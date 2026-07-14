"""
Multi-Store Retriever — Heterogeneous multi-store retrieval orchestrator.

Orchestrates retrieval from multiple heterogeneous stores (JSON, vector, graph,
etc.), each paired with its own sub-retriever. Results are fused through a
five-phase pipeline:

  Phase 1 — Per-store sub-retrieval:  Dispatch queries to each store's retriever
  Phase 2 — RRF fusion:               Reciprocal Rank Fusion across stores
  Phase 3 — Type quota:               Cap per-type output counts
  Phase 4 — MMR reranking:            Maximal Marginal Relevance for diversity
  Phase 5 — Contradiction detection:  Identify and resolve conflicting memories

Config options:
    rrf_k (int):            RRF constant. Default 60.
    multi_store_bonus (float): Bonus multiplier for units appearing in 2+ stores.
        Default 1.1.
    mmr_lambda (float):     MMR relevance/diversity trade-off. Default 0.7.
    type_quota (Dict[str, int]): Per-type output quotas.
        Default: {"tip": 3, "insight": 2, "workflow": 2, "trajectory": 1, "shortcut": 2}
    sub_retriever_top_k_multiplier (float): Multiplier applied to top_k for
        sub-retriever calls to ensure a broad candidate pool. Default 3.0.
    enable_contradiction_detection (bool): Whether to run Phase 5. Default True.
    retriever_map (Dict[str, str]): Override default store_type -> retriever_type
        mapping.

Usage:
    from automem.retrieval import MultiStoreRetriever, QueryContext

    retriever = MultiStoreRetriever(
        stores={"json": json_store, "vector": vector_store},
        embedding_model=embed_model,
    )
    pack = retriever.retrieve(QueryContext(query="How to parse PDF?"), top_k=10)
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from ..memory_schema import MemoryUnit
from .base_retriever import (
    BaseRetriever,
    EvidenceRef,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Default configuration constants
# ======================================================================

DEFAULT_RETRIEVER_MAP: Dict[str, str] = {
    "json": "keyword",
    "vector": "semantic",
    "hybrid": "hybrid",
    "graph": "graph",
    "llm_graph": "hybrid_graph",
}

DEFAULT_TYPE_QUOTA: Dict[str, int] = {
    "tip": 3,
    "insight": 2,
    "workflow": 1,     # workflow units are rare during cold-start (success-only source)
    "trajectory": 2,   # GAIA needs tool-strategy signals from multiple trajectories
    "shortcut": 2,
}

RRF_K = 60
MULTI_STORE_BONUS = 1.1
MMR_LAMBDA = 0.7
SUB_RETRIEVER_TOP_K_MULTIPLIER = 3.0


# ======================================================================
# Helper: lazy imports for sub-retrievers
# ======================================================================

def _get_retriever_class(retriever_type: str) -> type:
    """
    Lazily import and return the retriever class for a given type string.

    This avoids circular imports and only loads what is actually needed.
    """
    if retriever_type == "semantic":
        from .semantic_retriever import SemanticRetriever
        return SemanticRetriever
    elif retriever_type == "keyword":
        from .keyword_retriever import KeywordRetriever
        return KeywordRetriever
    elif retriever_type == "hybrid":
        from .hybrid_retriever import HybridRetriever
        return HybridRetriever
    elif retriever_type == "graph":
        from .graph_retriever import GraphRetriever
        return GraphRetriever
    elif retriever_type == "hybrid_graph":
        from .hybrid_graph_retriever import HybridGraphRetriever
        return HybridGraphRetriever
    elif retriever_type == "tag":
        from .tag_retriever import TagRetriever
        return TagRetriever
    # Codex CR2-8: search-space valid retrievers that the multi-store
    # orchestrator must dispatch as well, otherwise architectures with
    # heterogeneous storage + contrastive/cbr/cbr_rerank silently fall back
    # to no-result.
    elif retriever_type == "contrastive":
        from .contrastive_retriever import ContrastiveRetriever
        return ContrastiveRetriever
    elif retriever_type == "cbr":
        from .cbr_retriever import CBRRetriever
        return CBRRetriever
    elif retriever_type == "cbr_rerank":
        from .cbr_rerank_retriever import CBRRerankRetriever
        return CBRRerankRetriever
    # Stage-1 (2026-05-17) adoptions:
    elif retriever_type == "hyde":
        from .hyde_retriever import HydeRetriever
        return HydeRetriever
    elif retriever_type == "mmr":
        from .mmr_retriever import MmrRetriever
        return MmrRetriever
    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type!r}. "
            f"Supported: semantic, keyword, hybrid, graph, hybrid_graph, tag, "
            f"contrastive, cbr, cbr_rerank, hyde, mmr"
        )


def _get_embedding(unit: MemoryUnit) -> Optional[np.ndarray]:
    """Extract embedding vector from a MemoryUnit, if available."""
    emb = getattr(unit, "embedding", None)
    if emb is not None and isinstance(emb, np.ndarray) and emb.size > 0:
        return emb
    return None


# ======================================================================
# MultiStoreRetriever
# ======================================================================

class MultiStoreRetriever(BaseRetriever):
    """
    Multi-store heterogeneous retrieval orchestrator.

    Manages multiple storage backends, each paired with an appropriate
    sub-retriever. Results from all stores are fused through a five-phase
    pipeline: sub-retrieval → RRF fusion → type quota → MMR reranking →
    contradiction detection.

    Unlike other retrievers that operate on a single store, this class
    accepts a dictionary of stores and does NOT call super().__init__
    with a single store reference.
    """

    def __init__(
        self,
        stores: Dict[str, Any],
        embedding_model: Any = None,
        model: Any = None,
        model_resolver: Optional[Callable[[], Any]] = None,
        usage_in_task_metrics_resolver: Optional[Callable[[Any], bool]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            stores: Mapping from store type name (e.g. "json", "vector")
                to the storage backend instance.
            embedding_model: Shared embedding model for semantic retrievers
                and MMR cosine similarity computation.
            model: LLM model instance for ContradictionDetector Layer C
                (LLM-based arbitration). Optional.
            config: Strategy-specific configuration overrides.
        """
        # Intentionally skip BaseRetriever.__init__ — we hold multiple stores
        self.stores = stores
        self.embedding_model = embedding_model
        self.model = model
        self.model_resolver = model_resolver
        self.usage_in_task_metrics_resolver = usage_in_task_metrics_resolver
        self.config = config or {}

        # Unpack configuration with defaults
        self.rrf_k: int = self.config.get("rrf_k", RRF_K)
        self.multi_store_bonus: float = self.config.get(
            "multi_store_bonus", MULTI_STORE_BONUS
        )
        self.mmr_lambda: float = self.config.get("mmr_lambda", MMR_LAMBDA)
        self.type_quota: Dict[str, int] = self.config.get(
            "type_quota", DEFAULT_TYPE_QUOTA.copy()
        )
        self.sub_top_k_mult: float = self.config.get(
            "sub_retriever_top_k_multiplier", SUB_RETRIEVER_TOP_K_MULTIPLIER
        )
        self.enable_contradiction: bool = self.config.get(
            "enable_contradiction_detection", True
        )
        self.retriever_map: Dict[str, str] = self.config.get(
            "retriever_map", DEFAULT_RETRIEVER_MAP.copy()
        )
        # Tag-boost (Phase 2.5): optional rerank via query tag Jaccard
        self.tag_boost_weight: float = self.config.get("tag_boost_weight", 0.15)
        self.tag_weights: Dict[str, float] = self.config.get("tag_weights", {
            "task_domain": 0.5,
            "cognitive_skill": 0.35,
            "risk_pattern": 0.15,
        })
        # Shared QueryClassifier injected by ModularMemoryProvider when tag_aware=True
        self._query_classifier = self.config.get("query_classifier")

        # Cache for created sub-retrievers (avoid re-creation on each call)
        self._sub_retriever_cache: Dict[str, BaseRetriever] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "MultiStoreRetriever"

    # ------------------------------------------------------------------
    # Main retrieve entry point
    # ------------------------------------------------------------------

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        """
        Execute the full five-phase retrieval pipeline.

        Args:
            ctx: Query context with query text and optional embedding.
            top_k: Maximum number of units to return in the final pack.

        Returns:
            MemoryPack with fused, quota-filtered, diversity-reranked,
            contradiction-resolved results.
        """
        trace: List[TraceEntry] = []
        step_counter = 0

        # ----------------------------------------------------------
        # Phase 1: Per-store sub-retrieval
        # ----------------------------------------------------------
        sub_top_k = max(top_k, int(top_k * self.sub_top_k_mult))
        per_store_results: Dict[str, List[ScoredUnit]] = {}
        # Aggregate graph-expansion edge traversals from sub-packs (G1):
        # sub-pack fields are otherwise dropped when the final pack is built.
        collected_used_edges: List[Any] = []

        for store_type, store in self.stores.items():
            try:
                sub_retriever = self._create_sub_retriever(store_type, store)
                pack = sub_retriever.retrieve(ctx, top_k=sub_top_k)

                # Tag each unit with its source store
                for su in pack.scored_units:
                    su.method = f"{su.method}@{store_type}" if su.method else store_type

                per_store_results[store_type] = pack.scored_units
                collected_used_edges.extend(getattr(pack, "used_edges", []) or [])
                step_counter += 1
                trace.append(TraceEntry(
                    step=step_counter,
                    method=f"sub_retrieve:{store_type}",
                    candidates=len(pack.scored_units),
                    selected=len(pack.scored_units),
                    params={
                        "retriever": sub_retriever.name,
                        "store_type": store_type,
                        "sub_top_k": sub_top_k,
                    },
                ))
            except Exception as e:
                logger.warning(
                    "Sub-retrieval failed for store %s: %s", store_type, e
                )
                step_counter += 1
                trace.append(TraceEntry(
                    step=step_counter,
                    method=f"sub_retrieve:{store_type}",
                    candidates=0,
                    selected=0,
                    params={"error": str(e)},
                ))

        total_candidates = sum(len(v) for v in per_store_results.values())
        if total_candidates == 0:
            return self._make_empty_pack(ctx, trace)

        # ----------------------------------------------------------
        # Phase 2: Reciprocal Rank Fusion
        # ----------------------------------------------------------
        fused = self._rrf_fusion(per_store_results)
        step_counter += 1
        trace.append(TraceEntry(
            step=step_counter,
            method="rrf_fusion",
            candidates=total_candidates,
            selected=len(fused),
            params={
                "rrf_k": self.rrf_k,
                "multi_store_bonus": self.multi_store_bonus,
                "stores": list(per_store_results.keys()),
            },
        ))

        # ----------------------------------------------------------
        # Phase 2.5: Tag Boost (optional — only when QueryClassifier present)
        # ----------------------------------------------------------
        if self._query_classifier is not None:
            fused = self._apply_tag_boost(fused, ctx)
            step_counter += 1
            trace.append(TraceEntry(
                step=step_counter,
                method="tag_boost",
                candidates=len(fused),
                selected=len(fused),
                params={
                    "tag_boost_weight": self.tag_boost_weight,
                    "tag_weights": self.tag_weights,
                },
            ))

        # ----------------------------------------------------------
        # Phase 3: Type Quota
        # ----------------------------------------------------------
        quota_filtered = self._apply_type_quota(fused)
        step_counter += 1
        trace.append(TraceEntry(
            step=step_counter,
            method="type_quota",
            candidates=len(fused),
            selected=len(quota_filtered),
            params={"quota": self.type_quota},
        ))

        # ----------------------------------------------------------
        # Phase 4: MMR Reranking
        # ----------------------------------------------------------
        mmr_results = self._mmr_rerank(quota_filtered, top_k)
        step_counter += 1
        trace.append(TraceEntry(
            step=step_counter,
            method="mmr_rerank",
            candidates=len(quota_filtered),
            selected=len(mmr_results),
            params={"lambda": self.mmr_lambda, "top_k": top_k},
        ))

        # ----------------------------------------------------------
        # Phase 5: Contradiction Detection
        # ----------------------------------------------------------
        final_results = mmr_results
        if self.enable_contradiction and len(mmr_results) >= 2:
            final_results = self._run_contradiction_detection(
                mmr_results, ctx, trace
            )
            step_counter += 1
            trace.append(TraceEntry(
                step=step_counter,
                method="contradiction_detection",
                candidates=len(mmr_results),
                selected=len(final_results),
                params={"removed": len(mmr_results) - len(final_results)},
            ))

        final_pack = self._make_final_pack(ctx, final_results, trace)
        final_pack.used_edges = collected_used_edges
        return final_pack

    # ------------------------------------------------------------------
    # Phase 2.5: Tag Boost
    # ------------------------------------------------------------------

    def _apply_tag_boost(
        self, units: List[ScoredUnit], ctx: QueryContext
    ) -> List[ScoredUnit]:
        """Blend weighted Jaccard tag score into each unit's RRF score."""
        try:
            query_tags = self._query_classifier.classify(ctx.query)
        except Exception as e:
            logger.warning("tag_boost: classifier failed (%s), skipping", e)
            return units

        total_query_tags = sum(len(v) for v in query_tags.values())
        if total_query_tags == 0:
            return units

        # Rank by the boosted value but keep ScoredUnit.score untouched —
        # score is the absolute-cosine injection-gate signal (2026-06-29
        # invariant); writing the boost into it let a 0.30-cosine unit cross
        # the 0.40 gate floor.
        boosted = []
        for su in units:
            tag_score = self._tag_score_unit(query_tags, su.unit)
            boosted.append((su.score + self.tag_boost_weight * tag_score, su))

        boosted.sort(key=lambda pair: pair[0], reverse=True)
        return [su for _, su in boosted]

    def _tag_score_unit(
        self,
        query_tags: Dict[str, List[str]],
        unit: MemoryUnit,
    ) -> float:
        """Weighted Jaccard similarity across three tag dimensions."""
        content = unit.content
        fp = content.get("failure_pattern")
        unit_tags: Dict[str, List[str]] = {
            "task_domain": (
                unit.applicable_task_types
                or content.get("task_type_tags", [])
            ),
            "cognitive_skill": content.get("cognitive_skill_tags", []),
            "risk_pattern": (
                [fp] if isinstance(fp, str) and fp
                else list(fp) if fp else []
            ),
        }

        total = 0.0
        for dim, weight in self.tag_weights.items():
            q_set = set(query_tags.get(dim, []))
            u_set = set(unit_tags.get(dim, []))
            if not q_set:
                continue
            union = q_set | u_set
            if not union:
                continue
            total += weight * len(q_set & u_set) / len(union)
        return total

    # ------------------------------------------------------------------
    # Phase 1: Sub-retriever creation
    # ------------------------------------------------------------------

    def _create_sub_retriever(
        self, store_type: str, store: Any
    ) -> BaseRetriever:
        """
        Create (or return cached) sub-retriever for a given store type.

        Uses the retriever_map to determine which retriever class to
        instantiate. Falls back to keyword retriever for unknown store types.
        """
        if store_type in self._sub_retriever_cache:
            return self._sub_retriever_cache[store_type]

        # Resolution chain: injected map -> DEFAULT_RETRIEVER_MAP -> keyword.
        # The provider injects a map covering only the PRIMARY store; without
        # the DEFAULT fallback here, secondary stores (failure_bank, compiler
        # extras) silently landed on keyword regardless of DEFAULT_RETRIEVER_MAP.
        retriever_type = (
            self.retriever_map.get(store_type)
            or DEFAULT_RETRIEVER_MAP.get(store_type, "keyword")
        )
        cls = _get_retriever_class(retriever_type)

        # Construct the sub-retriever with appropriate arguments
        if retriever_type == "tag":
            # TagRetriever needs a QueryClassifier
            classifier = self.config.get("query_classifier")
            if classifier is None:
                from .tag_vocabulary import TagVocabulary
                from .query_classifier import QueryClassifier as QC
                vocab = TagVocabulary()
                classifier = QC(model=self.model, vocabulary=vocab)
            retriever = cls(
                store=store,
                classifier=classifier,
                config=self.config.get("tag_config"),
            )
        elif retriever_type in ("semantic", "graph", "hybrid_graph"):
            # These retrievers accept an embedding_model parameter
            retriever = cls(
                store=store,
                embedding_model=self.embedding_model,
                config=self.config.get(f"{retriever_type}_config"),
            )
        elif retriever_type in ("contrastive", "cbr", "cbr_rerank"):
            # Codex Round 3 R3-6: CBR + contrastive need embedding_model
            # to encode source_task_query for case matching. Without it
            # they silently degrade to a content-only fallback while the
            # search pretends it tested the requested retriever.
            #
            # CBR reranking receives the same injected model contract as the
            # single-store path. It never discovers a different model through
            # process environment.
            kwargs = {
                "store": store,
                "embedding_model": self.embedding_model,
                "config": self.config.get(f"{retriever_type}_config"),
            }
            if retriever_type == "cbr_rerank":
                kwargs.update(
                    {
                        "model": self.model,
                        "model_resolver": self.model_resolver,
                        "usage_in_task_metrics_resolver": (
                            self.usage_in_task_metrics_resolver
                        ),
                    }
                )
            try:
                retriever = cls(**kwargs)
            except TypeError as _e:
                logger.warning(
                    "multi_store_retriever: %s does not accept embedding_model "
                    "(%s); falling back to content-only constructor — "
                    "case-similarity matching is degraded.",
                    cls.__name__, _e,
                )
                retriever = cls(
                    store=store,
                    config=self.config.get(f"{retriever_type}_config"),
                )
        elif retriever_type == "hybrid":
            # HybridRetriever needs sub_retrievers
            from .semantic_retriever import SemanticRetriever
            from .keyword_retriever import KeywordRetriever

            sem = SemanticRetriever(
                store=store,
                embedding_model=self.embedding_model,
            )
            kw = KeywordRetriever(store=store)
            retriever = cls(
                store=store,
                sub_retrievers=[sem, kw],
                config=self.config.get("hybrid_config"),
            )
        elif retriever_type == "hyde":
            # Codex F-7 fix (2026-05-18): hyde needs embedding_model AND a
            # hypothesis_model (the LLM that writes a candidate answer
            # passage). The compiler does not populate hypothesis_model in
            # retriever_config, so fall back to the provider's task/extraction
            # model (self.model). Without this fallback, HyDE silently
            # degraded to semantic-only retrieval.
            hyde_cfg = dict(self.config.get("hyde_config") or {})
            hypo_model = hyde_cfg.get("hypothesis_model") or self.model
            retriever = cls(
                store=store,
                embedding_model=self.embedding_model,
                hypothesis_model=hypo_model,
                config=hyde_cfg,
            )
        elif retriever_type == "mmr":
            # MMR also needs embedding_model — without it sub-retriever
            # construction defaults to keyword-only fallback (the else
            # branch below).
            retriever = cls(
                store=store,
                embedding_model=self.embedding_model,
                config=self.config.get("mmr_config"),
            )
        else:
            # keyword and other simple retrievers
            retriever = cls(
                store=store,
                config=self.config.get(f"{retriever_type}_config"),
            )

        self._sub_retriever_cache[store_type] = retriever
        return retriever

    # ------------------------------------------------------------------
    # Phase 2: Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self, per_store_results: Dict[str, List[ScoredUnit]]
    ) -> List[ScoredUnit]:
        """
        Fuse results from multiple stores using Reciprocal Rank Fusion.

        For each unique unit, accumulate:
            rrf_score = sum( 1 / (k + rank_in_store) )
        across all stores where the unit appears. Units appearing in
        2+ stores receive an additional multiplicative bonus.

        Args:
            per_store_results: Mapping from store name to ranked ScoredUnit
                lists, each in its sub-retriever's OWN ranking order (which
                for contrastive / cbr_rerank / mmr is intentionally NOT
                score-descending — score is the absolute-cosine gate signal,
                order is the ranking signal).

        Returns:
            Deduplicated list of ScoredUnit sorted by fused RRF score
            descending.
        """
        # Accumulate RRF contributions per unit ID
        rrf_scores: Dict[str, float] = defaultdict(float)
        unit_map: Dict[str, ScoredUnit] = {}
        store_presence: Dict[str, Set[str]] = defaultdict(set)

        for store_name, scored_units in per_store_results.items():
            # Ordering contract (2026-07-11): use the sub-retriever's GIVEN
            # order to assign RRF ranks. Since the 2026-06-29 fix made
            # ScoredUnit.score an absolute cosine for the injection gate,
            # re-sorting by score here collapsed contrastive / cbr_rerank /
            # mmr / hybrid rankings back to pure cosine order.
            for rank, su in enumerate(scored_units):
                uid = su.unit.id
                rrf_scores[uid] += 1.0 / (self.rrf_k + rank + 1)
                store_presence[uid].add(store_name)

                # Keep the ScoredUnit with the highest original score
                if uid not in unit_map or su.score > unit_map[uid].score:
                    unit_map[uid] = su

        # Apply multi-store bonus
        for uid in rrf_scores:
            if len(store_presence[uid]) >= 2:
                rrf_scores[uid] *= self.multi_store_bonus

        # Build fused list sorted by RRF score
        fused: List[ScoredUnit] = []
        for uid in sorted(rrf_scores, key=rrf_scores.get, reverse=True):
            su = unit_map[uid]
            stores_hit = store_presence[uid]
            # FIX (2026-06-25): the score field must carry the original max SIMILARITY
            # (e.g. 0.64), NOT the tiny RRF rank score (1/(60+rank) ≈ 0.016). RRF is used
            # only for RANKING (the sort above). The score field feeds the injection gate
            # (min_relevance / gate_threshold, default 0.40) AND the agent-visible
            # "relevance:" string. The old code put rrf_scores[uid] here, so EVERY unit
            # showed relevance ≈ 0.02 and the 0.40 gate dropped ALL memory (or, gate-off,
            # injected off-topic units with a meaningless 0.02 score → toxic on strong
            # backbones). Keep RRF for ordering; report similarity for gating/display.
            fused.append(ScoredUnit(
                unit=su.unit,
                score=float(su.score),
                method=f"rrf({'|'.join(sorted(stores_hit))})",
            ))

        return fused

    # ------------------------------------------------------------------
    # Phase 3: Type Quota
    # ------------------------------------------------------------------

    def _apply_type_quota(
        self, units: List[ScoredUnit]
    ) -> List[ScoredUnit]:
        """
        Apply per-type quota caps to the fused result list.

        Walks the list (in RRF-fused rank order) and selects up
        to quota[type] units of each type. Units whose type has no quota
        entry pass through uncapped.

        Args:
            units: Score-sorted list of ScoredUnit from RRF fusion.

        Returns:
            Filtered list preserving original score ordering.
        """
        type_counts: Dict[str, int] = defaultdict(int)
        selected: List[ScoredUnit] = []

        for su in units:
            unit_type = su.unit.type.value  # e.g. "tip", "insight"
            quota_limit = self.type_quota.get(unit_type)

            if quota_limit is not None and type_counts[unit_type] >= quota_limit:
                continue

            selected.append(su)
            type_counts[unit_type] += 1

        return selected

    # ------------------------------------------------------------------
    # Phase 4: MMR Reranking
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self, units: List[ScoredUnit], top_k: int
    ) -> List[ScoredUnit]:
        """
        Apply Maximal Marginal Relevance reranking for diversity.

        MMR score for a candidate c given already-selected set S:
            MMR(c) = lambda * relevance(c)
                     - (1 - lambda) * max_{s in S} similarity(c, s)

        Uses cosine similarity between unit embeddings. Falls back to
        score-only selection if embeddings are unavailable.

        Args:
            units: Candidate units (sorted by relevance score).
            top_k: Number of units to select.

        Returns:
            Reranked list of top_k ScoredUnit.
        """
        if len(units) <= top_k:
            return units

        # Collect embeddings; fall back to truncated list if none available
        embeddings: List[Optional[np.ndarray]] = []
        has_any_embedding = False
        for su in units:
            emb = _get_embedding(su.unit)
            embeddings.append(emb)
            if emb is not None:
                has_any_embedding = True

        if not has_any_embedding:
            # No embeddings available — just truncate by score
            logger.debug(
                "MMR: no embeddings available, falling back to score truncation"
            )
            return units[:top_k]

        # Determine embedding dimension from first available embedding
        emb_dim = next(e.shape[0] for e in embeddings if e is not None)

        # Build embedding matrix; use zero vectors for missing embeddings
        emb_matrix = np.zeros((len(units), emb_dim), dtype=np.float32)
        for i, emb in enumerate(embeddings):
            if emb is not None:
                emb_matrix[i] = emb.flatten()[:emb_dim]

        # Normalize relevance scores to [0, 1] for fair comparison
        scores = np.array([su.score for su in units], dtype=np.float64)
        score_max = scores.max()
        score_min = scores.min()
        if score_max > score_min:
            norm_scores = (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.ones_like(scores)

        # Precompute full pairwise similarity matrix
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        normalized = np.divide(
            emb_matrix,
            norms,
            out=np.zeros_like(emb_matrix, dtype=float),
            where=norms != 0,
        )
        sim_matrix = normalized @ normalized.T

        # Greedy MMR selection
        selected_indices: List[int] = []
        remaining = set(range(len(units)))
        lam = self.mmr_lambda

        for _ in range(top_k):
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                relevance = norm_scores[idx]

                if selected_indices:
                    max_sim = max(
                        sim_matrix[idx, s] for s in selected_indices
                    )
                else:
                    max_sim = 0.0

                mmr_score = lam * relevance - (1.0 - lam) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx < 0:
                break

            selected_indices.append(best_idx)
            remaining.discard(best_idx)

        return [units[i] for i in selected_indices]

    # ------------------------------------------------------------------
    # Phase 5: Contradiction Detection
    # ------------------------------------------------------------------

    def _run_contradiction_detection(
        self,
        units: List[ScoredUnit],
        ctx: QueryContext,
        trace: List[TraceEntry],
    ) -> List[ScoredUnit]:
        """
        Run ContradictionDetector on the final result set and apply
        resolution strategies.

        Attempts to import and instantiate ContradictionDetector. If the
        module is unavailable or detection fails, returns units unchanged.

        Args:
            units: Current result list after MMR reranking.
            ctx: Original query context.
            trace: Mutable trace list (for logging detection details).

        Returns:
            Filtered list with contradictions resolved.
        """
        try:
            from .contradiction_detector import ContradictionDetector
        except ImportError:
            logger.debug(
                "ContradictionDetector not available; skipping Phase 5"
            )
            return units

        try:
            detector = ContradictionDetector(model=self.model)
            result = detector.detect(units)

            if result is None or not hasattr(result, "conflicts"):
                return units

            if not result.conflicts:
                return units

            return self._apply_contradiction_resolution(units, result)

        except Exception as e:
            logger.warning("Contradiction detection failed: %s", e)
            return units

    def _apply_contradiction_resolution(
        self,
        units: List[ScoredUnit],
        result: Any,
    ) -> List[ScoredUnit]:
        """
        Apply contradiction resolution actions to the unit list.

        Resolution strategies:
            - "keep_a":     Remove unit_b from results.
            - "keep_b":     Remove unit_a from results.
            - "both_valid": Keep both; annotate units with conflict info.
            - "merge":      Keep the higher-scored unit, remove the other.

        Args:
            units: Current scored units.
            result: ContradictionResult with a .conflicts list.

        Returns:
            Filtered list with contradictions resolved.
        """
        ids_to_remove: Set[str] = set()
        annotated_ids: Set[str] = set()

        # Build a score lookup for merge resolution
        score_map: Dict[str, float] = {su.unit.id: su.score for su in units}

        for conflict in result.conflicts:
            resolution = getattr(conflict, "resolution", None)
            unit_a_id = getattr(conflict, "unit_a_id", None)
            unit_b_id = getattr(conflict, "unit_b_id", None)

            if not unit_a_id or not unit_b_id:
                continue

            if resolution == "keep_a":
                ids_to_remove.add(unit_b_id)
            elif resolution == "keep_b":
                ids_to_remove.add(unit_a_id)
            elif resolution == "both_valid":
                # Keep both, but annotate them
                annotated_ids.add(unit_a_id)
                annotated_ids.add(unit_b_id)
            elif resolution == "merge":
                # Keep the higher-scored unit
                score_a = score_map.get(unit_a_id, 0.0)
                score_b = score_map.get(unit_b_id, 0.0)
                if score_a >= score_b:
                    ids_to_remove.add(unit_b_id)
                else:
                    ids_to_remove.add(unit_a_id)
            else:
                # Unknown resolution — keep both as a safe default
                logger.debug(
                    "Unknown contradiction resolution %r for (%s, %s); "
                    "keeping both",
                    resolution, unit_a_id, unit_b_id,
                )

        # Apply removal filter and annotation
        filtered: List[ScoredUnit] = []
        for su in units:
            if su.unit.id in ids_to_remove:
                continue

            if su.unit.id in annotated_ids:
                # Add conflict annotation to the method tag
                su = ScoredUnit(
                    unit=su.unit,
                    score=su.score,
                    method=f"{su.method}|conflict:both_valid",
                )

            filtered.append(su)

        return filtered

    # ------------------------------------------------------------------
    # Pack builders
    # ------------------------------------------------------------------

    def _make_empty_pack(
        self, ctx: QueryContext, trace: List[TraceEntry]
    ) -> MemoryPack:
        """Build an empty MemoryPack when no results are found."""
        return MemoryPack(
            query_context=ctx,
            scored_units=[],
            trace=trace,
            evidence=[],
            retriever_name=self.name,
        )

    def _make_final_pack(
        self,
        ctx: QueryContext,
        scored_units: List[ScoredUnit],
        trace: List[TraceEntry],
    ) -> MemoryPack:
        """Build the final MemoryPack with evidence references."""
        evidence = self._build_evidence_refs(scored_units)
        return MemoryPack(
            query_context=ctx,
            scored_units=scored_units,
            trace=trace,
            evidence=evidence,
            retriever_name=self.name,
        )

    def _build_evidence_refs(
        self, scored_units: List[ScoredUnit]
    ) -> List[EvidenceRef]:
        """Build evidence references from scored units."""
        evidence: List[EvidenceRef] = []
        for su in scored_units:
            text = su.unit.content_text()
            snippet = text[:120] + "..." if len(text) > 120 else text
            evidence.append(EvidenceRef(
                unit_id=su.unit.id,
                unit_type=su.unit.type.value,
                snippet=snippet,
                score=su.score,
                source_task_id=su.unit.source_task_id,
            ))
        return evidence

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_store_types(self) -> List[str]:
        """Return the list of registered store type names."""
        return list(self.stores.keys())

    def get_retriever_for_store(self, store_type: str) -> Optional[str]:
        """Return the retriever type string mapped to a store type."""
        return self.retriever_map.get(store_type)

    def clear_cache(self) -> None:
        """Clear the sub-retriever cache, forcing re-creation on next call."""
        self._sub_retriever_cache.clear()

    def __repr__(self) -> str:
        store_info = ", ".join(
            f"{k}->{self.retriever_map.get(k, '?')}"
            for k in self.stores
        )
        return f"MultiStoreRetriever(stores=[{store_info}])"
