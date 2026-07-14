"""
HybridGraphRetriever — Three-path parallel search with RRF fusion and MMR reranking.

Designed for LLMGraphStore's enhanced graph structure (entity embeddings,
fact edges, entity summaries/labels), but works with plain GraphStore too.

Three search paths executed in parallel:
  Path 1 — Semantic:  Cosine similarity on MemoryUnit embeddings
  Path 2 — BM25:      Keyword matching on content text (same as KeywordRetriever)
  Path 3 — Graph BFS: Entity name embedding search → BFS expansion along
            HAS_ENTITY / FACT edges → collect connected MemoryUnits

Fusion: Reciprocal Rank Fusion (RRF)
  score = Σ 1 / (k + rank_i)   for each path i where the unit appears

Reranking: Maximal Marginal Relevance (MMR)
  MMR = λ * relevance - (1-λ) * max_sim_to_selected

Config:
    rrf_k (int):         RRF constant. Default 60.
    mmr_lambda (float):  MMR diversity/relevance trade-off. Default 0.7.
    semantic_weight (float): Weight multiplier for semantic path. Default 1.0.
    bm25_weight (float):     Weight multiplier for BM25 path. Default 1.0.
    graph_weight (float):    Weight multiplier for graph path. Default 1.0.
    entity_top_k (int):  Number of entity seeds from embedding search. Default 5.
    graph_max_hops (int): BFS expansion depth. Default 2.
    min_score (float):   Minimum score threshold. Default 0.0.
    active_only (bool):  Only consider active units. Default True.
"""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from automem.memory_schema import MemoryUnit
from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)

import logging

logger = logging.getLogger(__name__)


# ======================================================================
# BM25 helpers (lightweight, no external dependency)
# ======================================================================

# Tokenizer shared with KeywordRetriever (CJK-aware since 2026-07-11) so the
# BM25 arm matches Chinese queries the same way the lexical arm does.
from automem.retrieval.keyword_retriever import _tokenize


def _bm25_score_batch(
    query_tokens: List[str],
    doc_token_lists: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Compute BM25 scores for a batch of documents against a query."""
    n_docs = len(doc_token_lists)
    if n_docs == 0:
        return []

    # Document frequencies
    doc_freq: Dict[str, int] = Counter()
    doc_lengths = []
    for tokens in doc_token_lists:
        doc_lengths.append(len(tokens))
        for t in set(tokens):
            doc_freq[t] += 1

    avg_dl = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0

    # IDF
    idf = {}
    for t in set(query_tokens):
        df = doc_freq.get(t, 0)
        idf[t] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    scores = []
    for i, tokens in enumerate(doc_token_lists):
        tf = Counter(tokens)
        dl = doc_lengths[i]
        score = 0.0
        for qt in query_tokens:
            if qt not in idf:
                continue
            f = tf.get(qt, 0)
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * dl / avg_dl)
            score += idf[qt] * numerator / denominator
        scores.append(score)
    return scores


# ======================================================================
# Reciprocal Rank Fusion
# ======================================================================

def _rrf_fuse(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: List[float],
    k: int = 60,
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.

    Each list is sorted by score descending. Weights scale contributions.
    Returns: {unit_id: fused_score}
    """
    fused: Dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (uid, _score) in enumerate(ranked):
            fused[uid] += weight * (1.0 / (k + rank + 1))
    return dict(fused)


# ======================================================================
# Maximal Marginal Relevance
# ======================================================================

def _mmr_rerank(
    candidates: List[ScoredUnit],
    query_emb: Optional[np.ndarray],
    top_k: int,
    lam: float = 0.7,
) -> List[ScoredUnit]:
    """
    MMR reranking: selects diverse top-k from candidates.

    MMR = λ * relevance - (1-λ) * max_sim_to_selected

    If no embeddings available, falls back to score-only ranking.
    """
    if not candidates or top_k <= 0:
        return []
    if len(candidates) <= top_k:
        return candidates

    # Build embedding matrix for candidates
    emb_list = []
    has_emb = []
    for su in candidates:
        if su.unit.embedding is not None:
            emb_list.append(su.unit.embedding)
            has_emb.append(True)
        else:
            emb_list.append(np.zeros(1))
            has_emb.append(False)

    if not any(has_emb):
        # No embeddings: just return top-k by score
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]

    # Normalize relevance scores to [0, 1]
    max_score = max(su.score for su in candidates) or 1.0
    min_score = min(su.score for su in candidates)
    score_range = max_score - min_score if max_score != min_score else 1.0

    selected: List[ScoredUnit] = []
    remaining = list(range(len(candidates)))

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            rel = (candidates[idx].score - min_score) / score_range

            # Max similarity to already selected
            max_sim = 0.0
            if selected and has_emb[idx]:
                for sel_su in selected:
                    if sel_su.unit.embedding is not None:
                        sim = float(cosine_similarity(
                            candidates[idx].unit.embedding.reshape(1, -1),
                            sel_su.unit.embedding.reshape(1, -1),
                        )[0][0])
                        max_sim = max(max_sim, sim)

            mmr = lam * rel - (1 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        if best_idx >= 0:
            selected.append(candidates[best_idx])
            remaining.remove(best_idx)

    return selected


# ======================================================================
# HybridGraphRetriever
# ======================================================================

class HybridGraphRetriever(BaseRetriever):
    """
    Three-path parallel search with RRF fusion and MMR reranking.

    Works with both GraphStore and LLMGraphStore. When used with
    LLMGraphStore, the graph path leverages entity name embeddings
    for more accurate entity-based search.
    """

    def __init__(self, store, embedding_model=None,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(store, config)
        self.embedding_model = embedding_model

        # RRF config
        self.rrf_k = self.config.get("rrf_k", 60)
        self.semantic_weight = self.config.get("semantic_weight", 1.0)
        self.bm25_weight = self.config.get("bm25_weight", 1.0)
        self.graph_weight = self.config.get("graph_weight", 1.0)

        # MMR config
        self.mmr_lambda = self.config.get("mmr_lambda", 0.7)

        # Graph path config
        self.entity_top_k = self.config.get("entity_top_k", 5)
        # `graph_hop` is the architecture-search dimension name; accept it as
        # an alias so flat/direct configs work in the single-store path too.
        self.graph_max_hops = self.config.get(
            "graph_max_hops", self.config.get("graph_hop", 2)
        )

        # General
        self.min_score = self.config.get("min_score", 0.0)
        self.active_only = self.config.get("active_only", True)

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        trace: List[TraceEntry] = []

        # Cold-start guard: zero out graph path weight when the graph is too sparse.
        # Uses StorageHealthReport when available; graph BFS on a sparse graph returns
        # nothing and wastes time.
        graph_weight = self.graph_weight
        if hasattr(self.store, 'get_health_report'):
            report = self.store.get_health_report()
            if report.is_cold_start or report.retrieval_mode == "graph_sparse":
                graph_weight = 0.0
                trace.append(TraceEntry(
                    step=0,
                    method="graph_cold_start_fallback",
                    candidates=report.unit_count,
                    selected=0,
                    params={
                        "reason": report.retrieval_mode,
                        "unit_count": report.unit_count,
                        "threshold": report.cold_start_threshold,
                        "graph_weight_zeroed": True,
                    },
                ))

        # Get query embedding
        query_emb = ctx.embedding
        if query_emb is None and self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                ctx.query, convert_to_numpy=True
            )

        # Get all candidate units
        units = self.store.get_all(active_only=self.active_only)
        if not units:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="hybrid_graph", candidates=0, selected=0,
            )])

        unit_map = {u.id: u for u in units}

        # ---- Path 1: Semantic ----
        semantic_ranked = self._path_semantic(query_emb, units, trace)

        # ---- Path 2: BM25 ----
        bm25_ranked = self._path_bm25(ctx.query, units, trace)

        # ---- Path 3: Graph BFS (skipped when graph_weight == 0.0) ----
        graph_ranked = []
        if graph_weight > 0.0:
            graph_ranked = self._path_graph(query_emb, ctx.query, unit_map, trace)

        # ---- RRF Fusion ----
        ranked_lists = [semantic_ranked, bm25_ranked, graph_ranked]
        weights = [self.semantic_weight, self.bm25_weight, graph_weight]
        fused_scores = _rrf_fuse(ranked_lists, weights, k=self.rrf_k)

        trace.append(TraceEntry(
            step=4,
            method="rrf_fusion",
            candidates=len(fused_scores),
            selected=len(fused_scores),
            params={
                "rrf_k": self.rrf_k,
                "weights": {"semantic": self.semantic_weight,
                            "bm25": self.bm25_weight,
                            "graph": self.graph_weight},
            },
        ))

        # Build candidate ScoredUnits.
        # FIX 2026-06-29: the final ScoredUnit.score must be the absolute SEMANTIC
        # cosine (the scale the Tier-1 injection gate compares against the 0.40
        # floor), NOT the RRF rank score (~0.016 at k=60) — identical class to the
        # multi_store _rrf_fusion fix. RRF ranks candidates; cosine is the gate score.
        sem_cos = dict(semantic_ranked)   # uid -> raw cosine (0..1)

        # Backfill an absolute cosine for units surfaced ONLY by the BM25 /
        # graph paths: sem_cos covers just the semantic top-N, and the old
        # hard 0.0 made those units permanently un-injectable under the
        # Tier-1 gate (which floors on absolute cosine at 0.40).
        if query_emb is not None:
            _missing = [
                (uid, unit_map[uid]) for uid in fused_scores
                if uid not in sem_cos
                and unit_map.get(uid) is not None
                and unit_map[uid].embedding is not None
            ]
            if _missing:
                _embs = np.vstack([u.embedding for _, u in _missing])
                _sims = cosine_similarity(
                    np.asarray(query_emb, dtype=np.float32).reshape(1, -1), _embs
                )[0]
                for (_uid, _), _s in zip(_missing, _sims):
                    sem_cos[_uid] = float(_s)

        ranked_cands = []
        for uid, rrf in fused_scores.items():
            unit = unit_map.get(uid)
            if unit is None:
                continue
            score = float(sem_cos.get(uid, 0.0))
            # min_score is documented (and configured) in COSINE scale; it was
            # mistakenly compared against RRF scores, whose ceiling is
            # ~sum(weights)/rrf_k ≈ 0.05 — any cosine-scale setting silently
            # filtered EVERYTHING. Default 0.0 keeps behavior identical.
            if score < self.min_score:
                continue
            ranked_cands.append((rrf, ScoredUnit(
                unit=unit, score=score, method="hybrid_graph",
            )))

        # ---- MMR Reranking ----
        # Rank by RRF (fused) to form the MMR pool; each ScoredUnit.score stays the
        # raw cosine so MMR's relevance term and the downstream gate both see cosine.
        mmr_pool = min(top_k * 3, len(ranked_cands))
        ranked_cands.sort(key=lambda t: t[0], reverse=True)
        candidates = [su for _, su in ranked_cands[:mmr_pool]]

        selected = _mmr_rerank(candidates, query_emb, top_k, self.mmr_lambda)

        trace.append(TraceEntry(
            step=5,
            method="mmr_rerank",
            candidates=mmr_pool,
            selected=len(selected),
            params={"lambda": self.mmr_lambda, "top_k": top_k},
        ))


        return self._make_pack(ctx, selected, trace)

    # ------------------------------------------------------------------
    # Path 1: Semantic search on MemoryUnit embeddings
    # ------------------------------------------------------------------

    def _path_semantic(
        self,
        query_emb: Optional[np.ndarray],
        units: List[MemoryUnit],
        trace: List[TraceEntry],
    ) -> List[Tuple[str, float]]:
        """Returns ranked list of (unit_id, score) by embedding similarity."""
        if query_emb is None:
            trace.append(TraceEntry(
                step=1, method="semantic", candidates=0, selected=0,
                params={"error": "no_query_embedding"},
            ))
            return []

        emb_matrix, emb_units = self.store.get_embedding_index(
            active_only=self.active_only
        )
        if emb_matrix is None or len(emb_units) == 0:
            trace.append(TraceEntry(
                step=1, method="semantic", candidates=0, selected=0,
            ))
            return []

        q = query_emb.reshape(1, -1)
        sims = cosine_similarity(q, emb_matrix)[0]

        ranked = []
        for i, sim in enumerate(sims):
            ranked.append((emb_units[i].id, float(sim)))
        ranked.sort(key=lambda x: x[1], reverse=True)

        trace.append(TraceEntry(
            step=1,
            method="semantic",
            candidates=len(emb_units),
            selected=len(ranked),
        ))
        return ranked

    # ------------------------------------------------------------------
    # Path 2: BM25 keyword search
    # ------------------------------------------------------------------

    def _path_bm25(
        self,
        query: str,
        units: List[MemoryUnit],
        trace: List[TraceEntry],
    ) -> List[Tuple[str, float]]:
        """Returns ranked list of (unit_id, score) by BM25 score."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            trace.append(TraceEntry(
                step=2, method="bm25", candidates=0, selected=0,
                params={"error": "empty_query"},
            ))
            return []

        doc_tokens = [_tokenize(u.content_text()) for u in units]
        scores = _bm25_score_batch(query_tokens, doc_tokens)

        ranked = [(units[i].id, scores[i]) for i in range(len(units)) if scores[i] > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)

        trace.append(TraceEntry(
            step=2,
            method="bm25",
            candidates=len(units),
            selected=len(ranked),
            params={"query_tokens": len(query_tokens)},
        ))
        return ranked

    # ------------------------------------------------------------------
    # Path 3: Graph BFS from entity seeds
    # ------------------------------------------------------------------

    def _path_graph(
        self,
        query_emb: Optional[np.ndarray],
        query_text: str,
        unit_map: Dict[str, MemoryUnit],
        trace: List[TraceEntry],
    ) -> List[Tuple[str, float]]:
        """
        Entity-based graph search:
        1. Find entity nodes by embedding similarity (LLMGraphStore)
           or by keyword match (fallback for plain GraphStore)
        2. BFS along HAS_ENTITY (reverse) + FACT edges
        3. Collect connected MemoryUnits with propagated scores
        """
        has_graph = hasattr(self.store, 'neighbors') and hasattr(self.store, 'get')
        if not has_graph:
            trace.append(TraceEntry(
                step=3, method="graph_bfs", candidates=0, selected=0,
                params={"error": "no_graph_store"},
            ))
            return []

        # Step 1: Find seed entity nodes
        seed_entities = self._find_entity_seeds(query_emb, query_text)
        if not seed_entities:
            trace.append(TraceEntry(
                step=3, method="graph_bfs", candidates=0, selected=0,
                params={"error": "no_entity_seeds"},
            ))
            return []

        # Step 2: BFS from entity nodes to collect MemoryUnit IDs
        unit_scores: Dict[str, float] = {}
        visited_entities: Set[str] = set()

        # Frontier: (entity_nid, base_score, hop)
        frontier = [(nid, score) for nid, score in seed_entities]

        for hop in range(self.graph_max_hops):
            next_frontier = []
            decay = 0.7 ** hop

            for entity_nid, base_score in frontier:
                if entity_nid in visited_entities:
                    continue
                visited_entities.add(entity_nid)

                propagated = base_score * decay

                # Follow HAS_ENTITY edges in reverse: find content nodes
                # that link to this entity
                content_nids = self.store.neighbors(
                    entity_nid, edge_type="HAS_ENTITY", direction="in"
                )
                for cnid in content_nids:
                    if not cnid.startswith("m:"):
                        continue
                    uid = cnid[2:]
                    if uid in unit_map:
                        old = unit_scores.get(uid, 0.0)
                        unit_scores[uid] = max(old, propagated)

                # Follow FACT edges to other entities (expand)
                if hop < self.graph_max_hops - 1:
                    all_neighbors = self.store.neighbors(
                        entity_nid, direction="both"
                    )
                    for nbr_nid in all_neighbors:
                        if nbr_nid.startswith("e:") and nbr_nid not in visited_entities:
                            # Check if connected via a FACT-like edge
                            next_frontier.append((nbr_nid, propagated * 0.7))

            frontier = next_frontier

        ranked = sorted(unit_scores.items(), key=lambda x: x[1], reverse=True)

        trace.append(TraceEntry(
            step=3,
            method="graph_bfs",
            candidates=len(seed_entities),
            selected=len(ranked),
            params={
                "seed_entities": len(seed_entities),
                "max_hops": self.graph_max_hops,
            },
        ))
        return ranked

    def _find_entity_seeds(
        self,
        query_emb: Optional[np.ndarray],
        query_text: str,
    ) -> List[Tuple[str, float]]:
        """
        Find entity nodes most relevant to the query.

        Strategy 1 (LLMGraphStore): Use entity name embeddings
        Strategy 2 (fallback): Keyword match on entity display names
        """
        seeds: List[Tuple[str, float]] = []

        # Strategy 1: Entity embedding search (LLMGraphStore)
        if hasattr(self.store, 'get_all_entity_embeddings') and query_emb is not None:
            emb_matrix, nids = self.store.get_all_entity_embeddings()
            if emb_matrix is not None and len(nids) > 0:
                q = query_emb.reshape(1, -1)
                sims = cosine_similarity(q, emb_matrix)[0]
                top_k = min(self.entity_top_k, len(nids))
                top_indices = sims.argsort()[-top_k:][::-1]
                for idx in top_indices:
                    if sims[idx] > 0.1:
                        seeds.append((nids[idx], float(sims[idx])))
                if seeds:
                    return seeds

        # Strategy 2: Keyword match on entity display names
        if not hasattr(self.store, '_graph'):
            return []

        query_tokens = set(_tokenize(query_text))
        if not query_tokens:
            return []

        for nid, data in self.store._graph.nodes(data=True):
            if data.get("layer") != "entity":
                continue
            name = data.get("display_name", "")
            name_tokens = set(_tokenize(name))
            # Also check aliases
            aliases = data.get("aliases", [])
            for alias in aliases:
                name_tokens.update(_tokenize(alias))

            overlap = query_tokens & name_tokens
            if overlap:
                score = len(overlap) / max(len(query_tokens), 1)
                seeds.append((nid, score))

        seeds.sort(key=lambda x: x[1], reverse=True)
        return seeds[:self.entity_top_k]
