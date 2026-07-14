"""
CBRRetriever — Case-Based Reasoning retrieval.

F3 in the memory system redesign. Core idea:

  Current memory retrievers embed ``MemoryUnit.content_text()`` (the LESSON)
  and match user query against that. This is *reverse* CBR: we are asking
  "is the current question similar to a past solution?" instead of
  "is the current question similar to a past question?".

  CBRRetriever fixes this by using ``MemoryUnit.source_task_query`` (the
  original task that produced this unit) as the retrieval anchor.

Pipeline:

  Stage 1 — Case match:   encode(current_query) vs encode(source_task_query)
                          for every unit in the store. Produces case_score.

  Stage 2 — Content fallback: runs the existing SemanticRetriever to
                              compute content_score per unit.

  Merge: ``final_score = max(case_score, content_weight * content_score)``

         Interpretation:
           - If past question is similar, trust stage 1 (CBR behavior).
           - If not, fall back to the old content-based retrieval, scaled
             down by ``content_weight`` so case hits outrank content hits
             of equal raw cosine.

Typical config::

    content_weight: 0.7     # how much to trust content-only matches
    case_score_floor: 0.0   # filter in provider via min_relevance instead
    encode_batch_size: 64
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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
from automem.retrieval.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)


class CBRRetriever(BaseRetriever):
    """Case-Based Reasoning retriever with content fallback."""

    def __init__(self, store, embedding_model=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self.content_weight = float(self.config.get("content_weight", 0.7))
        self.case_score_floor = float(self.config.get("case_score_floor", 0.0))
        self.encode_batch_size = int(self.config.get("encode_batch_size", 64))
        self.active_only = bool(self.config.get("active_only", True))

        # In-memory cache: unit_id -> query_embedding (np.ndarray) or None if
        # the unit has no source_task_query. _case_text_cache records the
        # text each embedding was computed FROM, so a changed
        # source_task_query invalidates the entry instead of silently
        # reusing a stale vector.
        self._case_emb_cache: Dict[str, Optional[np.ndarray]] = {}
        self._case_text_cache: Dict[str, str] = {}

        # Internal semantic retriever for content-side scoring.
        self._content_retriever = SemanticRetriever(
            store, embedding_model, self.config
        )

    # ------------------------------------------------------------------
    # Case embedding computation
    # ------------------------------------------------------------------
    def _ensure_case_embeddings(self, units: List[MemoryUnit]) -> None:
        """Compute and cache query embeddings for units missing them.

        Uses batched encoding (batch=64 by default) for speed. Units with
        an empty source_task_query get None entries so we skip them next time.
        """
        if self.embedding_model is None:
            return
        needed_texts: List[str] = []
        needed_units: List[MemoryUnit] = []
        for u in units:
            text = (u.source_task_query or "").strip()
            if (
                u.id in self._case_emb_cache
                and self._case_text_cache.get(u.id, "") == text
            ):
                continue
            if not text:
                self._case_emb_cache[u.id] = None
                self._case_text_cache[u.id] = ""
                continue
            needed_texts.append(text)
            needed_units.append(u)

        if not needed_texts:
            return

        try:
            # Batched encoding
            embs = self.embedding_model.encode(
                needed_texts,
                batch_size=self.encode_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for u, e in zip(needed_units, embs):
                # Normalize to unit vector (sentence-transformers usually
                # already does this, but be explicit for cosine stability).
                n = np.linalg.norm(e)
                self._case_emb_cache[u.id] = e / n if n > 0 else e
                self._case_text_cache[u.id] = (u.source_task_query or "").strip()
        except Exception as e:
            logger.warning(f"[CBR] case embedding batch failed: {e}")
            for u in needed_units:
                self._case_emb_cache.pop(u.id, None)
                self._case_text_cache.pop(u.id, None)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        # 1. Resolve query embedding
        query_emb = ctx.embedding
        if query_emb is None and self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                ctx.query, convert_to_numpy=True
            )
        if query_emb is None:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="cbr", candidates=0, selected=0,
                params={"error": "no_embedding"},
            )])

        # Normalize query vec
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        # 2. Pull all candidate units from the store
        try:
            all_units = self.store.get_all(active_only=self.active_only)
        except Exception as e:
            logger.warning(f"[CBR] store.get_all failed: {e}; fallback to content")
            return self._content_retriever.retrieve(ctx, top_k)

        # 3. Try store-side query index first (P3). If present, skip text
        # encoding entirely for those units.
        store_idx_mat = None
        store_idx_units = []
        if hasattr(self.store, "get_query_embedding_index"):
            try:
                store_idx_mat, store_idx_units = self.store.get_query_embedding_index(
                    active_only=self.active_only
                )
            except Exception as e:
                logger.warning(f"[CBR] store.get_query_embedding_index failed: {e}")

        # Only encode the units that are NOT already in the store-side index
        store_idx_ids = {u.id for u in store_idx_units}
        units_to_encode = [u for u in all_units if u.id not in store_idx_ids]
        if units_to_encode:
            self._ensure_case_embeddings(units_to_encode)
            # Push newly-encoded embeddings into store-side index for next call
            if hasattr(self.store, "set_query_embedding"):
                for u in units_to_encode:
                    emb = self._case_emb_cache.get(u.id)
                    if emb is not None:
                        try:
                            self.store.set_query_embedding(
                                u.id, u.source_task_query or "", emb
                            )
                        except Exception:
                            pass

        # 4. Stage 1: case cosine similarity
        case_units: List[MemoryUnit] = []
        case_emb_list: List[np.ndarray] = []
        # Prefer store-side index (faster), fall back to local cache
        if store_idx_mat is not None and len(store_idx_units) > 0:
            for i, u in enumerate(store_idx_units):
                emb_vec = store_idx_mat[i]
                if emb_vec is not None:
                    case_units.append(u)
                    case_emb_list.append(emb_vec)
        for u in all_units:
            if u.id in store_idx_ids:
                continue
            emb = self._case_emb_cache.get(u.id)
            if emb is not None:
                case_units.append(u)
                case_emb_list.append(emb)

        if case_emb_list:
            case_mat = np.vstack(case_emb_list)
            case_scores = cosine_similarity(
                query_emb.reshape(1, -1), case_mat
            )[0]
        else:
            case_scores = np.zeros(0, dtype=np.float32)

        case_score_by_id: Dict[str, float] = {
            u.id: float(s) for u, s in zip(case_units, case_scores)
        }

        # 5. Stage 2: content scoring via existing semantic retriever
        # We ask for more units than top_k so merge has enough candidates.
        content_pool_k = max(top_k * 3, 30)
        try:
            content_pack = self._content_retriever.retrieve(
                ctx, top_k=content_pool_k
            )
            content_score_by_id: Dict[str, float] = {
                su.unit.id: float(su.score) for su in content_pack.scored_units
            }
            content_units = {su.unit.id: su.unit for su in content_pack.scored_units}
        except Exception as e:
            logger.warning(f"[CBR] content retrieve failed: {e}")
            content_score_by_id = {}
            content_units = {}

        # 6. Merge: max(case_score, content_weight * content_score)
        # Union of unit IDs from both stages
        unit_by_id: Dict[str, MemoryUnit] = {u.id: u for u in case_units}
        for uid, u in content_units.items():
            unit_by_id.setdefault(uid, u)

        # FIX 2026-06-29: content_weight is a RANKING knob (case-hits should rank
        # above content-only hits) — it must NOT scale the unit's final score, which
        # is the absolute cosine the Tier-1 gate compares against the 0.40 floor.
        # Previously `ts = content_cosine * 0.7` deflated e.g. 0.50 -> 0.35 and the
        # gate dropped real content matches (same class as the contrastive/hybrid
        # fixes). Gate score = raw max cosine; content_weight ranks only.
        scored_pairs: List[tuple] = []
        for uid, u in unit_by_id.items():
            cs = case_score_by_id.get(uid, 0.0)            # raw case cosine
            raw_ts = content_score_by_id.get(uid, 0.0)      # raw content cosine
            gate_score = max(cs, raw_ts)                     # absolute cosine -> gate
            rank_score = max(cs, raw_ts * self.content_weight)  # case-preferring rank
            method = "cbr_case" if cs >= raw_ts * self.content_weight else "cbr_content"
            scored_pairs.append((rank_score, ScoredUnit(unit=u, score=gate_score, method=method)))

        # 7. Sort by rank_score, filter (on the cosine gate score) + truncate
        scored_pairs.sort(key=lambda t: t[0], reverse=True)
        merged: List[ScoredUnit] = [su for _, su in scored_pairs]
        if self.case_score_floor > 0:
            merged = [su for su in merged if su.score >= self.case_score_floor]
        merged = merged[:top_k]

        # 9. Trace
        best_case = float(np.max(case_scores)) if len(case_scores) else 0.0
        best_content_raw = (
            max((v for v in content_score_by_id.values()), default=0.0)
        )
        trace = [TraceEntry(
            step=1, method="cbr",
            candidates=len(unit_by_id),
            selected=len(merged),
            params={
                "top_k": top_k,
                "content_weight": self.content_weight,
                "case_pool_size": len(case_units),
                "content_pool_size": len(content_units),
                "best_case_score": round(best_case, 4),
                "best_content_raw_score": round(best_content_raw, 4),
                "cache_size": len(self._case_emb_cache),
            },
        )]

        return self._make_pack(ctx, merged, trace)
