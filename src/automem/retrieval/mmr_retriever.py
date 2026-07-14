"""MMR Retriever — Maximal Marginal Relevance (Carbonell & Goldstein 1998).

Stage 1 adoption (2026-05-17). Implements the classic MMR re-ranking formula
to encourage diversity in retrieved units. No external repo needed — the
formula is a 30-line classic.

Algorithm:
  Stage 1: dense retrieval to get top-N candidates (N >= 2*top_k)
  Stage 2: iteratively pick the candidate with highest:
             score(u) = λ · sim(u, query) − (1 − λ) · max sim(u, S)
           where S = already-selected, λ ∈ [0, 1] balances relevance vs diversity.

Why this matters for AutoMem:
  Current graph / contrastive retrievers tend to return clustered near-duplicates
  when the pool has many semantically similar units (e.g. several tip units about
  "answer in single word"). MMR enforces diverse coverage of the pool. Cost: 0
  LLM calls; only one extra cosine pass per candidate.
"""

from typing import Any, Dict, List, Optional

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)

logger = logging.getLogger(__name__)


def mmr_select(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidate_scores: np.ndarray,
    top_k: int = 5,
    lambda_: float = 0.5,
) -> List[int]:
    """Pure MMR selector — return indices of selected candidates in order.

    Args:
        query_emb: (D,) query embedding, normalized.
        candidate_embs: (N, D) candidate embeddings, normalized.
        candidate_scores: (N,) pre-computed sim(candidate, query). Provided
            so we don't recompute when caller already has it (e.g. from
            stage-1 FAISS results).
        top_k: number to select.
        lambda_: 0 = pure diversity, 1 = pure relevance. 0.5 = balanced.

    Returns:
        Indices into ``candidate_embs`` of the selected candidates in MMR order.
    """
    n = candidate_embs.shape[0]
    if n == 0:
        return []
    top_k = min(top_k, n)
    selected: List[int] = []
    remaining = list(range(n))

    while len(selected) < top_k and remaining:
        if not selected:
            # First pick = max relevance
            winner_rel_idx = int(np.argmax(candidate_scores[remaining]))
            winner = remaining[winner_rel_idx]
        else:
            # MMR formula on each remaining candidate
            selected_embs = candidate_embs[selected]  # (k, D)
            best_score = -np.inf
            winner = remaining[0]
            for i in remaining:
                rel = candidate_scores[i]
                # max sim to any already-selected
                sims_to_sel = candidate_embs[i] @ selected_embs.T  # (k,)
                max_redundancy = float(np.max(sims_to_sel))
                mmr_score = lambda_ * rel - (1.0 - lambda_) * max_redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    winner = i
        selected.append(winner)
        remaining.remove(winner)

    return selected


class MmrRetriever(BaseRetriever):
    """Two-stage MMR retriever: dense recall then diversity re-rank.

    Config options:
        lambda_ (float):        MMR trade-off, 0=diverse, 1=relevant. Default 0.5.
        candidate_pool_size:    Stage-1 recall depth. Default max(2*top_k, 20).
        min_score (float):      Minimum sim threshold (default 0.0).
        active_only (bool):     Default True.
    """

    def __init__(
        self,
        store,
        embedding_model=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self.lambda_ = float(self.config.get("lambda_", 0.5))
        self.candidate_pool_size = self.config.get("candidate_pool_size", None)
        self.min_score = self.config.get("min_score", 0.0)
        self.active_only = self.config.get("active_only", True)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def _encode_query(self, ctx: QueryContext) -> Optional[np.ndarray]:
        if ctx.embedding is not None:
            return ctx.embedding
        if self.embedding_model is not None:
            try:
                return self.embedding_model.encode(ctx.query, convert_to_numpy=True)
            except Exception as e:
                logger.warning("[MMR] embedding_model.encode failed: %s", e)
        try:
            from automem.search.attribution import _get_embedder
            emb = _get_embedder()
            if emb is None:
                return None
            return emb.encode(ctx.query, normalize_embeddings=True, show_progress_bar=False)
        except Exception as e:
            logger.warning("[MMR] fallback embedder failed: %s", e)
            return None

    # ──────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────
    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        query_emb = self._encode_query(ctx)
        if query_emb is None:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="mmr", candidates=0, selected=0,
                params={"error": "no_embedding"},
            )])

        # Stage 1: dense recall (over-retrieve)
        pool_size = self.candidate_pool_size or max(2 * top_k, 20)
        if hasattr(self.store, "search"):
            try:
                stage1 = self.store.search(
                    query_emb, top_k=pool_size, active_only=self.active_only,
                )
                # stage1 is [(unit, score), ...]
                candidate_units = [u for u, _ in stage1]
                candidate_scores_list = [s for _, s in stage1]
            except Exception:
                stage1 = None
                candidate_units, candidate_scores_list = [], []
        else:
            stage1 = None
            candidate_units, candidate_scores_list = [], []

        # Fallback: numpy cosine over full pool
        if not candidate_units:
            emb_matrix, units = self.store.get_embedding_index(
                active_only=self.active_only,
            )
            if emb_matrix is None or len(units) == 0:
                return self._make_pack(ctx, [], [TraceEntry(
                    step=1, method="mmr", candidates=0, selected=0,
                )])
            sims = cosine_similarity(query_emb.reshape(1, -1), emb_matrix)[0]
            order = sims.argsort()[-min(pool_size, len(units)):][::-1]
            candidate_units = [units[i] for i in order]
            candidate_scores_list = [float(sims[i]) for i in order]

        if not candidate_units:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="mmr", candidates=0, selected=0,
            )])

        # Stage 2: collect embeddings for selected candidates so MMR can compute
        # in-set diversity. Fall back to re-encoding if the unit has no
        # ``embedding`` attribute.
        cand_emb_rows: List[np.ndarray] = []
        for u in candidate_units:
            e = getattr(u, "embedding", None)
            if e is None:
                # Re-encode from unit text — costly but only when stage-1 had no
                # cached embedding (e.g. JsonStorage).
                text = getattr(u, "content", None)
                if isinstance(text, dict):
                    text = " ".join(str(v) for v in text.values() if isinstance(v, str))
                e = self._encode_query(QueryContext(query=str(text or "")))
                if e is None:
                    e = np.zeros_like(query_emb)
            cand_emb_rows.append(np.asarray(e, dtype=np.float32))

        candidate_embs = np.stack(cand_emb_rows)
        # Normalize for cosine consistency (FAISS / our embedder return
        # normalized embs already, but be defensive).
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-12
        candidate_embs = candidate_embs / norms
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        candidate_scores = np.asarray(candidate_scores_list, dtype=np.float32)

        # Run MMR selection
        selected_idx = mmr_select(
            q_norm, candidate_embs, candidate_scores,
            top_k=top_k, lambda_=self.lambda_,
        )

        scored: List[ScoredUnit] = []
        for i in selected_idx:
            score = float(candidate_scores[i])
            if score < self.min_score:
                continue
            scored.append(ScoredUnit(
                unit=candidate_units[i], score=score, method="mmr",
            ))


        trace = [
            TraceEntry(
                step=1, method="dense_recall",
                candidates=len(candidate_units), selected=len(candidate_units),
                params={"pool_size": pool_size},
            ),
            TraceEntry(
                step=2, method="mmr_rerank",
                candidates=len(candidate_units), selected=len(scored),
                params={"lambda": self.lambda_, "top_k": top_k},
            ),
        ]
        return self._make_pack(ctx, scored, trace)


__all__ = ["MmrRetriever", "mmr_select"]
