"""
Contrastive Retriever — Success/failure differentiated search.

Separates memory pool into success-sourced and failure-sourced units,
applies different retrieval strategies to each pool, and merges with
configurable weights.

Inspired by ExPeL's pattern: insights from failures (text-heavy) vs
trajectories from successes (hybrid), with different scoring weights.

This enables the agent to learn from both positive exemplars and
negative lessons simultaneously.
"""

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


class ContrastiveRetriever(BaseRetriever):
    """
    Retrieves memories with differentiated treatment of success vs failure units.

    Success pool: tips, workflows, trajectories from successful tasks
    Failure pool: insights, tips from failed tasks

    Config options:
        success_weight (float): Weight for success-pool results. Default 0.6.
        failure_weight (float): Weight for failure-pool results. Default 0.4.
        success_k (int): Max units from success pool. Default 3.
        failure_k (int): Max units from failure pool. Default 2.
        min_score (float): Minimum similarity threshold. Default 0.0.
        active_only (bool): Only consider active units. Default True.
    """

    def __init__(self, store, embedding_model=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self.success_weight = self.config.get("success_weight", 0.6)
        self.failure_weight = self.config.get("failure_weight", 0.4)
        self.success_k = self.config.get("success_k", 3)
        self.failure_k = self.config.get("failure_k", 2)
        self.min_score = self.config.get("min_score", 0.0)
        self.active_only = self.config.get("active_only", True)

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        trace: List[TraceEntry] = []

        # Get query embedding
        query_emb = ctx.embedding
        if query_emb is None and self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                ctx.query, convert_to_numpy=True
            )

        # Get all active units
        all_units = self.store.get_all(active_only=self.active_only)
        if not all_units:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="contrastive", candidates=0, selected=0,
            )])

        # Split into success and failure pools
        success_pool = [u for u in all_units if u.task_outcome == "success"]
        failure_pool = [u for u in all_units if u.task_outcome == "failure"]

        # Retrieve from each pool
        success_scored = self._search_pool(
            query_emb, success_pool, self.success_k, "success"
        )
        failure_scored = self._search_pool(
            query_emb, failure_pool, self.failure_k, "failure"
        )

        trace.append(TraceEntry(
            step=1,
            method="contrastive_success",
            candidates=len(success_pool),
            selected=len(success_scored),
            params={"weight": self.success_weight, "k": self.success_k},
        ))
        trace.append(TraceEntry(
            step=2,
            method="contrastive_failure",
            candidates=len(failure_pool),
            selected=len(failure_scored),
            params={"weight": self.failure_weight, "k": self.failure_k},
        ))

        # Apply pool weights for RANKING ONLY, then merge.
        # FIX 2026-06-29: the pool weight (0.6 success / 0.4 failure) must NOT
        # scale the unit's final `score` — that score is the absolute cosine the
        # Tier-1 injection gate compares against the 0.40 floor. Multiplying the
        # cosine by 0.6/0.4 deflated e.g. a real 0.49 match to 0.29/0.20, so the
        # gate silently dropped ALL relevant memory (observed: xBench 0/110
        # injected despite offline top-1 cosine 0.49). Same class as the
        # hybrid_retriever / multi_store RRF fix: weight ranks, raw cosine scores.
        ranked: List[tuple] = []
        # Pass su.method through: _search_pool tags degraded-fallback results
        # ("contrastive_degraded_*"), and re-hardcoding the method here erased
        # that diagnostic from the final pack/trace.
        for su in success_scored:
            ranked.append((su.score * self.success_weight, ScoredUnit(
                unit=su.unit, score=su.score, method=su.method)))
        for su in failure_scored:
            ranked.append((su.score * self.failure_weight, ScoredUnit(
                unit=su.unit, score=su.score, method=su.method)))

        # Sort by the weighted RANK key, take top_k; final score stays raw cosine.
        ranked.sort(key=lambda t: t[0], reverse=True)
        merged: List[ScoredUnit] = [su for _, su in ranked[:top_k]]

        trace.append(TraceEntry(
            step=3,
            method="contrastive_merge",
            candidates=len(success_scored) + len(failure_scored),
            selected=len(merged),
            params={
                "success_weight": self.success_weight,
                "failure_weight": self.failure_weight,
                "top_k": top_k,
            },
        ))


        return self._make_pack(ctx, merged, trace)

    def _search_pool(
        self,
        query_emb: Optional[np.ndarray],
        pool: List[MemoryUnit],
        k: int,
        pool_name: str,
    ) -> List[ScoredUnit]:
        """Search within a specific pool using embedding similarity."""
        if not pool:
            return []

        # Filter units with embeddings
        units_with_emb = [u for u in pool if u.embedding is not None]
        if not units_with_emb or query_emb is None:
            # Degraded fallback: rank by effective_score but do NOT report it
            # as ScoredUnit.score — score is the absolute-cosine injection-gate
            # signal, and effective_score is query-INDEPENDENT quality (a
            # fresh high-confidence unit scores ~0.45+ and used to cross the
            # 0.40 gate floor with zero relevance to the query). score=0.0
            # keeps the ordering while letting the gate treat these as
            # unverifiable-relevance results.
            pool_sorted = sorted(pool, key=lambda u: u.effective_score, reverse=True)
            return [
                ScoredUnit(
                    unit=u, score=0.0,
                    method=f"contrastive_degraded_{pool_name}",
                )
                for u in pool_sorted[:k]
            ]

        emb_matrix = np.vstack([u.embedding for u in units_with_emb])
        q = query_emb.reshape(1, -1)
        sims = cosine_similarity(q, emb_matrix)[0]

        actual_k = min(k, len(units_with_emb))
        top_indices = sims.argsort()[-actual_k:][::-1]

        scored = []
        for idx in top_indices:
            score = float(sims[idx])
            if score >= self.min_score:
                scored.append(ScoredUnit(
                    unit=units_with_emb[idx],
                    score=score,
                    method=f"contrastive_{pool_name}",
                ))
        return scored
