"""
Hybrid Retriever — Weighted combination of sub-retrievers.

A meta-retriever that runs multiple retrieval strategies in parallel,
normalizes their scores, and merges results using configurable weights.

Default configuration uses semantic (0.7) + keyword (0.3), matching
the proven pattern from cerebra_fusion and agent_kb providers.
"""

from typing import Any, Dict, List, Optional

from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)


class HybridRetriever(BaseRetriever):
    """
    Combines multiple retrievers with weighted score fusion.

    Config options:
        weights (Dict[str, float]): Retriever name -> weight.
            Default: {"SemanticRetriever": 0.7, "KeywordRetriever": 0.3}
        dedup (bool): Deduplicate by unit ID. Default True.
    """

    def __init__(
        self,
        store,
        sub_retrievers: List[BaseRetriever],
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(store, config)
        self.sub_retrievers = sub_retrievers
        self.weights: Dict[str, float] = self.config.get("weights", {})
        self.dedup = self.config.get("dedup", True)

        # Default equal weights if not specified
        if not self.weights:
            n = len(sub_retrievers)
            for r in sub_retrievers:
                self.weights[r.name] = 1.0 / n

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        # Run all sub-retrievers
        sub_packs: List[MemoryPack] = []
        all_trace: List[TraceEntry] = []
        step_counter = 0

        for retriever in self.sub_retrievers:
            # Request more from each sub-retriever to allow fusion
            sub_top_k = max(top_k * 2, 10)
            pack = retriever.retrieve(ctx, top_k=sub_top_k)
            sub_packs.append(pack)

            # Relabel trace steps
            weight = self.weights.get(retriever.name, 0.0)
            for t in pack.trace:
                step_counter += 1
                t.step = step_counter
                t.params["weight"] = weight
                all_trace.append(t)

        # Merge scores using weighted score board
        score_board: Dict[str, float] = {}
        unit_map: Dict[str, ScoredUnit] = {}

        for pack, retriever in zip(sub_packs, self.sub_retrievers):
            weight = self.weights.get(retriever.name, 0.0)
            if not pack.scored_units:
                continue

            # Normalize scores within this retriever's results
            max_score = max(su.score for su in pack.scored_units)
            min_score = min(su.score for su in pack.scored_units)
            score_range = max_score - min_score

            for su in pack.scored_units:
                uid = su.unit.id
                # Normalize to [0, 1]
                if score_range > 0:
                    norm_score = (su.score - min_score) / score_range
                else:
                    norm_score = 1.0

                weighted_score = norm_score * weight
                score_board[uid] = score_board.get(uid, 0.0) + weighted_score

                # Keep the ScoredUnit with highest raw score
                if uid not in unit_map or su.score > unit_map[uid].score:
                    unit_map[uid] = su

        # Rank by fused score and select top-k
        ranked = sorted(score_board.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        scored = []
        for uid, fused_score in ranked:
            su = unit_map[uid]
            # FIX (2026-06-25): expose the raw absolute similarity (semantic cosine or
            # keyword TF-IDF, both in [0,1]) — NOT the within-batch-normalized fused_score
            # (whose top is always ≈max-weight regardless of absolute relevance). The score
            # field feeds the injection gate (min_relevance 0.40) and the agent-visible
            # "relevance:"; with the normalized score, a no-relevant-match task still had a
            # top unit ≈1.0 and got off-topic memory injected. fused_score is used only for
            # ranking (the sort above). Same class of bug as the RRF fusion fix.
            scored.append(ScoredUnit(
                unit=su.unit,
                score=float(su.score),
                method="hybrid",
            ))

        # Final trace entry for the fusion step
        step_counter += 1
        all_trace.append(TraceEntry(
            step=step_counter,
            method="hybrid_fusion",
            candidates=len(score_board),
            selected=len(scored),
            params={
                "weights": self.weights,
                "top_k": top_k,
            },
        ))


        return self._make_pack(ctx, scored, all_trace)
