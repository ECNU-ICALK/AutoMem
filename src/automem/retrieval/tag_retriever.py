"""
Tag Retriever — Structured tag matching for memory retrieval.

A first-class retrieval strategy that scores memories by tag overlap
rather than embedding similarity. This provides a genuinely orthogonal
retrieval signal: while SemanticRetriever and KeywordRetriever both
compute similarity in continuous vector spaces, TagRetriever operates
on discrete categorical features.

Scoring: Weighted Jaccard similarity across three tag dimensions
(task_domain, cognitive_skill, risk_pattern). Produces scores in [0, 1]
that are directly comparable with other retrievers' scores.

Requires:
  - A QueryClassifier to convert the raw query into structured tags
  - MemoryUnits with populated applicable_task_types and/or content tags
"""

import logging
from typing import Any, Dict, List, Optional, Set

from automem.memory_schema import MemoryUnit
from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)
from automem.retrieval.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class TagRetriever(BaseRetriever):
    """
    Retrieves memories by structured tag matching (Jaccard similarity).

    Config options:
        tag_weights (dict): Per-dimension weights. Default:
            {"task_domain": 0.5, "cognitive_skill": 0.35, "risk_pattern": 0.15}
        active_only (bool): Only consider active units. Default True.
    """

    def __init__(
        self,
        store,
        classifier: QueryClassifier,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(store, config)
        self.classifier = classifier
        self.active_only = self.config.get("active_only", True)
        self.weights: Dict[str, float] = self.config.get("tag_weights", {
            "task_domain": 0.5,
            "cognitive_skill": 0.35,
            "risk_pattern": 0.15,
        })

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        # Step 1: Classify query into tags
        query_tags = self.classifier.classify(ctx.query)
        ctx.metadata["query_tags"] = query_tags

        # Check if classification produced any tags
        total_query_tags = sum(len(v) for v in query_tags.values())
        if total_query_tags == 0:
            logger.info("TagRetriever: query classified with no tags, returning empty")
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="tag_classify", candidates=0, selected=0,
                params={"error": "no_query_tags"},
            )])

        # Step 2: Get all units from store
        units = self.store.get_all(active_only=self.active_only)
        if not units:
            return self._make_pack(ctx, [], [TraceEntry(
                step=1, method="tag", candidates=0, selected=0,
            )])

        # Step 3: Score every unit by tag overlap
        scored = []
        for unit in units:
            score = self._compute_tag_score(query_tags, unit)
            if score > 0:
                scored.append(ScoredUnit(
                    unit=unit, score=score, method="tag",
                ))

        # Step 4: Sort and top-k
        scored.sort(key=lambda su: su.score, reverse=True)
        scored = scored[:top_k]

        trace = [
            TraceEntry(
                step=1, method="tag_classify", candidates=0, selected=0,
                params={"query_tags": query_tags},
            ),
            TraceEntry(
                step=2, method="tag_match", candidates=len(units),
                selected=len(scored),
                params={"weights": self.weights, "top_k": top_k},
            ),
        ]


        logger.info(
            f"TagRetriever: query_tags={query_tags}, "
            f"matched {len(scored)}/{len(units)} units, "
            f"best_score={scored[0].score:.3f}" if scored else "no matches"
        )

        return self._make_pack(ctx, scored, trace)

    def _compute_tag_score(
        self, query_tags: Dict[str, List[str]], unit: MemoryUnit
    ) -> float:
        """Weighted Jaccard similarity across tag dimensions."""
        unit_tags = self._extract_unit_tags(unit)
        total = 0.0

        for dim, weight in self.weights.items():
            q_set: Set[str] = set(query_tags.get(dim, []))
            u_set: Set[str] = set(unit_tags.get(dim, []))
            if not q_set:
                continue
            union = q_set | u_set
            if not union:
                continue
            jaccard = len(q_set & u_set) / len(union)
            total += weight * jaccard

        return total

    @staticmethod
    def _extract_unit_tags(unit: MemoryUnit) -> Dict[str, List[str]]:
        """Extract tags from MemoryUnit fields into unified 3-dimension format."""
        content = unit.content

        # task_domain: prefer top-level, fallback to content field
        task_domain = (
            unit.applicable_task_types
            or content.get("task_type_tags", [])
        )

        # cognitive_skill: from content field (added by classifier during encoding)
        cognitive_skill = content.get("cognitive_skill_tags", [])

        # risk_pattern: from insight's failure_pattern field
        risk_pattern = []
        fp = content.get("failure_pattern")
        if fp:
            risk_pattern = [fp] if isinstance(fp, str) else list(fp)

        return {
            "task_domain": task_domain,
            "cognitive_skill": cognitive_skill,
            "risk_pattern": risk_pattern,
        }
