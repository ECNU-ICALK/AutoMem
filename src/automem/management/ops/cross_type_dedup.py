"""
Cross-type deduplication operation.

Detects when an INSIGHT is semantically covered by a TIP (higher similarity
than the threshold) and deactivates the INSIGHT, since TIPs are more
actionable. Also adds REINFORCES relations for moderate similarity pairs.
"""

import time
import logging
from typing import Any, Dict, List

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class CrossTypeDedupOp(BaseManageOp):
    """Deduplicate INSIGHTs that are covered by existing TIPs."""

    op_name = "cross_type_dedup"
    op_group = "deduplication"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = True
    rl_action_id = 6

    def execute(self, context: Dict[str, Any]) -> OpResult:
        result = OpResult(op_name=self.op_name)
        t0 = time.time()

        cross_threshold = self.config.get("cross_type_threshold", 0.85)
        reinforce_threshold = 0.7

        try:
            all_units = self.store.get_all()
            active_units = [u for u in all_units if u.is_active and u.embedding is not None]

            insights: List[MemoryUnit] = [
                u for u in active_units if u.type == MemoryUnitType.INSIGHT
            ]
            tips: List[MemoryUnit] = [
                u for u in active_units if u.type == MemoryUnitType.TIP
            ]

            total_deactivated = 0
            total_reinforced = 0
            deactivated_ids: list = []

            for insight in insights:
                best_sim = 0.0
                best_tip: MemoryUnit | None = None

                # Find the most similar TIP
                for tip in tips:
                    sim = _cosine_similarity(insight.embedding, tip.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_tip = tip

                if best_tip is None:
                    continue

                if best_sim > cross_threshold:
                    # INSIGHT is covered by TIP -> deactivate INSIGHT
                    insight.is_active = False
                    deactivated_ids.append(insight.id)

                    # TIP supersedes INSIGHT
                    supersedes_rel = MemoryRelation(
                        target_id=insight.id,
                        relation_type=RelationType.SUPERSEDES,
                        weight=best_sim,
                    )
                    best_tip.relations.append(supersedes_rel)

                    # NOTE: no direct graph.add_edge — graph stores project
                    # content->content edges from unit.relations inside
                    # update() (_sync_relation_edges, uppercase edge_type).
                    # The old direct edge used the wrong attribute
                    # (relation=, lowercase), invisible to graph expansion's
                    # edge_type filter.

                    self.store.update(insight)
                    self.store.update(best_tip)
                    total_deactivated += 1
                    logger.debug(
                        "Cross-type dedup: deactivated INSIGHT %s (sim=%.3f with TIP %s)",
                        insight.id, best_sim, best_tip.id,
                    )

                elif best_sim > reinforce_threshold:
                    # Moderate similarity: add REINFORCES between both — ONCE.
                    # This op is PERIODIC and used to append a duplicate
                    # relation every cycle (unbounded relations growth). Graph
                    # stores project content->content edges FROM unit.relations
                    # inside update() (_sync_relation_edges), so duplicate
                    # relations also meant ever-multiplying parallel edges;
                    # the old direct graph.add_edge here was redundant AND
                    # used the wrong attribute (relation=, lowercase) that
                    # graph expansion's edge_type filter never matched.
                    ins_has = any(
                        r.relation_type == RelationType.REINFORCES
                        and r.target_id == best_tip.id
                        for r in insight.relations
                    )
                    tip_has = any(
                        r.relation_type == RelationType.REINFORCES
                        and r.target_id == insight.id
                        for r in best_tip.relations
                    )
                    if ins_has and tip_has:
                        continue  # already linked in a previous cycle

                    if not ins_has:
                        insight.relations.append(MemoryRelation(
                            target_id=best_tip.id,
                            relation_type=RelationType.REINFORCES,
                            weight=best_sim,
                        ))
                    if not tip_has:
                        best_tip.relations.append(MemoryRelation(
                            target_id=insight.id,
                            relation_type=RelationType.REINFORCES,
                            weight=best_sim,
                        ))

                    self.store.update(insight)
                    self.store.update(best_tip)
                    total_reinforced += 1
                    logger.debug(
                        "Cross-type: added REINFORCES between INSIGHT %s and TIP %s (sim=%.3f)",
                        insight.id, best_tip.id, best_sim,
                    )

            # Rebuild FAISS index if any deactivations occurred
            if total_deactivated > 0:
                self._rebuild_faiss_if_needed()

            result.triggered = total_deactivated > 0 or total_reinforced > 0
            result.units_affected = total_deactivated + total_reinforced
            result.units_deleted = total_deactivated  # deactivated, not hard-deleted
            result.units_modified = total_reinforced
            result.details = {
                "cross_type_threshold": cross_threshold,
                "reinforce_threshold": reinforce_threshold,
                "insights_scanned": len(insights),
                "tips_scanned": len(tips),
                "deactivated_ids": deactivated_ids,
                "reinforced_pairs": total_reinforced,
            }

        except Exception:
            logger.exception("CrossTypeDedupOp failed")
            result.details["error"] = "exception during execution"

        result.duration_ms = (time.time() - t0) * 1000
        return result
