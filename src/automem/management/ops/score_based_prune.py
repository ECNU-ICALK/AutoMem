"""
ScoreBasedPruneOp — Periodic operation that deactivates low-quality memory
units based on their effective_score, and enforces a maximum memory budget.

For graph-backed stores the unit is soft-deactivated (is_active=False in node
attributes) but edges are preserved for audit / relation traversal.
For vector/hybrid stores the FAISS index is rebuilt after the batch.
"""

import time
import logging
from typing import Any, Dict, List

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit

logger = logging.getLogger(__name__)


class ScoreBasedPruneOp(BaseManageOp):
    """Prune low-scoring memory units to stay within quality and budget limits."""

    op_name = "score_based_prune"
    op_group = "maintenance"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = False
    rl_action_id = 14

    def execute(self, context: Dict[str, Any]) -> OpResult:
        """
        Two-phase pruning:

        Phase 1 — Quality gate:
            Deactivate any active unit whose effective_score < min_effective_score.

        Phase 2 — Budget cap:
            If the number of remaining active units still exceeds
            max_memory_count, sort by effective_score ascending and
            deactivate the lowest-scoring units until the budget is met.

        Storage-specific behaviour:
          * Graph stores: set is_active=False in node attrs, keep edges.
          * Vector / Hybrid stores: rebuild FAISS index after batch.

        Config keys:
          min_effective_score (float): Threshold for phase 1.  Default 0.05.
          max_memory_count (int):      Budget cap for phase 2.  Default 500.

        Returns:
            OpResult with units_deleted = total deactivated count.
        """
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            min_score = self.config.get("min_effective_score", 0.05)
            max_count = self.config.get("max_memory_count", 500)

            all_active = self.store.get_all(active_only=True)
            total_before = len(all_active)
            deactivated: List[str] = []

            # --- Phase 1: quality gate ---
            surviving = []
            for unit in all_active:
                if unit.effective_score < min_score:
                    self._deactivate_unit(unit)
                    deactivated.append(unit.id)
                else:
                    surviving.append(unit)

            phase1_count = len(deactivated)

            # --- Phase 2: budget cap ---
            phase2_count = 0
            if len(surviving) > max_count:
                # Sort ascending by effective_score; prune the weakest
                surviving.sort(key=lambda u: u.effective_score)
                excess = len(surviving) - max_count
                for unit in surviving[:excess]:
                    self._deactivate_unit(unit)
                    deactivated.append(unit.id)
                    phase2_count += 1

            # Rebuild FAISS index for vector/hybrid stores after batch pruning
            if deactivated and not self._is_graph_store():
                self._rebuild_faiss_if_needed()

            result.units_deleted = len(deactivated)
            result.units_affected = total_before
            result.details = {
                "total_before": total_before,
                "phase1_pruned": phase1_count,
                "phase2_pruned": phase2_count,
                "total_pruned": len(deactivated),
                "remaining_active": total_before - len(deactivated),
                "min_effective_score": min_score,
                "max_memory_count": max_count,
            }
            logger.info(
                "score_based_prune: pruned %d units (phase1=%d, phase2=%d), "
                "%d remaining from %d total.",
                len(deactivated), phase1_count, phase2_count,
                total_before - len(deactivated), total_before,
            )

        except Exception as e:
            logger.error("score_based_prune failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deactivate_unit(self, unit: MemoryUnit) -> None:
        """
        Soft-deactivate a unit.

        For graph stores, update the node attribute directly so that edges
        are preserved for audit purposes.  For all other backends, simply
        mark is_active=False and persist.
        """
        unit.is_active = False

        if self._is_graph_store():
            # Update node attrs in the graph while keeping edges intact
            nid = self.store._content_nid(unit.id)
            if nid in self.store._graph:
                self.store._graph.nodes[nid]["is_active"] = False
            self.store.update(unit)
        else:
            self.store.update(unit)
