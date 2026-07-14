"""
SizeCappedPruneOp — Hard size-cap-triggered emergency pruning.

Differs from score_based_prune (gradual quality gate) in two ways:
  1. Triggered only when active pool size > cap; otherwise no-op.
  2. When triggered, deactivates the bottom-K to bring pool down to
     cap * shrink_to_ratio, regardless of their effective_score.

Rationale:
  Without a hard cap, the canonical pool grows ~150 units/round on GAIA
  (observed in evolution_run_1: 0 → 628 in 4 rounds). Long runs see pool
  bloat → retrieval noise → lift regression. This op sits at the periodic
  layer and trips only when the soft tools (time_decay, score_based_prune,
  utility_audit) have not kept the pool lean enough.

Selection priority for "who to prune first" (high → low):
  1. Already conflict_count > 0    (memory_stale candidates)
  2. is_negative_example == True   (failure-trajectory evidence; useful but
                                    expendable)
  3. usage_count == 0              (never retrieved → no proven value)
  4. lowest effective_score
  5. oldest age_hours

Config keys:
  cap (int):              Active pool size cap. Default 300.
  shrink_to_ratio (float):After triggering, deactivate down to cap*ratio.
                          Default 0.85 (= 255 if cap=300).
"""
import time
import logging
from typing import Any, Dict, List

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit

logger = logging.getLogger(__name__)


class SizeCappedPruneOp(BaseManageOp):
    """Emergency pruning when pool exceeds cap. Periodic mandatory."""

    op_name = "size_capped_prune"
    op_group = "maintenance"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 17

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=False)

        cap = int(self.config.get("cap", 300))
        shrink_ratio = float(self.config.get("shrink_to_ratio", 0.85))
        target_size = max(1, int(cap * shrink_ratio))

        try:
            active = self.store.get_all(active_only=True)
            n_active = len(active)
            if n_active <= cap:
                result.details = {
                    "active_pool": n_active, "cap": cap, "action": "no-op"
                }
                result.duration_ms = (time.time() - t0) * 1000
                return result

            n_to_drop = n_active - target_size
            sorted_units = sorted(
                active,
                key=lambda u: (
                    -int(getattr(u, "conflict_count", 0)),    # prefer pruning conflicted units
                    -int(bool(getattr(u, "is_negative_example", False))),  # then negative examples
                    -int(int(getattr(u, "usage_count", 0)) == 0),  # then never-used
                    float(u.effective_score),                  # then low score
                    -float(getattr(u, "age_hours", 0.0)),      # then oldest
                ),
            )
            victims: List[MemoryUnit] = sorted_units[:n_to_drop]

            for u in victims:
                u.is_active = False
                self.store.update(u)

            # Codex Q4-3 fix (2026-04-28): rebuild FAISS once after the
            # batch. Without this, vector/hybrid stores leave deactivated
            # vectors in the FAISS index, where they continue occupying
            # top-k slots and the cap effectively does not work for
            # vector-backed architectures.
            if victims:
                try:
                    self._rebuild_faiss_if_needed()
                except Exception as e:
                    logger.debug(f"size_capped_prune FAISS rebuild skipped: {e}")

            result.triggered = True
            result.units_affected = n_active
            result.units_deleted = len(victims)
            result.details = {
                "active_pool_before": n_active,
                "active_pool_after": n_active - len(victims),
                "cap": cap,
                "shrink_to_ratio": shrink_ratio,
                "target_size": target_size,
                "deactivated_ids": [u.id for u in victims][:32],
                "selection_breakdown": {
                    "by_conflict": sum(1 for u in victims
                                       if getattr(u, "conflict_count", 0) > 0),
                    "by_negative": sum(1 for u in victims
                                       if getattr(u, "is_negative_example", False)),
                    "by_zero_usage": sum(1 for u in victims
                                         if getattr(u, "usage_count", 0) == 0),
                },
            }
            logger.info(
                "size_capped_prune: pool %d > cap %d, deactivated %d → %d "
                "(breakdown: conflict=%d, negative=%d, zero_usage=%d)",
                n_active, cap, len(victims), n_active - len(victims),
                result.details["selection_breakdown"]["by_conflict"],
                result.details["selection_breakdown"]["by_negative"],
                result.details["selection_breakdown"]["by_zero_usage"],
            )
        except Exception as e:
            logger.error("size_capped_prune failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result
