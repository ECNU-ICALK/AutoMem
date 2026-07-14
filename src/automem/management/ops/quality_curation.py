"""
QualityCurationOp — Periodic operation that refines memory unit confidence
scores using a Bayesian update that blends the unit's observed success rate
with its prior confidence.
"""

import time
import logging
from typing import Any, Dict

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class QualityCurationOp(BaseManageOp):
    """Refine confidence via Bayesian blending of success rate and prior."""

    op_name = "quality_curation"
    op_group = "maintenance"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 15

    def execute(self, context: Dict[str, Any]) -> OpResult:
        """
        For every active unit that has been used at least once, compute a
        Bayesian-blended confidence update:

            new_confidence = bayesian_weight * success_rate
                           + (1 - bayesian_weight) * confidence

        The update is applied only when the change exceeds 0.001 to avoid
        unnecessary write amplification.

        Config keys:
          bayesian_weight (float): Weight given to observed success_rate.
                                   Default 0.7.

        Returns:
            OpResult with units_modified count.
        """
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            bayesian_weight = self.config.get("bayesian_weight", 0.7)

            all_active = self.store.get_all(active_only=True)
            candidates = [u for u in all_active if u.usage_count > 0]
            modified = 0

            for unit in candidates:
                old_confidence = unit.confidence

                # Bayesian blend of empirical success rate and prior confidence
                new_confidence = (
                    bayesian_weight * unit.success_rate
                    + (1.0 - bayesian_weight) * unit.confidence
                )

                # Clamp to valid range
                new_confidence = max(0.01, min(new_confidence, 1.0))

                # Only persist if the change is meaningful
                if abs(new_confidence - old_confidence) > 0.001:
                    unit.confidence = new_confidence
                    self.store.update(unit)
                    modified += 1

            result.units_modified = modified
            result.units_affected = len(candidates)
            result.details = {
                "total_active": len(all_active),
                "candidates_with_usage": len(candidates),
                "modified": modified,
                "bayesian_weight": bayesian_weight,
            }
            logger.info(
                "quality_curation: updated %d/%d candidates (bayesian_weight=%.2f).",
                modified, len(candidates), bayesian_weight,
            )

        except Exception as e:
            logger.error("quality_curation failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result
