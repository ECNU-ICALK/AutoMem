"""
PenalizeOnFailureOp — Reduce confidence and decay weight for memories
that were used in a failed task.

Part of the 'failure_adjustment' operation group.
"""

import time
import logging
from typing import Any, Dict, List

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class PenalizeOnFailureOp(BaseManageOp):
    """
    After a failed task, penalize all memory units that were used during
    the task by reducing their confidence and decay weight.  This makes
    underperforming memories less likely to surface in future retrievals.
    """

    op_name = "penalize_on_failure"
    op_group = "failure_adjustment"
    trigger_type = TriggerType.POST_TASK
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 8
    rl_param_range = (0.0, 0.5)

    _DEFAULT_CONFIG = {
        "confidence_penalty": 0.15,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name)

        try:
            # Only trigger on task failure
            if context.get("task_succeeded", True):
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            result.triggered = True
            penalty = self.config.get(
                "confidence_penalty",
                self._DEFAULT_CONFIG["confidence_penalty"],
            )

            used_unit_ids: List[str] = context.get("used_unit_ids", [])
            units_modified = 0

            for unit_id in used_unit_ids:
                unit = self.store.get(unit_id)
                if unit is None:
                    logger.warning(
                        "penalize_on_failure: unit %s not found, skipping",
                        unit_id,
                    )
                    continue

                # Record the failed outcome
                unit.record_outcome(False)

                # Reduce confidence (clamp to 0)
                unit.confidence = max(0.0, unit.confidence - penalty)

                # Decay weight reduction
                unit.decay_weight *= 0.9

                self.store.update(unit)
                units_modified += 1

                logger.debug(
                    "penalize_on_failure: unit %s confidence=%.2f decay=%.2f",
                    unit_id[:8], unit.confidence, unit.decay_weight,
                )

            result.units_modified = units_modified
            result.units_affected = units_modified
            result.details = {
                "confidence_penalty": penalty,
                "used_unit_ids": used_unit_ids,
            }

        except Exception as e:
            logger.error(
                "penalize_on_failure: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info(
            "penalize_on_failure: completed in %.1fms, modified %d units",
            result.duration_ms, result.units_modified,
        )
        return result
