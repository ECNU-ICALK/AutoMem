"""
BoostOnSuccessOp — Boost confidence for memories used in a successful task.

Part of the 'failure_adjustment' operation group.
"""

import time
import logging
from typing import Any, Dict, List

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class BoostOnSuccessOp(BaseManageOp):
    """
    After a successful task, boost confidence of all memory units that were
    used during the task.  This reinforces high-value memories so they rank
    higher in future retrievals.
    """

    op_name = "boost_on_success"
    op_group = "failure_adjustment"
    trigger_type = TriggerType.POST_TASK
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 9
    rl_param_range = (0.0, 0.2)

    _DEFAULT_CONFIG = {
        "success_boost": 0.05,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name)

        try:
            # Only trigger on task success
            if not context.get("task_succeeded", False):
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            result.triggered = True
            boost = self.config.get(
                "success_boost",
                self._DEFAULT_CONFIG["success_boost"],
            )

            used_unit_ids: List[str] = context.get("used_unit_ids", [])
            units_modified = 0

            for unit_id in used_unit_ids:
                unit = self.store.get(unit_id)
                if unit is None:
                    logger.warning(
                        "boost_on_success: unit %s not found, skipping",
                        unit_id,
                    )
                    continue

                # Record the successful outcome
                unit.record_outcome(True)

                # Boost confidence (clamp to 1.0)
                unit.confidence = min(1.0, unit.confidence + boost)

                self.store.update(unit)
                units_modified += 1

                logger.debug(
                    "boost_on_success: unit %s confidence=%.2f",
                    unit_id[:8], unit.confidence,
                )

            result.units_modified = units_modified
            result.units_affected = units_modified
            result.details = {
                "success_boost": boost,
                "used_unit_ids": used_unit_ids,
            }

        except Exception as e:
            logger.error(
                "boost_on_success: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info(
            "boost_on_success: completed in %.1fms, modified %d units",
            result.duration_ms, result.units_modified,
        )
        return result
