"""
AccessStatsUpdateOp — Post-task operation that updates access statistics
for memory units that were used during the most recent task.
"""

import time
import logging
from typing import Any, Dict

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class AccessStatsUpdateOp(BaseManageOp):
    """Update access statistics for memory units used in the latest task."""

    op_name = "access_stats_update"
    op_group = "maintenance"
    trigger_type = TriggerType.POST_TASK
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 12

    def execute(self, context: Dict[str, Any]) -> OpResult:
        """
        Update access stats for each unit referenced in context['used_unit_ids'].

        For each unit id:
          1. Fetch the unit from the store
          2. Call unit.record_access() to bump access_count and last_accessed
          3. Persist the updated unit

        Args:
            context: Must contain 'used_unit_ids' (List[str]).

        Returns:
            OpResult with units_modified = number of successfully updated units.
        """
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            used_ids = context.get("used_unit_ids", [])
            if not used_ids:
                logger.debug("access_stats_update: no used_unit_ids in context, skipping.")
                result.duration_ms = (time.time() - t0) * 1000
                return result

            modified = 0
            missing = 0

            for unit_id in used_ids:
                unit = self.store.get(unit_id)
                if unit is None:
                    logger.warning("access_stats_update: unit %s not found in store.", unit_id)
                    missing += 1
                    continue

                unit.record_access()
                self.store.update(unit)
                modified += 1

            result.units_modified = modified
            result.units_affected = modified + missing
            result.details = {
                "used_ids_count": len(used_ids),
                "modified": modified,
                "missing": missing,
            }
            logger.info(
                "access_stats_update: updated %d/%d units (%d missing).",
                modified, len(used_ids), missing,
            )

        except Exception as e:
            logger.error("access_stats_update failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result
