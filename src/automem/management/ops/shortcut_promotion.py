"""
ShortcutPromotionOp — Tool-Manager-inspired: promote reusable shortcut units.

After a successful task, if a SHORTCUT unit was retrieved and used, bump its
usage_count/confidence aggressively — so that high-reuse shortcuts behave
like "first-class tools" in future retrievals.

Works alongside boost_on_success (which applies uniformly) by applying an
EXTRA boost specifically to SHORTCUT type units — i.e. differentiating
reusable macros from ephemeral insights / tips.
"""

import time
import logging
from typing import Any, Dict, List

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnitType

logger = logging.getLogger(__name__)


class ShortcutPromotionOp(BaseManageOp):
    op_name = "shortcut_promotion"
    op_group = "tool_manager"
    trigger_type = TriggerType.POST_TASK
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 20
    rl_param_range = (0.0, 0.3)

    _DEFAULT_CONFIG = {
        "shortcut_boost": 0.10,          # extra confidence boost beyond boost_on_success
        "promote_usage_threshold": 3,    # >=this many uses -> mark as "promoted"
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name)

        try:
            if not context.get("task_succeeded", False):
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            result.triggered = True
            boost = self.config.get(
                "shortcut_boost", self._DEFAULT_CONFIG["shortcut_boost"]
            )
            thresh = int(self.config.get(
                "promote_usage_threshold",
                self._DEFAULT_CONFIG["promote_usage_threshold"],
            ))

            used_unit_ids: List[str] = context.get("used_unit_ids", []) or []
            promoted_ids = []
            boosted_ids = []

            for unit_id in used_unit_ids:
                unit = self.store.get(unit_id)
                if unit is None or unit.type != MemoryUnitType.SHORTCUT:
                    continue
                # Tool-Manager gate: only promote schema-valid shortcuts.
                # (A shortcut without `tool_valid` in applicable_task_types
                # has not passed ShortcutValidationOp — treat as plain memory,
                # no tool-style promotion.)
                if "tool_valid" not in unit.applicable_task_types:
                    continue
                unit.confidence = min(1.0, unit.confidence + boost)
                unit.usage_count += 1
                unit.success_count += 1
                boosted_ids.append(unit_id)
                if unit.usage_count >= thresh:
                    # Promote: mark with metadata flag (persisted via to_dict)
                    if "promoted" not in unit.applicable_task_types:
                        unit.applicable_task_types = list(
                            unit.applicable_task_types
                        ) + ["promoted"]
                        promoted_ids.append(unit_id)
                self.store.update(unit)

            result.units_modified = len(boosted_ids)
            result.details = {
                "shortcut_boost": boost,
                "boosted_unit_ids": boosted_ids,
                "promoted_unit_ids": promoted_ids,
            }
            result.duration_ms = (time.time() - t0) * 1000
            return result
        except Exception as e:
            logger.exception("ShortcutPromotionOp failed: %s", e)
            result.error = str(e)
            result.duration_ms = (time.time() - t0) * 1000
            return result
