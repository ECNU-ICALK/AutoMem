"""
TimeDecayOp — Periodic operation that applies exponential time decay
to all active memory units, with a recency refresh bonus for recently
accessed units.
"""

import time
import logging
from typing import Any, Dict

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class TimeDecayOp(BaseManageOp):
    """Apply exponential time decay to memory unit weights."""

    op_name = "time_decay"
    op_group = "maintenance"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 13
    rl_param_range = (0.8, 1.0)

    def execute(self, context: Dict[str, Any]) -> OpResult:
        """
        Apply decay to every active unit's decay_weight.

        Algorithm:
          1. decay_weight *= decay_rate
          2. If the unit was accessed within the last 24 hours,
             add access_refresh to decay_weight (clamped to 1.0)
          3. Clamp final decay_weight to [0.01, 1.0]

        Config keys:
          decay_rate (float): Multiplicative decay factor.  Default 0.95.
          access_refresh (float): Bonus for recently accessed units.  Default 0.1.

        Returns:
            OpResult with units_modified count.
        """
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            decay_rate = self.config.get("decay_rate", 0.95)
            access_refresh = self.config.get("access_refresh", 0.1)

            units = self.store.get_all(active_only=True)
            modified = 0

            for unit in units:
                old_weight = unit.decay_weight

                # Step 1: exponential decay
                unit.decay_weight *= decay_rate

                # Step 2: recency refresh for recently accessed units
                if unit.hours_since_access < 24:
                    unit.decay_weight = min(unit.decay_weight + access_refresh, 1.0)

                # Step 3: clamp to valid range
                unit.decay_weight = max(0.01, min(unit.decay_weight, 1.0))

                # Only persist if the value actually changed
                if abs(unit.decay_weight - old_weight) > 1e-9:
                    self.store.update(unit)
                    modified += 1

            result.units_modified = modified
            result.units_affected = len(units)
            result.details = {
                "total_active": len(units),
                "modified": modified,
                "decay_rate": decay_rate,
                "access_refresh": access_refresh,
            }
            logger.info(
                "time_decay: decayed %d/%d active units (rate=%.3f, refresh=%.2f).",
                modified, len(units), decay_rate, access_refresh,
            )

        except Exception as e:
            logger.error("time_decay failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result
