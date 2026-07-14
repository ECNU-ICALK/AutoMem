"""Adapt graph edge weights from usage/success feedback without deleting edges.

The effective weight is rebuilt as ``base_weight * feedback_multiplier`` on
every run.  It is therefore idempotent for unchanged statistics and can be
recomputed after relation reindexing instead of compounding a multiplier onto
the previous effective weight.
"""

import logging
import time
from typing import Any, Dict

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class EdgeWeightOptimizeOp(BaseManageOp):
    """Apply a bounded success-rate multiplier to graph relation weights."""

    op_name = "edge_weight_optimize"
    op_group = "graph_consolidate"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = False
    rl_action_id = 24

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "min_usage_for_adjust": 5,
        "low_success_threshold": 0.4,
        "high_success_threshold": 0.6,
        "max_penalty": 0.2,
        "max_boost": 0.2,
        "weight_floor": 0.0,
        "weight_ceiling": 1.0,
        "adjust_edge_types": ["SIMILAR"],
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name)

        try:
            if not self._is_graph_store():
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            cfg = {**self._DEFAULT_CONFIG, **(self.config or {})}
            min_adjust = int(cfg["min_usage_for_adjust"])
            low_threshold = float(cfg["low_success_threshold"])
            high_threshold = float(cfg["high_success_threshold"])
            max_penalty = float(cfg["max_penalty"])
            max_boost = float(cfg["max_boost"])
            floor = float(cfg["weight_floor"])
            ceiling = float(cfg["weight_ceiling"])
            adjust_types = set(cfg["adjust_edge_types"] or [])

            if not 0.0 <= low_threshold < high_threshold <= 1.0:
                raise ValueError(
                    "success thresholds must satisfy "
                    "0 <= low_success_threshold < high_success_threshold <= 1"
                )
            if not 0.0 <= max_penalty < 1.0 or max_boost < 0.0:
                raise ValueError("feedback bounds must be non-negative and penalty < 1")
            if floor > ceiling:
                raise ValueError("weight_floor must not exceed weight_ceiling")

            graph = self.store._graph
            adjusted = 0
            at_floor = 0
            at_ceiling = 0

            for s, t, _key, data in graph.edges(keys=True, data=True):
                if data.get("edge_type") not in adjust_types:
                    continue
                if not (str(s).startswith("m:") and str(t).startswith("m:")):
                    continue

                usage = int(data.get("usage_count", 0) or 0)
                if usage < min_adjust:
                    continue
                success = int(data.get("success_count", 0) or 0)
                rate = max(0.0, min(1.0, success / usage))

                current_weight = float(data.get("weight", 1.0) or 0.0)
                previous_base = data.get("base_weight")
                previous_multiplier = data.get("feedback_multiplier")
                base_weight = float(
                    current_weight if previous_base is None else previous_base
                )
                if rate > high_threshold:
                    span = 1.0 - high_threshold
                    fraction = (rate - high_threshold) / span if span else 1.0
                    multiplier = 1.0 + max_boost * fraction
                elif rate < low_threshold:
                    span = low_threshold
                    fraction = (low_threshold - rate) / span if span else 1.0
                    multiplier = 1.0 - max_penalty * fraction
                else:
                    multiplier = 1.0

                new_weight = max(floor, min(ceiling, base_weight * multiplier))
                data["base_weight"] = base_weight
                data["feedback_multiplier"] = multiplier
                data["weight"] = new_weight
                if (
                    new_weight != current_weight
                    or previous_base != base_weight
                    or previous_multiplier != multiplier
                ):
                    adjusted += 1
                if new_weight == floor:
                    at_floor += 1
                if new_weight == ceiling:
                    at_ceiling += 1

            if adjusted:
                try:
                    self.store.save()
                except Exception as e:
                    logger.warning("edge_weight_optimize: store.save failed: %s", e)

            result.triggered = bool(adjusted)
            result.units_affected = adjusted
            result.details = {
                "edges_adjusted": adjusted,
                "edges_at_floor": at_floor,
                "edges_at_ceiling": at_ceiling,
                "edges_pruned": 0,
            }
            if adjusted:
                logger.info(
                    "edge_weight_optimize: adjusted=%d floor=%d ceiling=%d",
                    adjusted, at_floor, at_ceiling,
                )

        except Exception as e:
            logger.error("edge_weight_optimize: execution failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result
