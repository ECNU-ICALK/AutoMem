"""
Signature-based deduplication operation.

Groups memory units by their content signature and merges exact duplicates,
keeping the unit with the highest effective_score and transferring stats
from victims to the survivor.
"""

import time
import logging
from collections import defaultdict
from typing import Any, Dict

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class SignatureDedupOp(BaseManageOp):
    """Deduplicate memory units that share the same content signature."""

    op_name = "signature_dedup"
    op_group = "deduplication"
    trigger_type = TriggerType.ON_INSERT
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = False
    rl_action_id = 4

    def execute(self, context: Dict[str, Any]) -> OpResult:
        result = OpResult(op_name=self.op_name)
        t0 = time.time()

        try:
            all_units = self.store.get_all()
            active_units = [u for u in all_units if u.is_active]

            # Group by signature
            sig_groups: Dict[str, list] = defaultdict(list)
            for unit in active_units:
                sig = unit.signature or unit.compute_signature()
                if sig:
                    sig_groups[sig].append(unit)

            total_deleted = 0
            total_modified = 0

            for sig, group in sig_groups.items():
                if len(group) <= 1:
                    continue

                # Keep the unit with the highest effective_score
                group.sort(key=lambda u: u.effective_score, reverse=True)
                survivor = group[0]
                victims = group[1:]

                # Transfer stats from victims to survivor
                for victim in victims:
                    survivor.usage_count += victim.usage_count
                    survivor.success_count += victim.success_count

                    # Delete victim (graph stores cascade edge cleanup automatically)
                    self.store.delete(victim.id)
                    total_deleted += 1
                    logger.debug(
                        "Signature dedup: deleted %s (sig=%s), merged into %s",
                        victim.id, sig, survivor.id,
                    )

                # Persist the updated survivor
                self.store.update(survivor)
                total_modified += 1

            result.triggered = total_deleted > 0
            result.units_affected = total_deleted + total_modified
            result.units_deleted = total_deleted
            result.units_modified = total_modified
            result.details = {
                "total_active_scanned": len(active_units),
                "duplicate_groups": sum(
                    1 for g in sig_groups.values() if len(g) > 1
                ),
            }

        except Exception:
            logger.exception("SignatureDedupOp failed")
            result.details["error"] = "exception during execution"

        result.duration_ms = (time.time() - t0) * 1000
        return result
