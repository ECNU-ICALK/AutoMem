"""
UtilityAuditOp — Periodic harmful-memory pruning based on observed success rate.

Differs from ``score_based_prune`` (which thresholds the composite ``effective_score``
that blends confidence, success rate, and recency decay): utility_audit looks at
the *empirical* success_rate alone, computed from how often a unit was retrieved
and how often the task succeeded after using it.

Rationale (MODULE_AUDIT_REPORT.md, idea 3 step A):
  Marginal success_rate is not the same as causal contribution, but for units
  that have been retrieved enough times we can detect "harmful or net-zero"
  memories — they get retrieved, the task fails more often than not, yet they
  survive because their composite score still hovers above the prune threshold.
  This op deactivates them so they stop being injected into agent prompts.

Config keys:
    min_usage_count (int):     minimum usage_count required before judging a unit.
                               Default 3 — below this, the empirical rate is too
                               noisy.
    harmful_threshold (float): success_rate strictly below this → deactivate.
                               Default 0.30. Combined with min_usage_count this
                               targets units that produced at least 3 retrievals
                               and hit < 1 success per 3 uses.
    require_negative_lift (bool): if True (default), only deactivate when the
                               unit's success_rate is meaningfully BELOW the
                               provider-wide success rate (success_rate <
                               provider_baseline - margin). Prevents pruning
                               low-rate units in already-hard task families.
    margin (float):            tolerance below provider_baseline. Default 0.10.
    leakage_risk_threshold_bump (dict): (added 2026-04-28) per-leakage_risk
                               threshold adjustment. High-leakage units must
                               clear a stricter success_rate bar to survive.
                               Default {"high": +0.20, "medium": +0.10}.
                               Implementation: unit's effective threshold =
                               harmful_threshold + bump[leakage_risk].
"""

import time
import logging
from typing import Any, Dict, List, Optional

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit

logger = logging.getLogger(__name__)


class UtilityAuditOp(BaseManageOp):
    """Deactivate units whose empirical success rate is below threshold."""

    op_name = "utility_audit"
    op_group = "maintenance"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 16

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        min_usage = int(self.config.get("min_usage_count", 3))
        harmful_threshold = float(self.config.get("harmful_threshold", 0.30))
        require_negative_lift = bool(self.config.get("require_negative_lift", True))
        margin = float(self.config.get("margin", 0.10))
        # Leakage-risk-aware threshold bumps (added 2026-04-28). Stricter bars
        # for high/medium leakage_risk units so that a "high"-tagged unit
        # surviving a few mixed-result retrievals gets pruned faster than a
        # "low"-tagged unit with the same success_rate.
        leakage_bumps: Dict[str, float] = dict(
            self.config.get("leakage_risk_threshold_bump", {"high": 0.20, "medium": 0.10})
        )

        # Stale-unused branch (added 2026-04-28). Bypasses min_usage gate
        # for units that are old but never retrieved.
        # Codex Q4-6 fix (2026-04-28): default to OFF for in-candidate
        # invocation, because age_hours is computed from `created_at`
        # which is preserved across canonical→candidate import. A
        # candidate that runs for ~1 hour against a 5-day-old canonical
        # pool would otherwise deactivate every imported unit on its very
        # first periodic tick, before the agent ever has a chance to
        # retrieve them. The CANONICAL-level periodic invocation
        # (automem_search._run_canonical_periodic_ops) re-enables this
        # branch by passing handle_stale_unused=True explicitly, where
        # the wall-clock age comparison is appropriate.
        handle_stale_unused = bool(self.config.get("handle_stale_unused", False))
        stale_unused_age_hours = float(self.config.get("stale_unused_age_hours", 24.0))

        try:
            all_active = self.store.get_all(active_only=True)
            total_active = len(all_active)

            eligible: List[MemoryUnit] = [
                u for u in all_active if int(getattr(u, "usage_count", 0)) >= min_usage
            ]
            stale_unused: List[MemoryUnit] = []
            if handle_stale_unused:
                stale_unused = [
                    u for u in all_active
                    if int(getattr(u, "usage_count", 0)) == 0
                    and float(getattr(u, "age_hours", 0.0)) >= stale_unused_age_hours
                ]

            # Provider-wide baseline (mean success_rate over all eligible units).
            # Used only when require_negative_lift is True.
            provider_baseline: Optional[float] = None
            if require_negative_lift and eligible:
                rates = [float(u.success_rate) for u in eligible]
                provider_baseline = sum(rates) / len(rates)

            harmful: List[MemoryUnit] = []
            leakage_pruned = 0
            for unit in eligible:
                rate = float(unit.success_rate)
                # Effective threshold honors leakage_risk: high/medium bumps it up.
                lr = str((unit.content or {}).get("leakage_risk", "low")).strip().lower()
                effective_threshold = harmful_threshold + float(leakage_bumps.get(lr, 0.0))

                below_threshold = rate < effective_threshold
                if not below_threshold:
                    continue
                if require_negative_lift and provider_baseline is not None:
                    if rate >= provider_baseline - margin:
                        # Not meaningfully worse than the provider-wide baseline,
                        # so the bad rate is more likely "hard task family" than
                        # "harmful memory". Skip — UNLESS leakage_risk is high,
                        # in which case the precautionary principle wins.
                        if lr != "high":
                            continue
                harmful.append(unit)
                if lr in ("high", "medium"):
                    leakage_pruned += 1

            for unit in harmful:
                self._deactivate(unit)

            # Stale-unused: deactivate separately, count separately so the
            # log distinguishes "harmful" from "never used".
            for unit in stale_unused:
                self._deactivate(unit)

            # Codex Q4-2 fix (2026-04-28): rebuild FAISS after the batch.
            # `update()` only persists the is_active flag; vector/hybrid
            # backends do not refresh their FAISS index on metadata-only
            # changes, so deactivated rows would still occupy retrieval
            # slots. Without this rebuild, utility_audit silently
            # under-prunes for vector / hybrid architectures.
            if (harmful or stale_unused):
                try:
                    self._rebuild_faiss_if_needed()
                except Exception as e:
                    logger.debug(f"utility_audit FAISS rebuild skipped: {e}")

            result.units_affected = total_active
            result.units_deleted = len(harmful) + len(stale_unused)
            result.details = {
                "total_active": total_active,
                "eligible_for_audit": len(eligible),
                "deactivated_harmful": len(harmful),
                "deactivated_stale_unused": len(stale_unused),
                "deactivated_due_to_leakage": leakage_pruned,
                "min_usage_count": min_usage,
                "harmful_threshold": harmful_threshold,
                "leakage_bumps": leakage_bumps,
                "stale_unused_age_hours": stale_unused_age_hours,
                "handle_stale_unused": handle_stale_unused,
                "require_negative_lift": require_negative_lift,
                "provider_baseline": (
                    round(provider_baseline, 4)
                    if provider_baseline is not None
                    else None
                ),
                "deactivated_ids": [u.id for u in (harmful + stale_unused)][:32],  # cap log spam
            }
            logger.info(
                "utility_audit: harmful=%d/%d eligible, stale_unused=%d "
                "(min_usage=%d, harmful_threshold=%.2f, baseline=%s, stale_age_h=%.1f)",
                len(harmful),
                len(eligible),
                len(stale_unused),
                min_usage,
                harmful_threshold,
                f"{provider_baseline:.3f}" if provider_baseline is not None else "n/a",
                stale_unused_age_hours,
            )

        except Exception as e:
            logger.error("utility_audit failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _deactivate(self, unit: MemoryUnit) -> None:
        """Soft-deactivate a unit (mirrors score_based_prune behaviour)."""
        unit.is_active = False
        # Some graph backends mirror is_active onto the graph node attribute.
        if self._is_graph_store():
            try:
                nid = self.store._content_nid(unit.id)
                if nid in self.store._graph:
                    self.store._graph.nodes[nid]["is_active"] = False
            except AttributeError:
                pass
        self.store.update(unit)

    def _is_graph_store(self) -> bool:
        return type(self.store).__name__ in {"GraphStore", "GraphStorage", "LLMGraphStorage"}
