"""QuarantineZone — two-stage memory unit promotion.

Created 2026-05-13 as part of the H-plan modular decomposition.

Before this module, every unit a candidate extracted went straight into the
candidate's storage. Bad units (LLM hallucinated tips, poisoned trajectories,
etc.) would influence the candidate's own evaluation AND, on canonical sync,
pollute the shared pool for all future rounds.

This module introduces a quarantine zone:

  candidate extraction → quarantine pool
                          │
                          │ promote_to_canonical() if usage_count >= K
                          │ AND success_rate >= threshold
                          ▼
                       canonical pool

Units in quarantine ARE retrievable during the candidate's eval (so we can
measure their utility), but only "graduate" to the cross-round canonical pool
after demonstrating real value over K usages.

API:
    qz = QuarantineZone(run_dir, min_usages=3, min_success_rate=0.40)
    qz.stage(units)           # add new units (called from take_in_memory)
    promoted = qz.promote()   # called by sync; returns list of units that
                              #   met criteria; remaining stay in quarantine.

The zone is stored at run_dir/quarantine/units.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    """Result of evaluating one quarantined unit for promotion."""

    unit_id: str
    promote: bool
    reason: str
    success_rate: float
    usage_count: int


class QuarantineZone:
    """Two-stage memory unit promotion.

    A unit enters quarantine on extraction. It "graduates" to canonical
    when:
      - usage_count >= min_usages (proven non-trivially active)
      - success_rate >= min_success_rate (proven net-positive)

    Units that fail are dropped (not promoted, not retried).
    """

    DEFAULT_MIN_USAGES = 3
    DEFAULT_MIN_SUCCESS_RATE = 0.40

    def __init__(
        self,
        run_dir: Path,
        min_usages: int = DEFAULT_MIN_USAGES,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    ):
        self.run_dir = Path(run_dir)
        self.min_usages = min_usages
        self.min_success_rate = min_success_rate
        self._zone_path = self.run_dir / "quarantine" / "units.json"
        self._zone_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> List[Dict[str, Any]]:
        if not self._zone_path.exists():
            return []
        try:
            with open(self._zone_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[quarantine] failed to load zone file: {e}")
            return []

    def _save(self, units: List[Dict[str, Any]]) -> None:
        with open(self._zone_path, "w", encoding="utf-8") as f:
            json.dump(units, f, indent=1, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stage(self, units: List[Dict[str, Any]]) -> int:
        """Add new units to quarantine. Returns count of newly added.

        Existing units (same id) are kept — caller is expected to update
        usage / success counters via update_stats() instead.
        """
        if not units:
            return 0
        existing = self._load()
        existing_ids = {u.get("id") for u in existing if u.get("id")}
        added = 0
        for u in units:
            uid = u.get("id") if isinstance(u, dict) else None
            if not uid or uid in existing_ids:
                continue
            existing.append(u)
            existing_ids.add(uid)
            added += 1
        if added > 0:
            self._save(existing)
            logger.info(f"[quarantine] staged {added} new unit(s); zone size = {len(existing)}")
        return added

    def update_stats(self, unit_id: str, used: bool, success: bool) -> None:
        """Bump usage_count / success_count for a quarantined unit.

        Called from the management pipeline after each task. Idempotent
        on a missing unit_id. Skips disk write if nothing changed.
        """
        if not used:
            return  # no-op: no counter to bump
        units = self._load()
        changed = False
        for u in units:
            if u.get("id") == unit_id:
                u["usage_count"] = int(u.get("usage_count", 0) or 0) + 1
                if success:
                    u["success_count"] = int(u.get("success_count", 0) or 0) + 1
                changed = True
                break
        if changed:
            self._save(units)

    def promote(self) -> List[Dict[str, Any]]:
        """Evaluate every quarantined unit and promote those that pass.

        Returns: list of units that should be added to canonical. The
        zone file is updated to drop both promoted and rejected units
        (rejected = met min_usages but failed success_rate).

        Units that have not yet reached min_usages remain in quarantine.
        """
        units = self._load()
        if not units:
            return []

        promoted: List[Dict[str, Any]] = []
        rejected_ids: List[str] = []
        kept: List[Dict[str, Any]] = []
        decisions: List[PromotionDecision] = []

        for u in units:
            usage = int(u.get("usage_count", 0) or 0)
            success = int(u.get("success_count", 0) or 0)
            success_rate = (success / usage) if usage > 0 else 0.0
            uid = u.get("id", "<missing>")

            if usage < self.min_usages:
                kept.append(u)
                decisions.append(PromotionDecision(
                    unit_id=uid, promote=False, reason=f"insufficient usage ({usage} < {self.min_usages})",
                    success_rate=success_rate, usage_count=usage,
                ))
            elif success_rate >= self.min_success_rate:
                promoted.append(u)
                decisions.append(PromotionDecision(
                    unit_id=uid, promote=True, reason=f"passed: success_rate={success_rate:.2f} ≥ {self.min_success_rate}",
                    success_rate=success_rate, usage_count=usage,
                ))
            else:
                rejected_ids.append(uid)
                decisions.append(PromotionDecision(
                    unit_id=uid, promote=False,
                    reason=f"rejected: success_rate={success_rate:.2f} < {self.min_success_rate} after {usage} usages",
                    success_rate=success_rate, usage_count=usage,
                ))

        self._save(kept)
        logger.info(
            "[quarantine] promote(): total=%d, promoted=%d, rejected=%d, still_in_quarantine=%d",
            len(units), len(promoted), len(rejected_ids), len(kept),
        )
        return promoted

    def count(self) -> int:
        return len(self._load())

    def list_units(self) -> List[Dict[str, Any]]:
        return self._load()


__all__ = ["QuarantineZone", "PromotionDecision"]
