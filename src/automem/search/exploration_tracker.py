"""ExplorationTracker — proposer-side diversity book-keeping.

Smart-5 fix (2026-05-16). Records, per architecture-space dimension, how many
times each value has been evaluated. Surfaces under-explored values to the
proposer so later rounds don't keep proposing variations of the same Pareto
sweet-spot.

Used by ``automem_search.generate_candidates``: every round, before invoking
the proposer LLM, the tracker is updated with every previously-evaluated
architecture, and ``render_hints()`` injects an "Under-explored options" block
into the proposer prompt.

The tracker is intentionally lightweight (no LLM calls, no persistence beyond
the in-memory dict — Pareto front already stores past architectures and the
tracker can be rebuilt from them on resume).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


class ExplorationTracker:
    """Tracks per-dimension evaluation counts across all proposed architectures.

    The space dict shape mirrors RECOMMENDED_ARCHITECTURE_SPACE:
        {
          "extract_types":    [...],
          "storage_types":    [...],
          "retrieval_types":  [...],
          "management_types": [...],
        }
    """

    # Only these 4 are actual proposer-decision dimensions; relation_types
    # appears in RECOMMENDED_ARCHITECTURE_SPACE for documentation but is
    # runtime-only (graph storage emits relations automatically).
    _PROPOSER_DIMS = (
        "extract_types", "storage_types", "retrieval_types", "management_types",
    )

    def __init__(self, recommended_space: Dict[str, List[str]]):
        # Smart-5 fix (2026-05-16): subset to proposer-decision dims so the
        # hint block stays focused on actionable values.
        self.space = {
            dim: recommended_space[dim]
            for dim in self._PROPOSER_DIMS
            if dim in recommended_space
        }
        # Dimension → Counter[value → eval_count]
        self.counts: Dict[str, Counter] = {dim: Counter() for dim in self.space.keys()}
        self.total_evaluated = 0

    # ──────────────────────────────────────────────────────────────────
    # Update
    # ──────────────────────────────────────────────────────────────────
    def update(self, architecture: Dict[str, Any]) -> None:
        """Record one architecture evaluation across all four dimensions."""
        if not isinstance(architecture, dict):
            return
        self.total_evaluated += 1

        # extract_types — a list, count each member
        for et in architecture.get("extract_types", []) or []:
            self.counts["extract_types"][et] += 1

        # storage_routing — dict {extract_type: storage_backend}; count backends
        routing = architecture.get("storage_routing", {}) or {}
        for backend in routing.values():
            self.counts["storage_types"][backend] += 1

        # retrieval — scalar
        ret = architecture.get("retrieval")
        if ret:
            self.counts["retrieval_types"][ret] += 1

        # management — scalar (may be "management" or "management_preset")
        mgmt = architecture.get("management_preset", architecture.get("management"))
        if mgmt:
            self.counts["management_types"][mgmt] += 1

    def replay(self, architectures: List[Dict[str, Any]]) -> None:
        """Update from a list of architectures (e.g. on resume from Pareto front)."""
        for arch in architectures:
            self.update(arch)

    # ──────────────────────────────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────────────────────────────
    def under_explored(self, dim: str, min_count: int = 2) -> List[str]:
        """Return values in ``dim`` evaluated < ``min_count`` times."""
        space_vals = self.space.get(dim, [])
        return [v for v in space_vals if self.counts[dim].get(v, 0) < min_count]

    def coverage_summary(self) -> Dict[str, Dict[str, int]]:
        """Return per-dim {value: count} dict for all values in the space."""
        out: Dict[str, Dict[str, int]] = {}
        for dim, values in self.space.items():
            out[dim] = {v: self.counts[dim].get(v, 0) for v in values}
        return out

    # ──────────────────────────────────────────────────────────────────
    # Rendering for prompt
    # ──────────────────────────────────────────────────────────────────
    def render_hints(self, min_count: int = 2, max_per_dim: int = 5) -> str:
        """Format under-explored options as a readable hint block.

        Returns empty string when total_evaluated == 0 (round 1 — no history).
        """
        if self.total_evaluated == 0:
            return ""

        lines: List[str] = []
        for dim, _ in self.space.items():
            under = self.under_explored(dim, min_count=min_count)[:max_per_dim]
            if not under:
                continue
            current_counts = ", ".join(
                f"{v}×{self.counts[dim].get(v, 0)}"
                for v in self.space[dim]
            )
            short_dim = dim.replace("_types", "").replace("_", "-")
            lines.append(
                f"  {short_dim:<11}: under-tested → {under}  (all: {current_counts})"
            )
        if not lines:
            return ""
        return (
            f"Exploration coverage across {self.total_evaluated} evaluated architectures:\n"
            + "\n".join(lines)
            + "\n\nPrefer proposals that exercise an under-tested value above before "
            "rehashing already-tested combos."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "counts": {dim: dict(c) for dim, c in self.counts.items()},
        }

    # ──────────────────────────────────────────────────────────────────
    # Persistence (optional, but allows surviving across resume)
    # ──────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str,
        recommended_space: Dict[str, List[str]],
    ) -> "ExplorationTracker":
        tracker = cls(recommended_space)
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return tracker
        tracker.total_evaluated = data.get("total_evaluated", 0)
        for dim, c in (data.get("counts") or {}).items():
            tracker.counts[dim] = Counter(c)
        return tracker


__all__ = ["ExplorationTracker"]
