"""Pareto front maintenance for multi-objective architecture search.

Objectives (all maximized after normalization):
  - accuracy:     fraction of tasks solved correctly
  - memory_lift:  accuracy - no_memory_baseline (memory contribution)
  - hit_rate:     fraction of retrievals returning non-empty results
  - token_eff:    1 - normalized_token_cost  (lower cost = higher efficiency)

An architecture A dominates B if A >= B on ALL objectives and A > B on
at least one objective.  The Pareto front is the set of non-dominated
configurations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParetoEntry:
    """A single evaluated architecture on (or off) the Pareto front."""

    config_id: str                          # unique, e.g. "r1_c0"
    architecture: Dict[str, Any]            # flat arch dict as given to LLM
    round_id: int = 0

    # Objective values (all in [0, 1], all maximized)
    accuracy: float = 0.0
    memory_lift: float = 0.0               # may be negative if memory hurts
    hit_rate: float = 0.0
    token_eff: float = 0.0                 # 1 - normalized_token_cost

    # Scalar fitness (weighted sum, for quick ranking)
    fitness: float = 0.0

    # Number of pooled measurements behind the objective values (protocol
    # v2 "pooled" mode; always 1 in legacy "max" mode).
    n_evals: int = 1

    # Raw metrics kept for logging
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    # Attribution breakdown for this candidate
    attribution_summary: Dict[str, Any] = field(default_factory=dict)

    def objectives(self) -> Tuple[float, float]:
        """Return objective tuple for dominance checks.

        Accuracy-first profile (2026-05-13): dominance considers only
        (accuracy, memory_lift). hit_rate and token_eff are still recorded
        in raw_metrics for logging/inspection but do NOT determine which
        candidates enter or leave the Pareto front. Rationale: run9 evidence
        (ledger principle P004) showed higher hit_rate does not translate to
        higher accuracy when failures are reasoning-bound; allowing hit_rate
        to dominate let low-acc/high-hit candidates clutter the front and
        anchor the proposer's exploitation phase to retrieval geometry
        rather than the actual objective (task accuracy).
        """
        return (self.accuracy, self.memory_lift)

    def dominates(self, other: "ParetoEntry") -> bool:
        """Return True if self Pareto-dominates other."""
        s = self.objectives()
        o = other.objectives()
        return all(a >= b for a, b in zip(s, o)) and any(a > b for a, b in zip(s, o))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "architecture": self.architecture,
            "round_id": self.round_id,
            "accuracy": self.accuracy,
            "memory_lift": self.memory_lift,
            "hit_rate": self.hit_rate,
            "token_eff": self.token_eff,
            "fitness": self.fitness,
            "n_evals": self.n_evals,
            "raw_metrics": self.raw_metrics,
            "attribution_summary": self.attribution_summary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParetoEntry":
        return cls(
            config_id=d.get("config_id", ""),
            architecture=d.get("architecture", {}),
            round_id=d.get("round_id", 0),
            accuracy=d.get("accuracy", 0.0),
            memory_lift=d.get("memory_lift", 0.0),
            hit_rate=d.get("hit_rate", 0.0),
            token_eff=d.get("token_eff", 0.0),
            fitness=d.get("fitness", 0.0),
            n_evals=int(d.get("n_evals", 1)),
            raw_metrics=d.get("raw_metrics", {}),
            attribution_summary=d.get("attribution_summary", {}),
        )

    def summary_str(self) -> str:
        arch = self.architecture
        return (
            f"[{self.config_id}] "
            f"acc={self.accuracy:.3f} lift={self.memory_lift:+.3f} "
            f"hit={self.hit_rate:.3f} teff={self.token_eff:.3f} "
            f"fit={self.fitness:.4f} | "
            f"extract={arch.get('extract_types',[])} "
            f"storage={arch.get('storage_routing',{})} "
            f"ret={arch.get('retrieval','?')} "
            f"mgmt={arch.get('management','?')}"
        )


class ParetoFront:
    """Maintains the non-dominated set of evaluated architectures."""

    def __init__(self) -> None:
        self._front: List[ParetoEntry] = []
        self._all_evaluated: List[ParetoEntry] = []   # history (dominated + front)
        # Protocol-v2 pooled measurement mode ("max" = legacy behavior).
        # In "pooled" mode repeated measurements of the SAME architecture
        # are averaged (running mean per objective) instead of keeping the
        # single highest-fitness draw — removes the winner's-curse bias
        # that max-selection puts on noisy 30-50 task evaluations.
        self.measurement_mode: str = "max"
        # arch_key -> accumulated sums, independent of front membership so
        # an architecture that drops off the front keeps its history.
        self._arch_stats: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _arch_key(architecture: Dict[str, Any]) -> str:
        try:
            return json.dumps(architecture, sort_keys=True, default=str)
        except Exception:
            return repr(architecture)

    def _accumulate(self, entry: ParetoEntry) -> ParetoEntry:
        """Update the per-arch accumulator with a raw measurement and return
        a pooled entry whose objectives are the running means."""
        key = self._arch_key(entry.architecture)
        st = self._arch_stats.setdefault(key, {
            "n": 0.0, "accuracy": 0.0, "memory_lift": 0.0,
            "hit_rate": 0.0, "token_eff": 0.0, "fitness": 0.0,
        })
        st["n"] += 1.0
        for f_name in ("accuracy", "memory_lift", "hit_rate", "token_eff", "fitness"):
            st[f_name] += float(getattr(entry, f_name))
        n = st["n"]
        pooled = ParetoEntry(
            config_id=entry.config_id,
            architecture=entry.architecture,
            round_id=entry.round_id,
            accuracy=st["accuracy"] / n,
            memory_lift=st["memory_lift"] / n,
            hit_rate=st["hit_rate"] / n,
            token_eff=st["token_eff"] / n,
            fitness=st["fitness"] / n,
            n_evals=int(n),
            raw_metrics=dict(entry.raw_metrics),
            attribution_summary=dict(entry.attribution_summary),
        )
        pooled.raw_metrics["pooled"] = {
            "n_evals": int(n),
            "last_accuracy": entry.accuracy,
            "last_fitness": entry.fitness,
        }
        return pooled

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, entry: ParetoEntry) -> bool:
        """Add an entry and update the front.

        Returns True if the entry was added to (or retained on) the front.

        Codex Q5-4 fix (2026-04-28): dedup by architecture. With LLM
        stochasticity two evaluations of the same architecture can
        produce slightly different metrics (one wins on hit_rate, the
        other on accuracy → both non-dominated under 4-obj Pareto). The
        front then carries duplicate slots and the architect "learns"
        that an architecture is diverse just by being lucky twice. Now
        we keep only the higher-fitness instance per canonicalized
        architecture key.
        """
        # A crash may happen after candidate evaluation but before the Pareto
        # checkpoint. On resume the same round/config is replayed; counting it
        # again would bias pooled means toward that one measurement.
        existing_config = next(
            (item for item in self._all_evaluated if item.config_id == entry.config_id),
            None,
        )
        if existing_config is not None:
            arch_key = self._arch_key(existing_config.architecture)
            logger.info("Pareto entry %s already recorded; skipping replay.", entry.config_id)
            return any(self._arch_key(item.architecture) == arch_key for item in self._front)

        self._all_evaluated.append(entry)

        # Protocol-v2 pooled mode (A1): fold this raw measurement into the
        # per-architecture running mean and continue with the POOLED entry.
        # The history above keeps the raw draw for variance diagnostics.
        if self.measurement_mode == "pooled":
            entry = self._accumulate(entry)

        # Architecture canonical key (deterministic JSON of normalized arch)
        try:
            import json as _json
            arch_key = _json.dumps(entry.architecture, sort_keys=True, default=str)
        except Exception:
            arch_key = repr(entry.architecture)

        # If a same-architecture entry is already on the front, drop the
        # old one then fall through to the standard dominance check on the
        # new entry.
        #
        # Codex Q6-A3 fix (2026-04-28): the previous version returned
        # immediately after replacement, never re-checking whether some
        # OTHER front member Pareto-dominates the replacement. With LLM
        # stochasticity this lets a dominated architecture sit on the
        # front (and into the architect's context) just because its
        # scalar fitness happened to be higher than the previous
        # same-architecture instance. Now: remove the duplicate, then
        # run the normal dominated-by-existing / surviving logic so
        # only Pareto-non-dominated entries stay on the front.
        same_arch_indices: List[int] = []
        same_arch_max_fitness = float("-inf")
        for i, e in enumerate(self._front):
            try:
                e_key = _json.dumps(e.architecture, sort_keys=True, default=str)
            except Exception:
                e_key = repr(e.architecture)
            if e_key == arch_key:
                same_arch_indices.append(i)
                same_arch_max_fitness = max(same_arch_max_fitness, e.fitness)

        # Codex Q7-A5 fix (2026-04-28): defer dropping the same-arch
        # duplicate until the replacement is confirmed non-dominated.
        # Previously we removed `old` immediately, then if `new` turned
        # out to be dominated by ANOTHER architecture (e.g. old had high
        # token_eff but new traded it for accuracy), we returned False
        # and the front lost a valid Pareto point with no replacement.
        # Now: snapshot the duplicates, drop, re-check; on rejection
        # restore.
        same_arch_entries: List[ParetoEntry] = []
        if same_arch_indices:
            # Pooled mode: the pooled entry is the architecture's current
            # best estimate — it always supersedes the stale front copy,
            # even when the pooled fitness went DOWN (that is the point:
            # no max-keeping under noise).
            if self.measurement_mode != "pooled" and entry.fitness <= same_arch_max_fitness:
                logger.debug(
                    "Entry %s has same architecture as a front member with "
                    "equal-or-higher fitness; not added.",
                    entry.config_id,
                )
                return False
            same_arch_entries = [self._front[i] for i in same_arch_indices]
            self._front = [
                e for i, e in enumerate(self._front) if i not in same_arch_indices
            ]

        # Check if entry is dominated by any existing front member
        dominated_by_existing = any(e.dominates(entry) for e in self._front)
        if dominated_by_existing:
            # Restore the duplicates we tentatively removed; the front
            # had a valid same-arch point and rejecting `entry` should
            # not silently shrink the front. (Legacy mode only — in pooled
            # mode the removed copy is a stale estimate of the SAME
            # architecture; restoring it would resurrect exactly the
            # inflated draw pooling exists to remove, so the architecture
            # correctly leaves the front instead.)
            if same_arch_entries and self.measurement_mode != "pooled":
                self._front.extend(same_arch_entries)
                logger.debug(
                    "Entry %s superseded same-arch but is dominated by "
                    "another architecture; restoring %d original duplicate(s).",
                    entry.config_id, len(same_arch_entries),
                )
            else:
                logger.debug(
                    "Entry %s is dominated; not added to front.", entry.config_id,
                )
            return False

        if same_arch_entries:
            if self.measurement_mode == "pooled":
                logger.info(
                    "Entry %s refreshes %d same-architecture front member(s) "
                    "with pooled estimate n=%d (fitness %.4f; previous %.4f).",
                    entry.config_id,
                    len(same_arch_entries),
                    entry.n_evals,
                    entry.fitness,
                    same_arch_max_fitness,
                )
            else:
                logger.info(
                    "Entry %s supersedes %d same-architecture front member(s) "
                    "(fitness %.4f > prev_max %.4f).",
                    entry.config_id,
                    len(same_arch_entries),
                    entry.fitness,
                    same_arch_max_fitness,
                )

        # Remove existing front members that entry dominates
        surviving = [e for e in self._front if not entry.dominates(e)]
        dominated_count = len(self._front) - len(surviving)
        surviving.append(entry)
        self._front = surviving

        if dominated_count:
            logger.info(
                "Entry %s dominates %d existing front member(s); front size: %d",
                entry.config_id, dominated_count, len(self._front),
            )
        else:
            logger.info(
                "Entry %s added to Pareto front (no domination); front size: %d",
                entry.config_id, len(self._front),
            )
        return True

    def best(self) -> Optional[ParetoEntry]:
        """Return the entry with highest fitness (primary tiebreaker)."""
        if not self._front:
            return None
        return max(self._front, key=lambda e: e.fitness)

    def top_k(self, k: int = 5) -> List[ParetoEntry]:
        """Return up to k front entries sorted by fitness descending."""
        return sorted(self._front, key=lambda e: e.fitness, reverse=True)[:k]

    def top_k_all(self, k: int = 5) -> List[ParetoEntry]:
        """Rank distinct architectures across all completed evaluations.

        M3 runoff must have access to dominated candidates too; otherwise a
        one-point Pareto front degenerates a requested top-2 runoff into a
        single contender. Pooled mode ranks each architecture by its running
        mean, while legacy mode retains its best observed measurement.
        """

        grouped: Dict[str, List[ParetoEntry]] = {}
        for entry in self._all_evaluated:
            grouped.setdefault(self._arch_key(entry.architecture), []).append(entry)

        ranked: List[ParetoEntry] = []
        for arch_key, entries in grouped.items():
            if self.measurement_mode != "pooled":
                ranked.append(max(entries, key=lambda item: item.fitness))
                continue

            latest = entries[-1]
            stats = self._arch_stats.get(arch_key)
            if stats and float(stats.get("n", 0.0)) > 0:
                count = float(stats["n"])
                sums = stats
            else:
                count = float(len(entries))
                sums = {
                    field_name: sum(float(getattr(item, field_name)) for item in entries)
                    for field_name in (
                        "accuracy",
                        "memory_lift",
                        "hit_rate",
                        "token_eff",
                        "fitness",
                    )
                }
            ranked.append(
                ParetoEntry(
                    config_id=latest.config_id,
                    architecture=dict(latest.architecture),
                    round_id=latest.round_id,
                    accuracy=float(sums["accuracy"]) / count,
                    memory_lift=float(sums["memory_lift"]) / count,
                    hit_rate=float(sums["hit_rate"]) / count,
                    token_eff=float(sums["token_eff"]) / count,
                    fitness=float(sums["fitness"]) / count,
                    n_evals=int(count),
                    raw_metrics=dict(latest.raw_metrics),
                    attribution_summary=dict(latest.attribution_summary),
                )
            )
        return sorted(ranked, key=lambda item: item.fitness, reverse=True)[:k]

    def size(self) -> int:
        return len(self._front)

    def all_evaluated(self) -> List[ParetoEntry]:
        return list(self._all_evaluated)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "front": [e.to_dict() for e in self._front],
            "all_evaluated": [e.to_dict() for e in self._all_evaluated],
            "measurement_mode": self.measurement_mode,
            "arch_stats": self._arch_stats,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParetoFront":
        pf = cls()
        pf._front = [ParetoEntry.from_dict(x) for x in d.get("front", [])]
        pf._all_evaluated = [ParetoEntry.from_dict(x) for x in d.get("all_evaluated", [])]
        pf.measurement_mode = d.get("measurement_mode", "max")
        pf._arch_stats = d.get("arch_stats", {}) or {}
        return pf

    def save(self, path: str) -> None:
        import os
        import tempfile

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=p.parent, suffix=".pareto.json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, p)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str) -> "ParetoFront":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Context builder (for LLM prompt injection)
    # ------------------------------------------------------------------

    def to_llm_context(self, max_front: int = 5, max_history: int = 15) -> Dict[str, Any]:
        """Build a compact context dict for injection into the search LLM prompt."""
        front_entries = self.top_k(max_front)
        recent_history = sorted(
            self._all_evaluated, key=lambda e: e.round_id, reverse=True
        )[:max_history]

        return {
            "pareto_front": [
                {
                    "config_id": e.config_id,
                    "architecture": e.architecture,
                    "metrics": {
                        "accuracy": round(e.accuracy, 4),
                        "memory_lift": round(e.memory_lift, 4),
                        "hit_rate": round(e.hit_rate, 4),
                        "token_eff": round(e.token_eff, 4),
                        "fitness": round(e.fitness, 4),
                        "n_evals": e.n_evals,
                    },
                    "attribution_summary": e.attribution_summary,
                }
                for e in front_entries
            ],
            "history_table": [
                {
                    "config_id": e.config_id,
                    "round": e.round_id,
                    "extract": e.architecture.get("extract_types", []),
                    "retrieval": e.architecture.get("retrieval", "?"),
                    "management": e.architecture.get("management", "?"),
                    "accuracy": round(e.accuracy, 3),
                    "fitness": round(e.fitness, 4),
                    "on_front": e in self._front,
                }
                for e in recent_history
            ],
            "front_size": len(self._front),
            "total_evaluated": len(self._all_evaluated),
        }
