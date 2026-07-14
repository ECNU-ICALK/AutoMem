"""Data split manager for automem experiments.

Ensures four non-overlapping splits so that profiling, optimisation,
validation, and final testing never share the same data points.
"""

import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DataSplitConfig:
    """Configuration for 4-way non-overlapping data splits."""

    profile_indices: List[int] = field(default_factory=list)       # for task_profiling
    optimization_indices: List[int] = field(default_factory=list)   # for prompt optimization rounds
    validation_indices: List[int] = field(default_factory=list)     # for incumbent vs challenger comparison
    final_test_indices: List[int] = field(default_factory=list)     # held out, never used during optimization

    def validate(self, total_tasks: int | None = None) -> tuple:
        """Validate index types, uniqueness, bounds and split disjointness."""
        errors: List[str] = []
        raw_splits = {
            "profile": self.profile_indices,
            "optimization": self.optimization_indices,
            "validation": self.validation_indices,
            "final_test": self.final_test_indices,
        }
        for name, indices in raw_splits.items():
            if not isinstance(indices, list):
                errors.append(f"{name} indices must be a list")
                continue
            invalid_types = [value for value in indices if type(value) is not int]
            if invalid_types:
                errors.append(
                    f"{name} contains non-integer indices: {invalid_types[:5]}"
                )
                continue
            duplicates = sorted(
                value for value in set(indices) if indices.count(value) > 1
            )
            if duplicates:
                errors.append(f"{name} contains duplicate indices: {duplicates[:5]}")
            negative = sorted(value for value in indices if value < 0)
            if negative:
                errors.append(f"{name} contains negative indices: {negative[:5]}")
            if total_tasks is not None:
                out_of_range = sorted(
                    value for value in indices if value >= total_tasks
                )
                if out_of_range:
                    errors.append(
                        f"{name} indices exceed task count {total_tasks}: "
                        f"{out_of_range[:5]}"
                    )

        sets = {
            name: {value for value in indices if type(value) is int}
            if isinstance(indices, list)
            else set()
            for name, indices in raw_splits.items()
        }
        names = list(sets.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = sets[names[i]] & sets[names[j]]
                if overlap:
                    preview = sorted(overlap)[:5]
                    errors.append(
                        f"Overlap between {names[i]} and {names[j]}: {preview}..."
                    )
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_indices": sorted(self.profile_indices),
            "optimization_indices": sorted(self.optimization_indices),
            "validation_indices": sorted(self.validation_indices),
            "final_test_indices": sorted(self.final_test_indices),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataSplitConfig":
        """Construct a :class:`DataSplitConfig` from a plain dict."""
        return cls(
            profile_indices=d.get("profile_indices", []),
            optimization_indices=d.get("optimization_indices", []),
            validation_indices=d.get("validation_indices", []),
            final_test_indices=d.get("final_test_indices", []),
        )

    def save(self, path: str) -> None:
        """Persist the split configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=p.parent, suffix=".split.json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, p)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise
        logger.info("Saved data split config to %s", path)

    @classmethod
    def load(cls, path: str) -> "DataSplitConfig":
        """Load a split configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        logger.info("Loaded data split config from %s", path)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Split creation helpers
# ---------------------------------------------------------------------------

def create_default_split(
    total_tasks: int,
    profile_n: int = 5,
    optimization_n: int = 15,
    validation_n: int = 10,
    final_test_n: int = 10,
) -> DataSplitConfig:
    """Create a default 4-way split from sequential task indices.

    Allocation (all indices are 0-based):
      - profile:      first *profile_n* tasks
      - optimization: next *optimization_n* tasks
      - validation:   next *validation_n* tasks
      - final_test:   next *final_test_n* tasks

    Tasks beyond the four requested split sizes remain unused.
    """
    sizes = (profile_n, optimization_n, validation_n, final_test_n)
    if any(size < 0 for size in sizes):
        raise ValueError("Split sizes must be non-negative")
    required = sum(sizes)
    if total_tasks < required:
        raise ValueError(
            f"Need at least {required} tasks, but only {total_tasks} available."
        )

    cursor = 0
    profile = list(range(cursor, cursor + profile_n))
    cursor += profile_n
    optimization = list(range(cursor, cursor + optimization_n))
    cursor += optimization_n
    validation = list(range(cursor, cursor + validation_n))
    cursor += validation_n
    final_test = list(range(cursor, cursor + final_test_n))

    config = DataSplitConfig(
        profile_indices=profile,
        optimization_indices=optimization,
        validation_indices=validation,
        final_test_indices=final_test,
    )
    is_valid, errors = config.validate()
    assert is_valid, f"Default split produced overlaps: {errors}"
    return config


def create_level_aware_split(
    tasks: List[Dict],
    profile_n: int = 5,
    optimization_n: int = 15,
    validation_n: int = 10,
    final_test_n: int = 10,
) -> DataSplitConfig:
    """Create splits that maintain the level distribution across all 4 splits.

    Each entry in *tasks* must have a ``"Level"`` key.  Tasks are grouped
    by level and distributed round-robin into the four buckets so that every
    split mirrors (as closely as possible) the overall level proportions.
    """
    total = len(tasks)
    target_sizes = [profile_n, optimization_n, validation_n, final_test_n]
    if any(size < 0 for size in target_sizes):
        raise ValueError("Split sizes must be non-negative")
    required = sum(target_sizes)
    if total < required:
        raise ValueError(
            f"Need at least {required} tasks, but only {total} available."
        )

    # Group task indices by level
    level_to_indices: Dict[Any, List[int]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        level_to_indices[task["Level"]].append(idx)

    buckets: List[List[int]] = [[] for _ in range(4)]
    remaining = {level: list(indices) for level, indices in level_to_indices.items()}
    levels = sorted(remaining, key=str)

    # Allocate each split exactly, using largest remainders to preserve the
    # level distribution among the tasks still available.
    for bucket, target_size in zip(buckets, target_sizes):
        remaining_total = sum(len(remaining[level]) for level in levels)
        if target_size == 0:
            continue
        ideals = {
            level: target_size * len(remaining[level]) / remaining_total
            for level in levels
        }
        allocation = {level: int(ideals[level]) for level in levels}
        slots_left = target_size - sum(allocation.values())
        ranked = sorted(
            levels,
            key=lambda level: (-(ideals[level] - allocation[level]), str(level)),
        )
        for level in ranked:
            if slots_left == 0:
                break
            if allocation[level] < len(remaining[level]):
                allocation[level] += 1
                slots_left -= 1

        if slots_left:
            raise RuntimeError("Unable to allocate the requested stratified split")
        for level in levels:
            count = allocation[level]
            bucket.extend(remaining[level][:count])
            del remaining[level][:count]

    config = DataSplitConfig(
        profile_indices=buckets[0],
        optimization_indices=buckets[1],
        validation_indices=buckets[2],
        final_test_indices=buckets[3],
    )

    is_valid, errors = config.validate()
    if not is_valid:
        logger.error("Level-aware split produced overlaps: %s", errors)
    return config
