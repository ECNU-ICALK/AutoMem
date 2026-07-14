"""Adaptive level-weighted batch sampling for optimization rounds.

Instead of running all optimization tasks every round, sample a batch
with quotas proportional to each level's improvement headroom
(weight = 1 - historical_accuracy).  Every level gets at least one
task for regression detection.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

DEFAULT_ACCURACY = 0.5  # used when no history exists (round 0)


def compute_level_quotas(
    level_to_indices: Dict[str, List[int]],
    level_accuracy: Dict[str, float],
    batch_size: int,
    min_per_level: int = 1,
) -> Dict[str, int]:
    """Compute per-level sampling quotas based on improvement headroom.

    Parameters
    ----------
    level_to_indices : dict
        Mapping from level string (e.g. "1", "2", "3") to list of task
        indices available in the optimization split for that level.
    level_accuracy : dict
        Historical EMA accuracy per level.  Missing levels default to
        ``DEFAULT_ACCURACY``.
    batch_size : int
        Total number of tasks to sample this round.
    min_per_level : int
        Minimum tasks per level (regression guard).

    Returns
    -------
    dict
        ``{level: count}`` quota allocation.
    """
    levels = sorted(level_to_indices.keys(), key=str)
    n_levels = len(levels)

    if n_levels == 0 or batch_size <= 0:
        return {}

    # Cap min_per_level so total minimums don't exceed batch_size
    effective_min = min(min_per_level, batch_size // max(n_levels, 1))

    # Also cap by available indices per level
    mins: Dict[str, int] = {}
    for lv in levels:
        mins[lv] = min(effective_min, len(level_to_indices[lv]))

    remaining = batch_size - sum(mins.values())

    # Compute weights = 1 - accuracy (improvement headroom)
    weights: Dict[str, float] = {}
    for lv in levels:
        acc = level_accuracy.get(lv, DEFAULT_ACCURACY)
        weights[lv] = max(1.0 - acc, 0.01)  # floor to avoid zero weight

    total_weight = sum(weights.values())

    # Distribute remaining slots proportionally
    quotas: Dict[str, int] = dict(mins)
    if remaining > 0 and total_weight > 0:
        fractional: Dict[str, float] = {}
        for lv in levels:
            share = remaining * weights[lv] / total_weight
            quotas[lv] += int(share)
            fractional[lv] = share - int(share)

        # Distribute leftover from rounding (largest remainder method)
        leftover = remaining - sum(int(remaining * weights[lv] / total_weight) for lv in levels)
        for lv in sorted(fractional, key=lambda k: fractional[k], reverse=True):
            if leftover <= 0:
                break
            quotas[lv] += 1
            leftover -= 1

    # Final cap: don't exceed available indices per level
    for lv in levels:
        quotas[lv] = min(quotas[lv], len(level_to_indices[lv]))

    return quotas


def sample_batch(
    level_to_indices: Dict[str, List[int]],
    quotas: Dict[str, int],
    used_indices: Set[int],
    rng: random.Random,
) -> List[int]:
    """Sample task indices according to quotas, preferring unused indices.

    If all indices for a level have been used, allow reuse.

    Returns
    -------
    list
        Flat list of selected task indices.
    """
    selected: List[int] = []

    for level in sorted(quotas.keys(), key=str):
        count = quotas[level]
        pool = level_to_indices.get(level, [])
        if not pool or count <= 0:
            continue

        # Prefer unused indices
        unused = [idx for idx in pool if idx not in used_indices]
        if len(unused) >= count:
            chosen = rng.sample(unused, count)
        else:
            # Take all unused, then sample from already-used
            chosen = list(unused)
            reuse_pool = [idx for idx in pool if idx in used_indices]
            need = count - len(chosen)
            if need > 0 and reuse_pool:
                chosen.extend(rng.sample(reuse_pool, min(need, len(reuse_pool))))

        selected.extend(chosen)

    return selected


def get_level_accuracy_history(
    run_dir: Path,
    current_round: int,
    ema_alpha: float = 0.3,
) -> Dict[str, float]:
    """Compute EMA of per-level accuracy from previous rounds.

    Reads ``evaluation_report.json`` from each past round directory.
    For round 0 (no history), returns ``DEFAULT_ACCURACY`` for all levels
    found in the data split.

    Parameters
    ----------
    run_dir : Path
        Root directory of the optimization run.
    current_round : int
        The round about to start (reads rounds 0..current_round-1).
    ema_alpha : float
        Smoothing factor for exponential moving average.  Higher values
        weight recent rounds more.

    Returns
    -------
    dict
        ``{level_str: ema_accuracy}``
    """
    if current_round <= 0:
        return {}

    ema: Dict[str, float] = {}

    for r in range(current_round):
        report_path = run_dir / f"round_{r}" / "evaluation_report.json"
        if not report_path.exists():
            continue

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        by_level = report.get("score_summary", {}).get("by_level", {})
        for level, stats in by_level.items():
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            if total == 0:
                continue
            acc = correct / total

            if level not in ema:
                ema[level] = acc  # first observation
            else:
                ema[level] = ema_alpha * acc + (1 - ema_alpha) * ema[level]

    return ema


def build_level_index_map(
    tasks: list,
    indices: List[int],
) -> Dict[str, List[int]]:
    """Group optimization indices by their task level.

    Parameters
    ----------
    tasks : list
        Full task list (from metadata.jsonl).
    indices : list
        Indices into *tasks* belonging to the optimization split.

    Returns
    -------
    dict
        ``{level_str: [idx, ...]}``
    """
    level_map: Dict[str, List[int]] = defaultdict(list)
    for idx in indices:
        if 0 <= idx < len(tasks):
            level = str(tasks[idx].get("Level", "1"))
            level_map[level].append(idx)
    return dict(level_map)
