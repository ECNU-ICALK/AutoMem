"""Benchmark-agnostic task-complexity inference.

The evolution system was originally written for GAIA, which ships a hand-labelled
difficulty Level (1/2/3). Several components keyed off that Level: extraction
prompts (how many tips / how many workflow steps to emit), the Observation Graph
(per-difficulty performance buckets), and batch sampling. On benchmarks without a
Level field (xBench-DeepSearch, WebWalkerQA, ...) those components either degraded
(OG collapsed to a single bucket) or fed dead L1/L2/L3 guidance to the extractor.

This module replaces the hard GAIA "Level" dependency with a single complexity
label derived from whatever signal is available, in priority order:

  1. an explicit difficulty field (GAIA `Level` / `level`), mapped 1->simple,
     2->medium, 3->complex  -> GAIA behaviour is byte-equivalent;
  2. the agent trajectory length (benchmark-agnostic, available at extraction
     time): <=3 steps -> simple, 4-8 -> medium, >8 -> complex;
  3. "unknown" when neither is present.

Labels are simple/medium/complex/unknown — a 3-level scale that matches GAIA's so
the prompt budgets (e.g. "simple: 0-1 tip; complex: up to 4") stay identical to
the old L1/L2/L3 numbers.
"""
from typing import Any, Dict, Optional, Sequence

# explicit-difficulty field aliases that mean the same 3 levels
_LEVEL_MAP = {
    "1": "simple", "2": "medium", "3": "complex",
    "l1": "simple", "l2": "medium", "l3": "complex",
    "level1": "simple", "level2": "medium", "level3": "complex",
    "simple": "simple", "medium": "medium", "complex": "complex",
    "easy": "simple", "hard": "complex",
}

# trajectory-step-count thresholds (calibrated to deep-search agents, where the
# xBench mean was ~9.6 steps): short traces are simple, long ones complex.
_STEP_SIMPLE_MAX = 3
_STEP_MEDIUM_MAX = 8


def _from_explicit_level(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    return _LEVEL_MAP.get(key)


def _from_step_count(n_steps: Optional[int]) -> Optional[str]:
    if not isinstance(n_steps, int) or n_steps <= 0:
        return None
    if n_steps <= _STEP_SIMPLE_MAX:
        return "simple"
    if n_steps <= _STEP_MEDIUM_MAX:
        return "medium"
    return "complex"


def task_complexity(
    task: Optional[Dict[str, Any]] = None,
    trajectory: Optional[Sequence[Any]] = None,
    explicit_level: Any = None,
) -> str:
    """Return one of 'simple' | 'medium' | 'complex' | 'unknown'.

    Args:
        task: a task/metadata dict; checked for a `level` / `Level` field.
        trajectory: the agent trajectory (any sized sequence); its length is the
            fallback complexity signal when no explicit level exists.
        explicit_level: an already-extracted level value (takes precedence over
            `task`), e.g. when the caller already pulled it out of metadata.
    """
    # 1. explicit difficulty field (GAIA Level) — keeps GAIA byte-equivalent
    lvl = explicit_level
    if lvl is None and task:
        lvl = task.get("level", task.get("Level"))
        # WebWalkerQA ships its official difficulty under "difficulty"
        # (easy/medium/hard); honor it as a fallback alias. level/Level
        # keeps precedence so GAIA behaviour stays byte-equivalent.
        if lvl is None:
            lvl = task.get("difficulty", task.get("Difficulty"))
    mapped = _from_explicit_level(lvl)
    if mapped is not None:
        return mapped

    # 2. trajectory step count (benchmark-agnostic)
    n = None
    if trajectory is not None:
        try:
            n = len(trajectory)
        except TypeError:
            n = None
    mapped = _from_step_count(n)
    if mapped is not None:
        return mapped

    # 3. nothing to go on
    return "unknown"
