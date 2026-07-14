"""The fixed AutoMem architecture-search evaluation protocol.

All search runs use the same protocol so results cannot silently diverge due
to command-line mechanism switches.

Mechanisms:
  M1  no mid-search validation checkpoints
  M2  fold rotation: the search pool is split into stratified folds and
      round t evaluates fold (t-1) mod n — within-round comparisons stay
      paired, while the champion's pooled estimate spans folds
  M3  final runoff: top-K distinct architectures re-evaluated on the
      full fold union before final validation
  M4  no-self-retrieval guard (fixed inside ModularMemoryProvider)
  C7  canonical merge gate: only the round winner's memories merge back
  A1  pooled champion scoring (implemented in ParetoFront pooled mode)
  A2  paired sign-test acceptance for champion succession
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Protocol resolution
# ======================================================================

@dataclass(frozen=True, init=False)
class ProtocolConfig:
    """Immutable effective settings for every public AutoMem search run."""

    name: str = "automem-v1"
    fold_rotation: int = 2
    canonical_merge: str = "winner"
    champion_scoring: str = "pooled"
    acceptance: str = "paired"
    accept_alpha: float = 0.10
    final_runoff: int = 2
    val_every: int = 0

    @classmethod
    def resolve(cls, args: Any) -> "ProtocolConfig":
        """Return the single fixed protocol; ``args`` is intentionally ignored."""

        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fold_rotation": self.fold_rotation,
            "canonical_merge": self.canonical_merge,
            "champion_scoring": self.champion_scoring,
            "acceptance": self.acceptance,
            "accept_alpha": self.accept_alpha,
            "final_runoff": self.final_runoff,
            "val_every": self.val_every,
        }

    def digest_fields(self) -> Dict[str, Any]:
        """Return behavior-affecting fields folded into the protocol digest."""

        d = self.to_dict()
        d.pop("val_every", None)  # checkpoint cadence does not change metrics
        return d


# ======================================================================
# M2 — stratified fold construction
# ======================================================================

def make_folds(
    pool: List[int],
    n_folds: int,
    seed: int,
    tasks: Optional[List[Dict[str, Any]]] = None,
    fold_size: Optional[int] = None,
) -> List[List[int]]:
    """Partition (a sample of) the search pool into n stratified folds.

    When tasks carry a "Level" field (GAIA), indices are grouped by level
    and dealt round-robin so each fold preserves the level mix. Deal order
    is seeded-shuffled => deterministic for a given (pool, seed).

    fold_size caps each fold (total sample = n_folds * fold_size when the
    pool is large enough); None uses the whole pool.
    """
    if n_folds <= 1:
        return [sorted(pool)]
    rng = random.Random(seed)

    # Group by level when available, else one bucket.
    buckets: Dict[Any, List[int]] = {}
    for idx in pool:
        lvl = None
        if tasks and idx < len(tasks):
            lvl = tasks[idx].get("Level")
        buckets.setdefault(lvl, []).append(idx)

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    cap_total = fold_size * n_folds if fold_size else None

    # Proportional quota per level (largest-remainder) when capping, so no
    # single level absorbs the truncation.
    if cap_total is not None:
        total = sum(len(b) for b in buckets.values())
        if total > cap_total:
            keys = sorted(buckets.keys(), key=lambda x: str(x))
            raw = {lvl: cap_total * len(buckets[lvl]) / total for lvl in keys}
            quota = {lvl: int(raw[lvl]) for lvl in keys}
            remaining = cap_total - sum(quota.values())
            for lvl in sorted(keys, key=lambda l: -(raw[l] - quota[l])):
                if remaining <= 0:
                    break
                if quota[lvl] < len(buckets[lvl]):
                    quota[lvl] += 1
                    remaining -= 1
            for lvl in keys:
                idxs = list(buckets[lvl])
                rng.shuffle(idxs)
                buckets[lvl] = idxs[: quota[lvl]]

    # Deal each bucket round-robin starting at a rotating offset so small
    # buckets do not all land in fold 0.
    start = 0
    for lvl in sorted(buckets.keys(), key=lambda x: str(x)):
        idxs = list(buckets[lvl])
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[(start + j) % n_folds].append(idx)
        start += len(idxs) % n_folds
    return [sorted(f) for f in folds]


def fold_for_round(folds: List[List[int]], round_id: int) -> List[int]:
    """Round t (1-based) evaluates fold (t-1) mod n."""
    if not folds:
        return []
    return list(folds[(round_id - 1) % len(folds)])


# ======================================================================
# A2 — paired sign test (exact binomial)
# ======================================================================

def paired_sign_test(
    champion_scores: Dict[str, float],
    challenger_scores: Dict[str, float],
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """Exact one-sided sign test on per-task flips (same tasks, same round).

    champion_scores / challenger_scores: item_index -> task_score (>=1.0
    counts as pass). Only tasks present in BOTH maps are compared; ties
    (both pass / both fail) carry no information and are discarded, per
    the standard sign-test treatment.

    promote=True iff the challenger wins strictly more flips than the
    champion AND the one-sided binomial p-value (P[X >= n10 | n01+n10,
    p=0.5]) is below alpha.
    """
    common = sorted(set(champion_scores) & set(challenger_scores))
    n10 = sum(1 for k in common
              if challenger_scores[k] >= 1.0 > champion_scores[k])   # champ fail -> chall pass
    n01 = sum(1 for k in common
              if champion_scores[k] >= 1.0 > challenger_scores[k])   # champ pass -> chall fail
    m = n10 + n01
    if m == 0:
        p_value = 1.0
    else:
        # One-sided exact binomial: P[X >= n10] with X ~ Bin(m, 0.5)
        p_value = sum(math.comb(m, x) for x in range(n10, m + 1)) / (2 ** m)
    promote = (n10 > n01) and (p_value < alpha)
    return {
        "n_common": len(common),
        "flips_to_pass": n10,
        "flips_to_fail": n01,
        "net_flips": n10 - n01,
        "p_value": round(p_value, 5),
        "alpha": alpha,
        "promote": promote,
    }


def per_task_scores_from_results(task_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """item_index -> task_score map from load_task_results() output."""
    out: Dict[str, float] = {}
    for r in task_results:
        idx = str(r.get("item_index", ""))
        if idx:
            out[idx] = float(r.get("task_score", 0.0))
    return out


# ======================================================================
# A2 — champion lineage state
# ======================================================================

def load_champion_state(run_dir) -> Optional[Dict[str, Any]]:
    from pathlib import Path
    p = Path(run_dir) / "champion_state.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # corrupted state must not kill the run
        logger.warning("champion_state.json unreadable (%s); ignoring.", e)
        return None


def save_champion_state(run_dir, state: Dict[str, Any]) -> None:
    import os
    import tempfile
    from pathlib import Path

    p = Path(run_dir) / "champion_state.json"
    fd, temp_path = tempfile.mkstemp(dir=p.parent, suffix=".champion.json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, p)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def update_champion_after_round(
    run_dir,
    round_id: int,
    round_results: List[Dict[str, Any]],
    per_task_by_config: Dict[str, Dict[str, float]],
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """Paired-acceptance champion succession (A2).

    round_results: the round's result entries (config_id, architecture,
    metrics, diversity_role). per_task_by_config: config_id -> per-task
    score map for this round's fold.

    Rules:
      * no incumbent yet -> round best becomes champion (no test);
      * incumbent re-eval present (diversity_role == "champion") -> best
        non-champion challenger must win the paired sign test against the
        SAME-round champion measurement to take over;
      * incumbent missing this round (e.g. elitism disabled) -> fall back
        to initializing from round best, flagged in the decision.
    """
    ok = [r for r in round_results if r.get("metrics")]
    if not ok:
        return {"updated": False, "reason": "no_successful_candidates"}

    state = load_champion_state(run_dir)
    if state is not None and int(state.get("last_applied_round", -1)) >= round_id:
        return {
            "round_id": round_id,
            "updated": False,
            "new_champion": state.get("config_id"),
            "reason": "round_already_applied",
        }
    champ_rows = [r for r in ok if r.get("diversity_role") == "champion"]
    challengers = [r for r in ok if r.get("diversity_role") != "champion"]

    def _fit(r: Dict[str, Any]) -> float:
        return float(r.get("metrics", {}).get("fitness", 0.0))

    decision: Dict[str, Any] = {"round_id": round_id, "updated": False}

    if state is None or not champ_rows:
        best = max(ok, key=_fit)
        new_state = {
            "config_id": best["config_id"],
            "architecture": best["architecture"],
            "since_round": round_id,
            "mode": "init" if state is None else "reinit_no_champion_row",
            "last_applied_round": round_id,
        }
        save_champion_state(run_dir, new_state)
        decision.update({
            "updated": True,
            "new_champion": best["config_id"],
            "test": None,
            "reason": new_state["mode"],
        })
        return decision

    champ_row = champ_rows[0]
    if not challengers:
        decision["reason"] = "no_challengers"
        state["last_applied_round"] = round_id
        save_champion_state(run_dir, state)
        return decision
    challenger = max(challengers, key=_fit)

    champ_scores = per_task_by_config.get(champ_row["config_id"], {})
    chall_scores = per_task_by_config.get(challenger["config_id"], {})
    test = paired_sign_test(champ_scores, chall_scores, alpha=alpha)
    decision["test"] = test
    decision["challenger"] = challenger["config_id"]
    decision["incumbent"] = state.get("config_id")

    if test["promote"]:
        new_state = {
            "config_id": challenger["config_id"],
            "architecture": challenger["architecture"],
            "since_round": round_id,
            "mode": "paired_promotion",
            "promoted_over": state.get("config_id"),
            "test": test,
            "last_applied_round": round_id,
        }
        save_champion_state(run_dir, new_state)
        decision.update({"updated": True, "new_champion": challenger["config_id"],
                         "reason": "paired_test_passed"})
    else:
        decision["reason"] = "paired_test_failed"
        state["last_applied_round"] = round_id
        save_champion_state(run_dir, state)
    return decision


# ======================================================================
# M3 — final runoff contender selection
# ======================================================================

def select_runoff_contenders(
    pareto: Any,
    champion_state: Optional[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Top-K distinct architectures by fitness, champion always included."""
    seen: set = set()
    contenders: List[Dict[str, Any]] = []

    def _key(arch: Dict[str, Any]) -> str:
        return json.dumps(arch, sort_keys=True, default=str)

    if champion_state and champion_state.get("architecture"):
        contenders.append({
            "config_id": champion_state.get("config_id", "champion"),
            "architecture": champion_state["architecture"],
            "source": "champion_state",
        })
        seen.add(_key(champion_state["architecture"]))

    rank_all = getattr(pareto, "top_k_all", pareto.top_k)
    for e in rank_all(max(k * 3, k)):
        ak = _key(e.architecture)
        if ak in seen:
            continue
        contenders.append({
            "config_id": e.config_id,
            "architecture": dict(e.architecture),
            "source": "evaluated_history",
        })
        seen.add(ak)
        if len(contenders) >= k:
            break
    return contenders[:k]
