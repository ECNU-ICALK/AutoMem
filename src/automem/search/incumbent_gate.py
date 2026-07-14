"""IncumbentGate — statistical-significance check before replacing the incumbent.

Created 2026-05-13 as part of the H-plan modular decomposition.

Before this module, `_make_champion_candidate` simply pulled `pareto.best()`
to designate the next round's champion — a pure `max` over fitness scores.
That meant a challenger 1pp above the incumbent (well within evaluation
noise) could displace it as the search reference point, sending the next
round in the wrong direction.

This module adds a one-sided proportion test (or bootstrap CI, when scipy
isn't available) before promoting a challenger. The Pareto front still
holds whatever non-dominated points exist, but the *champion-as-reference*
identity only updates when the difference clears noise.

API:
    gate = IncumbentGate(alpha=0.10, min_lift=0.02)
    is_sig, p_value, reason = gate.is_challenger_significant(
        incumbent_acc=0.46, challenger_acc=0.52, n_tasks=50,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IncumbentDecision:
    """Outcome of an incumbent-vs-challenger check."""

    promote: bool                # True → challenger replaces incumbent
    p_value: Optional[float]     # one-sided p; None if test could not run
    delta_acc: float             # challenger - incumbent
    reason: str                  # human-readable rationale


class IncumbentGate:
    """Decide whether to replace the incumbent with a challenger.

    Two gates (both must pass):
      1. Effect size: challenger_acc - incumbent_acc >= min_lift (default 0.02).
      2. Statistical significance: one-sided proportion test p < alpha
         (default 0.10). When `n_tasks` is small or scipy is missing, falls
         back to a bootstrap-based CI; if neither is feasible, the gate
         requires a larger min_lift to compensate.

    Defaults are tuned for GAIA-style 50-task search batches with acc
    typically in 0.4-0.6 range — a 2pp absolute lift translates to a
    z-score around 0.6-0.7 under the standard Wald approximation.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        min_lift: float = 0.02,
        bootstrap_iters: int = 2000,
        bootstrap_seed: int = 7,
    ):
        self.alpha = alpha
        self.min_lift = min_lift
        self.bootstrap_iters = bootstrap_iters
        self.bootstrap_seed = bootstrap_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_challenger_significant(
        self,
        incumbent_acc: float,
        challenger_acc: float,
        n_tasks: int,
    ) -> IncumbentDecision:
        """Check both gates and return a promote/reject decision.

        Args:
            incumbent_acc: incumbent's accuracy on the same n_tasks.
            challenger_acc: challenger's accuracy on the same n_tasks.
            n_tasks: number of tasks both architectures were evaluated on.
                Must be the same for both — otherwise the comparison is
                cross-batch and this method is not appropriate.

        Returns:
            IncumbentDecision
        """
        delta = float(challenger_acc) - float(incumbent_acc)

        # Gate 1: effect size
        if delta < self.min_lift:
            return IncumbentDecision(
                promote=False, p_value=None, delta_acc=delta,
                reason=(
                    f"Effect size {delta:+.3f} below min_lift={self.min_lift:.3f}; "
                    f"challenger does not clearly improve over incumbent."
                ),
            )

        # Gate 2: statistical significance (proportion test)
        p = self._one_sided_proportion_test(
            incumbent_acc, challenger_acc, n_tasks,
        )

        if p is None:
            # Fall back to bootstrap if scipy unavailable / numerical edge
            p = self._bootstrap_p_value(incumbent_acc, challenger_acc, n_tasks)

        if p is None:
            # Both tests failed — compensate with a stricter lift requirement.
            stricter = self.min_lift * 2
            if delta >= stricter:
                return IncumbentDecision(
                    promote=True, p_value=None, delta_acc=delta,
                    reason=(
                        f"Statistical test unavailable but effect size "
                        f"{delta:+.3f} ≥ stricter threshold {stricter:.3f}; "
                        f"promoting under the precautionary stricter-lift rule."
                    ),
                )
            return IncumbentDecision(
                promote=False, p_value=None, delta_acc=delta,
                reason=(
                    f"Statistical test unavailable AND effect size {delta:+.3f} "
                    f"< stricter threshold {stricter:.3f}; refusing to promote."
                ),
            )

        if p < self.alpha:
            return IncumbentDecision(
                promote=True, p_value=p, delta_acc=delta,
                reason=(
                    f"Challenger acc={challenger_acc:.3f} vs incumbent {incumbent_acc:.3f} "
                    f"(Δ={delta:+.3f}), one-sided p={p:.3f} < α={self.alpha}; promoting."
                ),
            )
        return IncumbentDecision(
            promote=False, p_value=p, delta_acc=delta,
            reason=(
                f"Effect size {delta:+.3f} clears min_lift but one-sided p={p:.3f} ≥ α={self.alpha}; "
                f"insufficient evidence to promote — incumbent retains champion role."
            ),
        )

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------
    def _one_sided_proportion_test(
        self, p1: float, p2: float, n: int,
    ) -> Optional[float]:
        """Wald one-sided z-test for H1: p2 > p1.

        Returns p-value, or None if degenerate (n too small / both rates 0 or 1).
        """
        if n <= 0:
            return None
        # Pool variance (Wald)
        try:
            pooled = (p1 + p2) / 2.0
            var = pooled * (1 - pooled) * (2.0 / n)
            if var <= 0:
                return None
            z = (p2 - p1) / math.sqrt(var)
            # One-sided p-value via normal CDF (use math.erfc)
            # P(Z > z) = 0.5 * erfc(z / sqrt(2))
            p_value = 0.5 * math.erfc(z / math.sqrt(2))
            return float(p_value)
        except (ValueError, ZeroDivisionError):
            return None

    def _bootstrap_p_value(
        self, p1: float, p2: float, n: int,
    ) -> Optional[float]:
        """Bootstrap p-value via paired-rate resampling.

        Approximates each architecture's per-task outcomes as Bernoulli(p),
        resamples n outcomes for both, and counts how often the resampled
        challenger fails to beat the resampled incumbent. Crude but useful
        when scipy isn't available.
        """
        try:
            import random
            rng = random.Random(self.bootstrap_seed)
            wins = 0
            for _ in range(self.bootstrap_iters):
                a = sum(1 for _ in range(n) if rng.random() < p1)
                b = sum(1 for _ in range(n) if rng.random() < p2)
                if b > a:
                    wins += 1
            # one-sided p = P(challenger fails to beat incumbent)
            p = 1.0 - (wins / self.bootstrap_iters)
            return p
        except Exception:
            return None


__all__ = ["IncumbentGate", "IncumbentDecision"]
