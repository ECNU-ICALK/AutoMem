"""CandidateScheduler — code-layer role allocator for evolutionary search.

Created 2026-05-13 as part of the H-plan modular decomposition.

Before this module, the architect prompt (`architecture_search.txt` §3) carried
~100 lines of Jinja that asked the LLM to figure out, for each round, what
role each candidate should play (`exploit`, `explore_retrieval`, etc.). The
LLM had to read `round_id`, `max_rounds`, the Pareto front, the attribution
diagnosis, the ledger — and then *infer* the right role distribution.

This module moves that decision to Python:

  scheduler.assign_roles(round_id, max_rounds, num_candidates, has_champion,
                         ledger_has_open_questions)
    → List[ScheduledCandidate]

Intent: the proposer prompt should be reduced to receive an explicit
`roles_this_round` list, and the LLM only needs to generate one architecture
per stated role. The Jinja phase logic could then shrink to a small lookup
table. (As of 2026-05-13 this module is built but the prompt rewrite to
consume `roles_this_round` is staged for a follow-up — the prompt's §3
still carries the explicit phase Jinja.)

Phase boundaries match the prompt's accuracy-first profile:
  - round 1                  → BOOTSTRAP   (no Pareto front yet)
  - progress ≤ 0.5           → BROAD_EXPLORE
  - progress ≤ 0.75          → MIXED_REFINE
  - progress > 0.75          → PURE_REFINE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Role(str, Enum):
    """Roles a candidate can be assigned in a given round."""

    CHAMPION = "champion"                  # re-evaluate current Pareto best
    BASELINE = "baseline"                  # R1 only: a simple competent config
    EXPLOIT = "exploit"                    # small mutation of champion
    EXPLORE_RETRIEVAL = "explore_retrieval"
    EXPLORE_EXTRACT = "explore_extract"
    EXPLORE_STORAGE = "explore_storage"
    VALIDATE_OPEN_Q = "validate_open_question"  # MIXED phase: test ledger open Q
    EXPLOIT_A = "exploit_a"                # PURE phase: 1-layer mutation, dominant attr
    EXPLOIT_B = "exploit_b"                # PURE phase: 1-layer mutation, 2nd attr
    EXPLOIT_C = "exploit_c"                # PURE phase: validate low-evidence principle


class Phase(str, Enum):
    BOOTSTRAP = "bootstrap"
    BROAD_EXPLORE = "broad_explore"
    MIXED_REFINE = "mixed_refine"
    PURE_REFINE = "pure_refine"


@dataclass
class ScheduledCandidate:
    candidate_id: int
    role: Role
    is_champion: bool = False  # if True, the runner injects without calling LLM


def phase_for(round_id: int, max_rounds: int) -> Phase:
    """Determine which phase this round belongs to."""
    if round_id == 1:
        return Phase.BOOTSTRAP
    if max_rounds <= 0:
        max_rounds = 1
    progress = round_id / max_rounds
    if progress <= 0.5:
        return Phase.BROAD_EXPLORE
    if progress <= 0.75:
        return Phase.MIXED_REFINE
    return Phase.PURE_REFINE


class CandidateScheduler:
    """Decide N candidate roles per round.

    Behavior is deterministic — same (round_id, max_rounds, num_candidates,
    has_champion) input produces the same role list.
    """

    def __init__(self):
        pass

    def assign_roles(
        self,
        round_id: int,
        max_rounds: int,
        num_candidates: int,
        has_champion: bool,
        ledger_has_open_questions: bool = False,
    ) -> List[ScheduledCandidate]:
        """Return the candidate role list for this round.

        Args:
            round_id: 1-indexed round number.
            max_rounds: total rounds for the run (drives phase boundaries).
            num_candidates: total candidates the orchestrator wants (incl. champion).
            has_champion: True if a Pareto best exists (round_id >= 2 + non-empty front).
            ledger_has_open_questions: enables the validate_open_question role.

        Returns:
            List[ScheduledCandidate] of length `num_candidates`.
        """
        if num_candidates <= 0:
            return []

        phase = phase_for(round_id, max_rounds)
        scheduled: List[ScheduledCandidate] = []

        # Round 1 bootstrap: no champion, LLM generates ALL N candidates.
        if phase == Phase.BOOTSTRAP:
            roles = [Role.BASELINE, Role.EXPLORE_RETRIEVAL, Role.EXPLORE_EXTRACT]
            # If num_candidates > 3, fill the rest with storage exploration.
            while len(roles) < num_candidates:
                roles.append(Role.EXPLORE_STORAGE)
            for i, role in enumerate(roles[:num_candidates]):
                scheduled.append(ScheduledCandidate(candidate_id=i, role=role))
            return scheduled

        # R2+: champion is candidate 0 (when present); LLM produces the rest.
        if has_champion:
            scheduled.append(
                ScheduledCandidate(candidate_id=0, role=Role.CHAMPION, is_champion=True)
            )
            remaining = num_candidates - 1
        else:
            # No Pareto best yet (e.g. first round failed); treat like bootstrap.
            return self.assign_roles(
                round_id=1,
                max_rounds=max_rounds,
                num_candidates=num_candidates,
                has_champion=False,
                ledger_has_open_questions=ledger_has_open_questions,
            )

        # Compose remaining roles by phase
        if phase == Phase.BROAD_EXPLORE:
            llm_roles = [Role.EXPLOIT, Role.EXPLORE_RETRIEVAL, Role.EXPLORE_STORAGE]
        elif phase == Phase.MIXED_REFINE:
            third = Role.VALIDATE_OPEN_Q if ledger_has_open_questions else Role.EXPLORE_STORAGE
            llm_roles = [Role.EXPLOIT_A, Role.EXPLOIT_B, third]
        else:  # PURE_REFINE
            llm_roles = [Role.EXPLOIT_A, Role.EXPLOIT_B, Role.EXPLOIT_C]

        # Truncate / pad to remaining slots
        for i in range(remaining):
            role = llm_roles[i] if i < len(llm_roles) else Role.EXPLOIT
            scheduled.append(ScheduledCandidate(candidate_id=i + 1, role=role))

        return scheduled

    def describe_phase(self, round_id: int, max_rounds: int) -> str:
        """Human-readable phase description (used in prompt + logging)."""
        phase = phase_for(round_id, max_rounds)
        return {
            Phase.BOOTSTRAP: "Bootstrap (no Pareto front yet — generate all candidates broadly)",
            Phase.BROAD_EXPLORE: "Broad exploration (ledger young — sample widely; ≥3-component mutations)",
            Phase.MIXED_REFINE: "Mixed refinement (double down on exploit while testing one open question)",
            Phase.PURE_REFINE: "Pure refinement (final rounds — 1-layer mutations only)",
        }[phase]


__all__ = ["CandidateScheduler", "Role", "Phase", "ScheduledCandidate", "phase_for"]
