"""Memory compliance — does agent follow the instructions inside retrieved units?

Smart-8 fix (2026-05-16). Attribution-stage version (no agent_runtime hook).

For each failed task with retrieved memory text, this module:
  1. Asks the diagnosis LLM to extract 6-dim instructions from the unit text:
     - tool_choice         (which tools to use)
     - step_order          (recommended action sequence)
     - answer_format       (single_word, single_number, csv, sentence, ...)
     - scope_constraint    (e.g. ``year=2019``)
     - data_source         (recommended domains / APIs)
     - avoid_rule          (anti-patterns to NOT do)
  2. Compares each dimension against the agent's actual behavior, parsed from
     the agent_trajectory and final_answer:
     - match    — agent followed the unit's recommendation
     - partial  — agent partially followed
     - violate  — agent did the opposite or ignored a hard constraint
     - n/a      — unit didn't give an instruction in this dimension
  3. Emits a per-task ``memory_compliance`` dict with:
       {
         "avg_followed_score": 0.0-1.0,
         "violations_by_dim":  {"D3.answer_format": 1, "D4.scope_constraint": 1},
         "per_unit": [{...}, ...],   # debug-friendly per-unit breakdown
       }

This is NOT a real-time agent-runtime signal — the agent didn't see the units
labeled by id. We're approximating which dimensions the agent likely matched
by parsing trajectory + final_answer with LLM judgement. This is intentional:
the goal is improving the LLM diagnostician's input (post-hoc), not changing
the agent's runtime.

Cost: ~1 LLM call per (task, unit) pair, but instruction-extract is cached
per unit_id / hash so reruns on the same pool are free. Typical eval (50 task,
~3 retrieved units each) → ~150 calls; deduped to <100 with cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Schemas ───────────────────────────────────────────────────────────────

INSTRUCTION_DIMS = (
    "tool_choice",
    "step_order",
    "answer_format",
    "scope_constraint",
    "data_source",
    "avoid_rule",
)

# Schema for LLM instruction-extract call. Empty arrays / null are allowed
# (most units don't give an instruction in every dimension).
INSTRUCTION_SCHEMA: Dict[str, Any] = {
    "tool_choice":      list,    # list of tool names like "web_search"
    "step_order":       list,    # list of action verbs in order
    "answer_format":    list,    # list of format hints like "single_word"
    "scope_constraint": list,    # list of constraint phrases
    "data_source":      list,    # list of recommended domains
    "avoid_rule":       list,    # list of anti-patterns
}


_EXTRACTION_PROMPT_TEMPLATE = """You are extracting actionable instructions from a memory unit produced by a prior LLM-agent run.

The unit contains advice like "use web_search before crawl_page" or "answer must be a single word". Your job: parse it into a 6-dimensional JSON of *what the unit tells the agent to do*. Most units do NOT give instructions in every dimension — leave those empty.

Memory unit text:
```
{{ unit_text }}
```

For each dimension, extract 0+ atomic instructions:

1. **tool_choice**     — recommended tool names (e.g. `["web_search", "arxiv_api"]`).
2. **step_order**      — ordered action verbs the agent should follow (e.g. `["search", "verify_year", "extract"]`).
3. **answer_format**   — format hints. Choose from: `single_word`, `single_number`, `single_phrase`, `comma_separated`, `sentence`, `paragraph`, OR a literal phrase like `"4-digit year"`.
4. **scope_constraint** — constraints the agent must respect (e.g. `["year=2019"]`, `["British_Museum_only"]`).
5. **data_source**     — recommended domains / APIs (e.g. `["arxiv.org", "wikipedia"]`).
6. **avoid_rule**      — anti-patterns explicitly warned against (e.g. `["don't_use_wikipedia_history"]`).

Output strictly valid JSON, no markdown fences, no prose:

{
  "tool_choice":      [],
  "step_order":       [],
  "answer_format":    [],
  "scope_constraint": [],
  "data_source":      [],
  "avoid_rule":       []
}
"""


_COMPARE_PROMPT_TEMPLATE = """You are a memory-compliance auditor.

You have:
- The agent's actual trajectory + final answer for a task.
- A memory unit's extracted 6-dim instructions (what the unit told the agent to do).

Judge whether the agent FOLLOWED each instruction dimension. Return one verdict per dimension:
- `match`   — agent did what the unit suggested.
- `partial` — agent did some of what the unit suggested.
- `violate` — agent did the opposite, ignored a hard constraint, or triggered an avoid_rule.
- `n_a`     — the unit had no instruction in this dimension (the instruction array was empty).

Agent question:
{{ question }}

Agent's golden answer:
{{ golden_answer }}

Agent's actual answer:
{{ agent_answer }}

Agent trajectory tail (last 5 steps):
{{ trajectory_tail }}

Unit's extracted instructions:
```json
{{ instructions_json }}
```

For each non-empty dimension, output one of {match, partial, violate}. For empty dimensions, output `n_a`. Also provide a 1-line evidence per dimension.

Output strictly valid JSON:

{
  "tool_choice":      {"verdict": "match|partial|violate|n_a", "evidence": "..."},
  "step_order":       {"verdict": "...",                       "evidence": "..."},
  "answer_format":    {"verdict": "...",                       "evidence": "..."},
  "scope_constraint": {"verdict": "...",                       "evidence": "..."},
  "data_source":      {"verdict": "...",                       "evidence": "..."},
  "avoid_rule":       {"verdict": "...",                       "evidence": "..."}
}
"""


COMPARE_SCHEMA: Dict[str, Any] = {
    dim: {"verdict", "evidence"} for dim in INSTRUCTION_DIMS
}


# ── Cache ─────────────────────────────────────────────────────────────────

# Per-process cache of extracted instructions by unit text hash.
# Survives across candidate evaluations within one run.
_INSTRUCTION_CACHE: Dict[str, Dict[str, Any]] = {}


def _unit_text_hash(unit_text: str) -> str:
    return hashlib.md5(unit_text.encode("utf-8", errors="ignore")).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────

def extract_instructions(unit_text: str, model) -> Dict[str, Any]:
    """LLM-extract 6-dim instructions from a memory unit text. Cached.

    Returns dict with the 6 INSTRUCTION_DIMS keys, each a list. On LLM failure
    returns an empty instructions dict (all dims empty), so downstream compare
    treats every dim as n_a.
    """
    from ..llm_utils import call_llm_json

    key = _unit_text_hash(unit_text)
    if key in _INSTRUCTION_CACHE:
        return _INSTRUCTION_CACHE[key]

    result = call_llm_json(
        model, _EXTRACTION_PROMPT_TEMPLATE, {"unit_text": unit_text},
        schema=INSTRUCTION_SCHEMA,
        max_retries=2,
        retry_with_feedback=True,
    )
    if result.get("_parse_failed"):
        logger.warning("[memory_compliance] extract_instructions failed: %s",
                       result.get("_last_err"))
        result = {dim: [] for dim in INSTRUCTION_DIMS}

    # Normalize — ensure every dim is a list
    for dim in INSTRUCTION_DIMS:
        if dim not in result or not isinstance(result[dim], list):
            result[dim] = []

    _INSTRUCTION_CACHE[key] = result
    return result


def compare_against_trajectory(
    instructions: Dict[str, Any],
    task_result: Dict[str, Any],
    model,
) -> Dict[str, Dict[str, str]]:
    """LLM-judge whether the agent's trajectory + final answer followed the
    extracted instructions, per dimension.

    Returns dict with the 6 INSTRUCTION_DIMS keys, each containing
    ``{"verdict": "match|partial|violate|n_a", "evidence": "..."}``.
    """
    from ..llm_utils import call_llm_json

    # Skip the LLM call entirely if EVERY dim is empty — no point comparing.
    if all(not instructions.get(dim) for dim in INSTRUCTION_DIMS):
        return {dim: {"verdict": "n_a", "evidence": "unit gave no instructions"}
                for dim in INSTRUCTION_DIMS}

    question = task_result.get("question", task_result.get("Question", ""))
    golden_answer = task_result.get("golden_answer", "")
    ar = task_result.get("agent_result", "")
    if isinstance(ar, dict):
        agent_answer = str(ar.get("final_answer", ""))
    else:
        agent_answer = str(ar) if ar is not None else ""

    traj = task_result.get("agent_trajectory") or task_result.get("trajectory") or []
    if isinstance(traj, list):
        tail = traj[-5:]
        trajectory_tail = "\n".join(f"- {str(s)[:200]}" for s in tail)
    else:
        trajectory_tail = "(no trajectory)"

    result = call_llm_json(
        model, _COMPARE_PROMPT_TEMPLATE, {
            "question": str(question)[:240],
            "golden_answer": str(golden_answer)[:120],
            "agent_answer": agent_answer[:200],
            "trajectory_tail": trajectory_tail,
            "instructions_json": json.dumps(instructions, indent=2),
        },
        schema=COMPARE_SCHEMA,
        max_retries=2,
        retry_with_feedback=True,
    )
    if result.get("_parse_failed"):
        logger.warning("[memory_compliance] compare failed: %s",
                       result.get("_last_err"))
        return {dim: {"verdict": "n_a", "evidence": "compare LLM failed"}
                for dim in INSTRUCTION_DIMS}

    # Normalize
    for dim in INSTRUCTION_DIMS:
        if dim not in result or not isinstance(result[dim], dict):
            result[dim] = {"verdict": "n_a", "evidence": "missing in LLM output"}
        if result[dim].get("verdict") not in ("match", "partial", "violate", "n_a"):
            result[dim]["verdict"] = "n_a"
    return result


def compute_per_task_compliance(
    task_result: Dict[str, Any],
    retrieved_units_text: List[str],
    model,
    max_units: int = 3,
) -> Optional[Dict[str, Any]]:
    """Compute memory compliance for one task.

    Smart-8 fix (2026-05-16). Only the first ``max_units`` retrieved units are
    examined to keep LLM cost bounded — these are typically the highest-ranked
    candidates anyway.

    Returns:
        {
          "avg_followed_score": 0.0-1.0 (mean of per-unit followed scores),
          "violations_by_dim":  {"D3.answer_format": 2, ...} (count of violates),
          "n_units_examined":   int,
          "per_unit":           [{unit_idx, instructions, verdicts, followed_score}],
        }
        OR None if no retrieved units / agent succeeded.
    """
    if not retrieved_units_text or task_result.get("task_score", 0.0) >= 1.0:
        return None

    units = retrieved_units_text[:max_units]
    per_unit_results = []
    violation_tally: Dict[str, int] = {f"D{i+1}.{dim}": 0
                                       for i, dim in enumerate(INSTRUCTION_DIMS)}
    total_followed = 0.0
    n_units = 0

    for idx, unit_text in enumerate(units):
        unit_text = (unit_text or "").strip()
        if not unit_text:
            continue
        instr = extract_instructions(unit_text, model)
        verdicts = compare_against_trajectory(instr, task_result, model)

        # Score: match=1, partial=0.5, violate=0, n_a doesn't count
        scored = [
            (1.0 if v["verdict"] == "match"
             else 0.5 if v["verdict"] == "partial"
             else 0.0)
            for dim, v in verdicts.items()
            if v["verdict"] != "n_a"
        ]
        followed = (sum(scored) / len(scored)) if scored else 0.0

        for i, dim in enumerate(INSTRUCTION_DIMS):
            if verdicts.get(dim, {}).get("verdict") == "violate":
                violation_tally[f"D{i+1}.{dim}"] += 1

        per_unit_results.append({
            "unit_idx": idx,
            "instructions": instr,
            "verdicts": verdicts,
            "followed_score": round(followed, 3),
        })
        total_followed += followed
        n_units += 1

    if n_units == 0:
        return None

    return {
        "avg_followed_score": round(total_followed / n_units, 3),
        "violations_by_dim": {k: v for k, v in violation_tally.items() if v > 0},
        "n_units_examined": n_units,
        "per_unit": per_unit_results,
    }


def aggregate_candidate_compliance(
    per_task_compliances: List[Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate compliance across all failed tasks in a candidate evaluation.

    Returns:
        {
          "avg_followed_score": ...,
          "n_tasks_with_compliance_data": int,
          "violations_by_dim": {dim: total_count},
          "violation_distribution": {dim: pct of failures with this violation},
          "interpretation": "..."  # short human-readable summary
        }
    """
    valid = [c for c in per_task_compliances if c is not None]
    if not valid:
        return {"n_tasks_with_compliance_data": 0}

    avg = sum(c["avg_followed_score"] for c in valid) / len(valid)

    violations: Dict[str, int] = {}
    for c in valid:
        for dim, n in (c.get("violations_by_dim") or {}).items():
            violations[dim] = violations.get(dim, 0) + n

    violation_dist = {
        dim: round(n / len(valid), 3)
        for dim, n in sorted(violations.items(), key=lambda kv: -kv[1])
    }

    # Build interpretation
    if avg >= 0.75:
        interp = (
            f"Agent generally follows retrieved-memory instructions "
            f"(avg_followed_score={avg:.2f}); memory IS being effectively "
            f"acted upon. Failure mode is NOT instruction-bypass."
        )
    elif violation_dist:
        top_dim = next(iter(violation_dist))
        interp = (
            f"Agent low compliance (avg={avg:.2f}); primary violation is "
            f"{top_dim} ({violation_dist[top_dim]*100:.0f}% of compliant tasks). "
            f"Memory mutations may not fix this — investigate task-agent prompt."
        )
    else:
        interp = (
            f"Mixed compliance (avg={avg:.2f}); no single dim dominates "
            f"violations. Look at per-unit detail for case-by-case patterns."
        )

    return {
        "avg_followed_score": round(avg, 3),
        "n_tasks_with_compliance_data": len(valid),
        "violations_by_dim": violations,
        "violation_distribution": violation_dist,
        "interpretation": interp,
    }


__all__ = [
    "INSTRUCTION_DIMS",
    "INSTRUCTION_SCHEMA",
    "COMPARE_SCHEMA",
    "extract_instructions",
    "compare_against_trajectory",
    "compute_per_task_compliance",
    "aggregate_candidate_compliance",
]
