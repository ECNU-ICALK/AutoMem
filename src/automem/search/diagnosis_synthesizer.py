"""Diagnosis Synthesizer — LLM-driven consolidation of all per-candidate
diagnostic signals into a single natural-language verdict for the proposer.

Smart-9a (2026-05-17). Replaces code-based consolidation with a gpt-5.5 call
that reads 6 independent signals (rule_diagnosis, layer_diagnosis,
memory_compliance, breakdown, evidence_by_category, historical_metrics) and
emits a synthesized_verdict in natural language.

This module also defines a sibling Round-level Synthesizer (Smart-9b) that
runs once per round, combining the 3 candidates' synthesized verdicts plus the
round-level differential_diagnosis to produce a round-level summary for the
next-round proposer.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from automem.resources import prompt_path

logger = logging.getLogger(__name__)

# ── Schemas ──────────────────────────────────────────────────────────────

SYNTHESIZED_VERDICT_SCHEMA: Dict[str, Any] = {
    "primary_signal":          str,
    "confidence":              str,
    "recommended_action":      str,
    "memory_bypass":           bool,
    "out_of_scope_for_memory": bool,
    "evidence_task_ids":       list,
    "cross_source_agreement":  str,
    "reasoning":               str,
}

ROUND_LEVEL_VERDICT_SCHEMA: Dict[str, Any] = {
    "round_verdict":           str,
    "next_round_focus":        str,
    "exploration_vs_exploit":  str,
    "stop_recommendation":     bool,
    "key_finding":             str,
    "evidence_from_best":      str,
    "reasoning":               str,
}


# ── Prompt paths ─────────────────────────────────────────────────────────

_PROMPT_DIR = prompt_path("meta")
_CANDIDATE_PROMPT = str(_PROMPT_DIR / "diagnosis_synthesizer.txt")
_ROUND_PROMPT = str(_PROMPT_DIR / "round_level_synthesizer.txt")


# ── Public API ───────────────────────────────────────────────────────────

def synthesize_candidate_verdict(
    model,
    rule_diagnosis: str,
    layer_diagnosis: Dict[str, Any],
    memory_compliance: Optional[Dict[str, Any]],
    breakdown: Dict[str, int],
    evidence_by_category: Dict[str, Any],
    historical_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Smart-9a synthesizer — let gpt-5.5 consolidate 6 signals.

    Returns a dict matching ``SYNTHESIZED_VERDICT_SCHEMA``. On LLM failure,
    returns a stub dict with ``_parse_failed=True`` (from call_llm_json) so
    the caller can fall back to surfacing the raw signals.
    """
    from ..llm_utils import call_llm_json

    template_vars = {
        "rule_diagnosis": (rule_diagnosis or "").strip() or "(no rule diagnosis)",
        "layer_diagnosis_json":  json.dumps(layer_diagnosis or {}, indent=2, ensure_ascii=False),
        "memory_compliance_json": json.dumps(memory_compliance or {}, indent=2, ensure_ascii=False),
        "breakdown_json":         json.dumps(breakdown or {}, indent=2, ensure_ascii=False),
        "evidence_by_category_json": json.dumps(evidence_by_category or {}, indent=2, ensure_ascii=False),
        "historical_metrics_json":   json.dumps(historical_metrics or [], indent=2, ensure_ascii=False),
    }

    return call_llm_json(
        model, _CANDIDATE_PROMPT, template_vars,
        schema=SYNTHESIZED_VERDICT_SCHEMA,
        max_retries=3,
        retry_with_feedback=True,
    )


def synthesize_round_verdict(
    model,
    round_id: int,
    candidate_verdicts: List[Dict[str, Any]],
    differential_diagnosis: Optional[Dict[str, Any]],
    pareto_status: Dict[str, Any],
    cumulative_tracking: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Smart-9b synthesizer — round-level consolidation.

    Inputs:
      - candidate_verdicts: list of per-candidate synthesized_verdict dicts
        (already from Smart-9a). Should include config_id for traceability.
      - differential_diagnosis: best-vs-worst gap analysis (Smart-4).
      - pareto_status: e.g. {"front_size": 2, "best_fitness": 0.582, "best_acc": 0.66}.
      - cumulative_tracking: rounds_to_discovery progress.

    Returns a dict matching ``ROUND_LEVEL_VERDICT_SCHEMA``. Used by the next
    round's proposer to set high-level focus (explore vs exploit, early stop).
    """
    from ..llm_utils import call_llm_json

    template_vars = {
        "round_id": round_id,
        "n_candidates": len(candidate_verdicts),
        "candidate_verdicts_json": json.dumps(
            candidate_verdicts, indent=2, ensure_ascii=False,
        ),
        "differential_diagnosis_json": json.dumps(
            differential_diagnosis or {}, indent=2, ensure_ascii=False,
        ),
        "pareto_status_json": json.dumps(
            pareto_status or {}, indent=2, ensure_ascii=False,
        ),
        "cumulative_tracking_json": json.dumps(
            cumulative_tracking or {}, indent=2, ensure_ascii=False,
        ),
    }

    return call_llm_json(
        model, _ROUND_PROMPT, template_vars,
        schema=ROUND_LEVEL_VERDICT_SCHEMA,
        max_retries=3,
        retry_with_feedback=True,
    )


__all__ = [
    "SYNTHESIZED_VERDICT_SCHEMA",
    "ROUND_LEVEL_VERDICT_SCHEMA",
    "synthesize_candidate_verdict",
    "synthesize_round_verdict",
]
