"""MutationAcquisition — code-layer decision of which architecture layer to
mutate this round, based on attribution diagnosis + ledger evidence.

Created 2026-05-13 as part of the H-plan modular decomposition.

Before this module, the architect prompt asked the LLM to reason from
the §2E attribution breakdown + §2F ledger and pick a mutation target.
Two problems with LLM-as-acquisition:

  1. Non-deterministic: same data, different round → different choice.
  2. Hard to ablate: can't test "is acquisition rule better than LLM intuition?"

This module produces a deterministic mutation recommendation:

  acq.select_layer_to_mutate(champion, attribution, ledger)
    → ('retrieval', reason_string)
    → ('extract_types', reason_string)
    → ('storage_routing', reason_string)
    → ('management', reason_string)
    → (None, reason_string)  # when memory layers are not the bottleneck

The proposer prompt now receives `recommended_mutation_layer` and the LLM
chooses a concrete value within that layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Map attribution category → the architecture layer most likely to fix it.
# Aligned with prompts/meta/layer_diagnosis_fixed.txt routing.
ATTRIBUTION_TO_LAYER: Dict[str, Optional[str]] = {
    "extraction_gap":         "extract_types",
    "extraction_low_quality": "extract_types",
    "retrieval_miss_topk":    "retrieval",
    "retrieval_miss_gate":    "retrieval",
    "judge_rejected_all":     "retrieval",
    "retrieval_noise":        "retrieval",
    "injection_bad":          "extract_types",  # injection runtime-fixed; fix unit content
    "memory_stale":           "management",
    "budget_capped":          None,  # infrastructure — not a memory layer
    "tool_failure":           None,
    "multimodal_failure":     None,
    "reasoning_error":        None,  # model bottleneck — memory is fine
    # F9 fix (codex review, 2026-05-16): A3 new categories.
    # Only DOMAIN_KNOWLEDGE_GAP is memory-actionable — the pool needs broader
    # warm-up sourcing, conceptually an extraction-side fix. The other four
    # (judge_format_mismatch, near_miss, numeric_precision, time_range_mismatch)
    # are judge / answer-formatting / agent-reasoning issues — memory cannot fix.
    "domain_knowledge_gap":   "extract_types",
    "judge_format_mismatch":  None,
    "near_miss":              None,
    "numeric_precision":      None,
    "time_range_mismatch":    None,
}


@dataclass
class AcquisitionRecommendation:
    """What the next mutation should target."""

    layer: Optional[str]                # 'retrieval' / 'extract_types' / ...; None = no memory fix
    reason: str                          # human-readable rationale
    suggested_values: List[str]          # specific options to consider (under-sampled first)
    confidence: float = 0.5              # 0-1, higher = stronger signal


class MutationAcquisition:
    """Rule-based mutation target selector.

    Heuristics, ordered by priority:
      1. If attribution.diagnosis has a clear bottleneck (>50% one category),
         mutate the layer that category maps to.
      2. Otherwise, mutate the under-sampled layer (lowest evaluation count
         in ledger.component_performance).
      3. If neither signal is clear, fall back to retrieval (most stable axis).
    """

    def __init__(
        self,
        dominant_threshold: float = 0.5,
        recommended_space: Optional[Dict[str, List[str]]] = None,
    ):
        self.dominant_threshold = dominant_threshold
        # Allowed values per layer (used for "suggested under-sampled values")
        if recommended_space is None:
            from automem.architecture_space import RECOMMENDED_ARCHITECTURE_SPACE
            recommended_space = RECOMMENDED_ARCHITECTURE_SPACE
        self._space = recommended_space

    def select_layer_to_mutate(
        self,
        champion: Optional[Dict[str, Any]],
        attribution_summary: Optional[Dict[str, Any]],
        ledger_state: Optional[Dict[str, Any]] = None,
    ) -> AcquisitionRecommendation:
        """Decide which architecture layer to mutate.

        Args:
            champion: dict with current champion's `architecture` (for ref).
            attribution_summary: dict matching round_done.json's
                candidate_results[i].attribution_summary structure.
                Expected: {breakdown: {category: count}, failure_count: int}
            ledger_state: optional dict from ExperienceLedger.render_dict()
                or ledger.json — expected keys: component_performance,
                open_questions.

        Returns:
            AcquisitionRecommendation
        """
        # Step 1: try attribution-driven decision
        attr_decision = self._from_attribution(attribution_summary)
        if attr_decision is not None:
            return attr_decision

        # Step 2: try ledger-driven decision (under-sampled component)
        ledger_decision = self._from_ledger_undersampled(ledger_state, champion)
        if ledger_decision is not None:
            return ledger_decision

        # Step 3: fallback
        return AcquisitionRecommendation(
            layer="retrieval",
            reason="No clear signal from attribution or ledger; defaulting to retrieval (historically the most stable axis).",
            suggested_values=list(self._space.get("retrieval_types", [])),
            confidence=0.2,
        )

    # ------------------------------------------------------------------
    # Internal heuristics
    # ------------------------------------------------------------------
    def _from_attribution(
        self, attr: Optional[Dict[str, Any]],
    ) -> Optional[AcquisitionRecommendation]:
        if not isinstance(attr, dict):
            return None
        breakdown = attr.get("breakdown") or {}
        failure_count = attr.get("failure_count") or 0
        if not isinstance(breakdown, dict) or failure_count <= 0:
            return None

        # Find dominant category (excluding non-memory ones)
        memory_layer_failures = {
            cat: cnt for cat, cnt in breakdown.items()
            if cnt > 0 and ATTRIBUTION_TO_LAYER.get(cat) is not None
        }
        if not memory_layer_failures:
            # All failures are reasoning_error / tool_failure / etc.
            non_memory = sum(
                cnt for cat, cnt in breakdown.items()
                if cnt > 0 and ATTRIBUTION_TO_LAYER.get(cat) is None
            )
            return AcquisitionRecommendation(
                layer=None,
                reason=(
                    f"Memory layers are not the bottleneck "
                    f"({non_memory}/{failure_count} failures are non-memory). "
                    f"No mutation will likely help."
                ),
                suggested_values=[],
                confidence=0.9,
            )

        dominant_cat = max(memory_layer_failures.items(), key=lambda kv: kv[1])
        cat, cnt = dominant_cat
        share = cnt / failure_count
        if share < self.dominant_threshold:
            return None  # no clear dominance — try ledger

        layer = ATTRIBUTION_TO_LAYER[cat]
        # Map our layer name to the key used in RECOMMENDED_ARCHITECTURE_SPACE.
        # (Codex review fix 2026-05-13: previous f"{layer}_types" produced
        # "extract_types_types" for extract_types and returned [].)
        _layer_to_space_key = {
            "extract_types":    "extract_types",
            "retrieval":        "retrieval_types",
            "management":       "management_types",
            "storage_routing":  "storage_types",
        }
        space_key = _layer_to_space_key.get(layer, "")
        suggested = list(self._space.get(space_key, [])) if space_key else []
        return AcquisitionRecommendation(
            layer=layer,
            reason=(
                f"Attribution dominant: '{cat}' = {cnt}/{failure_count} failures "
                f"({share:.0%}) → mutate {layer}."
            ),
            suggested_values=suggested,
            confidence=min(0.9, 0.4 + share),
        )

    def _from_ledger_undersampled(
        self,
        ledger_state: Optional[Dict[str, Any]],
        champion: Optional[Dict[str, Any]],
    ) -> Optional[AcquisitionRecommendation]:
        if not isinstance(ledger_state, dict):
            return None
        comp_perf_raw = ledger_state.get("component_performance")
        if not comp_perf_raw:
            return None

        # ExperienceLedger stores component_performance as a LIST of dicts
        # (each {component, value, n_evaluations, mean_acc, ...}). Convert to
        # the per-layer nested form this fn uses internally.
        # (Codex review fix 2026-05-13: previous code assumed dict-of-dicts
        # and silently fell back to the no-op default.)
        comp_perf: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if isinstance(comp_perf_raw, list):
            for row in comp_perf_raw:
                if not isinstance(row, dict):
                    continue
                layer = row.get("component")
                value = row.get("value")
                if not layer or not value:
                    continue
                comp_perf.setdefault(str(layer), {})[str(value)] = row
        elif isinstance(comp_perf_raw, dict):
            # already in nested form (custom callers)
            for k, v in comp_perf_raw.items():
                if isinstance(v, dict):
                    comp_perf[str(k)] = v
        if not comp_perf:
            return None

        # comp_perf is keyed by layer; each value is a dict of {value: stats}.
        # Layer-level evaluation count = sum of n_evaluations across values.
        layer_total: Dict[str, int] = {}
        layer_to_undertested_values: Dict[str, List[Tuple[str, int]]] = {}
        for layer, values_dict in comp_perf.items():
            if not isinstance(values_dict, dict):
                continue
            tot = 0
            value_counts: List[Tuple[str, int]] = []
            for val, stats in values_dict.items():
                if not isinstance(stats, dict):
                    continue
                n = int(stats.get("n_evaluations", 0) or 0)
                tot += n
                value_counts.append((val, n))
            if tot > 0:
                layer_total[layer] = tot
                value_counts.sort(key=lambda vc: vc[1])  # least-sampled first
                layer_to_undertested_values[layer] = value_counts

        if not layer_total:
            return None

        # Pick the layer with lowest total exploration
        target_layer = min(layer_total.items(), key=lambda kv: kv[1])
        layer_name, total = target_layer
        undertested = [v for v, n in layer_to_undertested_values.get(layer_name, []) if n <= 1]
        suggested = undertested or [v for v, _ in layer_to_undertested_values.get(layer_name, [])][:3]

        return AcquisitionRecommendation(
            layer=layer_name,
            reason=(
                f"Ledger under-sampling: layer '{layer_name}' has only {total} cumulative "
                f"evaluations (lowest). Under-tested values: {undertested[:3] or '(all values tested ≥1)'}."
            ),
            suggested_values=suggested,
            confidence=0.5,
        )


__all__ = ["MutationAcquisition", "AcquisitionRecommendation", "ATTRIBUTION_TO_LAYER"]
