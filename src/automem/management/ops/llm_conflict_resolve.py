"""LLMConflictResolveOp — LLM-based conflict resolution.

Stage-1 (2026-05-17) adoption. Complements the existing rule-based
``conflict_detection`` op: when conflict_detection finds two units with high
embedding similarity but opposite ``task_outcome``, instead of just flagging
them as CONFLICTS, this op asks an LLM to judge which unit is more
trustworthy and deactivates the other.

Design inspiration: Letta (formerly MemGPT) archival memory consolidation
(github.com/letta-ai/letta), specifically the LLM-based memory pruning
pattern. We adopt the principle ("LLM as judge between conflicting memories")
but rewrite to fit our BaseManageOp interface; no Letta code is imported.

This op is `requires_llm=True` and runs `periodic` (not on_insert) to bound
LLM cost — it processes at most ``max_pairs_per_run`` conflict pairs per
trigger. Pairs are picked by highest cosine similarity (most likely to be
genuine duplicates / contradictions).
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit

logger = logging.getLogger(__name__)


_CONFLICT_RESOLUTION_PROMPT = """You are a memory curator for an LLM agent system.

Two memory units appear to be in conflict (high embedding similarity, opposite recommended actions or claims). Decide which one should be KEPT as the more trustworthy record, and which should be DEACTIVATED.

## Unit A (id={{ unit_a_id }})
- created at: {{ unit_a_created_at }}
- access_count: {{ unit_a_access_count }}
- success_count: {{ unit_a_success_count }} / failures: {{ unit_a_failure_count }}
- confidence: {{ unit_a_confidence }}
- content (truncated):
{{ unit_a_content }}

## Unit B (id={{ unit_b_id }})
- created at: {{ unit_b_created_at }}
- access_count: {{ unit_b_access_count }}
- success_count: {{ unit_b_success_count }} / failures: {{ unit_b_failure_count }}
- confidence: {{ unit_b_confidence }}
- content (truncated):
{{ unit_b_content }}

## Decision rules

Prefer the unit that has:
1. Higher empirical success_count (it has worked more often).
2. Higher overall confidence.
3. More recent created_at (newer evidence supersedes older).
4. Higher access_count (more frequently used, suggesting consistent value).

If neither unit is clearly superior — they truly conflict and BOTH have evidence — choose `keep_both` and let downstream code merge their evidence with a `superseded` relation.

Output strictly valid JSON only, no markdown:

{
  "decision":  "keep_a" | "keep_b" | "keep_both",
  "reasoning": "<1-2 sentence rationale citing the decision rule used>"
}"""


def _truncate_content(content: Any, max_chars: int = 300) -> str:
    """Render a MemoryUnit's content field as a short string."""
    if isinstance(content, dict):
        parts = [f"{k}: {v}" for k, v in content.items() if isinstance(v, (str, int, float))]
        text = " | ".join(parts)
    elif isinstance(content, (list, tuple)):
        text = " | ".join(str(x) for x in content)
    else:
        text = str(content or "")
    return text[:max_chars]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class LLMConflictResolveOp(BaseManageOp):
    """LLM-judged resolution of pairs flagged by conflict_detection.

    Differs from ``ConflictDetectionOp`` which only flags pairs via embedding
    similarity. This op makes a binding kept/discarded decision using an LLM
    judge — turning conflicts into actual memory churn.

    Config options:
        max_pairs_per_run (int): cap on LLM calls per periodic trigger
            (default: 5)
        similarity_threshold (float): minimum cosine between unit pair to
            consider as conflict candidate (default: 0.80, lower than
            conflict_detection's 0.85 so we catch borderline cases)
        require_opposite_outcome (bool): if True (default), only pairs with
            opposite task_outcome are processed; if False, any high-similarity
            pair is examined (useful for stale-memory cleanup).
    """

    op_name = "llm_conflict_resolve"
    op_group = "deduplication"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = True
    requires_embedding = True
    rl_action_id = 20  # next free id after the existing 19 ops

    def execute(self, context: Dict[str, Any]) -> OpResult:
        result = OpResult(op_name=self.op_name)
        t0 = time.time()

        max_pairs = int(self.config.get("max_pairs_per_run", 5))
        sim_threshold = float(self.config.get("similarity_threshold", 0.80))
        require_opp_outcome = bool(self.config.get("require_opposite_outcome", True))

        # ManagementPipeline wires the LLM client at op instantiation
        # (BaseManageOp.__init__ stores it as `self.llm_client`). Earlier
        # versions of this op tried `context["management_llm"]`, which the
        # pipeline never sets — so the op silently no-op'd every periodic
        # tick. Codex F-4 fix (2026-05-18). Fall back to context only for
        # legacy callers that bypass the pipeline.
        llm = (
            self.llm_client
            or (context.get("management_llm") if context else None)
            or (context.get("llm_model") if context else None)
        )
        if llm is None:
            logger.warning("[llm_conflict_resolve] no LLM client; skipping")
            result.details = {"skipped": True, "reason": "no_llm_client"}
            result.duration_ms = (time.time() - t0) * 1000
            return result

        try:
            from automem.llm_utils import call_llm_json
        except Exception as e:
            logger.warning("[llm_conflict_resolve] failed to import call_llm_json: %s", e)
            result.details = {"skipped": True, "reason": "import_failed"}
            result.duration_ms = (time.time() - t0) * 1000
            return result

        # Step 1: gather conflict candidates ─────────────────────────────
        all_units = self.store.get_all()
        active = [u for u in all_units if u.is_active and u.embedding is not None]
        candidates = self._find_conflict_pairs(
            active, sim_threshold, require_opp_outcome,
        )
        # Sort by similarity descending so the most-likely duplicates come first.
        candidates.sort(key=lambda p: -p[2])
        candidates = candidates[:max_pairs]

        kept_a = kept_b = kept_both = 0
        deactivated_units: List[MemoryUnit] = []

        # Step 2: LLM judge each pair ────────────────────────────────────
        for unit_a, unit_b, sim in candidates:
            try:
                template_vars = self._build_template_vars(unit_a, unit_b)
                response = call_llm_json(
                    llm,
                    _CONFLICT_RESOLUTION_PROMPT,
                    template_vars,
                    max_retries=2,
                    retry_with_feedback=True,
                )
                if response.get("_parse_failed"):
                    logger.warning(
                        "[llm_conflict_resolve] LLM parse failed for %s vs %s: %s",
                        unit_a.id[:8], unit_b.id[:8], response.get("_last_err"),
                    )
                    continue
                decision = (response.get("decision") or "").lower()
                if decision == "keep_a":
                    unit_b.is_active = False
                    deactivated_units.append(unit_b)
                    kept_a += 1
                elif decision == "keep_b":
                    unit_a.is_active = False
                    deactivated_units.append(unit_a)
                    kept_b += 1
                elif decision == "keep_both":
                    kept_both += 1
                    # No action — leave both active. A future cluster_merge / graph
                    # op may add a `supersedes` relation; not done here to keep
                    # this op single-purpose.
                else:
                    logger.warning(
                        "[llm_conflict_resolve] unexpected decision=%r for %s vs %s",
                        decision, unit_a.id[:8], unit_b.id[:8],
                    )
            except Exception as e:
                logger.warning(
                    "[llm_conflict_resolve] LLM call failed for %s vs %s: %s",
                    unit_a.id[:8], unit_b.id[:8], e,
                )
                continue

        # Step 3: persist deactivations ─────────────────────────────────
        # Codex F-5 fix (2026-05-18): every storage backend's `update()`
        # takes a MemoryUnit (see JsonStorage.update / VectorStorage.update),
        # not (id, patch). The previous code raised TypeError that was
        # silently swallowed → deactivations lived only on in-memory objects
        # and were lost after the next reload / canonical sync.
        for unit in deactivated_units:
            try:
                self.store.update(unit)
            except Exception as e:
                logger.warning(
                    "[llm_conflict_resolve] failed to persist deactivation "
                    "of %s: %s", unit.id[:8], e,
                )

        result.units_affected = len(deactivated_units)
        result.units_modified = len(deactivated_units)
        result.details = {
            "pairs_examined": len(candidates),
            "keep_a": kept_a,
            "keep_b": kept_b,
            "keep_both": kept_both,
            "deactivated_ids": [u.id for u in deactivated_units[:10]],
            "similarity_threshold": sim_threshold,
        }
        if deactivated_units:
            result.triggered = True
        result.duration_ms = (time.time() - t0) * 1000
        return result

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    def _find_conflict_pairs(
        self,
        active_units: List[MemoryUnit],
        sim_threshold: float,
        require_opp_outcome: bool,
    ) -> List[Tuple[MemoryUnit, MemoryUnit, float]]:
        """Find pairs of units with high cosine similarity and (optionally)
        opposite outcomes. Returns list of (a, b, similarity)."""
        pairs: List[Tuple[MemoryUnit, MemoryUnit, float]] = []
        n = len(active_units)
        # O(n^2) but bounded by n ~ 300 in practice; fast for modest pools.
        for i in range(n):
            for j in range(i + 1, n):
                u_a, u_b = active_units[i], active_units[j]
                # Outcome filter (only when both have outcome set).
                if require_opp_outcome:
                    out_a = getattr(u_a, "task_outcome", None)
                    out_b = getattr(u_b, "task_outcome", None)
                    if out_a is None or out_b is None:
                        continue
                    if out_a == out_b:
                        continue
                sim = _cosine_similarity(u_a.embedding, u_b.embedding)
                if sim < sim_threshold:
                    continue
                pairs.append((u_a, u_b, sim))
        return pairs

    def _build_template_vars(
        self, unit_a: MemoryUnit, unit_b: MemoryUnit,
    ) -> Dict[str, Any]:
        """Build the Jinja2 context for the LLM resolution prompt."""
        def _g(u, attr, default=""):
            v = getattr(u, attr, default)
            if v is None:
                return default
            return v

        return {
            "unit_a_id":               str(unit_a.id)[:8],
            "unit_a_created_at":       str(_g(unit_a, "created_at", "?")),
            "unit_a_access_count":     int(_g(unit_a, "access_count", 0) or 0),
            "unit_a_success_count":    int(_g(unit_a, "success_count", 0) or 0),
            "unit_a_failure_count":    int(_g(unit_a, "failure_count", 0) or 0),
            "unit_a_confidence":       f"{_g(unit_a, 'confidence', 0.0) or 0.0:.2f}",
            "unit_a_content":          _truncate_content(unit_a.content),
            "unit_b_id":               str(unit_b.id)[:8],
            "unit_b_created_at":       str(_g(unit_b, "created_at", "?")),
            "unit_b_access_count":     int(_g(unit_b, "access_count", 0) or 0),
            "unit_b_success_count":    int(_g(unit_b, "success_count", 0) or 0),
            "unit_b_failure_count":    int(_g(unit_b, "failure_count", 0) or 0),
            "unit_b_confidence":       f"{_g(unit_b, 'confidence', 0.0) or 0.0:.2f}",
            "unit_b_content":          _truncate_content(unit_b.content),
        }
