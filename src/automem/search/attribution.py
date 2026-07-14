"""Post-hoc audit and failure attribution for architecture search.

For each failed task we classify the failure into one of five categories:

  extraction_gap      - No memories in the canonical pool matched this task type.
                        The memory system lacks the right knowledge entirely.
  retrieval_miss_gate - Relevant memories exist in the pool but were blocked
                        by the retrieval gate (e.g. score threshold or type filter).
  retrieval_miss_topk - Memories were retrieved but the right one was not in top-k.
  retrieval_noise     - Retrieved memories were irrelevant and may have hurt the agent.
  reasoning_error     - Helpful memories were retrieved but the agent still failed.

The audit requires:
  1. per-task result dicts (task_score, question, retrieved_memory_context, status)
  2. canonical pool units (list of MemoryUnit.to_dict())

We use simple TF-IDF bag-of-words overlap as proxy for semantic relevance
(no additional embedding calls needed).
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from automem.resources import prompt_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribution type
# ---------------------------------------------------------------------------

class AttributionType(str, Enum):
    EXTRACTION_GAP = "extraction_gap"
    EXTRACTION_LOW_QUALITY = "extraction_low_quality"   # NEW 2026-04-27: extracted units exist but have weak signal
    RETRIEVAL_MISS_TOPK = "retrieval_miss_topk"
    JUDGE_REJECTED_ALL = "judge_rejected_all"   # NEW (Fix 1): retriever returned candidates, LLM judge dropped all
    RETRIEVAL_NOISE = "retrieval_noise"
    INJECTION_BAD = "injection_bad"             # NEW 2026-04-27: kept memories existed but agent did not act on them
    MEMORY_STALE = "memory_stale"               # NEW 2026-04-27: kept memories had conflict_count > 0 or were superseded
    # ── 2026-05-10 (run9 R1+R2 retro analysis): non-memory infrastructure
    # failure modes that the rule-based audit was previously folding into
    # REASONING_ERROR. Each is OUT-OF-SCOPE for memory-architecture changes
    # — the proposer should not mutate retrieval/management to "fix" them.
    BUDGET_CAPPED = "budget_capped"             # agent hit max_steps before terminating
    TOOL_FAILURE = "tool_failure"               # web_search / file_inspection found nothing despite many attempts
    MULTIMODAL_FAILURE = "multimodal_failure"   # image/audio/video attachment, agent could not extract content
    INFRA_ERROR = "infra_error"                 # NEW 2026-07-07: agent run raised an API/infra error (e.g. HTTP 400 context-length blowup) before answering — OUT-OF-SCOPE
    # A3 fix (2026-05-16): 5 new categories that previously got mis-routed.
    # All are OUT-OF-SCOPE for memory architecture (proposer should not mutate
    # extract/retrieval/management to "fix" them).
    JUDGE_FORMAT_MISMATCH = "judge_format_mismatch"  # agent answered correct concept but judge strict-matched
    NEAR_MISS = "near_miss"                          # answer ≥0.7 string-similar to golden, judge gave 0
    NUMERIC_PRECISION = "numeric_precision"          # both numeric, |a - g|/|g| < 0.05
    DOMAIN_KNOWLEDGE_GAP = "domain_knowledge_gap"    # pool has 0 unit with non-trivial relevance (esoteric query)
    TIME_RANGE_MISMATCH = "time_range_mismatch"      # answer 答错时间窗 (golden 限定时间 X 但 agent 用 ALL-time)
    REASONING_ERROR = "reasoning_error"
    SUCCESS = "success"
    # Deprecated: retained for backward compatibility with old runs
    RETRIEVAL_MISS_GATE = "retrieval_miss_gate"


# A3 fix helpers (placed at top so the rule decision tree can call them)

_TIME_KEYWORDS = ("year", "month", "century", "decade", "since", "until",
                  "between", "before", "after", "first", "last", "earliest",
                  "latest", "from", "to", "during", "as of")


def _detect_near_miss(golden: Any, agent_answer: Any) -> bool:
    """Return True if agent's answer is ≥0.7 string-similar to golden.

    Catches the failure mode where agent said "Egalitarianism" and golden is
    "egalitarian" — strict judge gives 0 but the underlying concept is right.
    """
    import difflib
    if golden is None or agent_answer is None:
        return False
    g = str(golden).strip().lower()
    a = str(agent_answer).strip().lower()
    if not g or not a:
        return False
    ratio = difflib.SequenceMatcher(None, g, a).ratio()
    return ratio >= 0.7


def _detect_numeric_precision(golden: Any, agent_answer: Any,
                              rel_tol: float = 0.05) -> bool:
    """Return True if both golden and agent are numeric within rel_tol."""
    try:
        g = float(str(golden).strip().replace(",", ""))
        a = float(str(agent_answer).strip().replace(",", ""))
        if g == 0.0:
            return abs(a) < rel_tol
        return abs(g - a) / abs(g) < rel_tol
    except (ValueError, TypeError):
        return False


def _detect_time_range_mismatch(question: Any, golden: Any,
                                agent_answer: Any) -> bool:
    """Heuristic: question mentions time keyword AND agent answer is numeric
    AND agent's number ≫ golden's number (×3 or more). Suggests the agent
    didn't restrict to the asked time window.
    """
    if not question:
        return False
    q = str(question).lower()
    if not any(k in q for k in _TIME_KEYWORDS):
        return False
    try:
        g = float(str(golden).strip().replace(",", ""))
        a = float(str(agent_answer).strip().replace(",", ""))
        if g <= 0 or a <= 0:
            return False
        ratio = a / g
        return ratio >= 3.0 or ratio <= 1.0 / 3.0
    except (ValueError, TypeError):
        return False


# ── 2026-05-10 detection helpers ────────────────────────────────────────────

_MULTIMODAL_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".mp3", ".wav", ".m4a", ".flac",
    ".mp4", ".mov", ".avi", ".mkv",
}

_TOOL_FAILURE_PHRASES = (
    "unable to determine", "unable to locate", "unable to verify",
    "unable to extract", "unable to access", "unable to find",
    "could not find", "could not locate", "could not retrieve",
    "no results found", "no information available", "no data available",
    "not available online", "after extensive searching",
    "after numerous searches", "404", "access denied",
    "page not found", "broken link", "blog post is no longer",
)

_MULTIMODAL_FAILURE_PHRASES = (
    "unable to view", "cannot view", "unable to read the image",
    "cannot process the audio", "unable to transcribe",
    "unable to inspect the image", "image cannot be accessed",
    "audio file could not be processed",
)


def _detect_budget_capped(task_data: Dict[str, Any], max_steps: int) -> bool:
    """Return True if the agent's trajectory length is within 5% of max_steps —
    likely truncated by the step cap before reaching a real answer.

    Heuristic conservative (95% threshold) so we don't false-positive on tasks
    that legitimately needed many steps but finished on their own.
    """
    if max_steps <= 0:
        return False
    traj = task_data.get("agent_trajectory") or []
    return len(traj) >= int(max_steps * 0.95)


def _detect_tool_failure(task_data: Dict[str, Any], min_search_attempts: int = 3) -> bool:
    """Return True iff (a) agent's final answer contains a "could not find"-style
    phrase, AND (b) the agent actually attempted ≥ min_search_attempts web
    searches. The conjunction prevents false positives on tasks where the agent
    confidently — but wrongly — answered without searching enough."""
    ar = task_data.get("agent_result")
    if isinstance(ar, dict):
        answer = str(ar.get("final_answer", ""))
    else:
        answer = str(ar) if ar is not None else ""
    answer_lo = answer.lower()
    matched_phrase = any(p in answer_lo for p in _TOOL_FAILURE_PHRASES)
    if not matched_phrase:
        return False
    msgs = task_data.get("agent_messages") or []
    web_count = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        content_repr = str(m.get("content", ""))[:1000]
        if "web_search" in content_repr or "visit_webpage" in content_repr:
            web_count += 1
    return web_count >= min_search_attempts


def _detect_multimodal_failure(task_data: Dict[str, Any]) -> bool:
    """Return True iff the task has an image/audio/video attachment AND the
    agent's final answer contains a 'cannot view/process' phrase, OR the
    trajectory shows no successful image_inspection / audio_transcribe call."""
    file_name = (task_data.get("file_name") or "").lower()
    ext = ""
    if "." in file_name:
        ext = "." + file_name.rsplit(".", 1)[-1]
    if ext not in _MULTIMODAL_EXTS:
        return False
    ar = task_data.get("agent_result")
    if isinstance(ar, dict):
        answer = str(ar.get("final_answer", ""))
    else:
        answer = str(ar) if ar is not None else ""
    answer_lo = answer.lower()
    return any(p in answer_lo for p in _MULTIMODAL_FAILURE_PHRASES)


@dataclass
class AttributionResult:
    """Attribution for a single task."""

    task_id: str
    question: str
    task_score: float
    attribution: AttributionType
    evidence: str                          # one-line explanation
    pool_matches: int = 0                  # how many pool units matched the query
    retrieved_count: int = 0              # how many units were actually retrieved


@dataclass
class AuditSummary:
    """Aggregate attribution statistics for one candidate evaluation."""

    total_tasks: int = 0
    success_count: int = 0
    extraction_gap: int = 0
    extraction_low_quality: int = 0
    retrieval_miss_gate: int = 0
    judge_rejected_all: int = 0
    retrieval_miss_topk: int = 0
    retrieval_noise: int = 0
    injection_bad: int = 0
    memory_stale: int = 0
    # ── 2026-05-10: out-of-scope (non-memory) failure categories ──
    budget_capped: int = 0
    tool_failure: int = 0
    multimodal_failure: int = 0
    infra_error: int = 0
    reasoning_error: int = 0
    # A3 fix (2026-05-16): new fine-grained categories.
    judge_format_mismatch: int = 0
    near_miss: int = 0
    numeric_precision: int = 0
    domain_knowledge_gap: int = 0
    time_range_mismatch: int = 0

    # Derived rates
    @property
    def failure_count(self) -> int:
        return self.total_tasks - self.success_count

    @property
    def extractable_failure_rate(self) -> float:
        """Fraction of failures caused by extraction gap (memory never learned)."""
        if self.failure_count == 0:
            return 0.0
        return self.extraction_gap / self.failure_count

    @property
    def retrieval_failure_rate(self) -> float:
        """Fraction of failures caused by retrieval-side issues (knowledge exists but not used).
        Includes: retrieval_miss_topk + judge_rejected_all + legacy retrieval_miss_gate."""
        if self.failure_count == 0:
            return 0.0
        return (
            self.retrieval_miss_topk
            + self.judge_rejected_all
            + self.retrieval_miss_gate
        ) / self.failure_count

    def to_dict(self) -> Dict[str, Any]:
        # B2 fix (2026-05-16): rename "diagnosis" → "rule_diagnosis" to make
        # explicit this is rule-based (not LLM). Keep "diagnosis" as an alias
        # for ≥1 release so in-flight runs / old ledger readers don't break.
        # The LLM-produced layer diagnosis is stored separately as
        # "layer_diagnosis" by save_attribution_report (B1 fix).
        rule_diag = self._diagnose()
        return {
            "total_tasks": self.total_tasks,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "breakdown": {
                "extraction_gap": self.extraction_gap,
                "extraction_low_quality": self.extraction_low_quality,
                "retrieval_miss_topk": self.retrieval_miss_topk,
                "judge_rejected_all": self.judge_rejected_all,
                "retrieval_noise": self.retrieval_noise,
                "injection_bad": self.injection_bad,
                "memory_stale": self.memory_stale,
                "budget_capped": self.budget_capped,
                "tool_failure": self.tool_failure,
                "multimodal_failure": self.multimodal_failure,
                "infra_error": self.infra_error,
                "reasoning_error": self.reasoning_error,
                # A3 fix (2026-05-16): new fine-grained categories.
                "judge_format_mismatch": self.judge_format_mismatch,
                "near_miss": self.near_miss,
                "numeric_precision": self.numeric_precision,
                "domain_knowledge_gap": self.domain_knowledge_gap,
                "time_range_mismatch": self.time_range_mismatch,
                # Deprecated field kept for backward compatibility with old runs
                "retrieval_miss_gate": self.retrieval_miss_gate,
            },
            "rates": {
                "extractable_failure_rate": round(self.extractable_failure_rate, 4),
                "retrieval_failure_rate": round(self.retrieval_failure_rate, 4),
            },
            "rule_diagnosis": rule_diag,
            # Deprecated alias — readers should migrate to rule_diagnosis.
            "diagnosis": rule_diag,
        }

    def _diagnose(self) -> str:
        """One-line LLM-readable diagnosis of the dominant failure mode (post Fix 1)."""
        if self.failure_count == 0:
            return "No failures — all tasks succeeded."

        counts = {
            "extraction_gap": self.extraction_gap,
            "extraction_low_quality": self.extraction_low_quality,
            "retrieval_miss_topk": self.retrieval_miss_topk,
            "judge_rejected_all": self.judge_rejected_all,
            "retrieval_noise": self.retrieval_noise,
            "injection_bad": self.injection_bad,
            "memory_stale": self.memory_stale,
            "budget_capped": self.budget_capped,
            "tool_failure": self.tool_failure,
            "multimodal_failure": self.multimodal_failure,
            "infra_error": self.infra_error,
            "reasoning_error": self.reasoning_error,
            # Legacy category kept only for backward-compat scoring
            "retrieval_miss_gate": self.retrieval_miss_gate,
        }
        dominant = max(counts, key=counts.get)
        pct = counts[dominant] / self.failure_count * 100

        diagnoses = {
            "extraction_gap": (
                f"{pct:.0f}% of failures are extraction_gap: "
                "the pool lacks memories for these task types — add more diverse "
                "extract_types, run more warm-up, or broaden pool sourcing."
            ),
            "retrieval_miss_topk": (
                f"{pct:.0f}% of failures are retrieval_miss_topk: "
                "memories exist but retriever did not surface them — try a different "
                "retrieval strategy (semantic vs keyword vs hybrid), increase top_k, "
                "or change storage_routing."
            ),
            "judge_rejected_all": (
                f"{pct:.0f}% of failures are judge_rejected_all (post-Fix1 category): "
                "retriever surfaced candidates but the LLM consistency-judge dropped "
                "ALL of them. This means the retriever is surfacing semantically-off "
                "candidates that only look relevant by TF-IDF. Try: (1) tighter retriever "
                "(cbr/tag over semantic), (2) expand pool with in-domain memories, "
                "(3) reconsider extract_types for this task family. "
                "Do NOT adjust gate_threshold (it is deprecated)."
            ),
            "retrieval_noise": (
                f"{pct:.0f}% of failures are retrieval_noise: "
                "judge let candidates through but they were not actually useful — "
                "tighten retriever quality or enable management ops to prune low-quality memories."
            ),
            "reasoning_error": (
                f"{pct:.0f}% of failures are reasoning_error: "
                "memories retrieved correctly but agent reasoning failed — "
                "memory system is not the bottleneck for these tasks."
            ),
            "retrieval_miss_gate": (
                f"{pct:.0f}% of failures are labeled retrieval_miss_gate (legacy): "
                "this category is deprecated after Fix 1; modern runs should see "
                "judge_rejected_all instead. If this shows up in a new run, "
                "provider/attribution pipeline version is mismatched."
            ),
            "extraction_low_quality": (
                f"{pct:.0f}% of failures are extraction_low_quality: "
                "memories WERE extracted for these task families but their "
                "quality / actionability was poor (low judge keep rate or low "
                "average confidence). Try: (1) tighten extract prompts toward "
                "reusable 'recipes' rather than task-specific facts, (2) lower "
                "extract budget to favor quality over volume, (3) add "
                "utility_audit pruning."
            ),
            "injection_bad": (
                f"{pct:.0f}% of failures are injection_bad: "
                "retrieval and context composition both passed, but the agent did "
                "not use the cited guidance. Runtime injection is fixed; improve "
                "the encoded unit's actionability or retrieval relevance."
            ),
            "memory_stale": (
                f"{pct:.0f}% of failures are memory_stale: "
                "kept memories had unresolved conflict_count or were superseded. "
                "Try: (1) enable reflection_correction, (2) tighten "
                "conflict_detection threshold, (3) shorten time_decay half-life."
            ),
            "budget_capped": (
                f"{pct:.0f}% of failures are budget_capped: "
                "agent hit max_steps before producing a real answer. This is "
                "OUT-OF-SCOPE for memory architecture changes — "
                "DO NOT mutate retrieval/management/extract_types to fix; "
                "increase --max_steps or accept these as agent-budget-bound."
            ),
            "tool_failure": (
                f"{pct:.0f}% of failures are tool_failure: "
                "agent attempted multiple web searches but the underlying "
                "information was not findable (deleted page, dead link, "
                "absent from indexed sources). OUT-OF-SCOPE for memory "
                "architecture; proposer should NOT mutate retrieval to fix."
            ),
            "multimodal_failure": (
                f"{pct:.0f}% of failures are multimodal_failure: "
                "task has image / audio / video attachment that the agent "
                "could not view or process. Perception-layer issue — "
                "OUT-OF-SCOPE for memory architecture changes."
            ),
            "infra_error": (
                f"{pct:.0f}% of failures are infra_error: "
                "the agent run raised an API/infrastructure error (e.g. HTTP 400 "
                "from a context-length blowup) before producing an answer. "
                "OUT-OF-SCOPE for memory architecture changes — do NOT mutate "
                "extract/retrieval/management; reduce token growth or retry."
            ),
        }
        return diagnoses.get(dominant, f"{pct:.0f}% of failures are {dominant} — see breakdown.")


# ---------------------------------------------------------------------------
# Layer-level metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_layer_metrics(task_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate per-task memory_metrics into four-layer statistics.

    Returns a dict with keys: encode, store, retrieve, manage — each
    containing numeric summaries that an LLM diagnostician can reason over.
    """
    n = len(task_results)
    if n == 0:
        return {"encode": {}, "store": {}, "retrieve": {}, "manage": {}}

    # ---- Encode ----
    total_extracted = 0
    total_inserted = 0
    total_deduped = 0

    # ---- Store ----
    pool_sizes: List[int] = []
    backends_seen: Set[str] = set()

    # ---- Retrieve ----
    total_ret_calls = 0
    total_retrieved = 0
    empty_ret_calls = 0
    retriever_name = None

    # ---- Manage ----
    mgmt_preset = None
    ops_agg: Dict[str, Dict[str, int]] = {}  # op_name -> counters

    for r in task_results:
        mm = r.get("memory_metrics") or {}

        # Encode
        total_extracted += mm.get("num_extracted", 0)
        total_inserted += mm.get("num_inserted", 0)
        total_deduped += mm.get("num_deduped", 0)

        # Store
        backend = mm.get("storage_backend")
        if backend:
            backends_seen.add(backend)
        pool_size = mm.get("num_memory_units", 0)
        if pool_size > 0:
            pool_sizes.append(pool_size)

        # Retrieve
        calls = mm.get("retrieval_calls", 0)
        retrieved = mm.get("num_retrieved", 0)
        total_ret_calls += calls
        total_retrieved += retrieved
        if calls > 0 and retrieved == 0:
            empty_ret_calls += 1
        if mm.get("retriever_name"):
            retriever_name = mm["retriever_name"]

        # Manage
        if mm.get("management_preset"):
            mgmt_preset = mm["management_preset"]
        for op in mm.get("management_results", []):
            name = op.get("op_name", "unknown")
            if name not in ops_agg:
                ops_agg[name] = {
                    "invoked": 0, "triggered": 0,
                    "affected": 0, "deleted": 0, "modified": 0,
                }
            ops_agg[name]["invoked"] += 1
            if op.get("triggered"):
                ops_agg[name]["triggered"] += 1
            ops_agg[name]["affected"] += op.get("units_affected", 0)
            ops_agg[name]["deleted"] += op.get("units_deleted", 0)
            ops_agg[name]["modified"] += op.get("units_modified", 0)

    return {
        "encode": {
            "total_extracted": total_extracted,
            "total_inserted": total_inserted,
            "total_deduped": total_deduped,
            "avg_per_task": round(total_extracted / n, 1) if n else 0,
            "dedup_rate": round(total_deduped / total_extracted, 3) if total_extracted else 0,
        },
        "store": {
            "backends": sorted(backends_seen),
            # E10 fix (2026-05-16): use min/max instead of first/last to be
            # robust to file ordering — task_results comes from sorted-by-
            # filename glob, which is NOT guaranteed chronological. Previously
            # this could report negative pool growth on non-monotone filenames,
            # confusing the LLM diagnostician.
            "initial_pool": min(pool_sizes) if pool_sizes else 0,
            "final_pool": max(pool_sizes) if pool_sizes else 0,
            "growth": (max(pool_sizes) - min(pool_sizes)) if pool_sizes else 0,
        },
        "retrieve": {
            "strategy": retriever_name,
            "total_calls": total_ret_calls,
            "total_retrieved": total_retrieved,
            "empty_calls": empty_ret_calls,
            "empty_rate": round(empty_ret_calls / n, 3) if n else 0,
            "avg_per_task": round(total_retrieved / n, 1) if n else 0,
        },
        "manage": {
            "preset": mgmt_preset,
            "ops": ops_agg,
        },
    }


# ---------------------------------------------------------------------------
# LLM-powered four-layer diagnosis
# ---------------------------------------------------------------------------

# 2026-07-11: switched to the _fixed variant, which whitelists the five valid
# extract_types and forbids invented ones (the original prompt let the
# diagnosis LLM hallucinate a "fact" extract type in ~17 real run outputs,
# polluting the experience ledger) and adds an anti-thrash rule for
# retriever flip-flopping. NOTE: this file is a digest input — resuming a
# run PAUSED before this change will see digest drift (rounds wiped).
_DIAGNOSIS_PROMPT = str(prompt_path("meta", "layer_diagnosis_fixed.txt"))


def _select_evidence_tasks(
    attributions: List["AttributionResult"],
    task_results: List[Dict[str, Any]],
    top_k_per_category: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pick representative tasks per attribution category for evidence-grounded
    LLM diagnosis (Smart-1 fix, 2026-05-16).

    For each non-SUCCESS category, picks up to ``top_k_per_category`` tasks
    ordered by lowest ``task_score`` (worst first). The diagnosis prompt
    receives these so the LLM cites real task_ids instead of just citing
    aggregate counts.

    Parameters
    ----------
    attributions : list of AttributionResult from run_posthoc_audit
    task_results : raw task result dicts (from tasks/*.json)
    top_k_per_category : how many representative tasks per category (default 3)

    Returns
    -------
    Dict[category_name, List[evidence_dict]] where each evidence_dict has:
        task_id, question, golden_answer, agent_answer,
        retrieved_units_snippet, rule_attribution_evidence.
    """
    from collections import defaultdict
    by_cat: Dict[str, List["AttributionResult"]] = defaultdict(list)
    for a in attributions:
        if a.attribution == AttributionType.SUCCESS:
            continue
        by_cat[a.attribution.value].append(a)

    result_lookup = {str(r.get("task_id") or r.get("id") or ""): r
                     for r in task_results}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for cat, attrs in by_cat.items():
        # Worst task_score first (smallest score = clearest failure case)
        sorted_attrs = sorted(attrs, key=lambda a: a.task_score)
        top = sorted_attrs[:top_k_per_category]

        evidence: List[Dict[str, Any]] = []
        for a in top:
            r = result_lookup.get(str(a.task_id), {})
            ar = r.get("agent_result", "")
            if isinstance(ar, dict):
                agent_answer = str(ar.get("final_answer", ""))
            else:
                agent_answer = str(ar) if ar is not None else ""
            evidence.append({
                "task_id": str(a.task_id)[:24],
                "question": (a.question or "")[:200],
                "golden_answer": str(r.get("golden_answer", ""))[:120],
                "agent_answer": agent_answer[:150],
                "retrieved_units_snippet": str(
                    r.get("retrieved_memory_text")
                    or r.get("retrieved_memory_context")
                    or ""
                )[:400],
                "rule_attribution_evidence": a.evidence,
            })
        out[cat] = evidence
    return out


def _gather_historical_metrics(
    run_dir: Any, round_id: int, max_lookback: int = 4,
) -> List[Dict[str, Any]]:
    """Read past round_done.json files and extract per-layer metric trajectory.

    Smart-2 fix (2026-05-16): give the diagnostician historical context so it
    can spot trends like ``hit_rate dropping over 3 rounds`` or ``pool size
    plateauing'' instead of only seeing the current snapshot.

    Returns a list ordered oldest → newest, one entry per past round.
    """
    from pathlib import Path as _P
    out: List[Dict[str, Any]] = []
    if run_dir is None or round_id <= 1:
        return out
    rd_root = _P(run_dir)
    start = max(1, round_id - max_lookback)
    for r in range(start, round_id):
        rd_path = rd_root / f"round_{r}" / "round_done.json"
        if not rd_path.is_file():
            continue
        try:
            rd = json.loads(rd_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Pick a small set of trend-informative metrics.
        out.append({
            "round": r,
            "best_accuracy": rd.get("round_best_accuracy"),
            "mean_accuracy": rd.get("round_mean_accuracy"),
            "best_fitness": rd.get("best_fitness"),
            "front_size": rd.get("pareto_front_size"),
            "cum_max_acc": (rd.get("cumulative_tracking") or {}).get("cumulative_max_accuracy"),
            "breakdown_top3": _top_k_breakdown(rd, k=3),
            "best_candidate": _best_candidate_summary(rd),
        })
    return out


def _top_k_breakdown(round_done: Dict[str, Any], k: int = 3) -> Dict[str, int]:
    """Return the k most frequent attribution categories across all
    candidates in this round."""
    from collections import Counter
    tally: Counter = Counter()
    for c in round_done.get("candidate_results", []) or []:
        b = ((c.get("attribution_summary") or {}).get("breakdown") or {})
        for cat, n in b.items():
            if isinstance(n, int) and n > 0:
                tally[cat] += n
    return dict(tally.most_common(k))


def _best_candidate_summary(round_done: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact summary of the best candidate from this round
    (by accuracy then fitness).

    G9 fix (codex review, 2026-05-16): filter out failed / skipped candidates
    BEFORE picking the max — otherwise a candidate whose subprocess crashed
    (metrics={}) could surface as 'best' purely because failed candidates and
    legitimate 0-acc candidates score the same (0.0, 0.0).
    """
    cands = [
        c for c in (round_done.get("candidate_results", []) or [])
        if not c.get("skipped") and not c.get("failed") and c.get("metrics")
    ]
    if not cands:
        return {}
    def _score(c):
        m = c.get("metrics") or {}
        return (m.get("accuracy", 0.0), m.get("fitness", 0.0))
    best = max(cands, key=_score)
    m = best.get("metrics") or {}
    arch = best.get("architecture") or {}
    return {
        "config_id": best.get("config_id"),
        "acc": m.get("accuracy"),
        "lift": m.get("memory_lift"),
        "hit": m.get("hit_rate"),
        "retrieval": arch.get("retrieval"),
        "management": arch.get("management_preset", arch.get("management")),
    }


def build_layer_diagnosis(
    model,
    task_results: List[Dict],
    summary: "AuditSummary",
    architecture: Dict[str, Any],
    attributions: Optional[List["AttributionResult"]] = None,
    run_dir: Any = None,
    round_id: int = 0,
) -> Dict[str, Any]:
    """Call an LLM to produce a structured four-layer diagnosis.

    Parameters
    ----------
    model : callable LLM (OpenAIServerModel or compatible)
    task_results : raw task result dicts (from tasks/*.json)
    summary : AuditSummary from run_posthoc_audit
    architecture : the candidate architecture dict
    attributions : per-task AttributionResult list. When provided, the diagnosis
        prompt also receives `evidence_by_category` — for each non-zero
        attribution category, up to 3 worst-task_score representative tasks
        (Smart-1 fix, 2026-05-16). This makes the LLM cite specific task_ids
        in its observations instead of only quoting aggregate counts.

    Returns
    -------
    Dict with keys: encode, store, retrieve, manage, overall, priority_action.
    Each layer has status/observation/suggestion.
    """
    from ..llm_utils import call_llm_json

    metrics = aggregate_layer_metrics(task_results)

    # Strip per-task detail from summary to keep prompt concise.
    # Codex Q2-2 (2026-04-28): include ALL 9 attribution classes so the
    # diagnostician does not blame retrieval/manage when the dominant
    # cause is extraction_low_quality / judge_rejected_all / injection_bad
    # / memory_stale.
    summary_compact = {
        "total_tasks": summary.total_tasks,
        "success_count": summary.success_count,
        "failure_count": summary.failure_count,
        "breakdown": {
            "extraction_gap": summary.extraction_gap,
            "extraction_low_quality": summary.extraction_low_quality,
            "retrieval_miss_topk": summary.retrieval_miss_topk,
            "judge_rejected_all": summary.judge_rejected_all,
            "retrieval_noise": summary.retrieval_noise,
            "injection_bad": summary.injection_bad,
            "memory_stale": summary.memory_stale,
            "reasoning_error": summary.reasoning_error,
            # Legacy field kept for backward compatibility with old runs
            "retrieval_miss_gate": summary.retrieval_miss_gate,
        },
        "rates": {
            "extractable_failure_rate": round(summary.extractable_failure_rate, 4),
            "retrieval_failure_rate": round(summary.retrieval_failure_rate, 4),
        },
    }

    # Smart-1 fix (2026-05-16): give the LLM concrete representative task
    # evidence per attribution category (3 worst-task_score tasks each), so
    # diagnoses are evidence-grounded rather than just citing aggregate counts.
    evidence_by_cat: Dict[str, Any] = {}
    if attributions:
        evidence_by_cat = _select_evidence_tasks(
            attributions, task_results, top_k_per_category=3,
        )

    # Smart-2 fix (2026-05-16): expose historical metric trajectory so the
    # diagnostician can spot trends (hit_rate dropping over rounds, pool
    # plateauing, breakdown shifting from extraction→retrieval, etc.).
    historical: List[Dict[str, Any]] = []
    if run_dir is not None and round_id > 1:
        historical = _gather_historical_metrics(run_dir, round_id, max_lookback=4)

    template_vars = {
        "architecture_json": json.dumps(architecture, indent=2, ensure_ascii=False),
        "metrics_json": json.dumps(metrics, indent=2, ensure_ascii=False),
        "attribution_json": json.dumps(summary_compact, indent=2),
        "n_tasks": len(task_results),
        "current_round": round_id,
        # Smart-1 fix (2026-05-16): evidence injection.
        "evidence_by_category_json": json.dumps(
            evidence_by_cat, indent=2, ensure_ascii=False,
        ) if evidence_by_cat else "",
        "n_categories_with_evidence": len(evidence_by_cat),
        # Smart-2 fix (2026-05-16): historical trajectory.
        "historical_metrics_json": json.dumps(
            historical, indent=2, ensure_ascii=False,
        ) if historical else "",
        "n_historical_rounds": len(historical),
    }

    # B6 fix (2026-05-16): pass schema so the LLM output is validated structurally,
    # not just for raw-JSON parseability. Retry with feedback when the LLM misses
    # required keys (e.g. omits "priority_action" or returns a flat string).
    layer_diag = call_llm_json(
        model, _DIAGNOSIS_PROMPT, template_vars,
        schema=LAYER_DIAGNOSIS_SCHEMA,
        max_retries=3,
        retry_with_feedback=True,
    )

    # Smart-3 fix (2026-05-16): rule-based self-check. Catch obvious
    # contradictions between LLM diagnosis and metrics (e.g. "retrieve critical"
    # but hit_rate=0.9). On contradiction, give the LLM one explicit retry
    # with the contradictions listed as feedback.
    if not layer_diag.get("_parse_failed"):
        is_consistent, contradictions = _validate_layer_diag(layer_diag, metrics)
        if not is_consistent:
            logger.info(
                "Layer diagnosis self-check failed (%d contradictions); retrying with feedback.",
                len(contradictions),
            )
            template_vars["validation_feedback"] = (
                "Your previous diagnosis contradicted the metrics:\n  - "
                + "\n  - ".join(contradictions)
                + "\n\nReconsider each layer's status using the actual metric values, "
                "then re-emit the JSON."
            )
            layer_diag = call_llm_json(
                model, _DIAGNOSIS_PROMPT, template_vars,
                schema=LAYER_DIAGNOSIS_SCHEMA,
                max_retries=2,
                retry_with_feedback=True,
            )
            # Record the contradictions even if retry didn't fix them.
            if isinstance(layer_diag, dict) and not layer_diag.get("_parse_failed"):
                layer_diag.setdefault("_self_check", {})["contradictions_round_1"] = contradictions

    return layer_diag


# Smart-9a (2026-05-17): thin wrapper for `synthesize_candidate_verdict` so
# attribution.py's existing import surface contains the consolidation step too.
# Smart-10 (2026-05-17): A3↔subclass reclassification — when the LLM
# subclassifier produces a finer subclass that matches one of the new A3 top
# categories (unit_or_format_mismatch / numeric_calculation_error /
# time_range_misinterpretation), promote that task to the top category so
# the breakdown is single-source-of-truth.

_SUBCLASS_TO_TOP_CATEGORY: Dict[str, "AttributionType"] = {
    "unit_or_format_mismatch":      AttributionType.NEAR_MISS,
    "numeric_calculation_error":    AttributionType.NUMERIC_PRECISION,
    "time_range_misinterpretation": AttributionType.TIME_RANGE_MISMATCH,
}


def reclassify_reasoning_from_subclasses(
    attributions: List[AttributionResult],
    reasoning_subclasses: Optional[Dict[str, Dict[str, str]]],
    audit_summary: AuditSummary,
) -> Dict[str, int]:
    """Smart-10 (2026-05-17): promote REASONING_ERROR tasks whose LLM subclass
    matches a more specific A3 top category. Mutates ``attributions`` and
    ``audit_summary`` in place; returns a {old_class → new_class: count} dict
    for logging / audit.

    Without this step, a task is double-counted: once as REASONING_ERROR in the
    breakdown and once as a subclass in reasoning_error_subclasses. After
    reclassification, breakdown is the single source of truth.

    I3 fix (codex review, 2026-05-17): explicit idempotency guard using
    ``_reclassified_by_s10`` sentinel attribute. Without this the function
    was only idempotent because subsequent passes saw ``a.attribution !=
    REASONING_ERROR`` — that protection breaks if a future caller scans
    by ``a.attribution in _SUBCLASS_TO_TOP_CATEGORY.values()``. The sentinel
    makes the contract explicit and protects the evidence prefix too.
    """
    if not reasoning_subclasses:
        return {}

    movements: Dict[str, int] = {}
    for a in attributions:
        # I3 fix: explicit idempotency.
        if getattr(a, "_reclassified_by_s10", False):
            continue
        if a.attribution != AttributionType.REASONING_ERROR:
            continue
        info = reasoning_subclasses.get(a.task_id, {}) or {}
        sub = info.get("subclass")
        if sub not in _SUBCLASS_TO_TOP_CATEGORY:
            continue
        new_attr = _SUBCLASS_TO_TOP_CATEGORY[sub]
        # Move count from REASONING_ERROR to the new top category.
        audit_summary.reasoning_error = max(0, audit_summary.reasoning_error - 1)
        setattr(audit_summary, new_attr.value,
                getattr(audit_summary, new_attr.value, 0) + 1)
        a.attribution = new_attr
        a.evidence = (f"[reclassified from REASONING_ERROR via subclass={sub}] "
                      + a.evidence)
        # I3 fix: mark so re-runs (resume paths / future refactors) don't
        # double-decrement.
        a._reclassified_by_s10 = True
        key = f"REASONING_ERROR→{new_attr.value}"
        movements[key] = movements.get(key, 0) + 1
    if movements:
        logger.info(
            "[Smart-10] Reclassified %d task(s) from REASONING_ERROR to A3 categories: %s",
            sum(movements.values()), movements,
        )
    return movements


def build_synthesized_verdict(
    model,
    rule_diagnosis: str,
    layer_diagnosis: Dict[str, Any],
    memory_compliance: Optional[Dict[str, Any]],
    breakdown: Dict[str, int],
    evidence_by_category: Dict[str, Any],
    historical_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Smart-9a (2026-05-17): synthesize 6 diagnostic signals into a single
    natural-language verdict via gpt-5.5.

    Thin wrapper around ``diagnosis_synthesizer.synthesize_candidate_verdict``
    that lives in attribution.py for caller convenience.
    """
    from .diagnosis_synthesizer import synthesize_candidate_verdict
    return synthesize_candidate_verdict(
        model=model,
        rule_diagnosis=rule_diagnosis,
        layer_diagnosis=layer_diagnosis,
        memory_compliance=memory_compliance,
        breakdown=breakdown,
        evidence_by_category=evidence_by_category,
        historical_metrics=historical_metrics,
    )


def _validate_layer_diag(
    layer_diag: Dict[str, Any], metrics: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Rule-based contradiction check between LLM diagnosis and metrics.

    Smart-3 fix (2026-05-16). Returns (is_consistent, [contradictions]).

    Currently catches:
      - retrieve.status='critical' but hit_rate >= 0.7
      - encode.status='critical' but extraction_count >= 100
      - store.status='critical' but final_pool >= 50
      - manage.status='critical' but manage.ops is non-empty AND no memory_stale failures
      - overall claims a layer as bottleneck but that layer's status != 'critical'/'warning'
    """
    contradictions: List[str] = []

    retrieve = metrics.get("retrieve") or {}
    encode = metrics.get("encode") or {}
    store = metrics.get("store") or {}

    # G1 fix (codex review, 2026-05-16): use the actual key names emitted by
    # aggregate_layer_metrics. Previously we read `extraction_count` and
    # `avg_per_task` — neither exists, so 2 of 3 status-vs-metric rules never
    # fired. The real keys: encode["total_extracted"], retrieve["empty_rate"].
    empty_rate = retrieve.get("empty_rate", 0)
    extraction_count = encode.get("total_extracted", 0)
    final_pool = store.get("final_pool", 0)

    d_retrieve = (layer_diag.get("retrieve") or {}).get("status", "")
    d_encode = (layer_diag.get("encode") or {}).get("status", "")
    d_store = (layer_diag.get("store") or {}).get("status", "")

    if d_retrieve == "critical" and empty_rate < 0.3:
        contradictions.append(
            f"retrieve.status='critical' but empty_rate={empty_rate:.2f} (<0.3 means retrieval is mostly succeeding)"
        )
    if d_encode == "critical" and extraction_count >= 100:
        contradictions.append(
            f"encode.status='critical' but total_extracted={extraction_count} (>=100 means encoding is producing units)"
        )
    if d_store == "critical" and final_pool >= 50:
        contradictions.append(
            f"store.status='critical' but final_pool={final_pool} (>=50 means store has units)"
        )

    # overall must match at least one layer's status
    overall = (layer_diag.get("overall") or "").lower()
    statuses = {
        "encode": d_encode, "store": d_store, "retrieve": d_retrieve,
        "manage": (layer_diag.get("manage") or {}).get("status", ""),
    }
    for layer in ["encode", "store", "retrieve", "manage"]:
        # if overall claims this layer as bottleneck, that layer should be warning/critical
        if f"{layer} is the primary bottleneck" in overall or f"{layer} layer is bottleneck" in overall:
            if statuses[layer] not in ("warning", "critical"):
                contradictions.append(
                    f"overall claims {layer} is bottleneck but {layer}.status='{statuses[layer]}'"
                )

    return (len(contradictions) == 0, contradictions)


# B6 fix (2026-05-16): schema for build_layer_diagnosis LLM output.
LAYER_DIAGNOSIS_SCHEMA: Dict[str, Any] = {
    "encode":   {"status", "observation", "suggestion"},
    "store":    {"status", "observation", "suggestion"},
    "retrieve": {"status", "observation", "suggestion"},
    "manage":   {"status", "observation", "suggestion"},
    "overall":         str,
    "priority_action": str,
}


# Smart-4 fix (2026-05-16): schema for build_differential_diagnosis LLM output.
# Note: ``confounder`` is intentionally NOT in the schema — the prompt allows it
# to be an empty string when no confounder exists, but our _validate_schema
# rejects empty strings for `str`-typed fields. Downstream readers can safely
# call ``.get("confounder", "")``.
DIFFERENTIAL_DIAGNOSIS_SCHEMA: Dict[str, Any] = {
    "key_change":                    str,
    "impact_attribution":            str,
    "secondary_factors":             list,
    "recommendation_for_next_round": str,
    "confidence":                    str,
}


_DIFF_DIAGNOSIS_PROMPT = str(prompt_path("meta", "differential_diagnosis.txt"))


def build_differential_diagnosis(
    model,
    best_candidate: Dict[str, Any],
    worst_candidate: Dict[str, Any],
    round_id: int = 0,
) -> Dict[str, Any]:
    """LLM-driven differential diagnosis between the best and worst candidate
    of a round.

    Smart-4 fix (2026-05-16). Augments the per-candidate `layer_diagnosis` by
    explaining the *gap* — what one change caused the accuracy delta. The
    output feeds the next round's proposer prompt as a concrete "try this"
    direction.

    Parameters
    ----------
    model : callable LLM (OpenAIServerModel-compatible)
    best_candidate, worst_candidate : dict with keys
        ``config_id``, ``architecture``, ``metrics`` (raw_metrics).
    round_id : int

    Returns
    -------
    Dict matching ``DIFFERENTIAL_DIAGNOSIS_SCHEMA``, OR a stub dict with
    ``_parse_failed=True`` if all retries fail.
    """
    from ..llm_utils import call_llm_json

    best_m = best_candidate.get("metrics") or best_candidate.get("raw_metrics") or {}
    worst_m = worst_candidate.get("metrics") or worst_candidate.get("raw_metrics") or {}

    template_vars = {
        "best_config_id":    best_candidate.get("config_id", "?"),
        "worst_config_id":   worst_candidate.get("config_id", "?"),
        "best_arch_json":    json.dumps(
            best_candidate.get("architecture", {}), indent=2, ensure_ascii=False),
        "worst_arch_json":   json.dumps(
            worst_candidate.get("architecture", {}), indent=2, ensure_ascii=False),
        "best_metrics_json": json.dumps(best_m, indent=2, ensure_ascii=False),
        "worst_metrics_json":json.dumps(worst_m, indent=2, ensure_ascii=False),
        "round_id":  round_id,
        "acc_delta": round((best_m.get("accuracy") or 0.0) - (worst_m.get("accuracy") or 0.0), 4),
        "fit_delta": round((best_m.get("fitness")  or 0.0) - (worst_m.get("fitness")  or 0.0), 4),
        "hit_delta": round((best_m.get("hit_rate") or 0.0) - (worst_m.get("hit_rate") or 0.0), 4),
    }

    return call_llm_json(
        model, _DIFF_DIAGNOSIS_PROMPT, template_vars,
        schema=DIFFERENTIAL_DIAGNOSIS_SCHEMA,
        max_retries=3,
        retry_with_feedback=True,
    )


# ---------------------------------------------------------------------------
# Round-over-round step verification (R vs R-1)
# ---------------------------------------------------------------------------
#
# FGMD's cross-round "verify the step" stage. The per-candidate / per-layer
# diagnoses above explain failures *within* a round; this pure-rule check looks
# *across* rounds. It pairs the previous round's best architecture with the
# current round's best, attributes the change in the objective to the edit
# between them, and flags a direction that has stopped paying off so the
# proposer can pivot instead of descending a dead gradient. No LLM call — it
# reuses the per-round metrics the search already produced.

STEP_VERIFY_MIN_LIFT = 0.02   # accuracy delta below this is treated as noise
                              # (matches IncumbentGate's default min_lift).

_ARCH_COORDS = ("extract_types", "storage_routing", "retrieval", "management")


def _architecture_diff(prev_arch: Dict[str, Any],
                       curr_arch: Dict[str, Any]) -> Dict[str, Any]:
    """Return the coordinates that changed between two architectures.

    ``extract_types`` is compared order-insensitively (it is a set-valued
    extraction policy, not an ordered list).
    """
    changed: Dict[str, Any] = {}
    for c in _ARCH_COORDS:
        pv = (prev_arch or {}).get(c)
        cv = (curr_arch or {}).get(c)
        if c == "extract_types":
            pv = sorted(pv or [])
            cv = sorted(cv or [])
        if pv != cv:
            changed[c] = {"from": pv, "to": cv}
    return changed


def _best_candidate_of(round_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the best non-skipped candidate (by accuracy, then fitness) from a
    ``round_summary`` (``candidate_results``) or a last-round context
    (``candidates``)."""
    cands = (round_obj.get("candidate_results")
             or round_obj.get("candidates") or [])
    scored = [
        c for c in cands
        if not c.get("skipped") and not c.get("failed")
        and (c.get("metrics") or {}).get("accuracy") is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda c: (
        (c.get("metrics") or {}).get("accuracy", 0.0),
        (c.get("metrics") or {}).get("fitness", 0.0),
    ))


def _champion_round(champion: Any) -> Optional[int]:
    """Extract the round a Pareto champion was discovered in, from its
    ``config_id`` (``r{round}_c{cand}``) or a ``round_id`` field. Returns None
    when unparseable."""
    if champion is None:
        return None
    cid = (getattr(champion, "config_id", None)
           or (champion.get("config_id") if isinstance(champion, dict) else None))
    if isinstance(cid, str) and cid.startswith("r") and "_" in cid:
        try:
            return int(cid.split("_", 1)[0][1:])
        except (ValueError, IndexError):
            pass
    rid = (getattr(champion, "round_id", None)
           or (champion.get("round_id") if isinstance(champion, dict) else None))
    try:
        return int(rid) if rid is not None else None
    except (ValueError, TypeError):
        return None


def build_step_verification(
    round_id: int,
    prev_round_context: Dict[str, Any],
    curr_round_summary: Dict[str, Any],
    champion: Any = None,
    min_lift: float = STEP_VERIFY_MIN_LIFT,
) -> Optional[Dict[str, Any]]:
    """Round-over-round step verification (round ``R`` vs ``R-1``).

    Pairs the previous round's best architecture with the current round's best,
    attributes the change in the objective to the edit between them, and returns
    a structured verdict used as a validation-guided check so the search does
    not keep descending a direction that has stopped paying off.

    Pure rule-based (no LLM). Returns ``None`` when either round lacks a scored
    candidate (e.g. round 1, or an all-skipped round).

    Verdict semantics
    -----------------
      ``confirmed``    : the last edit improved accuracy by ``>= min_lift``.
      ``refuted``      : the last edit regressed accuracy by ``>= min_lift``.
      ``stalled``      : no edit beat the incumbent this round (the champion did
                         not advance) — the direction has stopped paying off.
      ``inconclusive`` : accuracy moved within the ``+/-min_lift`` noise band.
    """
    prev_best = _best_candidate_of(prev_round_context or {})
    curr_best = _best_candidate_of(curr_round_summary or {})
    if prev_best is None or curr_best is None:
        return None

    prev_m = prev_best.get("metrics") or {}
    curr_m = curr_best.get("metrics") or {}

    def _d(key: str) -> float:
        return round(float(curr_m.get(key, 0.0) or 0.0)
                     - float(prev_m.get(key, 0.0) or 0.0), 4)

    delta = {"accuracy": _d("accuracy"),
             "memory_lift": _d("memory_lift"),
             "fitness": _d("fitness")}

    edit = _architecture_diff(prev_best.get("architecture") or {},
                              curr_best.get("architecture") or {})
    champion_advanced = (_champion_round(champion) == round_id)
    changed_coords = ", ".join(edit.keys()) or "no coordinate"
    d_acc = delta["accuracy"]

    if not champion_advanced and not edit:
        verdict, paying_off = "stalled", False
        rec = ("The previous direction produced no architecture that beat the "
               "incumbent this round; it has stopped paying off — pivot to a "
               "different coordinate next round.")
    elif d_acc >= min_lift:
        verdict, paying_off = "confirmed", True
        rec = (f"The last edit ({changed_coords}) improved accuracy by "
               f"{d_acc:+.3f} (>= {min_lift}); continue along this direction.")
    elif d_acc <= -min_lift:
        verdict, paying_off = "refuted", False
        rec = (f"The last edit ({changed_coords}) regressed accuracy by "
               f"{d_acc:+.3f}; revert it and try a different coordinate.")
    else:
        verdict, paying_off = "inconclusive", bool(champion_advanced)
        rec = (f"The last edit ({changed_coords}) moved accuracy by only "
               f"{d_acc:+.3f} (within +/-{min_lift} noise band); treat as "
               f"inconclusive and diversify rather than doubling down.")

    return {
        "round": round_id,
        "compared_with_round": (prev_round_context or {}).get("round_id", round_id - 1),
        "prev_best": prev_best.get("config_id"),
        "curr_best": curr_best.get("config_id"),
        "last_edit": edit,
        "champion_advanced": champion_advanced,
        "objective_delta": delta,
        "verdict": verdict,
        "paying_off": paying_off,
        "min_lift": min_lift,
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Smart-7 fix (2026-05-16): embedding-based pool relevance.
# ---------------------------------------------------------------------------
#
# TF-IDF (kept below as fallback) over-includes — at the legacy 0.05 threshold,
# 80-90% of pool units register as "relevant" purely on stop-word overlap.
# This kills the signal in the decision tree (relevant_in_pool > 0 nearly
# always true). Sentence-transformers cosine similarity is semantically aware:
# "Unlambda question" vs "esoteric programming language tip" scores ~0.6 even
# without word overlap, while truly unrelated units score ~0.1.
#
# Cost: ~5-8s per candidate eval (encode 50 queries + cached unit embeddings).
# Embeddings are cached in memory by unit_id so re-evaluating an unchanged
# pool across rounds is free.

_EMBEDDER = None
_EMBED_CACHE: Dict[str, Any] = {}  # unit_id (or hash) → ndarray


def _get_embedder():
    """Lazy-load the sentence-transformers model. Single instance per process."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — embedding relevance "
                "disabled, falling back to TF-IDF for pool relevance."
            )
            return None
        # Smart-7 fix (2026-05-16): try a sequence of models; pick the first
        # that loads. `all-MiniLM-L6-v2` (384d) is small + usually pre-cached
        # on developer machines (used elsewhere in the project). `bge-base-en`
        # (768d) is bigger / better but requires a working network download.
        for model_name in (
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-base-en-v1.5",
        ):
            try:
                _EMBEDDER = SentenceTransformer(model_name)
                logger.info("[Smart-7] Loaded sentence-transformer: %s", model_name)
                break
            except Exception as e:
                logger.warning(
                    "[Smart-7] failed to load %s: %s — trying next.", model_name, e
                )
        if _EMBEDDER is None:
            logger.warning(
                "[Smart-7] no embedding model could be loaded; "
                "falling back to TF-IDF for pool relevance."
            )
    return _EMBEDDER


def _embed_texts(texts: List[str]) -> Optional[Any]:
    """Encode a list of texts to normalized embeddings (NxD ndarray).

    Returns None if embedder failed to load (caller should fall back to
    TF-IDF). Caches per-text on hash key to amortize re-runs.
    """
    embedder = _get_embedder()
    if embedder is None or not texts:
        return None

    import hashlib
    cache_keys = [hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()
                  for t in texts]
    missing_idx = [i for i, k in enumerate(cache_keys) if k not in _EMBED_CACHE]
    if missing_idx:
        to_encode = [texts[i] for i in missing_idx]
        try:
            new_embs = embedder.encode(
                to_encode, normalize_embeddings=True, show_progress_bar=False,
            )
        except Exception as e:
            logger.warning("[Smart-7] embedding encode failed (%s); falling back.", e)
            return None
        for i, emb in zip(missing_idx, new_embs):
            _EMBED_CACHE[cache_keys[i]] = emb

    import numpy as _np
    return _np.stack([_EMBED_CACHE[k] for k in cache_keys])


def _embedding_pool_scores(
    query: str, pool_texts: List[str],
) -> Optional[List[float]]:
    """Return cosine sims [N_pool] of query vs each pool unit.

    Returns None if embedder unavailable — caller falls back to TF-IDF.
    """
    if not pool_texts:
        return None
    q_emb = _embed_texts([query])
    if q_emb is None:
        return None
    p_embs = _embed_texts(pool_texts)
    if p_embs is None:
        return None
    # cosine: (q_emb [1, D]) @ (p_embs [N, D]^T) → [N]
    sims = (p_embs @ q_emb[0]).tolist()
    return sims


# ---------------------------------------------------------------------------
# TF-IDF helpers (lightweight relevance estimation)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _build_idf(corpus: List[List[str]]) -> Dict[str, float]:
    """Compute smoothed IDF for a token corpus.

    Uses the +1 additive smoothing formula so that terms appearing in every
    document still carry weight (avoids IDF=0 on small corpora).
    """
    n = len(corpus)
    df: Dict[str, int] = {}
    for doc in corpus:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1
    # Smoothed IDF: log((n+1)/(df+1)) + 1  → minimum value is 1.0
    return {tok: math.log((n + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}


def _tf_idf_score(query_tokens: List[str], doc_tokens: List[str], idf: Dict[str, float]) -> float:
    """Compute TF-IDF cosine-like similarity between query and doc."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    score = sum(idf.get(t, 0.0) for t in query_tokens if t in doc_set)
    # Normalize by query length so short queries don't dominate
    return score / (len(query_tokens) + 1e-9)


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def run_posthoc_audit(
    task_results: List[Dict[str, Any]],
    canonical_units: List[Dict[str, Any]],
    top_k: int = 5,
    relevance_threshold: float = 0.05,
    max_steps: int = 0,
) -> Tuple[List[AttributionResult], AuditSummary]:
    """Run post-hoc attribution audit.

    Parameters
    ----------
    task_results : list of task result dicts (from tasks/*.json)
    canonical_units : list of MemoryUnit.to_dict() from canonical_units.json
    top_k : retrieval top-k used during evaluation (to estimate miss_topk)
    relevance_threshold : minimum TF-IDF score to count as a pool match

    Returns
    -------
    (attributions, summary)
    """
    summary = AuditSummary(total_tasks=len(task_results))

    # Build pool text corpus for TF-IDF
    pool_texts = []
    for u in canonical_units:
        # Combine all textual fields
        parts = []
        for field_name in ("content", "summary", "key_decision", "critical_observation",
                           "tool_strategy", "principle", "description"):
            val = u.get(field_name, "")
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, dict):
                parts.extend(str(v) for v in val.values() if isinstance(v, str))
        pool_texts.append(" ".join(parts))

    pool_token_lists = [_tokenize(t) for t in pool_texts]
    idf = _build_idf(pool_token_lists) if pool_token_lists else {}

    # Smart-7 fix (2026-05-16): try to use embedding for pool relevance.
    # `pool_embedding_available` controls per-query whether we use embedding
    # cosine or fall back to TF-IDF. Cached across all queries in this run.
    pool_embedding_available = bool(pool_texts) and (_get_embedder() is not None)
    if pool_embedding_available:
        logger.info(
            "[Smart-7] Using sentence-transformer (bge-base-en-v1.5) for pool relevance — "
            "TF-IDF fallback active if embedding fails per-query."
        )

    # A1 fix (2026-05-16): cap "relevant_in_pool" via adaptive threshold so the
    # signal stays meaningful even when pool grows past a few hundred units.
    # Pre-A1: fixed 0.05 caused 80-90% of units to register as "relevant".
    # Adaptive logic: take max of (passed_threshold, median+1.5*std) so the
    # bar rises with pool quality but never drops below the user-supplied
    # ``relevance_threshold``. Per-query computation; cheap.
    #
    # Smart-7 fix (2026-05-16): when embeddings are in use, the threshold
    # interpretation changes — cosine sim has natural range ~[0, 0.9]. We use
    # 0.45 as the embedding threshold (matches BGE recommendations).
    import statistics as _stats
    EMB_DEFAULT_THRESHOLD = 0.45
    def _effective_threshold(per_query_scores: List[float],
                             using_embedding: bool = False) -> float:
        base = EMB_DEFAULT_THRESHOLD if using_embedding else relevance_threshold
        if not per_query_scores:
            return base
        nonzero = [s for s in per_query_scores if s > 0.0]
        if len(nonzero) < 5:
            return base
        try:
            med = _stats.median(nonzero)
            sd = _stats.pstdev(nonzero)
            adaptive = med + 1.5 * sd
        except _stats.StatisticsError:
            adaptive = base
        return max(base, adaptive)

    attributions: List[AttributionResult] = []

    for result in task_results:
        task_id = str(result.get("task_id", result.get("id", "unknown")))
        question = result.get("question", result.get("Question", ""))
        task_score = float(result.get("task_score", 0.0))

        mm = result.get("memory_metrics") or {}
        retrieved_count = mm.get("num_retrieved", 0)
        retrieved_pre_judge = mm.get("num_retrieved_pre_judge", 0)
        judge_dropped_all_count = mm.get("judge_dropped_all_count", 0)
        # Optional fine-grained signals (added 2026-04-27). When the provider
        # exports them, the decision tree below can split coarse categories
        # into the newer 8-class taxonomy. When absent, behaviour falls back
        # to the previous 5-class decisions, so legacy task_results still
        # parse correctly.
        avg_kept_confidence = mm.get("avg_kept_confidence")  # float in [0,1] or None
        kept_units_stale_count = mm.get("kept_units_stale_count", 0)

        if task_score >= 1.0:
            summary.success_count += 1
            attributions.append(AttributionResult(
                task_id=task_id,
                question=question,
                task_score=task_score,
                attribution=AttributionType.SUCCESS,
                evidence="Task succeeded.",
                pool_matches=0,
                retrieved_count=retrieved_count,
            ))
            continue

        # ---- Infra / API error (out-of-scope), checked BEFORE the memory tree ----
        # A task that raised an API/infrastructure error (e.g. HTTP 400 from a
        # context-length blowup) crashed before producing a real answer. It is
        # NOT a memory-architecture failure, so route it out of scope here so it
        # never pollutes the per-module blame signal that FGMD feeds to the
        # proposer. (Confirmed 2026-07-07: 17 xBench tasks with a populated
        # `error` field were being mislabeled as extraction_gap / retrieval_noise
        # / retrieval_miss_topk; GAIA runs had 0 such tasks.)
        _infra_err = result.get("error")
        if _infra_err:
            summary.infra_error += 1
            attributions.append(AttributionResult(
                task_id=task_id,
                question=question,
                task_score=task_score,
                attribution=AttributionType.INFRA_ERROR,
                evidence=(
                    "Agent run raised an infrastructure/API error before "
                    f"producing an answer: {str(_infra_err)[:140]}. OUT-OF-SCOPE "
                    "for memory architecture (context-length blowup / provider error)."
                ),
                pool_matches=0,
                retrieved_count=retrieved_count,
            ))
            continue

        # ---- Failed task: diagnose ----
        query_tokens = _tokenize(question)

        # Count how many pool units are relevant to this query.
        # Smart-7 fix (2026-05-16): prefer embedding cosine; fall back to TF-IDF
        # if the embedder failed to load or this particular query failed.
        emb_scores = (_embedding_pool_scores(question, pool_texts)
                      if pool_embedding_available else None)
        if emb_scores is not None:
            pool_scores = emb_scores
            _using_embedding = True
        else:
            pool_scores = [
                _tf_idf_score(query_tokens, pool_toks, idf)
                for pool_toks in pool_token_lists
            ]
            _using_embedding = False
        # A1 fix (2026-05-16): adaptive threshold to compensate for over-
        # inclusion at the static baseline.
        _thr = _effective_threshold(pool_scores, using_embedding=_using_embedding)
        relevant_in_pool = sum(1 for s in pool_scores if s >= _thr)

        # Determine whether retrieved memories were (likely) relevant.
        # Prefer persisted retrieved text when available. Fall back to the older
        # count-only approximation for historical task results.
        retrieved_text = (
            result.get("retrieved_memory_context")
            or result.get("retrieved_memory_text")
            or ""
        )
        # Fix 1 (2026-04-22): gate_threshold is removed; the filter is now an
        # LLM judge.  Decision tree rewritten around three distinct states
        # produced by the new pipeline:
        #   (a) retriever returned 0 candidates (pool empty on this topic, or
        #       retriever ranking didn't surface anything)  -> extraction_gap
        #                                                     or retrieval_miss_topk
        #   (b) retriever returned >0 but judge dropped all -> judge_rejected_all
        #   (c) retriever returned >0 and judge kept >0     -> noise or reasoning
        judge_dropped_all = (
            retrieved_pre_judge > 0 and retrieved_count == 0
        ) or (judge_dropped_all_count > 0 and retrieved_count == 0)

        if retrieved_count > 0 and retrieved_text:
            # Smart-7 fix (2026-05-16): use embedding for retrieved-text
            # relevance when available, same as pool_scores. Same threshold.
            if _using_embedding:
                ret_emb = _embedding_pool_scores(question, [retrieved_text])
                retrieved_score = ret_emb[0] if ret_emb else 0.0
            else:
                retrieved_score = _tf_idf_score(
                    query_tokens, _tokenize(retrieved_text), idf,
                )
            retrieved_relevant = retrieved_score >= _thr
        else:
            retrieved_relevant = (retrieved_count > 0 and relevant_in_pool > 0)

        # A3 fix (2026-05-16): answer-format checks BEFORE memory analysis.
        # These detect mis-failures where memory worked but the judge gave 0
        # due to format / precision / time-window issues. They must come first
        # because the agent's answer string already contains the signal.
        golden_answer = result.get("golden_answer", "")
        if isinstance(ar := result.get("agent_result"), dict):
            agent_answer_str = str(ar.get("final_answer", ""))
        else:
            agent_answer_str = str(ar) if ar is not None else ""

        if _detect_numeric_precision(golden_answer, agent_answer_str):
            attr = AttributionType.NUMERIC_PRECISION
            evidence = (
                f"golden={golden_answer!r} vs agent={agent_answer_str!r} — "
                "values within 5% relative; judge scored 0 but answer is numerically close."
            )
            summary.numeric_precision += 1
            attributions.append(AttributionResult(
                task_id=task_id, question=question, task_score=task_score,
                attribution=attr, evidence=evidence,
                pool_matches=relevant_in_pool, retrieved_count=retrieved_count,
            ))
            continue
        if _detect_time_range_mismatch(question, golden_answer, agent_answer_str):
            attr = AttributionType.TIME_RANGE_MISMATCH
            evidence = (
                f"question mentions time window; golden={golden_answer!r}, agent={agent_answer_str!r} "
                "— ratio ≥3× suggests agent didn't restrict to the asked time."
            )
            summary.time_range_mismatch += 1
            attributions.append(AttributionResult(
                task_id=task_id, question=question, task_score=task_score,
                attribution=attr, evidence=evidence,
                pool_matches=relevant_in_pool, retrieved_count=retrieved_count,
            ))
            continue
        if _detect_near_miss(golden_answer, agent_answer_str):
            attr = AttributionType.NEAR_MISS
            evidence = (
                f"golden={golden_answer!r} vs agent={agent_answer_str!r} — "
                "≥0.7 string similarity; likely format/morphology mismatch (e.g. 'egalitarian' vs 'Egalitarianism')."
            )
            summary.near_miss += 1
            attributions.append(AttributionResult(
                task_id=task_id, question=question, task_score=task_score,
                attribution=attr, evidence=evidence,
                pool_matches=relevant_in_pool, retrieved_count=retrieved_count,
            ))
            continue

        max_pool_relevance = max(pool_scores) if pool_scores else 0.0

        # E4 fix (2026-05-16): re-ordered so PROVIDER signals (kept_units_stale,
        # avg_kept_confidence, injection_failed_signal) fire BEFORE the broad
        # TF-IDF-based RETRIEVAL_NOISE branch. Previously the TF-IDF gate at
        # the top short-circuited MEMORY_STALE / EXTRACTION_LOW_QUALITY /
        # INJECTION_BAD whenever it happened to score the retrieved text low.
        # F2 fix (codex review): DOMAIN_KNOWLEDGE_GAP moved from before this
        # block to after RETRIEVAL_NOISE — a task with kept_units_stale_count
        # > 0 should be classified as MEMORY_STALE even if the pool happens
        # to be sparse for this query.
        # Attribution decision tree (order matters).
        if retrieved_count > 0 and retrieved_relevant and kept_units_stale_count > 0:
            # Memory existed and was retrieved but at least one kept unit had
            # conflict_count > 0 / superseded / inactive — stale memory.
            attr = AttributionType.MEMORY_STALE
            evidence = (
                f"Retrieved {retrieved_count} relevant unit(s) but "
                f"{kept_units_stale_count} of the kept units had unresolved conflicts "
                "or were superseded. Memory pool needs reconciliation."
            )
            summary.memory_stale += 1

        elif (
            retrieved_count > 0
            and retrieved_relevant
            and avg_kept_confidence is not None
            and avg_kept_confidence < 0.5
            # A2 fix (2026-05-16) + F3 fix (codex review): cross-validate against
            # domain-gap (pool must have strong on-topic units), and optionally
            # against actionable_usefulness_rate when the provider emits it.
            #   - pool_relevance ≥ 0.3 always required (excludes domain gap)
            #   - actionable_usefulness check is opt-in: when provider emits
            #     the signal, value < 0.3 is required; when signal is missing
            #     (None or 1.0 — the historical default), fall back to
            #     confidence-only behavior so the branch is still reachable.
            and (max(pool_scores) if pool_scores else 0.0) >= 0.3
            and (
                mm.get("actionable_usefulness_rate") is None
                or mm.get("actionable_usefulness_rate") == 1.0  # legacy default
                or mm.get("actionable_usefulness_rate") < 0.3
            )
        ):
            attr = AttributionType.EXTRACTION_LOW_QUALITY
            evidence = (
                f"Pool has {relevant_in_pool} relevant units (max_pool_rel="
                f"{max(pool_scores):.2f} ≥0.3); judge kept {retrieved_count} but "
                f"avg_confidence={(avg_kept_confidence or 0):.2f} <0.5 AND "
                f"actionable_usefulness={(mm.get('actionable_usefulness_rate') or 0):.2f} <0.3 — "
                "extraction produced shallow units that the agent could not act on. "
                "Tighten extract prompts or enable utility_audit pruning."
            )
            summary.extraction_low_quality += 1

        elif (
            retrieved_count > 0
            and retrieved_relevant
            and mm.get("injection_failed_signal") is True
        ):
            # INJECTION_BAD requires the provider's explicit agent-use signal;
            # task failure alone is not enough to blame fixed context injection.
            attr = AttributionType.INJECTION_BAD
            evidence = (
                f"Retrieved {retrieved_count} relevant unit(s) with cited context, "
                "but the agent did not act on them. Improve encoded actionability "
                "or retrieval relevance; the runtime composer is fixed."
            )
            summary.injection_bad += 1

        elif _detect_budget_capped(result, max_steps) and max_steps > 0:
            # ── 2026-05-10: agent hit (or got within 5% of) max_steps before
            # producing a real answer. NOT a memory architecture failure;
            # proposer should NOT mutate retrieval/management to "fix" this.
            attr = AttributionType.BUDGET_CAPPED
            traj_len = len(result.get("agent_trajectory") or [])
            evidence = (
                f"Agent trajectory reached {traj_len} steps (cap={max_steps}); "
                "likely truncated before reaching a real answer. Memory "
                "architecture is OUT-OF-SCOPE — increase --max_steps to fix, "
                "or accept this task as agent-budget-bound."
            )
            summary.budget_capped += 1

        elif _detect_tool_failure(result):
            # ── 2026-05-10: agent tried ≥3 web searches but answer admits
            # "unable to find / no information / 404 / blog deleted" etc. Pure
            # tool/data-availability failure; not a memory issue.
            attr = AttributionType.TOOL_FAILURE
            evidence = (
                "Agent attempted multiple web searches but final answer admits "
                "the information is not available (deleted page, dead link, "
                "or absent from indexed sources). Memory architecture is "
                "OUT-OF-SCOPE — proposer should NOT mutate retrieval to fix."
            )
            summary.tool_failure += 1

        elif _detect_multimodal_failure(result):
            # ── 2026-05-10: task has image/audio/video attachment AND agent
            # admitted it could not extract the multimodal content. This is a
            # perception-layer failure — memory cannot help.
            attr = AttributionType.MULTIMODAL_FAILURE
            evidence = (
                "Task has multimodal attachment (image/audio/video) but agent "
                "admitted it could not view/process the content. Memory "
                "architecture is OUT-OF-SCOPE — perception-layer issue."
            )
            summary.multimodal_failure += 1

        elif retrieved_count > 0 and not retrieved_relevant:
            # E4 fix (2026-05-16): moved from top to AFTER provider-signal
            # branches. RETRIEVAL_NOISE = judge let candidates through but
            # TF-IDF says they're off-topic.
            attr = AttributionType.RETRIEVAL_NOISE
            evidence = (
                f"Judge passed {retrieved_count} unit(s) but TF-IDF says none are relevant "
                f"(pool has {relevant_in_pool} relevant). Judge + retriever relevance criteria "
                "diverge; loosen TF-IDF threshold or tighten judge prompt."
            )
            summary.retrieval_noise += 1

        elif max_pool_relevance < 0.10:
            # F2 fix (codex review, 2026-05-16): DOMAIN_KNOWLEDGE_GAP — pool has
            # no on-topic unit. Placed AFTER provider-signal branches so that
            # a stale-memory or low-quality-extraction signal (which depends on
            # provider metadata, not pool relevance) is not shadowed.
            attr = AttributionType.DOMAIN_KNOWLEDGE_GAP
            evidence = (
                f"Max pool TF-IDF relevance={max_pool_relevance:.3f} (<0.10) — "
                "no unit in the pool covers this query's topic at all. "
                "Out-of-domain task; broaden warm-up sourcing rather than tweaking extract prompts."
            )
            summary.domain_knowledge_gap += 1

        elif retrieved_count > 0 and retrieved_relevant:
            # Judge passed a relevant candidate and agent still failed → reasoning
            attr = AttributionType.REASONING_ERROR
            evidence = (
                f"Pool has {relevant_in_pool} relevant units; "
                f"retrieved & judge-kept {retrieved_count} relevant unit(s) but task still failed — "
                "memory is not the bottleneck."
            )
            summary.reasoning_error += 1

        elif judge_dropped_all and relevant_in_pool > 0:
            # Retriever DID surface candidates but the LLM judge dropped them all.
            # This is the NEW diagnostic category (Fix 1 replaced gate with judge).
            attr = AttributionType.JUDGE_REJECTED_ALL
            evidence = (
                f"Pool has {relevant_in_pool} relevant units and retriever surfaced "
                f"{retrieved_pre_judge} candidate(s), but LLM judge dropped ALL of them — "
                "judge is over-strict OR retriever surfaced candidates that look relevant "
                "by TF-IDF but not by semantic judge. Consider: (1) tighten retriever quality "
                "so it returns fewer false positives, (2) expand pool with in-domain memories, "
                "(3) loosen judge prompt. NOT a gate_threshold issue (deprecated)."
            )
            summary.judge_rejected_all += 1

        elif judge_dropped_all and relevant_in_pool == 0:
            # Judge dropped all AND pool has nothing genuinely relevant -> extraction gap.
            # The retriever found *something* but it was noise; judge correctly filtered.
            attr = AttributionType.EXTRACTION_GAP
            evidence = (
                f"Pool has 0 TF-IDF-relevant units for this query; retriever surfaced "
                f"{retrieved_pre_judge} noisy candidate(s) and LLM judge correctly rejected them. "
                f"Pool lacks memories for this task family — need broader extraction."
            )
            summary.extraction_gap += 1

        elif relevant_in_pool == 0:
            # Pool has nothing and retriever found nothing.
            attr = AttributionType.EXTRACTION_GAP
            evidence = f"Pool has 0 units matching query (TF-IDF threshold={relevance_threshold}); retriever returned 0."
            summary.extraction_gap += 1

        else:
            # Pool has relevant units, retriever returned 0 → ranking / top-k miss
            attr = AttributionType.RETRIEVAL_MISS_TOPK
            evidence = (
                f"Pool has {relevant_in_pool} TF-IDF-relevant units; "
                f"retriever returned 0 candidates — retriever ranking missed them "
                "(retr_strategy / top_k / embedding mismatch)."
            )
            summary.retrieval_miss_topk += 1

        attributions.append(AttributionResult(
            task_id=task_id,
            question=question,
            task_score=task_score,
            attribution=attr,
            evidence=evidence,
            pool_matches=relevant_in_pool,
            retrieved_count=retrieved_count,
        ))

    return attributions, summary


def save_attribution_report(
    attributions: List[AttributionResult],
    summary: AuditSummary,
    output_path: str,
    reasoning_subclasses: Optional[Dict[str, Dict[str, str]]] = None,
    layer_diag: Optional[Dict[str, Any]] = None,
    memory_compliance: Optional[Dict[str, Any]] = None,
    evidence_by_category: Optional[Dict[str, Any]] = None,
    synthesized_verdict: Optional[Dict[str, Any]] = None,
) -> None:
    """Save attribution results to JSON.

    B1 fix (2026-05-16): now accepts ``layer_diag`` so the LLM-produced
    layer diagnosis is persisted alongside the rule-based summary. Previously
    only rule-based stats were saved and the ~5-10s/candidate LLM call output
    was lost on disk (it was only kept in memory inside ParetoEntry).

    Smart-8 fix (2026-05-16): also accepts aggregated ``memory_compliance``
    (per-candidate stats over failed tasks of how well the agent followed
    retrieved-unit instructions across 6 dimensions).
    """
    data = {
        "summary": summary.to_dict(),
        "per_task": [
            {
                "task_id": a.task_id,
                "question": a.question[:200],
                "task_score": a.task_score,
                "attribution": a.attribution.value,
                "evidence": a.evidence,
                "pool_matches": a.pool_matches,
                "retrieved_count": a.retrieved_count,
                # I5 fix (codex review, 2026-05-17): preserve the subclass
                # even when S10 has reclassified the task out of REASONING_ERROR.
                # Otherwise H4's reasoning_subclass_examples would systematically
                # miss the most actionable subclasses (unit_or_format /
                # numeric / time) that S10 promotes to A3 top categories.
                "reasoning_subclass":
                    (reasoning_subclasses or {}).get(a.task_id, {}).get("subclass"),
                "reasoning_subclass_evidence":
                    (reasoning_subclasses or {}).get(a.task_id, {}).get("evidence"),
            }
            for a in attributions
        ],
    }
    if reasoning_subclasses:
        # Aggregate counts so cumulative_principles can quote them.
        # E3 fix (2026-05-16): skip "__meta__" sentinel key — it has no
        # 'subclass' field, so the previous code silently counted it as
        # +1 true_reasoning_error.
        sub_counts: Dict[str, int] = {}
        for tid, info in reasoning_subclasses.items():
            if tid == "__meta__":
                continue
            sub = info.get("subclass") or "true_reasoning_error"
            sub_counts[sub] = sub_counts.get(sub, 0) + 1
        data["summary"]["reasoning_error_subclasses"] = sub_counts

    # B1 fix (2026-05-16): persist LLM 4-layer diagnosis if provided.
    if layer_diag:
        data["summary"]["layer_diagnosis"] = layer_diag

    # Smart-8 fix (2026-05-16): persist memory compliance summary if provided.
    if memory_compliance:
        data["summary"]["memory_compliance"] = memory_compliance

    # Smart-12 (2026-05-17): persist evidence_by_category so the ledger LLM
    # and post-hoc audit can see which 3 worst-task examples drove the
    # diagnosis.
    if evidence_by_category:
        data["summary"]["evidence_by_category"] = evidence_by_category

    # Smart-9a (2026-05-17): persist synthesized natural-language verdict.
    # The proposer reads this as the PRIMARY signal in §2A.
    if synthesized_verdict:
        data["summary"]["synthesized_verdict"] = synthesized_verdict

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Attribution report saved to %s", output_path)


# ---------------------------------------------------------------------------
# LLM-based deep sub-classification of REASONING_ERROR (H plan, 2026-04-28)
# ---------------------------------------------------------------------------
# Rationale: rule-based audit lumps ~79% of failures into REASONING_ERROR
# because once retrieval and judge both pass, no rule can tell why the agent
# still failed. Calling the diagnosis LLM on this subset (cheap, ~6-10
# extra calls per round) recovers fine-grained signal for the architect.

_REASONING_SUBCLASS_TAXONOMY = (
    "multimodal_extraction_error",   # video / audio / image misread
    "numeric_calculation_error",      # arithmetic / averaging error
    "unit_or_format_mismatch",        # value correct, unit/format off
    "entity_disambiguation_error",    # picked wrong same-named entity
    "time_range_misinterpretation",   # A5 fix (2026-05-16): didn't restrict to asked time window
    "question_constraint_missed",     # A5 fix: ignored a hard constraint
    "scope_misinterpretation",        # last resort — answered a different question (no narrower label)
    "multi_hop_reasoning_collapse",   # multi-step chain broke mid-way
    "true_reasoning_error",           # genuinely unclassifiable
)

_REASONING_SUBCLASS_PROMPT = """You are a failure-mode classifier for LLM agent task runs.

A task FAILED but rule-based audit cannot pinpoint why — retrieval and judge both passed.
Choose ONE sub-class from the taxonomy below.

Taxonomy:
  multimodal_extraction_error : video/audio/image content was misread (only when task has such files)
  numeric_calculation_error   : agent's number is wrong despite correct facts (math/averaging error)
  unit_or_format_mismatch     : value/entity correct but unit / format off (e.g. "17 thousand" vs "17000")
  entity_disambiguation_error : agent picked wrong same-named entity (different person / film / paper)
  time_range_misinterpretation: agent didn't restrict to the asked time window (e.g. "in 2019" → all-time)
  question_constraint_missed  : agent ignored a hard constraint (e.g. "exclude X", "first added", "exact form")
  scope_misinterpretation     : agent answered a different question than asked (use only when no narrower label fits)
  multi_hop_reasoning_collapse: multi-step chain broke mid-way (used file but lost track of question)
  true_reasoning_error        : none of the above; genuine LLM reasoning failure

Few-shot examples (A5 fix, 2026-05-16: differentiating between formerly-overloaded scope_misinterpretation):

Example 1 (time_range_misinterpretation):
  Question: "How many studio albums were published by Mercedes Sosa between 2000 and 2009?"
  Golden: "3"
  Agent: "11"  ← agent counted all albums in her career, not 2000-2009
  → {"subclass": "time_range_misinterpretation", "evidence": "counted all-time albums (11) instead of 2000-2009"}

Example 2 (question_constraint_missed):
  Question: "What is the latest predictor base command that received a bug fix (just the name)?"
  Golden: "BaseLabelPropagation"
  Agent: "MultiOutputClassifier"  ← ignored "latest" constraint, picked an earlier one
  → {"subclass": "question_constraint_missed", "evidence": "ignored 'latest' constraint; picked earlier predictor"}

Example 3 (entity_disambiguation_error):
  Question: "Who directed Inception (the 2010 sci-fi)?"
  Golden: "Christopher Nolan"
  Agent: "Edward Zwick"  ← confused two same-named films
  → {"subclass": "entity_disambiguation_error", "evidence": "confused 2010 Inception with another film"}

Example 4 (unit_or_format_mismatch):
  Question: "What is the average rating?"
  Golden: "0.0424"
  Agent: "4.24%"  ← correct value, wrong format
  → {"subclass": "unit_or_format_mismatch", "evidence": "expressed as percent instead of decimal"}

Task data:
  Question: {{ question }}
  Files in task: {{ task_files }}
  Golden answer: {{ golden }}
  Agent answered: {{ agent_answer }}
  Trajectory length: {{ n_steps }} steps
  Trajectory summary (last 3 steps):
{{ trajectory_tail }}

Output JSON only (no markdown):
{"subclass": "<one of the 9 keys>", "evidence": "<≤15 words from trajectory or answer>"}"""


def _format_trajectory_tail(trajectory: Any, n: int = 3) -> str:
    """Return a compact rendering of the last N trajectory steps."""
    if not isinstance(trajectory, list) or not trajectory:
        return "  (no trajectory)"
    tail = trajectory[-n:]
    lines = []
    for s in tail:
        if isinstance(s, dict):
            content = s.get("content") or s.get("action") or str(s)
            lines.append(f"  - {str(content)[:140]}")
        else:
            lines.append(f"  - {str(s)[:140]}")
    return "\n".join(lines)


def llm_subclassify_reasoning_errors(
    task_results: List[Dict[str, Any]],
    rule_attributions: List[AttributionResult],
    model,
    max_calls: int = 12,
    sample_seed: int = 0,
) -> Dict[str, Dict[str, str]]:
    """Sub-classify all REASONING_ERROR-tagged tasks via LLM.

    Returns: dict task_id -> {"subclass": ..., "evidence": ...}.
    Only fails individual tasks soft (returns "true_reasoning_error" with
    parse-error evidence) so a flaky LLM call never breaks the search loop.

    Codex CR2-13: bounded by ``max_calls`` to prevent unbounded LLM cost on
    reasoning-heavy rounds. With batch_size=50 + 3 candidates a single round
    can produce 150 reasoning errors; without the cap this dominates run cost
    and may trip provider rate limits. When more targets exist than max_calls,
    a deterministic random sample (seeded by sample_seed) is taken so the
    distribution is unbiased and reproducible.
    """
    import random as _random
    targets = [
        (a, r) for a, r in zip(rule_attributions, task_results)
        if a.attribution == AttributionType.REASONING_ERROR
    ]
    if not targets:
        logger.info("llm_subclassify: no REASONING_ERROR tasks; skipping LLM calls")
        return {}

    n_total_reasoning = len(targets)
    n_sampled = n_total_reasoning
    if n_total_reasoning > max_calls:
        rng = _random.Random(sample_seed + n_total_reasoning)
        sampled = rng.sample(targets, max_calls)
        logger.info(
            "llm_subclassify: %d REASONING_ERROR tasks > max_calls=%d; "
            "sampling %d via seed=%d",
            n_total_reasoning, max_calls, max_calls, sample_seed + n_total_reasoning,
        )
        targets = sampled
        n_sampled = max_calls

    logger.info(
        "llm_subclassify: calling diagnosis_model on %d REASONING_ERROR tasks "
        "(cap=%d)",
        len(targets), max_calls,
    )

    # Codex Round 3 R3-7: track sampling so the architect knows when the
    # distribution is estimated. We attach this to a sentinel key in the
    # output that downstream code can detect without colliding with task_id.
    out: Dict[str, Dict[str, str]] = {
        "__meta__": {
            "n_total_reasoning": str(n_total_reasoning),
            "n_sampled": str(n_sampled),
            "is_sampled": "true" if n_sampled < n_total_reasoning else "false",
        }
    }
    for attr, result in targets:
        # Pull task metadata without assuming key shape.
        question = result.get("question", attr.question)
        golden = result.get("golden_answer", "")
        agent_answer = str(result.get("agent_result", ""))[:200]
        traj = result.get("agent_trajectory") or result.get("trajectory") or []
        task_files = result.get("task_files") or []
        if not task_files:
            # Fall back to deriving from file_name if present
            fn = result.get("file_name") or ""
            if isinstance(fn, str) and "." in fn:
                task_files = [f"file_{fn.rsplit('.', 1)[-1].lower()}"]

        # E1 fix (2026-05-16): use render_prompt (Jinja2) instead of str.format()
        # to avoid KeyError when agent_answer or trajectory_tail contains literal
        # '{' (e.g. JSON-shaped agent outputs).
        from ..llm_utils import render_prompt
        prompt = render_prompt(_REASONING_SUBCLASS_PROMPT, {
            "question": str(question)[:240],
            "task_files": task_files,
            "golden": str(golden)[:120],
            "agent_answer": agent_answer,
            "n_steps": len(traj) if isinstance(traj, list) else "?",
            "trajectory_tail": _format_trajectory_tail(traj),
        })

        try:
            response = model([
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            parsed = _parse_subclass_response(raw)
        except Exception as e:
            logger.warning(
                "llm_subclassify failed on task %s: %s",
                attr.task_id, e,
            )
            parsed = {"subclass": "true_reasoning_error",
                      "evidence": f"llm_call_error: {type(e).__name__}"}

        out[attr.task_id] = parsed
    return out


def _parse_subclass_response(raw: str) -> Dict[str, str]:
    """Parse LLM JSON output; fall back to true_reasoning_error on garbage."""
    import re as _re
    cleaned = raw.strip()
    m = _re.search(r"```(?:json)?\s*([\s\S]+?)```", cleaned)
    if m:
        cleaned = m.group(1).strip()
    if not cleaned.startswith("{"):
        m = _re.search(r"\{[\s\S]+\}", cleaned)
        if m:
            cleaned = m.group(0)
    try:
        d = json.loads(cleaned)
        sub = str(d.get("subclass", "")).strip()
        if sub not in _REASONING_SUBCLASS_TAXONOMY:
            sub = "true_reasoning_error"
        evidence = str(d.get("evidence", ""))[:200]
        return {"subclass": sub, "evidence": evidence}
    except Exception:
        return {"subclass": "true_reasoning_error",
                "evidence": "parse_failed"}
