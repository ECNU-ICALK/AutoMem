"""
Build an EvaluationReport from per-task JSON result files produced by
the evaluation scripts (run_flash_searcher_mm_gaia.py, etc.).

Each task JSON is expected to carry:
  - task_score, status, level, question, golden_answer, agent_result
  - metrics.{total_tokens, prompt_tokens, completion_tokens, api_calls, elapsed_time}
  - memory_metrics  (dict from provider.get_experiment_metrics())
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from automem.contracts import (
    CostSummary,
    EvaluationReport,
    FailureCase,
    MemoryUsageSummary,
    RetrievalTraceSummary,
)
from automem.evaluation.utils import task_result_validation_error


# ------------------------------------------------------------------ #
# 1. Load task results
# ------------------------------------------------------------------ #

def load_task_results(tasks_dir: str) -> List[Dict[str, Any]]:
    """Load an exact set of strictly validated per-task JSON checkpoints."""
    results: List[Dict[str, Any]] = []
    invalid: List[str] = []
    tasks_path = Path(tasks_dir)
    if not tasks_path.is_dir():
        return results
    SKIP_FILES = {"extract_plan.json"}
    for fpath in sorted(tasks_path.glob("*.json")):
        if fpath.name in SKIP_FILES:
            continue
        try:
            if fpath.is_symlink() or not fpath.is_file():
                raise ValueError("task result must be a regular non-symlink file")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            validation_error = task_result_validation_error(data)
            if validation_error is not None:
                raise ValueError(validation_error)
            if fpath.name != f"{data['item_index']}.json":
                raise ValueError("task result filename does not match item_index")
            results.append(data)
        except (json.JSONDecodeError, OSError, TypeError, ValueError, KeyError) as exc:
            invalid.append(f"{fpath.name}: {exc}")
    if invalid:
        raise ValueError("Invalid task-result checkpoint(s): " + "; ".join(invalid))
    return results


# ------------------------------------------------------------------ #
# 2. Cost summary
# ------------------------------------------------------------------ #

def build_cost_summary(results: List[Dict]) -> CostSummary:
    """Sum token / call metrics across all task results.

    Task-model metrics already include fixed runtime calls when the same model
    is reused. Usage from an explicitly separate runtime client is added from
    ``memory_metrics``; older auxiliary counters remain readable for migration.
    """
    total_input = 0
    total_output = 0
    total_calls = 0

    for r in results:
        m = r.get("metrics", {})
        total_input += m.get("prompt_tokens", 0)
        total_output += m.get("completion_tokens", 0)
        total_calls += m.get("api_calls", 0)

        # Prefer per-task memory metrics and fall back field-by-field to old
        # top-level result files. Never sum both copies of the same counter.
        mm = r.get("memory_metrics") or {}

        def auxiliary_value(key: str) -> int:
            return int(mm.get(key, r.get(key, 0)) or 0)

        has_runtime_usage = any(
            key in mm or key in r
            for key in ("runtime_input_tokens", "runtime_output_tokens", "runtime_api_calls")
        )
        if has_runtime_usage:
            already_counted = bool(
                mm.get(
                    "runtime_usage_in_task_metrics",
                    r.get("runtime_usage_in_task_metrics", False),
                )
            )
            if not already_counted:
                total_input += auxiliary_value("runtime_input_tokens")
                total_output += auxiliary_value("runtime_output_tokens")
                total_calls += auxiliary_value("runtime_api_calls")
        else:
            # Compatibility with result files produced before automem-runtime-v1.
            total_input += auxiliary_value("judge_input_tokens")
            total_output += auxiliary_value("judge_output_tokens")
            total_calls += auxiliary_value("judge_api_calls")

        for k_in, k_out, k_call in (
            ("compress_input_tokens", "compress_output_tokens", "compression_calls"),
            ("strategy_input_tokens", "strategy_output_tokens", "strategy_api_calls"),
        ):
            total_input += auxiliary_value(k_in)
            total_output += auxiliary_value(k_out)
            total_calls += auxiliary_value(k_call)

        for k_in, k_out, k_call, included_flag in (
            (
                "rerank_input_tokens",
                "rerank_output_tokens",
                "rerank_calls",
                "rerank_usage_in_task_metrics",
            ),
            (
                "llm_graph_input_tokens",
                "llm_graph_output_tokens",
                "llm_graph_calls",
                "llm_graph_usage_in_task_metrics",
            ),
        ):
            if not bool(mm.get(included_flag, r.get(included_flag, False))):
                total_input += auxiliary_value(k_in)
                total_output += auxiliary_value(k_out)
                total_calls += auxiliary_value(k_call)

    return CostSummary(
        total_llm_calls=total_calls,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


# ------------------------------------------------------------------ #
# 3. Retrieval trace summary
# ------------------------------------------------------------------ #

def build_retrieval_trace_summary(results: List[Dict]) -> RetrievalTraceSummary:
    """Aggregate retrieval statistics from ``memory_metrics``.

    E8 fix (2026-05-16): empty_retrieval_rate is now strictly per-task, not
    per-query. Previously `empty_count` was incremented once per task while
    `total_queries` accumulated across multiple retrieval calls — so a task
    that made N calls and got nothing useful contributed empty=1/N rather
    than 1/1, *under-reporting* the empty signal. Architectures whose judge
    is most aggressively pruning candidates were silently rewarded.

    Per-task semantics: a task contributes `empty_count += 1` if it received
    no useful memory (no retrieval attempted, retriever returned 0, or judge
    dropped all candidates) and `total_queries += 1` regardless. So
    empty_retrieval_rate is exactly "fraction of tasks that saw no useful
    memory". Per-call retrieval count is still tracked separately as
    `avg_memories_per_query`.
    """
    total_queries = 0          # now: number of tasks (one bucket per task)
    empty_count = 0            # number of tasks with no useful memory
    total_retrieved = 0
    total_call_count = 0       # number of actual retriever calls (for avg)
    type_dist: Counter = Counter()

    for r in results:
        mm = r.get("memory_metrics") or {}
        calls = mm.get("retrieval_calls", 0)
        retrieved = mm.get("num_retrieved", 0)
        # Codex Q5-2 fix (2026-04-28): also count tasks where the provider
        # short-circuited BEFORE incrementing retrieval_calls (e.g.
        # primary store empty AND no secondary stores). The old
        # `calls > 0 AND retrieved == 0` condition treated those as
        # "no retrieval attempted" instead of empty, hiding the empty
        # signal from hit_rate. We also count tasks with explicit
        # `judge_dropped_all_count > 0` as empty retrievals (the judge
        # rejected everything → agent saw no memory).
        #
        # Codex Q6-A1 fix (2026-04-28): increment total_queries by AT LEAST 1
        # for the short-circuit case too. Previously only `calls` was added,
        # so an architecture that never imports any units gets total_queries=0
        # and `empty_retrieval_rate = empty / total = 0/0 → 0`, making the
        # empty-pool architecture appear to have a perfect hit rate. Use
        # `effective_calls = max(calls, 1 if attempted/short_circuit else 0)`.
        total_retrieved += retrieved
        total_call_count += max(calls, 0)
        judge_dropped_all = int(mm.get("judge_dropped_all_count", 0) or 0)
        had_attempt = (calls > 0) or (judge_dropped_all > 0)

        # E8 fix: empty_count/total_queries is now strictly per-task.
        total_queries += 1
        task_saw_no_useful_memory = (
            (had_attempt and retrieved == 0)
            or (not had_attempt)
            or (judge_dropped_all > 0 and retrieved == 0)
        )
        if task_saw_no_useful_memory:
            empty_count += 1

        # Accumulate type distribution if present
        for tname, cnt in (mm.get("type_distribution") or {}).items():
            type_dist[tname] += cnt

    # avg_memories_per_query is still per-call (informational only)
    avg_per_query = (total_retrieved / total_call_count) if total_call_count > 0 else 0.0

    return RetrievalTraceSummary(
        total_queries=total_queries,
        empty_retrieval_count=empty_count,
        avg_memories_per_query=avg_per_query,
        type_distribution=dict(type_dist),
    )


# ------------------------------------------------------------------ #
# 4. Memory usage summary
# ------------------------------------------------------------------ #

def build_memory_usage_summary(results: List[Dict]) -> MemoryUsageSummary:
    """Build memory-usage summary from per-task results.

    Snapshot fields (total_memories, entities, relations, type_counts) come
    from the **last** task (final store state).  Cumulative fields
    (total_extracted, total_inserted, total_management_ops) are summed across
    all tasks.
    """
    if not results:
        return MemoryUsageSummary()

    # Snapshot from last task
    last_mm = results[-1].get("memory_metrics") or {}

    # Cumulative counters across all tasks
    total_extracted = 0
    total_inserted = 0
    total_mgmt_ops = 0
    for r in results:
        mm = r.get("memory_metrics") or {}
        total_extracted += mm.get("num_extracted", 0)
        total_inserted += mm.get("num_inserted", 0)
        total_mgmt_ops += mm.get("management_ops_triggered", 0)

    return MemoryUsageSummary(
        total_memories=last_mm.get("num_memory_units", 0),
        total_entities=last_mm.get("graph_nodes") or 0,
        total_relations=last_mm.get("graph_edges") or 0,
        type_counts=last_mm.get("type_counts", {}),
        total_extracted=total_extracted,
        total_inserted=total_inserted,
        total_management_ops=total_mgmt_ops,
    )


# ------------------------------------------------------------------ #
# 5. Failure cases
# ------------------------------------------------------------------ #

def _categorize_failure(r: Dict) -> str:
    """Heuristic failure categorization for a single incorrect task."""
    if r.get("status") == "error":
        return "tool_error"
    mm = r.get("memory_metrics") or {}
    num_retrieved = mm.get("num_retrieved", 0)
    if num_retrieved == 0:
        return "extraction_gap"
    # Memory was retrieved but the answer is still wrong.
    return "retrieval_miss"


def build_failure_cases(
    results: List[Dict], max_cases: int = 10
) -> List[FailureCase]:
    """Extract up to *max_cases* failure diagnostics."""
    cases: List[FailureCase] = []
    for r in results:
        if r.get("task_score", 0.0) >= 1.0:
            continue

        category = _categorize_failure(r)

        # If memory was retrieved but answer is wrong and no tool error
        # and there *was* retrieval, it might actually be a reasoning error.
        mm = r.get("memory_metrics") or {}
        if category == "retrieval_miss" and mm.get("num_retrieved", 0) > 0:
            # Could be reasoning; keep retrieval_miss as the heuristic label
            # unless retrieval_calls == 0 (no retrieval attempted).
            pass
        elif category not in ("tool_error", "extraction_gap", "retrieval_miss"):
            category = "reasoning_error"

        cases.append(
            FailureCase(
                task_id=r.get("task_id", ""),
                task_query=r.get("question", ""),
                expected_answer=r.get("golden_answer", ""),
                agent_answer=str(r.get("agent_result", "")),
                failure_category=category,
            )
        )
        if len(cases) >= max_cases:
            break

    return cases


# ------------------------------------------------------------------ #
# 6. Master builder
# ------------------------------------------------------------------ #

def build_evaluation_report(
    tasks_dir: str,
    architecture_decision_dict: Dict[str, Any],
    round_id: int,
    benchmark_name: str = "GAIA",
) -> EvaluationReport:
    """Build a complete :class:`EvaluationReport` from per-task JSONs."""
    results = load_task_results(tasks_dir)

    n_tasks = len(results)
    # E5 fix (2026-05-16): expose mean_score / n_partial alongside the strict
    # binary accuracy. The binary version stays for GAIA leaderboard
    # comparability; the continuous version is used by the proposer to
    # distinguish "near-miss" from "completely wrong" failures.
    task_scores = [float(r.get("task_score", 0.0) or 0.0) for r in results]
    n_correct = sum(1 for s in task_scores if s >= 1.0)
    n_partial = sum(1 for s in task_scores if 0.0 < s < 1.0)
    mean_score = (sum(task_scores) / n_tasks) if n_tasks > 0 else 0.0
    accuracy = n_correct / n_tasks if n_tasks > 0 else 0.0

    # Per-level breakdown
    by_level: Dict[str, Dict[str, int]] = {}
    for r in results:
        level = str(r.get("level", "unknown"))
        bucket = by_level.setdefault(level, {"total": 0, "correct": 0, "partial": 0})
        bucket["total"] += 1
        ts = float(r.get("task_score", 0.0) or 0.0)
        if ts >= 1.0:
            bucket["correct"] += 1
        elif ts > 0.0:
            bucket["partial"] += 1

    # Codex Q5-1 fix (2026-04-28): aggregate avg_time_per_task so the
    # FitnessWeights.latency_penalty actually has data to subtract.
    # Previously avg_time_per_task was missing → normalized_latency=0 →
    # latency penalty silently never applied → slow architectures (graph,
    # helper-model architectures tied with fast ones at the same accuracy.
    elapsed_times = []
    for r in results:
        m = r.get("metrics", {})
        t = m.get("elapsed_time")
        if t is None:
            t = r.get("elapsed_time")
        if isinstance(t, (int, float)) and t > 0:
            elapsed_times.append(float(t))
    avg_time_per_task = (
        sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
    )

    score_summary: Dict[str, Any] = {
        "accuracy": accuracy,           # strict binary (n_correct / n_tasks)
        "n_tasks": n_tasks,
        "n_correct": n_correct,
        "n_partial": n_partial,          # E5 fix (2026-05-16): partial-credit count
        "mean_score": mean_score,        # E5 fix: continuous accuracy
        "by_level": by_level,
        "avg_time_per_task": avg_time_per_task,
    }

    return EvaluationReport(
        round_id=round_id,
        benchmark_name=benchmark_name,
        architecture_decision=architecture_decision_dict,
        score_summary=score_summary,
        cost_summary=build_cost_summary(results),
        retrieval_trace_summary=build_retrieval_trace_summary(results),
        memory_usage_summary=build_memory_usage_summary(results),
        failure_cases=build_failure_cases(results),
    )
