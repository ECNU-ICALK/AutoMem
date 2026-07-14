"""Regression tests for auxiliary model cost accounting."""

from automem.evaluation.aggregation import build_cost_summary


def test_composer_candidates_are_not_counted_as_api_calls():
    summary = build_cost_summary(
        [
            {
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "api_calls": 1,
                },
                "memory_metrics": {
                    "judge_input_tokens": 40,
                    "judge_output_tokens": 10,
                    "judge_api_calls": 1,
                    "judge_decisions_total": 4,
                },
            }
        ]
    )

    assert summary.total_input_tokens == 140
    assert summary.total_output_tokens == 30
    assert summary.total_llm_calls == 2


def test_shared_task_runtime_usage_is_not_counted_twice():
    summary = build_cost_summary(
        [
            {
                "metrics": {
                    "prompt_tokens": 160,
                    "completion_tokens": 35,
                    "api_calls": 3,
                },
                "memory_metrics": {
                    "runtime_input_tokens": 60,
                    "runtime_output_tokens": 15,
                    "runtime_api_calls": 2,
                    "runtime_usage_in_task_metrics": True,
                    # Composer aliases must not be billed again either.
                    "judge_input_tokens": 40,
                    "judge_output_tokens": 10,
                    "judge_api_calls": 1,
                },
            }
        ]
    )

    assert summary.total_input_tokens == 160
    assert summary.total_output_tokens == 35
    assert summary.total_llm_calls == 3


def test_external_runtime_client_usage_is_added_once_with_planner():
    summary = build_cost_summary(
        [
            {
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "api_calls": 1,
                },
                "memory_metrics": {
                    "runtime_input_tokens": 60,
                    "runtime_output_tokens": 15,
                    "runtime_api_calls": 2,
                    "runtime_usage_in_task_metrics": False,
                    "judge_input_tokens": 40,
                    "judge_output_tokens": 10,
                    "judge_api_calls": 1,
                },
            }
        ]
    )

    assert summary.total_input_tokens == 160
    assert summary.total_output_tokens == 35
    assert summary.total_llm_calls == 3


def test_auxiliary_top_level_alias_does_not_duplicate_memory_metrics():
    result = {
        "metrics": {"prompt_tokens": 10, "completion_tokens": 2, "api_calls": 1},
        "memory_metrics": {
            "judge_input_tokens": 5,
            "judge_output_tokens": 1,
            "judge_api_calls": 1,
        },
        "judge_input_tokens": 5,
        "judge_output_tokens": 1,
        "judge_api_calls": 1,
    }

    summary = build_cost_summary([result])
    assert (summary.total_input_tokens, summary.total_output_tokens) == (15, 3)
    assert summary.total_llm_calls == 2


def test_shared_reranker_and_graph_usage_are_not_counted_twice():
    summary = build_cost_summary(
        [
            {
                "metrics": {
                    "prompt_tokens": 150,
                    "completion_tokens": 30,
                    "api_calls": 4,
                },
                "memory_metrics": {
                    "rerank_input_tokens": 20,
                    "rerank_output_tokens": 4,
                    "rerank_calls": 2,
                    "rerank_usage_in_task_metrics": True,
                    "llm_graph_input_tokens": 30,
                    "llm_graph_output_tokens": 6,
                    "llm_graph_calls": 1,
                    "llm_graph_usage_in_task_metrics": True,
                },
            }
        ]
    )

    assert summary.total_input_tokens == 150
    assert summary.total_output_tokens == 30
    assert summary.total_llm_calls == 4


def test_external_reranker_and_graph_usage_are_added_once():
    summary = build_cost_summary(
        [
            {
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "api_calls": 1,
                },
                "memory_metrics": {
                    "rerank_input_tokens": 20,
                    "rerank_output_tokens": 4,
                    "rerank_calls": 2,
                    "rerank_usage_in_task_metrics": False,
                    "llm_graph_input_tokens": 30,
                    "llm_graph_output_tokens": 6,
                    "llm_graph_calls": 1,
                    "llm_graph_usage_in_task_metrics": False,
                },
            }
        ]
    )

    assert summary.total_input_tokens == 150
    assert summary.total_output_tokens == 30
    assert summary.total_llm_calls == 4
