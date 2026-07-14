"""Behavioral contracts for AutoMem's single fixed runtime policy."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from automem.memory_types import MemoryStatus
from automem.runtime import (
    DEFAULT_RUNTIME_POLICY,
    InjectionSessionRegistry,
    MemoryContextComposer,
    QueryPlanner,
)


class _Completions:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


class _Client:
    def __init__(self, *contents: str):
        self.chat = SimpleNamespace(completions=_Completions(list(contents)))


class _CallableModel:
    def __init__(self, *contents: str):
        self.contents = list(contents)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content=self.contents.pop(0), raw=None)


def test_runtime_policy_has_stable_identity_and_digest() -> None:
    assert DEFAULT_RUNTIME_POLICY.policy_id == "automem-runtime-v1"
    assert len(DEFAULT_RUNTIME_POLICY.digest) == 64
    assert DEFAULT_RUNTIME_POLICY.digest == DEFAULT_RUNTIME_POLICY.digest


def test_query_planner_preserves_literal_and_only_supplements_semantics() -> None:
    client = _Client('{"abstract_query":"find a reusable document parsing workflow"}')

    plan = QueryPlanner().plan(
        "Download report-17.pdf and return its 2025 revenue",
        context="No result yet",
        client=client,
        model="test-model",
    )

    assert plan.literal == "Download report-17.pdf and return its 2025 revenue"
    assert plan.abstract == "find a reusable document parsing workflow"
    assert not plan.used_fallback
    assert plan.input_tokens == 11
    assert plan.output_tokens == 7
    assert len(client.chat.completions.calls) == 1


def test_query_planner_offline_fallback_keeps_literal_query() -> None:
    plan = QueryPlanner().plan("literal entity 123", client=None, model="")

    assert plan.literal == "literal entity 123"
    assert plan.abstract == ""
    assert plan.used_fallback


def test_context_composer_selects_and_synthesizes_in_one_call() -> None:
    client = _Client(
        '{"keep_ids":[1],"guidance":"Consider validating the parser first [M1]",'
        '"no_guidance":false}'
    )
    candidates = [
        {"id": "a", "score": 0.9, "text": "irrelevant entity-specific answer"},
        {"id": "b", "score": 0.8, "text": "validate parser before extraction"},
    ]

    result = MemoryContextComposer().compose(
        "parse a report",
        candidates,
        client=client,
        model="test-model",
    )

    assert result.kept_indices == [1]
    assert "[M1]" in result.guidance
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert len(client.chat.completions.calls) == 1


@pytest.mark.parametrize(
    "guidance",
    (
        "Use the selected workflow [M1], not the unrelated result [M0]",
        "Use the selected workflow [M1] and an invented source [M99]",
    ),
)
def test_context_composer_rejects_unselected_or_unknown_citations(guidance) -> None:
    client = _Client(
        json.dumps(
            {"keep_ids": [1], "guidance": guidance, "no_guidance": False}
        )
    )
    candidates = [
        {"id": "a", "score": 0.9, "text": "top fallback memory"},
        {"id": "b", "score": 0.8, "text": "selected workflow"},
    ]

    result = MemoryContextComposer().compose(
        "query",
        candidates,
        client=client,
        model="test-model",
    )

    assert result.used_fallback
    assert result.kept_indices == [0]
    assert "[M1]" not in result.guidance
    assert "[M99]" not in result.guidance


def test_context_composer_offline_fallback_is_top_one_raw_memory() -> None:
    result = MemoryContextComposer().compose(
        "query",
        [
            {"id": "top", "score": 0.8, "text": "top memory"},
            {"id": "other", "score": 0.7, "text": "other memory"},
        ],
        client=None,
        model="",
    )

    assert result.kept_indices == [0]
    assert "top memory" in result.guidance
    assert "other memory" not in result.guidance
    assert result.used_fallback


def test_context_composer_preserves_usage_when_model_output_falls_back() -> None:
    client = _Client("not valid json")

    result = MemoryContextComposer().compose(
        "query",
        [{"id": "top", "score": 1.0, "text": "fallback memory"}],
        client=client,
        model="test-model",
    )

    assert result.used_fallback
    assert result.input_tokens == 11
    assert result.output_tokens == 7


def test_runtime_components_accept_the_agent_callable_model_contract() -> None:
    model = _CallableModel(
        '{"abstract_query":"reusable parsing operation"}',
        '{"keep_ids":[0],"guidance":"Validate the parser first [M0]",'
        '"no_guidance":false}',
    )

    plan = QueryPlanner().plan("parse this report", client=model)
    result = MemoryContextComposer().compose(
        plan.literal,
        [{"id": "a", "score": 1.0, "text": "validate the parser"}],
        client=model,
        model="",
    )

    assert plan.abstract == "reusable parsing operation"
    assert result.kept_indices == [0]
    assert len(model.calls) == 2


def test_phase_registry_allows_only_one_explicit_refresh_attempt() -> None:
    registry = InjectionSessionRegistry()
    key = registry.key("query", "task-1")

    assert registry.phase_allowed(key, MemoryStatus.BEGIN)
    assert not registry.phase_allowed(key, MemoryStatus.IN, refresh_boundary=False)
    assert registry.phase_allowed(key, MemoryStatus.IN, refresh_boundary=True)
    # The first attempt consumes the budget even when no guidance was committed.
    assert not registry.phase_allowed(key, MemoryStatus.IN, refresh_boundary=True)


def test_phase_registry_filters_seen_units_and_drops_duplicate_guidance() -> None:
    registry = InjectionSessionRegistry()
    key = registry.key("query", "task-2")
    registry.phase_allowed(key, MemoryStatus.BEGIN)

    assert registry.commit(key, MemoryStatus.BEGIN, ["a"], "guidance")
    assert registry.unseen_indices(key, ["a", "b", "c"]) == [1, 2]
    assert not registry.commit(key, MemoryStatus.IN, ["b"], "guidance")
