"""Offline regressions for architecture-bound model dependencies and cost."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from automem.memory_schema import MemoryUnit, MemoryUnitType
from automem.providers.modular_memory_provider import ModularMemoryProvider
from automem.retrieval.base_retriever import MemoryPack, QueryContext, ScoredUnit
from automem.retrieval.cbr_rerank_retriever import CBRRerankRetriever
from automem.storage.llm_graph_storage import LLMGraphStore


class _CallableModel:
    model_id = "offline-model"

    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        usage = SimpleNamespace(prompt_tokens=13, completion_tokens=3)
        return SimpleNamespace(
            content=self.content,
            raw=SimpleNamespace(usage=usage),
        )


class _EmbeddingModel:
    def encode(self, text, convert_to_numpy=True):
        value = float(sum(str(text).encode("utf-8")) % 11 + 1)
        vector = np.asarray([value, value + 1.0, value + 2.0])
        return vector / np.linalg.norm(vector)


def _tip(unit_id: str, principle: str) -> MemoryUnit:
    return MemoryUnit(
        id=unit_id,
        type=MemoryUnitType.TIP,
        content={"topic": unit_id, "principle": principle},
        source_task_query=f"source {unit_id}",
        task_outcome="success",
    )


def test_cbr_rerank_uses_injected_model_independent_of_environment(monkeypatch):
    model = _CallableModel("9")
    retriever = CBRRerankRetriever(
        store=object(),
        model=model,
        usage_in_task_metrics=True,
        config={"rerank_pool_size": 2},
    )
    ctx = QueryContext(query="How should this task be solved?")

    def candidate_pack(_ctx, top_k):
        assert top_k == 2
        return MemoryPack(
            query_context=ctx,
            scored_units=[
                ScoredUnit(_tip("a", "first"), 0.8),
                ScoredUnit(_tip("b", "second"), 0.6),
            ],
        )

    retriever._cbr.retrieve = candidate_pack
    monkeypatch.setenv("MEMORY_TASK_GATE_MODEL", "wrong-model")
    monkeypatch.setenv("MEMORY_TASK_GATE_API_KEY", "wrong-key")
    first = retriever.retrieve(ctx, top_k=2)
    monkeypatch.delenv("MEMORY_TASK_GATE_MODEL")
    monkeypatch.setenv("MEMORY_TASK_GATE_API_BASE", "http://invalid.local")
    second = retriever.retrieve(ctx, top_k=2)

    assert [unit.unit.id for unit in first.scored_units] == ["a", "b"]
    assert [unit.unit.id for unit in second.scored_units] == ["a", "b"]
    assert first.retriever_name == second.retriever_name == "cbr_rerank"
    assert len(model.calls) == 4
    assert retriever.get_usage_metrics() == {
        "rerank_calls": 4,
        "rerank_input_tokens": 52,
        "rerank_output_tokens": 12,
        "rerank_usage_in_task_metrics": True,
    }


def test_cbr_rerank_fails_closed_without_model(monkeypatch):
    monkeypatch.setenv("MEMORY_TASK_GATE_MODEL", "ignored")
    monkeypatch.setenv("MEMORY_TASK_GATE_API_KEY", "ignored")
    monkeypatch.setenv("MEMORY_TASK_GATE_API_BASE", "http://ignored.local")

    with pytest.raises(ValueError, match="requires an injected"):
        CBRRerankRetriever(store=object())


@pytest.mark.parametrize("response", ["not-a-score", "11", "score: 9"])
def test_cbr_rerank_fails_closed_on_invalid_model_output(response):
    retriever = CBRRerankRetriever(store=object(), model=_CallableModel(response))

    with pytest.raises(RuntimeError, match="exactly one integer"):
        retriever._score_candidate("query", _tip("candidate", "principle"))


def test_cbr_rerank_fails_closed_on_model_error():
    def broken_model(*args, **kwargs):
        raise OSError("offline")

    retriever = CBRRerankRetriever(store=object(), model=broken_model)

    with pytest.raises(RuntimeError, match="model call failed"):
        retriever._score_candidate("query", _tip("candidate", "principle"))


def test_provider_injects_task_model_into_llm_graph_and_exports_usage(tmp_path):
    model = _CallableModel('{"entities": []}')
    provider = ModularMemoryProvider(
        {
            "storage_dir": str(tmp_path / "memory"),
            "storage_type": "llm_graph",
            "retriever_type": "hybrid",
            "enabled_prompts": ["tip"],
            "embedding_model": _EmbeddingModel(),
            "management_preset": "lightweight",
            "model": model,
        }
    )

    assert provider.initialize()
    provider.store.upsert_memory_unit(_tip("graph-unit", "persist atomically"))

    assert provider.store.llm_client is model
    assert len(model.calls) == 1
    metrics = provider.get_experiment_metrics()
    assert metrics["llm_graph_calls"] == 1
    assert metrics["llm_graph_input_tokens"] == 13
    assert metrics["llm_graph_output_tokens"] == 3
    assert metrics["llm_graph_usage_in_task_metrics"] is True


def test_llm_graph_ignores_ambient_credentials_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://ambient.invalid")
    monkeypatch.setenv("LLM_GRAPH_MODEL", "ambient-model")

    with pytest.raises(ValueError, match=r"requires config\['model'\]"):
        LLMGraphStore({"storage_dir": str(tmp_path / "graph")})


def test_llm_graph_maintenance_mode_loads_without_enabling_extraction(tmp_path):
    store = LLMGraphStore(
        {"storage_dir": str(tmp_path / "graph"), "maintenance_mode": True}
    )

    assert store.initialize()
    assert store.add([_tip("seed", "carry canonical memory")]) == 1
    assert [unit.id for unit in store.get_all()] == ["seed"]
    with pytest.raises(RuntimeError, match="maintenance mode"):
        store.upsert_memory_unit(_tip("new", "requires extraction"))
