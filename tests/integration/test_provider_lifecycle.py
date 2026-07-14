"""Offline integration test for the real modular provider read path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from automem.architecture.compiler import RuntimeConfig
from automem.config import get_memory_config
from automem.management.presets import get_preset
from automem.memory_schema import MemoryUnit, MemoryUnitType
from automem.memory_types import MemoryRequest, MemoryStatus, MemoryType
from automem.providers.modular_memory_provider import ModularMemoryProvider


class _EmbeddingModel:
    """Small deterministic encoder with no model download or network access."""

    def encode(self, text, convert_to_numpy=True):
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        vector = np.asarray([digest[0] + 1, digest[1] + 1, digest[2] + 1], dtype=float)
        return vector / np.linalg.norm(vector)


class _RuntimeModel:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("usage identity test must not invoke the model")


def _tip(unit_id: str, topic: str, principle: str, embedder: _EmbeddingModel) -> MemoryUnit:
    unit = MemoryUnit(
        id=unit_id,
        type=MemoryUnitType.TIP,
        content={
            "topic": topic,
            "principle": principle,
            "micro_example": principle,
        },
        source_task_id=f"source-{unit_id}",
        source_task_query=f"previous task {unit_id}",
        task_outcome="success",
    )
    unit.compute_signature()
    unit.embedding = embedder.encode(unit.content_text())
    return unit


def test_provider_begin_then_one_refresh_without_external_services(tmp_path, monkeypatch):
    for name in (
        "MEMORY_DISAGREEMENT_API_KEY",
        "MODULAR_STORAGE_TYPE",
        "MODULAR_STORAGE_DIR",
        "MODULAR_RETRIEVER_TYPE",
        "MODULAR_ADDITIONAL_STORES",
    ):
        monkeypatch.delenv(name, raising=False)

    embedder = _EmbeddingModel()
    provider = ModularMemoryProvider(
        {
            "storage_dir": str(tmp_path / "memory"),
            "storage_type": "json",
            "retriever_type": "hybrid",
            "enabled_prompts": ["tip"],
            "embedding_model": embedder,
            "management_preset": "lightweight",
            "top_k": 2,
        }
    )
    assert provider.initialize()
    assert provider.store.add(
        [
            _tip("atomic", "atomic persistence", "write updates atomically", embedder),
            _tip("refresh", "planning refresh", "refresh after a summary", embedder),
        ]
    ) == 2

    common = {
        "query": "How should an agent persist memory and refresh its plan?",
        "context": "",
        "additional_params": {"task_id": "integration-task"},
    }
    begin = provider.provide_memory(
        MemoryRequest(status=MemoryStatus.BEGIN, **common)
    )

    assert begin.total_count == 1
    assert begin.memories[0].metadata["runtime_policy_id"] == "automem-runtime-v1"
    begin_ids = set(begin.memories[0].metadata["citation_ids"])
    assert len(begin_ids) == 1
    assert "[memory:" in begin.memories[0].content

    refresh_params = {**common["additional_params"], "refresh_boundary": True}
    refresh = provider.provide_memory(
        MemoryRequest(
            query=common["query"],
            context="A summary has just been produced.",
            status=MemoryStatus.IN,
            additional_params=refresh_params,
        )
    )
    assert refresh.total_count == 1
    assert begin_ids.isdisjoint(refresh.memories[0].metadata["citation_ids"])

    second_refresh = provider.provide_memory(
        MemoryRequest(
            query=common["query"],
            context="A later summary.",
            status=MemoryStatus.IN,
            additional_params=refresh_params,
        )
    )
    assert second_refresh.total_count == 0
    assert provider.get_experiment_metrics()["phase_refresh_denied"] == 1


def test_edge_feedback_tracks_only_paths_supporting_injected_units():
    edges = [
        ("m:seed", "m:middle", "SIMILAR"),
        ("m:middle", "m:selected", "SIMILAR"),
        ("m:noise", "m:unused", "SIMILAR"),
    ]

    supporting = ModularMemoryProvider._supporting_edges(edges, ["selected"])

    assert supporting == [
        ("m:middle", "m:selected", "SIMILAR"),
        ("m:seed", "m:middle", "SIMILAR"),
    ]


def test_provider_distinguishes_shared_and_external_runtime_usage():
    task_model = _RuntimeModel()
    shared = ModularMemoryProvider({"model": task_model})
    shared_client = shared._build_runtime_client()

    assert shared_client is task_model
    assert shared._runtime_usage_is_in_task_metrics(shared_client)

    external_model = _RuntimeModel()
    external = ModularMemoryProvider(
        {"model": task_model, "runtime_client": external_model}
    )
    external_client = external._build_runtime_client()

    assert external_client is external_model
    assert not external._runtime_usage_is_in_task_metrics(external_client)


def test_structured_runtime_config_initializes_real_provider(tmp_path):
    runtime = RuntimeConfig(
        extract_plan={
            "extract_types": ["tip"],
            "storage_routing": {"tip": "json"},
        },
        primary_storage_type="json",
        storage_dir=str(tmp_path / "compiled-memory"),
        retrieval_type="hybrid",
        retrieval_config={"top_k": 2, "gate_threshold": 0.4},
        management_preset="lightweight",
        management_config=asdict(get_preset("lightweight")),
    )
    runtime_path = tmp_path / "runtime_config.json"
    runtime_path.write_text(json.dumps(runtime.to_dict()), encoding="utf-8")

    config = get_memory_config(MemoryType.MODULAR, runtime_path)
    config["embedding_model"] = _EmbeddingModel()
    provider = ModularMemoryProvider(config)

    assert provider.initialize()
    assert provider.storage_dir == str(tmp_path / "compiled-memory")
    assert provider.storage_type == "json"
    assert provider.retriever_type == "hybrid"
    assert provider.enabled_prompts == ["tip"]
    assert provider._management_preset == "lightweight"
