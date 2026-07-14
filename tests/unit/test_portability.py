from __future__ import annotations

import json
import sys
from argparse import Namespace
from importlib.resources import files
from pathlib import Path

import pytest

from automem.architecture.compiler import RuntimeConfig
from automem.config import DEFAULT_CONFIG, PROMPT_DIR, get_memory_config
from automem.memory_types import PROVIDER_MAPPING, MemoryType
from automem.resources import prompt_path, read_prompt_text
from automem.search.protocol import ProtocolConfig


def test_packaged_prompts_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert prompt_path().is_dir()
    assert prompt_path("tips_prompt.txt").is_file()
    assert "management" in read_prompt_text("meta", "architecture_search.txt")
    assert Path(PROMPT_DIR).resolve() == prompt_path().resolve()


def test_query_classifier_uses_the_installed_prompt_resource(tmp_path, monkeypatch):
    from automem.retrieval.query_classifier import QueryClassifier
    from automem.retrieval.tag_vocabulary import TagVocabulary

    monkeypatch.chdir(tmp_path)
    classifier = QueryClassifier(lambda *_args, **_kwargs: None, TagVocabulary())

    assert "task_domain" in classifier._template_str
    assert "{{ query }}" in classifier._template_str


def test_gaia_attachment_resolves_relative_to_metadata(tmp_path, monkeypatch):
    from automem.benchmarks.gaia import resolve_attachment_path

    metadata = tmp_path / "dataset" / "metadata.jsonl"
    metadata.parent.mkdir()
    metadata.write_text("", encoding="utf-8")
    attachment = metadata.parent / "attachments" / "input.zip"
    attachment.parent.mkdir()
    attachment.write_bytes(b"placeholder")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_attachment_path("attachments/input.zip", metadata)

    assert Path(resolved) == attachment


def test_only_shipped_providers_are_publicly_registered():
    assert set(MemoryType) == {MemoryType.MODULAR}
    assert set(PROVIDER_MAPPING) == set(MemoryType)
    assert set(DEFAULT_CONFIG["providers"]) == set(MemoryType)

    providers = files("automem").joinpath("providers")
    for class_name, module_name in PROVIDER_MAPPING.values():
        assert class_name.endswith("Provider")
        assert providers.joinpath(f"{module_name}.py").is_file()
    assert providers.joinpath("prompt_support.py").is_file()
    assert not providers.joinpath("prompt_based_memory_provider.py").is_file()

    assert DEFAULT_CONFIG["providers"][MemoryType.MODULAR]["enabled_prompts"] == [
        "tip"
    ]


def test_search_and_benchmark_sources_have_no_repo_bootstrap_paths():
    automem = files("automem")
    engine = automem.joinpath("search", "engine.py").read_text(encoding="utf-8")
    xbench = automem.joinpath(
        "benchmarks", "xbench_deepsearch", "runner.py"
    ).read_text(encoding="utf-8")

    for retired in (
        "REPO_ROOT",
        "scripts/eval",
        'env["PYTHONPATH"]',
        "MODULAR_",
        "env_overrides",
        "extract_plan_json",
        'add_argument("--evo_protocol"',
        'add_argument("--no_self_retrieval"',
        'add_argument("--fold_rotation"',
        'add_argument("--canonical_merge"',
        'add_argument("--champion_scoring"',
        'add_argument("--acceptance"',
        'add_argument("--final_runoff"',
        'add_argument("--val_every"',
    ):
        assert retired not in engine
    assert "python -m" in engine
    assert "xbench-evals-main" not in xbench
    assert "sys.path.insert" not in xbench


def test_evaluation_protocol_is_single_and_not_runtime_configurable():
    protocol = ProtocolConfig.resolve(
        Namespace(
            evo_protocol="legacy",
            no_self_retrieval="off",
            fold_rotation=99,
            canonical_merge="all",
            champion_scoring="last",
            acceptance="threshold",
            final_runoff=0,
            val_every=9,
        )
    )

    assert protocol.to_dict() == {
        "name": "automem-v1",
        "fold_rotation": 2,
        "canonical_merge": "winner",
        "champion_scoring": "pooled",
        "acceptance": "paired",
        "accept_alpha": 0.1,
        "final_runoff": 2,
        "val_every": 0,
    }
    with pytest.raises(TypeError):
        ProtocolConfig(fold_rotation=3)


def test_runtime_config_is_structured_and_merges_into_provider_config(tmp_path):
    runtime = RuntimeConfig(
        extract_plan={
            "extract_types": ["tip"],
            "storage_routing": {"tip": "json"},
        },
        primary_storage_type="json",
        storage_dir=str(tmp_path / "store_json"),
        additional_stores={},
        retrieval_type="hybrid",
        retrieval_config={"top_k": 4, "gate_threshold": 0.4},
        management_preset="json_full",
        management_config={
            "post_task_ops": ["reflection_correction"],
            "periodic_ops": ["quality_curation"],
            "on_insert_ops": ["signature_dedup"],
            "periodic_interval": 10,
            "op_configs": {},
        },
    )
    runtime_path = tmp_path / "runtime_config.json"
    runtime_path.write_text(json.dumps(runtime.to_dict()), encoding="utf-8")

    config = get_memory_config(MemoryType.MODULAR, runtime_path)
    assert config["storage_type"] == "json"
    assert config["retriever_type"] == "hybrid"
    assert config["enabled_prompts"] == ["tip"]
    assert config["additional_stores"] == {}
    assert config["management_preset"] == "json_full"
    assert config["management_config"]["periodic_interval"] == 10
    assert config["top_k"] == 4
    assert "env_overrides" not in runtime.to_dict()


def test_runtime_config_rejects_unknown_fields_and_non_modular_provider(tmp_path):
    payload = RuntimeConfig().to_dict()
    payload["legacy_toggle"] = True
    runtime_path = tmp_path / "runtime_config.json"
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown runtime config fields"):
        get_memory_config(MemoryType.MODULAR, runtime_path)

    runtime_path.write_text(json.dumps(RuntimeConfig().to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="ships only the modular provider"):
        get_memory_config("legacy", runtime_path)


def test_search_launches_installed_runner_with_structured_config(tmp_path, monkeypatch):
    from automem.search import engine

    infile = tmp_path / "tasks.jsonl"
    infile.write_text("{}\n", encoding="utf-8")
    tasks_dir = tmp_path / "candidate" / "tasks"
    tasks_dir.mkdir(parents=True)
    captured = {}

    class FakeProcess:
        returncode = 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
    args = Namespace(
        infile=str(infile),
        eval_script=None,
        benchmark="GAIA",
        max_steps=4,
        token_budget=512,
        concurrency=1,
        model=None,
        judge_model=None,
    )
    runtime = RuntimeConfig(
        extract_plan={"extract_types": ["tip"], "storage_routing": {"tip": "json"}}
    )

    process = engine._start_eval_subprocess(tasks_dir, [0], runtime, args)
    process._log_file.close()

    command = captured["command"]
    assert command[:3] == [sys.executable, "-m", "automem.benchmarks.gaia.runner"]
    assert "--runtime_config_json" in command
    assert "--extract_plan_json" not in command
    assert "env" not in captured["kwargs"]

    runtime_path = tasks_dir.parent / "runtime_config.json"
    assert json.loads(runtime_path.read_text(encoding="utf-8")) == runtime.to_dict()


def test_search_baseline_omits_memory_provider_flag(tmp_path, monkeypatch):
    from automem.search import engine

    infile = tmp_path / "tasks.jsonl"
    infile.write_text("{}\n", encoding="utf-8")
    tasks_dir = tmp_path / "baseline" / "tasks"
    tasks_dir.mkdir(parents=True)
    captured = {}

    class FakeProcess:
        returncode = 0

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
    args = Namespace(
        infile=str(infile),
        eval_script=None,
        benchmark="GAIA",
        max_steps=4,
        token_budget=512,
        concurrency=1,
        model=None,
        judge_model=None,
    )

    process = engine._start_eval_subprocess(tasks_dir, [0], None, args)
    process._log_file.close()

    assert "--memory_provider" not in captured["command"]


def test_eval_subprocess_start_failure_is_raised_and_closes_log(tmp_path, monkeypatch):
    from automem.search import engine

    infile = tmp_path / "tasks.jsonl"
    infile.write_text("{}\n", encoding="utf-8")
    tasks_dir = tmp_path / "candidate" / "tasks"
    tasks_dir.mkdir(parents=True)
    captured = {}

    def fail_popen(command, **kwargs):
        captured["log"] = kwargs["stdout"]
        raise OSError("cannot start runner")

    monkeypatch.setattr(engine.subprocess, "Popen", fail_popen)
    args = Namespace(
        infile=str(infile),
        eval_script=None,
        benchmark="GAIA",
        max_steps=4,
        token_budget=512,
        concurrency=1,
        model=None,
        judge_model=None,
    )

    with pytest.raises(OSError, match="cannot start runner"):
        engine._run_eval_subprocess(tasks_dir, [0], None, args)

    assert captured["log"].closed


def test_no_memory_eval_omits_the_modular_only_provider_flag(tmp_path, monkeypatch):
    from automem.search import engine

    infile = tmp_path / "tasks.jsonl"
    infile.write_text("{}\n", encoding="utf-8")
    tasks_dir = tmp_path / "baseline" / "tasks"
    tasks_dir.mkdir(parents=True)
    captured = {}

    class FakeProcess:
        returncode = 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)
    args = Namespace(
        infile=str(infile),
        eval_script=None,
        benchmark="GAIA",
        max_steps=4,
        token_budget=512,
        concurrency=1,
        model=None,
        judge_model=None,
    )

    process = engine._start_eval_subprocess(tasks_dir, [0], None, args)
    process._log_file.close()

    assert "--memory_provider" not in captured["command"]
    assert "--runtime_config_json" not in captured["command"]


def test_validation_summary_uses_packaged_aggregation_module(tmp_path):
    from automem.search import engine

    task = {
        "task_id": "portable-task",
        "item_index": 1,
        "task_score": 1.0,
        "question": "What is the answer?",
        "status": "success",
        "judge_unjudged": False,
        "task_identity": "0" * 64,
    }
    (tmp_path / "1.json").write_text(json.dumps(task), encoding="utf-8")

    summary = engine._build_per_task_pass_fail_summary(tmp_path)

    assert len(summary) == 1
    assert summary[0]["task_id"] == "portable-task"
    assert summary[0]["passed"] is True


def test_proposer_prompts_only_name_canonical_management_presets():
    prompt_names = (
        "architecture_search.txt",
        "layer_diagnosis_fixed.txt",
        "ledger_update.txt",
    )
    retired = ("skywork_unified", "graph_adaptive", "graph_full", "json_basic")

    for name in prompt_names:
        content = read_prompt_text("meta", name)
        for old_name in retired:
            assert old_name not in content

    architecture_prompt = read_prompt_text("meta", "architecture_search.txt")
    for canonical in (
        "lightweight",
        "json_full",
        "tool_manager",
        "graph_consolidate",
    ):
        assert canonical in architecture_prompt
