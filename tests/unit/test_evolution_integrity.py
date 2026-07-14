from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from automem.data_split import DataSplitConfig
from automem.evaluation.utils import require_complete_task_run
from automem.memory_schema import MemoryUnit, MemoryUnitType
from automem.search import engine
from automem.search.pareto_front import ParetoEntry, ParetoFront
from automem.search.protocol import ProtocolConfig
from automem.search.protocol import (
    load_champion_state,
    select_runoff_contenders,
    update_champion_after_round,
)
from automem.storage.graph_storage import GraphStore
from automem.storage.json_storage import JsonStorage


VALID_TASK_IDENTITY = "0" * 64


def _write_valid_result(path, item_index: int, score: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "item_index": item_index,
                "task_score": score,
                "status": "success",
                "judge_unjudged": False,
                "task_identity": VALID_TASK_IDENTITY,
            }
        ),
        encoding="utf-8",
    )


def _unit(unit_id: str, *, usage_count: int = 0) -> MemoryUnit:
    unit = MemoryUnit(
        id=unit_id,
        type=MemoryUnitType.TIP,
        content={"topic": unit_id, "principle": f"principle-{unit_id}"},
        source_task_id="task-1",
        usage_count=usage_count,
    )
    unit.compute_signature()
    return unit


def _protocol_args(infile: str, **overrides) -> Namespace:
    values = {
        "infile": infile,
        "search_prompt": None,
        "eval_script": None,
        "benchmark": "GAIA",
        "model": "task-model",
        "search_model": None,
        "judge_model": "judge-model",
        "diagnosis_model": "diagnosis-model",
        "max_steps": 40,
        "token_budget": 8192,
        "dry_run": False,
        "_resolved_split": {
            "profile_indices": [0],
            "optimization_indices": [1],
            "validation_indices": [],
            "final_test_indices": [],
        },
        "_loaded_tasks": [],
    }
    values.update(overrides)
    return Namespace(**values)


def test_protocol_digest_tracks_task_bytes_split_and_agent_limits(tmp_path):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one","Question":"first"}\n', encoding="utf-8")
    protocol = ProtocolConfig.resolve(None)
    args = _protocol_args(str(infile))

    first = engine._compute_eval_protocol_signature("task-model", protocol, args)
    args.max_steps = 41
    steps_changed = engine._compute_eval_protocol_signature("task-model", protocol, args)
    args.max_steps = 40
    args._resolved_split["optimization_indices"] = []
    split_changed = engine._compute_eval_protocol_signature("task-model", protocol, args)
    args._resolved_split["optimization_indices"] = [1]
    infile.write_text('{"task_id":"one","Question":"changed"}\n', encoding="utf-8")
    input_changed = engine._compute_eval_protocol_signature("task-model", protocol, args)

    assert len(
        {
            first["digest"],
            steps_changed["digest"],
            split_changed["digest"],
            input_changed["digest"],
        }
    ) == 4


def test_protocol_digest_tracks_role_endpoints_and_web_controls(tmp_path, monkeypatch):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one"}\n', encoding="utf-8")
    args = _protocol_args(str(infile))

    monkeypatch.delenv("TASK_API_BASE", raising=False)
    monkeypatch.delenv("JUDGE_API_BASE", raising=False)
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    first = engine._compute_eval_protocol_signature(args=args)

    monkeypatch.setenv("TASK_API_BASE", "https://task.example/v1")
    monkeypatch.setenv("JUDGE_API_BASE", "https://judge.example/v1")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "custom-provider")
    changed = engine._compute_eval_protocol_signature(args=args)

    assert first["digest"] != changed["digest"]
    assert first["baseline_digest"] != changed["baseline_digest"]
    assert changed["endpoints"]["task_api_base"] == "https://task.example/v1"
    assert changed["endpoints"]["judge_api_base"] == "https://judge.example/v1"


def test_protocol_digest_tracks_effective_diagnosis_role_configuration(
    tmp_path, monkeypatch
):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one"}\n', encoding="utf-8")
    args = _protocol_args(str(infile))
    monkeypatch.setenv("DIAGNOSIS_API_BASE", "https://diagnosis.example/v1")
    monkeypatch.delenv("DIAGNOSIS_API_KEY", raising=False)

    generic_fallback = engine._compute_eval_protocol_signature(args=args)
    monkeypatch.setenv("DIAGNOSIS_API_KEY", "configured-secret")
    dedicated = engine._compute_eval_protocol_signature(args=args)

    assert generic_fallback["digest"] != dedicated["digest"]
    assert not generic_fallback["endpoints"]["diagnosis_role_configured"]
    assert dedicated["endpoints"]["diagnosis_role_configured"]
    assert generic_fallback["endpoints"]["diagnosis_api_base"] != dedicated[
        "endpoints"
    ]["diagnosis_api_base"]


def test_task_model_is_not_used_as_proposer_model(tmp_path, monkeypatch):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one"}\n', encoding="utf-8")
    monkeypatch.delenv("SEARCH_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL", "default-proposer")
    args = _protocol_args(
        str(infile), model="task-only", search_model="explicit-proposer"
    )

    signature = engine._compute_eval_protocol_signature(
        eval_model=args.model,
        args=args,
    )

    assert signature["eval_model"] == "task-only"
    assert signature["search_model"] == "explicit-proposer"


def test_proposer_loader_uses_explicit_search_model_not_task_model(monkeypatch):
    from flashoagents import models as model_module

    captured = {}

    class _Model:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(model_module, "OpenAIServerModel", _Model)
    monkeypatch.setattr(
        engine,
        "resolve_openai_endpoint",
        lambda role: ("key", "https://search.example/v1"),
    )

    engine.load_model(
        Namespace(model="task-only", search_model="explicit-proposer")
    )

    assert captured["model_id"] == "explicit-proposer"
    assert captured["api_base"] == "https://search.example/v1"


def test_blank_search_model_uses_the_same_fallback_in_loader_and_digest(
    tmp_path, monkeypatch
):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one"}\n', encoding="utf-8")
    args = _protocol_args(str(infile), search_model="   ")
    monkeypatch.delenv("SEARCH_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL", "proposer-a")

    first = engine._compute_eval_protocol_signature(args=args)
    first_model, _ = engine._resolve_search_model(args)
    monkeypatch.setenv("DEFAULT_MODEL", "proposer-b")
    second = engine._compute_eval_protocol_signature(args=args)
    second_model, _ = engine._resolve_search_model(args)

    assert (first_model, second_model) == ("proposer-a", "proposer-b")
    assert (first["search_model"], second["search_model"]) == (
        "proposer-a",
        "proposer-b",
    )
    assert first["digest"] != second["digest"]


def test_xbench_defaults_apply_only_without_explicit_split():
    defaults = Namespace(
        benchmark="xBench-DeepSearch",
        data_split=None,
        _split_sizes_explicit=False,
        warmup_n=19,
        search_n=100,
        validation_n=30,
        test_n=15,
    )
    engine._apply_benchmark_split_defaults(defaults)
    assert (
        defaults.warmup_n,
        defaults.search_n,
        defaults.validation_n,
        defaults.test_n,
    ) == (10, 70, 10, 10)

    explicit = Namespace(
        benchmark="xbench",
        data_split=None,
        _split_sizes_explicit=True,
        warmup_n=19,
        search_n=100,
        validation_n=30,
        test_n=15,
    )
    engine._apply_benchmark_split_defaults(explicit)
    assert (
        explicit.warmup_n,
        explicit.search_n,
        explicit.validation_n,
        explicit.test_n,
    ) == (19, 100, 30, 15)


def test_search_task_loader_rejects_duplicate_xbench_ids(tmp_path):
    infile = tmp_path / "xbench.csv"
    infile.write_text(
        "id,prompt,answer,reference_steps,canary\n"
        "same,cQ==,YQ==,,k\n"
        "same,cQ==,YQ==,,k\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate id"):
        engine.load_tasks(str(infile))


def test_final_validation_is_a_post_search_action_not_a_search_digest_input(tmp_path):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one"}\n', encoding="utf-8")
    args = _protocol_args(str(infile), final_validation=False)

    without_final = engine._compute_eval_protocol_signature(args=args)
    args.final_validation = True
    with_final = engine._compute_eval_protocol_signature(args=args)

    assert without_final["digest"] == with_final["digest"]
    assert not without_final["post_search_actions"]["final_validation"]
    assert with_final["post_search_actions"]["final_validation"]


def test_protocol_signature_covers_all_shipped_prompt_resources(tmp_path):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one"}\n', encoding="utf-8")

    signature = engine._compute_eval_protocol_signature(
        args=_protocol_args(str(infile))
    )

    assert signature["runtime_info"]["prompt_resource_files"] >= 10
    assert len(signature["runtime_info"]["prompt_resources_sha256"]) == 64


def test_custom_runner_bytes_are_part_of_protocol_digest(tmp_path):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text('{"task_id":"one"}\n', encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("print('first')\n", encoding="utf-8")
    args = _protocol_args(str(infile), eval_script=str(runner))

    first = engine._compute_eval_protocol_signature(args=args)["digest"]
    runner.write_text("print('second')\n", encoding="utf-8")
    second = engine._compute_eval_protocol_signature(args=args)["digest"]

    assert first != second


@pytest.mark.parametrize(
    "failure_fields",
    [{"status": "error"}, {"status": "success", "judge_unjudged": True}],
)
def test_exact_result_guard_rejects_runner_and_judge_failures(
    tmp_path, failure_fields
):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    payload = {
        "task_id": "task-1",
        "item_index": 1,
        "task_score": 0.0,
        **failure_fields,
    }
    (tasks_dir / "1.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid_files"):
        engine._require_exact_task_results(tasks_dir, [0], "candidate")


@pytest.mark.parametrize(
    "task_score",
    [None, True, float("nan"), -0.1, 1.1],
)
def test_exact_result_guard_rejects_missing_or_invalid_scores(tmp_path, task_score):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    payload = {
        "task_id": "task-1",
        "item_index": 1,
        "status": "success",
    }
    if task_score is not None:
        payload["task_score"] = task_score
    (tasks_dir / "1.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid_files"):
        engine._require_exact_task_results(tasks_dir, [0], "candidate")


@pytest.mark.parametrize(
    ("results", "expected", "future_errors"),
    [
        ([{"status": "error"}], 1, []),
        ([{"status": "success", "judge_unjudged": True}], 1, []),
        ([{"status": "success"}], 2, []),
        ([{"status": "success"}], 1, ["worker crashed"]),
    ],
)
def test_benchmark_completion_guard_raises_for_infrastructure_failures(
    results, expected, future_errors
):
    with pytest.raises(RuntimeError, match="run incomplete"):
        require_complete_task_run("test", results, expected, future_errors)


def test_data_split_rejects_duplicates_negative_boolean_and_out_of_range():
    split = DataSplitConfig(
        profile_indices=[0, 0],
        optimization_indices=[-1],
        validation_indices=[True],
        final_test_indices=[5],
    )

    valid, errors = split.validate(total_tasks=5)

    assert not valid
    assert any("duplicate" in error for error in errors)
    assert any("negative" in error for error in errors)
    assert any("non-integer" in error for error in errors)
    assert any("exceed" in error for error in errors)


def test_runoff_uses_distinct_dominated_history_when_front_has_one_point():
    pareto = ParetoFront()
    architectures = [
        {"encode": "tip", "store": "json", "retrieve": "hybrid", "manage": "lightweight"},
        {"encode": "workflow", "store": "json", "retrieve": "hybrid", "manage": "lightweight"},
        {"encode": "shortcut", "store": "json", "retrieve": "hybrid", "manage": "lightweight"},
    ]
    for index, (accuracy, fitness) in enumerate(
        [(0.9, 0.9), (0.7, 0.7), (0.5, 0.5)]
    ):
        pareto.add(
            ParetoEntry(
                config_id=f"candidate-{index}",
                architecture=architectures[index],
                accuracy=accuracy,
                memory_lift=accuracy,
                fitness=fitness,
            )
        )

    assert pareto.size() == 1
    contenders = select_runoff_contenders(
        pareto,
        {"config_id": "candidate-0", "architecture": architectures[0]},
        k=2,
    )

    assert [row["config_id"] for row in contenders] == [
        "candidate-0",
        "candidate-1",
    ]


def test_pareto_config_id_replay_is_idempotent_in_pooled_mode():
    pareto = ParetoFront()
    pareto.measurement_mode = "pooled"
    architecture = {
        "encode": "tip",
        "store": "json",
        "retrieve": "hybrid",
        "manage": "lightweight",
    }
    first = ParetoEntry(
        config_id="r1_c0",
        architecture=architecture,
        accuracy=1.0,
        memory_lift=1.0,
        fitness=1.0,
    )

    assert pareto.add(first)
    assert pareto.add(first)
    assert len(pareto.all_evaluated()) == 1
    assert pareto.top_k_all(1)[0].n_evals == 1

    pareto.add(
        ParetoEntry(
            config_id="r2_c0",
            architecture=architecture,
            accuracy=0.0,
            memory_lift=0.0,
            fitness=0.0,
        )
    )
    pooled = pareto.top_k_all(1)[0]
    assert pooled.n_evals == 2
    assert pooled.fitness == 0.5


def test_champion_round_replay_does_not_reapply_promotion(tmp_path):
    architecture_a = {"name": "a"}
    architecture_b = {"name": "b"}
    update_champion_after_round(
        tmp_path,
        1,
        [
            {
                "config_id": "r1_c0",
                "architecture": architecture_a,
                "metrics": {"fitness": 0.5},
                "diversity_role": "explore",
            }
        ],
        {"r1_c0": {"1": 0.0}},
    )
    round_results = [
        {
            "config_id": "r2_c0",
            "architecture": architecture_a,
            "metrics": {"fitness": 0.5},
            "diversity_role": "champion",
        },
        {
            "config_id": "r2_c1",
            "architecture": architecture_b,
            "metrics": {"fitness": 0.8},
            "diversity_role": "explore",
        },
    ]
    scores = {
        "r2_c0": {str(index): 0.0 for index in range(1, 5)},
        "r2_c1": {str(index): 1.0 for index in range(1, 5)},
    }

    first = update_champion_after_round(tmp_path, 2, round_results, scores)
    state_after_first = load_champion_state(tmp_path)
    replay = update_champion_after_round(tmp_path, 2, round_results, scores)

    assert first["new_champion"] == "r2_c1"
    assert replay["reason"] == "round_already_applied"
    assert load_champion_state(tmp_path) == state_after_first
    assert state_after_first["promoted_over"] == "r1_c0"


def test_canonical_merge_receipt_prevents_counter_double_apply(tmp_path):
    run_dir = tmp_path / "run"
    existing = _unit("shared", usage_count=10)
    engine.save_canonical_pool(run_dir, [existing.to_dict()])

    store_dir = tmp_path / "candidate" / "tasks" / "store_json"
    store = JsonStorage({"db_path": str(store_dir / "memory_db.json")})
    assert store.initialize()
    store.add([_unit("shared", usage_count=2)])

    assert engine.sync_tasks_to_canonical(
        run_dir, tmp_path / "candidate" / "tasks", merge_id="round:1:candidate:1"
    ) == 0
    assert engine.load_canonical_pool(run_dir)[0]["usage_count"] == 12

    assert engine.sync_tasks_to_canonical(
        run_dir, tmp_path / "candidate" / "tasks", merge_id="round:1:candidate:1"
    ) == 0
    assert engine.load_canonical_pool(run_dir)[0]["usage_count"] == 12


def test_graph_edge_feedback_survives_canonical_handoff(tmp_path):
    source_tasks = tmp_path / "candidate" / "tasks"
    source_store = GraphStore(
        {"storage_dir": str(source_tasks / "store_graph")}
    )
    assert source_store.initialize()
    source_store.add([_unit("source"), _unit("target")])
    source_store._graph.add_edge(
        "m:source",
        "m:target",
        key="SIMILAR",
        edge_type="SIMILAR",
        base_weight=0.4,
        feedback_multiplier=1.5,
        usage_count=7,
        success_count=5,
        effective_weight=0.6,
    )
    source_store.save()

    run_dir = tmp_path / "run"
    engine.sync_tasks_to_canonical(
        run_dir, source_tasks, merge_id="round:1:candidate:graph"
    )
    runtime = engine._compile_architecture(
        {
            "extract_types": ["tip"],
            "storage_routing": {"tip": "graph"},
            "retrieval": "graph",
            "management": "graph_consolidate",
        },
        str(tmp_path / "destination"),
    )
    assert runtime is not None
    assert engine.import_canonical_to_storage(run_dir, runtime) == 2

    destination = GraphStore({"storage_dir": runtime.storage_dir})
    assert destination.initialize()
    edges = destination.export_relation_edges()
    matching = next(edge for edge in edges if edge["edge_type"] == "SIMILAR")

    assert matching["attributes"]["feedback_multiplier"] == 1.5
    assert matching["attributes"]["usage_count"] == 7
    assert matching["attributes"]["success_count"] == 5


def test_corrupt_canonical_pool_fails_closed(tmp_path):
    run_dir = tmp_path / "run"
    engine.save_canonical_pool(run_dir, [{"id": "bad", "type": "not-a-type"}])
    runtime = engine._compile_architecture(
        {
            "extract_types": ["tip"],
            "storage_routing": {"tip": "json"},
            "retrieval": "hybrid",
            "management": "lightweight",
        },
        str(tmp_path / "destination"),
    )
    assert runtime is not None

    with pytest.raises(RuntimeError, match="deserialize canonical pool"):
        engine.import_canonical_to_storage(run_dir, runtime)


def test_canonical_state_uses_versioned_envelope(tmp_path):
    run_dir = tmp_path / "run"
    engine.save_canonical_pool(run_dir, [_unit("one").to_dict()])

    payload = json.loads(engine.canonical_pool_path(run_dir).read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["applied_merges"] == []
    assert payload["periodic_rounds"] == []
    assert len(payload["units"]) == 1


def test_failed_canonical_maintenance_is_retryable(tmp_path, monkeypatch):
    from automem.management.ops.quality_curation import QualityCurationOp

    run_dir = tmp_path / "run"
    engine.save_canonical_pool(run_dir, [_unit("one").to_dict()])

    def fail_maintenance(self, context):
        raise RuntimeError("injected maintenance failure")

    monkeypatch.setattr(QualityCurationOp, "execute", fail_maintenance)

    with pytest.raises(RuntimeError, match="injected maintenance failure"):
        engine._run_canonical_periodic_ops(run_dir, round_id=3)

    payload = json.loads(engine.canonical_pool_path(run_dir).read_text(encoding="utf-8"))
    assert payload["periodic_rounds"] == []
    assert len(payload["units"]) == 1


def test_partial_candidate_retry_rebuilds_full_batch_from_round_snapshot(
    tmp_path, monkeypatch
):
    cand_dir = tmp_path / "candidate"
    tasks_dir = cand_dir / "tasks"
    tasks_dir.mkdir(parents=True)
    (cand_dir / "storage" / "sentinel").mkdir(parents=True)
    (tasks_dir / "1.json").write_text(
        json.dumps(
            {
                "item_index": 1,
                "task_score": 1.0,
                "status": "success",
                "judge_unjudged": False,
                "task_identity": VALID_TASK_IDENTITY,
            }
        ),
        encoding="utf-8",
    )
    launched = []
    imported = []

    class _Process:
        returncode = 0

        def wait(self):
            return 0

    def fake_start(*, tasks_dir, task_indices, runtime_config, args):
        launched.append(list(task_indices))
        for index in task_indices:
            (tasks_dir / f"{index + 1}.json").write_text(
                json.dumps(
                    {
                            "item_index": index + 1,
                            "task_score": 1.0,
                            "status": "success",
                            "judge_unjudged": False,
                            "task_identity": VALID_TASK_IDENTITY,
                    }
                ),
                encoding="utf-8",
            )
        return _Process()

    monkeypatch.setattr(engine, "_compile_architecture", lambda arch, path: object())
    monkeypatch.setattr(engine, "_start_eval_subprocess", fake_start)
    monkeypatch.setattr(
        engine,
        "import_canonical_to_storage",
        lambda source, runtime: imported.append(source) or 0,
    )
    snapshot = tmp_path / "round_start"

    complete, missing, extras = engine._ensure_batch_complete(
        tasks_dir=tasks_dir,
        expected_indices=[0, 1],
        arch={"name": "candidate"},
        cand_dir=cand_dir,
        args=Namespace(no_canonical_import=False),
        config_id="r1_c0",
        canonical_source_dir=snapshot,
    )

    assert complete and not missing and not extras
    assert launched == [[0, 1]]
    assert imported == [snapshot]
    assert not (cand_dir / "storage" / "sentinel").exists()


def test_stateful_stage_partial_outputs_remove_nested_and_split_storage(tmp_path):
    nested = tmp_path / "warmup-or-runoff"
    nested_tasks = nested / "tasks"
    nested_tasks.mkdir(parents=True)
    (nested / "storage" / "sentinel").mkdir(parents=True)
    (nested_tasks / "1.json").write_text(
        json.dumps(
            {
                "item_index": 1,
                "task_score": 1.0,
                "status": "success",
                "judge_unjudged": False,
                "task_identity": VALID_TASK_IDENTITY,
            }
        ),
        encoding="utf-8",
    )

    assert not engine._reuse_or_reset_stateful_stage(
        nested_tasks,
        [0, 1],
        [nested],
        "stateful stage",
    )
    assert not nested.exists()

    final_tasks = tmp_path / "final" / "tasks"
    final_storage = tmp_path / "final" / "storage"
    final_tasks.mkdir(parents=True)
    (final_storage / "sentinel").mkdir(parents=True)
    (final_tasks / "1.json").write_text(
        json.dumps(
            {
                "item_index": 1,
                "task_score": 1.0,
                "status": "success",
                "judge_unjudged": False,
                "task_identity": VALID_TASK_IDENTITY,
            }
        ),
        encoding="utf-8",
    )

    assert not engine._reuse_or_reset_stateful_stage(
        final_tasks,
        [0, 1],
        [final_tasks, final_storage],
        "final memory",
    )
    assert not final_tasks.exists()
    assert not final_storage.exists()


def test_stateful_stage_reuses_only_exact_results_with_required_storage(tmp_path):
    stage = tmp_path / "warmup"
    tasks = stage / "tasks"
    storage = stage / "storage"
    tasks.mkdir(parents=True)
    (storage / "sentinel").mkdir(parents=True)
    for item_index in (1, 2):
        (tasks / f"{item_index}.json").write_text(
            json.dumps(
                {
                    "item_index": item_index,
                    "task_score": 1.0,
                    "status": "success",
                    "judge_unjudged": False,
                    "task_identity": VALID_TASK_IDENTITY,
                }
            ),
            encoding="utf-8",
        )

    assert engine._reuse_or_reset_stateful_stage(
        tasks,
        [0, 1],
        [stage],
        "warmup",
        required_state_paths=[storage],
    )
    assert stage.exists()


def test_warmup_partial_resume_replays_full_batch_from_empty_storage(
    tmp_path, monkeypatch
):
    warmup = tmp_path / "warmup"
    (warmup / "storage" / "sentinel").mkdir(parents=True)
    _write_valid_result(warmup / "tasks" / "1.json", 1)
    launched = []

    def fake_eval(*, tasks_dir, task_indices, **_kwargs):
        assert not (warmup / "storage" / "sentinel").exists()
        launched.append(list(task_indices))
        for index in task_indices:
            _write_valid_result(tasks_dir / f"{index + 1}.json", index + 1)
        (warmup / "storage" / "rebuilt").mkdir(parents=True)
        return True

    monkeypatch.setattr(engine, "_compile_architecture", lambda *_args: object())
    monkeypatch.setattr(engine, "_run_eval_subprocess", fake_eval)
    monkeypatch.setattr(engine, "sync_tasks_to_canonical", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "load_canonical_pool", lambda *_args: [])

    engine.run_warmup(
        tmp_path,
        DataSplitConfig(profile_indices=[0, 1]),
        Namespace(warmup_n=2, dry_run=False),
    )

    assert launched == [[0, 1]]
    assert (warmup / "warmup_done.json").is_file()


def test_warmup_done_marker_without_backing_state_replays_full_batch(
    tmp_path, monkeypatch
):
    warmup = tmp_path / "warmup"
    warmup.mkdir()
    (warmup / "warmup_done.json").write_text(
        json.dumps({"n_tasks": 2, "pool_size": 0}),
        encoding="utf-8",
    )
    launched = []

    def fake_eval(*, tasks_dir, task_indices, **_kwargs):
        launched.append(list(task_indices))
        for index in task_indices:
            _write_valid_result(tasks_dir / f"{index + 1}.json", index + 1)
        (warmup / "storage" / "rebuilt").mkdir(parents=True)
        return True

    monkeypatch.setattr(engine, "_compile_architecture", lambda *_args: object())
    monkeypatch.setattr(engine, "_run_eval_subprocess", fake_eval)
    monkeypatch.setattr(engine, "sync_tasks_to_canonical", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(engine, "load_canonical_pool", lambda *_args: [])

    engine.run_warmup(
        tmp_path,
        DataSplitConfig(profile_indices=[0, 1]),
        Namespace(warmup_n=2, dry_run=False),
    )

    assert launched == [[0, 1]]
    assert json.loads((warmup / "warmup_done.json").read_text(encoding="utf-8")) == {
        "n_tasks": 2,
        "pool_size": 0,
    }


def test_runoff_partial_contender_replays_full_union(tmp_path, monkeypatch):
    from automem.search import protocol as protocol_module

    contender_dir = tmp_path / "final_runoff" / "contender_0_cfg"
    runoff_snapshot = tmp_path / "final_runoff" / "runoff_start_state"
    engine._save_bound_canonical_snapshot(tmp_path, runoff_snapshot)
    (contender_dir / "storage" / "sentinel").mkdir(parents=True)
    _write_valid_result(contender_dir / "tasks" / "1.json", 1)
    launched = []
    imported = []

    monkeypatch.setattr(protocol_module, "load_champion_state", lambda *_args: {})
    monkeypatch.setattr(
        protocol_module,
        "select_runoff_contenders",
        lambda *_args, **_kwargs: [
            {
                "config_id": "cfg",
                "source": "history",
                "architecture": {"management": "lightweight"},
            }
        ],
    )
    monkeypatch.setattr(engine, "_compile_architecture", lambda *_args: object())
    monkeypatch.setattr(
        engine,
        "import_canonical_to_storage",
        lambda source, _runtime: imported.append(source) or 0,
    )

    def fake_eval(*, tasks_dir, task_indices, **_kwargs):
        assert not (contender_dir / "storage" / "sentinel").exists()
        launched.append(list(task_indices))
        for index in task_indices:
            _write_valid_result(tasks_dir / f"{index + 1}.json", index + 1)
        return True

    monkeypatch.setattr(engine, "_run_eval_subprocess", fake_eval)
    monkeypatch.setattr(
        engine,
        "compute_candidate_fitness",
        lambda **_kwargs: (0.8, {"accuracy": 0.75}),
    )
    monkeypatch.setattr(engine, "load_canonical_pool", lambda *_args: [])

    winner = engine.run_final_runoff(
        tmp_path,
        object(),
        [0, 1],
        {"baseline_accuracy": 0.0, "per_task_scores": {}},
        Namespace(
            dry_run=False,
            no_canonical_import=False,
            token_cap_per_task=100,
            latency_cap_per_task=10.0,
        ),
        k=1,
    )

    assert launched == [[0, 1]]
    assert imported == [runoff_snapshot]
    assert winner["config_id"] == "cfg"


def test_bound_canonical_snapshot_detects_mutation(tmp_path):
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot"
    state = engine._empty_canonical_state()
    state["units"] = [{"id": "original"}]
    engine._save_canonical_state(source, state)
    engine._save_bound_canonical_snapshot(source, snapshot)

    changed = engine._load_canonical_state(snapshot)
    changed["units"] = [{"id": "mutated"}]
    engine._save_canonical_state(snapshot, changed)

    with pytest.raises(ValueError, match="snapshot manifest digest mismatch"):
        engine._load_bound_canonical_snapshot(snapshot)


def test_final_memory_validation_partial_resume_replays_full_batch(
    tmp_path, monkeypatch
):
    validation_dir = tmp_path / "final_validation"
    (validation_dir / "storage" / "sentinel").mkdir(parents=True)
    _write_valid_result(validation_dir / "tasks" / "1.json", 1)
    calls = []

    def fake_eval(*, tasks_dir, task_indices, **_kwargs):
        calls.append((tasks_dir.name, list(task_indices)))
        if tasks_dir.name == "tasks":
            assert not (validation_dir / "storage" / "sentinel").exists()
        for index in task_indices:
            _write_valid_result(tasks_dir / f"{index + 1}.json", index + 1)
        return True

    monkeypatch.setattr(engine, "_run_eval_subprocess", fake_eval)
    monkeypatch.setattr(engine, "_compile_architecture", lambda *_args: object())
    monkeypatch.setattr(engine, "import_canonical_to_storage", lambda *_args: 0)
    monkeypatch.setattr(engine, "load_canonical_pool", lambda *_args: [])
    monkeypatch.setattr(
        engine,
        "compute_candidate_fitness",
        lambda **_kwargs: (0.9, {"accuracy": 1.0}),
    )
    best = SimpleNamespace(
        config_id="winner",
        architecture={"management": "lightweight"},
        raw_metrics={"accuracy": 0.8},
    )

    engine.run_final_validation(
        tmp_path,
        SimpleNamespace(best=lambda: best),
        DataSplitConfig(final_test_indices=[0, 1]),
        {},
        Namespace(
            dry_run=False,
            no_canonical_import=False,
            token_cap_per_task=100,
            latency_cap_per_task=10.0,
        ),
    )

    assert calls == [("baseline_tasks", [0, 1]), ("tasks", [0, 1])]
    assert (validation_dir / "validation_result.json").is_file()


def test_corrupt_persisted_batch_and_folds_fail_closed(tmp_path):
    split = DataSplitConfig(optimization_indices=[0, 1, 2, 3])
    args = Namespace(batch_size=2, search_batch_seed=42)
    batch_path = tmp_path / "search_batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "indices": [0, 0],
                "seed": 42,
                "n": 2,
                "pool_size": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        engine.load_or_create_search_batch(tmp_path, split, args)

    batch_path.unlink()
    folds_path = tmp_path / "search_folds.json"
    folds_path.write_text(
        json.dumps(
            {
                "folds": [[0, 1], [1, 2]],
                "n_folds": 2,
                "fold_sizes": [2, 2],
                "seed": 42,
                "pool_size": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="current protocol"):
        engine.load_or_create_search_folds(tmp_path, split, args, n_folds=2)


def test_candidate_checkpoint_manifest_prevents_architecture_result_misbinding(
    tmp_path,
):
    candidates_path = tmp_path / "candidates.json"
    manifest_path = tmp_path / "candidates_manifest.json"
    original = [{"candidate_id": 0, "architecture": {"encode": "tip"}}]
    candidates_path.write_text(json.dumps(original), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sha256": engine._candidate_checkpoint_digest(original),
            }
        ),
        encoding="utf-8",
    )
    assert engine._load_bound_candidate_checkpoint(
        candidates_path, manifest_path
    ) == original

    candidates_path.write_text(
        json.dumps(
            [{"candidate_id": 0, "architecture": {"encode": "workflow"}}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        engine._load_bound_candidate_checkpoint(candidates_path, manifest_path)
