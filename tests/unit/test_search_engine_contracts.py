from __future__ import annotations

import random
import sys
from argparse import Namespace

import pytest

from automem.architecture.models import ArchitectureSpec
from automem.runtime import RuntimePolicy
from automem.search import engine
from automem.search.protocol import ProtocolConfig


def _signature_args(search_prompt: str | None = None, judge_model: str = "judge-a"):
    return Namespace(
        search_prompt=search_prompt,
        judge_model=judge_model,
        diagnosis_model="diagnosis-a",
        dry_run=False,
    )


def _assert_public_architecture_compiles(architecture, storage_dir):
    spec = ArchitectureSpec.from_search_dict(architecture)
    runtime = engine._compile_architecture(architecture, str(storage_dir))

    assert runtime is not None
    assert runtime.extract_plan["extract_types"] == list(spec.encode)
    assert runtime.extract_plan["storage_routing"] == {
        encode_type: spec.store for encode_type in spec.encode
    }
    assert runtime.primary_storage_type == spec.store
    assert runtime.additional_stores == {}


def test_every_builtin_search_candidate_compiles_in_public_space(tmp_path):
    candidates = [("warmup", engine.WARMUP_ARCHITECTURE)]
    candidates.extend(
        (f"seed-{index}", candidate["architecture"])
        for index, candidate in enumerate(engine._round1_hardcoded_seeds())
    )
    candidates.extend(
        (f"fallback-{index}", candidate["architecture"])
        for index, candidate in enumerate(engine._fallback_candidates(1, 7))
    )
    candidates.extend(
        (f"random-{index}", candidate["architecture"])
        for index, candidate in enumerate(
            engine._sample_random_candidates(2, 64, random.Random(90210))
        )
    )

    for name, architecture in candidates:
        _assert_public_architecture_compiles(architecture, tmp_path / name)


def test_builtin_candidates_exercise_the_multi_encode_dimension():
    seed_sizes = {
        len(candidate["architecture"]["extract_types"])
        for candidate in engine._round1_hardcoded_seeds()
    }
    assert any(size > 1 for size in seed_sizes)

    fallback_sizes = {
        len(candidate["architecture"]["extract_types"])
        for candidate in engine._fallback_candidates(1, 3)
    }
    assert any(size > 1 for size in fallback_sizes)
    assert any(size == 1 for size in fallback_sizes)

    random_sizes = {
        len(candidate["architecture"]["extract_types"])
        for candidate in engine._sample_random_candidates(2, 64, random.Random(90210))
    }
    assert any(size > 1 for size in random_sizes)
    assert any(size == 1 for size in random_sizes)


def test_candidate_validation_accepts_subsets_and_rejects_mixed_stores(tmp_path):
    subset = {
        "extract_types": ["tip", "workflow"],
        "storage_routing": {"tip": "json", "workflow": "json"},
        "retrieval": "hybrid",
        "management": "lightweight",
    }
    valid, reason = engine._validate_candidate({"architecture": subset})
    assert valid, reason
    _assert_public_architecture_compiles(subset, tmp_path / "subset")

    mixed = {
        "extract_types": ["tip", "workflow"],
        "storage_routing": {"tip": "json", "workflow": "vector"},
        "retrieval": "hybrid",
        "management": "lightweight",
    }
    valid, reason = engine._validate_candidate({"architecture": mixed})
    assert not valid
    assert "common store" in reason
    assert engine._compile_architecture(mixed, str(tmp_path)) is None


def test_search_and_diagnosis_prompts_expose_the_subset_encode_schema():
    search_prompt = engine.SEARCH_PROMPT.read_text(encoding="utf-8")
    diagnosis_prompt = engine.prompt_path(
        "meta", "layer_diagnosis_fixed.txt"
    ).read_text(encoding="utf-8")
    differential_prompt = engine.prompt_path(
        "meta", "differential_diagnosis.txt"
    ).read_text(encoding="utf-8")

    assert "non-empty subset" in search_prompt
    assert "pick exactly one" not in search_prompt
    assert '"extract_types": ["tip", "shortcut"]' in search_prompt
    assert "SAME backend" in search_prompt
    assert "non-empty subset of these five" in diagnosis_prompt
    # Guard against ANY single-encode phrasing resurfacing (the 2026-07-14
    # audit found "exactly one value" / "Exactly one valid extract type"
    # leftovers that the narrower phrase check above missed).
    assert "exactly one" not in diagnosis_prompt.lower()
    assert "[tip] → [tip, trajectory]" in diagnosis_prompt
    assert "ONE common backend" in diagnosis_prompt
    assert "{tip,workflow}" not in differential_prompt
    with pytest.raises(FileNotFoundError):
        engine.prompt_path("meta", "architecture_selection.txt")
    with pytest.raises(FileNotFoundError):
        engine.prompt_path("meta", "layer_diagnosis.txt")


def test_protocol_digest_changes_with_runtime_policy(monkeypatch):
    baseline = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), _signature_args()
    )["digest"]
    monkeypatch.setattr(
        engine,
        "DEFAULT_RUNTIME_POLICY",
        RuntimePolicy(max_injected_units=2),
    )

    changed = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), _signature_args()
    )["digest"]

    assert changed != baseline


@pytest.mark.parametrize(
    ("owner", "attribute"),
    [
        (engine.QueryPlanner, "_PROMPT"),
        (engine.MemoryContextComposer, "_SYSTEM"),
    ],
)
def test_protocol_digest_changes_with_runtime_prompt(monkeypatch, owner, attribute):
    baseline = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), _signature_args()
    )["digest"]
    monkeypatch.setattr(owner, attribute, getattr(owner, attribute) + " changed")

    changed = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), _signature_args()
    )["digest"]

    assert changed != baseline


def test_protocol_digest_changes_with_actual_judge_model():
    first = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), _signature_args(judge_model="judge-a")
    )
    second = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), _signature_args(judge_model="judge-b")
    )

    assert first["judge_model"] == "judge-a"
    assert second["judge_model"] == "judge-b"
    assert first["digest"] != second["digest"]


def test_protocol_digest_changes_with_actual_search_prompt_bytes(tmp_path):
    prompt = tmp_path / "search.txt"
    prompt.write_text("first search behavior", encoding="utf-8")
    args = _signature_args(str(prompt))
    first = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), args
    )

    prompt.write_text("second search behavior", encoding="utf-8")
    second = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), args
    )

    assert first["runtime_info"]["search_prompt_sha"] != second["runtime_info"][
        "search_prompt_sha"
    ]
    assert first["digest"] != second["digest"]


def test_protocol_digest_separates_dry_run_from_online_execution():
    online_args = _signature_args()
    dry_args = _signature_args()
    dry_args.dry_run = True

    online = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), online_args
    )
    dry = engine._compute_eval_protocol_signature(
        "task-model", ProtocolConfig.resolve(None), dry_args
    )

    assert online["execution_mode"] == "online"
    assert dry["execution_mode"] == "dry_run"
    assert online["digest"] != dry["digest"]


def test_synthetic_metrics_are_deterministic():
    kwargs = {
        "architecture": engine.WARMUP_ARCHITECTURE,
        "round_id": 3,
        "candidate_id": 1,
        "total_tasks": 12,
        "baseline_accuracy": 0.4,
    }

    assert engine._synthetic_candidate_metrics(**kwargs) == (
        engine._synthetic_candidate_metrics(**kwargs)
    )


def test_fixed_protocol_runoff_ignores_optional_final_validation(monkeypatch, tmp_path):
    captured = {}

    def fake_runoff(run_dir, pareto, indices, baseline, args, k):
        captured.update(k=k, final_validation=args.final_validation)
        return {"config_id": "winner"}

    monkeypatch.setattr(engine, "run_final_runoff", fake_runoff)
    args = Namespace(
        _protocol=ProtocolConfig.resolve(None),
        final_validation=False,
    )

    winner = engine.run_protocol_runoff(tmp_path, object(), [1, 2], {}, args)

    assert winner == {"config_id": "winner"}
    assert captured == {"k": 2, "final_validation": False}


def test_dry_run_is_offline_and_executes_mandatory_runoff(monkeypatch, tmp_path):
    infile = tmp_path / "tasks.jsonl"
    infile.write_text(
        "".join(
            '{"task_id":"t%d","Question":"question %d"}\n' % (index, index)
            for index in range(5)
        ),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run touched an online evaluation path")

    monkeypatch.setattr(engine, "load_model", forbidden)
    monkeypatch.setattr(engine, "load_diagnosis_model", forbidden)
    monkeypatch.setattr(engine, "_start_eval_subprocess", forbidden)
    monkeypatch.setattr(engine, "_ensure_batch_complete", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "automem-search",
            "--run_name",
            "offline",
            "--output_dir",
            str(tmp_path / "runs"),
            "--infile",
            str(infile),
            "--max_rounds",
            "1",
            "--num_candidates",
            "3",
            "--warmup_n",
            "1",
            "--search_n",
            "2",
            "--batch_size",
            "1",
            "--validation_n",
            "1",
            "--test_n",
            "1",
            "--dry_run",
            "--no_ledger",
        ],
    )

    engine.main()

    run_dir = tmp_path / "runs" / "offline"
    assert (run_dir / "final_runoff" / "runoff_result.json").is_file()
    assert not (run_dir / "final_validation").exists()
