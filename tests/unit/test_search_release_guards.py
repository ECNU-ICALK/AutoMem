from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from automem.data_split import DataSplitConfig
from automem.search import engine


VALID_TASK_IDENTITY = "0" * 64


def _write_result(path: Path, item_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_id": f"task-{item_index}",
                "item_index": item_index,
                "task_score": 1.0,
                "status": "success",
                "judge_unjudged": False,
                "task_identity": VALID_TASK_IDENTITY,
            }
        ),
        encoding="utf-8",
    )


def test_new_run_refuses_nonempty_existing_directory(tmp_path):
    run_dir = tmp_path / "runs" / "same-name"
    run_dir.mkdir(parents=True)
    (run_dir / "baseline_done.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        engine.setup_run_dir(
            Namespace(
                run_name="same-name",
                output_dir=str(tmp_path / "runs"),
                resume=False,
            )
        )


def test_resume_requires_an_existing_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cannot resume"):
        engine.setup_run_dir(
            Namespace(
                run_name="missing",
                output_dir=str(tmp_path / "runs"),
                resume=True,
            )
        )


def test_exact_result_guard_rejects_same_count_for_wrong_indices(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _write_result(tasks_dir / "1.json", 1)
    _write_result(tasks_dir / "3.json", 3)

    with pytest.raises(RuntimeError, match=r"missing=\[1\].*extras=\[2\]"):
        engine._require_exact_task_results(tasks_dir, [0, 1], "test stage")


def test_exact_result_guard_rejects_filename_index_mismatch(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _write_result(tasks_dir / "1.json", 1)
    _write_result(tasks_dir / "copy.json", 1)

    with pytest.raises(RuntimeError, match=r"invalid_files=\['copy.json'\]"):
        engine._require_exact_task_results(tasks_dir, [0], "test stage")


def test_final_runoff_fails_when_runner_fails(tmp_path, monkeypatch):
    entry = SimpleNamespace(
        config_id="candidate-a",
        architecture={"encode": "tip", "store": "json"},
    )
    pareto = SimpleNamespace(top_k=lambda _k: [entry])

    monkeypatch.setattr(engine, "_compile_architecture", lambda *_args: object())
    monkeypatch.setattr(engine, "_run_eval_subprocess", lambda **_kwargs: False)

    with pytest.raises(RuntimeError, match="Final runoff subprocess failed"):
        engine.run_final_runoff(
            tmp_path,
            pareto,
            [0, 1],
            {"baseline_accuracy": 0.0},
            Namespace(dry_run=False, no_canonical_import=True),
            k=2,
        )

    assert not (tmp_path / "final_runoff" / "runoff_result.json").exists()


def test_final_validation_fails_when_memory_runner_fails(tmp_path, monkeypatch):
    best = SimpleNamespace(
        config_id="winner",
        architecture={"encode": "tip", "store": "json"},
        raw_metrics={"accuracy": 0.5},
    )
    pareto = SimpleNamespace(best=lambda: best)
    split = DataSplitConfig(
        profile_indices=[0],
        optimization_indices=[1],
        validation_indices=[2],
        final_test_indices=[7],
    )
    calls = 0

    def fake_eval(*, tasks_dir, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            _write_result(tasks_dir / "8.json", 8)
            return True
        return False

    monkeypatch.setattr(engine, "_compile_architecture", lambda *_args: object())
    monkeypatch.setattr(engine, "_run_eval_subprocess", fake_eval)

    with pytest.raises(RuntimeError, match="memory candidate subprocess failed"):
        engine.run_final_validation(
            tmp_path,
            pareto,
            split,
            {},
            Namespace(dry_run=False, no_canonical_import=True),
        )

    assert not (tmp_path / "final_validation" / "validation_result.json").exists()


def test_core_dependencies_cover_default_embedding_runtime():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    project_root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = metadata["project"]["dependencies"]

    assert any(requirement.startswith("sentence-transformers") for requirement in requirements)
    assert any(requirement.startswith("filelock") for requirement in requirements)

    benchmark_requirements = metadata["project"]["optional-dependencies"]["benchmarks"]
    assert any(requirement.startswith("xlrd") for requirement in benchmark_requirements)
