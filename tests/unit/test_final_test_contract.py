from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

from automem.data_split import (
    DataSplitConfig,
    create_default_split,
    create_level_aware_split,
)
from automem.search import engine


def test_default_split_honors_all_four_requested_sizes():
    split = create_default_split(
        total_tasks=30,
        profile_n=2,
        optimization_n=5,
        validation_n=3,
        final_test_n=4,
    )

    assert list(map(len, split.to_dict().values())) == [2, 5, 3, 4]
    assert split.final_test_indices == [10, 11, 12, 13]


def test_level_aware_split_honors_exact_final_test_size():
    tasks = [{"Level": level} for level in (1, 2, 3) for _ in range(10)]

    split = create_level_aware_split(
        tasks,
        profile_n=2,
        optimization_n=8,
        validation_n=4,
        final_test_n=5,
    )

    assert [
        len(split.profile_indices),
        len(split.optimization_indices),
        len(split.validation_indices),
        len(split.final_test_indices),
    ] == [2, 8, 4, 5]
    assert split.validate() == (True, [])


def test_final_validation_uses_only_held_out_final_test_indices(tmp_path, monkeypatch):
    best = SimpleNamespace(
        config_id="winner",
        architecture={"encode": "tip", "store": "json"},
        raw_metrics={"accuracy": 0.75},
    )
    pareto = SimpleNamespace(best=lambda: best)
    split = DataSplitConfig(
        profile_indices=[0],
        optimization_indices=[1, 2],
        validation_indices=[3, 4],
        final_test_indices=[7],
    )
    captured = {}

    monkeypatch.setattr(engine, "_compile_architecture", lambda *_args: object())
    monkeypatch.setattr(engine, "load_canonical_pool", lambda *_args: [])

    def synthetic(_architecture, **kwargs):
        captured.update(kwargs)
        return 0.25, {"accuracy": 0.5}

    monkeypatch.setattr(engine, "_synthetic_candidate_metrics", synthetic)
    args = Namespace(dry_run=True, no_canonical_import=False, max_rounds=1)

    engine.run_final_validation(
        tmp_path,
        pareto,
        split,
        {"baseline_accuracy": 0.0, "per_task_scores": {}},
        args,
    )

    result = json.loads(
        (tmp_path / "final_validation" / "validation_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert captured["total_tasks"] == 1
    assert captured["baseline_accuracy"] == 1.0
    assert result["evaluation_split"] == "final_test"
    assert result["final_test_indices"] == [7]
    assert result["baseline_accuracy"] == 1.0
    assert "validation_metrics" not in result
