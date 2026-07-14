"""Focused contract tests for the canonical graph management preset."""

from __future__ import annotations

import networkx as nx
import pytest

from automem.architecture.compiler import RuntimeConfig
from automem.architecture_space import (
    ARCHITECTURE_SPACE,
    RECOMMENDED_ARCHITECTURE_SPACE,
    get_valid_managements,
    validate_architecture,
)
from automem.management.ops.edge_weight_optimize import EdgeWeightOptimizeOp
from automem.management.preset_registry import (
    PUBLIC_PRESET_NAMES,
    normalize_preset_name,
    validate_preset_capabilities,
)
from automem.management.presets import get_preset, list_presets
from automem.search.validator import ArchitectureValidator


class _GraphStore:
    """Small graph-store double exposing only the operation's contract."""

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self.save_count = 0

    def neighbors(self, *_args, **_kwargs):
        return []

    def save(self) -> None:
        self.save_count += 1


def _graph_architecture(*, retrieval: str = "graph", management: str = "graph_consolidate"):
    return {
        "extract_types": ["tip"],
        "storage_routing": {"tip": "graph"},
        "retrieval": retrieval,
        "management": management,
    }


def test_public_management_surface_has_exactly_four_canonical_names() -> None:
    expected = ["lightweight", "json_full", "tool_manager", "graph_consolidate"]

    assert list(PUBLIC_PRESET_NAMES) == expected
    assert list_presets() == expected
    assert ARCHITECTURE_SPACE["management_types"] == expected
    assert RECOMMENDED_ARCHITECTURE_SPACE["management_types"] == expected


def test_graph_consolidate_contains_content_and_edge_ops_but_no_shortcuts() -> None:
    config = get_preset("graph_consolidate")
    all_ops = config.post_task_ops + config.periodic_ops + config.on_insert_ops

    assert "edge_stats_update" in config.post_task_ops
    assert "reflection_correction" in config.post_task_ops
    assert "reindex_relations" in config.periodic_ops
    assert "edge_weight_optimize" in config.periodic_ops
    assert "cluster_merge" in config.periodic_ops
    assert "cross_task_generalize" in config.periodic_ops
    assert "shortcut_promotion" not in all_ops
    assert "shortcut_validation" not in all_ops
    assert config.periodic_ops.index("size_capped_prune") < config.periodic_ops.index(
        "reindex_relations"
    )
    assert config.periodic_ops.index("reindex_relations") < config.periodic_ops.index(
        "edge_weight_optimize"
    )
    assert config.periodic_interval == 10


@pytest.mark.parametrize(
    "retired",
    ("skywork_unified", "graph_adaptive", "graph_full", "json_basic", "case_bank"),
)
def test_retired_management_names_are_rejected_everywhere(retired: str) -> None:
    with pytest.raises(ValueError, match="Unknown management preset"):
        normalize_preset_name(retired)
    with pytest.raises(ValueError, match="Unknown management preset"):
        RuntimeConfig.from_dict({"management_preset": retired})

    assert retired not in list_presets()


def test_graph_consolidate_capability_contract_is_enforced() -> None:
    assert validate_preset_capabilities(
        "graph_consolidate",
        storage_types=["graph"],
        retrieval_types=["graph"],
    ) == []
    assert any(
        "requires graph storage" in error
        for error in validate_preset_capabilities(
            "graph_consolidate",
            storage_types=["json"],
            retrieval_types=["graph"],
        )
    )
    assert any(
        "weighted SIMILAR edges" in error
        for error in validate_preset_capabilities(
            "graph_consolidate",
            storage_types=["graph"],
            retrieval_types=["hybrid"],
        )
    )


def test_architecture_validation_rejects_missing_edge_feedback() -> None:
    valid, errors = validate_architecture(_graph_architecture())
    assert valid, errors

    valid, errors = validate_architecture(_graph_architecture(retrieval="hybrid"))
    assert not valid
    assert any("weighted SIMILAR edges" in error for error in errors)

    assert "graph_consolidate" not in get_valid_managements(
        {"tip": "graph"}, retrieval="hybrid"
    )
    assert "graph_consolidate" in get_valid_managements(
        {"tip": "graph"}, retrieval="graph"
    )


def test_search_validator_does_not_repair_retired_management_names() -> None:
    validator = ArchitectureValidator(strict=True)
    repaired = validator.repair(
        _graph_architecture(retrieval="hybrid", management="skywork_unified")
    )

    assert repaired["management"] == "skywork_unified"
    assert not validator.validate(repaired).is_valid


def test_edge_weight_is_rebuilt_from_base_and_does_not_compound() -> None:
    store = _GraphStore()
    store._graph.add_edge(
        "m:a",
        "m:b",
        key="similar",
        edge_type="SIMILAR",
        weight=0.8,
        usage_count=10,
        success_count=9,
    )
    operation = EdgeWeightOptimizeOp(store, config={})

    first = operation.execute({})
    edge = store._graph["m:a"]["m:b"]["similar"]
    assert first.triggered
    assert edge["base_weight"] == pytest.approx(0.8)
    assert edge["feedback_multiplier"] == pytest.approx(1.15)
    assert edge["weight"] == pytest.approx(0.92)
    assert store.save_count == 1

    second = operation.execute({})
    assert not second.triggered
    assert edge["weight"] == pytest.approx(0.92)
    assert store.save_count == 1


def test_low_success_feedback_never_physically_prunes_an_edge() -> None:
    store = _GraphStore()
    store._graph.add_edge(
        "m:a",
        "m:b",
        key="similar",
        edge_type="SIMILAR",
        weight=0.8,
        usage_count=5,
        success_count=0,
    )
    operation = EdgeWeightOptimizeOp(
        store,
        config={"min_usage_for_prune": 0, "prune_below_weight": 1.0},
    )

    result = operation.execute({})

    assert store._graph.has_edge("m:a", "m:b", key="similar")
    edge = store._graph["m:a"]["m:b"]["similar"]
    assert edge["base_weight"] == pytest.approx(0.8)
    assert edge["feedback_multiplier"] == pytest.approx(0.8)
    assert edge["weight"] == pytest.approx(0.64)
    assert result.details["edges_pruned"] == 0
