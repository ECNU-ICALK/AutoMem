import json
from pathlib import Path

import pytest

from automem.architecture_space import ARCHITECTURE_SPACE, RECOMMENDED_ARCHITECTURE_SPACE
from automem.architecture.models import (
    ARCHITECTURE_CHOICES,
    SCHEMA_VERSION,
    SPACE_ID,
    ArchitectureSpec,
    architecture_space_manifest,
)
from automem.architecture.compiler import ArchitectureCompiler, RuntimeConfig
from automem.search.validator import ArchitectureValidator
from automem.providers.modular_memory_provider import ModularMemoryProvider


def _valid_payload():
    return {
        "schema_version": SCHEMA_VERSION,
        "encode": ["tip"],
        "store": "json",
        "retrieve": "hybrid",
        "manage": "lightweight",
    }


def test_round_trip_has_only_public_fields_and_stable_fingerprint():
    payload = _valid_payload()
    spec = ArchitectureSpec.from_dict(payload)

    assert spec.to_dict() == payload
    assert set(spec.to_dict()) == {
        "schema_version",
        "encode",
        "store",
        "retrieve",
        "manage",
    }
    assert ArchitectureSpec.from_dict(json.loads(json.dumps(payload))).fingerprint == (
        spec.fingerprint
    )


def test_encode_accepts_string_sugar_and_canonicalizes_subset_order():
    single = ArchitectureSpec.from_dict({**_valid_payload(), "encode": "tip"})
    assert single.encode == ("tip",)
    assert single.to_dict()["encode"] == ["tip"]
    assert single.fingerprint == ArchitectureSpec.from_dict(_valid_payload()).fingerprint

    shuffled = ArchitectureSpec.from_dict(
        {**_valid_payload(), "encode": ["workflow", "tip", "trajectory"]}
    )
    canonical = ArchitectureSpec.from_dict(
        {**_valid_payload(), "encode": ["tip", "trajectory", "workflow"]}
    )
    assert shuffled.encode == ("tip", "trajectory", "workflow")
    assert shuffled.to_dict()["encode"] == ["tip", "trajectory", "workflow"]
    assert shuffled.fingerprint == canonical.fingerprint


@pytest.mark.parametrize(
    ("encode", "match"),
    [
        ([], "encode must be a string or a non-empty list"),
        (["tip", "tip"], "duplicate encode"),
        (["tip", "fact"], "invalid encode value"),
        ([1], "encode must be a string or a non-empty list"),
    ],
)
def test_encode_rejects_empty_duplicate_and_unknown_subsets(encode, match):
    with pytest.raises((TypeError, ValueError), match=match):
        ArchitectureSpec.from_dict({**_valid_payload(), "encode": encode})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"runtime_options": {}}, "unknown architecture fields"),
        ({"retrieve": "keyword"}, "invalid retrieve"),
        ({"schema_version": "2"}, "unsupported schema_version"),
    ],
)
def test_rejects_unknown_or_invalid_values(mutation, match):
    payload = _valid_payload()
    payload.update(mutation)
    with pytest.raises(ValueError, match=match):
        ArchitectureSpec.from_dict(payload)


def test_rejects_missing_fields_and_non_string_values():
    payload = _valid_payload()
    del payload["manage"]
    with pytest.raises(ValueError, match="missing architecture fields"):
        ArchitectureSpec.from_dict(payload)

    payload = _valid_payload()
    payload["encode"] = 1
    with pytest.raises(TypeError, match="encode must be a string"):
        ArchitectureSpec.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {**_valid_payload(), "retrieve": "graph"},
        {**_valid_payload(), "manage": "graph_consolidate"},
        {
            **_valid_payload(),
            "store": "graph",
            "manage": "graph_consolidate",
        },
    ],
)
def test_rejects_incompatible_graph_selections(payload):
    with pytest.raises(ValueError):
        ArchitectureSpec.from_dict(payload)


def test_accepts_edge_aware_graph_consolidation_selection():
    spec = ArchitectureSpec.from_dict(
        {
            **_valid_payload(),
            "store": "graph",
            "retrieve": "graph",
            "manage": "graph_consolidate",
        }
    )
    assert spec.manage == "graph_consolidate"


def test_manifest_reports_the_subset_encode_space():
    manifest = architecture_space_manifest()

    assert manifest["space_id"] == SPACE_ID == "automem-esrm-v1"
    assert manifest["counts"] == {
        "encode": 5,
        "store": 5,
        "retrieve": 6,
        "manage": 4,
        "encode_subsets": 31,
        "cartesian_total": 3720,
        "compatible_total": 2573,
    }
    # compatible_total = 31 encode subsets x the 83 store/retrieve/manage
    # combinations that survive the graph constraints.
    assert manifest["counts"]["compatible_total"] == 31 * 83
    assert "graph_adaptive" not in manifest["space"]["manage"]


def test_search_adapter_preserves_one_public_four_tuple():
    spec = ArchitectureSpec.from_dict(_valid_payload())

    assert ArchitectureSpec.from_search_dict(spec.to_search_dict()) == spec
    assert spec.to_provider_config("./memory") == {
        "storage_dir": "./memory",
        "storage_type": "json",
        "retriever_type": "hybrid",
        "retriever_config": {},
        "enabled_prompts": ["tip"],
        "management_enabled": True,
        "management_preset": "lightweight",
    }


def test_search_adapter_accepts_multi_encode_with_one_common_store():
    payload = {
        "extract_types": ["tip", "workflow"],
        "storage_routing": {"tip": "json", "workflow": "json"},
        "retrieval": "hybrid",
        "management": "lightweight",
    }
    spec = ArchitectureSpec.from_search_dict(payload)

    assert spec.encode == ("tip", "workflow")
    assert spec.store == "json"
    assert spec.to_search_dict() == payload


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "extract_types": ["tip"],
                "storage_routing": {"tip": "json", "workflow": "json"},
                "retrieval": "hybrid",
                "management": "lightweight",
            },
            "exactly the selected encode types",
        ),
        (
            {
                "extract_types": ["tip", "workflow"],
                "storage_routing": {"tip": "json", "workflow": "vector"},
                "retrieval": "hybrid",
                "management": "lightweight",
            },
            "one common store",
        ),
        (
            {
                "extract_types": [],
                "storage_routing": {},
                "retrieval": "hybrid",
                "management": "lightweight",
            },
            "non-empty list of encode choices",
        ),
        (
            {
                "extract_types": ["tip", "tip"],
                "storage_routing": {"tip": "json"},
                "retrieval": "hybrid",
                "management": "lightweight",
            },
            "duplicate encode",
        ),
    ],
)
def test_search_adapter_rejects_malformed_architectures(payload, match):
    with pytest.raises(ValueError, match=match):
        ArchitectureSpec.from_search_dict(payload)


def test_optimizer_validator_accepts_subsets_and_repairs_mixed_routing():
    validator = ArchitectureValidator()
    subset = {
        "extract_types": ["tip", "workflow"],
        "storage_routing": {"tip": "json", "workflow": "json"},
        "retrieval": "hybrid",
        "management": "lightweight",
    }
    assert validator.validate(subset).is_valid

    mixed = {
        "extract_types": ["tip", "workflow", "trajectory"],
        "storage_routing": {"tip": "json", "workflow": "vector", "trajectory": "json"},
        "retrieval": "hybrid",
        "management": "lightweight",
    }
    report = validator.validate(mixed)
    assert not report.is_valid
    assert any("common store" in violation for violation in report.violations)

    repaired = validator.repair(mixed)
    assert repaired["extract_types"] == ["tip", "workflow", "trajectory"]
    assert repaired["storage_routing"] == {
        "tip": "json",
        "workflow": "json",
        "trajectory": "json",
    }
    assert validator.validate(repaired).is_valid

    noisy = {
        "extract_types": ["tip", "fact", "tip", "workflow"],
        "storage_routing": {"tip": "json"},
        "retrieval": "hybrid",
        "management": "lightweight",
    }
    repaired_noisy = validator.repair(noisy)
    assert repaired_noisy["extract_types"] == ["tip", "workflow"]
    assert repaired_noisy["storage_routing"] == {"tip": "json", "workflow": "json"}
    assert validator.validate(repaired_noisy).is_valid


def test_public_spec_compiles_to_one_store_runtime(tmp_path):
    spec = ArchitectureSpec.from_dict(_valid_payload())
    runtime = ArchitectureCompiler(str(tmp_path)).compile_spec(spec)

    assert runtime.extract_plan["extract_types"] == ["tip"]
    assert runtime.extract_plan["storage_routing"] == {"tip": "json"}
    assert runtime.primary_storage_type == "json"
    assert runtime.additional_stores == {}
    assert runtime.retrieval_type == "hybrid"
    assert runtime.management_preset == "lightweight"


def test_multi_encode_spec_compiles_to_one_store_runtime(tmp_path):
    spec = ArchitectureSpec.from_dict(
        {
            **_valid_payload(),
            "encode": ["tip", "trajectory", "workflow"],
            "store": "hybrid",
            "retrieve": "contrastive",
        }
    )
    runtime = ArchitectureCompiler(str(tmp_path)).compile_spec(spec)

    assert runtime.extract_plan["extract_types"] == ["tip", "trajectory", "workflow"]
    assert runtime.extract_plan["storage_routing"] == {
        "tip": "hybrid",
        "trajectory": "hybrid",
        "workflow": "hybrid",
    }
    assert runtime.primary_storage_type == "hybrid"
    assert runtime.additional_stores == {}
    assert runtime.retrieval_type == "contrastive"


def test_shipped_example_uses_the_strict_public_schema():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "configs" / "example.architecture.json").read_text(encoding="utf-8")
    )

    spec = ArchitectureSpec.from_dict(payload)
    assert spec.to_dict() == payload


def test_runtime_config_rejects_expanded_internal_space():
    with pytest.raises(ValueError, match="one common store"):
        RuntimeConfig(
            extract_plan={
                "extract_types": ["tip", "workflow"],
                "storage_routing": {"tip": "json", "workflow": "vector"},
            },
            primary_storage_type="json",
        )

    with pytest.raises(ValueError, match="unknown runtime config fields"):
        RuntimeConfig.from_dict({"hidden_runtime_toggle": True})

    with pytest.raises(ValueError, match="at least one Encode"):
        ModularMemoryProvider({"enabled_prompts": []})

    with pytest.raises(ValueError, match="duplicate encode"):
        ModularMemoryProvider({"enabled_prompts": ["tip", "tip"]})


def test_legacy_search_api_exposes_the_same_single_space():
    expected = {
        "extract_types": list(ARCHITECTURE_CHOICES["encode"]),
        "storage_types": list(ARCHITECTURE_CHOICES["store"]),
        "retrieval_types": list(ARCHITECTURE_CHOICES["retrieve"]),
        "management_types": list(ARCHITECTURE_CHOICES["manage"]),
    }
    for key, values in expected.items():
        assert ARCHITECTURE_SPACE[key] == values
        assert RECOMMENDED_ARCHITECTURE_SPACE[key] == values
