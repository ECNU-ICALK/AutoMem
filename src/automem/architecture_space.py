"""Compatibility helpers for AutoMem's single public E/S/R/M space.

The canonical schema lives in :mod:`automem.architecture.models`.  Search
internals still use the historical ``extract_types``/``storage_routing``
shape; this module constrains that shape to a non-empty subset of the five
encode types routed to exactly one common store, so it matches the public
31 x 5 x 6 x 4 space (paper multi-Encode routes included) without adding
hidden dimensions.
"""

from __future__ import annotations

from typing import Any

from automem.architecture.models import ARCHITECTURE_CHOICES, ArchitectureSpec

ARCHITECTURE_SPACE: dict[str, list[str]] = {
    "extract_types": list(ARCHITECTURE_CHOICES["encode"]),
    "storage_types": list(ARCHITECTURE_CHOICES["store"]),
    "retrieval_types": list(ARCHITECTURE_CHOICES["retrieve"]),
    "management_types": list(ARCHITECTURE_CHOICES["manage"]),
}

# There is no second, broader public space.  Keep the historical name as an
# alias because the optimizer imports it in several places.
RECOMMENDED_ARCHITECTURE_SPACE = ARCHITECTURE_SPACE

RELATION_TYPES: tuple[str, ...] = (
    "ABOUT",
    "MENTIONS",
    "SUPPORTS",
    "CONTRADICTS",
    "SEQUENCE",
    "GENERALIZES",
    "CO_OCCURS",
)

CONSTRAINTS: dict[str, Any] = {
    "extract_conditions": {
        "insight": "failure_only",
        "workflow": "success_only",
    },
    "retrieval_requires_graph_storage": ["graph"],
    "management_requires_graph_storage": ["graph_consolidate"],
    "management_requires_edge_feedback": ["graph_consolidate"],
    "edge_feedback_retrieval_types": ["graph"],
    "graph_storage_types": ["graph", "llm_graph"],
}


def validate_architecture(
    config: dict[str, Any],
    task_outcome: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate the optimizer representation of one public architecture."""

    if not isinstance(config, dict):
        return False, ["architecture must be a dict"]

    errors: list[str] = []
    core = {key: value for key, value in config.items() if key != "relation_types"}
    try:
        spec = ArchitectureSpec.from_search_dict(core)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        spec = None

    relation_types = config.get("relation_types")
    if relation_types is not None:
        if not isinstance(relation_types, list):
            errors.append("relation_types must be a list if provided")
        else:
            invalid = [value for value in relation_types if value not in RELATION_TYPES]
            if invalid:
                errors.append(
                    f"invalid relation type(s): {invalid}; valid options: {list(RELATION_TYPES)}"
                )

    if task_outcome not in (None, "success", "failure"):
        errors.append("task_outcome must be 'success', 'failure', or None")
    elif spec is not None and task_outcome is not None:
        for encode_type in spec.encode:
            condition = CONSTRAINTS["extract_conditions"].get(encode_type)
            if task_outcome == "success" and condition == "failure_only":
                errors.append(
                    f"encode '{encode_type}' is restricted to failure outcomes"
                )
            if task_outcome == "failure" and condition == "success_only":
                errors.append(
                    f"encode '{encode_type}' is restricted to success outcomes"
                )

    return not errors, errors


def get_valid_retrievals(storage_routing: dict[str, str]) -> list[str]:
    """Return retrieval choices compatible with the selected store."""

    active_storages = set(storage_routing.values())
    has_graph = bool(active_storages & set(CONSTRAINTS["graph_storage_types"]))
    return [
        retrieval
        for retrieval in ARCHITECTURE_SPACE["retrieval_types"]
        if retrieval not in CONSTRAINTS["retrieval_requires_graph_storage"] or has_graph
    ]


def get_valid_managements(
    storage_routing: dict[str, str],
    retrieval: str | None = None,
) -> list[str]:
    """Return management choices compatible with the store and retriever."""

    active_storages = set(storage_routing.values())
    has_graph = bool(active_storages & set(CONSTRAINTS["graph_storage_types"]))
    valid: list[str] = []
    for management in ARCHITECTURE_SPACE["management_types"]:
        if management in CONSTRAINTS["management_requires_graph_storage"] and not has_graph:
            continue
        if (
            retrieval is not None
            and management in CONSTRAINTS["management_requires_edge_feedback"]
            and retrieval not in CONSTRAINTS["edge_feedback_retrieval_types"]
        ):
            continue
        valid.append(management)
    return valid


def describe_space() -> str:
    """Return a compact prompt-safe description of the public search space."""

    lines = [
        "AutoMem architecture space (extract_types selects a non-empty subset; "
        "every other layer selects exactly one value):"
    ]
    for key in (
        "extract_types",
        "storage_types",
        "retrieval_types",
        "management_types",
    ):
        lines.append(f"- {key}: {', '.join(ARCHITECTURE_SPACE[key])}")
    lines.append("- all selected extract types are routed to the one selected store")
    lines.append("- graph retrieval requires graph or llm_graph storage")
    lines.append("- graph_consolidate additionally requires graph retrieval")
    return "\n".join(lines)


__all__ = [
    "ARCHITECTURE_SPACE",
    "CONSTRAINTS",
    "RECOMMENDED_ARCHITECTURE_SPACE",
    "RELATION_TYPES",
    "describe_space",
    "get_valid_managements",
    "get_valid_retrievals",
    "validate_architecture",
]
