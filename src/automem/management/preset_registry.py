"""Canonical management preset names and cross-layer capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Dict, List


@dataclass(frozen=True)
class PresetCapabilities:
    """Runtime capabilities required by a management preset."""

    requires_graph_storage: bool = False
    requires_weighted_edge_feedback: bool = False


PUBLIC_PRESET_NAMES = (
    "lightweight",
    "json_full",
    "tool_manager",
    "graph_consolidate",
)

PRESET_CAPABILITIES: Dict[str, PresetCapabilities] = {
    "lightweight": PresetCapabilities(),
    "json_full": PresetCapabilities(),
    "tool_manager": PresetCapabilities(),
    "graph_consolidate": PresetCapabilities(
        requires_graph_storage=True,
        requires_weighted_edge_feedback=True,
    ),
}

GRAPH_STORAGE_TYPES = frozenset({"graph", "llm_graph"})
WEIGHTED_EDGE_FEEDBACK_RETRIEVERS = frozenset({"graph"})


def normalize_preset_name(name: str) -> str:
    """Return a canonical preset name or raise ``ValueError``."""

    normalized = str(name or "").strip().lower()
    if normalized in PRESET_CAPABILITIES:
        return normalized

    valid = ", ".join(PUBLIC_PRESET_NAMES)
    raise ValueError(f"Unknown management preset '{normalized}'; valid presets: {valid}.")


def get_preset_capabilities(name: str) -> PresetCapabilities:
    """Return capability requirements for a canonical preset."""

    canonical = normalize_preset_name(name)
    return PRESET_CAPABILITIES[canonical]


def validate_preset_capabilities(
    name: str,
    *,
    storage_types: Collection[str],
    retrieval_types: Collection[str],
) -> List[str]:
    """Return cross-layer capability violations for ``name``.

    Edge adaptation is enabled only when the resolved retriever both traverses
    weighted memory edges and reports their usage.  ``graph`` is currently the
    only retriever with that contract; ``hybrid_graph`` does neither for
    memory-to-memory ``SIMILAR`` edges.
    """

    canonical = normalize_preset_name(name)
    capabilities = PRESET_CAPABILITIES[canonical]
    storages = {str(value).strip().lower() for value in storage_types}
    retrievers = {str(value).strip().lower() for value in retrieval_types}
    errors: List[str] = []

    if capabilities.requires_graph_storage and not (storages & GRAPH_STORAGE_TYPES):
        errors.append(
            f"Management preset '{canonical}' requires graph storage "
            f"({sorted(GRAPH_STORAGE_TYPES)})."
        )
    if (
        capabilities.requires_weighted_edge_feedback
        and not (retrievers & WEIGHTED_EDGE_FEEDBACK_RETRIEVERS)
    ):
        errors.append(
            f"Management preset '{canonical}' requires a retriever that consumes "
            "weighted SIMILAR edges and emits edge-usage traces "
            f"({sorted(WEIGHTED_EDGE_FEEDBACK_RETRIEVERS)})."
        )

    return errors


__all__ = [
    "GRAPH_STORAGE_TYPES",
    "PRESET_CAPABILITIES",
    "PUBLIC_PRESET_NAMES",
    "PresetCapabilities",
    "WEIGHTED_EDGE_FEEDBACK_RETRIEVERS",
    "get_preset_capabilities",
    "normalize_preset_name",
    "validate_preset_capabilities",
]
