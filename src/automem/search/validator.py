"""Validate optimizer candidates against the single public E/S/R/M space.

Two-stage workflow:
  1. `validate(arch)` → (is_valid, [violation strings])
  2. `repair(arch)`   → returns a best-effort fix-then-pass version
                       (used to rescue stray LLM outputs); raises if
                       irreparable (e.g. missing extract_types).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from automem.architecture_space import (
    ARCHITECTURE_SPACE,
    CONSTRAINTS,
    RECOMMENDED_ARCHITECTURE_SPACE,
)
from automem.architecture.models import ArchitectureSpec

# Repair-map: legacy / removed values → closest recommended fallback.
# When the LLM emits a value outside RECOMMENDED, repair() routes it to the
# fallback so the candidate is still runnable instead of being discarded.
RETRIEVAL_REPAIR_MAP: Dict[str, str] = {
    "semantic":     "hybrid",       # semantic dropped in 2026-04-24 audit
    "keyword":      "hybrid",       # keyword soft-deleted from proposer set
    "cbr":          "cbr_rerank",   # cbr-only superseded by cbr_rerank
    "tag":          "hybrid",       # tag dropped (high empty_rate)
    "hybrid_graph": "graph",        # collapsed (use plain graph)
}

@dataclass
class ValidationReport:
    """Result of a single architecture validation."""

    is_valid: bool
    violations: List[str]
    repaired_from: Optional[Dict[str, Any]] = None  # original arch if repair() ran


class ArchitectureValidator:
    """Code-layer architecture validator.

    Both modes use the same public value sets. ``strict`` remains only as a
    compatibility argument for callers migrated from the earlier two-space API.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self._space = RECOMMENDED_ARCHITECTURE_SPACE if strict else ARCHITECTURE_SPACE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(self, arch: Dict[str, Any]) -> ValidationReport:
        """Validate through :class:`ArchitectureSpec`, the sole public schema."""
        try:
            ArchitectureSpec.from_search_dict(arch)
        except (TypeError, ValueError) as exc:
            return ValidationReport(False, [str(exc)])
        return ValidationReport(True, [])

    def repair(self, arch: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort fix for common LLM mistakes (deprecated values, etc.).

        Returns a NEW deep-copied dict (input is never mutated). Output may
        still fail `validate()` if irreparably malformed (e.g. missing
        extract_types entirely) — caller is expected to revalidate.
        """
        if not isinstance(arch, dict):
            raise ValueError(f"arch must be a dict, got {type(arch).__name__}")

        import copy as _copy
        repaired = _copy.deepcopy(arch)
        notes: List[str] = []

        # Retrieval: map deprecated → recommended fallback
        retrieval = repaired.get("retrieval")
        if retrieval not in self._space["retrieval_types"]:
            fallback = RETRIEVAL_REPAIR_MAP.get(retrieval)
            if fallback:
                repaired["retrieval"] = fallback
                notes.append(f"retrieval '{retrieval}' → '{fallback}'")
            else:
                # unknown value — fall through to validate() failure
                pass

        # Normalize the encode subset at the explicit repair boundary:
        # drop unknown values and duplicates deterministically, preserving the
        # multi-Encode selection itself (subsets are inside the public space).
        extract_types = repaired.get("extract_types") or []
        if isinstance(extract_types, list):
            known = [value for value in extract_types if value in self._space["extract_types"]]
            deduped = list(dict.fromkeys(known))
            if deduped != extract_types:
                dropped = [value for value in extract_types if value not in deduped]
                repaired["extract_types"] = deduped
                extract_types = deduped
                notes.append(
                    "extract_types normalized to "
                    f"{deduped} (dropped {dropped})"
                )

        # Complete, unify, and trim routing: the public contract routes every
        # selected type to one common store.
        routing = repaired.get("storage_routing")
        if isinstance(routing, dict) and isinstance(extract_types, list) and extract_types:
            valid_stores = [
                routing[et]
                for et in extract_types
                if routing.get(et) in self._space["storage_types"]
            ]
            if valid_stores:
                # Deterministic: majority store, ties broken by menu order.
                target = max(
                    set(valid_stores),
                    key=lambda s: (
                        valid_stores.count(s),
                        -self._space["storage_types"].index(s),
                    ),
                )
            else:
                # Default to json (lowest cold-start threshold, always valid)
                target = "json"
            for et in extract_types:
                if routing.get(et) != target:
                    previous = routing.get(et)
                    routing[et] = target
                    if previous is None:
                        notes.append(f"storage_routing['{et}'] defaulted to '{target}'")
                    else:
                        notes.append(
                            f"storage_routing['{et}'] unified from '{previous}' to '{target}'"
                        )
            # Trim entries for types not in extract_types
            extra_keys = [k for k in routing if k not in extract_types]
            for k in extra_keys:
                routing.pop(k)
                notes.append(f"removed storage_routing['{k}'] (not in extract_types)")

        # Cross-layer: if retrieval=graph but no graph storage, demote retrieval
        active_storages = (
            set(repaired.get("storage_routing", {}).values())
            if isinstance(repaired.get("storage_routing"), dict)
            else set()
        )
        has_graph = bool(active_storages & set(CONSTRAINTS["graph_storage_types"]))
        if (
            repaired.get("retrieval") in CONSTRAINTS["retrieval_requires_graph_storage"]
            and not has_graph
        ):
            repaired["retrieval"] = "hybrid"
            notes.append(
                "retrieval='graph' requires graph storage; demoted to 'hybrid'"
            )

        # Similarly for management
        if (
            repaired.get("management") in CONSTRAINTS["management_requires_graph_storage"]
            and not has_graph
        ):
            repaired["management"] = "lightweight"
            notes.append(
                "management requires graph storage; demoted to 'lightweight'"
            )

        # Edge-adaptive management is meaningful only with a retriever that
        # consumes weighted SIMILAR edges and reports traversed edges. Preserve
        # the requested retrieval strategy and fall back to content-only
        # consolidation when that feedback contract is unavailable.
        if (
            repaired.get("management") in CONSTRAINTS["management_requires_edge_feedback"]
            and repaired.get("retrieval") not in CONSTRAINTS["edge_feedback_retrieval_types"]
        ):
            repaired["management"] = "json_full"
            notes.append(
                "management requires weighted-edge feedback; demoted to 'json_full'"
            )

        # Repair is an explicit boundary, but its output is still a canonical
        # four-key architecture. Diagnostic notes must not leak into the schema.
        return repaired


__all__ = ["ArchitectureValidator", "ValidationReport"]
