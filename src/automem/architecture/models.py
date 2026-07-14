"""Stable public model for one AutoMem architecture selection."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

SCHEMA_VERSION = "1"
SPACE_ID = "automem-esrm-v1"

ENCODE_CHOICES: tuple[str, ...] = (
    "tip",
    "insight",
    "trajectory",
    "workflow",
    "shortcut",
)
STORE_CHOICES: tuple[str, ...] = (
    "json",
    "vector",
    "hybrid",
    "graph",
    "llm_graph",
)
RETRIEVE_CHOICES: tuple[str, ...] = (
    "hybrid",
    "contrastive",
    "cbr_rerank",
    "graph",
    "hyde",
    "mmr",
)
MANAGE_CHOICES: tuple[str, ...] = (
    "lightweight",
    "json_full",
    "tool_manager",
    "graph_consolidate",
)

ARCHITECTURE_CHOICES: dict[str, tuple[str, ...]] = {
    "encode": ENCODE_CHOICES,
    "store": STORE_CHOICES,
    "retrieve": RETRIEVE_CHOICES,
    "manage": MANAGE_CHOICES,
}


@dataclass(frozen=True)
class ArchitectureSpec:
    """A strict, versioned selection from AutoMem's public E/S/R/M space.

    ``encode`` selects a non-empty subset of the five extraction types (the
    paper's multi-Encode routes, e.g. ``tip+trajectory+workflow``); ``store``,
    ``retrieve``, and ``manage`` each select exactly one value. All selected
    encode types are persisted to the single selected store.

    Runtime execution policy is intentionally absent. Query planning, context
    composition, and phase refresh behavior are fixed code-level behavior and
    are not searchable architecture dimensions.
    """

    schema_version: str
    encode: tuple[str, ...]
    store: str
    retrieve: str
    manage: str

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "encode",
        "store",
        "retrieve",
        "manage",
    )

    def __post_init__(self) -> None:
        for field_name in ("schema_version", "store", "retrieve", "manage"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"{field_name} must be a string")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

        encode = self.encode
        if isinstance(encode, str):
            encode = (encode,)
        elif isinstance(encode, Sequence) and not isinstance(encode, (bytes, bytearray)):
            encode = tuple(encode)
        else:
            raise TypeError("encode must be a string or a list of strings")
        if not encode or any(type(value) is not str for value in encode):
            raise TypeError("encode must be a string or a non-empty list of strings")
        unknown = [value for value in encode if value not in ENCODE_CHOICES]
        if unknown:
            allowed = ", ".join(ENCODE_CHOICES)
            raise ValueError(
                f"invalid encode value(s) {unknown!r}; expected values from: {allowed}"
            )
        if len(set(encode)) != len(encode):
            duplicates = sorted({value for value in encode if encode.count(value) > 1})
            raise ValueError(f"duplicate encode value(s): {', '.join(duplicates)}")
        # Canonical menu order makes serialization and fingerprints
        # independent of the order the caller listed the subset in.
        object.__setattr__(
            self,
            "encode",
            tuple(sorted(encode, key=ENCODE_CHOICES.index)),
        )

        for field_name in ("store", "retrieve", "manage"):
            value = getattr(self, field_name)
            choices = ARCHITECTURE_CHOICES[field_name]
            if value not in choices:
                allowed = ", ".join(choices)
                raise ValueError(
                    f"invalid {field_name} {value!r}; expected one of: {allowed}"
                )
        graph_stores = {"graph", "llm_graph"}
        if self.retrieve == "graph" and self.store not in graph_stores:
            raise ValueError("graph retrieval requires graph or llm_graph storage")
        if self.manage == "graph_consolidate":
            if self.store not in graph_stores:
                raise ValueError(
                    "graph_consolidate requires graph or llm_graph storage"
                )
            if self.retrieve != "graph":
                raise ValueError(
                    "graph_consolidate requires graph retrieval with weighted "
                    "SIMILAR edges feedback"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical public representation with no hidden fields."""
        return {
            "schema_version": self.schema_version,
            "encode": list(self.encode),
            "store": self.store,
            "retrieve": self.retrieve,
            "manage": self.manage,
        }

    def to_search_dict(self) -> dict[str, Any]:
        """Return the optimizer's internal shape without expanding the space."""
        return {
            "extract_types": list(self.encode),
            "storage_routing": {encode_type: self.store for encode_type in self.encode},
            "retrieval": self.retrieve,
            "management": self.manage,
        }

    def to_provider_config(self, storage_dir: str) -> dict[str, Any]:
        """Build the modular provider configuration for this architecture."""
        if not isinstance(storage_dir, str) or not storage_dir.strip():
            raise ValueError("storage_dir must be a non-empty string")
        return {
            "storage_dir": storage_dir,
            "storage_type": self.store,
            "retriever_type": self.retrieve,
            "retriever_config": {},
            "enabled_prompts": list(self.encode),
            "management_enabled": True,
            "management_preset": self.manage,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureSpec:
        """Build a spec while rejecting both missing and unknown fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("architecture payload must be a mapping")
        supplied = set(payload)
        expected = set(cls._FIELDS)
        unknown = sorted(supplied - expected)
        missing = sorted(expected - supplied)
        if unknown:
            raise ValueError(f"unknown architecture fields: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"missing architecture fields: {', '.join(missing)}")
        return cls(**{field_name: payload[field_name] for field_name in cls._FIELDS})

    @classmethod
    def from_search_dict(cls, payload: Mapping[str, Any]) -> ArchitectureSpec:
        """Validate the optimizer's internal shape against the public 4-tuple."""
        if not isinstance(payload, Mapping):
            raise TypeError("search architecture must be a mapping")
        expected = {"extract_types", "storage_routing", "retrieval", "management"}
        supplied = set(payload)
        if supplied != expected:
            unknown = sorted(supplied - expected)
            missing = sorted(expected - supplied)
            details = []
            if unknown:
                details.append(f"unknown fields: {', '.join(unknown)}")
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            raise ValueError("invalid search architecture; " + "; ".join(details))

        extract_types = payload["extract_types"]
        if (
            not isinstance(extract_types, list)
            or not extract_types
            or any(type(value) is not str for value in extract_types)
        ):
            raise ValueError(
                "extract_types must be a non-empty list of encode choices"
            )

        storage_routing = payload["storage_routing"]
        if not isinstance(storage_routing, Mapping) or set(storage_routing) != set(
            extract_types
        ):
            raise ValueError(
                "storage_routing must map exactly the selected encode types"
            )
        routed_stores = set(storage_routing.values())
        if len(routed_stores) != 1:
            raise ValueError(
                "storage_routing must route every selected encode type to one "
                "common store"
            )

        return cls(
            schema_version=SCHEMA_VERSION,
            encode=tuple(extract_types),
            store=next(iter(routed_stores)),
            retrieve=payload["retrieval"],
            manage=payload["management"],
        )

    @property
    def fingerprint(self) -> str:
        """SHA-256 of canonical JSON; stable across processes and key order."""
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def encode_subsets() -> list[tuple[str, ...]]:
    """All non-empty encode subsets in canonical menu order (2^5 - 1 = 31)."""
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(ENCODE_CHOICES) + 1):
        subsets.extend(itertools.combinations(ENCODE_CHOICES, size))
    return subsets


def architecture_space_manifest() -> dict[str, Any]:
    """Return the sole public search space and its per-layer/total counts.

    ``encode`` counts both the five-type menu and the 31 non-empty subsets an
    architecture may select; ``cartesian_total`` is subsets x store x retrieve
    x manage before compatibility validation.
    """
    counts = {name: len(values) for name, values in ARCHITECTURE_CHOICES.items()}
    subsets = encode_subsets()
    total = len(subsets)
    for name in ("store", "retrieve", "manage"):
        total *= counts[name]
    compatible_total = 0
    for encode in subsets:
        for store in STORE_CHOICES:
            for retrieve in RETRIEVE_CHOICES:
                for manage in MANAGE_CHOICES:
                    try:
                        ArchitectureSpec(
                            schema_version=SCHEMA_VERSION,
                            encode=encode,
                            store=store,
                            retrieve=retrieve,
                            manage=manage,
                        )
                    except ValueError:
                        continue
                    compatible_total += 1
    return {
        "space_id": SPACE_ID,
        "schema_version": SCHEMA_VERSION,
        "space": {name: list(values) for name, values in ARCHITECTURE_CHOICES.items()},
        "counts": {
            **counts,
            "encode_subsets": len(subsets),
            "cartesian_total": total,
            "compatible_total": compatible_total,
        },
    }


__all__ = [
    "ARCHITECTURE_CHOICES",
    "ArchitectureSpec",
    "ENCODE_CHOICES",
    "MANAGE_CHOICES",
    "RETRIEVE_CHOICES",
    "SCHEMA_VERSION",
    "SPACE_ID",
    "STORE_CHOICES",
    "architecture_space_manifest",
    "encode_subsets",
]
