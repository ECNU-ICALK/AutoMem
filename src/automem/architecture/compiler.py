"""
Architecture Compiler — translates an ArchitectureDecision into a RuntimeConfig.

The compiler is the bridge between the prompt-driven optimization loop (which
produces ArchitectureDecision objects) and the eval subprocess (which needs a
structured, serializable provider configuration).

Responsibilities:
    1. Validate the decision against ARCHITECTURE_SPACE and CONSTRAINTS.
    2. Resolve cross-layer dependencies (e.g. graph retrieval needs graph storage).
    3. Build a structured RuntimeConfig consumed by the provider factory.
    4. Explicitly record any unsupported features rather than silently ignoring them.

Usage:
    from automem.architecture import ArchitectureCompiler

    compiler = ArchitectureCompiler(base_storage_dir="./storage/round_3")
    runtime_cfg = compiler.compile(decision)
    # runtime_cfg.to_dict() -> JSON passed to the benchmark subprocess
    # runtime_cfg.unsupported_features -> list of warnings
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from automem.architecture_space import ARCHITECTURE_SPACE, CONSTRAINTS, validate_architecture
from automem.architecture.models import ArchitectureSpec
from automem.contracts import ArchitectureDecision, ManagementPlan, RetrievalPlan
from automem.management.preset_registry import normalize_preset_name
from automem.management.presets import get_preset

logger = logging.getLogger(__name__)

RUNTIME_CONFIG_SCHEMA_VERSION = "1"


class CompilationError(Exception):
    """Raised when an ArchitectureDecision cannot be compiled to a valid runtime config."""
    pass


# Valid management operation names (from pipeline registry)
# 2026-05-13: dynamic_discard / case_rewrite removed — see pipeline.py.
VALID_MANAGEMENT_OPS = {
    "cluster_merge", "trajectory_to_workflow", "cross_task_generalize",
    "reindex_relations", "signature_dedup", "semantic_dedup",
    "cross_type_dedup", "conflict_detection", "penalize_on_failure",
    "boost_on_success", "reflection_correction",
    "access_stats_update", "time_decay", "score_based_prune",
    "quality_curation", "utility_audit", "size_capped_prune",
    "llm_conflict_resolve", "shortcut_promotion", "shortcut_validation",
    # G1 (2026-07-11): edge-level adaptive feedback ops (graph stores only).
    "edge_stats_update", "edge_weight_optimize",
}

# Valid rerank values (currently only "none" is fully supported)
VALID_RERANK = {"none"}


@dataclass
class RuntimeConfig:
    """Fully resolved runtime configuration for one evaluation round.

    This is the single source of truth passed to the eval script.
    No silent ignoring — every field is either set or explicitly unsupported.
    """

    # Extract layer
    extract_plan: Dict[str, Any] = field(default_factory=dict)
    # {"extract_types": [...], "storage_routing": {...}, "relation_types": [...]}

    # Storage layer
    primary_storage_type: str = "json"
    storage_dir: str = ""
    additional_stores: Dict[str, str] = field(default_factory=dict)
    # {"vector": "/path/to/store_vector", "graph": "/path/to/store_graph"}

    # Retrieval layer
    retrieval_type: str = "hybrid"
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    # top_k, graph_hop, type_quota, etc.

    # Management layer
    management_preset: str = "lightweight"
    management_config: Dict[str, Any] = field(default_factory=dict)
    # enabled_ops, intensity, post_task_budget, etc.

    # Compilation metadata
    unsupported_features: List[str] = field(default_factory=list)
    # Features requested by ArchitectureDecision but not yet implemented

    def __post_init__(self) -> None:
        if not self.extract_plan:
            return
        candidate = {
            "extract_types": self.extract_plan.get("extract_types", []),
            "storage_routing": self.extract_plan.get("storage_routing", {}),
            "retrieval": self.retrieval_type,
            "management": self.management_preset,
        }
        if self.extract_plan.get("relation_types") is not None:
            candidate["relation_types"] = self.extract_plan["relation_types"]
        is_valid, errors = validate_architecture(candidate)
        if not is_valid:
            raise ValueError(f"invalid runtime architecture: {'; '.join(errors)}")

        routed_stores = set(self.extract_plan["storage_routing"].values())
        if len(routed_stores) != 1:
            raise ValueError(
                "storage_routing must route every selected encode type to one "
                "common store"
            )
        if self.primary_storage_type != next(iter(routed_stores)):
            raise ValueError(
                "primary_storage_type must equal the architecture's selected store"
            )
        if self.additional_stores:
            raise ValueError(
                "additional_stores are not supported by the public E/S/R/M space"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime_schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
            "extract_plan": dict(self.extract_plan),
            "primary_storage_type": self.primary_storage_type,
            "storage_dir": self.storage_dir,
            "additional_stores": dict(self.additional_stores),
            "retrieval_type": self.retrieval_type,
            "retrieval_config": dict(self.retrieval_config),
            "management_preset": self.management_preset,
            "management_config": dict(self.management_config),
            "unsupported_features": list(self.unsupported_features),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuntimeConfig":
        if d is None:
            return cls()
        allowed = set(cls().to_dict())
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"unknown runtime config fields: {unknown}")
        schema_version = str(d.get("runtime_schema_version", RUNTIME_CONFIG_SCHEMA_VERSION))
        if schema_version != RUNTIME_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported runtime config schema version: {schema_version}"
            )
        management_preset = normalize_preset_name(
            d.get("management_preset", "lightweight")
        )
        return cls(
            extract_plan=d.get("extract_plan", {}),
            primary_storage_type=d.get("primary_storage_type", "json"),
            storage_dir=d.get("storage_dir", ""),
            additional_stores=d.get("additional_stores", {}),
            retrieval_type=d.get("retrieval_type", "hybrid"),
            retrieval_config=d.get("retrieval_config", {}),
            management_preset=management_preset,
            management_config=d.get("management_config", {}),
            unsupported_features=d.get("unsupported_features", []),
        )


class ArchitectureCompiler:
    """Compiles ArchitectureDecision -> RuntimeConfig.

    Validates all layers and produces a structured configuration that can be
    passed to the eval subprocess. No silent ignoring: unsupported
    features are explicitly listed.
    """

    VALID_STORAGE_TYPES = set(ARCHITECTURE_SPACE.get("storage_types", []))

    def __init__(self, base_storage_dir: str = "./storage"):
        self.base_storage_dir = base_storage_dir

    def compile_spec(self, spec: ArchitectureSpec) -> RuntimeConfig:
        """Compile the canonical public architecture without hidden choices."""
        if not isinstance(spec, ArchitectureSpec):
            raise TypeError("spec must be an ArchitectureSpec")
        decision = ArchitectureDecision(
            enabled_memory_types=list(spec.encode),
            storage_routing={encode_type: spec.store for encode_type in spec.encode},
            retrieval_plan=RetrievalPlan(
                primary_routes=[spec.retrieve],
                top_k=4,
                graph_hop=1 if spec.retrieve == "graph" else 0,
            ),
            management_plan=ManagementPlan(preset=spec.manage),
        )
        return self.compile(decision)

    def _normalize_storage_type(self, raw: str) -> str:
        """Validate storage type name. Falls back to 'json' for unknown names."""
        normalized = raw.lower().strip()
        if normalized in self.VALID_STORAGE_TYPES:
            return normalized
        logger.warning(
            f"Unknown storage type '{raw}', falling back to 'json'. "
            f"Valid types: {sorted(self.VALID_STORAGE_TYPES)}"
        )
        return "json"

    def compile(self, decision: ArchitectureDecision) -> RuntimeConfig:
        """Compile an ArchitectureDecision into a RuntimeConfig.

        Raises CompilationError if the decision is fundamentally invalid.
        Lists unsupported features in RuntimeConfig.unsupported_features.
        """
        # Step 1: Extract plan and normalize storage types
        extract_plan = decision.to_extract_plan()

        # Normalize storage routing aliases (LLM may hallucinate names like "canonical_json")
        raw_routing = extract_plan.get("storage_routing", {})
        normalized_routing = {
            k: self._normalize_storage_type(v) for k, v in raw_routing.items()
        }
        extract_plan["storage_routing"] = normalized_routing

        # Build config for validation
        storage_types = set(normalized_routing.values())
        primary_storage = self._pick_primary_storage(storage_types)

        retrieval_type = self._resolve_retrieval(decision.retrieval_plan, storage_types)
        management_preset = self._resolve_management(decision.management_plan, storage_types)

        valid_config = {
            "extract_types": extract_plan.get("extract_types", []),
            "storage_routing": extract_plan.get("storage_routing", {}),
            "retrieval": retrieval_type,
            "management": management_preset,
        }
        if extract_plan.get("relation_types"):
            valid_config["relation_types"] = extract_plan["relation_types"]

        is_valid, errors = validate_architecture(valid_config)
        if not is_valid:
            raise CompilationError(
                f"Architecture validation failed: {'; '.join(errors)}"
            )

        # Step 2: Build RuntimeConfig
        unsupported: List[str] = []

        # Storage setup — use type-based directory names so different storage
        # types never collide (critical for persistent/shared storage mode).
        storage_dir = os.path.join(self.base_storage_dir, f"store_{primary_storage}")
        additional_stores: Dict[str, str] = {}
        for _mem_type, store_type in extract_plan.get("storage_routing", {}).items():
            if store_type != primary_storage:
                store_path = os.path.join(self.base_storage_dir, f"store_{store_type}")
                additional_stores[store_type] = store_path

        # Retrieval config
        rp = decision.retrieval_plan
        retrieval_config = {
            "top_k": rp.top_k,
            "graph_hop": rp.graph_hop,
            "type_quota": dict(rp.type_quota) if rp.type_quota else {},
            "rerank": rp.rerank,
            "post_retrieval": rp.post_retrieval,
            "memory_token_budget": rp.memory_token_budget,
            "contradiction_confirm_threshold": rp.contradiction_confirm_threshold,
            "contradiction_suspect_threshold": rp.contradiction_suspect_threshold,
            "gate_threshold": rp.gate_threshold,
        }
        # graph_hop wiring (2026-07-11): GraphRetriever reads `max_hops`, and
        # in the (always-on) multi-store path per-retriever settings only
        # reach sub-retrievers when nested under `<type>_config`. The flat
        # graph_hop key above never reached any retriever, making this
        # searched dimension a silent no-op (candidates differing only in
        # graph_hop were the same architecture).
        if rp.graph_hop:
            retrieval_config["graph_config"] = {"max_hops": rp.graph_hop}
            retrieval_config["hybrid_graph_config"] = {"graph_max_hops": rp.graph_hop}

        # Build per-store retriever_map for MultiStoreRetriever.
        # Default mapping (json→keyword, vector→semantic, hybrid→hybrid,
        # graph→graph, llm_graph→hybrid_graph) is sensible for most cases.
        # Override the primary store's retriever with primary_routes[0].
        if additional_stores and rp.primary_routes:
            from automem.retrieval.multi_store_retriever import DEFAULT_RETRIEVER_MAP
            retriever_map = DEFAULT_RETRIEVER_MAP.copy()
            retriever_map[primary_storage] = retrieval_type
            retrieval_config["retriever_map"] = retriever_map

        # secondary_routes is deprecated and removed from prompt schema
        if rp.secondary_routes:
            logger.info("Ignoring deprecated secondary_routes field")

        # Validate and filter rerank
        if rp.rerank not in VALID_RERANK:
            logger.warning(
                f"Unsupported rerank value '{rp.rerank}', resetting to 'none'."
            )
            retrieval_config["rerank"] = "none"

        # The canonical preset is the runtime source of truth. Architecture
        # proposals select a preset, not an ad-hoc operation list.
        preset_config = get_preset(management_preset)
        management_config = {
            "post_task_ops": list(preset_config.post_task_ops),
            "periodic_ops": list(preset_config.periodic_ops),
            "on_insert_ops": list(preset_config.on_insert_ops),
            "periodic_interval": preset_config.periodic_interval,
            "op_configs": dict(preset_config.op_configs),
        }

        if unsupported:
            logger.warning(f"Unsupported features: {unsupported}")

        return RuntimeConfig(
            extract_plan=extract_plan,
            primary_storage_type=primary_storage,
            storage_dir=storage_dir,
            additional_stores=additional_stores,
            retrieval_type=retrieval_type,
            retrieval_config=retrieval_config,
            management_preset=management_preset,
            management_config=management_config,
            unsupported_features=unsupported,
        )

    def _pick_primary_storage(self, storage_types: set) -> str:
        """Pick primary storage type. Prefer hybrid/json over graph backends."""
        if not storage_types:
            return "json"
        # Priority: hybrid > json > vector > graph > llm_graph
        for pref in ["hybrid", "json", "vector", "graph", "llm_graph"]:
            if pref in storage_types:
                return pref
        return list(storage_types)[0]

    def _resolve_retrieval(self, rp: RetrievalPlan, storage_types: set) -> str:
        """Resolve retrieval type from RetrievalPlan, respecting storage constraints."""
        if rp.primary_routes:
            candidate = rp.primary_routes[0]
        else:
            candidate = "hybrid"

        # Check graph constraint
        graph_stores = storage_types & set(CONSTRAINTS.get("graph_storage_types", []))
        graph_retrievals = CONSTRAINTS.get("retrieval_requires_graph_storage", [])
        needs_graph = candidate in graph_retrievals
        if needs_graph and not graph_stores:
            logger.warning(
                f"Retrieval '{candidate}' requires graph storage but none available. "
                f"Falling back to 'hybrid'."
            )
            return "hybrid"

        if candidate in ARCHITECTURE_SPACE.get("retrieval_types", []):
            return candidate

        logger.warning(f"Unknown retrieval type '{candidate}', falling back to 'hybrid'.")
        return "hybrid"

    def _resolve_management(self, mp: ManagementPlan, storage_types: set) -> str:
        """Resolve management preset from ManagementPlan."""
        # Try preset first
        preset = mp.preset
        if not preset:
            # Map intensity to preset
            intensity_map = {
                "none": "lightweight",
                "light": "lightweight",
                "medium": "lightweight",
                "heavy": "json_full",
            }
            preset = intensity_map.get(mp.intensity, "lightweight")

        try:
            preset = normalize_preset_name(preset)
        except ValueError:
            logger.warning(
                "Unknown management preset '%s', falling back to 'lightweight'.",
                preset,
            )
            return "lightweight"

        # Check graph constraint
        graph_management = CONSTRAINTS.get("management_requires_graph_storage", [])
        if preset in graph_management:
            graph_stores = storage_types & set(CONSTRAINTS.get("graph_storage_types", []))
            if not graph_stores:
                logger.warning(
                    f"Management '{preset}' requires graph storage. "
                    f"Falling back to 'json_full'."
                )
                return "json_full"

        if preset in ARCHITECTURE_SPACE.get("management_types", []):
            return preset

        logger.warning(
            f"Unknown management preset '{preset}', falling back to 'lightweight'."
        )
        return "lightweight"
