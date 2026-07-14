"""
Base classes for memory management operations.

Every management operation is a standalone, pluggable tool class that declares:
  - Which storage backends it supports (StorageCompatibility)
  - When it should be triggered (TriggerType)
  - Whether it needs an LLM or embedding model

Operations are composed into pipelines (ManagementPipeline) and orchestrated
by ManagementConfig, analogous to how HybridRetriever composes sub-retrievers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from ..storage.base_storage import BaseMemoryStorage

logger = logging.getLogger(__name__)


# ======================================================================
# Enums
# ======================================================================

class StorageCompatibility(Enum):
    """Declares which storage backends an operation supports."""
    ALL = "all"                    # Works with any storage backend
    JSON_ONLY = "json"             # Only JsonStorage
    VECTOR_ONLY = "vector"         # Only VectorStorage
    GRAPH_ONLY = "graph"           # Only GraphStore / LLMGraphStore
    NON_GRAPH = "non_graph"        # JsonStorage, VectorStorage, HybridStorage
    GRAPH_ENHANCED = "graph_enhanced"  # All backends, but graph has enhanced logic


class TriggerType(Enum):
    """When the operation should be triggered."""
    POST_TASK = "post_task"        # After each task completes
    PERIODIC = "periodic"          # Every N tasks
    ON_INSERT = "on_insert"        # When new memories are inserted


# ======================================================================
# OpResult — execution outcome of a single operation
# ======================================================================

@dataclass
class OpResult:
    """Result of executing a single management operation."""
    op_name: str
    triggered: bool = False
    units_affected: int = 0
    units_created: int = 0
    units_deleted: int = 0
    units_modified: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        if not self.triggered:
            return f"OpResult({self.op_name}: not triggered)"
        return (
            f"OpResult({self.op_name}: affected={self.units_affected}, "
            f"created={self.units_created}, deleted={self.units_deleted}, "
            f"modified={self.units_modified}, {self.duration_ms:.1f}ms)"
        )


# ======================================================================
# ManagementConfig — pipeline-level configuration
# ======================================================================

@dataclass
class ManagementConfig:
    """Configuration for a management pipeline."""
    # Ordered lists of operation names for each trigger phase
    post_task_ops: List[str] = field(default_factory=list)
    periodic_ops: List[str] = field(default_factory=list)
    on_insert_ops: List[str] = field(default_factory=list)

    # How often periodic ops run (every N tasks)
    periodic_interval: int = 10

    # Per-operation config overrides: op_name -> config dict
    op_configs: Dict[str, Dict] = field(default_factory=dict)


# ======================================================================
# ManagementResult — aggregated result of a pipeline run
# ======================================================================

@dataclass
class ManagementResult:
    """Aggregated result from running a pipeline phase."""
    phase: str  # "post_task", "periodic", "on_insert"
    results: List[OpResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def total_affected(self) -> int:
        return sum(r.units_affected for r in self.results if r.triggered)

    def __repr__(self) -> str:
        triggered = [r for r in self.results if r.triggered]
        return (
            f"ManagementResult({self.phase}: {len(triggered)}/{len(self.results)} ops triggered, "
            f"total_affected={self.total_affected}, {self.total_duration_ms:.1f}ms)"
        )


# ======================================================================
# BaseManageOp — abstract base class for all management operations
# ======================================================================

class BaseManageOp(ABC):
    """
    Abstract base class for management operations.

    Subclasses must set class-level attributes:
        op_name: str                    — unique identifier (e.g. "semantic_dedup")
        op_group: str                   — group name (e.g. "deduplication")
        trigger_type: TriggerType       — when to run
        storage_compatibility: StorageCompatibility
        requires_llm: bool              — whether execute() needs llm_client
        requires_embedding: bool        — whether execute() needs embedding_model
    """

    # --- Subclass must declare these ---
    op_name: str = ""
    op_group: str = ""
    trigger_type: TriggerType = TriggerType.POST_TASK
    storage_compatibility: StorageCompatibility = StorageCompatibility.ALL
    requires_llm: bool = False
    requires_embedding: bool = False

    # --- RL interface (reserved) ---
    rl_action_id: int = -1
    rl_param_range: Tuple = (0.0, 1.0)

    def __init__(
        self,
        store: BaseMemoryStorage,
        config: Dict,
        embedding_model=None,
        llm_client=None,
    ):
        self.store = store
        self.config = config
        self.embedding_model = embedding_model
        self.llm_client = llm_client

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> OpResult:
        """
        Execute the management operation.

        Args:
            context: Runtime context with keys like:
                - task_id: str
                - task_succeeded: bool
                - used_unit_ids: List[str]
                - task_query: str
                - new_unit_ids: List[str]  (for on_insert ops)

        Returns:
            OpResult describing what happened.
        """
        ...

    # --- Storage type detection helpers ---

    def is_compatible(self, storage_type: str) -> bool:
        """Check if this op is compatible with the given storage type string."""
        compat = self.storage_compatibility
        if compat == StorageCompatibility.ALL:
            return True
        if compat == StorageCompatibility.GRAPH_ENHANCED:
            return True
        if compat == StorageCompatibility.JSON_ONLY:
            return storage_type == "json"
        if compat == StorageCompatibility.VECTOR_ONLY:
            return storage_type == "vector"
        if compat == StorageCompatibility.GRAPH_ONLY:
            return storage_type in ("graph", "llm_graph")
        if compat == StorageCompatibility.NON_GRAPH:
            return storage_type not in ("graph", "llm_graph")
        return True

    def _is_graph_store(self) -> bool:
        """Check if current store is a GraphStore or LLMGraphStore."""
        return hasattr(self.store, 'neighbors')

    def _is_llm_graph_store(self) -> bool:
        """Check if current store is an LLMGraphStore."""
        return hasattr(self.store, 'get_all_entity_embeddings')

    def _has_faiss(self) -> bool:
        """Check if current store has a FAISS index that needs rebuilding."""
        return hasattr(self.store, '_rebuild_faiss_index')

    def _rebuild_faiss_if_needed(self) -> None:
        """Trigger FAISS index rebuild for Vector/Hybrid stores.

        Codex Q5-A3 fix (2026-04-28): the previous version called
        `_rebuild_faiss_index()` which still indexes inactive units (it
        rebuilds from `self._units` regardless of `is_active`). After
        utility_audit / size_capped_prune flip is_active=False, those
        vectors stayed in FAISS and continued to occupy top-k slots,
        starving genuinely active memories. The fix: prefer
        `compact()` (active-only physical removal) when the store
        provides it, then save the rebuilt index/state to disk so the
        next provider load doesn't restore the inactive vectors.
        """
        if not self._has_faiss():
            return
        # Prefer compact() — physically removes inactive units from
        # _units AND the FAISS index, then saves.
        compact = getattr(self.store, "compact", None)
        if callable(compact):
            try:
                compact()
                return
            except Exception:
                pass
        # Fallback: rebuild + save (so persisted index excludes inactive)
        self.store._rebuild_faiss_index()
        save = getattr(self.store, "save", None)
        if callable(save):
            try:
                save()
            except Exception:
                pass

    # --- RL interface (reserved) ---

    def to_rl_action(self, context: Dict) -> Dict:
        """Convert to an RL action descriptor (reserved for future use)."""
        return {
            "action_id": self.rl_action_id,
            "param_range": self.rl_param_range,
            "state": context.get("unit_rl_state"),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(op={self.op_name}, group={self.op_group})"
