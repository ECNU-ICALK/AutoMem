"""
ModularMemoryProvider — Adapter that bridges decoupled Storage + Retrieval
layers into the existing BaseMemoryProvider interface.

Extraction uses the package's internal prompt support helpers.
Storage and Retrieval are pluggable via config.

Config keys:
    enabled_prompts: List[str]       — non-empty subset of public Encode choices
    storage_type: str                — one public Store choice
    retriever_type: str              — one public Retrieve choice
    retriever_config: Dict           — strategy-specific params (weights, top_k, etc.)
    storage_dir: str                 — base dir for persistence
    top_k: int                       — max memories to retrieve (default 5)
    embedding_model_name: str        — sentence-transformers model id
    embedding_cache_dir: str         — local model cache
    prompt_dir: str                  — directory containing prompt .txt files

Usage:
    provider = ModularMemoryProvider(config={
        "enabled_prompts": ["tip"],
        "storage_type": "json",
        "retriever_type": "hybrid",
        "retriever_config": {"weights": {"SemanticRetriever": 0.7, "KeywordRetriever": 0.3}},
        "storage_dir": "./storage/modular_experiment_1",
    })
    provider.config["model"] = llm_model
    provider.initialize()
"""

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..base_memory import BaseMemoryProvider
from ..architecture.models import SCHEMA_VERSION, ArchitectureSpec
from ..memory_types import (
    MemoryItem,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)
from ..memory_schema import MemoryUnit, split_extraction_output
from ..runtime import (
    DEFAULT_RUNTIME_POLICY,
    InjectionSessionRegistry,
    MemoryContextComposer,
    QueryPlan,
    QueryPlanner,
)

from .prompt_support import (
    PROMPT_FILE_NAMES,
    PROMPT_TO_UNIT_TYPE,
    _build_template_context,
    _load_embedding_model,
    _parse_json_from_response,
    _render_prompt,
    format_memory_unit,
)

logger = logging.getLogger(__name__)


# Storage type → (class, module path, config builder)
_STORAGE_FACTORIES = {
    "json": lambda cfg: _make_json_storage(cfg),
    "vector": lambda cfg: _make_vector_storage(cfg),
    "hybrid": lambda cfg: _make_hybrid_storage(cfg),
    "graph": lambda cfg: _make_graph_storage(cfg),
    "llm_graph": lambda cfg: _make_llm_graph_storage(cfg),
}

_GRAPH_STORAGE_TYPES = {"graph", "llm_graph"}
_GRAPH_RETRIEVER_TYPES = {"graph", "hybrid_graph"}


def _make_json_storage(cfg):
    from ..storage import JsonStorage
    storage_cfg = dict(cfg)
    storage_cfg.setdefault("db_path", os.path.join(cfg["storage_dir"], "memory_db.json"))
    return JsonStorage(storage_cfg)


def _make_vector_storage(cfg):
    from ..storage import VectorStorage
    return VectorStorage(dict(cfg))


def _make_hybrid_storage(cfg):
    from ..storage import HybridStorage
    return HybridStorage(dict(cfg))


def _make_graph_storage(cfg):
    from ..storage import GraphStore
    return GraphStore(dict(cfg))


def _make_llm_graph_storage(cfg):
    from ..storage import LLMGraphStore
    return LLMGraphStore(dict(cfg))


def _make_retriever(
    retriever_type,
    store,
    embedding_model,
    retriever_config,
    model=None,
    model_resolver=None,
    usage_in_task_metrics_resolver=None,
):
    """Create a retriever instance by type string."""
    from ..retrieval import (
        SemanticRetriever,
        KeywordRetriever,
        HybridRetriever,
        GraphRetriever,
        ContrastiveRetriever,
        HybridGraphRetriever,
        CBRRetriever,
        CBRRerankRetriever,
    )

    if retriever_type == "semantic":
        return SemanticRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "keyword":
        return KeywordRetriever(store, retriever_config)

    elif retriever_type == "hybrid":
        semantic = SemanticRetriever(store, embedding_model, retriever_config)
        keyword = KeywordRetriever(store, retriever_config)
        return HybridRetriever(store, [semantic, keyword], retriever_config)

    elif retriever_type == "graph":
        return GraphRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "contrastive":
        return ContrastiveRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "hybrid_graph":
        return HybridGraphRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "cbr":
        # Case-Based Reasoning: match on source_task_query first, content fallback.
        return CBRRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "cbr_rerank":
        # Memento-inspired: CBR candidate pool + LLM rerank.
        return CBRRerankRetriever(
            store,
            embedding_model,
            retriever_config,
            model=model,
            model_resolver=model_resolver,
            usage_in_task_metrics_resolver=usage_in_task_metrics_resolver,
        )

    elif retriever_type == "tag":
        from ..retrieval import TagRetriever, TagVocabulary, QueryClassifier
        vocab = TagVocabulary()
        classifier = QueryClassifier(
            model=retriever_config.get("_model"),
            vocabulary=vocab,
        )
        return TagRetriever(store, classifier, retriever_config)

    # Stage-1 (2026-05-17) adoptions:
    elif retriever_type == "hyde":
        # HyDE: LLM-generate hypothesis passage + average emb with query + dense retrieval.
        # Codex F-7 fix (2026-05-18): the architecture compiler does not
        # populate retriever_config["hypothesis_model"], so previously HyDE
        # silently fell back to no-hypothesis (semantic-only) for every
        # sampled candidate. Use the provider's task/extraction model as a
        # default so HyDE actually does HyDE.
        from ..retrieval import HydeRetriever
        hypo_model = retriever_config.get("hypothesis_model") or model
        return HydeRetriever(
            store,
            embedding_model=embedding_model,
            hypothesis_model=hypo_model,
            config=retriever_config,
        )

    elif retriever_type == "mmr":
        # MMR: dense recall + diversity re-rank. 0 LLM calls.
        from ..retrieval import MmrRetriever
        return MmrRetriever(store, embedding_model, retriever_config)

    else:
        logger.warning(f"Unknown retriever_type '{retriever_type}', falling back to semantic")
        return SemanticRetriever(store, embedding_model, retriever_config)


class ModularMemoryProvider(BaseMemoryProvider):
    """
    Adapter provider that composes:
      - Extraction: uses the shared prompt support helpers
      - Storage: pluggable backend (JsonStorage / GraphStore)
      - Retrieval: pluggable strategy (5 options)

    Implements BaseMemoryProvider interface so it works with existing
    GAIA runner and agent framework unchanged.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.MODULAR, config)

        # Codex Q14-1 fix (2026-04-28): under --shared_memory_provider
        # --concurrency > 1, the eval script reassigns
        # `memory_provider.model = task_model` per task. Without
        # thread-isolation, worker A's extraction could fire with
        # worker B's freshly-assigned model. Make `model` a property
        # backed by `_task_model_local` (per-thread) with a process
        # default `_default_model` for any thread that hasn't set it.
        self._task_model_local = threading.local()
        self._default_model = self.config.get("model")
        self.runtime_policy = DEFAULT_RUNTIME_POLICY
        self.runtime_policy_id = self.runtime_policy.policy_id
        self.runtime_policy_digest = self.runtime_policy.digest
        self._query_planner = QueryPlanner(self.runtime_policy)
        self._context_composer = MemoryContextComposer(self.runtime_policy)
        self._injection_sessions = InjectionSessionRegistry(self.runtime_policy)
        self._extraction_model_cache = self.config.get("extraction_model")

        # Architecture and operational behavior come from one resolved config;
        # shell variables cannot silently mutate a candidate between workers.
        self.storage_dir = str(self.config.get("storage_dir", "./storage/modular"))
        self.storage_type = str(self.config.get("storage_type", "json"))
        self.storage_config = dict(self.config.get("storage_config", {}))
        self.retriever_type = str(self.config.get("retriever_type", "hybrid"))
        self.retriever_config = dict(self.config.get("retriever_config", {}))
        self.top_k = int(self.config.get("top_k", 4))

        self._metrics_local = threading.local()

        # G2/G4 reuse the run's memory/task model by default. A pre-built
        # OpenAI-compatible client may be injected for deployments that assign
        # a dedicated memory model, but endpoint/model selection is not a
        # searchable runtime option.
        self.runtime_model_id = str(
            self.config.get("runtime_model_id")
            or getattr(self._default_model, "model_id", "runtime-model")
        )
        self._runtime_client = self.config.get("runtime_client")

        self.injection_type = "context_composer"

        # Protocol-v2 M4: no-self-retrieval guard. During architecture search
        # the same task batch is re-attempted round after round while the
        # canonical pool already holds units extracted from those very tasks
        # — retrieving a task's own trajectory/answer is leakage, not memory
        # transfer (xBench: search hit_rate 0.93 vs validation 0.27). When
        # enabled, units whose source_task_query matches the incoming query
        # (or whose source_task_id matches request.additional_params.task_id)
        # are filtered out right after retrieval.
        self._no_self_retrieval = True

        self.tag_aware = False

        # Extraction config
        self.enabled_prompts: List[str] = list(
            self.config.get("enabled_prompts", ["tip"])
        )
        # Resolve prompts from installed package data, independent of cwd.
        _default_prompt_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
        )
        self.prompt_dir = (
            self.config.get("prompt_dir")
            or _default_prompt_dir
        )

        self.embedding_model_name = self.config.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_cache_dir = self.config.get(
            "embedding_cache_dir", "./storage/models"
        )

        # Management config
        self._management_enabled = True
        self._management_preset = str(
            self.config.get("management_preset", "lightweight")
        )
        self._management_ops_override = ""

        # A public architecture selects a non-empty subset of Encode types and
        # persists all of them to its one selected store.
        self._additional_stores_config = dict(
            self.config.get("additional_stores", {})
        )
        if not self.enabled_prompts:
            raise ValueError("enabled_prompts must contain at least one Encode choice")
        if self._additional_stores_config:
            raise ValueError("additional_stores are outside the public E/S/R/M space")
        ArchitectureSpec(
            schema_version=SCHEMA_VERSION,
            encode=tuple(self.enabled_prompts),
            store=self.storage_type,
            retrieve=self.retriever_type,
            manage=self._management_preset,
        )

        # Parallel safety: serialize provider-level reads/writes so
        # concurrent tasks sharing this provider don't clobber shared state.
        self._provider_lock = threading.RLock()

        # Internal state (initialized during initialize())
        self.store = None
        self.retriever = None
        self.embedding_model = None
        self.manager = None
        self._management_config = None
        self._managers: Dict[str, Any] = {}
        self._prompt_templates: Dict[str, str] = {}
        self._last_provided_ids: List[str] = []
        self._experiment_metrics: Dict[str, Any] = {}
        self._stores: Dict[str, Any] = {}  # Heterogeneous storage: {"json": JsonStorage, "vector": VectorStorage, ...}
        self.reset_experiment_metrics()

    @property
    def model(self):
        """Codex Q14-1 fix: per-thread task model.

        Eval workers assign `memory_provider.model = task_model` per
        task. Without thread-local routing, worker A's extraction
        could run with worker B's freshly-set model. The property
        returns the calling thread's model if set; otherwise the
        process-wide `_default_model`.

        An explicitly injected extraction model is preferred over the task
        model for memory-side calls.
        """
        ext = self._get_extraction_model()
        if ext is not None:
            return ext
        local_model = getattr(
            getattr(self, "_task_model_local", None), "value", None
        )
        if local_model is not None:
            return local_model
        return getattr(self, "_default_model", None)

    def _get_extraction_model(self):
        """Return the resolved optional extraction model."""
        return getattr(self, "_extraction_model_cache", None)

    @model.setter
    def model(self, value):
        # The setter routes to thread-local. Initial __init__ uses
        # `_default_model` directly (NOT this setter) to avoid race
        # with the property protocol.
        if not hasattr(self, "_task_model_local"):
            self._task_model_local = threading.local()
        self._task_model_local.value = value

    def _resolved_management_preset(self) -> str:
        if not self._management_enabled:
            return "lightweight"
        if self._management_preset:
            return self._management_preset
        if self.storage_type in _GRAPH_STORAGE_TYPES and self.retriever_type == "graph":
            return "graph_consolidate"
        return "lightweight"

    def _graph_stats(self) -> Dict[str, Any]:
        total_nodes = 0
        total_edges = 0
        has_graph_stats = False
        for store in self._stores.values():
            if not hasattr(store, "stats"):
                continue
            try:
                stats = store.stats()
            except Exception:
                continue
            nodes = stats.get("total_nodes")
            edges = stats.get("total_edges")
            if nodes is not None:
                total_nodes += nodes
                has_graph_stats = True
            if edges is not None:
                total_edges += edges
                has_graph_stats = True

        if not has_graph_stats:
            return {"graph_nodes": None, "graph_edges": None}
        return {"graph_nodes": total_nodes, "graph_edges": total_edges}

    def _signature_exists_anywhere(self, signature: str) -> bool:
        if not signature:
            return False
        for store in self._stores.values():
            try:
                if store.exists_signature(signature):
                    return True
            except Exception:
                continue
        return False

    def _update_memory_totals(self) -> None:
        num_units = 0
        by_store: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}

        for store_type, store in self._stores.items():
            try:
                count = store.count()
            except Exception:
                count = 0
            by_store[store_type] = count
            num_units += count

            try:
                for unit in store.get_all(active_only=False):
                    type_key = unit.type.value
                    type_counts[type_key] = type_counts.get(type_key, 0) + 1
            except Exception:
                continue

        self._experiment_metrics["num_memory_units"] = num_units
        self._experiment_metrics["num_memory_units_by_store"] = by_store
        self._experiment_metrics["type_counts"] = type_counts
        self._experiment_metrics.update(self._graph_stats())

    def _record_management_results(self, results: List[Any], store_type: Optional[str] = None) -> None:
        triggered = 0
        serialized = self._experiment_metrics.setdefault("management_results", [])
        for phase_result in results:
            for op_result in phase_result.results:
                if op_result.triggered:
                    triggered += 1
                entry = {
                    "phase": phase_result.phase,
                    "op_name": op_result.op_name,
                    "triggered": op_result.triggered,
                    "units_affected": op_result.units_affected,
                    "units_created": op_result.units_created,
                    "units_deleted": op_result.units_deleted,
                    "units_modified": op_result.units_modified,
                    "duration_ms": op_result.duration_ms,
                    "details": op_result.details,
                }
                if store_type is not None:
                    entry["store_type"] = store_type
                serialized.append(entry)
        self._experiment_metrics["management_ops_triggered"] += triggered

    def _build_management_config(self):
        from ..management import ManagementConfig
        from ..management.presets import get_preset

        mgmt_config = self.config.get("management_config")
        if self._management_ops_override:
            ops_list = [
                o.strip()
                for o in self._management_ops_override.split(",")
                if o.strip()
            ]
            return ManagementConfig(
                post_task_ops=ops_list,
                periodic_ops=ops_list,
                on_insert_ops=[
                    o for o in ops_list
                    if o in ("signature_dedup", "conflict_detection")
                ],
            )
        if mgmt_config is not None:
            return ManagementConfig(**mgmt_config)
        return get_preset(self._resolved_management_preset())

    def _get_or_create_manager(self, store_type: str, store):
        if not self._management_enabled or self._management_config is None:
            return None
        if store_type in self._managers:
            return self._managers[store_type]

        from ..management import ManagementPipeline

        manager = ManagementPipeline(
            store=store,
            config=self._management_config,
            embedding_model=self.embedding_model,
            llm_client=self.model,
        )
        self._managers[store_type] = manager
        if store_type == self.storage_type:
            self.manager = manager
        return manager

    def _validate_runtime_compatibility(self) -> None:
        # Check all storage types (primary + additional) for graph capability
        all_storage_types = {self.storage_type} | set(self._additional_stores_config.keys())
        has_graph = bool(all_storage_types & _GRAPH_STORAGE_TYPES)

        if self.retriever_type in _GRAPH_RETRIEVER_TYPES and not has_graph:
            raise ValueError(
                f"Retriever '{self.retriever_type}' requires graph storage, "
                f"got '{self.storage_type}'"
            )

        if self._management_enabled:
            from ..management.preset_registry import validate_preset_capabilities

            errors = validate_preset_capabilities(
                self._resolved_management_preset(),
                storage_types=all_storage_types,
                retrieval_types=[self.retriever_type],
            )
            if errors:
                raise ValueError(" ".join(errors))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            self._validate_runtime_compatibility()

            # 1. Load embedding model, or use an explicitly injected instance
            # for embedding-service adapters and deterministic offline tests.
            self.embedding_model = self.config.get("embedding_model")
            if self.embedding_model is None:
                self.embedding_model = _load_embedding_model(
                    self.embedding_model_name, self.embedding_cache_dir
                )

            # 2. Create and initialize storage backend
            factory = _STORAGE_FACTORIES.get(self.storage_type)
            if factory is None:
                logger.error(f"Unknown storage_type: {self.storage_type}")
                return False

            primary_storage_config = {
                "storage_dir": self.storage_dir,
                **self.storage_config,
            }
            if self.storage_type == "llm_graph":
                has_explicit_model = any(
                    primary_storage_config.get(key) is not None
                    for key in ("model", "llm_client", "model_resolver")
                ) or bool(primary_storage_config.get("llm_api_key"))
                if not has_explicit_model:
                    primary_storage_config["model"] = self._build_runtime_client()
                    primary_storage_config["model_resolver"] = (
                        self._build_runtime_client
                    )
                primary_storage_config.setdefault(
                    "usage_in_task_metrics_resolver",
                    self._runtime_usage_is_in_task_metrics,
                )
            self.store = factory(primary_storage_config)
            if not self.store.initialize():
                logger.error("Storage backend initialization failed")
                return False
            # P1 — give the store a handle to the embedding model so update()
            # can re-encode on content changes (case_rewrite / rename tag).
            if hasattr(self.store, "set_embedding_model"):
                try:
                    self.store.set_embedding_model(self.embedding_model)
                except Exception as e:
                    logger.warning(f"set_embedding_model failed on primary store: {e}")

            # Register primary store in heterogeneous store dict
            self._stores[self.storage_type] = self.store

            # Pre-create secondary stores from compiler config.
            # Fix 2: only load stores that have an EXPLICIT absolute path in
            # _additional_stores_config.  Refuse to create stores from bare
            # type names (which would fall back to default paths like
            # ./storage/json) because those paths accumulate stale data
            # across runs and poison the retriever with pre-RRF multi-store
            # fusion.
            for store_type, explicit_path in self._additional_stores_config.items():
                if store_type == self.storage_type:
                    continue
                if not explicit_path or not isinstance(explicit_path, str):
                    logger.warning(
                        f"Refusing to pre-create secondary store '{store_type}': "
                        f"no explicit path in _additional_stores_config "
                        f"(got {explicit_path!r}). This prevents stale default-path pollution."
                    )
                    continue
                try:
                    self._get_or_create_store(store_type)
                    logger.info(f"Pre-created secondary store: {store_type} at {explicit_path}")
                except Exception as e:
                    logger.warning(f"Failed to pre-create secondary store {store_type}: {e}")

            # 3. Create retriever (MultiStoreRetriever when multiple stores)
            # Inject model reference for TagRetriever / QueryClassifier
            self.retriever_config["_model"] = self.model

            # When tag_aware is enabled and we have multiple stores, build a
            # shared QueryClassifier and inject it so MultiStoreRetriever can
            # apply a tag-boost after RRF fusion (Phase 2.5).
            if self.tag_aware and len(self._stores) > 1:
                try:
                    from ..retrieval.tag_vocabulary import TagVocabulary
                    from ..retrieval.query_classifier import QueryClassifier as _QC
                    _vocab = TagVocabulary()
                    _classifier = _QC(model=self.model, vocabulary=_vocab)
                    self.retriever_config["query_classifier"] = _classifier
                    logger.info("Tag-aware retrieval: QueryClassifier injected into MultiStoreRetriever config")
                except Exception as _e:
                    logger.warning(f"Failed to build QueryClassifier for tag-boost: {_e}")

            if len(self._stores) > 1:
                from ..retrieval import MultiStoreRetriever
                # Inject retriever_map so the primary store keeps the user's
                # chosen retriever_type. The default
                # DEFAULT_RETRIEVER_MAP would silently overwrite e.g.
                # contrastive/cbr_rerank/hyde/mmr with the storage-default
                # retriever (json→keyword, hybrid→hybrid, …). Secondary
                # explicitly configured secondary stores inherit
                # DEFAULT_RETRIEVER_MAP via the dict-get fallback inside
                # MultiStoreRetriever.
                mstore_cfg = dict(self.retriever_config)
                if "retriever_map" not in mstore_cfg:
                    mstore_cfg["retriever_map"] = {
                        self.storage_type: self.retriever_type,
                    }
                self.retriever = MultiStoreRetriever(
                    stores=self._stores,
                    embedding_model=self.embedding_model,
                    model=self._build_runtime_client(),
                    model_resolver=self._build_runtime_client,
                    usage_in_task_metrics_resolver=(
                        self._runtime_usage_is_in_task_metrics
                    ),
                    config=mstore_cfg,
                )
                logger.info(
                    f"Using MultiStoreRetriever with stores: {list(self._stores.keys())}, "
                    f"retriever_map={mstore_cfg['retriever_map']}"
                )
            else:
                self.retriever = _make_retriever(
                    self.retriever_type,
                    self.store,
                    self.embedding_model,
                    self.retriever_config,
                    model=self._build_runtime_client(),
                    model_resolver=self._build_runtime_client,
                    usage_in_task_metrics_resolver=(
                        self._runtime_usage_is_in_task_metrics
                    ),
                )

            # 4. Load prompt templates
            self._load_prompt_templates()

            # Convert silent extraction-disabled state into a loud init-time
            # failure: if the user asked for prompts but none loaded, the run
            # would otherwise complete with num_extracted=0 across every task
            # and look like a successful "no_memory" run. Root-cause typically
            # an incorrect resolved config or a missing package-data file.
            if self.enabled_prompts and not self._prompt_templates:
                raise RuntimeError(
                    f"ModularMemoryProvider: enabled_prompts="
                    f"{self.enabled_prompts} but no templates loaded from "
                    f"prompt_dir={self.prompt_dir!r}. Check `prompt_dir` in the "
                    f"resolved config. Without templates, "
                    f"memory extraction silently no-ops."
                )

            # 5. Initialize management pipeline(s)
            self.manager = None
            self._managers = {}
            self._management_config = None
            if self._management_enabled:
                try:
                    self._management_config = self._build_management_config()
                    for store_type, store in self._stores.items():
                        self._get_or_create_manager(store_type, store)
                except Exception as e:
                    logger.warning(f"Management pipeline init failed (non-fatal): {e}")
                    self.manager = None
                    self._management_config = None
                    self._managers = {}

            logger.info(
                f"ModularMemoryProvider initialized: "
                f"storage={self.storage_type}, retriever={self.retriever_type}, "
                f"prompts={list(self._prompt_templates.keys())}, "
                f"management={'enabled' if self._managers else 'disabled'}, "
                f"existing_units={sum(store.count() for store in self._stores.values())}"
            )
            self.reset_experiment_metrics()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ModularMemoryProvider: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_or_create_store(self, store_type: str):
        """Lazy-load a secondary storage backend."""
        if store_type in self._stores:
            return self._stores[store_type]

        from automem.storage import JsonStorage, VectorStorage, HybridStorage, GraphStore, LLMGraphStore

        # Use compiler-specified path if available, else default
        store_dir = self._additional_stores_config.get(
            store_type,
            os.path.join(self.storage_dir, f"store_{store_type}")
        )
        os.makedirs(store_dir, exist_ok=True)

        store_map = {
            "json": JsonStorage,
            "vector": VectorStorage,
            "hybrid": HybridStorage,
            "graph": GraphStore,
            "llm_graph": LLMGraphStore,
        }

        cls = store_map.get(store_type)
        if cls is None:
            logger.warning(f"Unknown store type: {store_type}, falling back to json")
            cls = JsonStorage

        if store_type == "json" or cls is JsonStorage:
            config = {"db_path": os.path.join(store_dir, "memory_db.json")}
        else:
            config = {"storage_dir": store_dir}
        if store_type == "llm_graph":
            config["model"] = self._build_runtime_client()
            config["model_resolver"] = self._build_runtime_client
            config["usage_in_task_metrics_resolver"] = (
                self._runtime_usage_is_in_task_metrics
            )

        store = cls(config)
        if not store.initialize():
            raise RuntimeError(f"Failed to initialize secondary store: {store_type}")
        # P1 — inject embedding model so update() can re-encode on content change
        if hasattr(store, "set_embedding_model") and self.embedding_model is not None:
            try:
                store.set_embedding_model(self.embedding_model)
            except Exception as e:
                logger.warning(f"set_embedding_model failed on secondary store {store_type}: {e}")
        self._stores[store_type] = store
        self._get_or_create_manager(store_type, store)
        return store

    def _route_units_to_stores(self, units, storage_routing: Dict[str, str]):
        """Route memory units to different storage backends by type."""
        routed: Dict[str, List[MemoryUnit]] = {}
        for unit in units:
            type_key = unit.type.value
            store_type = storage_routing.get(
                type_key,
                list(storage_routing.values())[0] if storage_routing else "json",
            )
            routed.setdefault(store_type, []).append(unit)

        inserted_counts: Dict[str, int] = {}
        for store_type, store_units in routed.items():
            store = self._get_or_create_store(store_type)
            before_count = store.count()
            if store_type == "llm_graph" and hasattr(store, "upsert_memory_unit"):
                for unit in store_units:
                    store.upsert_memory_unit(unit)
            else:
                store.add(store_units)
            inserted_counts[store_type] = max(store.count() - before_count, 0)
            logger.debug(
                "Routed %d unit(s) to %s store",
                len(store_units),
                store_type,
            )

        return routed, inserted_counts

    # ------------------------------------------------------------------
    # Memory ingestion
    # ------------------------------------------------------------------

    def take_in_memory(self, trajectory_data: TrajectoryData, extract_plan: Optional[Dict[str, Any]] = None) -> tuple:
        with self._provider_lock:
            return self._take_in_memory_unlocked(trajectory_data, extract_plan)

    def _take_in_memory_unlocked(self, trajectory_data: TrajectoryData, extract_plan: Optional[Dict[str, Any]] = None) -> tuple:
        # Utilization check (added 2026-04-28): did the agent actually
        # reference any memory we injected at task start? When no kept unit
        # appears in the trajectory text, fire `injection_failed_signal` so
        # attribution.py can label the task as INJECTION_BAD. Done first so
        # the signal is captured even if extraction early-returns below.
        # Codex CR2-2: prefer per-query lookup (race-safe) over the shared
        # _last_provided_units attribute. Falls back to the shared field
        # only when the query key is missing (e.g. legacy paths).
        per_query = getattr(self, "_provided_units_by_query", None) or {}
        last_units = per_query.pop(trajectory_data.query, None)
        if last_units is None:
            last_units = getattr(self, "_last_provided_units", None) or []
        # Race-safe used_unit_ids for post-task management (boost_on_success):
        # derive from THIS query's popped units, not the shared
        # _last_provided_ids which a concurrent worker's
        # reset_experiment_metrics() may clear mid-task under
        # shared_memory_provider + concurrency > 1. Codex P1 fix.
        _used_unit_ids = [u.id for u in last_units if getattr(u, "id", None)] if last_units else []
        # G1 (2026-07-11): pop this query's traversed graph edges for the
        # edge_stats_update post-task op (same query-keyed race safety).
        _edges_per_query = getattr(self, "_provided_edges_by_query", None) or {}
        _used_edge_pairs = _edges_per_query.pop(trajectory_data.query, None) or []
        if last_units:
            try:
                used = self._check_injection_utilization(trajectory_data, last_units)
                self._experiment_metrics["injection_failed_signal"] = (not used)
            except Exception as e:
                logger.debug(f"injection utilization check failed: {e}")
            finally:
                self._last_provided_units = []

        if not self.model:
            return False, "No model provided for memory extraction"

        if not self._prompt_templates:
            return False, "No prompt templates loaded"

        metadata = trajectory_data.metadata or {}
        is_correct = metadata.get("is_correct", False)
        task_outcome = "success" if is_correct else "failure"
        task_id = metadata.get("task_id", str(uuid.uuid4())[:8])

        context = _build_template_context(trajectory_data, is_correct)

        new_units: List[MemoryUnit] = []
        prompts_used = []
        extracted_count = 0
        llm_error_count = 0          # extraction LLM calls that failed after retries
        prompts_attempted = 0        # extraction prompts we actually tried to run

        # If an architecture supplies an extraction plan, run exactly those
        # encode types. Search failures belong in the run ExperienceLedger;
        # they must not silently add INSIGHT to a candidate that did not select
        # that encode component.
        if extract_plan is not None:
            extract_types = list(extract_plan.get("extract_types", []))
            ArchitectureSpec.from_search_dict(
                {
                    "extract_types": extract_types,
                    "storage_routing": dict(
                        extract_plan.get("storage_routing", {})
                    ),
                    "retrieval": self.retriever_type,
                    "management": self._resolved_management_preset(),
                }
            )
            templates_to_run = {
                k: v for k, v in self._prompt_templates.items()
                if k in extract_types
            }
        else:
            templates_to_run = self._prompt_templates

        for prompt_name, template_str in templates_to_run.items():
            unit_type = PROMPT_TO_UNIT_TYPE.get(prompt_name)
            if unit_type is None:
                continue

            # W2 provenance gate (write-side): failure tasks only contribute
            # INSIGHT units (failure-mode lessons), not "what worked" types.
            # Success tasks contribute every type EXCEPT insight.
            if not is_correct and prompt_name != "insight":
                logger.debug(
                    f"[provenance] skipping {prompt_name} extraction on failed "
                    f"task {task_id} (only INSIGHT allowed from failures)"
                )
                continue
            if is_correct and prompt_name == "insight":
                continue

            try:
                filled_prompt = _render_prompt(template_str, context)
            except Exception as e:
                logger.error(f"Template rendering failed for {prompt_name}: {e}")
                continue

            prompts_attempted += 1
            messages = [
                {"role": "user", "content": [{"type": "text", "text": filled_prompt}]}
            ]
            response_text = None
            for _attempt in range(3):
                try:
                    response = self.model(messages)
                    response_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )
                    break
                except Exception as e:
                    if _attempt < 2:
                        time.sleep(1.5 * (_attempt + 1))
                        continue
                    # Exhausted retries: the extraction endpoint/credentials are
                    # likely down. Count it so a silently-failing extraction
                    # pipeline (0 units across a whole round) is surfaced loudly
                    # below instead of being discovered rounds later.
                    logger.error(
                        f"LLM call failed for {prompt_name} after 3 attempts: {e}"
                    )
                    llm_error_count += 1
            if response_text is None:
                continue

            parsed = self._parse_extraction_with_retry(
                messages, response_text, prompt_name
            )
            if parsed is None:
                # Count it: parse failures used to be 100% silent (320
                # occurrences across historical runs), each dropping a whole
                # prompt's units while take_in_memory still reported success.
                self._experiment_metrics["extraction_parse_failures"] = (
                    self._experiment_metrics.get("extraction_parse_failures", 0) + 1
                )
                logger.warning(
                    f"Failed to parse extraction result for {prompt_name} "
                    f"(after one corrective retry)"
                )
                continue

            if isinstance(parsed, dict) and parsed.get("skipped"):
                continue

            try:
                _src_q = (
                    metadata.get("original_question")
                    or trajectory_data.query
                )
                units = split_extraction_output(
                    extraction_result=parsed,
                    unit_type=unit_type,
                    source_task_id=task_id,
                    source_task_query=_src_q,
                    task_outcome=task_outcome,
                    extraction_model=str(getattr(self.model, "model_id", "unknown")),
                )
            except Exception as e:
                logger.error(f"split_extraction_output failed for {prompt_name}: {e}")
                continue

            extracted_count += len(units)

            # Dedup + embed + add to storage
            for unit in units:
                if self._signature_exists_anywhere(unit.signature):
                    continue

                text = unit.content_text()
                if text and self.embedding_model is not None:
                    unit.embedding = self.embedding_model.encode(
                        text, convert_to_numpy=True
                    )

                new_units.append(unit)

            prompts_used.append(prompt_name)

        # Batch add to storage
        inserted_count = 0
        routed_units_by_store: Dict[str, List[MemoryUnit]] = {}
        storage_routing = dict(extract_plan.get("storage_routing", {})) if extract_plan is not None else {}

        if new_units and storage_routing:
            # Heterogeneous routing: send units to different stores by type
            routed_units_by_store, inserted_by_store = self._route_units_to_stores(new_units, storage_routing)
            inserted_count = sum(inserted_by_store.values())
            logger.info(
                f"ModularMemoryProvider: routed {inserted_count} units via heterogeneous storage "
                f"from {', '.join(prompts_used)}"
            )
        elif new_units:
            before_count = self.store.count()
            if self.storage_type in ("graph", "llm_graph") and hasattr(self.store, 'upsert_memory_unit'):
                # For GraphStore/LLMGraphStore: use upsert_memory_unit
                # LLMGraphStore will auto-run LLM entity extraction pipeline
                for unit in new_units:
                    if self.storage_type == "llm_graph":
                        # LLMGraphStore extracts entities via LLM internally
                        self.store.upsert_memory_unit(unit)
                    else:
                        from ..storage.graph_storage import extract_entities_from_unit
                        entities = extract_entities_from_unit(unit)
                        self.store.upsert_memory_unit(unit, entities=entities)
            else:
                self.store.add(new_units)
            inserted_count = max(self.store.count() - before_count, 0)
            self.store.save()
            routed_units_by_store = {self.storage_type: new_units}
            logger.info(
                f"ModularMemoryProvider: added {inserted_count} units from "
                f"{', '.join(prompts_used)} (total: {self.store.count()})"
            )

        if llm_error_count:
            logger.warning(
                "[extraction] %d/%d extraction LLM call(s) FAILED for task %s "
                "(extracted=%d units). The extraction model endpoint/credentials "
                "may be down; fix it before continuing or the search wastes rounds "
                "on an empty memory pool.",
                llm_error_count, prompts_attempted, task_id, len(new_units),
            )
        msg = (
            f"Extracted {len(new_units)} units from "
            f"{len(prompts_used)} prompts ({', '.join(prompts_used)})"
            + (f" [WARN: {llm_error_count}/{prompts_attempted} extraction LLM calls failed]"
               if llm_error_count else "")
        )
        self._experiment_metrics["num_extracted"] += extracted_count
        self._experiment_metrics["num_inserted"] += inserted_count
        self._experiment_metrics["num_deduped"] += max(extracted_count - inserted_count, 0)

        # Build failure_context for reflection_correction (empty on success)
        if not is_correct:
            result_text = str(trajectory_data.result or "")
            failure_context = (
                f"Task failed. Agent answer: {result_text[:500]}"
                if result_text else "Task failed (no result captured)."
            )
        else:
            failure_context = ""

        # Run management pipeline
        if self._managers:
            try:
                for store_type, units_for_store in routed_units_by_store.items():
                    manager = self._managers.get(store_type)
                    if manager is None or not units_for_store:
                        continue
                    manager.run_on_insert(
                        units_for_store,
                        {"new_unit_ids": [u.id for u in units_for_store]},
                    )

                post_task_context = {
                    "task_id": task_id,
                    "task_succeeded": is_correct,
                    "used_unit_ids": _used_unit_ids or list(self._last_provided_ids or []),
                    # G1: graph edges traversed by this task's retrieval, for
                    # edge_stats_update (empty list for non-graph retrieval).
                    "used_edge_pairs": _used_edge_pairs,
                    "task_query": trajectory_data.query,
                    "failure_context": failure_context,
                }
                for manager in self._managers.values():
                    manager.run_post_task(dict(post_task_context))

                self._last_provided_ids = []
                for store_type, manager in self._managers.items():
                    self._record_management_results(
                        manager.consume_recent_results(),
                        store_type=store_type,
                    )
            except Exception as e:
                logger.warning(f"Management pipeline error (non-fatal): {e}")

        self._update_memory_totals()

        # If every extraction call errored and nothing was produced, report
        # failure so the eval layer logs a WARNING. take_in_memory otherwise
        # returns success and an all-failed extraction stays invisible (debug).
        if llm_error_count and not new_units:
            return False, msg
        return True, msg

    # ------------------------------------------------------------------
    # Memory retrieval (delegates to Retriever → MemoryPack)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Disagreement Gate (post-retrieval consistency filter)
    # ------------------------------------------------------------------
    _UTIL_STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "for",
        "on", "and", "or", "but", "with", "by", "this", "that", "it", "its",
        "as", "be", "do", "if", "you", "we", "they", "have", "has", "had",
        "from", "into", "than", "then", "when", "what", "which", "who",
        "step", "tool", "memory", "task", "query", "agent", "answer",
    })

    @classmethod
    def _extract_distinctive_tokens(cls, text: str) -> list:
        """Return rare, lowercased ≥4-char tokens for keyword-overlap matching."""
        if not text or not isinstance(text, str):
            return []
        import re as _re
        cleaned = _re.sub(r"[^\w\s]", " ", text.lower())
        return [
            t for t in cleaned.split()
            if len(t) >= 4 and t not in cls._UTIL_STOPWORDS
        ]

    @classmethod
    def _unit_distinctive_tokens(cls, unit) -> list:
        """Pull distinctive matchable tokens from a MemoryUnit's content."""
        from automem.memory_schema import MemoryUnitType
        c = unit.content or {}
        toks: list = []

        if unit.type == MemoryUnitType.TIP:
            for k in ("topic", "principle", "applicability"):
                v = c.get(k)
                if isinstance(v, str):
                    toks.extend(cls._extract_distinctive_tokens(v))
        elif unit.type == MemoryUnitType.SHORTCUT:
            name = c.get("name", "")
            if isinstance(name, str) and name:
                # shortcut names are snake_case — split into parts
                toks.extend(name.lower().replace("_", " ").split())
            toks.extend(cls._extract_distinctive_tokens(c.get("description", "")))
        elif unit.type == MemoryUnitType.WORKFLOW:
            for wf_key in ("agent_workflow", "search_workflow"):
                for s in c.get(wf_key, []) or []:
                    if isinstance(s, dict):
                        toks.extend(cls._extract_distinctive_tokens(
                            s.get("action", "") or s.get("query_formulation", "")
                        ))
        elif unit.type == MemoryUnitType.INSIGHT:
            toks.extend(cls._extract_distinctive_tokens(
                c.get("root_cause_conclusion", "")
            ))
            toks.extend(cls._extract_distinctive_tokens(
                c.get("corrective_strategy", "")
            ))
        elif unit.type == MemoryUnitType.TRAJECTORY:
            toks.extend(cls._extract_distinctive_tokens(c.get("key_decision", "")))
            anchor = c.get("reusable_anchor", "")
            if isinstance(anchor, str):
                toks.extend(cls._extract_distinctive_tokens(anchor))

        # Add use_when triggers — these are the agent-facing applicability hints
        for uw in getattr(unit, "use_when", []) or []:
            toks.extend(cls._extract_distinctive_tokens(str(uw)))

        # Dedup while preserving order
        seen, out = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # Markers that delimit memory injection; we strip everything between
    # them before doing utilization keyword overlap. Without this strip the
    # injected prompt itself contributes token hits, falsely zeroing
    # injection_failed_signal (Codex CR2, 2026-04-28).
    _MEMORY_BLOCK_BEGIN = "Memory System Guidance"
    _MEMORY_BLOCK_END = "End Memory"

    @classmethod
    def _strip_memory_block(cls, text: str) -> str:
        """Remove anything between Memory System Guidance ... End Memory."""
        if not text:
            return ""
        import re as _re
        # The block uses unicode dashes (————) wrapping the markers; match
        # both ASCII and unicode em-dash variants. Greedy across newlines.
        pattern = (
            r"[—\-]+\s*" + _re.escape(cls._MEMORY_BLOCK_BEGIN) + r"\s*[—\-]+"
            r"[\s\S]*?"
            r"[—\-]+\s*" + _re.escape(cls._MEMORY_BLOCK_END) + r"\s*[—\-]+"
        )
        return _re.sub(pattern, " ", text, flags=_re.IGNORECASE)

    @staticmethod
    def _render_judge_head(unit, score: float) -> str:
        """Codex Q3-3 fix (2026-04-28): render the priority signals
        (Apply when / Avoid when / Source / negative-example) FIRST so
        the judge sees them even when the memory body is truncated."""
        bits = []
        if getattr(unit, "is_negative_example", False):
            bits.append("[NEG]")
        uw = getattr(unit, "use_when", None) or []
        if uw:
            bits.append("Apply when: " + "; ".join(str(t) for t in uw if t)[:200])
        aw = getattr(unit, "avoid_when", None) or []
        if aw:
            bits.append("Avoid when: " + "; ".join(str(t) for t in aw if t)[:200])
        src = (getattr(unit, "source_task_query", "") or "").strip()
        if src:
            bits.append(f"Source: {src[:120]}")
        if not bits:
            return ""
        return "(" + " | ".join(bits) + ") "

    @classmethod
    def _check_injection_utilization(cls, trajectory_data, units) -> bool:
        """Heuristic: did the agent's trajectory reference any injected memory?

        Counts distinctive token overlap between each kept unit and the
        agent's OWN trajectory text — explicitly stripping the injected
        memory block itself so the prompt we just inserted does not count
        as "evidence of use" (Codex CR2).

        Returns True if at least one unit has ≥2 distinctive tokens that
        appear in the cleaned trajectory text. Returns True for empty
        units list (nothing was injected — not a failed injection).
        """
        if not units:
            return True
        traj = getattr(trajectory_data, "trajectory", None)
        if isinstance(traj, list):
            raw = " ".join(str(s) for s in traj)
        elif traj is None:
            raw = ""
        else:
            raw = str(traj)
        # Strip the injected memory block before token matching.
        cleaned = cls._strip_memory_block(raw).lower()
        if not cleaned:
            return False
        for u in units:
            tokens = cls._unit_distinctive_tokens(u)
            if not tokens:
                continue
            hits = sum(1 for t in tokens if t in cleaned)
            # Need ≥2 distinctive token hits to count as "agent referenced it"
            # (single-token match is unreliable due to surface-form collisions
            # with the original question wording).
            if hits >= 2:
                return True
        return False

    # ------------------------------------------------------------------
    def _build_runtime_client(self):
        """Return the fixed runtime model/client, or None for offline fallback."""
        if self._runtime_client is not None:
            return self._runtime_client
        runtime_model = self.model
        if runtime_model is not None and (
            callable(runtime_model) or hasattr(runtime_model, "chat")
        ):
            return runtime_model
        return None

    def _runtime_usage_is_in_task_metrics(self, runtime_client: Any) -> bool:
        """Whether the runner's task-model counter already observes this client."""
        task_model = getattr(
            getattr(self, "_task_model_local", None), "value", None
        ) or getattr(self, "_default_model", None)
        return runtime_client is not None and runtime_client is task_model

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        with self._provider_lock:
            return self._provide_memory_unlocked(request)

    def _composer_candidates(self, scored_units: list) -> list:
        """Render candidates once for the fixed relevance-and-composition call."""
        candidates = []
        for scored in scored_units:
            head = self._render_judge_head(scored.unit, float(scored.score))
            try:
                body = format_memory_unit(scored.unit, float(scored.score))
            except Exception:
                body = str(scored.unit.content)
            candidates.append(
                {
                    "id": str(scored.unit.id),
                    "score": float(scored.score),
                    "text": f"{head}{body}",
                }
            )
        return candidates

    @staticmethod
    def _supporting_edges(used_edges: list, selected_unit_ids: list) -> list:
        """Keep only graph paths that contributed to an injected memory."""
        needed = {f"m:{unit_id}" for unit_id in selected_unit_ids}
        supporting = []
        remaining = list(used_edges or [])
        changed = True
        while changed:
            changed = False
            next_remaining = []
            for edge in remaining:
                if not isinstance(edge, (tuple, list)) or len(edge) < 3:
                    continue
                source, target, edge_type = edge[:3]
                if str(target) in needed:
                    item = (str(source), str(target), str(edge_type))
                    if item not in supporting:
                        supporting.append(item)
                    if str(source) not in needed:
                        needed.add(str(source))
                        changed = True
                else:
                    next_remaining.append(edge)
            remaining = next_remaining
        return supporting

    def _provide_memory_unlocked(self, request: MemoryRequest) -> MemoryResponse:
        empty_response = MemoryResponse(
            memories=[],
            memory_type=self.memory_type,
            total_count=0,
            request_id=str(uuid.uuid4()),
        )

        params = request.additional_params or {}
        task_id = str(params.get("task_id", "") or "")
        session_key = self._injection_sessions.key(request.query, task_id)
        refresh_boundary = bool(params.get("refresh_boundary", False))
        if not self._injection_sessions.phase_allowed(
            session_key,
            request.status,
            refresh_boundary=refresh_boundary,
        ):
            self._experiment_metrics["phase_refresh_denied"] += 1
            return empty_response

        if self.retriever is None or self.store is None:
            return empty_response

        # Codex CR2-7: in heterogeneous architectures the primary store can be
        # empty while a secondary store holds the relevant units (e.g. workflow
        # memories routed to graph store, primary is json with no tip units
        # yet). Falling through here lets MultiStoreRetriever consult the
        # secondaries; only short-circuit when ALL stores are empty.
        primary_count = self.store.count()
        secondary_total = sum(
            s.count() for k, s in getattr(self, "_stores", {}).items()
            if s is not self.store
        )
        if primary_count == 0 and secondary_total == 0:
            return empty_response

        # Fixed G4 policy: preserve the literal query for lexical/entity
        # matching and supplement only the semantic embedding with an abstract
        # retrieval focus. HyDE already performs its own expansion.
        from ..retrieval import QueryContext
        auxiliary_client = self._build_runtime_client()
        if self._runtime_usage_is_in_task_metrics(auxiliary_client):
            self._experiment_metrics["runtime_usage_in_task_metrics"] = True
        if self.retriever_type == "hyde":
            query_plan = QueryPlan(literal=request.query, used_fallback=True)
        else:
            self._experiment_metrics["query_planner_calls"] += 1
            if auxiliary_client is not None:
                self._experiment_metrics["query_planner_model_calls"] += 1
            query_plan = self._query_planner.plan(
                request.query,
                context=request.context,
                client=auxiliary_client,
                model=self.runtime_model_id,
            )
            if query_plan.used_fallback:
                self._experiment_metrics["query_planner_fallbacks"] += 1
            self._experiment_metrics["query_planner_input_tokens"] += (
                query_plan.input_tokens
            )
            self._experiment_metrics["query_planner_output_tokens"] += (
                query_plan.output_tokens
            )
        query_emb = None
        if self.embedding_model is not None:
            literal_emb = self.embedding_model.encode(
                query_plan.literal, convert_to_numpy=True
            )
            query_emb = literal_emb
            if query_plan.abstract:
                abstract_emb = self.embedding_model.encode(
                    query_plan.abstract, convert_to_numpy=True
                )
                query_emb = (np.asarray(literal_emb) + np.asarray(abstract_emb)) / 2.0

        ctx = QueryContext(
            query=query_plan.literal,
            embedding=query_emb,
            task_id=task_id,
            metadata={"abstract_query": query_plan.abstract},
        )

        # Retrieve — upgrade to MultiStoreRetriever if new stores appeared
        self._experiment_metrics["retrieval_calls"] += 1
        active_stores = {k: v for k, v in self._stores.items() if v.count() > 0}
        from ..retrieval import MultiStoreRetriever
        if len(active_stores) > 1 and not isinstance(self.retriever, MultiStoreRetriever):
            # FIX 2026-06-29: inject retriever_map so this runtime upgrade keeps the
            # architecture's chosen retriever for the primary store. Without it the
            # config has no retriever_map and DEFAULT_RETRIEVER_MAP silently overrides
            # e.g. contrastive/cbr_rerank/hyde/mmr with the storage default — the exact
            # leak the compile-time path (L851-855) guards against. Mirror it here.
            _mcfg = dict(self.retriever_config) if getattr(self, "retriever_config", None) else {}
            _mcfg.setdefault("retriever_map", {self.storage_type: self.retriever_type})
            self.retriever = MultiStoreRetriever(
                stores=active_stores,
                embedding_model=self.embedding_model,
                model=self._build_runtime_client(),
                model_resolver=self._build_runtime_client,
                usage_in_task_metrics_resolver=(
                    self._runtime_usage_is_in_task_metrics
                ),
                config=_mcfg,
            )
            logger.info(f"Upgraded to MultiStoreRetriever with stores: {list(active_stores.keys())}, retriever_map={_mcfg['retriever_map']}")
        pack = self.retriever.retrieve(ctx, top_k=self.top_k)

        # Protocol-v2 M4: drop units sourced from the CURRENT task before any
        # downstream counter/judge sees them — semantics: "the retriever never
        # returned it". Matches by exact source_task_query (the agent's query
        # IS the task question, so no extra plumbing is needed and the check
        # is safe under concurrency) and, when provided, by explicit task_id.
        if self._no_self_retrieval and pack.scored_units:
            _qnorm = (request.query or "").strip()
            _req_task_id = str(
                (request.additional_params or {}).get("task_id", "") or ""
            )
            _before = len(pack.scored_units)
            pack.scored_units = [
                su for su in pack.scored_units
                if not (
                    (_qnorm and (getattr(su.unit, "source_task_query", None) or "").strip() == _qnorm)
                    or (_req_task_id and str(getattr(su.unit, "source_task_id", "") or "") == _req_task_id)
                )
            ]
            _dropped = _before - len(pack.scored_units)
            if _dropped:
                self._experiment_metrics["self_retrieval_filtered"] = (
                    self._experiment_metrics.get("self_retrieval_filtered", 0) + _dropped
                )

        if request.status == MemoryStatus.IN and pack.scored_units:
            unit_ids = [str(scored.unit.id) for scored in pack.scored_units]
            unseen = self._injection_sessions.unseen_indices(session_key, unit_ids)
            before = len(pack.scored_units)
            pack.scored_units = [pack.scored_units[index] for index in unseen]
            self._experiment_metrics["phase_refresh_seen_dropped"] += (
                before - len(pack.scored_units)
            )

        # Track pre-judge retrieval count so attribution can distinguish
        # "retriever returned 0" from "judge dropped all".
        self._experiment_metrics["num_retrieved_pre_judge"] += len(pack.scored_units)

        if pack.is_empty():
            self._update_memory_totals()
            return empty_response

        # Fixed G2 policy: relevance selection and tentative context synthesis
        # happen in one composer call. With no auxiliary endpoint, use the
        # deterministic top-1 raw-memory fallback.
        candidates = self._composer_candidates(pack.scored_units)
        self._experiment_metrics["context_composer_calls"] += 1
        self._experiment_metrics["context_composer_candidates"] += len(candidates)
        if auxiliary_client is not None:
            self._experiment_metrics["context_composer_model_calls"] += 1
        composition = self._context_composer.compose(
            request.query,
            candidates,
            client=auxiliary_client,
            model=self.runtime_model_id,
        )
        self._experiment_metrics["context_composer_input_tokens"] += (
            composition.input_tokens
        )
        self._experiment_metrics["context_composer_output_tokens"] += (
            composition.output_tokens
        )
        if composition.used_fallback:
            self._experiment_metrics["context_composer_fallbacks"] += 1
        if composition.no_guidance:
            self._experiment_metrics["context_composer_no_guidance"] += 1
            self._update_memory_totals()
            return empty_response

        selected = [pack.scored_units[index] for index in composition.kept_indices]
        guidance_text = composition.guidance
        for index in composition.kept_indices:
            guidance_text = guidance_text.replace(
                f"[M{index}]", f"[memory:{pack.scored_units[index].unit.id}]"
            )
        pack.scored_units = selected
        self._experiment_metrics["context_composer_selected"] += len(selected)
        selected_ids = [str(scored.unit.id) for scored in selected]
        if not self._injection_sessions.commit(
            session_key,
            request.status,
            selected_ids,
            guidance_text,
        ):
            self._experiment_metrics["duplicate_guidance_dropped"] += 1
            self._update_memory_totals()
            return empty_response
        # Per-task fine-grained signals consumed by attribution.py decision tree.
        if pack.scored_units:
            confidences = [
                float(getattr(su.unit, "confidence", 1.0))
                for su in pack.scored_units
            ]
            stale_count = 0
            for su in pack.scored_units:
                unit = su.unit
                if not getattr(unit, "is_active", True):
                    stale_count += 1
                    continue
                if getattr(unit, "conflict_count", 0) > 0:
                    stale_count += 1
                elif getattr(unit, "superseded_by", None):
                    stale_count += 1
            self._experiment_metrics["avg_kept_confidence"] = (
                sum(confidences) / len(confidences) if confidences else None
            )
            self._experiment_metrics["kept_units_stale_count"] = stale_count
        self._experiment_metrics["injection_type"] = self.injection_type
        self._experiment_metrics["num_retrieved"] += len(pack.scored_units)
        self._experiment_metrics["retriever_name"] = pack.retriever_name

        memory_item = MemoryItem(
            id=f"modular_{uuid.uuid4()}",
            content=guidance_text,
            metadata={
                "retriever": pack.retriever_name,
                "num_units": len(pack.scored_units),
                "by_type": {k: len(v) for k, v in pack.by_type.items()},
                "total_memory_units": self.store.count(),
                "citation_ids": selected_ids,
                "phase": request.status.value,
                "abstract_query": query_plan.abstract,
                "runtime_policy_id": self.runtime_policy_id,
                "runtime_policy_digest": self.runtime_policy_digest,
            },
            score=float(np.mean([su.score for su in pack.scored_units])) if pack.scored_units else 0.0,
        )

        logger.info(
            f"provide_memory: {pack.retriever_name} returned {len(pack.scored_units)} units "
            f"(query='{request.query[:60]}...', total={self.store.count()})"
        )

        # Track provided memory IDs for management feedback
        self._last_provided_ids = [su.unit.id for su in pack.scored_units]
        # Track full units per-query so take_in_memory can run a utilization
        # check against the agent trace (powers INJECTION_BAD attribution).
        # Codex CR2-2: under shared_memory_provider + concurrency > 1 a single
        # provider-wide attribute would race; keyed by query so each task can
        # find its own injected pack.
        self._last_provided_units = [su.unit for su in pack.scored_units]
        provided_units = self._provided_units_by_query.setdefault(request.query, [])
        known_unit_ids = {str(unit.id) for unit in provided_units}
        provided_units.extend(
            unit
            for unit in self._last_provided_units
            if str(unit.id) not in known_unit_ids
        )
        # G1 (2026-07-11): track graph edges the retriever traversed for this
        # query so edge_stats_update can credit them post-task. Same
        # query-keyed race-safety rationale as _provided_units_by_query.
        provided_edges = self._provided_edges_by_query.setdefault(request.query, [])
        for edge in self._supporting_edges(
            getattr(pack, "used_edges", []) or [], selected_ids
        ):
            if edge not in provided_edges:
                provided_edges.append(edge)
        self._update_memory_totals()

        return MemoryResponse(
            memories=[memory_item],
            memory_type=self.memory_type,
            total_count=1,
            request_id=str(uuid.uuid4()),
        )

    def _parse_extraction_with_retry(self, messages, response_text, prompt_name):
        """Parse an extraction LLM response; on parse failure, re-ask ONCE
        with an explicit JSON-only reminder before giving up.

        Returns the parsed object or None. The caller counts the terminal
        failure in extraction_parse_failures.
        """
        parsed = _parse_json_from_response(response_text)
        if parsed is not None:
            return parsed
        logger.warning(
            f"Extraction output for {prompt_name} was not valid JSON; retrying once"
        )
        retry_messages = list(messages) + [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "Your previous output could not be parsed as JSON. "
                    "Redo the extraction and respond with STRICTLY VALID JSON "
                    "only — no markdown fences, no commentary."
                ),
            }],
        }]
        try:
            resp = self.model(retry_messages)
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.warning(f"Parse-retry LLM call failed for {prompt_name}: {e}")
            return None
        return _parse_json_from_response(text)

    def _load_prompt_templates(self) -> None:
        """Load templates for the encode components selected by the architecture."""
        prompts_to_load = list(self.enabled_prompts or [])
        for prompt_name in prompts_to_load:
            fname = PROMPT_FILE_NAMES.get(prompt_name)
            if not fname:
                logger.warning(f"Unknown prompt name: {prompt_name}, skipping")
                continue
            fpath = os.path.join(self.prompt_dir, fname)
            if not os.path.exists(fpath):
                logger.warning(
                    "Prompt file not found for '%s': %s", prompt_name, fpath
                )
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                self._prompt_templates[prompt_name] = f.read()

    @property
    def _experiment_metrics(self) -> Dict[str, Any]:
        local = getattr(self, "_metrics_local", None)
        if local is None:
            local = threading.local()
            self._metrics_local = local
        d = getattr(local, "experiment_metrics", None)
        if d is None:
            d = self._new_experiment_metrics_dict()
            local.experiment_metrics = d
        return d

    @_experiment_metrics.setter
    def _experiment_metrics(self, value: Dict[str, Any]) -> None:
        local = getattr(self, "_metrics_local", None)
        if local is None:
            local = threading.local()
            self._metrics_local = local
        local.experiment_metrics = value

    def _new_experiment_metrics_dict(self) -> Dict[str, Any]:
        """Fresh per-task experiment-metrics dict (used by the thread-local
        property's lazy init and by reset_experiment_metrics)."""
        store = getattr(self, "store", None)
        try:
            management_preset = self._resolved_management_preset()
        except Exception:
            management_preset = None
        return {
            "storage_backend": getattr(self, "storage_type", None),
            "retrieval_strategy": getattr(self, "retriever_type", None),
            "management_preset": management_preset,
            "num_extracted": 0,
            "num_memory_units": store.count() if store is not None else 0,
            "num_inserted": 0,
            "num_deduped": 0,
            "num_retrieved": 0,
            "num_retrieved_pre_judge": 0,
            "self_retrieval_filtered": 0,
            "judge_dropped_all_count": 0,
            "retrieval_calls": 0,
            "query_planner_calls": 0,
            "query_planner_model_calls": 0,
            "query_planner_fallbacks": 0,
            "query_planner_input_tokens": 0,
            "query_planner_output_tokens": 0,
            "context_composer_calls": 0,
            "context_composer_model_calls": 0,
            "context_composer_candidates": 0,
            "context_composer_selected": 0,
            "context_composer_fallbacks": 0,
            "context_composer_no_guidance": 0,
            "context_composer_input_tokens": 0,
            "context_composer_output_tokens": 0,
            "runtime_usage_in_task_metrics": False,
            "rerank_input_tokens": 0,
            "rerank_output_tokens": 0,
            "rerank_calls": 0,
            "rerank_usage_in_task_metrics": False,
            "llm_graph_input_tokens": 0,
            "llm_graph_output_tokens": 0,
            "llm_graph_calls": 0,
            "llm_graph_usage_in_task_metrics": False,
            "phase_refresh_denied": 0,
            "phase_refresh_seen_dropped": 0,
            "duplicate_guidance_dropped": 0,
            "management_ops_triggered": 0,
            "management_results": [],
            "retriever_name": None,
            "graph_nodes": None,
            "graph_edges": None,
            # Fine-grained signals (added 2026-04-27) — consumed by attribution
            "injection_type": getattr(self, "injection_type", "context_composer"),
            "avg_kept_confidence": None,
            "kept_units_stale_count": 0,
            # injection_failed_signal (added 2026-04-28): set by take_in_memory
            # when the agent trace did not reference any kept memory. Powers
            # the INJECTION_BAD attribution category.
            "injection_failed_signal": False,
        }

    def reset_experiment_metrics(self) -> None:
        self._last_provided_ids = []
        self._last_provided_units = []
        # Per-query injection cache (Codex CR2-2 + Round-3 R3-3).
        # Initialize on first call only; subsequent resets MUST NOT clear
        # in-flight entries because, under shared_memory_provider +
        # concurrency > 1, another task may have already cached its units
        # via provide_memory but not yet reached take_in_memory. Clearing
        # here would orphan that task's INJECTION_BAD signal.
        if not hasattr(self, "_provided_units_by_query"):
            self._provided_units_by_query: Dict[str, List[Any]] = {}
        # Defensive bound on size — drop oldest entries if dict grows beyond
        # 256. take_in_memory pops on the happy path, so this only kicks in
        # if a task crashed before take_in_memory.
        elif len(self._provided_units_by_query) > 256:
            keys = list(self._provided_units_by_query.keys())
            for k in keys[:len(keys) - 200]:
                self._provided_units_by_query.pop(k, None)
        # G1 (2026-07-11): mirror bookkeeping for traversed graph edges.
        if not hasattr(self, "_provided_edges_by_query"):
            self._provided_edges_by_query: Dict[str, List[Any]] = {}
        elif len(self._provided_edges_by_query) > 256:
            keys = list(self._provided_edges_by_query.keys())
            for k in keys[:len(keys) - 200]:
                self._provided_edges_by_query.pop(k, None)
        # Reset retriever-tree thread-local counters (cbr_rerank holds
        # its own threading.local for parity with the provider).
        def _reset_retriever_thread_locals(r):
            if r is None:
                return
            reset_usage = getattr(r, "reset_usage_metrics", None)
            if callable(reset_usage):
                reset_usage()
            else:
                tl = getattr(r, "_rerank_thread_local", None)
                if tl is not None:
                    tl.rerank_calls = 0
                    tl.rerank_input_tokens = 0
                    tl.rerank_output_tokens = 0
                    tl.rerank_usage_in_task_metrics = False
            sub_cache = getattr(r, "_sub_retriever_cache", None)
            if isinstance(sub_cache, dict):
                for sub in sub_cache.values():
                    _reset_retriever_thread_locals(sub)
            for attr in ("sub_retrievers", "_cbr"):
                child = getattr(r, attr, None)
                if isinstance(child, list):
                    for c in child:
                        _reset_retriever_thread_locals(c)
                elif child is not None:
                    _reset_retriever_thread_locals(child)
        try:
            _reset_retriever_thread_locals(getattr(self, "retriever", None))
        except Exception:
            pass
        for store in getattr(self, "_stores", {}).values():
            reset_usage = getattr(store, "reset_usage_metrics", None)
            if callable(reset_usage):
                reset_usage()
        self._experiment_metrics = self._new_experiment_metrics_dict()
        for manager in self._managers.values():
            if hasattr(manager, "clear_recent_results"):
                manager.clear_recent_results()
        self._update_memory_totals()

    def get_experiment_metrics(self) -> Dict[str, Any]:
        self._update_memory_totals()
        snapshot = dict(self._experiment_metrics)
        candidates = int(snapshot.get("context_composer_candidates", 0))
        selected = int(snapshot.get("context_composer_selected", 0))
        if candidates > 0:
            snapshot["judge_decisions_total"] = candidates
            snapshot["noise_injection_rate"] = round(
                (candidates - selected) / candidates, 4
            )
            snapshot["actionable_usefulness_rate"] = round(selected / candidates, 4)
        else:
            snapshot["judge_decisions_total"] = 0
            snapshot["noise_injection_rate"] = None
            snapshot["actionable_usefulness_rate"] = None
        snapshot["judge_input_tokens"] = int(
            snapshot.get("context_composer_input_tokens", 0)
        )
        snapshot["judge_output_tokens"] = int(
            snapshot.get("context_composer_output_tokens", 0)
        )
        snapshot["judge_api_calls"] = int(
            snapshot.get("context_composer_model_calls", 0)
        )
        snapshot["runtime_input_tokens"] = int(
            snapshot.get("query_planner_input_tokens", 0)
        ) + int(snapshot.get("context_composer_input_tokens", 0))
        snapshot["runtime_output_tokens"] = int(
            snapshot.get("query_planner_output_tokens", 0)
        ) + int(snapshot.get("context_composer_output_tokens", 0))
        snapshot["runtime_api_calls"] = int(
            snapshot.get("query_planner_model_calls", 0)
        ) + int(snapshot.get("context_composer_model_calls", 0))

        # Walk the retriever tree for per-thread rerank counters.
        rerank_in = rerank_out = rerank_calls = 0
        rerank_in_task_flags: List[bool] = []
        visited_retrievers = set()
        def _walk_for_rerank(r):
            nonlocal rerank_in, rerank_out, rerank_calls
            if r is None or id(r) in visited_retrievers:
                return
            visited_retrievers.add(id(r))
            get_usage = getattr(r, "get_usage_metrics", None)
            if callable(get_usage):
                usage = get_usage()
                calls = int(usage.get("rerank_calls", 0) or 0)
                rerank_in += int(usage.get("rerank_input_tokens", 0) or 0)
                rerank_out += int(usage.get("rerank_output_tokens", 0) or 0)
                rerank_calls += calls
                if calls:
                    rerank_in_task_flags.append(
                        bool(usage.get("rerank_usage_in_task_metrics", False))
                    )
            else:
                tl = getattr(r, "_rerank_thread_local", None)
                if tl is not None:
                    calls = int(getattr(tl, "rerank_calls", 0) or 0)
                    rerank_in += int(getattr(tl, "rerank_input_tokens", 0) or 0)
                    rerank_out += int(getattr(tl, "rerank_output_tokens", 0) or 0)
                    rerank_calls += calls
                    if calls:
                        rerank_in_task_flags.append(False)
            sub_cache = getattr(r, "_sub_retriever_cache", None)
            if isinstance(sub_cache, dict):
                for sub in sub_cache.values():
                    _walk_for_rerank(sub)
            for attr in ("sub_retrievers", "_cbr"):
                child = getattr(r, attr, None)
                if isinstance(child, list):
                    for c in child:
                        _walk_for_rerank(c)
                elif child is not None:
                    _walk_for_rerank(child)
        try:
            _walk_for_rerank(getattr(self, "retriever", None))
        except Exception:
            pass
        snapshot["rerank_input_tokens"] = rerank_in
        snapshot["rerank_output_tokens"] = rerank_out
        snapshot["rerank_calls"] = rerank_calls
        snapshot["rerank_usage_in_task_metrics"] = bool(
            rerank_in_task_flags and all(rerank_in_task_flags)
        )

        graph_in = graph_out = graph_calls = 0
        graph_in_task_flags: List[bool] = []
        visited_stores = set()
        for store in getattr(self, "_stores", {}).values():
            if id(store) in visited_stores:
                continue
            visited_stores.add(id(store))
            get_usage = getattr(store, "get_usage_metrics", None)
            if not callable(get_usage):
                continue
            usage = get_usage()
            calls = int(usage.get("llm_graph_calls", 0) or 0)
            graph_in += int(usage.get("llm_graph_input_tokens", 0) or 0)
            graph_out += int(usage.get("llm_graph_output_tokens", 0) or 0)
            graph_calls += calls
            if calls:
                graph_in_task_flags.append(
                    bool(usage.get("llm_graph_usage_in_task_metrics", False))
                )
        snapshot["llm_graph_input_tokens"] = graph_in
        snapshot["llm_graph_output_tokens"] = graph_out
        snapshot["llm_graph_calls"] = graph_calls
        snapshot["llm_graph_usage_in_task_metrics"] = bool(
            graph_in_task_flags and all(graph_in_task_flags)
        )
        snapshot["injection_stats"] = {
            "renderer": "context_composer",
            "candidates": candidates,
            "selected": selected,
            "fallbacks": int(snapshot.get("context_composer_fallbacks", 0)),
        }
        return snapshot
