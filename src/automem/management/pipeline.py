"""
ManagementPipeline — Orchestrates management operations in configurable order.

The pipeline instantiates operations from a registry, checks compatibility
with the current storage backend, and executes them in the configured order
for each trigger phase (post_task, periodic, on_insert).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Type

from .base_op import (
    BaseManageOp,
    ManagementConfig,
    ManagementResult,
    OpResult,
)
from ..storage.base_storage import BaseMemoryStorage

logger = logging.getLogger(__name__)


# ======================================================================
# Operation Registry — maps op_name -> OpClass
# ======================================================================

def _build_registry() -> Dict[str, Type[BaseManageOp]]:
    """Lazily import all op classes and build the registry."""
    from .ops.cluster_merge import ClusterMergeOp
    from .ops.trajectory_to_workflow import TrajectoryToWorkflowOp
    from .ops.cross_task_generalize import CrossTaskGeneralizeOp
    from .ops.reindex_relations import ReindexRelationsOp
    from .ops.signature_dedup import SignatureDedupOp
    from .ops.semantic_dedup import SemanticDedupOp
    from .ops.cross_type_dedup import CrossTypeDedupOp
    from .ops.conflict_detection import ConflictDetectionOp
    from .ops.penalize_on_failure import PenalizeOnFailureOp
    from .ops.boost_on_success import BoostOnSuccessOp
    from .ops.reflection_correction import ReflectionCorrectionOp
    from .ops.access_stats_update import AccessStatsUpdateOp
    from .ops.time_decay import TimeDecayOp
    from .ops.score_based_prune import ScoreBasedPruneOp
    from .ops.quality_curation import QualityCurationOp
    # Shortcut operations are registered for the tool-manager preset.
    from .ops.shortcut_promotion import ShortcutPromotionOp
    from .ops.shortcut_validation import ShortcutValidationOp
    # Registered 2026-04-27: utility_audit op (idea 3 step A) — deactivates
    # units with empirical success_rate < threshold; mandatory_periodic.
    from .ops.utility_audit import UtilityAuditOp
    # Registered 2026-04-28: size_capped_prune — emergency periodic pruning
    # when pool > cap. Sits in mandatory_periodic.
    from .ops.size_capped_prune import SizeCappedPruneOp
    # Stage-1 (2026-05-17) adoption: LLM-based conflict resolution.
    from .ops.llm_conflict_resolve import LLMConflictResolveOp
    # Registered 2026-07-11 (G1): edge-level usage feedback loop ported from
    # the cerebra_fusion_memory evolved architecture.
    from .ops.edge_stats_update import EdgeStatsUpdateOp
    from .ops.edge_weight_optimize import EdgeWeightOptimizeOp

    return {
        "cluster_merge": ClusterMergeOp,
        "trajectory_to_workflow": TrajectoryToWorkflowOp,
        "cross_task_generalize": CrossTaskGeneralizeOp,
        "reindex_relations": ReindexRelationsOp,
        "signature_dedup": SignatureDedupOp,
        "semantic_dedup": SemanticDedupOp,
        "cross_type_dedup": CrossTypeDedupOp,
        "conflict_detection": ConflictDetectionOp,
        "penalize_on_failure": PenalizeOnFailureOp,
        "boost_on_success": BoostOnSuccessOp,
        "reflection_correction": ReflectionCorrectionOp,
        "access_stats_update": AccessStatsUpdateOp,
        "time_decay": TimeDecayOp,
        "score_based_prune": ScoreBasedPruneOp,
        "quality_curation": QualityCurationOp,
        "shortcut_promotion": ShortcutPromotionOp,
        "shortcut_validation": ShortcutValidationOp,
        "utility_audit": UtilityAuditOp,
        "size_capped_prune": SizeCappedPruneOp,
        "llm_conflict_resolve": LLMConflictResolveOp,
        "edge_stats_update": EdgeStatsUpdateOp,
        "edge_weight_optimize": EdgeWeightOptimizeOp,
    }


# Module-level cache
_OP_REGISTRY: Optional[Dict[str, Type[BaseManageOp]]] = None


def get_op_registry() -> Dict[str, Type[BaseManageOp]]:
    """Get or build the operation registry (singleton)."""
    global _OP_REGISTRY
    if _OP_REGISTRY is None:
        _OP_REGISTRY = _build_registry()
    return _OP_REGISTRY


# ======================================================================
# ManagementPipeline
# ======================================================================

class ManagementPipeline:
    """
    Configurable orchestrator for memory management operations.

    Instantiates ops from the registry, skips incompatible or under-resourced
    ones, and runs them in the configured order for each trigger phase.

    Usage:
        pipeline = ManagementPipeline(store, config, embedding_model, llm_client)
        result = pipeline.run_post_task({"task_succeeded": True, "used_unit_ids": [...]})
        result = pipeline.run_periodic()
        result = pipeline.run_on_insert(new_units)
    """

    def __init__(
        self,
        store: BaseMemoryStorage,
        config: ManagementConfig,
        embedding_model=None,
        llm_client=None,
    ):
        self.store = store
        self.config = config
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self._task_counter = 0
        self._recent_results: List[ManagementResult] = []

        # Detect storage type string for compatibility checks
        self._storage_type = self._detect_storage_type()

        # Instantiate operations
        registry = get_op_registry()
        self._post_task_ops = self._instantiate_ops(config.post_task_ops, registry)
        self._periodic_ops = self._instantiate_ops(config.periodic_ops, registry)
        self._on_insert_ops = self._instantiate_ops(config.on_insert_ops, registry)

        logger.info(
            f"ManagementPipeline initialized: "
            f"post_task={[o.op_name for o in self._post_task_ops]}, "
            f"periodic={[o.op_name for o in self._periodic_ops]}, "
            f"on_insert={[o.op_name for o in self._on_insert_ops]}, "
            f"storage_type={self._storage_type}"
        )

    def _detect_storage_type(self) -> str:
        """Infer storage type string from the store instance."""
        cls_name = type(self.store).__name__
        if "LLMGraph" in cls_name:
            return "llm_graph"
        if "Graph" in cls_name:
            return "graph"
        if "Vector" in cls_name:
            return "vector"
        if "Hybrid" in cls_name:
            return "hybrid"
        return "json"

    def _instantiate_ops(
        self,
        op_names: List[str],
        registry: Dict[str, Type[BaseManageOp]],
    ) -> List[BaseManageOp]:
        """Instantiate ops, skipping incompatible or under-resourced ones."""
        ops = []
        for name in op_names:
            cls = registry.get(name)
            if cls is None:
                logger.warning(f"Unknown op '{name}', skipping")
                continue

            # Check storage compatibility
            if not cls.storage_compatibility:
                pass  # default: ALL
            else:
                # Create a temporary instance to check compatibility
                temp = cls.__new__(cls)
                temp.storage_compatibility = cls.storage_compatibility
                if not temp.is_compatible(self._storage_type):
                    logger.info(
                        f"Op '{name}' incompatible with storage '{self._storage_type}', skipping"
                    )
                    continue

            # Check resource requirements
            if cls.requires_llm and self.llm_client is None:
                logger.info(f"Op '{name}' requires LLM but none provided, skipping")
                continue
            if cls.requires_embedding and self.embedding_model is None:
                logger.info(f"Op '{name}' requires embedding model but none provided, skipping")
                continue

            # Instantiate
            op_config = self.config.op_configs.get(name, {})
            op = cls(
                store=self.store,
                config=op_config,
                embedding_model=self.embedding_model,
                llm_client=self.llm_client,
            )
            ops.append(op)
        return ops

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_post_task(self, context: Dict[str, Any]) -> ManagementResult:
        """
        Run post-task operations.

        Handles boost/penalize mutual exclusion: only runs the one matching
        context["task_succeeded"].
        """
        self._task_counter += 1
        task_succeeded = context.get("task_succeeded", False)

        result = ManagementResult(phase="post_task")
        start = time.time()

        for op in self._post_task_ops:
            # Mutual exclusion for boost/penalize
            if op.op_name == "boost_on_success" and not task_succeeded:
                result.results.append(OpResult(op_name=op.op_name, triggered=False))
                continue
            if op.op_name == "penalize_on_failure" and task_succeeded:
                result.results.append(OpResult(op_name=op.op_name, triggered=False))
                continue

            op_result = self._run_single_op(op, context)
            result.results.append(op_result)

        result.total_duration_ms = (time.time() - start) * 1000
        self._recent_results.append(result)

        # Check if periodic ops should run.
        # FIX 2026-06-29: guard against periodic_interval <= 0 (settable via presets)
        # which would raise ZeroDivisionError; <=0 means "never run periodic".
        if self.config.periodic_interval > 0 and self._task_counter % self.config.periodic_interval == 0:
            periodic_result = self.run_periodic(context)
            logger.info(f"Periodic management triggered at task #{self._task_counter}: {periodic_result}")

        return result

    def run_periodic(self, context: Optional[Dict[str, Any]] = None) -> ManagementResult:
        """Run periodic maintenance operations."""
        context = context or {}
        result = ManagementResult(phase="periodic")
        start = time.time()

        for op in self._periodic_ops:
            op_result = self._run_single_op(op, context)
            result.results.append(op_result)

        result.total_duration_ms = (time.time() - start) * 1000
        self._recent_results.append(result)
        return result

    def run_on_insert(
        self,
        new_units: List,
        context: Optional[Dict[str, Any]] = None,
    ) -> ManagementResult:
        """Run on-insert operations for newly added memories."""
        context = context or {}
        context["new_unit_ids"] = [u.id for u in new_units]

        result = ManagementResult(phase="on_insert")
        start = time.time()

        for op in self._on_insert_ops:
            op_result = self._run_single_op(op, context)
            result.results.append(op_result)

        result.total_duration_ms = (time.time() - start) * 1000
        self._recent_results.append(result)
        return result

    def consume_recent_results(self) -> List[ManagementResult]:
        """Return and clear recently executed management results."""
        results = list(self._recent_results)
        self._recent_results = []
        return results

    def clear_recent_results(self) -> None:
        """Clear buffered management results."""
        self._recent_results = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single_op(self, op: BaseManageOp, context: Dict[str, Any]) -> OpResult:
        """Execute a single op with timing and error handling."""
        start = time.time()
        try:
            op_result = op.execute(context)
            op_result.duration_ms = (time.time() - start) * 1000
            if op_result.triggered:
                logger.debug(f"Op '{op.op_name}': {op_result}")
            return op_result
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Op '{op.op_name}' failed: {e}", exc_info=True)
            return OpResult(
                op_name=op.op_name,
                triggered=False,
                duration_ms=duration,
                details={"error": str(e)},
            )
