"""
Management operation classes (post 2026-05-13 cleanup).

Removed (0% trigger in run9 audit, see audit notes in presets.py):
  - DynamicDiscardOp: min_usage=5 never reached in 50-task candidate batches
  - CaseRewriteOp:    LLM rewrite never fired in 19 calls (Memento-style)

Groups (current):
  - Episodic consolidation: ClusterMerge, TrajectoryToWorkflow, CrossTaskGeneralize, ReindexRelations
  - Deduplication: SignatureDedup, SemanticDedup, CrossTypeDedup, ConflictDetection
  - Failure adjustment: PenalizeOnFailure, BoostOnSuccess, ReflectionCorrection
  - Maintenance: AccessStatsUpdate, TimeDecay, ScoreBasedPrune, QualityCuration, UtilityAudit, SizeCappedPrune
  - Tool Manager inspired: ShortcutPromotion, ShortcutValidation
"""

# Episodic consolidation
from .cluster_merge import ClusterMergeOp
from .trajectory_to_workflow import TrajectoryToWorkflowOp
from .cross_task_generalize import CrossTaskGeneralizeOp
from .reindex_relations import ReindexRelationsOp

# Deduplication
from .signature_dedup import SignatureDedupOp
from .semantic_dedup import SemanticDedupOp
from .cross_type_dedup import CrossTypeDedupOp
from .conflict_detection import ConflictDetectionOp
# Stage-1 (2026-05-17) adoption — LLM-based conflict resolution
from .llm_conflict_resolve import LLMConflictResolveOp

# Failure adjustment
from .penalize_on_failure import PenalizeOnFailureOp
from .boost_on_success import BoostOnSuccessOp
from .reflection_correction import ReflectionCorrectionOp

# Maintenance
from .access_stats_update import AccessStatsUpdateOp
from .time_decay import TimeDecayOp
from .score_based_prune import ScoreBasedPruneOp
from .quality_curation import QualityCurationOp

# Tool Manager inspired
from .shortcut_promotion import ShortcutPromotionOp
from .shortcut_validation import ShortcutValidationOp

# Graph-adaptive edge feedback (G1, 2026-07-11 — ported from cerebra_fusion)
from .edge_stats_update import EdgeStatsUpdateOp
from .edge_weight_optimize import EdgeWeightOptimizeOp

__all__ = [
    "ClusterMergeOp",
    "TrajectoryToWorkflowOp",
    "CrossTaskGeneralizeOp",
    "ReindexRelationsOp",
    "SignatureDedupOp",
    "SemanticDedupOp",
    "CrossTypeDedupOp",
    "ConflictDetectionOp",
    "LLMConflictResolveOp",
    "PenalizeOnFailureOp",
    "BoostOnSuccessOp",
    "ReflectionCorrectionOp",
    "AccessStatsUpdateOp",
    "TimeDecayOp",
    "ScoreBasedPruneOp",
    "QualityCurationOp",
    "ShortcutPromotionOp",
    "ShortcutValidationOp",
    "EdgeStatsUpdateOp",
    "EdgeWeightOptimizeOp",
]
