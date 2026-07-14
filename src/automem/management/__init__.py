"""
Memory Management Module.

Provides pluggable management operations that maintain memory quality through:
  - Episodic consolidation (clustering, workflow extraction, cross-task generalization)
  - Deduplication (signature, semantic, cross-type, conflict detection)
  - Failure-driven adjustment (penalize, boost, reflection, dynamic discard)
  - Maintenance (access stats, time decay, pruning, quality curation)

Operations are composed into pipelines via ManagementConfig and orchestrated
by ManagementPipeline.

Usage:
    from automem.management import ManagementPipeline, ManagementConfig
    from automem.management.presets import get_preset

    config = get_preset("lightweight")
    pipeline = ManagementPipeline(store, config, embedding_model, llm_client)
    pipeline.run_post_task({"task_succeeded": True, "used_unit_ids": [...]})
"""

from .base_op import (
    BaseManageOp,
    ManagementConfig,
    ManagementResult,
    OpResult,
    StorageCompatibility,
    TriggerType,
)
from .pipeline import ManagementPipeline, get_op_registry
from .preset_registry import (
    PresetCapabilities,
    get_preset_capabilities,
    normalize_preset_name,
    validate_preset_capabilities,
)
from .presets import get_preset, list_presets

__all__ = [
    # Base types
    "BaseManageOp",
    "OpResult",
    "ManagementConfig",
    "ManagementResult",
    "StorageCompatibility",
    "TriggerType",
    # Pipeline
    "ManagementPipeline",
    "get_op_registry",
    # Presets
    "get_preset",
    "get_preset_capabilities",
    "list_presets",
    "normalize_preset_name",
    "PresetCapabilities",
    "validate_preset_capabilities",
]
