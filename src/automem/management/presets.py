"""
Preset pipeline configurations for memory lifecycle operations.

Two-layer design (refactored 2026-04-24):
  - MANDATORY_* ops are always present in every preset. They form the
    quality floor against which every architecture is evaluated.
  - Each preset's body declares only the EXPERIMENTAL ops it adds on top
    of the mandatory layer.

The refactor preserves the *set* of ops per preset (verified by smoke test);
ordering is normalized to:  [stats updates] + [experimental] + [cleanup]
for post_task, and [experimental] + [mandatory cleanup] for on_insert.
"""

from .base_op import ManagementConfig
from .preset_registry import (
    PRESET_CAPABILITIES,
    PUBLIC_PRESET_NAMES,
    get_preset_capabilities,
    normalize_preset_name,
)

# ============================================================
# MANDATORY OPS — always run, regardless of preset choice
# ============================================================
# Rationale (see MODULE_AUDIT_REPORT.md §2):
#   - access_stats_update / boost_on_success / penalize_on_failure : RL signals
#       and usage counters; near-zero cost, prerequisite for U1 bandit + U3 utility
#   - signature_dedup    : exact-string dedup; never harmful, idempotent
#   - quality_curation   : light filter; appears in every well-performing preset
#   - conflict_detection : on-insert pollution check; cheap
MANDATORY_POST_TASK_PRE = [
    "access_stats_update",
    "boost_on_success",
    "penalize_on_failure",
]
MANDATORY_POST_TASK_POST = [
    "signature_dedup",
]
MANDATORY_PERIODIC = [
    "quality_curation",
    # 2026-04-27: utility_audit (idea 3 step A) is now mandatory. It
    # deactivates units with empirical success_rate below threshold,
    # requiring at least min_usage_count retrievals before judging. Cheap
    # (no LLM/embedding), single-pass over active units.
    "utility_audit",
    # 2026-04-28: size_capped_prune — hard cap on active pool size. No-op
    # below cap; emergency aggressive pruning above. Cheap (sort + flag
    # is_active=False). Trips only when soft tools fail to keep the pool
    # lean, preventing long-term retrieval-noise growth.
    "size_capped_prune",
]
MANDATORY_ON_INSERT = [
    "signature_dedup",
    "conflict_detection",
]


def _build(
    post_task=(),
    periodic=(),
    on_insert=(),
    interval: int = 10,
) -> ManagementConfig:
    """Compose mandatory + preset-specific ops, dedup preserving canonical order.

    post_task ordering: [stats updates] + [preset experimental] + [cleanup]
    periodic ordering : [preset experimental] + [mandatory quality_curation]
    on_insert ordering: [preset experimental] + [mandatory dedup/conflict checks]
    """
    def _merge(*lists):
        seen, out = set(), []
        for lst in lists:
            for op in lst:
                if op not in seen:
                    seen.add(op)
                    out.append(op)
        return out

    return ManagementConfig(
        post_task_ops=_merge(MANDATORY_POST_TASK_PRE, post_task, MANDATORY_POST_TASK_POST),
        periodic_ops=_merge(periodic, MANDATORY_PERIODIC),
        on_insert_ops=_merge(on_insert, MANDATORY_ON_INSERT),
        periodic_interval=interval,
    )


# ============================================================
# Preset definitions — each adds experimental ops to the mandatory floor
# ============================================================

def lightweight() -> ManagementConfig:
    """Minimal experimental ops on top of the mandatory floor.

    Empirical winner in past 33 evals: highest mean lift (+0.089), lowest
    stddev (0.033). Recommended default when in doubt.
    """
    return _build(
        post_task=[],
        periodic=["time_decay", "score_based_prune"],
        on_insert=[],
    )


def json_full() -> ManagementConfig:
    """Full management for JsonStorage — includes LLM-based correction + clustering.

    Stage-1 (2026-05-17): added ``llm_conflict_resolve`` to periodic ops so
    pairs flagged by ``conflict_detection`` actually get pruned via LLM judge
    (rather than only flagged). Bounded to 5 LLM calls/run.
    """
    return _build(
        post_task=["reflection_correction"],
        periodic=[
            "time_decay",
            "semantic_dedup",
            "cross_type_dedup",
            "cluster_merge",
            "trajectory_to_workflow",
            "llm_conflict_resolve",   # Stage-1 adoption
            "score_based_prune",
            "reindex_relations",
        ],
        on_insert=[],
    )


def tool_manager() -> ManagementConfig:
    """Tool-Manager-inspired: promote reusable SHORTCUT units to first-class callables.

    Under-explored in past evals (n=0). Best with extract_types containing
    'shortcut'. Does NOT require graph storage (only SHORTCUT-extract
    machinery), making it the lightest "experimental" preset.
    """
    return _build(
        post_task=["shortcut_promotion"],
        periodic=[
            "time_decay",
            "semantic_dedup",
            "trajectory_to_workflow",
            "score_based_prune",
        ],
        on_insert=["shortcut_validation"],
    )


def graph_consolidate() -> ManagementConfig:
    """Graph content consolidation with success-aware edge adaptation.

    The preset intentionally excludes shortcut promotion/validation; those
    operations belong exclusively to ``tool_manager``.  Unit cleanup runs
    before relation rebuilding, and edge weights are optimized last so the
    adaptive value is not overwritten by ``reindex_relations``.
    """
    return _build(
        post_task=["edge_stats_update", "reflection_correction"],
        periodic=[
            "time_decay",
            "semantic_dedup",
            "cross_type_dedup",
            "cluster_merge",
            "trajectory_to_workflow",
            "cross_task_generalize",
            "llm_conflict_resolve",
            "score_based_prune",
            *MANDATORY_PERIODIC,
            "reindex_relations",
            "edge_weight_optimize",
        ],
        on_insert=[],
    )


# ============================================================
# Preset lookup
# ============================================================
_PRESET_BUILDERS = {
    "lightweight": lightweight,
    "json_full": json_full,
    "tool_manager": tool_manager,
    "graph_consolidate": graph_consolidate,
}

if set(_PRESET_BUILDERS) != set(PRESET_CAPABILITIES):
    raise RuntimeError("Preset builders and capability registry are out of sync")

# Storage type -> default preset mapping
_STORAGE_DEFAULT_PRESET = {
    "json": "lightweight",
    "vector": "lightweight",
    "hybrid": "lightweight",
    "graph": "graph_consolidate",
    "llm_graph": "graph_consolidate",
}


def get_preset(name_or_storage_type: str) -> ManagementConfig:
    """
    Get a preset ManagementConfig by name or storage type.

    Args:
        name_or_storage_type: Either a canonical preset name ("lightweight", etc.)
            or a storage type ("json", "graph", etc.) to auto-select the default preset.

    Returns:
        ManagementConfig instance.
    """
    raw_name = str(name_or_storage_type or "").strip().lower()
    preset_name = _STORAGE_DEFAULT_PRESET.get(raw_name)
    if preset_name is None:
        preset_name = normalize_preset_name(raw_name)
    return _PRESET_BUILDERS[preset_name]()


def list_presets() -> list:
    """Return public canonical preset names only."""
    return list(PUBLIC_PRESET_NAMES)


__all__ = [
    "get_preset",
    "get_preset_capabilities",
    "graph_consolidate",
    "json_full",
    "lightweight",
    "list_presets",
    "normalize_preset_name",
    "tool_manager",
]
