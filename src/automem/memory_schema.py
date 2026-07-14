"""
Unified Memory Unit Schema for RL-based Memory Management.

Design principles:
  1. Strict envelope (identity, lifecycle, quality, cost, scope) + flexible content (Dict)
  2. All RL-observable/controllable fields are explicit, numeric, top-level
  3. Downcastable to existing MemoryItem for retrieval compatibility
  4. Atomic granularity: one MemoryUnit = one independently manageable piece of knowledge
     - Insight  → one per failed task   (already atomic from extraction)
     - Tip      → one per individual tip (split from extraction batch)
     - Trajectory → one per task         (steps are a coherent sequence)
     - Workflow → one per task           (steps are a coherent sequence)
     - Shortcut → one per individual macro (split from extraction batch)

Usage:
  from automem.memory_schema import MemoryUnit, MemoryUnitType, RelationType

  unit = MemoryUnit(
      type=MemoryUnitType.TIP,
      content={"topic": "...", "principle": "...", ...},
      source_task_id="04a04a9b-...",
      source_task_query="If we assume all articles...",
      task_outcome="success",
      extraction_model="qwen3-max",
  )
  unit.compute_signature()

  # Persist
  d = unit.to_dict()
  unit2 = MemoryUnit.from_dict(d)

  # Feed to RL policy network
  state_vec = unit.to_rl_state()  # np.ndarray, shape (16,)

  # Pass to retrieval layer
  item = unit.to_memory_item(score=0.87)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import hashlib
import json
import uuid

import numpy as np

from automem.memory_types import MemoryItem, MemoryItemType


# ============================================================
# Enums
# ============================================================

class MemoryUnitType(Enum):
    """Atomic memory unit types, one per extraction prompt module."""
    INSIGHT    = "insight"      # Failure diagnostics    (failure-only)
    TIP        = "tip"          # Cognitive heuristics   (success + failure)
    TRAJECTORY = "trajectory"   # Compressed action-observation chain
    WORKFLOW   = "workflow"     # Orchestration logic    (success-only)
    SHORTCUT   = "shortcut"     # Executable macros      (success + failure)


class RelationType(Enum):
    """Edge types between memory units."""
    SIMILAR    = "similar"      # Semantically related content
    DEPENDS    = "depends"      # A requires knowledge from B
    CONFLICTS  = "conflicts"    # A contradicts B
    SUPERSEDES = "supersedes"   # A replaces B (newer / higher confidence)
    COOCCURS   = "cooccurs"     # Extracted from same task
    REINFORCES = "reinforces"   # A supports / strengthens B

    # --- Relation-First paradigm (new) ---
    ABOUT        = "about"         # memory → entity: primary topic
    MENTIONS     = "mentions"      # memory → entity: secondary reference
    SUPPORTS     = "supports"      # memory → memory: mutual reinforcement
    CONTRADICTS  = "contradicts"   # memory → memory: conflict signal
    SEQUENCE     = "sequence"      # memory → memory: execution ordering
    GENERALIZES  = "generalizes"   # memory → memory: abstraction hierarchy
    CO_OCCURS    = "co_occurs"     # entity → entity: co-appearance pattern


# ============================================================
# Relation (lightweight edge, stored on the unit itself)
# ============================================================

@dataclass
class MemoryRelation:
    """Directed edge from the owning MemoryUnit to target_id."""
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    annotation: str = ""  # LLM-generated edge description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "annotation": self.annotation,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryRelation":
        return MemoryRelation(
            target_id=d["target_id"],
            relation_type=RelationType(d["relation_type"]),
            weight=d.get("weight", 1.0),
            annotation=d.get("annotation", ""),
        )


@dataclass
class EntityNode:
    """An entity extracted from trajectory, used as a graph node."""
    name: str
    entity_type: str  # "tool", "concept", "resource", "person", "organization", "location"
    description: str = ""
    source_task_id: str = ""
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    mention_count: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_task_id": self.source_task_id,
            "first_seen": self.first_seen,
            "mention_count": self.mention_count,
            "attributes": self.attributes,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EntityNode":
        return EntityNode(
            name=d["name"],
            entity_type=d.get("entity_type", "concept"),
            description=d.get("description", ""),
            source_task_id=d.get("source_task_id", ""),
            first_seen=d.get("first_seen", datetime.now().isoformat()),
            mention_count=d.get("mention_count", 1),
            attributes=d.get("attributes", {}),
        )


@dataclass
class ExtractionResult:
    """Complete extraction output from a single trajectory processing."""
    memory_units: List["MemoryUnit"] = field(default_factory=list)
    entity_nodes: List[EntityNode] = field(default_factory=list)
    relations: List[MemoryRelation] = field(default_factory=list)
    extract_plan_used: Dict[str, Any] = field(default_factory=dict)
    extraction_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_units": [u.to_dict() for u in self.memory_units],
            "entity_nodes": [e.to_dict() for e in self.entity_nodes],
            "relations": [r.to_dict() for r in self.relations],
            "extract_plan_used": self.extract_plan_used,
            "extraction_time_ms": self.extraction_time_ms,
        }


# ============================================================
# Core Memory Unit
# ============================================================

@dataclass
class MemoryUnit:
    """
    Universal memory unit for RL-based memory management.

    Field groups
    ────────────
    Identity      id, type, signature
    Content       content  (Dict — type-specific, kept flexible)
    Source        source_task_id, source_task_query, task_outcome, extraction_model
    Quality       confidence, usage_count, success_count   ← RL-observable & adjustable
    Lifecycle     created_at, last_accessed, access_count,
                  decay_weight, is_active                  ← RL-observable & adjustable
    Cost          storage_tokens
    Scope         applicable_domains, applicable_task_types
    Relations     relations  (List[MemoryRelation])
    Embedding     embedding  (np.ndarray, 384-d by default)

    Content schemas per type (extraction prompt output → atomic unit)
    ─────────────────────────────────────────────────────────────────
    INSIGHT:
        root_cause_conclusion: str
        state_mismatch_analysis: str          # "Expected: X; Actual: Y"
        divergence_point: str                 # "[Step N - Action]: ..."
        knowledge_graph: List[List[str]]      # [[S, P, O], ...]

    TIP:
        category: str                         # "planning_and_decision" | "tool_and_search"
        topic: str
        principle: str
        micro_example: str
        counterfactual: str

    TRAJECTORY:
        steps: List[Dict]                     # [{step_id, action, observation}, ...]
        task_outcome: str
        failure_reason: Optional[str]

    WORKFLOW:
        agent_workflow: List[Dict]            # [{step, action, rationale, generalized_execution}]
        search_workflow: List[Dict]           # [{step, query_formulation, validation_criteria}]

    SHORTCUT:
        name: str
        description: str
        precondition: str
        extraction_type: str
        assumptions: List[str]
    """

    # === Identity ==========================================================
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryUnitType = MemoryUnitType.TIP
    signature: str = ""

    # === Content (type-specific, flexible Dict) ============================
    content: Dict[str, Any] = field(default_factory=dict)

    # === Source / Provenance ================================================
    source_task_id: str = ""
    source_task_query: str = ""
    task_outcome: str = ""                    # "success" | "failure"
    extraction_model: str = ""

    # === Quality (RL-observable & adjustable) ===============================
    confidence: float = 1.0                   # [0, 1]  — RL action: adjust
    usage_count: int = 0                      # Retrieved & presented to agent
    success_count: int = 0                    # Task succeeded after using this

    # === Lifecycle (RL-observable & adjustable) =============================
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None
    access_count: int = 0
    decay_weight: float = 1.0                 # [0, 1]  — RL action: decay speed
    is_active: bool = True                    # False   — RL action: forget (soft)

    # === Cost ==============================================================
    storage_tokens: int = 0

    # === Scope =============================================================
    applicable_domains: List[str] = field(default_factory=list)
    applicable_task_types: List[str] = field(default_factory=list)
    # When-to-use guidance (added 2026-04-28). These are observable trigger /
    # anti-trigger conditions emitted by extraction, consumed by:
    #   (a) LLM Judge to decide keep/drop based on applicability, not just content
    #   (b) Injection layer to render an explicit "Apply when:" hint to the agent
    #   (c) INJECTION_BAD attribution when agent ignored a kept memory
    use_when: List[str] = field(default_factory=list)
    avoid_when: List[str] = field(default_factory=list)
    # Negative-example flag (added 2026-04-28). True for trajectory units
    # extracted from FAILED tasks. Such units are kept in the pool for
    # diagnostic value but must be flagged at injection time so the agent
    # treats them as "what NOT to do" rather than templates to imitate.
    is_negative_example: bool = False
    # Conflict signal (added 2026-04-28). Incremented by conflict_detection
    # op every time this unit is found in a CONFLICTS pair. Consumed by the
    # MEMORY_STALE attribution branch — units with conflict_count > 0 are
    # treated as stale even if still active. Also lets utility_audit prefer
    # pruning conflicted units.
    conflict_count: int = 0
    superseded_by: Optional[str] = None

    # === Relations =========================================================
    relations: List[MemoryRelation] = field(default_factory=list)

    # === Embedding =========================================================
    embedding: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def success_rate(self) -> float:
        """Success rate when this memory was used. Returns 0 if never used."""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    @property
    def age_hours(self) -> float:
        """Hours since creation."""
        try:
            created = datetime.fromisoformat(self.created_at)
            return (datetime.now() - created).total_seconds() / 3600.0
        except (ValueError, TypeError):
            return 0.0

    @property
    def hours_since_access(self) -> float:
        """Hours since last retrieval. Returns age_hours if never accessed."""
        if self.last_accessed is None:
            return self.age_hours
        try:
            accessed = datetime.fromisoformat(self.last_accessed)
            return (datetime.now() - accessed).total_seconds() / 3600.0
        except (ValueError, TypeError):
            return self.age_hours

    @property
    def effective_score(self) -> float:
        """Composite score combining confidence, success rate, and recency decay."""
        recency = 1.0 / (1.0 + self.hours_since_access / 168.0)  # 168h = 1 week half-life
        return self.confidence * (0.5 + 0.5 * self.success_rate) * (self.decay_weight * recency)

    # -----------------------------------------------------------------------
    # Mutations (called by retrieval / management layers)
    # -----------------------------------------------------------------------

    def record_access(self) -> None:
        """Update access metadata.

        Canonical call site (since 2026-07-11): the post-task
        access_stats_update management op, for units that were actually
        INJECTED. Retrievers must NOT call this — retrieval-layer bumping
        counted a single hybrid injection up to 4 times (each sub-arm +
        fusion + post-task) and kept gate-dropped noise units alive via
        last_accessed refreshes. The modular provider keeps
        its own self-contained call.
        """
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()

    def record_outcome(self, task_succeeded: bool) -> None:
        """Update usage/success counts after a task completes."""
        self.usage_count += 1
        if task_succeeded:
            self.success_count += 1

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def compute_signature(self) -> str:
        """Compute content hash for deduplication. Sets and returns self.signature."""
        raw = json.dumps(self.content, sort_keys=True, ensure_ascii=False)
        self.signature = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.signature

    def content_text(self) -> str:
        """Flatten content dict to a single text string for embedding.

        Fields are ordered by retrieval relevance for each memory type:
        high-signal fields (principle, applicability, key_decision) are
        placed first so they dominate the embedding vector.
        """
        # Priority field order per type — most retrieval-relevant first.
        # Fields not listed here are appended afterwards in insertion order.
        _PRIORITY: Dict[MemoryUnitType, List[str]] = {
            MemoryUnitType.TIP: [
                "principle", "applicability", "topic", "counterfactual", "micro_example",
            ],
            MemoryUnitType.INSIGHT: [
                "root_cause_conclusion", "applicability", "failure_pattern",
                "state_mismatch_analysis", "divergence_point",
            ],
            MemoryUnitType.TRAJECTORY: [
                "key_decision", "critical_observation", "tool_strategy",
            ],
            MemoryUnitType.WORKFLOW: [],   # handled inline below
            MemoryUnitType.SHORTCUT: [
                "description", "precondition",
            ],
        }

        priority = _PRIORITY.get(self.type, [])
        seen_keys: set = set()
        parts: List[str] = []

        def _add_str(v: Any) -> None:
            s = str(v).strip()
            if s:
                parts.append(s)

        # 1. Priority fields first
        for key in priority:
            v = self.content.get(key)
            if v is None:
                continue
            seen_keys.add(key)
            if isinstance(v, str):
                _add_str(v)
            elif isinstance(v, list):
                for item in v:
                    _add_str(item) if isinstance(item, str) else None

        # 2. Remaining fields in insertion order
        for key, v in self.content.items():
            if key in seen_keys:
                continue
            if isinstance(v, str):
                _add_str(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        _add_str(item)
                    elif isinstance(item, dict):
                        if self.type == MemoryUnitType.WORKFLOW:
                            # For workflow steps prioritize decision-relevant sub-fields
                            for sub_key in ("action", "rationale", "query_formulation",
                                            "validation_criteria", "generalized_execution"):
                                if sub_key in item and isinstance(item[sub_key], str):
                                    _add_str(item[sub_key])
                        else:
                            for val in item.values():
                                if isinstance(val, str):
                                    _add_str(val)

        # Append use_when triggers so vector retrieval can match on
        # applicability conditions, not just memory content.
        for trigger in self.use_when:
            _add_str(trigger)

        return " ".join(parts)

    def token_estimate(self) -> int:
        """Rough token count estimate (1 token ≈ 4 chars)."""
        text = self.content_text()
        self.storage_tokens = max(1, len(text) // 4)
        return self.storage_tokens

    # -----------------------------------------------------------------------
    # RL interface
    # -----------------------------------------------------------------------

    # Type one-hot dimension order (for to_rl_state)
    _TYPE_INDEX = {
        MemoryUnitType.INSIGHT:    0,
        MemoryUnitType.TIP:        1,
        MemoryUnitType.TRAJECTORY: 2,
        MemoryUnitType.WORKFLOW:   3,
        MemoryUnitType.SHORTCUT:   4,
    }

    def to_rl_state(self) -> np.ndarray:
        """
        Convert to a fixed-size numeric vector for the RL policy network.

        Layout (16 dims):
          [0:5]   type one-hot            (5)
          [5]     confidence              (1)
          [6]     usage_count (log-scaled)(1)
          [7]     success_rate            (1)
          [8]     age_hours (log-scaled)  (1)
          [9]     hours_since_access (log)(1)
          [10]    access_count (log)      (1)
          [11]    decay_weight            (1)
          [12]    is_active (0/1)         (1)
          [13]    storage_tokens (log)    (1)
          [14]    num_relations           (1)
          [15]    task_outcome (1=success) (1)
        """
        vec = np.zeros(16, dtype=np.float32)

        # type one-hot
        idx = self._TYPE_INDEX.get(self.type, 0)
        vec[idx] = 1.0

        # quality
        vec[5] = self.confidence
        vec[6] = np.log1p(self.usage_count)
        vec[7] = self.success_rate

        # lifecycle
        vec[8]  = np.log1p(self.age_hours)
        vec[9]  = np.log1p(self.hours_since_access)
        vec[10] = np.log1p(self.access_count)
        vec[11] = self.decay_weight
        vec[12] = 1.0 if self.is_active else 0.0

        # cost
        vec[13] = np.log1p(self.storage_tokens)

        # relations
        vec[14] = float(len(self.relations))

        # source
        vec[15] = 1.0 if self.task_outcome == "success" else 0.0

        return vec

    # -----------------------------------------------------------------------
    # Compatibility with existing MemoryItem
    # -----------------------------------------------------------------------

    def to_memory_item(self, score: Optional[float] = None) -> MemoryItem:
        """Downcast to MemoryItem for the retrieval interface."""
        return MemoryItem(
            id=self.id,
            content=self.content_text(),
            metadata={
                "type": self.type.value,
                "source_task_id": self.source_task_id,
                "confidence": self.confidence,
                "success_rate": self.success_rate,
                "decay_weight": self.decay_weight,
            },
            score=score if score is not None else self.effective_score,
            type=MemoryItemType.TEXT,
        )

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict (embedding stored as list)."""
        return {
            "id": self.id,
            "type": self.type.value,
            "signature": self.signature,
            "content": self.content,
            "source_task_id": self.source_task_id,
            "source_task_query": self.source_task_query,
            "task_outcome": self.task_outcome,
            "extraction_model": self.extraction_model,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "decay_weight": self.decay_weight,
            "is_active": self.is_active,
            "storage_tokens": self.storage_tokens,
            "applicable_domains": self.applicable_domains,
            "applicable_task_types": self.applicable_task_types,
            "use_when": self.use_when,
            "avoid_when": self.avoid_when,
            "is_negative_example": self.is_negative_example,
            "conflict_count": self.conflict_count,
            "superseded_by": self.superseded_by,
            "relations": [r.to_dict() for r in self.relations],
            "embedding": (
                self.embedding.tolist()
                if isinstance(self.embedding, np.ndarray)
                else self.embedding
            ) if self.embedding is not None else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryUnit":
        """Deserialize from dict."""
        emb = d.get("embedding")
        return MemoryUnit(
            id=d.get("id", str(uuid.uuid4())),
            type=MemoryUnitType(d["type"]),
            signature=d.get("signature", ""),
            content=d.get("content", {}),
            source_task_id=d.get("source_task_id", ""),
            source_task_query=d.get("source_task_query", ""),
            task_outcome=d.get("task_outcome", ""),
            extraction_model=d.get("extraction_model", ""),
            confidence=d.get("confidence", 1.0),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
            created_at=d.get("created_at", datetime.now().isoformat()),
            last_accessed=d.get("last_accessed"),
            access_count=d.get("access_count", 0),
            decay_weight=d.get("decay_weight", 1.0),
            is_active=d.get("is_active", True),
            storage_tokens=d.get("storage_tokens", 0),
            applicable_domains=d.get("applicable_domains", []),
            applicable_task_types=d.get("applicable_task_types", []),
            use_when=d.get("use_when", []),
            avoid_when=d.get("avoid_when", []),
            is_negative_example=d.get("is_negative_example", False),
            conflict_count=d.get("conflict_count", 0),
            superseded_by=d.get("superseded_by"),
            relations=[MemoryRelation.from_dict(r) for r in d.get("relations", [])],
            embedding=np.array(emb, dtype=np.float32) if emb is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"MemoryUnit(id={self.id[:8]}..., type={self.type.value}, "
            f"confidence={self.confidence:.2f}, usage={self.usage_count}, "
            f"success_rate={self.success_rate:.2f}, active={self.is_active})"
        )


# ============================================================
# Batch splitter: extraction output → atomic MemoryUnits
# ============================================================

import logging as _logging
_split_logger = _logging.getLogger(__name__)

import numbers as _numbers

# Phrases that mark a degenerate trajectory (the extractor saw no real trace,
# only the final answer) — such units get clamped to a low confidence.
_DEGENERATE_TRAJ_MARKERS = (
    "no raw action trace",
    "only the final answer",
    "raw trajectory log is empty",
    "no recoverable retrieval",
    "raw trajectory contains no",
)


def _compute_initial_confidence(content: Dict[str, Any], unit_type: "MemoryUnitType") -> float:
    """Heuristic initial confidence so freshly-extracted units are not all 1.0.

    Combines three signals available at creation time:
      1. LLM self-rating  — the tips prompt already emits `quality_self_score`
         (0-1); honoured here (also accepts `confidence`/`quality`).
      2. Completeness      — fraction of high-signal fields (use_when,
         key_decision, principle, critical_observation, reusable_anchor) that
         are actually filled, scaled into [0.6, 1.0].
      3. Trajectory degeneracy — a single-step / answer-only stub is clamped to
         <=0.2 so it ranks below real multi-step traces.

    Returns a value in [0.05, 1.0].
    """
    if not isinstance(content, dict):
        return 0.6

    # (1) LLM self-rating
    self_score = None
    for k in ("quality_self_score", "confidence", "quality"):
        v = content.get(k)
        if isinstance(v, _numbers.Number) and 0.0 <= float(v) <= 1.0:
            self_score = float(v)
            break
    base = self_score if self_score is not None else 0.65

    # (2) completeness of high-signal fields actually present in content
    signal_fields = (
        "use_when", "key_decision", "principle",
        "critical_observation", "reusable_anchor",
    )
    total = filled = 0
    for k in signal_fields:
        if k not in content:
            continue
        total += 1
        v = content[k]
        if isinstance(v, (list, tuple, dict)):
            filled += 1 if len(v) > 0 else 0
        elif isinstance(v, str):
            filled += 1 if v.strip() else 0
        elif v:
            filled += 1
    completeness = 1.0 if total == 0 else (0.6 + 0.4 * filled / total)

    conf = base * completeness

    # (3) process-richness signal for the two procedural types: a stub with no
    # real steps is clamped low; a richer trace/workflow is mildly rewarded.
    if unit_type == MemoryUnitType.TRAJECTORY:
        steps = content.get("steps", [])
        n_steps = len(steps) if isinstance(steps, list) else 0
        blob = json.dumps(content, ensure_ascii=False).lower()
        degenerate = n_steps <= 1 or any(m in blob for m in _DEGENERATE_TRAJ_MARKERS)
        if degenerate:
            conf = min(conf, 0.2)
        else:
            conf = conf * (1.0 + 0.08 * min(n_steps, 6) / 6.0)  # saturates ~6 steps

    elif unit_type == MemoryUnitType.WORKFLOW:
        wf = content.get("agent_workflow") or content.get("search_workflow") or []
        n_steps = len(wf) if isinstance(wf, list) else 0
        blob = json.dumps(content, ensure_ascii=False).lower()
        degenerate = n_steps == 0 or any(m in blob for m in _DEGENERATE_TRAJ_MARKERS)
        if degenerate:
            conf = min(conf, 0.3)
        else:
            conf = conf * (1.0 + 0.10 * min(n_steps, 5) / 5.0)  # saturates ~5 steps

    return max(0.05, min(1.0, conf))


def split_extraction_output(
    extraction_result: Dict[str, Any],
    unit_type: MemoryUnitType,
    source_task_id: str,
    source_task_query: str,
    task_outcome: str,
    extraction_model: str = "",
) -> List[MemoryUnit]:
    """
    Split a raw extraction output (from LLM) into atomic MemoryUnits.

    Handles the fact that some extraction types produce batches:
      - Tips:      {planning_and_decision_tips: [...], tool_and_search_tips: [...]}
                   → one MemoryUnit per individual tip
      - Shortcuts: [macro1, macro2, ...]
                   → one MemoryUnit per macro
      - Others:    already atomic, wrapped as-is

    Source constraint enforcement (defence-in-depth):
      - WORKFLOW units must originate from success tasks.
      - INSIGHT units must originate from failure tasks.
      Violations are logged and an empty list is returned.
    """
    # --- Source constraint validation (defence-in-depth) ---
    if unit_type == MemoryUnitType.WORKFLOW and task_outcome != "success":
        _split_logger.warning(
            "split_extraction_output: WORKFLOW unit rejected — "
            "task_outcome=%r (must be 'success'). task_id=%s",
            task_outcome, source_task_id,
        )
        return []
    if unit_type == MemoryUnitType.INSIGHT and task_outcome != "failure":
        _split_logger.warning(
            "split_extraction_output: INSIGHT unit rejected — "
            "task_outcome=%r (must be 'failure'). task_id=%s",
            task_outcome, source_task_id,
        )
        return []

    common = dict(
        source_task_id=source_task_id,
        source_task_query=source_task_query,
        task_outcome=task_outcome,
        extraction_model=extraction_model,
    )

    units: List[MemoryUnit] = []

    if unit_type == MemoryUnitType.TIP:
        # 4 categories (added file_handling_tips 2026-04-28).
        for category in (
            "planning_and_decision_tips",
            "tool_and_search_tips",
            "answer_format_tips",
            "file_handling_tips",
        ):
            cat_label = category.replace("_tips", "")
            for item in extraction_result.get(category, []):
                content = dict(item)
                content["category"] = cat_label
                u = MemoryUnit(type=unit_type, content=content, **common)
                u.compute_signature()
                u.token_estimate()
                units.append(u)

    elif unit_type == MemoryUnitType.SHORTCUT:
        items = extraction_result if isinstance(extraction_result, list) else [extraction_result]
        for item in items:
            u = MemoryUnit(type=unit_type, content=dict(item), **common)
            u.compute_signature()
            u.token_estimate()
            units.append(u)

    elif unit_type == MemoryUnitType.TRAJECTORY:
        if isinstance(extraction_result, list):
            # Legacy format: bare list of steps (no summary fields)
            steps = extraction_result
            content = {
                "steps": steps,
                "task_outcome": task_outcome,
                "key_decision": "",
                "tool_strategy": "",
                "critical_observation": "",
            }
        else:
            # New format: dict with steps + summary fields. Start from full
            # extraction_result so newly-added top-level fields like
            # use_when / avoid_when / leakage_risk / reusable_anchor flow
            # through; then overwrite the canonical 5 fields to enforce shape.
            content = dict(extraction_result)
            steps = extraction_result.get("steps", [])
            content["steps"] = steps
            content["task_outcome"] = task_outcome
            content.setdefault("key_decision", "")
            content.setdefault("tool_strategy", "")
            content.setdefault("critical_observation", "")
        u = MemoryUnit(type=unit_type, content=content, **common)
        # Flag failure trajectories as negative examples so injection layer
        # can warn the agent NOT to imitate them.
        if task_outcome != "success":
            u.is_negative_example = True
        u.compute_signature()
        u.token_estimate()
        units.append(u)

    else:
        # INSIGHT, WORKFLOW — already atomic
        u = MemoryUnit(type=unit_type, content=dict(extraction_result), **common)
        # W2 provenance: INSIGHT is by definition a failure-mode lesson.
        # Mark explicitly so the injection layer renders it as 'past
        # failure to avoid' rather than 'past success to imitate'.
        if unit_type == MemoryUnitType.INSIGHT:
            u.is_negative_example = True
        u.compute_signature()
        u.token_estimate()
        units.append(u)

    # Promote content-level tags to top-level applicable_task_types
    # Also promote use_when / avoid_when emitted by extraction prompts
    # (added 2026-04-28) so they become first-class unit fields rather than
    # being buried inside the content dict.
    for u in units:
        content_tags = u.content.get("task_type_tags", [])
        if content_tags and not u.applicable_task_types:
            u.applicable_task_types = list(content_tags)
        uw = u.content.get("use_when")
        if isinstance(uw, list) and uw and not u.use_when:
            u.use_when = [str(x) for x in uw if str(x).strip()]
        elif isinstance(uw, str) and uw.strip() and not u.use_when:
            u.use_when = [uw.strip()]
        aw = u.content.get("avoid_when")
        if isinstance(aw, list) and aw and not u.avoid_when:
            u.avoid_when = [str(x) for x in aw if str(x).strip()]
        elif isinstance(aw, str) and aw.strip() and not u.avoid_when:
            u.avoid_when = [aw.strip()]

        # Discriminative initial confidence (was hard-defaulted to 1.0 for every
        # unit, leaving retrieval/ranking no quality signal to work with).
        u.confidence = _compute_initial_confidence(
            u.content if isinstance(u.content, dict) else {}, u.type
        )

    return units
