"""
Contract dataclasses for the outer prompt-driven optimization loop.

Four main contracts connect the optimization phases:

    task_profiling.txt  -->  BenchmarkProfile
    ArchitectureSpec  -->  ArchitectureDecision  -->  extract_plan
    [run eval]  -->  ExtractionBundle
    feedback_analysis.txt  <--  EvaluationReport

Helper dataclasses:
    MemoryDemandScore, FailureMode          (BenchmarkProfile)
    RetrievalPlan, ManagementPlan           (ArchitectureDecision)
    CostSummary, RetrievalTraceSummary,
    MemoryUsageSummary, FailureCase         (EvaluationReport)

All use @dataclass with manual to_dict()/from_dict() — same style as MemoryUnit.
No pydantic dependency.

Usage:
    from automem.contracts import (
        BenchmarkProfile, ArchitectureDecision,
        ExtractionBundle, EvaluationReport,
    )

    profile = BenchmarkProfile.from_dict(json.loads(raw))
    decision = ArchitectureDecision.from_dict(llm_output)
    extract_plan = decision.to_extract_plan()
    provider.take_in_memory(trajectory, extract_plan=extract_plan)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# Helper dataclasses — BenchmarkProfile
# ============================================================

@dataclass
class MemoryDemandScore:
    """Per-memory-type demand score with explanation."""

    score: int = 0
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "why": self.why}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryDemandScore:
        return cls(score=d.get("score", 0), why=d.get("why", ""))


@dataclass
class FailureMode:
    """A common failure pattern observed in the benchmark."""

    name: str = ""
    why_memory_matters: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "why_memory_matters": self.why_memory_matters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FailureMode:
        return cls(
            name=d.get("name", ""),
            why_memory_matters=d.get("why_memory_matters", ""),
        )


# ============================================================
# 1. BenchmarkProfile
# ============================================================

@dataclass
class BenchmarkProfile:
    """
    Produced by task_profiling.txt.
    Characterizes a benchmark / task family so the architecture selector
    knows what kind of memory system to build.
    """

    # --- Identity ---
    benchmark_name: str = ""
    task_family_summary: str = ""
    core_subskills: List[str] = field(default_factory=list)

    # --- Environment characteristics (0-5 scores or categorical) ---
    horizon: str = "medium"                # "short" | "medium" | "long"
    observability: str = "partial"         # "full" | "partial" | "minimal"
    environment_dynamics: str = "static"   # "static" | "slow" | "fast"
    tool_reliance: int = 0                 # 0-5
    entity_density: int = 0               # 0-5
    relation_density: int = 0             # 0-5
    state_revisitation: int = 0           # 0-5
    need_global_strategy: int = 0         # 0-5
    need_stepwise_guidance: int = 0       # 0-5
    cost_sensitivity: int = 0             # 0-5

    # --- Memory demand per type ---
    memory_demand: Dict[str, MemoryDemandScore] = field(default_factory=dict)

    # --- Failure modes ---
    major_failure_modes: List[FailureMode] = field(default_factory=list)

    # --- Architecture priors (hints, not binding) ---
    architecture_priors: Dict[str, Any] = field(default_factory=dict)

    # --- Evaluation ---
    evaluation_focus: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "task_family_summary": self.task_family_summary,
            "core_subskills": list(self.core_subskills),
            "horizon": self.horizon,
            "observability": self.observability,
            "environment_dynamics": self.environment_dynamics,
            "tool_reliance": self.tool_reliance,
            "entity_density": self.entity_density,
            "relation_density": self.relation_density,
            "state_revisitation": self.state_revisitation,
            "need_global_strategy": self.need_global_strategy,
            "need_stepwise_guidance": self.need_stepwise_guidance,
            "cost_sensitivity": self.cost_sensitivity,
            "memory_demand": {
                k: v.to_dict() for k, v in self.memory_demand.items()
            },
            "major_failure_modes": [fm.to_dict() for fm in self.major_failure_modes],
            "architecture_priors": dict(self.architecture_priors),
            "evaluation_focus": list(self.evaluation_focus),
            "uncertainties": list(self.uncertainties),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BenchmarkProfile:
        memory_demand_raw = d.get("memory_demand", {})
        memory_demand = {}
        for k, v in memory_demand_raw.items():
            if isinstance(v, dict):
                memory_demand[k] = MemoryDemandScore.from_dict(v)
            elif isinstance(v, MemoryDemandScore):
                memory_demand[k] = v
            else:
                memory_demand[k] = MemoryDemandScore(score=int(v))

        failure_modes_raw = d.get("major_failure_modes", [])
        failure_modes = []
        for fm in failure_modes_raw:
            if isinstance(fm, dict):
                failure_modes.append(FailureMode.from_dict(fm))
            elif isinstance(fm, FailureMode):
                failure_modes.append(fm)

        return cls(
            benchmark_name=d.get("benchmark_name", ""),
            task_family_summary=d.get("task_family_summary", ""),
            core_subskills=d.get("core_subskills", []),
            horizon=d.get("horizon", "medium"),
            observability=d.get("observability", "partial"),
            environment_dynamics=d.get("environment_dynamics", "static"),
            tool_reliance=d.get("tool_reliance", 0),
            entity_density=d.get("entity_density", 0),
            relation_density=d.get("relation_density", 0),
            state_revisitation=d.get("state_revisitation", 0),
            need_global_strategy=d.get("need_global_strategy", 0),
            need_stepwise_guidance=d.get("need_stepwise_guidance", 0),
            cost_sensitivity=d.get("cost_sensitivity", 0),
            memory_demand=memory_demand,
            major_failure_modes=failure_modes,
            architecture_priors=d.get("architecture_priors", {}),
            evaluation_focus=d.get("evaluation_focus", []),
            uncertainties=d.get("uncertainties", []),
        )


# ============================================================
# Helper dataclasses — ArchitectureDecision
# ============================================================

@dataclass
class RetrievalPlan:
    """Retrieval configuration selected by the architecture optimizer."""

    primary_routes: List[str] = field(default_factory=list)
    secondary_routes: List[str] = field(default_factory=list)
    rerank: str = "none"
    top_k: int = 5
    graph_hop: int = 0
    type_quota: Dict[str, int] = field(default_factory=dict)
    post_retrieval: str = "auto"   # "direct_merge" | "llm_summary" | "auto"
    memory_token_budget: int = 1500
    contradiction_confirm_threshold: float = 0.8
    contradiction_suspect_threshold: float = 0.4
    gate_threshold: float = 0.3  # minimum relevance score to inject memories
    tag_aware: bool = False  # enable tag-based retrieval as additional signal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_routes": list(self.primary_routes),
            "secondary_routes": list(self.secondary_routes),
            "rerank": self.rerank,
            "top_k": self.top_k,
            "graph_hop": self.graph_hop,
            "type_quota": dict(self.type_quota),
            "post_retrieval": self.post_retrieval,
            "memory_token_budget": self.memory_token_budget,
            "contradiction_confirm_threshold": self.contradiction_confirm_threshold,
            "contradiction_suspect_threshold": self.contradiction_suspect_threshold,
            "gate_threshold": self.gate_threshold,
            "tag_aware": self.tag_aware,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RetrievalPlan:
        if d is None:
            return cls()
        return cls(
            primary_routes=d.get("primary_routes", []),
            secondary_routes=d.get("secondary_routes", []),
            rerank=d.get("rerank", "none"),
            top_k=d.get("top_k", 5),
            graph_hop=d.get("graph_hop", 0),
            type_quota=d.get("type_quota", {}),
            post_retrieval=d.get("post_retrieval", "auto"),
            memory_token_budget=d.get("memory_token_budget", 1500),
            contradiction_confirm_threshold=d.get(
                "contradiction_confirm_threshold", 0.8
            ),
            contradiction_suspect_threshold=d.get(
                "contradiction_suspect_threshold", 0.4
            ),
            gate_threshold=d.get("gate_threshold", 0.3),
            tag_aware=d.get("tag_aware", False),
        )


@dataclass
class ManagementPlan:
    """Management configuration selected by the architecture optimizer."""

    enabled_ops: List[str] = field(default_factory=list)
    intensity: str = "none"   # "none" | "light" | "medium" | "heavy"
    post_task_budget: int = 0
    periodic_budget: int = 0

    # Extended fields for preset-based configuration
    preset: str = ""
    custom_ops: List[str] = field(default_factory=list)
    trigger_on_insert: bool = True
    trigger_periodic: bool = False
    periodic_interval: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled_ops": list(self.enabled_ops),
            "intensity": self.intensity,
            "post_task_budget": self.post_task_budget,
            "periodic_budget": self.periodic_budget,
            "preset": self.preset,
            "custom_ops": list(self.custom_ops),
            "trigger_on_insert": self.trigger_on_insert,
            "trigger_periodic": self.trigger_periodic,
            "periodic_interval": self.periodic_interval,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ManagementPlan:
        if d is None:
            return cls()
        return cls(
            enabled_ops=d.get("enabled_ops", []),
            intensity=d.get("intensity", "none"),
            post_task_budget=d.get("post_task_budget", 0),
            periodic_budget=d.get("periodic_budget", 0),
            preset=d.get("preset", ""),
            custom_ops=d.get("custom_ops", []),
            trigger_on_insert=d.get("trigger_on_insert", True),
            trigger_periodic=d.get("trigger_periodic", False),
            periodic_interval=d.get("periodic_interval", 10),
        )


# ============================================================
# 2. ArchitectureDecision
# ============================================================

@dataclass
class ArchitectureDecision:
    """
    Internal compiler contract derived from the public ArchitectureSpec.
    A selected subgraph of the four-layer architecture space.

    This is the central contract — it drives what the inner pipeline
    actually does during memory extraction, storage, retrieval, and
    management.
    """

    # --- Graph topology ---
    selected_nodes: Dict[str, List[str]] = field(default_factory=dict)
    selected_edges: List[Any] = field(default_factory=list)

    # --- Extract configuration ---
    enabled_memory_types: List[str] = field(default_factory=list)
    enabled_anchor_types: List[str] = field(default_factory=list)
    enabled_relation_types: List[str] = field(default_factory=list)

    # --- Storage configuration ---
    storage_routing: Dict[str, Any] = field(default_factory=dict)

    # --- Retrieval configuration ---
    retrieval_plan: RetrievalPlan = field(default_factory=RetrievalPlan)

    # --- Management configuration ---
    management_plan: ManagementPlan = field(default_factory=ManagementPlan)

    # --- Hyperparameters (free-form overrides) ---
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # --- Meta ---
    round_id: int = 0
    benchmark_name: str = ""
    rationale: Dict[str, str] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    risks: List[str] = field(default_factory=list)
    ablation_targets: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Bridge: convert to extract_plan for ModularMemoryProvider
    # ------------------------------------------------------------------

    def to_extract_plan(self) -> Dict[str, Any]:
        """
        Convert to the format accepted by
        ModularMemoryProvider.take_in_memory(extract_plan=...).

        Returns a dict with keys:
            - extract_types: List[str]
            - storage_routing: Dict[str, str]  (flattened to first backend)
            - relation_types: List[str]
        """
        # Flatten storage_routing: if value is a list, take the first element.
        # Only keep entries for enabled memory types to avoid creating
        # unnecessary additional stores for disabled types.
        enabled = set(self.enabled_memory_types)
        flat_routing: Dict[str, str] = {}
        for mem_type, backends in self.storage_routing.items():
            if mem_type not in enabled:
                continue
            if isinstance(backends, list):
                flat_routing[mem_type] = backends[0] if backends else "json"
            else:
                flat_routing[mem_type] = str(backends)

        return {
            "extract_types": list(self.enabled_memory_types),
            "storage_routing": flat_routing,
            "relation_types": list(self.enabled_relation_types),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_nodes": {
                k: list(v) for k, v in self.selected_nodes.items()
            },
            "selected_edges": list(self.selected_edges),
            "enabled_memory_types": list(self.enabled_memory_types),
            "enabled_anchor_types": list(self.enabled_anchor_types),
            "enabled_relation_types": list(self.enabled_relation_types),
            "storage_routing": _serialize_storage_routing(self.storage_routing),
            "retrieval_plan": self.retrieval_plan.to_dict(),
            "management_plan": self.management_plan.to_dict(),
            "hyperparameters": dict(self.hyperparameters),
            "round_id": self.round_id,
            "benchmark_name": self.benchmark_name,
            "rationale": dict(self.rationale),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "risks": list(self.risks),
            "ablation_targets": list(self.ablation_targets),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArchitectureDecision:
        if d is None:
            return cls()

        retrieval_raw = d.get("retrieval_plan")
        if isinstance(retrieval_raw, dict):
            retrieval_plan = RetrievalPlan.from_dict(retrieval_raw)
        elif isinstance(retrieval_raw, RetrievalPlan):
            retrieval_plan = retrieval_raw
        else:
            retrieval_plan = RetrievalPlan()

        management_raw = d.get("management_plan")
        if isinstance(management_raw, dict):
            management_plan = ManagementPlan.from_dict(management_raw)
        elif isinstance(management_raw, ManagementPlan):
            management_plan = management_raw
        else:
            management_plan = ManagementPlan()

        rationale_raw = d.get("rationale", {})
        if isinstance(rationale_raw, str):
            rationale = {"summary": rationale_raw}
        else:
            rationale = dict(rationale_raw)

        return cls(
            selected_nodes=d.get("selected_nodes", {}),
            selected_edges=d.get("selected_edges", []),
            enabled_memory_types=d.get("enabled_memory_types", []),
            enabled_anchor_types=d.get("enabled_anchor_types", []),
            enabled_relation_types=d.get("enabled_relation_types", []),
            storage_routing=d.get("storage_routing", {}),
            retrieval_plan=retrieval_plan,
            management_plan=management_plan,
            hyperparameters=d.get("hyperparameters", {}),
            round_id=d.get("round_id", 0),
            benchmark_name=d.get("benchmark_name", ""),
            rationale=rationale,
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 0.0),
            risks=d.get("risks", []),
            ablation_targets=d.get("ablation_targets", []),
        )


# ============================================================
# 3. ExtractionBundle
# ============================================================

@dataclass
class ExtractionBundle:
    """
    Produced by ModularMemoryProvider.take_in_memory().
    Complete extraction output for a single task.

    Contains serialized (dict) forms of MemoryUnit, EntityNode, and
    MemoryRelation — no numpy dependency.
    """

    # --- Source task ---
    source_task_id: str = ""
    source_task_query: str = ""
    task_outcome: str = ""          # "success" | "failure"

    # --- Extracted data ---
    memory_units: List[Dict[str, Any]] = field(default_factory=list)
    entity_nodes: List[Dict[str, Any]] = field(default_factory=list)
    relation_records: List[Dict[str, Any]] = field(default_factory=list)

    # --- Plan and provenance ---
    extract_plan_used: Dict[str, Any] = field(default_factory=dict)
    prompts_used: List[str] = field(default_factory=list)
    extraction_model: str = ""
    extraction_timestamp: str = ""
    extraction_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def memory_count(self) -> int:
        """Total number of extracted memory units."""
        return len(self.memory_units)

    @property
    def entity_count(self) -> int:
        """Total number of extracted entity nodes."""
        return len(self.entity_nodes)

    @property
    def relation_count(self) -> int:
        """Total number of extracted relations."""
        return len(self.relation_records)

    def memory_type_counts(self) -> Dict[str, int]:
        """Count of memory units grouped by type."""
        counts: Dict[str, int] = {}
        for mu in self.memory_units:
            t = mu.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "source_task_query": self.source_task_query,
            "task_outcome": self.task_outcome,
            "memory_units": list(self.memory_units),
            "entity_nodes": list(self.entity_nodes),
            "relation_records": list(self.relation_records),
            "extract_plan_used": dict(self.extract_plan_used),
            "prompts_used": list(self.prompts_used),
            "extraction_model": self.extraction_model,
            "extraction_timestamp": self.extraction_timestamp,
            "extraction_time_ms": self.extraction_time_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExtractionBundle:
        if d is None:
            return cls()
        return cls(
            source_task_id=d.get("source_task_id", ""),
            source_task_query=d.get("source_task_query", ""),
            task_outcome=d.get("task_outcome", ""),
            memory_units=d.get("memory_units", []),
            entity_nodes=d.get("entity_nodes", []),
            relation_records=d.get("relation_records", []),
            extract_plan_used=d.get("extract_plan_used", {}),
            prompts_used=d.get("prompts_used", []),
            extraction_model=d.get("extraction_model", ""),
            extraction_timestamp=d.get("extraction_timestamp", ""),
            extraction_time_ms=d.get("extraction_time_ms", 0.0),
        )


# ============================================================
# Helper dataclasses — EvaluationReport
# ============================================================

@dataclass
class CostSummary:
    """LLM call and token cost breakdown."""

    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    extraction_calls: int = 0
    retrieval_calls: int = 0
    management_calls: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "extraction_calls": self.extraction_calls,
            "retrieval_calls": self.retrieval_calls,
            "management_calls": self.management_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CostSummary:
        if d is None:
            return cls()
        return cls(
            total_llm_calls=d.get("total_llm_calls", 0),
            total_input_tokens=d.get("total_input_tokens", 0),
            total_output_tokens=d.get("total_output_tokens", 0),
            extraction_calls=d.get("extraction_calls", 0),
            retrieval_calls=d.get("retrieval_calls", 0),
            management_calls=d.get("management_calls", 0),
            estimated_cost_usd=d.get("estimated_cost_usd", 0.0),
        )


@dataclass
class RetrievalTraceSummary:
    """Aggregated retrieval quality metrics across all tasks in a round."""

    total_queries: int = 0
    empty_retrieval_count: int = 0
    avg_memories_per_query: float = 0.0
    avg_relevance_score: float = 0.0
    type_distribution: Dict[str, int] = field(default_factory=dict)
    route_hit_counts: Dict[str, int] = field(default_factory=dict)
    avg_retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "empty_retrieval_count": self.empty_retrieval_count,
            "avg_memories_per_query": self.avg_memories_per_query,
            "avg_relevance_score": self.avg_relevance_score,
            "type_distribution": dict(self.type_distribution),
            "route_hit_counts": dict(self.route_hit_counts),
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RetrievalTraceSummary:
        if d is None:
            return cls()
        return cls(
            total_queries=d.get("total_queries", 0),
            empty_retrieval_count=d.get("empty_retrieval_count", 0),
            avg_memories_per_query=d.get("avg_memories_per_query", 0.0),
            avg_relevance_score=d.get("avg_relevance_score", 0.0),
            type_distribution=d.get("type_distribution", {}),
            route_hit_counts=d.get("route_hit_counts", {}),
            avg_retrieval_time_ms=d.get("avg_retrieval_time_ms", 0.0),
        )


@dataclass
class MemoryUsageSummary:
    """Aggregated memory store statistics after a round."""

    total_memories: int = 0
    total_entities: int = 0
    total_relations: int = 0
    type_counts: Dict[str, int] = field(default_factory=dict)
    active_ratio: float = 1.0
    avg_confidence: float = 0.0
    avg_usage_count: float = 0.0
    avg_success_rate: float = 0.0
    storage_tokens_total: int = 0
    unused_edge_ratio: float = 0.0
    # Aggregated across all tasks in the round
    total_extracted: int = 0
    total_inserted: int = 0
    total_management_ops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_memories": self.total_memories,
            "total_entities": self.total_entities,
            "total_relations": self.total_relations,
            "type_counts": dict(self.type_counts),
            "active_ratio": self.active_ratio,
            "avg_confidence": self.avg_confidence,
            "avg_usage_count": self.avg_usage_count,
            "avg_success_rate": self.avg_success_rate,
            "storage_tokens_total": self.storage_tokens_total,
            "unused_edge_ratio": self.unused_edge_ratio,
            "total_extracted": self.total_extracted,
            "total_inserted": self.total_inserted,
            "total_management_ops": self.total_management_ops,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemoryUsageSummary:
        if d is None:
            return cls()
        return cls(
            total_memories=d.get("total_memories", 0),
            total_entities=d.get("total_entities", 0),
            total_relations=d.get("total_relations", 0),
            type_counts=d.get("type_counts", {}),
            active_ratio=d.get("active_ratio", 1.0),
            avg_confidence=d.get("avg_confidence", 0.0),
            avg_usage_count=d.get("avg_usage_count", 0.0),
            avg_success_rate=d.get("avg_success_rate", 0.0),
            storage_tokens_total=d.get("storage_tokens_total", 0),
            unused_edge_ratio=d.get("unused_edge_ratio", 0.0),
            total_extracted=d.get("total_extracted", 0),
            total_inserted=d.get("total_inserted", 0),
            total_management_ops=d.get("total_management_ops", 0),
        )


@dataclass
class FailureCase:
    """Per-task failure diagnostic for the feedback analyst."""

    task_id: str = ""
    task_query: str = ""
    expected_answer: str = ""
    agent_answer: str = ""
    failure_category: str = ""
    # Valid categories:
    #   "retrieval_miss"  — relevant memory exists but was not retrieved
    #   "extraction_gap"  — no useful memory was extracted
    #   "tool_error"      — tool execution failure
    #   "reasoning_error" — agent had correct info but reasoned wrong
    #   "other"           — uncategorized
    memories_retrieved: List[str] = field(default_factory=list)
    diagnosis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_query": self.task_query,
            "expected_answer": self.expected_answer,
            "agent_answer": self.agent_answer,
            "failure_category": self.failure_category,
            "memories_retrieved": list(self.memories_retrieved),
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FailureCase:
        if d is None:
            return cls()
        return cls(
            task_id=d.get("task_id", ""),
            task_query=d.get("task_query", ""),
            expected_answer=d.get("expected_answer", ""),
            agent_answer=d.get("agent_answer", ""),
            failure_category=d.get("failure_category", ""),
            memories_retrieved=d.get("memories_retrieved", []),
            diagnosis=d.get("diagnosis", ""),
        )


# ============================================================
# 4. EvaluationReport
# ============================================================

@dataclass
class EvaluationReport:
    """
    Produced by the evaluation pipeline.
    Carries everything the feedback analyst needs: what architecture was
    used, how it scored, what went wrong.
    """

    # --- Identity ---
    round_id: int = 0
    benchmark_name: str = ""

    # --- Architecture (nested ArchitectureDecision.to_dict()) ---
    architecture_decision: Dict[str, Any] = field(default_factory=dict)

    # --- Score ---
    score_summary: Dict[str, Any] = field(default_factory=dict)

    # --- Cost ---
    cost_summary: CostSummary = field(default_factory=CostSummary)

    # --- Retrieval quality ---
    retrieval_trace_summary: RetrievalTraceSummary = field(
        default_factory=RetrievalTraceSummary
    )

    # --- Memory store state ---
    memory_usage_summary: MemoryUsageSummary = field(
        default_factory=MemoryUsageSummary
    )

    # --- Failure diagnostics ---
    failure_cases: List[FailureCase] = field(default_factory=list)

    # --- Raw trajectory samples for the feedback prompt ---
    trajectory_samples: List[Dict[str, Any]] = field(default_factory=list)

    # --- Free-form notes ---
    notes: str = ""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def accuracy(self) -> float:
        """Task accuracy from score_summary."""
        return float(self.score_summary.get("accuracy", 0.0))

    @property
    def n_tasks(self) -> int:
        """Total tasks evaluated."""
        return int(self.score_summary.get("n_tasks", 0))

    @property
    def n_correct(self) -> int:
        """Number of tasks answered correctly."""
        return int(self.score_summary.get("n_correct", 0))

    @property
    def empty_retrieval_rate(self) -> float:
        """Fraction of queries that returned no memories."""
        total = self.retrieval_trace_summary.total_queries
        if total == 0:
            return 0.0
        return self.retrieval_trace_summary.empty_retrieval_count / total

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "benchmark_name": self.benchmark_name,
            "architecture_decision": dict(self.architecture_decision),
            "score_summary": dict(self.score_summary),
            "cost_summary": self.cost_summary.to_dict(),
            "retrieval_trace_summary": self.retrieval_trace_summary.to_dict(),
            "memory_usage_summary": self.memory_usage_summary.to_dict(),
            "failure_cases": [fc.to_dict() for fc in self.failure_cases],
            "trajectory_samples": list(self.trajectory_samples),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EvaluationReport:
        if d is None:
            return cls()

        # Deserialize cost_summary
        cost_raw = d.get("cost_summary")
        if isinstance(cost_raw, dict):
            cost_summary = CostSummary.from_dict(cost_raw)
        elif isinstance(cost_raw, CostSummary):
            cost_summary = cost_raw
        else:
            cost_summary = CostSummary()

        # Deserialize retrieval_trace_summary
        rts_raw = d.get("retrieval_trace_summary")
        if isinstance(rts_raw, dict):
            rts = RetrievalTraceSummary.from_dict(rts_raw)
        elif isinstance(rts_raw, RetrievalTraceSummary):
            rts = rts_raw
        else:
            rts = RetrievalTraceSummary()

        # Deserialize memory_usage_summary
        mus_raw = d.get("memory_usage_summary")
        if isinstance(mus_raw, dict):
            mus = MemoryUsageSummary.from_dict(mus_raw)
        elif isinstance(mus_raw, MemoryUsageSummary):
            mus = mus_raw
        else:
            mus = MemoryUsageSummary()

        # Deserialize failure_cases
        fc_raw = d.get("failure_cases", [])
        failure_cases = []
        for fc in fc_raw:
            if isinstance(fc, dict):
                failure_cases.append(FailureCase.from_dict(fc))
            elif isinstance(fc, FailureCase):
                failure_cases.append(fc)

        return cls(
            round_id=d.get("round_id", 0),
            benchmark_name=d.get("benchmark_name", ""),
            architecture_decision=d.get("architecture_decision", {}),
            score_summary=d.get("score_summary", {}),
            cost_summary=cost_summary,
            retrieval_trace_summary=rts,
            memory_usage_summary=mus,
            failure_cases=failure_cases,
            trajectory_samples=d.get("trajectory_samples", []),
            notes=d.get("notes", ""),
        )


# ============================================================
# 5. FeedbackAnalysisResult
# ============================================================

@dataclass
class PromptEdit:
    """A single structured edit suggestion from the feedback analyst."""

    target_prompt: str = "architecture_selection"
    # Valid targets: "architecture_selection", "task_profiling", "feedback_analysis"
    target_section: str = "editable_policy"
    # Valid sections: "editable_policy", "decision_rules", "constraints", "examples", "general"
    edit_type: str = "modify"
    # Valid types: "add_rule", "remove_rule", "modify", "add_example", "adjust_weight", "add_constraint"
    priority: str = "Medium"
    # "High", "Medium", "Low"
    what_to_change: str = ""
    why: str = ""
    expected_effect: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_prompt": self.target_prompt,
            "target_section": self.target_section,
            "edit_type": self.edit_type,
            "priority": self.priority,
            "what_to_change": self.what_to_change,
            "why": self.why,
            "expected_effect": self.expected_effect,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PromptEdit:
        if d is None:
            return cls()
        return cls(
            target_prompt=d.get("target_prompt", "architecture_selection"),
            target_section=d.get("target_section", "editable_policy"),
            edit_type=d.get("edit_type", "modify"),
            priority=d.get("priority", "Medium"),
            what_to_change=d.get("what_to_change", ""),
            why=d.get("why", ""),
            expected_effect=d.get("expected_effect", ""),
        )


@dataclass
class FeedbackAnalysisResult:
    """
    Produced by feedback_analysis.txt.
    Structured feedback from one optimization round to guide the next
    round's prompt rewrite.
    """

    learned_principles: List[str] = field(default_factory=list)

    edge_assessment: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {edge: [str,str], label: str, why: str, evidence: [str]}

    layer_diagnosis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Keys: extract, store, retrieve, manage
    # Each: {status: str, issues: [str], what_to_fix: [str]}

    failure_modes: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {name: str, root_cause_layer: str, why: str}

    suggested_prompt_edits: List[PromptEdit] = field(default_factory=list)

    next_ablation: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learned_principles": list(self.learned_principles),
            "edge_assessment": list(self.edge_assessment),
            "layer_diagnosis": dict(self.layer_diagnosis),
            "failure_modes": list(self.failure_modes),
            "suggested_prompt_edits": [e.to_dict() for e in self.suggested_prompt_edits],
            "next_ablation": list(self.next_ablation),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FeedbackAnalysisResult:
        if d is None:
            return cls()
        edits_raw = d.get("suggested_prompt_edits", [])
        edits = []
        for e in edits_raw:
            if isinstance(e, dict):
                edits.append(PromptEdit.from_dict(e))
            elif isinstance(e, PromptEdit):
                edits.append(e)
        return cls(
            learned_principles=d.get("learned_principles", []),
            edge_assessment=d.get("edge_assessment", []),
            layer_diagnosis=d.get("layer_diagnosis", {}),
            failure_modes=d.get("failure_modes", []),
            suggested_prompt_edits=edits,
            next_ablation=d.get("next_ablation", []),
        )

    @property
    def high_priority_edits(self) -> List[PromptEdit]:
        return [e for e in self.suggested_prompt_edits if e.priority == "High"]

    @property
    def harmful_edges(self) -> List[Dict[str, Any]]:
        return [e for e in self.edge_assessment if e.get("label") == "Harmful"]

    @property
    def missing_edges(self) -> List[Dict[str, Any]]:
        return [e for e in self.edge_assessment if e.get("label") == "Missing"]

    def edits_for_prompt(self, target: str) -> List[PromptEdit]:
        """Filter edits targeting a specific prompt."""
        return [e for e in self.suggested_prompt_edits if e.target_prompt == target]


# ============================================================
# 6. Fitness
# ============================================================

@dataclass
class FitnessWeights:
    """Configurable weights for multi-objective fitness computation.

    Accuracy-first profile (2026-05-13): user explicitly requested that
    accuracy dominate fitness while latency / token-cost / framework
    complexity carry near-zero weight. empty_retrieval_penalty kept small
    (0.05) to discourage trivially-empty pools from masquerading as
    high-accuracy architectures; unused_edge_penalty and the two cost
    penalties dropped to 0 — those concerns belong to a post-search
    deployment phase, not the search itself.
    """

    accuracy: float = 1.0
    latency_penalty: float = 0.0
    token_cost_penalty: float = 0.0
    empty_retrieval_penalty: float = 0.05
    unused_edge_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "latency_penalty": self.latency_penalty,
            "token_cost_penalty": self.token_cost_penalty,
            "empty_retrieval_penalty": self.empty_retrieval_penalty,
            "unused_edge_penalty": self.unused_edge_penalty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FitnessWeights:
        if d is None:
            return cls()
        return cls(
            accuracy=d.get("accuracy", 1.0),
            latency_penalty=d.get("latency_penalty", 0.0),
            token_cost_penalty=d.get("token_cost_penalty", 0.0),
            empty_retrieval_penalty=d.get("empty_retrieval_penalty", 0.05),
            unused_edge_penalty=d.get("unused_edge_penalty", 0.0),
        )


@dataclass
class FitnessResult:
    """Result of multi-objective fitness evaluation."""

    fitness: float = 0.0
    accuracy: float = 0.0
    normalized_latency: float = 0.0
    normalized_token_cost: float = 0.0
    empty_retrieval_rate: float = 0.0
    unused_edge_ratio: float = 0.0
    weights_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitness": self.fitness,
            "accuracy": self.accuracy,
            "normalized_latency": self.normalized_latency,
            "normalized_token_cost": self.normalized_token_cost,
            "empty_retrieval_rate": self.empty_retrieval_rate,
            "unused_edge_ratio": self.unused_edge_ratio,
            "weights_used": dict(self.weights_used),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FitnessResult:
        if d is None:
            return cls()
        return cls(
            fitness=d.get("fitness", 0.0),
            accuracy=d.get("accuracy", 0.0),
            normalized_latency=d.get("normalized_latency", 0.0),
            normalized_token_cost=d.get("normalized_token_cost", 0.0),
            empty_retrieval_rate=d.get("empty_retrieval_rate", 0.0),
            unused_edge_ratio=d.get("unused_edge_ratio", 0.0),
            weights_used=d.get("weights_used", {}),
        )


def compute_fitness(
    report: EvaluationReport,
    weights: Optional[FitnessWeights] = None,
    latency_cap: float = 120.0,
    token_cap: float = 100000.0,
) -> FitnessResult:
    """
    Compute multi-objective fitness from an EvaluationReport.

    fitness = accuracy
              - a * normalized_latency
              - b * normalized_token_cost
              - c * empty_retrieval_rate
              - d * unused_edge_ratio

    All penalties are normalized to [0, 1] before applying weights.
    """
    if weights is None:
        weights = FitnessWeights()

    acc = report.accuracy

    # Normalize latency: avg_time from score_summary, capped
    avg_time = float(report.score_summary.get("avg_time_per_task", 0.0))
    norm_latency = min(avg_time / latency_cap, 1.0) if latency_cap > 0 else 0.0

    # Normalize token cost
    total_tokens = report.cost_summary.total_input_tokens + report.cost_summary.total_output_tokens
    norm_tokens = min(total_tokens / token_cap, 1.0) if token_cap > 0 else 0.0

    # Empty retrieval rate
    empty_rate = report.empty_retrieval_rate

    # Unused edge ratio from memory usage
    unused_ratio = report.memory_usage_summary.unused_edge_ratio

    fitness = (
        weights.accuracy * acc
        - weights.latency_penalty * norm_latency
        - weights.token_cost_penalty * norm_tokens
        - weights.empty_retrieval_penalty * empty_rate
        - weights.unused_edge_penalty * unused_ratio
    )

    return FitnessResult(
        fitness=fitness,
        accuracy=acc,
        normalized_latency=norm_latency,
        normalized_token_cost=norm_tokens,
        empty_retrieval_rate=empty_rate,
        unused_edge_ratio=unused_ratio,
        weights_used=weights.to_dict(),
    )


# ============================================================
# 7. IncumbentRecord (for incumbent/challenger/archive)
# ============================================================

@dataclass
class IncumbentRecord:
    """Tracks the current best prompt version and its performance."""

    version_id: str = "v0"
    prompt_text: str = ""
    parent_version: str = ""
    round_id: int = 0
    fitness: FitnessResult = field(default_factory=FitnessResult)
    evaluation_hash: str = ""
    source: str = "initial"  # "initial", "challenger_win", "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "prompt_text": self.prompt_text,
            "parent_version": self.parent_version,
            "round_id": self.round_id,
            "fitness": self.fitness.to_dict(),
            "evaluation_hash": self.evaluation_hash,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> IncumbentRecord:
        if d is None:
            return cls()
        fitness_raw = d.get("fitness")
        if isinstance(fitness_raw, dict):
            fitness = FitnessResult.from_dict(fitness_raw)
        else:
            fitness = FitnessResult()
        return cls(
            version_id=d.get("version_id", "v0"),
            prompt_text=d.get("prompt_text", ""),
            parent_version=d.get("parent_version", ""),
            round_id=d.get("round_id", 0),
            fitness=fitness,
            evaluation_hash=d.get("evaluation_hash", ""),
            source=d.get("source", "initial"),
        )


# ============================================================
# Internal helpers
# ============================================================

def _serialize_storage_routing(routing: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize storage_routing, handling both str and list values."""
    result = {}
    for k, v in routing.items():
        if isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Main contracts
    "BenchmarkProfile",
    "ArchitectureDecision",
    "ExtractionBundle",
    "EvaluationReport",
    "FeedbackAnalysisResult",
    # Fitness
    "FitnessWeights",
    "FitnessResult",
    "compute_fitness",
    # Incumbent/archive
    "IncumbentRecord",
    "PromptEdit",
    # Helper dataclasses — BenchmarkProfile
    "MemoryDemandScore",
    "FailureMode",
    # Helper dataclasses — ArchitectureDecision
    "RetrievalPlan",
    "ManagementPlan",
    # Helper dataclasses — EvaluationReport
    "CostSummary",
    "RetrievalTraceSummary",
    "MemoryUsageSummary",
    "FailureCase",
]
