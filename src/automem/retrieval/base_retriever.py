"""
Base Retriever — Abstract interface and unified output types for the Retrieval layer.

Core types:
  - QueryContext:  Encapsulates query + optional embedding + metadata
  - ScoredUnit:    MemoryUnit + retrieval score + retrieval method tag
  - TraceEntry:    One step in the retrieval path (for explainability)
  - EvidenceRef:   Lightweight source reference (unit_id, type, snippet)
  - MemoryPack:    Unified retrieval output — the single return type for all retrievers

Usage:
  from automem.retrieval import SemanticRetriever, MemoryPack

  retriever = SemanticRetriever(store, embedding_model)
  pack = retriever.retrieve(QueryContext(query="How to parse PDF?"))
  prompt_str = pack.to_prompt_string()   # ready for LLM injection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from automem.memory_schema import MemoryUnit, MemoryUnitType


# ============================================================
# Query Context
# ============================================================

@dataclass
class QueryContext:
    """Input to a retriever: the current task query + optional enrichments."""
    query: str
    embedding: Optional[np.ndarray] = None
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Scored Unit (retrieval result atom)
# ============================================================

@dataclass
class ScoredUnit:
    """A MemoryUnit annotated with a retrieval score and method tag."""
    unit: MemoryUnit
    score: float
    method: str = ""          # e.g. "semantic", "keyword", "graph"
    source_store: str = ""    # e.g. "json", "vector", "graph" — which storage backend

    def __repr__(self) -> str:
        store_tag = f", store={self.source_store}" if self.source_store else ""
        return (
            f"ScoredUnit({self.unit.type.value}, "
            f"score={self.score:.3f}, method={self.method}{store_tag})"
        )


# ============================================================
# Trace & Evidence (explainability)
# ============================================================

@dataclass
class TraceEntry:
    """One step in the retrieval path, for debugging and explainability."""
    step: int
    method: str           # "semantic", "keyword", "graph_expand", etc.
    candidates: int       # number of candidates at this step
    selected: int         # number selected / passed to next step
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "method": self.method,
            "candidates": self.candidates,
            "selected": self.selected,
            "params": self.params,
        }


@dataclass
class EvidenceRef:
    """Lightweight source reference for a retrieved memory."""
    unit_id: str
    unit_type: str
    snippet: str          # short text excerpt
    score: float
    source_task_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_type": self.unit_type,
            "snippet": self.snippet,
            "score": self.score,
            "source_task_id": self.source_task_id,
        }


# ============================================================
# MemoryPack — Unified retrieval output
# ============================================================

# Type-specific formatting templates
def _format_tip(c: dict, s: float) -> str:
    lines = [
        f"[TIP] {c.get('topic', '')} (relevance: {s:.2f})",
        f"  Principle: {c.get('principle', '')}",
        f"  Example: {c.get('micro_example', '')}",
    ]
    if c.get('applicability'):
        lines.append(f"  Applies when: {c['applicability']}")
    tags = c.get('task_type_tags', [])
    if tags:
        lines.append(f"  Domain: {', '.join(tags)}")
    return "\n".join(lines)


def _format_insight(c: dict, s: float) -> str:
    lines = [
        f"[INSIGHT] (relevance: {s:.2f})",
        f"  Root cause: {c.get('root_cause_conclusion', '')}",
        f"  Mismatch: {c.get('state_mismatch_analysis', '')}",
    ]
    if c.get('failure_pattern'):
        lines.append(f"  Pattern: {c['failure_pattern']}")
    if c.get('applicability'):
        lines.append(f"  Applies when: {c['applicability']}")
    tags = c.get('task_type_tags', [])
    if tags:
        lines.append(f"  Domain: {', '.join(tags)}")
    return "\n".join(lines)


_TYPE_FORMATTERS = {
    MemoryUnitType.TIP: lambda c, s: _format_tip(c, s),
    MemoryUnitType.INSIGHT: lambda c, s: _format_insight(c, s),
    MemoryUnitType.WORKFLOW: lambda c, s: _format_workflow(c, s),
    MemoryUnitType.TRAJECTORY: lambda c, s: _format_trajectory(c, s),
    MemoryUnitType.SHORTCUT: lambda c, s: (
        f"[SHORTCUT] {c.get('name', '')} (relevance: {s:.2f})\n"
        f"  {c.get('description', '')}\n"
        f"  Precondition: {c.get('precondition', '')}"
    ),
}


def _format_workflow(c: dict, s: float) -> str:
    parts = [f"[WORKFLOW] (relevance: {s:.2f})"]
    for wf_key in ("agent_workflow", "search_workflow"):
        steps = c.get(wf_key, [])
        if not steps:
            continue
        if not isinstance(steps, list):
            # Legacy: pre-M4 fix units may store agent_workflow as a raw string
            parts.append(f"  {wf_key}: {str(steps)[:300]}")
            continue
        parts.append(f"  {wf_key}:")
        for st in steps:
            if not isinstance(st, dict):
                continue
            step_num = st.get("step", "?")
            action = st.get("action", st.get("query_formulation", ""))
            parts.append(f"    Step {step_num}: {action}")
    return "\n".join(parts)


def _format_trajectory(c: dict, s: float) -> str:
    steps = c.get("steps", [])
    parts = [f"[TRAJECTORY] ({len(steps)} steps, relevance: {s:.2f})"]
    # Show high-signal summary fields first (new format) — most reusable for future tasks
    key_decision = c.get("key_decision", "")
    tool_strategy = c.get("tool_strategy", "")
    critical_obs = c.get("critical_observation", "")
    if key_decision:
        parts.append(f"  Key decision: {key_decision}")
    if tool_strategy:
        parts.append(f"  Tool strategy: {tool_strategy}")
    if critical_obs:
        parts.append(f"  Critical observation: {critical_obs}")
    # Then show compressed steps
    for st in steps[:5]:  # truncate to first 5 steps
        sid = st.get("step_id", st.get("step", "?"))
        action = st.get("action", "")
        parts.append(f"  Step {sid}: {action}")
    if len(steps) > 5:
        parts.append(f"  ... ({len(steps) - 5} more steps)")
    return "\n".join(parts)


def _format_scored_unit(su: ScoredUnit, for_judge: bool = False) -> str:
    """Format a single ScoredUnit into human-readable text.

    Codex Q2 fixes (2026-04-28):
      - `for_judge=True` is set ONLY when this rendering feeds the LLM
        Judge. In that mode we expose the raw `Source: <source_task_query>`
        line and the full `critical_observation` for high-leakage units so
        the judge can detect cross-domain misuse.
      - In agent-injection mode (`for_judge=False`, default), `Source` is
        ABSTRACTED to the inferred source category only (no raw prior-task
        query → no entity/date leakage to the executor agent), and units
        with `content.leakage_risk == "high"` have their answer-like
        fields (critical_observation) removed before injection.
    """
    unit = su.unit
    is_neg = bool(getattr(unit, "is_negative_example", False))
    leakage_risk = ""
    if isinstance(unit.content, dict):
        leakage_risk = str(unit.content.get("leakage_risk", "")).strip().lower()

    formatter = _TYPE_FORMATTERS.get(unit.type)
    # Q2-A3: high-leakage trajectory — strip critical_observation from the
    # injection rendering. We do this by mutating a *copy* of content for
    # this render only.
    render_content = unit.content
    if (not for_judge) and leakage_risk == "high" \
            and isinstance(unit.content, dict):
        render_content = {**unit.content}
        if "critical_observation" in render_content:
            render_content["critical_observation"] = "[redacted: leakage_risk=high]"
    if formatter:
        body = formatter(render_content, su.score)
    else:
        # content_text() may include high-leakage fields too — for
        # agent-side render we cap aggressively when high leakage.
        text = unit.content_text()[:120 if (not for_judge and leakage_risk == "high") else 200]
        body = f"[{unit.type.value.upper()}] (relevance: {su.score:.2f})\n  {text}"

    if is_neg and body.startswith("[TRAJECTORY]"):
        body = body.replace("[TRAJECTORY]", "[TRAJECTORY ⚠ NEGATIVE EXAMPLE]", 1)

    use_hint = _USE_HINTS.get(unit.type, "")
    if is_neg:
        use_hint = (
            "AVOID this path. Read key_decision to learn what NOT to do; "
            "the steps below ENDED IN FAILURE."
        )
    if use_hint:
        body += f"\n  → Use: {use_hint}"

    use_when = getattr(unit, "use_when", None) or []
    if use_when:
        uw = "; ".join(str(t).strip() for t in use_when if str(t).strip())
        if uw:
            body += f"\n  Apply when: {uw}"
    avoid_when = getattr(unit, "avoid_when", None) or []
    if avoid_when:
        aw = "; ".join(str(t).strip() for t in avoid_when if str(t).strip())
        if aw:
            body += f"\n  Avoid when: {aw}"

    # Codex Q2-1: Source provenance only goes to the JUDGE; for the agent we
    # abstract it down to a coarse domain hint to avoid leaking entities/
    # dates/file names from the prior task into exact-match grading.
    src = getattr(unit, "source_task_query", "") or ""
    if isinstance(src, str) and src.strip():
        if for_judge:
            body += f"\n  Source: {src.strip()[:120]}"
        else:
            domain_hint = _abstract_source_domain(src)
            if domain_hint:
                body += f"\n  Source domain: {domain_hint}"

    note = getattr(su, "judge_note", None)
    if note:
        body += f"\n  Judge note: {note}"
    return body


def _abstract_source_domain(src: str) -> str:
    """Map a prior-task query string to a coarse source-domain label.

    The point is: the agent does NOT need to know the literal prior query;
    a category like "encyclopedia / discography / paper / file_xlsx /
    finance" is enough to assess applicability without leaking entities.
    Falls back to "" when no clear category — better silent than wrong.
    """
    s = (src or "").lower()
    if not s:
        return ""
    # Keep this list short and operation-class-flavored.
    rules = [
        ("xlsx", "file_xlsx task"),
        ("spreadsheet", "file_xlsx task"),
        ("pdf", "file_pdf task"),
        ("image", "file_image task"),
        ("youtube", "multimodal_video task"),
        ("audio", "file_audio task"),
        ("github", "github_repo task"),
        ("museum", "museum_collection task"),
        ("wikipedia", "encyclopedia task"),
        ("paper", "academic_paper task"),
        ("arxiv", "academic_paper task"),
        ("doi", "academic_paper task"),
        ("studio album", "discography task"),
        ("discograph", "discography task"),
        ("film", "filmography task"),
        ("price", "finance task"),
        ("how many", "count_query task"),
    ]
    for keyword, label in rules:
        if keyword in s:
            return label
    return ""


# Per-type usage hint shown to the agent (mirrors prompt_support's
# format_memory_unit so agent and judge see consistent guidance).
_USE_HINTS = {
    MemoryUnitType.TIP: "Apply this principle when planning if Apply-when matches.",
    MemoryUnitType.WORKFLOW: (
        "Reference step ordering and tool sequence. Adapt arguments to current "
        "task; do NOT copy parameters."
    ),
    MemoryUnitType.INSIGHT: (
        "AVOID this failure pattern. Apply the corrective_strategy proactively "
        "when the detection_signal appears."
    ),
    MemoryUnitType.TRAJECTORY: (
        "Read key_decision + critical_observation only. Do NOT replay steps verbatim."
    ),
    MemoryUnitType.SHORTCUT: (
        "Invoke as a parameterized macro when Apply-when matches; substitute "
        "placeholders with current values."
    ),
}


@dataclass
class MemoryPack:
    """
    Unified retrieval output.

    Contains the selected memories, grouped by type, with full trace
    and evidence references for explainability.

    The main consumer-facing method is `to_prompt_string()`, which
    produces a formatted string wrapped in begin/end markers, ready
    for injection into the LLM context.
    """
    query_context: QueryContext
    scored_units: List[ScoredUnit] = field(default_factory=list)
    trace: List[TraceEntry] = field(default_factory=list)
    evidence: List[EvidenceRef] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    retriever_name: str = ""
    # Graph edges traversed during retrieval expansion, as (source_nid,
    # target_nid, edge_type) triples. Populated only by graph-walking
    # retrievers and aggregated across sub-packs by MultiStoreRetriever;
    # consumed by the edge_stats_update management op (G1, 2026-07-11).
    # Defaulted so every existing constructor stays valid.
    used_edges: List[Tuple[str, str, str]] = field(default_factory=list)

    # ---- Derived accessors ----

    @property
    def by_type(self) -> Dict[str, List[ScoredUnit]]:
        """Group scored units by MemoryUnitType value."""
        groups: Dict[str, List[ScoredUnit]] = {}
        for su in self.scored_units:
            key = su.unit.type.value
            groups.setdefault(key, []).append(su)
        return groups

    @property
    def selected_units(self) -> List[MemoryUnit]:
        """Flat list of MemoryUnit objects (no scores)."""
        return [su.unit for su in self.scored_units]

    @property
    def total_tokens(self) -> int:
        """Rough total token estimate for all selected units."""
        return sum(su.unit.storage_tokens for su in self.scored_units)

    def is_empty(self) -> bool:
        return len(self.scored_units) == 0

    # ---- Formatting ----

    def to_prompt_string(
        self,
        begin_marker: str = "----Memory System Guidance----",
        end_marker: str = "----End Memory----",
        max_units: Optional[int] = None,
        group_by_type: bool = True,
    ) -> str:
        """
        Format the retrieval result into a string for LLM context injection.

        Compatible with the existing agent framework format:
          ----Memory System Guidance----
          <formatted memories>
          ----End Memory----
        """
        if self.is_empty():
            return ""

        units_to_format = self.scored_units
        if max_units is not None:
            units_to_format = units_to_format[:max_units]

        if group_by_type:
            body = self._format_grouped(units_to_format)
        else:
            body = "\n\n".join(_format_scored_unit(su) for su in units_to_format)

        return f"{begin_marker}\n{body}\n{end_marker}"

    def _format_grouped(self, units: List[ScoredUnit]) -> str:
        """Format units grouped by type, with section headers."""
        groups: Dict[str, List[ScoredUnit]] = {}
        for su in units:
            key = su.unit.type.value
            groups.setdefault(key, []).append(su)

        sections = []
        # Order: tip > insight > workflow > trajectory > shortcut
        type_order = ["tip", "insight", "workflow", "trajectory", "shortcut"]
        for t in type_order:
            if t not in groups:
                continue
            section_units = groups[t]
            formatted = [_format_scored_unit(su) for su in section_units]
            sections.append("\n\n".join(formatted))

        return "\n\n".join(sections)

    def to_guidance_text(self) -> str:
        """Return only the body text (no begin/end markers), for embedding into MemoryItem."""
        if self.is_empty():
            return ""
        return "\n\n".join(_format_scored_unit(su) for su in self.scored_units)

    # ---- Serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query_context.query,
            "retriever": self.retriever_name,
            "num_units": len(self.scored_units),
            "by_type": {k: len(v) for k, v in self.by_type.items()},
            "total_tokens": self.total_tokens,
            "trace": [t.to_dict() for t in self.trace],
            "evidence": [e.to_dict() for e in self.evidence],
            "created_at": self.created_at,
        }


# ============================================================
# Base Retriever — Abstract interface
# ============================================================

class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.

    Subclasses implement `retrieve()` which takes a QueryContext
    and returns a MemoryPack.

    All retrievers operate on a BaseMemoryStorage backend.
    """

    def __init__(self, store, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            store: A BaseMemoryStorage instance (JsonStorage, VectorStorage, etc.)
            config: Strategy-specific configuration dict.
        """
        self.store = store
        self.config = config or {}

    @property
    def name(self) -> str:
        """Retriever name, used in MemoryPack.retriever_name and trace."""
        return self.__class__.__name__

    @abstractmethod
    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        """
        Retrieve relevant memories for the given query context.

        Args:
            ctx: QueryContext with query text, optional embedding, metadata.
            top_k: Maximum number of units to return.

        Returns:
            MemoryPack with scored units, trace, and evidence.
        """
        ...

    def _build_evidence(self, scored_units: List[ScoredUnit]) -> List[EvidenceRef]:
        """Build evidence references from scored units."""
        evidence = []
        for su in scored_units:
            text = su.unit.content_text()
            snippet = text[:120] + "..." if len(text) > 120 else text
            evidence.append(EvidenceRef(
                unit_id=su.unit.id,
                unit_type=su.unit.type.value,
                snippet=snippet,
                score=su.score,
                source_task_id=su.unit.source_task_id,
            ))
        return evidence

    def _make_pack(
        self,
        ctx: QueryContext,
        scored_units: List[ScoredUnit],
        trace: List[TraceEntry],
    ) -> MemoryPack:
        """Convenience: build a complete MemoryPack from retrieval results."""
        return MemoryPack(
            query_context=ctx,
            scored_units=scored_units,
            trace=trace,
            evidence=self._build_evidence(scored_units),
            retriever_name=self.name,
        )
