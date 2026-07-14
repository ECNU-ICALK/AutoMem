"""ObservationGraph — rule-based structural experience map for the Proposer.

Granularity (v1): task patterns are keyed by GAIA level ("L1"/"L2"/"L3").
Category distribution is stored as node metadata for richness, but edges are
attributed at level granularity because that is the finest breakdown available
in per-candidate metrics (``score_summary.by_level``).

The graph is updated once per round from data the orchestrator already has
(round results + per-task scores + baseline). No LLM is involved in the update
path; the graph is only *serialised* into the Proposer prompt when a candidate
is proposed in observation-aware mode.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Edge relation constants
REL_TRIED_WITH = "tried_with"               # task_pattern -> extract combo (marginal)
REL_WORKED_WITH_RETRIEVER = "worked_with_retriever"  # task_pattern -> retriever (marginal)
REL_ARCH_EVALUATED = "arch_evaluated"       # task_pattern -> full joint architecture
REL_HIGH_ATTRIBUTION = "high_attribution"   # task_pattern -> memory_ref

# Complexity buckets (benchmark-agnostic). GAIA's raw Level 1/2/3 and any
# step-count signal are both folded into these via _to_bucket() so the
# Observation Graph stratifies the same way on GAIA, xBench, WebWalkerQA, ...
_VALID_LEVELS = {"simple", "medium", "complex"}


def _to_bucket(level=None, n_steps=None) -> str:
    """Map a raw difficulty level (e.g. GAIA Level 1/2/3) and/or a trajectory
    step count to a complexity bucket: simple|medium|complex|unknown.

    Every external `level` value entering the graph (task census, candidate
    by_level keys, attribution helps_levels) goes through this so GAIA's
    numeric levels and benchmarks-without-levels land in the same 3 buckets.
    """
    from automem.task_complexity import task_complexity
    traj = range(n_steps) if isinstance(n_steps, int) and n_steps > 0 else None
    return task_complexity(explicit_level=level, trajectory=traj)


@dataclass
class ObsNode:
    id: str
    kind: str                                # "task_pattern" | "memory_ref"
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObsEdge:
    src: str
    rel: str
    dst: str
    attrs: Dict[str, Any] = field(default_factory=dict)


class ObservationGraph:
    """Rule-based observation graph. Thread-safe via an internal RLock."""

    def __init__(self, path: Optional[Path] = None):
        self.path: Optional[Path] = Path(path) if path else None
        self.version: int = 0
        self.updated_at_round: int = 0
        self.nodes: Dict[str, ObsNode] = {}
        # edges keyed by (src, rel, dst) so repeated round updates accumulate
        # in place rather than appending duplicates.
        self._edges: Dict[tuple, ObsEdge] = {}
        # round_ids already folded in — makes update_from_round idempotent so a
        # --resume that re-enters an already-recorded round cannot double-count
        # n_trials / census.
        self.recorded_rounds: List[int] = []
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            try:
                self._load()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("ObservationGraph load failed (%s); starting empty.", e)

    # ------------------------------------------------------------------
    # Public read helpers
    # ------------------------------------------------------------------
    @property
    def edges(self) -> List[ObsEdge]:
        return list(self._edges.values())

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        with self._lock:
            if not self.path:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": self.version,
                "updated_at_round": self.updated_at_round,
                "recorded_rounds": self.recorded_rounds,
                "nodes": [
                    {"id": n.id, "kind": n.kind, "attrs": n.attrs}
                    for n in self.nodes.values()
                ],
                "edges": [
                    {"src": e.src, "rel": e.rel, "dst": e.dst, "attrs": e.attrs}
                    for e in self._edges.values()
                ],
            }
            self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self) -> None:
        d = json.loads(self.path.read_text())
        self.version = int(d.get("version", 0))
        self.updated_at_round = int(d.get("updated_at_round", 0))
        self.recorded_rounds = list(d.get("recorded_rounds", []))
        self.nodes = {
            n["id"]: ObsNode(id=n["id"], kind=n["kind"], attrs=n.get("attrs", {}))
            for n in d.get("nodes", [])
        }
        self._edges = {}
        for e in d.get("edges", []):
            key = (e["src"], e["rel"], e["dst"])
            self._edges[key] = ObsEdge(src=e["src"], rel=e["rel"],
                                       dst=e["dst"], attrs=e.get("attrs", {}))

    # ------------------------------------------------------------------
    # Update (rule-based; no LLM)
    # ------------------------------------------------------------------
    def update_from_round(
        self,
        round_id: int,
        candidates: List[Dict[str, Any]],
        task_results: Optional[List[Dict[str, Any]]] = None,
        baseline_per_level: Optional[Dict[str, float]] = None,
        top_attribution_k: int = 5,
    ) -> None:
        """Fold one round's outcomes into the graph.

        Args:
          round_id: current round (1-based).
          candidates: list of dicts, each with at least
              ``architecture`` (dict with extract_types/retrieval) and
              ``metrics`` (dict, may include accuracy/fitness and
              ``score_summary.by_level``). Optional ``attribution`` list of
              {unit_id, extract_type, score} for high-value units.
          task_results: optional per-task dicts with ``level`` + ``category``;
              used to populate task_pattern node census + category distribution.
          baseline_per_level: optional {"1": acc, "2": acc, "3": acc}.
          top_attribution_k: how many top attributed units to record per round.
        """
        with self._lock:
            # Idempotency: never fold the same round twice (guards --resume
            # re-entering an already-recorded round). Codex P2 fix 2026-05-20.
            if round_id in self.recorded_rounds:
                logger.info(
                    "ObservationGraph: round %d already recorded; skipping update.",
                    round_id,
                )
                return
            self.recorded_rounds.append(round_id)
            self.version += 1
            self.updated_at_round = round_id

            # 1) task_pattern census (from task_results, if provided)
            if task_results:
                self._update_task_patterns(task_results, baseline_per_level)

            # Determine the set of levels we "know about" this update — used
            # to attribute candidate metrics that lack a per-level breakdown,
            # and to link attribution units that don't carry helps_levels.
            known_levels = set()
            if task_results:
                for t in task_results:
                    lvl = _to_bucket(t.get("level") or t.get("Level"), t.get("n_steps"))
                    if lvl in _VALID_LEVELS:
                        known_levels.add(lvl)
            # also include levels already present as task_pattern nodes
            known_levels |= {
                n.id[1:] for n in self.nodes.values()
                if n.kind == "task_pattern" and n.id[1:] in _VALID_LEVELS
            }

            # 2) per-candidate edges (tried_with / worked_with_retriever)
            for cand in candidates:
                self._update_candidate_edges(cand, known_levels)

            # 3) high-attribution memory_ref nodes
            for cand in candidates:
                self._update_attribution(cand, top_attribution_k, known_levels)

        self.save()

    # ---- internal update helpers ----
    def _update_task_patterns(
        self,
        task_results: List[Dict[str, Any]],
        baseline_per_level: Optional[Dict[str, float]],
    ) -> None:
        # group by level
        by_level: Dict[str, List[Dict[str, Any]]] = {}
        for t in task_results:
            lvl = _to_bucket(t.get("level") or t.get("Level"), t.get("n_steps"))
            if lvl not in _VALID_LEVELS:
                continue
            by_level.setdefault(lvl, []).append(t)

        for lvl, tasks in by_level.items():
            node_id = f"L{lvl}"
            node = self.nodes.get(node_id)
            if node is None:
                node = ObsNode(id=node_id, kind="task_pattern",
                               attrs={"n_tasks_seen": 0, "categories": {},
                                      "best_acc_so_far": 0.0})
                self.nodes[node_id] = node

            node.attrs["n_tasks_seen"] = node.attrs.get("n_tasks_seen", 0) + len(tasks)

            # category distribution (informational)
            cats = node.attrs.setdefault("categories", {})
            for t in tasks:
                c = t.get("category") or "unknown"
                cats[c] = cats.get(c, 0) + 1

            # NOTE: best_acc_so_far is intentionally NOT computed here. The
            # census task_results come from a single representative candidate,
            # which is not necessarily the best performer. best_acc_so_far is
            # derived from each candidate's per-level metrics in
            # _update_candidate_edges (max over all candidates). Codex P2 fix.

            # baseline (set once, from provided per-level baseline)
            if baseline_per_level and lvl in baseline_per_level:
                node.attrs["baseline_acc"] = round(float(baseline_per_level[lvl]), 4)

    def _ensure_task_pattern_node(self, level: str) -> "ObsNode":
        """Create a stub task_pattern node if missing (so candidate-only
        updates still surface in to_proposer_json)."""
        node_id = f"L{level}"
        node = self.nodes.get(node_id)
        if node is None:
            node = ObsNode(id=node_id, kind="task_pattern",
                           attrs={"n_tasks_seen": 0, "categories": {},
                                  "best_acc_so_far": 0.0})
            self.nodes[node_id] = node
        return node

    def _update_candidate_edges(self, cand: Dict[str, Any],
                                known_levels: set) -> None:
        arch = cand.get("architecture") or {}
        metrics = cand.get("metrics") or {}
        extract_types = arch.get("extract_types") or []
        retrieval = arch.get("retrieval") or arch.get("retriever") or "?"
        if not extract_types:
            return
        combo = "extract:" + "+".join(sorted(extract_types))
        # Full joint architecture signature — captures which complete config
        # actually produced the accuracy (marginal edges below can otherwise
        # mislead the proposer into pairing a high-avg combo with a high-avg
        # retriever that were never good together). Codex P2 fix 2026-05-20.
        mgmt = arch.get("management") or arch.get("management_preset") or "?"
        # Preserve the FULL per-type routing (not just the set of backends), so
        # tip->json,traj->vector is distinct from tip->vector,traj->json in the
        # authoritative architectures_evaluated key. Codex P2 fix 2026-05-20.
        storage_routing = arch.get("storage_routing") or {}
        storage_sig = ",".join(
            f"{k}={v}" for k, v in sorted(storage_routing.items())) or "?"
        arch_sig = f"{combo} | ret:{retrieval} | mgmt:{mgmt} | store:{storage_sig}"

        # Build {level: accuracy}. Prefer the per-level breakdown; if absent,
        # attribute the candidate's overall accuracy to every known level
        # (dry-run / legacy reports only carry aggregate `accuracy`).
        level_accs: Dict[str, float] = {}
        by_level = (metrics.get("score_summary") or {}).get("by_level") or {}
        if by_level:
            for lvl, stat in by_level.items():
                bucket = _to_bucket(lvl)   # GAIA "1"/"2"/"3" -> simple/medium/complex
                if bucket not in _VALID_LEVELS:
                    continue
                total = stat.get("total") or 0
                if total <= 0:
                    continue
                level_accs[bucket] = (stat.get("correct") or 0) / total
        # Fallback: no usable per-bucket breakdown. Either by_level is absent
        # (dry-run / legacy reports) OR it only carried non-difficulty buckets
        # like "unknown" (benchmarks without a level field — by_level is then
        # truthy but every _to_bucket() is "unknown", leaving level_accs empty).
        # Attribute overall accuracy to every known complexity bucket so the
        # candidate still records arch/combo/retriever edges. Codex P2 fix.
        if not level_accs:
            overall = metrics.get("accuracy")
            if overall is not None:
                for lvl in known_levels:
                    level_accs[lvl] = float(overall)

        for lvl, level_acc in level_accs.items():
            node = self._ensure_task_pattern_node(lvl)  # so to_proposer_json emits it
            # best_acc_so_far = max per-level accuracy across ALL candidates
            # (not the census candidate). Codex P2 fix 2026-05-20.
            node.attrs["best_acc_so_far"] = max(
                node.attrs.get("best_acc_so_far", 0.0), round(level_acc, 4))
            src = f"L{lvl}"
            # joint (full-architecture) edge — the authoritative signal
            self._accumulate_edge(src, REL_ARCH_EVALUATED, arch_sig, level_acc)
            # marginal edges — useful for breadth / exploration hints
            self._accumulate_edge(src, REL_TRIED_WITH, combo, level_acc)
            self._accumulate_edge(src, REL_WORKED_WITH_RETRIEVER,
                                  f"retrieval:{retrieval}", level_acc)

    def _accumulate_edge(self, src: str, rel: str, dst: str, acc_value: float) -> None:
        key = (src, rel, dst)
        e = self._edges.get(key)
        if e is None:
            e = ObsEdge(src=src, rel=rel, dst=dst,
                        attrs={"n_trials": 0, "avg_acc": 0.0})
            self._edges[key] = e
        n = e.attrs.get("n_trials", 0)
        prev_avg = e.attrs.get("avg_acc", 0.0)
        new_avg = (prev_avg * n + acc_value) / (n + 1)
        e.attrs["n_trials"] = n + 1
        e.attrs["avg_acc"] = round(new_avg, 4)

    def _update_attribution(self, cand: Dict[str, Any], top_k: int,
                            known_levels: set) -> None:
        attribution = cand.get("attribution") or []
        if not attribution:
            return
        # sort by score desc, take top_k
        ranked = sorted(attribution,
                        key=lambda a: float(a.get("score") or 0), reverse=True)[:top_k]
        for a in ranked:
            uid = a.get("unit_id")
            if not uid:
                continue
            node_id = f"m_{str(uid)[:8]}"
            node = self.nodes.get(node_id)
            if node is None:
                node = ObsNode(id=node_id, kind="memory_ref",
                               attrs={"unit_id": uid,
                                      "extract_type": a.get("extract_type", "?"),
                                      "attribution": 0.0, "n_hits": 0})
                self.nodes[node_id] = node
            node.attrs["attribution"] = round(
                max(node.attrs.get("attribution", 0.0), float(a.get("score") or 0)), 4)
            node.attrs["n_hits"] = node.attrs.get("n_hits", 0) + 1
            # Link to levels. The public contract allows attribution entries
            # without `helps_levels`; in that case link to all known levels so
            # the unit still surfaces in per-pattern high_value_units. (The
            # node is also always exposed via the global section below.)
            helps = a.get("helps_levels")
            target_levels = ([_to_bucket(l) for l in helps] if helps else list(known_levels))
            for lvl in target_levels:
                if lvl in _VALID_LEVELS:
                    self._ensure_task_pattern_node(lvl)
                    self._edges.setdefault(
                        (f"L{lvl}", REL_HIGH_ATTRIBUTION, node_id),
                        ObsEdge(src=f"L{lvl}", rel=REL_HIGH_ATTRIBUTION, dst=node_id),
                    )

    # ------------------------------------------------------------------
    # Serialisation for the Proposer prompt
    # ------------------------------------------------------------------
    def to_proposer_json(self, max_memory_refs: int = 15) -> str:
        """Compact, Proposer-friendly view. Aggregates edges under each
        task_pattern so the LLM reads one block per level."""
        with self._lock:
            out: Dict[str, Any] = {
                "round": self.updated_at_round,
                "patterns": {},
            }

            # Global top attributed units — only emitted when actually present.
            # (Attribution is not wired in the automem P1 integration, so this
            # stays absent there rather than advertising an empty list.)
            mem_refs = [n for n in self.nodes.values() if n.kind == "memory_ref"]
            mem_refs.sort(key=lambda n: float(n.attrs.get("attribution") or 0),
                          reverse=True)
            if mem_refs:
                out["top_memory_units"] = [
                    {
                        "type": ref.attrs.get("extract_type"),
                        "attribution": ref.attrs.get("attribution"),
                        "n_hits": ref.attrs.get("n_hits"),
                    }
                    for ref in mem_refs[:max_memory_refs]
                ]

            for node in self.nodes.values():
                if node.kind != "task_pattern":
                    continue
                lvl_id = node.id
                block: Dict[str, Any] = {
                    "n_tasks": node.attrs.get("n_tasks_seen", 0),
                    "baseline_acc": node.attrs.get("baseline_acc"),
                    "best_acc_so_far": node.attrs.get("best_acc_so_far"),
                    "categories": node.attrs.get("categories", {}),
                    # Authoritative joint evidence: full architectures actually
                    # evaluated and their per-level accuracy.
                    "architectures_evaluated": {},
                    # Marginal aggregates (breadth hints only — do NOT pair two
                    # marginals as if they were jointly validated).
                    "extract_combos": {},
                    "retrievers": {},
                    "high_value_units": [],
                }
                out["patterns"][lvl_id] = block

            # fold edges into pattern blocks
            for e in self._edges.values():
                blk = out["patterns"].get(e.src)
                if blk is None:
                    continue
                if e.rel == REL_ARCH_EVALUATED:
                    blk["architectures_evaluated"][e.dst] = {
                        "avg_acc": e.attrs.get("avg_acc"),
                        "n": e.attrs.get("n_trials"),
                    }
                elif e.rel == REL_TRIED_WITH:
                    blk["extract_combos"][e.dst.replace("extract:", "")] = {
                        "avg_acc": e.attrs.get("avg_acc"),
                        "n": e.attrs.get("n_trials"),
                    }
                elif e.rel == REL_WORKED_WITH_RETRIEVER:
                    blk["retrievers"][e.dst.replace("retrieval:", "")] = {
                        "avg_acc": e.attrs.get("avg_acc"),
                        "n": e.attrs.get("n_trials"),
                    }
                elif e.rel == REL_HIGH_ATTRIBUTION:
                    ref = self.nodes.get(e.dst)
                    if ref and len(blk["high_value_units"]) < max_memory_refs:
                        blk["high_value_units"].append({
                            "type": ref.attrs.get("extract_type"),
                            "attribution": ref.attrs.get("attribution"),
                        })

            return json.dumps(out, indent=2, ensure_ascii=False)
