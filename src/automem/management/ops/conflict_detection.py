"""
Conflict detection operation.

Identifies memory units with high embedding similarity but opposite
task_outcome values, marking them as CONFLICTS. When the score gap is
large enough the weaker unit is deactivated.
"""

import time
import logging
from typing import Any, Dict, List, Set

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryRelation, RelationType

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class ConflictDetectionOp(BaseManageOp):
    """Detect conflicting memory units based on embedding similarity and outcome."""

    op_name = "conflict_detection"
    op_group = "deduplication"
    trigger_type = TriggerType.ON_INSERT
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = True
    rl_action_id = 7

    def execute(self, context: Dict[str, Any]) -> OpResult:
        result = OpResult(op_name=self.op_name)
        t0 = time.time()

        threshold = self.config.get("conflict_threshold", 0.85)

        try:
            all_units = self.store.get_all()
            active_units = [u for u in all_units if u.is_active and u.embedding is not None]

            # Determine which units to focus on
            new_unit_ids: List[str] | None = context.get("new_unit_ids")
            if new_unit_ids:
                target_units = [u for u in active_units if u.id in set(new_unit_ids)]
                candidate_units = active_units
            else:
                target_units = active_units
                candidate_units = active_units

            conflict_pairs: List[tuple] = []
            deactivated_ids: Set[str] = set()
            processed_pairs: Set[tuple] = set()
            total_conflicts = 0
            total_deactivated = 0

            for target in target_units:
                if target.id in deactivated_ids:
                    continue

                for candidate in candidate_units:
                    if candidate.id == target.id or candidate.id in deactivated_ids:
                        continue

                    # Avoid processing the same pair twice
                    pair_key = tuple(sorted((target.id, candidate.id)))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)

                    # Check embedding similarity
                    sim = _cosine_similarity(target.embedding, candidate.embedding)
                    if sim < threshold:
                        continue

                    # Codex Q4-4 fix (2026-04-28): require an explicit
                    # success/failure pair. Previously any non-equal,
                    # non-None outcome qualified, so a cluster-merged or
                    # legacy unit with task_outcome="" would conflict with
                    # a real "success" unit and one of them would be
                    # deactivated for no real reason.
                    OUTCOMES = {"success", "failure"}
                    t_o = (target.task_outcome or "").strip().lower()
                    c_o = (candidate.task_outcome or "").strip().lower()
                    if t_o not in OUTCOMES or c_o not in OUTCOMES:
                        continue
                    if t_o == c_o:
                        continue

                    # Opposite outcomes with high similarity -> CONFLICT
                    conflict_rel_a = MemoryRelation(
                        target_id=candidate.id,
                        relation_type=RelationType.CONFLICTS,
                        weight=sim,
                    )
                    conflict_rel_b = MemoryRelation(
                        target_id=target.id,
                        relation_type=RelationType.CONFLICTS,
                        weight=sim,
                    )
                    target.relations.append(conflict_rel_a)
                    candidate.relations.append(conflict_rel_b)

                    # Increment conflict_count signal so MEMORY_STALE
                    # attribution branch can fire on these units.
                    target.conflict_count = int(getattr(target, "conflict_count", 0)) + 1
                    candidate.conflict_count = int(getattr(candidate, "conflict_count", 0)) + 1

                    # Graph enhanced: add CONFLICTS edges
                    if self._is_graph_store():
                        graph = self.store._graph
                        target_nid = self.store._content_nid(target.id)
                        candidate_nid = self.store._content_nid(candidate.id)
                        graph.add_edge(
                            target_nid,
                            candidate_nid,
                            relation=RelationType.CONFLICTS.value,
                            weight=sim,
                        )
                        graph.add_edge(
                            candidate_nid,
                            target_nid,
                            relation=RelationType.CONFLICTS.value,
                            weight=sim,
                        )

                    total_conflicts += 1
                    conflict_pairs.append((target.id, candidate.id, sim))

                    # If score gap is large, deactivate the weaker unit
                    score_a = target.effective_score
                    score_b = candidate.effective_score
                    higher_score = max(score_a, score_b)
                    lower_score = min(score_a, score_b)

                    if higher_score > 0 and lower_score < 0.5 * higher_score:
                        if score_a < score_b:
                            weaker = target
                        else:
                            weaker = candidate

                        weaker.is_active = False
                        # Track who superseded the weaker unit so downstream
                        # ops + diagnostics can trace the chain.
                        if score_a < score_b:
                            weaker.superseded_by = candidate.id
                        else:
                            weaker.superseded_by = target.id
                        deactivated_ids.add(weaker.id)
                        total_deactivated += 1
                        logger.debug(
                            "Conflict detection: deactivated %s (score=%.3f vs %.3f)",
                            weaker.id, lower_score, higher_score,
                        )

                    self.store.update(target)
                    self.store.update(candidate)

                    # Codex Q4-5 fix (2026-04-28): if THIS target was just
                    # deactivated, stop comparing it to remaining
                    # candidates. Continuing would let a single weak unit
                    # stamp `conflict_count` on every nearby strong unit
                    # before being removed from the scan, biasing
                    # size_capped_prune toward those good units later.
                    if target.id in deactivated_ids:
                        break

            # Rebuild FAISS index if any deactivations occurred
            if total_deactivated > 0:
                self._rebuild_faiss_if_needed()

            result.triggered = total_conflicts > 0
            result.units_affected = total_conflicts * 2 + total_deactivated
            result.units_deleted = total_deactivated
            result.units_modified = total_conflicts * 2
            result.details = {
                "threshold": threshold,
                "targets_scanned": len(target_units),
                "candidates_scanned": len(candidate_units),
                "conflict_pairs": len(conflict_pairs),
                "deactivated_ids": list(deactivated_ids),
            }

        except Exception:
            logger.exception("ConflictDetectionOp failed")
            result.details["error"] = "exception during execution"

        result.duration_ms = (time.time() - t0) * 1000
        return result
