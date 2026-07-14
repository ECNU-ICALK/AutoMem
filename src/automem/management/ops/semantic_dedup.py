"""
Semantic deduplication operation.

Within each MemoryUnitType group, computes pairwise cosine similarity of
embeddings and deactivates near-duplicate units, transferring stats and
adding SUPERSEDES relations to the survivor.
"""

import time
import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SemanticDedupOp(BaseManageOp):
    """Deduplicate semantically similar memory units within the same type."""

    op_name = "semantic_dedup"
    op_group = "deduplication"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = True
    rl_action_id = 5

    def execute(self, context: Dict[str, Any]) -> OpResult:
        result = OpResult(op_name=self.op_name)
        t0 = time.time()

        threshold = self.config.get("semantic_threshold", 0.88)

        try:
            all_units = self.store.get_all()
            active_units = [u for u in all_units if u.is_active and u.embedding is not None]

            # Group by type
            type_groups: Dict[MemoryUnitType, List[MemoryUnit]] = defaultdict(list)
            for unit in active_units:
                type_groups[unit.type].append(unit)

            total_deactivated = 0
            total_modified = 0
            deactivated_ids: set = set()

            for utype, group in type_groups.items():
                if len(group) < 2:
                    continue

                # Pairwise cosine similarity
                for i in range(len(group)):
                    if group[i].id in deactivated_ids:
                        continue
                    for j in range(i + 1, len(group)):
                        if group[j].id in deactivated_ids:
                            continue

                        sim = _cosine_similarity(group[i].embedding, group[j].embedding)
                        if sim < threshold:
                            continue

                        # Keep the one with higher effective_score
                        if group[i].effective_score >= group[j].effective_score:
                            survivor, victim = group[i], group[j]
                        else:
                            survivor, victim = group[j], group[i]

                        # Transfer stats
                        survivor.usage_count += victim.usage_count
                        survivor.success_count += victim.success_count

                        # Deactivate victim
                        victim.is_active = False
                        deactivated_ids.add(victim.id)

                        # Add SUPERSEDES relation to survivor
                        supersedes_rel = MemoryRelation(
                            target_id=victim.id,
                            relation_type=RelationType.SUPERSEDES,
                            weight=sim,
                        )
                        survivor.relations.append(supersedes_rel)

                        # Graph-enhanced: redirect edges and add SUPERSEDES edge
                        if self._is_graph_store():
                            graph = self.store._graph
                            victim_nid = self.store._content_nid(victim.id)
                            survivor_nid = self.store._content_nid(survivor.id)

                            # Redirect incoming edges from victim to survivor
                            if graph.has_node(victim_nid):
                                for pred in list(graph.predecessors(victim_nid)):
                                    for key, data in list(
                                        graph[pred][victim_nid].items()
                                    ):
                                        if not graph.has_edge(pred, survivor_nid):
                                            graph.add_edge(pred, survivor_nid, **data)
                                # Redirect outgoing edges from victim to survivor
                                for succ in list(graph.successors(victim_nid)):
                                    for key, data in list(
                                        graph[victim_nid][succ].items()
                                    ):
                                        if not graph.has_edge(survivor_nid, succ):
                                            graph.add_edge(survivor_nid, succ, **data)

                            # Add SUPERSEDES edge in graph
                            graph.add_edge(
                                survivor_nid,
                                victim_nid,
                                relation=RelationType.SUPERSEDES.value,
                                weight=sim,
                            )

                        self.store.update(victim)
                        self.store.update(survivor)
                        total_deactivated += 1
                        total_modified += 1
                        logger.debug(
                            "Semantic dedup: deactivated %s (sim=%.3f with %s)",
                            victim.id, sim, survivor.id,
                        )

            # Rebuild FAISS index if any deactivations occurred
            if total_deactivated > 0:
                self._rebuild_faiss_if_needed()

            result.triggered = total_deactivated > 0
            result.units_affected = total_deactivated + total_modified
            result.units_deleted = total_deactivated  # deactivated, not hard-deleted
            result.units_modified = total_modified
            result.details = {
                "threshold": threshold,
                "total_active_with_embedding": len(active_units),
                "deactivated_ids": list(deactivated_ids),
            }

        except Exception:
            logger.exception("SemanticDedupOp failed")
            result.details["error"] = "exception during execution"

        result.duration_ms = (time.time() - t0) * 1000
        return result
