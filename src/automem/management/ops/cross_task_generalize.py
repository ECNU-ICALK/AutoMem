"""
CrossTaskGeneralizeOp — Identify insights and tips that appear across
different tasks and synthesize cross-domain principles via LLM.

Part of the 'episodic_consolidation' operation group.
"""

import time
import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


class CrossTaskGeneralizeOp(BaseManageOp):
    """
    Find TIP/INSIGHT units from different tasks that share high semantic
    similarity (or, in graph stores, share entity nodes), then ask an LLM
    to synthesize a cross-domain principle.  New units are linked via
    REINFORCES relations.
    """

    op_name = "cross_task_generalize"
    op_group = "episodic_consolidation"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = True
    requires_embedding = True
    rl_action_id = 2

    _DEFAULT_CONFIG = {
        "min_cross_task_pairs": 2,
        "similarity_threshold": 0.85,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            min_pairs = self.config.get(
                "min_cross_task_pairs",
                self._DEFAULT_CONFIG["min_cross_task_pairs"],
            )
            sim_threshold = self.config.get(
                "similarity_threshold",
                self._DEFAULT_CONFIG["similarity_threshold"],
            )

            # Step 1: Collect active TIP + INSIGHT units
            all_units: List[MemoryUnit] = self.store.get_all()
            candidates = [
                u for u in all_units
                if u.is_active
                and u.type in (MemoryUnitType.TIP, MemoryUnitType.INSIGHT)
            ]

            if len(candidates) < 2:
                logger.info(
                    "cross_task_generalize: only %d candidates, skipping",
                    len(candidates),
                )
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            # Collect groups of related units from different tasks
            groups_to_synthesize: List[List[MemoryUnit]] = []

            if self._is_graph_store():
                # Step 3 (graph-enhanced): find entity nodes connected to 3+
                # different content nodes from different tasks
                try:
                    groups_to_synthesize = self._find_graph_cross_task_groups(
                        candidates, min_pairs
                    )
                except Exception as e:
                    logger.warning(
                        "cross_task_generalize: graph path failed, "
                        "falling back to embedding: %s", e,
                    )
                    groups_to_synthesize = self._find_embedding_cross_task_groups(
                        candidates, sim_threshold, min_pairs
                    )
            else:
                # Step 2 (non-graph): embedding similarity
                groups_to_synthesize = self._find_embedding_cross_task_groups(
                    candidates, sim_threshold, min_pairs
                )

            if not groups_to_synthesize:
                logger.info("cross_task_generalize: no qualifying cross-task groups")
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            units_created = 0

            # Step 4: Synthesize cross-domain principles
            for group in groups_to_synthesize:
                summaries = []
                task_ids = set()
                for u in group:
                    summaries.append(
                        f"- [Task: {u.source_task_id[:8] if u.source_task_id else 'unknown'}] "
                        f"[{u.type.value}] {u.content_text()[:300]}"
                    )
                    task_ids.add(u.source_task_id)

                prompt = (
                    "You are a knowledge synthesis assistant. The following memory "
                    "entries come from DIFFERENT tasks but share common themes or "
                    "patterns.\n\n"
                    f"Source memories (from {len(task_ids)} different tasks):\n"
                    + "\n".join(summaries)
                    + "\n\nSynthesize a single cross-domain principle or tip that "
                    "captures the general pattern applicable across all these tasks. "
                    "Output ONLY the generalized principle, no preamble."
                )

                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
                response = self.llm_client(messages)
                response_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )

                # Step 5: Create new TIP unit with REINFORCES relations
                new_unit = MemoryUnit(
                    id=str(uuid.uuid4()),
                    type=MemoryUnitType.TIP,
                    content={
                        "topic": "cross_task_principle",
                        "principle": response_text.strip(),
                        "source_task_count": len(task_ids),
                        "source_unit_count": len(group),
                    },
                    source_task_id="cross_task",
                    source_task_query="",
                    task_outcome="",
                    confidence=min(u.confidence for u in group),
                    usage_count=0,
                    success_count=0,
                    decay_weight=1.0,
                    is_active=True,
                )
                new_unit.compute_signature()
                new_unit.token_estimate()

                # Compute embedding
                if self.embedding_model is not None:
                    try:
                        new_unit.embedding = self.embedding_model.encode(
                            new_unit.content_text()
                        )
                    except Exception as e:
                        logger.warning(
                            "cross_task_generalize: embedding failed: %s", e
                        )

                # Add REINFORCES relations to source units
                for u in group:
                    new_unit.relations.append(
                        MemoryRelation(
                            target_id=u.id,
                            relation_type=RelationType.REINFORCES,
                            weight=1.0,
                        )
                    )

                self.store.add([new_unit])  # fix: JsonStorage.add expects list
                units_created += 1

                logger.info(
                    "cross_task_generalize: created principle %s from %d units "
                    "across %d tasks",
                    new_unit.id[:8], len(group), len(task_ids),
                )

            result.units_created = units_created
            result.units_affected = units_created
            result.details = {
                "total_candidates": len(candidates),
                "groups_synthesized": units_created,
                "similarity_threshold": sim_threshold,
            }

        except Exception as e:
            logger.error(
                "cross_task_generalize: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info(
            "cross_task_generalize: completed in %.1fms", result.duration_ms
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_embedding_cross_task_groups(
        self,
        candidates: List[MemoryUnit],
        sim_threshold: float,
        min_pairs: int,
    ) -> List[List[MemoryUnit]]:
        """Find groups of cross-task units via pairwise embedding similarity."""
        # Filter to those with embeddings
        with_emb = [u for u in candidates if u.embedding is not None]
        if len(with_emb) < 2:
            return []

        embeddings = np.array([u.embedding for u in with_emb])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # Find cross-task pairs above threshold
        paired_indices: Set[int] = set()
        pairs: List[Tuple[int, int]] = []
        n = len(with_emb)
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    sim_matrix[i, j] >= sim_threshold
                    and with_emb[i].source_task_id != with_emb[j].source_task_id
                    and with_emb[i].source_task_id
                    and with_emb[j].source_task_id
                ):
                    pairs.append((i, j))
                    paired_indices.add(i)
                    paired_indices.add(j)

        if len(pairs) < min_pairs:
            return []

        # Simple greedy grouping: connected components via pairs
        groups: List[List[MemoryUnit]] = []
        visited: Set[int] = set()

        for i in paired_indices:
            if i in visited:
                continue
            # BFS to find connected component
            component = set()
            queue = [i]
            while queue:
                node = queue.pop()
                if node in component:
                    continue
                component.add(node)
                for a, b in pairs:
                    if a == node and b not in component:
                        queue.append(b)
                    elif b == node and a not in component:
                        queue.append(a)
            visited |= component
            group_units = [with_emb[idx] for idx in component]
            # Ensure at least 2 different tasks
            task_ids = set(u.source_task_id for u in group_units)
            if len(task_ids) >= 2 and len(group_units) >= 2:
                groups.append(group_units)

        return groups

    def _find_graph_cross_task_groups(
        self, candidates: List[MemoryUnit], min_pairs: int
    ) -> List[List[MemoryUnit]]:
        """Find cross-task groups via shared entity nodes in graph store."""
        graph = self.store._graph

        # Map entity nodes to the content nodes (units) they connect to
        entity_to_units: Dict[str, List[MemoryUnit]] = defaultdict(list)

        for unit in candidates:
            content_nid = self.store._content_nid(unit.id)
            if not graph.has_node(content_nid):
                continue
            for _, target, data in graph.edges(content_nid, data=True):
                if data.get("edge_type") == "HAS_ENTITY":
                    entity_to_units[target].append(unit)

        # Find entities connected to 3+ units from different tasks
        groups: List[List[MemoryUnit]] = []
        seen_unit_sets: Set[frozenset] = set()

        for entity_nid, units in entity_to_units.items():
            if len(units) < 3:
                continue
            task_ids = set(u.source_task_id for u in units if u.source_task_id)
            if len(task_ids) < 2:
                continue

            unit_key = frozenset(u.id for u in units)
            if unit_key in seen_unit_sets:
                continue
            seen_unit_sets.add(unit_key)
            groups.append(units)

        return groups
