"""
ClusterMergeOp — Cluster semantically similar TIP/INSIGHT units and merge
them into consolidated summaries via LLM synthesis.

Part of the 'episodic_consolidation' operation group.
"""

import time
import logging
import uuid
from typing import Any, Dict, List

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryUnitType, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


class ClusterMergeOp(BaseManageOp):
    """
    Cluster semantically similar TIP and INSIGHT memories using hierarchical
    clustering, then ask an LLM to synthesize each cluster into a single
    consolidated memory unit.  Old units are decayed and linked via SUPERSEDES.
    """

    op_name = "cluster_merge"
    op_group = "episodic_consolidation"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = True
    requires_embedding = True
    rl_action_id = 0

    # Default configuration
    _DEFAULT_CONFIG = {
        "similarity_threshold": 0.80,
        "min_cluster_size": 3,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            sim_threshold = self.config.get(
                "similarity_threshold",
                self._DEFAULT_CONFIG["similarity_threshold"],
            )
            min_cluster_size = self.config.get(
                "min_cluster_size",
                self._DEFAULT_CONFIG["min_cluster_size"],
            )

            # Step 1: Collect active TIP + INSIGHT units with embeddings
            all_units: List[MemoryUnit] = self.store.get_all()
            candidates = [
                u for u in all_units
                if u.is_active
                and u.type in (MemoryUnitType.TIP, MemoryUnitType.INSIGHT)
                and u.embedding is not None
            ]

            if len(candidates) < min_cluster_size:
                logger.info(
                    "cluster_merge: only %d candidates (need %d), skipping",
                    len(candidates), min_cluster_size,
                )
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            # Step 2: Compute pairwise cosine similarity matrix
            embeddings = np.array([u.embedding for u in candidates])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            normed = embeddings / norms
            sim_matrix = normed @ normed.T

            # Step 3: Hierarchical clustering
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            # Convert similarity to condensed distance
            dist_matrix = 1.0 - sim_matrix
            np.fill_diagonal(dist_matrix, 0.0)
            dist_matrix = np.clip(dist_matrix, 0.0, None)
            condensed = squareform(dist_matrix, checks=False)

            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=1.0 - sim_threshold, criterion="distance")

            # Group by cluster label
            clusters: Dict[int, List[int]] = {}
            for idx, label in enumerate(labels):
                clusters.setdefault(int(label), []).append(idx)

            units_created = 0
            units_modified = 0

            # Step 4: Process clusters meeting min_cluster_size
            for cluster_id, indices in clusters.items():
                if len(indices) < min_cluster_size:
                    continue

                cluster_units = [candidates[i] for i in indices]

                # Build LLM synthesis prompt
                summaries = []
                for cu in cluster_units:
                    summaries.append(
                        f"- [{cu.type.value}] {cu.content_text()[:300]}"
                    )
                prompt = (
                    "You are a memory consolidation assistant. The following memory "
                    "entries are semantically similar and should be merged into a "
                    "single, concise, comprehensive tip.\n\n"
                    "Source memories:\n"
                    + "\n".join(summaries)
                    + "\n\nWrite a single consolidated tip that captures all key "
                    "information. Output ONLY the merged text, no preamble."
                )

                # Step 5: LLM call
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
                response = self.llm_client(messages)
                response_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )

                # Step 6: Create merged MemoryUnit
                merged_unit = MemoryUnit(
                    id=str(uuid.uuid4()),
                    type=MemoryUnitType.TIP,
                    content={
                        "topic": "consolidated_cluster",
                        "principle": response_text.strip(),
                        "source_count": len(cluster_units),
                    },
                    source_task_id=cluster_units[0].source_task_id,
                    source_task_query="",
                    task_outcome="",
                    confidence=max(cu.confidence for cu in cluster_units),
                    usage_count=0,
                    success_count=0,
                    decay_weight=1.0,
                    is_active=True,
                )
                merged_unit.compute_signature()
                merged_unit.token_estimate()

                # Compute embedding for merged unit if embedding_model available
                if self.embedding_model is not None:
                    try:
                        merged_unit.embedding = self.embedding_model.encode(
                            merged_unit.content_text()
                        )
                    except Exception as e:
                        logger.warning("cluster_merge: embedding failed: %s", e)

                self.store.add([merged_unit])  # fix: JsonStorage.add expects list
                units_created += 1

                # Step 7: Decay old units; merged_unit SUPERSEDES each old unit
                # (same direction as semantic_dedup: survivor points to victim)
                for cu in cluster_units:
                    cu.decay_weight *= 0.3
                    self.store.update(cu)
                    units_modified += 1

                    merged_unit.relations.append(
                        MemoryRelation(
                            target_id=cu.id,
                            relation_type=RelationType.SUPERSEDES,
                            weight=1.0,
                        )
                    )

                # Persist merged_unit with all SUPERSEDES relations now attached
                self.store.update(merged_unit)

                # Step 8: Graph-enhanced logic
                if self._is_graph_store():
                    try:
                        graph = self.store._graph
                        merged_nid = self.store._content_nid(merged_unit.id)

                        for cu in cluster_units:
                            old_nid = self.store._content_nid(cu.id)

                            # Add SUPERSEDES edge from merged to old (survivor → victim)
                            if not self.store._has_edge(
                                merged_nid, old_nid, "SUPERSEDES"
                            ):
                                graph.add_edge(
                                    merged_nid,
                                    old_nid,
                                    key="SUPERSEDES",
                                    edge_type="SUPERSEDES",
                                    weight=1.0,
                                )

                            # Transfer HAS_ENTITY edges to merged unit
                            if graph.has_node(old_nid):
                                for _, target, data in list(
                                    graph.edges(old_nid, data=True)
                                ):
                                    if data.get("edge_type") == "HAS_ENTITY":
                                        if not self.store._has_edge(
                                            merged_nid, target, "HAS_ENTITY"
                                        ):
                                            graph.add_edge(
                                                merged_nid,
                                                target,
                                                key="HAS_ENTITY",
                                                edge_type="HAS_ENTITY",
                                                weight=data.get("weight", 1.0),
                                            )
                    except Exception as e:
                        logger.warning(
                            "cluster_merge: graph enhancement failed: %s", e
                        )

                logger.info(
                    "cluster_merge: merged %d units into %s",
                    len(cluster_units), merged_unit.id[:8],
                )

            result.units_created = units_created
            result.units_modified = units_modified
            result.units_affected = units_created + units_modified
            result.details = {
                "total_candidates": len(candidates),
                "clusters_merged": units_created,
                "similarity_threshold": sim_threshold,
            }

        except Exception as e:
            logger.error("cluster_merge: execution failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info("cluster_merge: completed in %.1fms", result.duration_ms)
        return result
