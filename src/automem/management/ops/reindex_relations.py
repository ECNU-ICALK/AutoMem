"""
ReindexRelationsOp — Recompute inter-unit similarity relations and
co-occurrence edges based on embeddings and task provenance.

Part of the 'episodic_consolidation' operation group.
"""

import time
import logging
from typing import Any, Dict, List

import numpy as np

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnit, MemoryRelation, RelationType

logger = logging.getLogger(__name__)


class ReindexRelationsOp(BaseManageOp):
    """
    Reindex similarity and co-occurrence relations between memory units.

    Graph stores (native): iterate content node pairs, compute cosine
    similarity, and upsert SIMILAR / COOCCURS edges directly in the graph.

    Non-graph stores (weak): compute pairwise embedding similarity,
    update each unit's relations list, and persist via store.update().
    """

    op_name = "reindex_relations"
    op_group = "episodic_consolidation"
    trigger_type = TriggerType.PERIODIC
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = True
    rl_action_id = 3

    _DEFAULT_CONFIG = {
        "similarity_threshold": 0.7,
    }

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name, triggered=True)

        try:
            sim_threshold = self.config.get(
                "similarity_threshold",
                self._DEFAULT_CONFIG["similarity_threshold"],
            )

            all_units: List[MemoryUnit] = self.store.get_all()
            active_units = [u for u in all_units if u.is_active]

            if len(active_units) < 2:
                logger.info("reindex_relations: fewer than 2 active units, skipping")
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            if self._is_graph_store():
                edges_added, edges_updated, edges_removed = self._reindex_graph(
                    active_units, sim_threshold
                )
                result.units_affected = edges_added + edges_updated + edges_removed
                result.details = {
                    "mode": "graph",
                    "edges_added": edges_added,
                    "edges_updated": edges_updated,
                    "edges_removed": edges_removed,
                    "active_units": len(active_units),
                }
            else:
                units_modified = self._reindex_non_graph(
                    active_units, sim_threshold
                )
                result.units_modified = units_modified
                result.units_affected = units_modified
                result.details = {
                    "mode": "non_graph",
                    "units_modified": units_modified,
                    "active_units": len(active_units),
                }

        except Exception as e:
            logger.error(
                "reindex_relations: execution failed: %s", e, exc_info=True
            )
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        logger.info("reindex_relations: completed in %.1fms", result.duration_ms)
        return result

    # ------------------------------------------------------------------
    # Graph-native reindexing
    # ------------------------------------------------------------------

    def _reindex_graph(
        self, units: List[MemoryUnit], sim_threshold: float
    ) -> tuple:
        """Reindex SIMILAR and COOCCURS edges in the graph store."""
        graph = self.store._graph
        edges_added = 0
        edges_updated = 0
        edges_removed = 0

        # Filter to units with embeddings
        with_emb = [u for u in units if u.embedding is not None]

        if len(with_emb) >= 2:
            embeddings = np.array([u.embedding for u in with_emb])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            normed = embeddings / norms
            sim_matrix = normed @ normed.T

            n = len(with_emb)
            for i in range(n):
                for j in range(i + 1, n):
                    u_i = with_emb[i]
                    u_j = with_emb[j]
                    sim = float(sim_matrix[i, j])

                    nid_i = self.store._content_nid(u_i.id)
                    nid_j = self.store._content_nid(u_j.id)

                    # SIMILAR edges. Existing edges are located by edge_type,
                    # NOT by key: GraphStore serialization drops edge keys, so
                    # after a save/load cycle the "SIMILAR" key becomes an
                    # integer and the old key-addressed remove silently failed
                    # (re-add then accumulated parallel duplicates across
                    # restarts). This also preserves and consolidates the G1
                    # usage/success statistics through the weight refresh.
                    _existing = [
                        (_k, _d)
                        for _k, _d in (graph.get_edge_data(nid_i, nid_j) or {}).items()
                        if _d.get("edge_type") == "SIMILAR"
                    ]
                    if sim >= sim_threshold:
                        if _existing:
                            _stats: Dict[str, int] = {}
                            for _k, _d in _existing:
                                for _sk in ("usage_count", "success_count"):
                                    if _sk in _d:
                                        _stats[_sk] = _stats.get(_sk, 0) + int(_d[_sk] or 0)
                            for _k, _d in _existing:
                                try:
                                    graph.remove_edge(nid_i, nid_j, key=_k)
                                except Exception:
                                    pass
                            graph.add_edge(
                                nid_i, nid_j,
                                key="SIMILAR",
                                edge_type="SIMILAR",
                                weight=sim,
                                **_stats,
                            )
                            edges_updated += 1
                        else:
                            graph.add_edge(
                                nid_i, nid_j,
                                key="SIMILAR",
                                edge_type="SIMILAR",
                                weight=sim,
                            )
                            edges_added += 1
                    elif _existing:
                        # Similarity dropped below the threshold since the
                        # edge was created — remove it. Retrieval walks
                        # SIMILAR edges without re-checking similarity, so a
                        # stale edge keeps propagating scores forever.
                        for _k, _d in _existing:
                            try:
                                graph.remove_edge(nid_i, nid_j, key=_k)
                                edges_removed += 1
                            except Exception:
                                pass

                    # COOCCURS edges for units from the same task
                    if (
                        u_i.source_task_id
                        and u_i.source_task_id == u_j.source_task_id
                        and not self.store._has_edge(nid_i, nid_j, "COOCCURS")
                    ):
                        graph.add_edge(
                            nid_i, nid_j,
                            key="COOCCURS",
                            edge_type="COOCCURS",
                            weight=1.0,
                        )
                        edges_added += 1

        # Also handle COOCCURS for units without embeddings
        task_groups: Dict[str, List[MemoryUnit]] = {}
        for u in units:
            if u.source_task_id:
                task_groups.setdefault(u.source_task_id, []).append(u)

        for task_id, group in task_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    nid_i = self.store._content_nid(group[i].id)
                    nid_j = self.store._content_nid(group[j].id)
                    if (
                        graph.has_node(nid_i)
                        and graph.has_node(nid_j)
                        and not self.store._has_edge(nid_i, nid_j, "COOCCURS")
                    ):
                        graph.add_edge(
                            nid_i, nid_j,
                            key="COOCCURS",
                            edge_type="COOCCURS",
                            weight=1.0,
                        )
                        edges_added += 1

        # Drop SIMILAR edges whose endpoints are no longer ACTIVE content
        # nodes (soft-deleted / physically removed units). Retrieval filters
        # inactive units at expansion time, but the dead edges bloat the
        # graph indefinitely. Restrict to m:* content nodes so entity/query
        # layer edges are never touched.
        active_nids = {self.store._content_nid(u.id) for u in units}
        stale_edges = [
            (s, t, k)
            for s, t, k, d in graph.edges(keys=True, data=True)
            if d.get("edge_type") == "SIMILAR"
            and s.startswith("m:") and t.startswith("m:")
            and (s not in active_nids or t not in active_nids)
        ]
        for s, t, k in stale_edges:
            try:
                graph.remove_edge(s, t, key=k)
                edges_removed += 1
            except Exception:
                pass

        # Persist. This op used to mutate the in-memory graph WITHOUT saving:
        # its edges only reached disk by riding along a later unrelated write,
        # and the final cycle before shutdown was lost entirely.
        try:
            self.store.save()
        except Exception as e:
            logger.warning("reindex_relations: store.save failed: %s", e)

        logger.info(
            "reindex_relations (graph): added=%d, updated=%d, removed=%d",
            edges_added, edges_updated, edges_removed,
        )
        return edges_added, edges_updated, edges_removed

    # ------------------------------------------------------------------
    # Non-graph (weak) reindexing
    # ------------------------------------------------------------------

    def _reindex_non_graph(
        self, units: List[MemoryUnit], sim_threshold: float
    ) -> int:
        """Reindex relations via unit.relations lists and store.update()."""
        # Filter to units with embeddings
        with_emb = [u for u in units if u.embedding is not None]
        units_modified = 0

        if len(with_emb) < 2:
            return 0

        embeddings = np.array([u.embedding for u in with_emb])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        # Active IDs for stale-relation pruning (units passed in are already active)
        active_ids = {u.id for u in units}

        n = len(with_emb)
        modified_ids = set()

        for i in range(n):
            u_i = with_emb[i]
            changed = False

            # --- Prune stale SIMILAR / COOCCURS pointing to inactive units ---
            before = len(u_i.relations)
            u_i.relations = [
                r for r in u_i.relations
                if r.relation_type not in (RelationType.SIMILAR, RelationType.COOCCURS)
                or r.target_id in active_ids
            ]
            if len(u_i.relations) < before:
                changed = True

            existing_similar = {
                r.target_id
                for r in u_i.relations
                if r.relation_type == RelationType.SIMILAR
            }
            existing_cooccurs = {
                r.target_id
                for r in u_i.relations
                if r.relation_type == RelationType.COOCCURS
            }

            for j in range(n):
                if i == j:
                    continue
                u_j = with_emb[j]
                sim = float(sim_matrix[i, j])

                # Add SIMILAR relation if above threshold
                if sim >= sim_threshold and u_j.id not in existing_similar:
                    u_i.relations.append(
                        MemoryRelation(
                            target_id=u_j.id,
                            relation_type=RelationType.SIMILAR,
                            weight=sim,
                        )
                    )
                    changed = True

                # Add COOCCURS relation for same-task units
                if (
                    u_i.source_task_id
                    and u_i.source_task_id == u_j.source_task_id
                    and u_j.id not in existing_cooccurs
                ):
                    u_i.relations.append(
                        MemoryRelation(
                            target_id=u_j.id,
                            relation_type=RelationType.COOCCURS,
                            weight=1.0,
                        )
                    )
                    changed = True

            if changed:
                self.store.update(u_i)
                modified_ids.add(u_i.id)

        units_modified = len(modified_ids)
        logger.info(
            "reindex_relations (non-graph): %d units modified", units_modified
        )
        return units_modified
