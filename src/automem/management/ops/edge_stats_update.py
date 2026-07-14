"""
EdgeStatsUpdateOp — Record usage/success statistics on graph edges that
retrieval expansion actually traversed.

Part of the 'graph_consolidate' operation group (G1, 2026-07-11 — ported from
the cerebra_fusion_memory evolved architecture, whose edge-level feedback
loop had no equivalent in the four-module framework).

Data flow: GraphRetriever records (source_nid, target_nid, edge_type)
triples for every m->m edge used during expansion -> MemoryPack.used_edges
-> MultiStoreRetriever aggregation -> ModularMemoryProvider tracks them per
query -> take_in_memory passes them into the management context as
``used_edge_pairs`` (mirroring ``used_unit_ids``) -> this op bumps
``usage_count`` (and ``success_count`` when the task succeeded) on the
matching edges. edge_weight_optimize later converts these statistics into
adaptive edge weights without deleting graph relations.

NOTE: only GraphRetriever reports used_edges today. llm_graph's default
retriever (hybrid_graph) does not, so on llm_graph architectures this op is
a harmless no-op unless retrieval is explicitly routed to `graph`.
"""

import logging
import time
from typing import Any, Dict

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType

logger = logging.getLogger(__name__)


class EdgeStatsUpdateOp(BaseManageOp):
    """Bump usage/success counters on graph edges used by this task's retrieval."""

    op_name = "edge_stats_update"
    op_group = "graph_consolidate"
    trigger_type = TriggerType.POST_TASK
    storage_compatibility = StorageCompatibility.GRAPH_ENHANCED
    requires_llm = False
    requires_embedding = False
    rl_action_id = 23

    _DEFAULT_CONFIG: Dict[str, Any] = {}

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name)

        try:
            edge_pairs = context.get("used_edge_pairs") or []
            if not edge_pairs or not self._is_graph_store():
                result.triggered = False
                result.duration_ms = (time.time() - t0) * 1000
                return result

            succeeded = bool(context.get("task_succeeded", False))
            graph = self.store._graph
            bumped = 0

            for pair in edge_pairs:
                try:
                    u, v, etype = pair[0], pair[1], pair[2]
                except Exception:
                    continue
                # Expansion walks direction="both", so the recorded
                # (source, target) may be the reverse of the stored edge.
                # Bump the FIRST direction that matches, not both — a single
                # traversal used a single edge.
                hit = False
                for a, b in ((u, v), (v, u)):
                    if hit:
                        break
                    if not graph.has_edge(a, b):
                        continue
                    for _key, data in graph[a][b].items():
                        if etype and data.get("edge_type") != etype:
                            continue
                        data["usage_count"] = int(data.get("usage_count", 0) or 0) + 1
                        if succeeded:
                            data["success_count"] = int(data.get("success_count", 0) or 0) + 1
                        bumped += 1
                        hit = True
                        break

            if bumped:
                try:
                    self.store.save()
                except Exception as e:
                    logger.warning("edge_stats_update: store.save failed: %s", e)

            result.triggered = bumped > 0
            result.units_affected = bumped
            result.details = {
                "edges_bumped": bumped,
                "edge_pairs_seen": len(edge_pairs),
                "task_succeeded": succeeded,
            }

        except Exception as e:
            logger.error("edge_stats_update: execution failed: %s", e, exc_info=True)
            result.details["error"] = str(e)

        result.duration_ms = (time.time() - t0) * 1000
        return result
