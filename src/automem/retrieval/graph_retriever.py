"""
Graph Retriever — Vector seed + graph neighbor expansion.

Two-phase retrieval:
  1. Seed: Use semantic similarity to find initial relevant units
  2. Expand: Walk the graph (1-hop or multi-hop) from seed nodes,
     propagating scores with decay along edges

Requires a GraphStore backend to access the graph structure.
Falls back to pure semantic retrieval if the store lacks graph methods.

Inspired by cerebra_fusion's _graph_expand() pattern:
  propagated_score = base_score * edge_weight * decay_factor
"""

from typing import Any, Dict, List, Optional, Set

from sklearn.metrics.pairwise import cosine_similarity

from automem.memory_schema import MemoryUnit
from automem.retrieval.base_retriever import (
    BaseRetriever,
    MemoryPack,
    QueryContext,
    ScoredUnit,
    TraceEntry,
)


class GraphRetriever(BaseRetriever):
    """
    Semantic seed + graph neighbor expansion retriever.

    Config options:
        seed_k (int): Number of seed nodes from semantic search. Default 3.
        max_hops (int): Maximum graph expansion hops. Default 1.
        decay_factor (float): Score decay per hop. Default 0.7.
        min_score (float): Minimum score threshold. Default 0.1.
        active_only (bool): Only consider active units. Default True.
        expand_edge_types (List[str]): Edge types to follow.
            Default: ["SIMILAR", "COOCCURS", "REINFORCES", "HAS_MEMORY"]
    """

    def __init__(self, store, embedding_model=None, config: Optional[Dict[str, Any]] = None):
        super().__init__(store, config)
        self.embedding_model = embedding_model
        self.seed_k = self.config.get("seed_k", 3)
        # `graph_hop` is the architecture-search dimension name; accept it as
        # an alias so direct/flat configs work too.
        self.max_hops = self.config.get("max_hops", self.config.get("graph_hop", 1))
        self.decay_factor = self.config.get("decay_factor", 0.7)
        self.min_score = self.config.get("min_score", 0.1)
        self.active_only = self.config.get("active_only", True)
        # Codex Q6-1 fix: include HAS_ENTITY by default so the m→e→m
        # bridge below actually fires. Without it, expansion only walks
        # explicit MemoryRelation edges, which are sparse.
        self.expand_edge_types = self.config.get("expand_edge_types", [
            "SIMILAR", "COOCCURS", "REINFORCES", "HAS_MEMORY", "HAS_ENTITY",
        ])
        # Codex Q7-A3 fix (2026-04-28): cap m→e→m fan-out so a generic
        # entity (e.g. tool:web_search appearing in 200 memories) does
        # NOT flood the second-frontier with unrelated candidates and
        # blow the judge budget. Skip entities with neighbor count
        # above this threshold (treat them as "topic noise" rather
        # than meaningful relational signal).
        self.entity_bridge_max_degree = int(
            self.config.get("entity_bridge_max_degree", 25)
        )
        # Per-entity cap on how many m2 we accept (in case the entity
        # passes the degree filter but still fans out heavily).
        self.entity_bridge_top_k = int(
            self.config.get("entity_bridge_top_k", 8)
        )

    def _neighbors_with_weights(self, node_id: str, edge_type: Optional[str]):
        """Return (neighbor_id, edge_weight) pairs for graph expansion.

        Uses the store's neighbors_with_weights() when available (GraphStore
        persists rel.weight on edges); falls back to weight=1.0 for stores
        that only expose neighbors() — which preserves the legacy behaviour
        for them.
        """
        fn = getattr(self.store, "neighbors_with_weights", None)
        if fn is not None:
            return fn(node_id, edge_type=edge_type, direction="both")
        return [
            (nbr, 1.0)
            for nbr in self.store.neighbors(
                node_id, edge_type=edge_type, direction="both"
            )
        ]

    def retrieve(self, ctx: QueryContext, top_k: int = 5) -> MemoryPack:
        trace: List[TraceEntry] = []
        has_graph = hasattr(self.store, 'neighbors') and hasattr(self.store, 'get')

        # Cold-start guard: if the graph is too sparse for traversal, skip Phase 2
        # and fall back to seed-only retrieval. Uses StorageHealthReport when available.
        if has_graph and hasattr(self.store, 'get_health_report'):
            report = self.store.get_health_report()
            no_structural_graph = getattr(report, "graph_edge_count", 0) <= 0
            if report.unit_count == 0 or (
                (report.is_cold_start or report.retrieval_mode == "graph_sparse")
                and no_structural_graph
            ):
                has_graph = False
                trace.append(TraceEntry(
                    step=0,
                    method="graph_cold_start_fallback",
                    candidates=report.unit_count,
                    selected=0,
                    params={
                        "reason": report.retrieval_mode,
                        "unit_count": report.unit_count,
                        "threshold": report.cold_start_threshold,
                        "avg_degree": getattr(report, "graph_avg_degree", None),
                        "graph_edge_count": getattr(report, "graph_edge_count", None),
                    },
                ))

        # Phase 1: Semantic seed
        query_emb = ctx.embedding
        if query_emb is None and self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                ctx.query, convert_to_numpy=True
            )

        emb_matrix, emb_units = self.store.get_embedding_index(
            active_only=self.active_only
        )

        seed_scored: List[ScoredUnit] = []
        if query_emb is not None and emb_matrix is not None and len(emb_units) > 0:
            query_emb = query_emb.reshape(1, -1)
            sims = cosine_similarity(query_emb, emb_matrix)[0]
            seed_count = min(self.seed_k, len(emb_units))
            top_indices = sims.argsort()[-seed_count:][::-1]

            for idx in top_indices:
                score = float(sims[idx])
                if score >= self.min_score:
                    seed_scored.append(ScoredUnit(
                        unit=emb_units[idx],
                        score=score,
                        method="graph_seed",
                    ))

        trace.append(TraceEntry(
            step=1,
            method="graph_seed",
            candidates=len(emb_units) if emb_matrix is not None else 0,
            selected=len(seed_scored),
            params={"seed_k": self.seed_k},
        ))

        if not has_graph or not seed_scored:
            # No graph available or no seeds — return seed results only
            return self._make_pack(ctx, seed_scored[:top_k], trace)

        # Phase 2: Graph expansion
        score_board: Dict[str, float] = {}
        unit_map: Dict[str, MemoryUnit] = {}
        # (source_nid, target_nid, edge_type) triples for every m->m edge
        # actually used by expansion — consumed by the edge_stats_update
        # management op (G1). Entity-bridge hops are structural (HAS_ENTITY)
        # and deliberately not tracked.
        used_edges: List[tuple] = []

        # Initialize with seed scores
        for su in seed_scored:
            score_board[su.unit.id] = su.score
            unit_map[su.unit.id] = su.unit

        visited: Set[str] = set()
        frontier = [(su.unit.id, su.score) for su in seed_scored]
        total_expanded = 0

        for hop in range(self.max_hops):
            next_frontier = []
            for unit_id, base_score in frontier:
                if unit_id in visited:
                    continue
                visited.add(unit_id)

                node_id = f"m:{unit_id}"
                # Codex Q6-1 fix (2026-04-28): also traverse m→e→m via
                # HAS_ENTITY edges. GraphStore creates `m:* -> e:*` links
                # for every entity mentioned by a memory; without this
                # path, two memories that share an entity but lack a
                # direct MemoryRelation edge are unreachable, and the
                # graph arm degenerates to "semantic seeds + a few SIMILAR
                # neighbors". The HAS_ENTITY hop converts the entity
                # graph into useful retrieval signal.
                for edge_type in self.expand_edge_types:
                    neighbors = self._neighbors_with_weights(node_id, edge_type)
                    for nbr_id, edge_weight in neighbors:
                        # Memory-to-memory neighbor: direct propagation.
                        if nbr_id.startswith("m:"):
                            nbr_unit_id = nbr_id[2:]
                            if nbr_unit_id in visited:
                                continue
                            # propagated = base * edge_weight * decay (matches
                            # the module docstring). edge_weight was silently
                            # dropped before 2026-07-11, so a 0.01-weight edge
                            # propagated exactly like a 1.0 one.
                            propagated = base_score * edge_weight * self.decay_factor
                            if propagated < self.min_score:
                                continue
                            nbr_unit = self.store.get(nbr_unit_id)
                            if nbr_unit is None:
                                continue
                            if self.active_only and not nbr_unit.is_active:
                                continue
                            old_score = score_board.get(nbr_unit_id, 0.0)
                            new_score = max(old_score, propagated)
                            score_board[nbr_unit_id] = new_score
                            unit_map[nbr_unit_id] = nbr_unit
                            next_frontier.append((nbr_unit_id, new_score))
                            used_edges.append((node_id, nbr_id, edge_type))
                            total_expanded += 1
                        # Memory-to-entity neighbor: take a second hop
                        # back to other memories that share the entity.
                        elif nbr_id.startswith("e:"):
                            entity_propagated = base_score * edge_weight * self.decay_factor
                            if entity_propagated < self.min_score:
                                continue
                            try:
                                second_hop = self._neighbors_with_weights(nbr_id, None)
                            except Exception:
                                continue
                            # Codex Q7-A3 fix: skip generic high-degree
                            # entities (e.g. tool:web_search shared by
                            # 200+ memories) to keep retrieval focused.
                            if len(second_hop) > self.entity_bridge_max_degree:
                                continue
                            # Codex Q8-A6 fix (2026-04-28): pre-rank bridge
                            # candidates by confidence × decay before top_k
                            # so the selection is deterministic. Since
                            # 2026-07-11 each e→m edge weight also scales the
                            # propagated score (weightless HAS_ENTITY edges
                            # default to 1.0, preserving prior behaviour).
                            base_bridge = entity_propagated * self.decay_factor
                            ranked_candidates = []
                            for m2_id, w2 in second_hop:
                                if not m2_id.startswith("m:"):
                                    continue
                                m2_unit_id = m2_id[2:]
                                if m2_unit_id == unit_id or m2_unit_id in visited:
                                    continue
                                propagated = base_bridge * w2
                                if propagated < self.min_score:
                                    continue
                                m2_unit = self.store.get(m2_unit_id)
                                if m2_unit is None:
                                    continue
                                if self.active_only and not m2_unit.is_active:
                                    continue
                                # Rank by unit confidence (high-quality
                                # memories sort first); usage_count is
                                # also a useful signal but may not be
                                # populated on cold pools.
                                conf = float(getattr(m2_unit, "confidence", 1.0) or 1.0)
                                rank_score = propagated * conf
                                ranked_candidates.append(
                                    (rank_score, m2_unit_id, m2_unit, propagated)
                                )
                            ranked_candidates.sort(
                                key=lambda x: (-x[0], x[1])  # score DESC, id ASC for determinism
                            )
                            for r_score, m2_unit_id, m2_unit, m2_propagated in (
                                ranked_candidates[: self.entity_bridge_top_k]
                            ):
                                old_score = score_board.get(m2_unit_id, 0.0)
                                new_score = max(old_score, m2_propagated)
                                score_board[m2_unit_id] = new_score
                                unit_map[m2_unit_id] = m2_unit
                                next_frontier.append((m2_unit_id, new_score))
                                total_expanded += 1

            frontier = next_frontier

        trace.append(TraceEntry(
            step=2,
            method="graph_expand",
            candidates=total_expanded,
            selected=len(score_board),
            params={
                "max_hops": self.max_hops,
                "decay_factor": self.decay_factor,
                "edge_types": self.expand_edge_types,
            },
        ))

        # Rank all candidates and select top-k. Ties are common here (every
        # 1-hop neighbor of the same seed shares base*decay), so break them
        # by unit id — otherwise truncation picks hash-order-dependent,
        # cross-process-nondeterministic winners.
        ranked = sorted(score_board.items(), key=lambda x: (-x[1], x[0]))
        ranked = ranked[:top_k]

        scored = [
            ScoredUnit(
                unit=unit_map[uid],
                score=score,
                method="graph" if uid not in {su.unit.id for su in seed_scored} else "graph_seed",
            )
            for uid, score in ranked
        ]


        pack = self._make_pack(ctx, scored, trace)
        pack.used_edges = used_edges
        return pack
