"""
GraphStore — Three-layer graph storage backend for MemoryUnit persistence.

Organizes memories into a tri-layer directed graph:

  Layer 1 — QueryNode   (q:{task_id})    Represents a task / question
  Layer 2 — ContentNode (m:{unit_id})    Wraps a MemoryUnit
  Layer 3 — EntityNode  (e:{type}:{name}) Anchors / entities extracted from content

Edge types:
  Query  -> Content:  HAS_MEMORY
  Content -> Content: SIMILAR / DEPENDS / CONFLICTS / SUPERSEDES / COOCCURS / REINFORCES
  Content -> Entity:  HAS_ENTITY

Persistence: JSON (graph structure + node/edge attrs) + .npz (embeddings).
Atomic save via write-to-temp + rename.

Inherits BaseMemoryStorage so it is a drop-in replacement for JsonStorage et al.,
while exposing additional graph-specific query methods for the retrieval layer.

Config:
    storage_dir: str       — Directory for persistence files (default ./storage/graph)
    embedding_dim: int     — Embedding vector dimension (default 384)
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from ..memory_schema import MemoryUnit, MemoryUnitType, RelationType
from .base_storage import BaseMemoryStorage, StorageHealthReport

logger = logging.getLogger(__name__)

# ======================================================================
# Node type constants
# ======================================================================
LAYER_QUERY = "query"
LAYER_CONTENT = "content"
LAYER_ENTITY = "entity"

# Edge type constants
EDGE_HAS_MEMORY = "HAS_MEMORY"
EDGE_HAS_ENTITY = "HAS_ENTITY"

# Map RelationType enum values to edge type strings used in the graph
_RELATION_EDGE_TYPES: Dict[str, str] = {
    rt.value: rt.value.upper() for rt in RelationType
}

# ======================================================================
# Lightweight node schemas (used only inside GraphStore)
# ======================================================================

@dataclass
class QueryNode:
    """Represents a task / question in the query layer."""
    node_id: str
    query_text: str
    task_outcome: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    query_embedding: Optional[np.ndarray] = None

    def to_attrs(self) -> Dict[str, Any]:
        return {
            "layer": LAYER_QUERY,
            "query_text": self.query_text,
            "task_outcome": self.task_outcome,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_attrs(node_id: str, attrs: Dict[str, Any]) -> "QueryNode":
        return QueryNode(
            node_id=node_id,
            query_text=attrs.get("query_text", ""),
            task_outcome=attrs.get("task_outcome", ""),
            created_at=attrs.get("created_at", ""),
        )


@dataclass
class EntityNode:
    """Represents an entity / anchor in the entity layer."""
    node_id: str
    display_name: str
    entity_type: str
    normalized_name: str
    aliases: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_attrs(self) -> Dict[str, Any]:
        return {
            "layer": LAYER_ENTITY,
            "display_name": self.display_name,
            "entity_type": self.entity_type,
            "normalized_name": self.normalized_name,
            "aliases": self.aliases,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_attrs(node_id: str, attrs: Dict[str, Any]) -> "EntityNode":
        return EntityNode(
            node_id=node_id,
            display_name=attrs.get("display_name", ""),
            entity_type=attrs.get("entity_type", ""),
            normalized_name=attrs.get("normalized_name", ""),
            aliases=attrs.get("aliases", []),
            created_at=attrs.get("created_at", ""),
        )


# ======================================================================
# Entity name normalizer
# ======================================================================

def normalize_entity_name(name: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for dedup."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_\- ]+", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


# ======================================================================
# Heuristic entity extractor (optional, simple)
# ======================================================================

def extract_entities_from_unit(unit: MemoryUnit) -> List[Dict[str, str]]:
    """
    Heuristic entity extraction from MemoryUnit content.

    Extracts:
      - TIP: topic -> concept, category -> category, task_type_tags -> tag
      - WORKFLOW: tool names from action text -> tool
      - TRAJECTORY: tool names from steps + key_decision concept + tool_strategy pattern
      - SHORTCUT: name -> action, precondition -> concept
      - INSIGHT: root_cause_conclusion -> concept, failure_pattern -> failure_pattern

    Tool name matching covers both:
      - GAIA-style snake_case tool names (web_search, crawl_page, final_answer, ...)
      - Legacy CamelCase patterns (XxxTool, xxx_tool)

    Returns list of dicts with keys: name, type, normalized_name.
    """
    # Known GAIA agent tool names for reliable snake_case matching
    _KNOWN_TOOLS: Set[str] = {
        "web_search", "crawl_page", "final_answer", "code_execution",
        "file_inspection", "calculator", "image_analysis", "python_interpreter",
        "browser", "search", "visit_webpage",
    }

    entities: List[Dict[str, str]] = []
    c = unit.content
    seen: Set[str] = set()

    def _add(name: str, etype: str) -> None:
        normed = normalize_entity_name(name)
        if normed and normed not in seen:
            seen.add(normed)
            entities.append({
                "name": name.strip(),
                "type": etype,
                "normalized_name": normed,
            })

    def _extract_tools_from_action(action: str) -> None:
        """Extract tool names from an action string."""
        # Match snake_case tool at start: web_search(args) or crawl_page(args)
        m = re.match(r"([a-z][a-z_]{2,})\s*\(", action)
        if m:
            tok = m.group(1)
            if tok in _KNOWN_TOOLS or tok.endswith("_tool") or tok.endswith("_search"):
                _add(tok, "tool")
        # Legacy CamelCase tool patterns
        for tok in re.findall(r"[A-Z][a-zA-Z]+Tool|[a-z_]+_tool", action):
            _add(tok, "tool")

    if unit.type == MemoryUnitType.TIP:
        if c.get("topic"):
            _add(c["topic"], "concept")
        if c.get("category"):
            _add(c["category"], "category")
        for tag in c.get("task_type_tags", []):
            _add(tag, "tag")

    elif unit.type == MemoryUnitType.WORKFLOW:
        for wf_key in ("agent_workflow", "search_workflow"):
            for step in c.get(wf_key, []):
                action = step.get("action", step.get("query_formulation", ""))
                if action:
                    _extract_tools_from_action(action)

    elif unit.type == MemoryUnitType.TRAJECTORY:
        # Extract tool names from compressed steps
        for step in c.get("steps", []):
            action = step.get("action", "")
            if action:
                _extract_tools_from_action(action)
        # Extract key_decision as a transferable concept entity
        key_decision = c.get("key_decision", "")
        if key_decision and len(key_decision) > 10:
            _add(key_decision[:80], "concept")
        # Extract tool_strategy as a tool-pattern entity
        tool_strategy = c.get("tool_strategy", "")
        if tool_strategy and len(tool_strategy) > 5:
            _add(tool_strategy[:60], "tool_pattern")

    elif unit.type == MemoryUnitType.SHORTCUT:
        if c.get("name"):
            _add(c["name"], "action")
        if c.get("precondition"):
            _add(c["precondition"][:60], "concept")

    elif unit.type == MemoryUnitType.INSIGHT:
        if c.get("root_cause_conclusion"):
            _add(c["root_cause_conclusion"][:80], "concept")
        if c.get("failure_pattern"):
            _add(c["failure_pattern"], "failure_pattern")
        for tag in c.get("task_type_tags", []):
            _add(tag, "tag")

    return entities


# ======================================================================
# GraphStore
# ======================================================================

class GraphStore(BaseMemoryStorage):
    """
    Three-layer graph storage for MemoryUnits.

    Organises memories as a directed multigraph with three node layers
    (Query, Content, Entity) and typed edges between them.

    Public API (beyond BaseMemoryStorage):
        upsert_memory_unit(unit, entities?)  -> str  (content node_id)
        upsert_entities(unit_id, entities)   -> None
        get_memory_unit(unit_id)             -> Optional[MemoryUnit]
        get_units_by_query(task_id)          -> List[MemoryUnit]
        get_units_by_entity(type, name)      -> List[MemoryUnit]
        neighbors(node_id, edge_type?, dir?) -> List[str]
        get_subgraph(node_ids)               -> nx.MultiDiGraph
        stats()                              -> dict
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.storage_dir: str = self.config.get(
            "storage_dir", "./storage/graph"
        )
        self.embedding_dim: int = self.config.get("embedding_dim", 384)

        self._graph_path = os.path.join(self.storage_dir, "graph.json")
        self._emb_path = os.path.join(self.storage_dir, "embeddings.npz")

        # Internal state
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._units: Dict[str, MemoryUnit] = {}       # unit.id -> MemoryUnit
        self._sig_to_id: Dict[str, str] = {}           # signature -> unit.id
        # P3 — lazy query-side cache: unit_id -> (query_hash, embedding)
        self._query_emb_cache: Dict[str, Tuple[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Node ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_nid(unit_id: str) -> str:
        return f"m:{unit_id}"

    @staticmethod
    def _query_nid(task_id: str) -> str:
        return f"q:{task_id}"

    @staticmethod
    def _entity_nid(entity_type: str, normalized_name: str) -> str:
        return f"e:{entity_type}:{normalized_name}"

    @staticmethod
    def _parse_nid(nid: str) -> Tuple[str, str]:
        """Return (layer, remainder) from a node id like 'm:xxxx'."""
        prefix, _, rest = nid.partition(":")
        layer_map = {"m": LAYER_CONTENT, "q": LAYER_QUERY, "e": LAYER_ENTITY}
        return layer_map.get(prefix, prefix), rest

    # ------------------------------------------------------------------
    # BaseMemoryStorage: lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            if os.path.exists(self._graph_path):
                self._load()
            logger.info(
                f"GraphStore initialized: {self._graph.number_of_nodes()} nodes, "
                f"{self._graph.number_of_edges()} edges, "
                f"{len(self._units)} memory units"
            )
            return True
        except Exception as e:
            logger.error(f"GraphStore initialization failed: {e}")
            return False

    def save(self) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
        self._save_atomic()

    # ------------------------------------------------------------------
    # BaseMemoryStorage: write operations
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> int:
        """Add MemoryUnits via upsert_memory_unit (with heuristic entities)."""
        added = 0
        for unit in units:
            if unit.signature and unit.signature in self._sig_to_id:
                # Dedup: merge stats into existing unit
                self._merge_duplicate(unit)
                continue
            entities = extract_entities_from_unit(unit)
            self.upsert_memory_unit(unit, entities=entities, _skip_save=True)
            self._tag_index_add(unit)              # P2
            added += 1
        if added > 0:
            self.save()
        return added

    def update(self, unit: MemoryUnit) -> bool:
        if unit.id not in self._units:
            return False
        nid = self._content_nid(unit.id)
        old_unit = self._units[unit.id]
        # P1 — invalidate stale embedding if content changed
        self._maybe_invalidate_stale_embedding(old_unit, unit)
        # P2 — keep tag inverted index in sync
        self._tag_index_update(old_unit, unit)
        # P3 — drop cached query embedding if source_task_query changed
        if hasattr(self, "_query_emb_cache"):
            if (old_unit.source_task_query or "") != (unit.source_task_query or ""):
                self._query_emb_cache.pop(unit.id, None)

        old_sig = old_unit.signature
        self._units[unit.id] = unit
        # Update signature index
        if old_sig and old_sig in self._sig_to_id:
            del self._sig_to_id[old_sig]
        if unit.signature:
            self._sig_to_id[unit.signature] = unit.id
        # Update graph node attrs
        self._graph.nodes[nid].update(self._content_node_attrs(unit))
        # Rebuild content->content edges from unit.relations
        self._sync_relation_edges(unit)
        self.save()
        return True

    def delete(self, unit_id: str) -> bool:
        if unit_id not in self._units:
            return False
        nid = self._content_nid(unit_id)
        unit = self._units.pop(unit_id)
        if unit.signature:
            self._sig_to_id.pop(unit.signature, None)
        self._tag_index_remove(unit_id)              # P2
        if hasattr(self, "_query_emb_cache"):
            self._query_emb_cache.pop(unit_id, None)  # P3
        if self._graph.has_node(nid):
            self._graph.remove_node(nid)  # removes all incident edges
        self.save()
        return True

    # ------------------------------------------------------------------
    # BaseMemoryStorage: read operations
    # ------------------------------------------------------------------

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        return self._units.get(unit_id)

    def get_all(
        self,
        active_only: bool = False,
        unit_type: Optional[MemoryUnitType] = None,
    ) -> List[MemoryUnit]:
        result = list(self._units.values())
        if active_only:
            result = [u for u in result if u.is_active]
        if unit_type is not None:
            result = [u for u in result if u.type == unit_type]
        return result

    def export_relation_edges(self) -> List[Dict[str, Any]]:
        """Return persisted content-to-content edge state using unit IDs.

        ``MemoryUnit.relations`` preserves the structural relation, but adaptive
        management statistics such as ``base_weight``, ``feedback_multiplier``,
        ``usage_count`` and ``success_count`` live only on graph edges.  The
        search coordinator uses this transport when handing the winning graph
        architecture to the next round.
        """

        exported: List[Dict[str, Any]] = []
        for source, target, data in self._graph.edges(data=True):
            source_layer, source_id = self._parse_nid(str(source))
            target_layer, target_id = self._parse_nid(str(target))
            edge_type = data.get("edge_type")
            if (
                source_layer != LAYER_CONTENT
                or target_layer != LAYER_CONTENT
                or not isinstance(edge_type, str)
                or not edge_type
            ):
                continue
            exported.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_type": edge_type,
                    "attributes": dict(data),
                }
            )
        return sorted(
            exported,
            key=lambda edge: (
                edge["source_id"], edge["target_id"], edge["edge_type"]
            ),
        )

    def import_relation_edges(self, edges: List[Dict[str, Any]]) -> int:
        """Upsert canonical content-edge state after units have been imported."""

        changed = 0
        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            source_id = str(edge.get("source_id") or "")
            target_id = str(edge.get("target_id") or "")
            edge_type = str(edge.get("edge_type") or "")
            if not source_id or not target_id or not edge_type:
                continue
            source = self._content_nid(source_id)
            target = self._content_nid(target_id)
            if not self._graph.has_node(source) or not self._graph.has_node(target):
                continue

            attributes = dict(edge.get("attributes") or {})
            attributes["edge_type"] = edge_type
            matching_key = None
            if self._graph.has_edge(source, target):
                for key, existing in self._graph[source][target].items():
                    if existing.get("edge_type") == edge_type:
                        matching_key = key
                        break
            if matching_key is None:
                self._graph.add_edge(
                    source, target, key=edge_type, **attributes
                )
                changed += 1
            else:
                existing = self._graph[source][target][matching_key]
                if any(existing.get(k) != v for k, v in attributes.items()):
                    existing.update(attributes)
                    changed += 1

        if changed:
            self.save()
        return changed

    def count(self) -> int:
        return len(self._units)

    # ------------------------------------------------------------------
    # BaseMemoryStorage: dedup
    # ------------------------------------------------------------------

    def exists_signature(self, signature: str) -> bool:
        return signature in self._sig_to_id

    # ------------------------------------------------------------------
    # BaseMemoryStorage: embedding access
    # ------------------------------------------------------------------

    def get_embedding_index(
        self,
        active_only: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[MemoryUnit]]:
        if active_only:
            units = [u for u in self._units.values()
                     if u.is_active and u.embedding is not None]
        else:
            units = [u for u in self._units.values()
                     if u.embedding is not None]
        if not units:
            return None, []
        emb_matrix = np.vstack([u.embedding for u in units])
        return emb_matrix, units

    # P3 query-side embedding index — inherited from BaseMemoryStorage.

    # ==================================================================
    # Graph-specific public API
    # ==================================================================

    def upsert_memory_unit(
        self,
        unit: MemoryUnit,
        entities: Optional[List[Dict[str, str]]] = None,
        _skip_save: bool = False,
    ) -> str:
        """
        Insert or merge a MemoryUnit into the graph.

        Creates:
          - ContentNode for the unit
          - QueryNode for source_task_id (if non-empty)
          - Query -> Content edge (HAS_MEMORY)
          - Content -> Content edges from unit.relations
          - Content -> Entity edges if entities provided

        When the unit's signature already exists and the existing unit is active,
        statistics are merged (usage_count, success_count, access_count accumulated;
        last_accessed updated; relations merged with dedup).

        When the existing unit has is_active=False (soft-deleted), it is reactivated
        with the new unit's content and statistics.

        Args:
            unit: MemoryUnit to upsert.
            entities: Optional entity dicts with keys name, type, normalized_name.

        Returns:
            The content node_id ("m:{unit.id}").
        """
        # --- Dedup / merge ---
        if unit.signature and unit.signature in self._sig_to_id:
            existing_id = self._sig_to_id[unit.signature]
            existing = self._units.get(existing_id)
            if existing is not None:
                if existing.is_active:
                    self._merge_into(existing, unit)
                    self._graph.nodes[self._content_nid(existing_id)].update(
                        self._content_node_attrs(existing)
                    )
                    self._ensure_query_edge(existing)
                    if entities:
                        self.upsert_entities(existing_id, entities)
                    if not _skip_save:
                        self.save()
                    return self._content_nid(existing_id)
                else:
                    # Reactivate soft-deleted unit
                    existing.is_active = True
                    existing.content = unit.content
                    existing.confidence = unit.confidence
                    existing.relations = unit.relations
                    self._graph.nodes[self._content_nid(existing_id)].update(
                        self._content_node_attrs(existing)
                    )
                    self._sync_relation_edges(existing)
                    self._ensure_query_edge(existing)
                    if entities:
                        self.upsert_entities(existing_id, entities)
                    if not _skip_save:
                        self.save()
                    return self._content_nid(existing_id)

        # --- New unit ---
        self._units[unit.id] = unit
        if unit.signature:
            self._sig_to_id[unit.signature] = unit.id

        # Content node
        nid = self._content_nid(unit.id)
        self._graph.add_node(nid, **self._content_node_attrs(unit))

        # Query node + edge
        self._ensure_query_edge(unit)

        # Content -> Content relation edges
        self._sync_relation_edges(unit)

        # Entities
        if entities:
            self.upsert_entities(unit.id, entities)

        if not _skip_save:
            self.save()
        return nid

    def upsert_entities(
        self,
        unit_id: str,
        entities: List[Dict[str, str]],
    ) -> None:
        """
        Add entity nodes and Content->Entity edges for a given MemoryUnit.

        Args:
            unit_id: The MemoryUnit id (not the graph node id).
            entities: List of dicts, each with at least 'name' and 'type'.
                      'normalized_name' is auto-computed if missing.
        """
        content_nid = self._content_nid(unit_id)
        if not self._graph.has_node(content_nid):
            logger.warning(f"Content node {content_nid} not in graph, skipping entities")
            return

        for ent in entities:
            name = ent.get("name", "")
            etype = ent.get("type", "unknown")
            normed = ent.get("normalized_name") or normalize_entity_name(name)
            if not normed:
                continue

            entity_nid = self._entity_nid(etype, normed)
            if not self._graph.has_node(entity_nid):
                enode = EntityNode(
                    node_id=entity_nid,
                    display_name=name,
                    entity_type=etype,
                    normalized_name=normed,
                    aliases=ent.get("aliases", []),
                )
                self._graph.add_node(entity_nid, **enode.to_attrs())
            else:
                # Merge aliases
                existing_aliases = self._graph.nodes[entity_nid].get("aliases", [])
                for alias in ent.get("aliases", []):
                    if alias not in existing_aliases:
                        existing_aliases.append(alias)
                self._graph.nodes[entity_nid]["aliases"] = existing_aliases

            # Add Content -> Entity edge (skip if already present)
            if not self._has_edge(content_nid, entity_nid, EDGE_HAS_ENTITY):
                self._graph.add_edge(
                    content_nid, entity_nid,
                    edge_type=EDGE_HAS_ENTITY,
                    entity_type=etype,
                    created_at=datetime.now().isoformat(),
                )

    def get_memory_unit(self, unit_id: str) -> Optional[MemoryUnit]:
        """Alias for get(), accepts bare unit_id (not node_id)."""
        return self._units.get(unit_id)

    def get_units_by_query(self, task_id: str) -> List[MemoryUnit]:
        """Return all MemoryUnits produced by a given task."""
        qnid = self._query_nid(task_id)
        if not self._graph.has_node(qnid):
            return []
        result = []
        for _, target, data in self._graph.out_edges(qnid, data=True):
            if data.get("edge_type") == EDGE_HAS_MEMORY:
                layer, uid = self._parse_nid(target)
                if layer == LAYER_CONTENT and uid in self._units:
                    result.append(self._units[uid])
        return result

    def get_units_by_entity(
        self,
        entity_type: str,
        normalized_name: str,
    ) -> List[MemoryUnit]:
        """Return all MemoryUnits linked to a specific entity."""
        entity_nid = self._entity_nid(entity_type, normalized_name)
        if not self._graph.has_node(entity_nid):
            return []
        result = []
        # Entity nodes receive HAS_ENTITY edges from content nodes
        for source, _, data in self._graph.in_edges(entity_nid, data=True):
            if data.get("edge_type") == EDGE_HAS_ENTITY:
                layer, uid = self._parse_nid(source)
                if layer == LAYER_CONTENT and uid in self._units:
                    result.append(self._units[uid])
        return result

    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "out",
    ) -> List[str]:
        """
        Return neighbor node_ids of a given node, optionally filtered by edge_type.

        Args:
            node_id: Graph node id (e.g. "m:xxxx", "q:xxxx", "e:tool:web_search").
            edge_type: If set, only follow edges of this type.
            direction: "out" (successors), "in" (predecessors), or "both".

        Returns:
            List of neighbor node_ids.
        """
        if not self._graph.has_node(node_id):
            return []

        result_set: Set[str] = set()

        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    result_set.add(target)

        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    result_set.add(source)

        # Sorted for cross-process determinism: list(set) order follows the
        # string-hash seed, which made graph expansion pick different
        # neighbors under different PYTHONHASHSEEDs.
        return sorted(result_set)

    def neighbors_with_weights(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[Tuple[str, float]]:
        """Like neighbors(), but returns (neighbor_id, edge_weight) pairs.

        Relation edges persist their weight (rel.weight) on the edge data;
        HAS_ENTITY and other structural edges carry no weight and default to
        1.0. For parallel edges (MultiDiGraph) the MAX matching weight wins.
        Sorted by neighbor id for determinism.
        """
        if not self._graph.has_node(node_id):
            return []

        best: Dict[str, float] = {}

        def _collect(edges, other_index: int) -> None:
            for edge in edges:
                data = edge[2]
                if edge_type is not None and data.get("edge_type") != edge_type:
                    continue
                other = edge[other_index]
                w = data.get("weight", 1.0)
                try:
                    w = float(w) if w is not None else 1.0
                except (TypeError, ValueError):
                    w = 1.0
                if other not in best or w > best[other]:
                    best[other] = w

        if direction in ("out", "both"):
            _collect(self._graph.out_edges(node_id, data=True), 1)
        if direction in ("in", "both"):
            _collect(self._graph.in_edges(node_id, data=True), 0)

        return sorted(best.items())

    def get_subgraph(self, node_ids: List[str]) -> nx.MultiDiGraph:
        """Return an induced subgraph containing the specified nodes."""
        valid = [n for n in node_ids if self._graph.has_node(n)]
        return self._graph.subgraph(valid).copy()

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the graph."""
        nodes_by_layer: Dict[str, int] = {
            LAYER_QUERY: 0, LAYER_CONTENT: 0, LAYER_ENTITY: 0
        }
        for _, data in self._graph.nodes(data=True):
            layer = data.get("layer", "unknown")
            nodes_by_layer[layer] = nodes_by_layer.get(layer, 0) + 1

        edges_by_type: Dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            etype = data.get("edge_type", "unknown")
            edges_by_type[etype] = edges_by_type.get(etype, 0) + 1

        units_by_type: Dict[str, int] = {}
        for u in self._units.values():
            t = u.type.value
            units_by_type[t] = units_by_type.get(t, 0) + 1

        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "nodes_by_layer": nodes_by_layer,
            "edges_by_type": edges_by_type,
            "memory_units": len(self._units),
            "units_by_type": units_by_type,
        }

    # ==================================================================
    # Health report
    # ==================================================================

    # Graph traversal is ineffective below this unit count: the graph is
    # too sparse for BFS/DFS to find meaningful multi-hop paths.
    # Empirically validated: <30 units → ~100% empty retrieval via graph.
    _COLD_START_THRESHOLD = 30

    def get_health_report(self) -> StorageHealthReport:
        total = len(self._units)
        active = sum(1 for u in self._units.values() if u.is_active)
        is_cold = total < self._COLD_START_THRESHOLD

        # Count content-to-content semantic edges (exclude HAS_MEMORY / HAS_ENTITY)
        _structural_edges = {EDGE_HAS_MEMORY, EDGE_HAS_ENTITY}
        content_edges = sum(
            1 for _, _, d in self._graph.edges(data=True)
            if d.get("edge_type") not in _structural_edges
        )
        # Average out-degree of content nodes (indicates graph connectivity)
        content_nids = [
            n for n, d in self._graph.nodes(data=True)
            if d.get("layer") == LAYER_CONTENT
        ]
        avg_degree = (
            sum(self._graph.out_degree(n) for n in content_nids) / len(content_nids)
            if content_nids else 0.0
        )

        if is_cold:
            mode = "graph_sparse"
            quality = "none" if total == 0 else "low"
        elif avg_degree < 1.0:
            mode = "graph_sparse"
            quality = "low"
        elif avg_degree < 2.0:
            mode = "standard"
            quality = "medium"
        else:
            mode = "standard"
            quality = "high"

        return StorageHealthReport(
            backend_type="graph",
            unit_count=total,
            active_unit_count=active,
            is_cold_start=is_cold,
            cold_start_threshold=self._COLD_START_THRESHOLD,
            retrieval_mode=mode,
            estimated_recall_quality=quality,
            graph_edge_count=content_edges,
            graph_avg_degree=avg_degree,
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _content_node_attrs(self, unit: MemoryUnit) -> Dict[str, Any]:
        """Build graph node attribute dict for a ContentNode."""
        return {
            "layer": LAYER_CONTENT,
            "unit_type": unit.type.value,
            "signature": unit.signature,
            "source_task_id": unit.source_task_id,
            "task_outcome": unit.task_outcome,
            "confidence": unit.confidence,
            "usage_count": unit.usage_count,
            "success_count": unit.success_count,
            "is_active": unit.is_active,
            "created_at": unit.created_at,
            "decay_weight": unit.decay_weight,
        }

    def _ensure_query_edge(self, unit: MemoryUnit) -> None:
        """Create QueryNode and HAS_MEMORY edge if source_task_id is present."""
        if not unit.source_task_id:
            return
        qnid = self._query_nid(unit.source_task_id)
        cnid = self._content_nid(unit.id)

        if not self._graph.has_node(qnid):
            qnode = QueryNode(
                node_id=qnid,
                query_text=unit.source_task_query,
                task_outcome=unit.task_outcome,
                created_at=unit.created_at,
            )
            self._graph.add_node(qnid, **qnode.to_attrs())

        if not self._has_edge(qnid, cnid, EDGE_HAS_MEMORY):
            self._graph.add_edge(
                qnid, cnid,
                edge_type=EDGE_HAS_MEMORY,
                memory_type=unit.type.value,
                signature=unit.signature,
                created_at=datetime.now().isoformat(),
            )

    def _sync_relation_edges(self, unit: MemoryUnit) -> None:
        """
        Synchronize Content->Content edges from unit.relations.

        Removes old relation edges from this content node, then re-adds
        current relations. Targets that don't exist in the graph yet
        are silently skipped (they will be linked when the target is added).
        """
        cnid = self._content_nid(unit.id)

        # Remove existing relation edges from this node
        edges_to_remove = []
        for _, target, key, data in self._graph.out_edges(cnid, data=True, keys=True):
            if data.get("edge_type") not in (EDGE_HAS_MEMORY, EDGE_HAS_ENTITY):
                edges_to_remove.append((cnid, target, key))
        for u, v, k in edges_to_remove:
            self._graph.remove_edge(u, v, key=k)

        # Add current relations
        for rel in unit.relations:
            target_nid = self._content_nid(rel.target_id)
            edge_type = _RELATION_EDGE_TYPES.get(
                rel.relation_type.value, rel.relation_type.value.upper()
            )
            # Only add if target exists in graph
            if self._graph.has_node(target_nid):
                self._graph.add_edge(
                    cnid, target_nid,
                    edge_type=edge_type,
                    weight=rel.weight,
                    created_at=datetime.now().isoformat(),
                )

    def _has_edge(self, source: str, target: str, edge_type: str) -> bool:
        """Check if an edge with the given type exists between source and target."""
        if not self._graph.has_node(source) or not self._graph.has_node(target):
            return False
        for _, _, data in self._graph.edges(source, data=True):
            if _ == target and data.get("edge_type") == edge_type:
                return True
        # More reliable check using edge data
        if self._graph.has_edge(source, target):
            for key in self._graph[source][target]:
                if self._graph[source][target][key].get("edge_type") == edge_type:
                    return True
        return False

    def _merge_into(self, existing: MemoryUnit, incoming: MemoryUnit) -> None:
        """Merge incoming unit stats into an existing unit (dedup merge)."""
        existing.usage_count += incoming.usage_count
        existing.success_count += incoming.success_count
        existing.access_count += incoming.access_count
        existing.last_accessed = datetime.now().isoformat()
        # Merge relations (dedup by target_id + relation_type)
        existing_rels = {
            (r.target_id, r.relation_type.value) for r in existing.relations
        }
        for rel in incoming.relations:
            key = (rel.target_id, rel.relation_type.value)
            if key not in existing_rels:
                existing.relations.append(rel)
                existing_rels.add(key)
        # Update confidence to max
        existing.confidence = max(existing.confidence, incoming.confidence)

    def _merge_duplicate(self, unit: MemoryUnit) -> None:
        """Handle add() dedup: merge stats and ensure query edge."""
        existing_id = self._sig_to_id.get(unit.signature)
        if existing_id is None:
            return
        existing = self._units.get(existing_id)
        if existing is None:
            return
        self._merge_into(existing, unit)
        nid = self._content_nid(existing_id)
        if self._graph.has_node(nid):
            self._graph.nodes[nid].update(self._content_node_attrs(existing))
        self._ensure_query_edge(existing)

    # ==================================================================
    # Persistence
    # ==================================================================

    def _save_atomic(self) -> None:
        """
        Persist graph (JSON) and embeddings (npz) with atomic writes.

        Strategy: write to temp file in same dir, then os.replace() (atomic on POSIX).
        """
        # --- Build JSON payload ---
        nodes = {}
        for nid, data in self._graph.nodes(data=True):
            nodes[nid] = dict(data)

        edges = []
        for src, tgt, data in self._graph.edges(data=True):
            edges.append({"source": src, "target": tgt, **dict(data)})

        units_data = {uid: u.to_dict() for uid, u in self._units.items()}

        payload = {
            "nodes": nodes,
            "edges": edges,
            "units": units_data,
            "sig_to_id": self._sig_to_id,
        }

        # Remove embeddings from units payload (stored separately in npz)
        for uid, udict in payload["units"].items():
            udict.pop("embedding", None)

        # Atomic JSON write
        fd, tmp_json = tempfile.mkstemp(
            dir=self.storage_dir, suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_json, self._graph_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_json):
                os.unlink(tmp_json)
            raise

        # --- Save embeddings (npz) ---
        emb_dict = {}
        for uid, unit in self._units.items():
            if unit.embedding is not None:
                emb_dict[uid] = unit.embedding.astype(np.float32)
        # Also save query embeddings if any
        for nid, data in self._graph.nodes(data=True):
            if data.get("layer") == LAYER_QUERY:
                qemb = data.get("_query_embedding")
                if qemb is not None:
                    emb_dict[f"_qemb_{nid}"] = np.asarray(qemb, dtype=np.float32)

        # np.savez_compressed auto-appends ".npz" if missing, so use a
        # temp path that already ends with ".npz" to avoid the mismatch.
        fd2, tmp_npz = tempfile.mkstemp(
            dir=self.storage_dir, suffix=".tmp.npz"
        )
        os.close(fd2)
        try:
            if emb_dict:
                np.savez_compressed(tmp_npz, **emb_dict)
            else:
                np.savez_compressed(tmp_npz)  # empty file
            os.replace(tmp_npz, self._emb_path)
        except Exception:
            if os.path.exists(tmp_npz):
                os.unlink(tmp_npz)
            raise

    def _load(self) -> None:
        """Load graph structure from JSON and embeddings from npz."""
        # --- Load JSON ---
        with open(self._graph_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # --- Load embeddings ---
        emb_dict = {}
        if os.path.exists(self._emb_path):
            with np.load(self._emb_path, allow_pickle=False) as npz:
                for key in npz.files:
                    emb_dict[key] = npz[key]

        # --- Rebuild units ---
        self._units = {}
        self._sig_to_id = {}
        self._init_tag_index()                     # P2
        self._tag_index = {}
        for uid, udict in payload.get("units", {}).items():
            # Restore embedding
            if uid in emb_dict:
                udict["embedding"] = emb_dict[uid].tolist()
            unit = MemoryUnit.from_dict(udict)
            self._units[uid] = unit
            if unit.signature:
                self._sig_to_id[unit.signature] = uid
            self._tag_index_add(unit)              # P2

        # Also restore sig_to_id from payload (covers edge cases)
        for sig, uid in payload.get("sig_to_id", {}).items():
            if sig not in self._sig_to_id and uid in self._units:
                self._sig_to_id[sig] = uid

        # --- Rebuild graph ---
        self._graph = nx.MultiDiGraph()
        for nid, attrs in payload.get("nodes", {}).items():
            self._graph.add_node(nid, **attrs)

        for edge in payload.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            self._graph.add_edge(src, tgt, **edge)
