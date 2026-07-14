"""
LLMGraphStore — Graph storage with LLM-based entity and fact extraction.

Extends GraphStore by replacing heuristic entity extraction with a
multi-stage LLM pipeline inspired by Graphiti/Zep:

  Stage 1: LLM Entity Extraction   (from MemoryUnit content)
  Stage 2: Entity Dedup             (embedding pre-filter + LLM judge)
  Stage 3: LLM Fact Extraction      (relations between entities)
  Stage 4: Fact Dedup               (heuristic: same source+target+type)

EntityNode is upgraded with: summary, name_embedding, labels, attributes.

The model is an explicit dependency. Initialization and extraction fail closed
instead of silently changing this backend into heuristic ``graph`` storage.

Config:
    storage_dir: str          — Directory for persistence (default ./storage/llm_graph)
    embedding_dim: int        — Embedding dimension (default 384)
    model: Any                — Callable model reused from the provider
    llm_client: Any           — Explicit OpenAI-compatible client (compatibility path)
    llm_model: str            — Required model id for OpenAI-compatible clients
    dedup_sim_threshold: float — Embedding similarity threshold for dedup candidates (default 0.7)
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..memory_schema import MemoryUnit
from ..model_io import invoke_text_model
from .graph_storage import (
    EDGE_HAS_ENTITY,
    EntityNode,
    GraphStore,
    normalize_entity_name,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Prompt templates for LLM extraction
# ======================================================================

ENTITY_EXTRACTION_SYSTEM = """You are a knowledge extraction assistant. Extract structured entities from agent memory units. Output valid JSON only."""

ENTITY_EXTRACTION_USER = """Given the following memory unit from an agent task:

Memory Type: {unit_type}
Task Query: {source_task_query}
Content: {content_json}
Task Outcome: {task_outcome}

Extract entity nodes that are explicitly or implicitly mentioned.

Guidelines:
1. Extract significant entities: tools, websites, data sources, APIs, file formats, domains, concepts, methods, error types, platforms.
2. DO NOT create entities for relationships or actions (these will be extracted as edges later).
3. DO NOT extract task-specific values such as specific numbers, dates, or proper nouns from answers.
4. Use full, explicit names. Provide aliases if applicable.
5. Assign one or more labels from: [Tool, Website, DataSource, Concept, Domain, Method, ErrorType, FileFormat, API, Platform]

Output JSON:
{{"entities": [{{"name": "entity name", "labels": ["Label1"], "aliases": ["alt name"], "summary": "one-line description of this entity in context"}}]}}

If no meaningful entities can be extracted, return: {{"entities": []}}"""

ENTITY_DEDUP_SYSTEM = """You are a deduplication assistant. Determine if two entities refer to the same real-world concept. Output valid JSON only."""

ENTITY_DEDUP_USER = """Existing Entity:
  Name: {existing_name}
  Labels: {existing_labels}
  Summary: {existing_summary}

New Entity:
  Name: {new_name}
  Labels: {new_labels}
  Summary: {new_summary}

Are these the same entity? Consider that duplicate entities may have different surface names (e.g., "ClinicalTrials.gov" and "CT.gov", or "web_search" and "WebSearchTool").

Output JSON:
{{"is_duplicate": true/false, "best_name": "the most complete full name"}}"""

FACT_EXTRACTION_SYSTEM = """You are a relationship extraction assistant. Extract factual relationships between entities from agent memory units. Output valid JSON only."""

FACT_EXTRACTION_USER = """Memory Unit:
Type: {unit_type}
Content: {content_json}
Task: {source_task_query}

Entities already extracted:
{entity_list_text}

Extract factual relationships between the listed entities.

Guidelines:
1. Each fact should connect two DISTINCT entities from the list above.
2. The relation_type should be concise and in ALL_CAPS (e.g., USED_FOR, REQUIRES, RETURNS_FORMAT, HOSTED_ON, CAUSES_ERROR, ALTERNATIVE_TO, PART_OF, DEPENDS_ON, EXTRACTS_FROM, VALIDATES_WITH).
3. Provide a detailed fact description containing all relevant context.
4. Only extract facts that are clearly supported by the content.

Output JSON:
{{"facts": [{{"source_entity": "name of source", "target_entity": "name of target", "relation_type": "RELATION_TYPE", "fact": "detailed description of this relationship"}}]}}

If no meaningful facts can be extracted, return: {{"facts": []}}"""


# ======================================================================
# LLM client helper
# ======================================================================

def _get_llm_client(config: Dict) -> Optional[Any]:
    """Create a client only from explicit configuration, never environment."""
    try:
        from openai import OpenAI
        import httpx

        api_key = str(config.get("llm_api_key") or "").strip()
        api_base = str(config.get("llm_api_base") or "").strip()
        if not api_key:
            return None
        return OpenAI(
            api_key=api_key,
            base_url=api_base or None,
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=300.0),
        )
    except Exception as e:
        logger.warning(f"Failed to create LLM client: {e}")
        return None


# ======================================================================
# LLMGraphStore
# ======================================================================

class LLMGraphStore(GraphStore):
    """Graph storage with LLM-based entity and fact extraction pipeline."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        cfg = config or {}
        self._maintenance_mode = bool(cfg.get("maintenance_mode", False))
        if cfg.get("use_llm_extraction", True) is not True:
            raise ValueError(
                "llm_graph cannot disable LLM extraction; use graph storage instead"
            )
        self._configured_model = cfg.get("model")
        self._model_resolver = cfg.get("model_resolver")
        self._usage_in_task_metrics = bool(cfg.get("usage_in_task_metrics", False))
        self._usage_in_task_metrics_resolver = cfg.get(
            "usage_in_task_metrics_resolver"
        )
        self._llm_client = self._configured_model or cfg.get("llm_client")
        if self._llm_client is None and self._model_resolver is None:
            self._llm_client = _get_llm_client(cfg)
        if (
            self._llm_client is None
            and self._model_resolver is None
            and not self._maintenance_mode
        ):
            raise ValueError(
                "llm_graph requires config['model'], config['llm_client'], "
                "a model_resolver, or explicit llm_api_key configuration"
            )
        initial_client = self._resolve_optional_client()
        if initial_client is None and not self._maintenance_mode:
            raise ValueError("llm_graph model resolver returned no model at initialization")
        self.llm_model = str(
            cfg.get("llm_model")
            or getattr(self._configured_model, "model_id", "")
            or getattr(initial_client, "model_id", "")
            or ""
        )
        if initial_client is not None and hasattr(initial_client, "chat") and not self.llm_model:
            raise ValueError(
                "llm_graph requires llm_model for an OpenAI-compatible client"
            )
        self.use_llm = True
        self.dedup_sim_threshold = cfg.get("dedup_sim_threshold", 0.7)
        self.llm_usage_stats = {
            "llm_graph_calls": 0,
            "llm_graph_input_tokens": 0,
            "llm_graph_output_tokens": 0,
        }
        self._llm_thread_local = threading.local()

        # Entity name embeddings cache: node_id -> np.ndarray
        self._entity_embeddings: Dict[str, np.ndarray] = {}
        # Entity summaries cache: node_id -> str
        self._entity_summaries: Dict[str, str] = {}
        # Entity labels cache: node_id -> List[str]
        self._entity_labels: Dict[str, List[str]] = {}
        # Fact edge descriptions: (source_nid, target_nid, edge_type) -> str
        self._fact_descriptions: Dict[Tuple[str, str, str], str] = {}

    @property
    def llm_client(self):
        client = self._resolve_optional_client()
        if client is None:
            raise RuntimeError(
                "llm_graph extraction is unavailable in model-free maintenance mode"
            )
        return client

    def _resolve_optional_client(self):
        return (
            self._model_resolver()
            if self._model_resolver is not None
            else self._llm_client
        )

    def _usage_is_in_task_metrics(self, client: Any) -> bool:
        if self._usage_in_task_metrics_resolver is not None:
            return bool(self._usage_in_task_metrics_resolver(client))
        return self._usage_in_task_metrics

    def reset_usage_metrics(self) -> None:
        tl = self._llm_thread_local
        tl.llm_graph_calls = 0
        tl.llm_graph_input_tokens = 0
        tl.llm_graph_output_tokens = 0
        tl.llm_graph_usage_in_task_metrics = False

    def get_usage_metrics(self) -> Dict[str, Any]:
        tl = self._llm_thread_local
        return {
            "llm_graph_calls": int(getattr(tl, "llm_graph_calls", 0) or 0),
            "llm_graph_input_tokens": int(
                getattr(tl, "llm_graph_input_tokens", 0) or 0
            ),
            "llm_graph_output_tokens": int(
                getattr(tl, "llm_graph_output_tokens", 0) or 0
            ),
            "llm_graph_usage_in_task_metrics": bool(
                getattr(tl, "llm_graph_usage_in_task_metrics", False)
            ),
        }

    def _call_llm(self, system: str, user: str) -> Dict[str, Any]:
        client = self.llm_client
        usage_in_task_metrics = self._usage_is_in_task_metrics(client)
        tl = self._llm_thread_local
        prior_calls = int(getattr(tl, "llm_graph_calls", 0) or 0)
        tl.llm_graph_calls = prior_calls + 1
        tl.llm_graph_usage_in_task_metrics = (
            usage_in_task_metrics
            if prior_calls == 0
            else bool(getattr(tl, "llm_graph_usage_in_task_metrics", False))
            and usage_in_task_metrics
        )
        self.llm_usage_stats["llm_graph_calls"] += 1

        result = invoke_text_model(
            client,
            model=self.llm_model or getattr(client, "model_id", ""),
            system=system,
            user=user,
            max_tokens=800,
            temperature=0.1,
        )
        tl.llm_graph_input_tokens = (
            getattr(tl, "llm_graph_input_tokens", 0) + result.input_tokens
        )
        tl.llm_graph_output_tokens = (
            getattr(tl, "llm_graph_output_tokens", 0) + result.output_tokens
        )
        self.llm_usage_stats["llm_graph_input_tokens"] += result.input_tokens
        self.llm_usage_stats["llm_graph_output_tokens"] += result.output_tokens

        text = result.text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("llm_graph model returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("llm_graph model must return a JSON object")
        return payload

    # ------------------------------------------------------------------
    # Override: upsert_memory_unit with LLM pipeline
    # ------------------------------------------------------------------

    def upsert_memory_unit(self, unit: MemoryUnit, entities=None,
                           _skip_save: bool = False) -> str:
        """Upsert with LLM-based entity and fact extraction."""
        fact_edges: List[Dict] = []

        if entities is None:
            entities, fact_edges = self._llm_extraction_pipeline(unit)
            logger.info(
                "LLM extracted %d entities, %d facts for unit %s",
                len(entities),
                len(fact_edges),
                unit.id,
            )

        # Call parent upsert
        nid = super().upsert_memory_unit(unit, entities, _skip_save=True)

        # Add fact edges between entity nodes
        for fact in fact_edges:
            src_nid = fact.get("source_nid")
            tgt_nid = fact.get("target_nid")
            rel_type = fact.get("relation_type", "RELATED")
            description = fact.get("fact", "")
            if src_nid and tgt_nid and self._graph.has_node(src_nid) and self._graph.has_node(tgt_nid):
                if not self._has_edge(src_nid, tgt_nid, rel_type):
                    self._graph.add_edge(
                        src_nid, tgt_nid,
                        edge_type=rel_type,
                        fact=description,
                        created_at=datetime.now().isoformat(),
                    )
                    self._fact_descriptions[(src_nid, tgt_nid, rel_type)] = description

        if not _skip_save:
            self._save_atomic()

        return nid

    # ------------------------------------------------------------------
    # Override: upsert_entities to store extra fields
    # ------------------------------------------------------------------

    def upsert_entities(self, unit_id: str, entities: List[Dict[str, str]]) -> None:
        """Extended upsert that also stores summary, labels, name_embedding."""
        content_nid = self._content_nid(unit_id)
        if not self._graph.has_node(content_nid):
            return

        for ent in entities:
            name = ent.get("name", "").strip()
            if not name:
                continue
            etype = ent.get("type", "concept")
            norm = normalize_entity_name(name)
            entity_nid = self._entity_nid(etype, norm)

            if self._graph.has_node(entity_nid):
                # Merge aliases
                existing_aliases = set(self._graph.nodes[entity_nid].get("aliases", []))
                new_aliases = set(ent.get("aliases", []))
                merged = list(existing_aliases | new_aliases | {name})
                self._graph.nodes[entity_nid]["aliases"] = merged
                # Update summary if provided and current is empty
                if ent.get("summary") and not self._entity_summaries.get(entity_nid):
                    self._entity_summaries[entity_nid] = ent["summary"]
                # Update labels
                if ent.get("labels"):
                    existing_labels = set(self._entity_labels.get(entity_nid, []))
                    self._entity_labels[entity_nid] = list(existing_labels | set(ent["labels"]))
            else:
                attrs = EntityNode(
                    node_id=entity_nid,
                    display_name=name,
                    entity_type=etype,
                    normalized_name=norm,
                    aliases=ent.get("aliases", []),
                ).to_attrs()
                self._graph.add_node(entity_nid, **attrs)

                # Store extended fields
                if ent.get("summary"):
                    self._entity_summaries[entity_nid] = ent["summary"]
                if ent.get("labels"):
                    self._entity_labels[entity_nid] = ent["labels"]

                # Compute entity name embedding
                self._compute_entity_embedding(entity_nid, name)

            # Ensure content -> entity edge
            if not self._has_edge(content_nid, entity_nid, EDGE_HAS_ENTITY):
                self._graph.add_edge(
                    content_nid, entity_nid,
                    edge_type=EDGE_HAS_ENTITY,
                    created_at=datetime.now().isoformat(),
                )

    # ------------------------------------------------------------------
    # LLM extraction pipeline
    # ------------------------------------------------------------------

    def _llm_extraction_pipeline(self, unit: MemoryUnit) -> Tuple[List[Dict], List[Dict]]:
        """
        Full LLM extraction pipeline:
        Stage 1: Extract entities
        Stage 2: Dedup entities against existing
        Stage 3: Extract facts between entities
        Stage 4: Dedup facts
        Returns: (entities, fact_edges)
        """
        # Stage 1: Extract entities
        raw_entities = self._llm_extract_entities(unit)
        if not raw_entities:
            return [], []

        # Stage 2: Dedup entities
        deduped_entities = self._llm_dedup_entities(raw_entities)

        # Stage 3: Extract facts
        fact_edges = self._llm_extract_facts(unit, deduped_entities)

        return deduped_entities, fact_edges

    def _llm_extract_entities(self, unit: MemoryUnit) -> List[Dict]:
        """Stage 1: LLM-based entity extraction from MemoryUnit."""
        content_json = json.dumps(unit.content, ensure_ascii=False, default=str)
        if len(content_json) > 3000:
            content_json = content_json[:3000] + "..."

        user_prompt = ENTITY_EXTRACTION_USER.format(
            unit_type=unit.type.value if unit.type else "unknown",
            source_task_query=unit.source_task_query or "",
            content_json=content_json,
            task_outcome=unit.task_outcome or "unknown",
        )

        result = self._call_llm(ENTITY_EXTRACTION_SYSTEM, user_prompt)
        if "entities" not in result or not isinstance(result["entities"], list):
            raise RuntimeError("llm_graph entity response is missing an entities list")

        entities = []
        for ent in result["entities"]:
            name = ent.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            labels = ent.get("labels", ["Concept"])
            # Map first label to entity type
            etype = labels[0].lower() if labels else "concept"
            entities.append({
                "name": name,
                "type": etype,
                "labels": labels,
                "aliases": ent.get("aliases", []),
                "summary": ent.get("summary", ""),
            })
        return entities

    def _llm_dedup_entities(self, raw_entities: List[Dict]) -> List[Dict]:
        """Stage 2: Dedup new entities against existing graph entities."""
        if not self._entity_embeddings:
            # No existing entities, nothing to dedup
            return raw_entities

        deduped = []
        existing_embs = []
        existing_nids = []
        for nid, emb in self._entity_embeddings.items():
            existing_embs.append(emb)
            existing_nids.append(nid)

        if not existing_embs:
            return raw_entities

        emb_matrix = np.stack(existing_embs)

        for ent in raw_entities:
            # Quick embedding-based pre-filter
            new_emb = self._encode_text(ent["name"])
            if new_emb is None:
                deduped.append(ent)
                continue

            sims = np.dot(emb_matrix, new_emb) / (
                np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(new_emb) + 1e-9
            )
            top_idx = np.argsort(sims)[::-1][:3]
            candidates = [(existing_nids[i], float(sims[i])) for i in top_idx
                          if sims[i] >= self.dedup_sim_threshold]

            if not candidates:
                deduped.append(ent)
                continue

            # LLM judge for top candidate
            best_nid, best_sim = candidates[0]
            existing_name = self._graph.nodes[best_nid].get("display_name", "")
            existing_summary = self._entity_summaries.get(best_nid, "")
            existing_labels = self._entity_labels.get(best_nid, [])

            user_prompt = ENTITY_DEDUP_USER.format(
                existing_name=existing_name,
                existing_labels=existing_labels,
                existing_summary=existing_summary,
                new_name=ent["name"],
                new_labels=ent.get("labels", []),
                new_summary=ent.get("summary", ""),
            )

            result = self._call_llm(ENTITY_DEDUP_SYSTEM, user_prompt)

            if result and result.get("is_duplicate"):
                # Merge: update existing entity with best name and aliases
                best_name = result.get("best_name", ent["name"])
                self._graph.nodes[best_nid]["display_name"] = best_name
                existing_aliases = set(self._graph.nodes[best_nid].get("aliases", []))
                existing_aliases.add(ent["name"])
                existing_aliases.add(existing_name)
                self._graph.nodes[best_nid]["aliases"] = list(existing_aliases)
                # Update summary if better
                if ent.get("summary") and not existing_summary:
                    self._entity_summaries[best_nid] = ent["summary"]
                logger.debug(f"Dedup: merged '{ent['name']}' into '{best_name}' ({best_nid})")
                # Still add to deduped with mapped info so content->entity edge is created
                _, layer_rest = best_nid.split(":", 1) if ":" in best_nid else ("entity", best_nid)
                parts = layer_rest.split(":", 1)
                deduped.append({
                    "name": best_name,
                    "type": parts[0] if len(parts) > 1 else ent.get("type", "concept"),
                    "labels": ent.get("labels", existing_labels),
                    "aliases": list(existing_aliases),
                    "summary": self._entity_summaries.get(best_nid, ent.get("summary", "")),
                })
            else:
                deduped.append(ent)

        return deduped

    def _llm_extract_facts(self, unit: MemoryUnit, entities: List[Dict]) -> List[Dict]:
        """Stage 3: LLM-based fact/relation extraction between entities."""
        if len(entities) < 2:
            return []

        content_json = json.dumps(unit.content, ensure_ascii=False, default=str)
        if len(content_json) > 3000:
            content_json = content_json[:3000] + "..."

        entity_list_text = "\n".join(
            f"- {e['name']} [{', '.join(e.get('labels', []))}]: {e.get('summary', '')}"
            for e in entities
        )

        user_prompt = FACT_EXTRACTION_USER.format(
            unit_type=unit.type.value if unit.type else "unknown",
            content_json=content_json,
            source_task_query=unit.source_task_query or "",
            entity_list_text=entity_list_text,
        )

        result = self._call_llm(FACT_EXTRACTION_SYSTEM, user_prompt)
        if "facts" not in result or not isinstance(result["facts"], list):
            raise RuntimeError("llm_graph fact response is missing a facts list")

        # Stage 4: Map entity names to graph node IDs and dedup
        fact_edges = []
        seen = set()
        entity_name_to_nid = {}
        for ent in entities:
            norm = normalize_entity_name(ent["name"])
            etype = ent.get("type", "concept")
            nid = self._entity_nid(etype, norm)
            entity_name_to_nid[ent["name"].lower()] = nid
            for alias in ent.get("aliases", []):
                entity_name_to_nid[alias.lower()] = nid

        for fact in result["facts"]:
            src_name = fact.get("source_entity", "").lower()
            tgt_name = fact.get("target_entity", "").lower()
            rel_type = fact.get("relation_type", "RELATED").upper()
            description = fact.get("fact", "")

            src_nid = entity_name_to_nid.get(src_name)
            tgt_nid = entity_name_to_nid.get(tgt_name)
            if not src_nid or not tgt_nid or src_nid == tgt_nid:
                continue

            dedup_key = (src_nid, tgt_nid, rel_type)
            if dedup_key in seen or dedup_key in self._fact_descriptions:
                continue
            seen.add(dedup_key)

            fact_edges.append({
                "source_nid": src_nid,
                "target_nid": tgt_nid,
                "relation_type": rel_type,
                "fact": description,
            })

        return fact_edges

    # ------------------------------------------------------------------
    # Entity embedding helpers
    # ------------------------------------------------------------------

    def _compute_entity_embedding(self, entity_nid: str, name: str) -> None:
        """Compute and cache entity name embedding."""
        emb = self._encode_text(name)
        if emb is not None:
            self._entity_embeddings[entity_nid] = emb

    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text using the embedding model from the retriever layer."""
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_embed_model'):
                model_path = os.environ.get(
                    "EMBEDDING_MODEL_PATH",
                    "./storage/models/sentence-transformers_all-MiniLM-L6-v2"
                )
                self._embed_model = SentenceTransformer(model_path)
            emb = self._embed_model.encode(text, normalize_embeddings=True)
            return emb.astype(np.float32)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Graph query extensions
    # ------------------------------------------------------------------

    def get_entity_summary(self, entity_nid: str) -> str:
        """Get the LLM-generated summary for an entity node."""
        return self._entity_summaries.get(entity_nid, "")

    def get_entity_labels(self, entity_nid: str) -> List[str]:
        """Get labels for an entity node."""
        return self._entity_labels.get(entity_nid, [])

    def get_fact_description(self, source_nid: str, target_nid: str, edge_type: str) -> str:
        """Get fact description for a specific edge."""
        return self._fact_descriptions.get((source_nid, target_nid, edge_type), "")

    def get_entity_embedding(self, entity_nid: str) -> Optional[np.ndarray]:
        """Get the name embedding for an entity node."""
        return self._entity_embeddings.get(entity_nid)

    def get_all_entity_embeddings(self) -> Tuple[Optional[np.ndarray], List[str]]:
        """Return stacked entity embeddings and their node IDs."""
        if not self._entity_embeddings:
            return None, []
        nids = list(self._entity_embeddings.keys())
        matrix = np.stack([self._entity_embeddings[n] for n in nids])
        return matrix, nids

    def get_all_fact_texts(self) -> Dict[str, str]:
        """Return all fact descriptions keyed by 'source|target|type'."""
        return {
            f"{s}|{t}|{et}": desc
            for (s, t, et), desc in self._fact_descriptions.items()
        }

    def stats(self) -> Dict:
        """Extended stats with entity and fact info."""
        base = super().stats()
        base["entity_summaries"] = len(self._entity_summaries)
        base["entity_embeddings"] = len(self._entity_embeddings)
        base["fact_descriptions"] = len(self._fact_descriptions)
        base.update(self.llm_usage_stats)
        return base

    # ------------------------------------------------------------------
    # Persistence: save/load extended fields
    # ------------------------------------------------------------------

    def _save_atomic(self):
        """Save graph + extended entity/fact data."""
        super()._save_atomic()

        # Save extended data in a separate JSON
        ext_path = os.path.join(self.storage_dir, "llm_graph_ext.json")
        ext_data = {
            "entity_summaries": self._entity_summaries,
            "entity_labels": self._entity_labels,
            "fact_descriptions": {
                f"{s}|||{t}|||{et}": desc
                for (s, t, et), desc in self._fact_descriptions.items()
            },
        }
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=self.storage_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(ext_data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, ext_path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

        # Save entity embeddings
        if self._entity_embeddings:
            emb_path = os.path.join(self.storage_dir, "entity_embeddings.npz")
            fd2, tmp2 = tempfile.mkstemp(dir=self.storage_dir, suffix=".tmp")
            os.close(fd2)
            try:
                np.savez_compressed(
                    tmp2,
                    **{nid: emb for nid, emb in self._entity_embeddings.items()}
                )
                os.replace(tmp2 + ".npz" if not tmp2.endswith(".npz") else tmp2, emb_path)
                # np.savez adds .npz suffix
                if os.path.exists(tmp2):
                    os.remove(tmp2)
            except Exception:
                for p in [tmp2, tmp2 + ".npz"]:
                    if os.path.exists(p):
                        os.remove(p)
                raise

    def _load(self):
        """Load graph + extended entity/fact data."""
        super()._load()

        # Load extended data
        ext_path = os.path.join(self.storage_dir, "llm_graph_ext.json")
        if os.path.exists(ext_path):
            try:
                with open(ext_path, "r", encoding="utf-8") as f:
                    ext_data = json.load(f)
                self._entity_summaries = ext_data.get("entity_summaries", {})
                self._entity_labels = ext_data.get("entity_labels", {})
                raw_facts = ext_data.get("fact_descriptions", {})
                self._fact_descriptions = {}
                for key, desc in raw_facts.items():
                    parts = key.split("|||")
                    if len(parts) == 3:
                        self._fact_descriptions[(parts[0], parts[1], parts[2])] = desc
            except Exception as e:
                logger.warning(f"Failed to load llm_graph_ext.json: {e}")

        # Load entity embeddings
        emb_path = os.path.join(self.storage_dir, "entity_embeddings.npz")
        if os.path.exists(emb_path):
            try:
                data = np.load(emb_path, allow_pickle=False)
                self._entity_embeddings = {k: data[k] for k in data.files}
            except Exception as e:
                logger.warning(f"Failed to load entity_embeddings.npz: {e}")
