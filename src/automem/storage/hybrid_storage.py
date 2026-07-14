"""
HybridStorage — JSON metadata + FAISS vector index combined storage backend.

Combines the full-fidelity JSON storage of MemoryUnit metadata with a FAISS
index for efficient vector similarity search. Supports structured filtering
(by type, active status, domains, etc.) combined with vector search in a
single query path.

Config:
    storage_dir: str          — Directory for all storage files
    embedding_dim: int        — Embedding dimension (default: 384)
    index_type: str           — FAISS index type: "flat", "ivfflat" (default: "flat")
    nlist: int                — IVFFlat Voronoi cells (default: 100)
    nprobe: int               — IVFFlat probe count (default: 10)

Files created:
    <storage_dir>/memory_db.json   — Full MemoryUnit data (with embeddings as lists)
    <storage_dir>/faiss.index      — FAISS binary index (synced with JSON)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..memory_schema import MemoryUnit, MemoryUnitType
from .base_storage import (
    BaseMemoryStorage,
    StorageHealthReport,
    atomic_write_json,
    embedding_fingerprint,
)

logger = logging.getLogger(__name__)


def _get_faiss():
    """Lazy import faiss."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "HybridStorage requires the 'faiss-cpu' or 'faiss-gpu' package. "
            "Install with: pip install faiss-cpu"
        )


class HybridStorage(BaseMemoryStorage):
    """
    Hybrid JSON + FAISS storage.

    Maintains two synchronized data stores:
      1. JSON file: Full MemoryUnit serialization (human-readable, supports
         structured queries and metadata filtering)
      2. FAISS index: Normalized embeddings for fast similarity search

    The JSON store is the source of truth. The FAISS index is always
    rebuildable from JSON data and is treated as a derived acceleration
    structure.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.storage_dir: str = self.config.get(
            "storage_dir", "./storage/hybrid"
        )
        self.embedding_dim: int = self.config.get("embedding_dim", 384)
        self.index_type: str = self.config.get("index_type", "flat")
        self.nlist: int = self.config.get("nlist", 100)
        self.nprobe: int = self.config.get("nprobe", 10)

        self._json_path = os.path.join(self.storage_dir, "memory_db.json")
        self._index_path = os.path.join(self.storage_dir, "faiss.index")

        # In-memory state
        self._units: List[MemoryUnit] = []
        self._id_index: Dict[str, int] = {}     # unit.id -> list position
        self._sig_set: set = set()
        self._faiss_index = None
        self._pending_deletions: set = set()             # P5 — soft-delete queue
        self._query_emb_cache: Dict[str, Tuple[str, np.ndarray]] = {}  # P3

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            faiss = _get_faiss()
            os.makedirs(self.storage_dir, exist_ok=True)

            # Always load from JSON (source of truth)
            if os.path.exists(self._json_path):
                self._load_json()
            else:
                self._units = []

            self._rebuild_indices()

            # ALWAYS rebuild the FAISS index from the JSON units. The JSON is
            # the declared source of truth and carries the embeddings, so a
            # rebuild is pure memcpy + faiss.add (no re-encoding) at pool
            # scale. The old path blindly loaded the on-disk index "assumed
            # in sync" — in-place update() bugs left 35 stale-vector desyncs
            # across 14 real experiment stores, and a torn save could
            # misalign positions entirely (FAISS hits mapping to the WRONG
            # units). Rebuilding on init self-heals every desync class.
            self._rebuild_faiss_index(faiss)

            logger.info(
                f"HybridStorage initialized: {len(self._units)} units, "
                f"FAISS entries={self._faiss_index.ntotal if self._faiss_index else 0}"
            )
            return True
        except Exception as e:
            logger.error(f"HybridStorage initialization failed: {e}")
            return False

    def save(self) -> None:
        faiss = _get_faiss()
        os.makedirs(self.storage_dir, exist_ok=True)

        # Save JSON (source of truth, includes embeddings) — atomically, so
        # an interrupt can never truncate the pool file.
        data = [u.to_dict() for u in self._units]
        atomic_write_json(self._json_path, data)

        # Save FAISS index (advisory only — initialize() always rebuilds
        # from the JSON) — atomically for the same reason.
        if self._faiss_index is not None:
            tmp_index = self._index_path + ".tmp"
            faiss.write_index(self._faiss_index, tmp_index)
            os.replace(tmp_index, self._index_path)

    def _load_json(self) -> None:
        with open(self._json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._units = [MemoryUnit.from_dict(d) for d in data]

    def _rebuild_indices(self) -> None:
        self._id_index = {u.id: i for i, u in enumerate(self._units)}
        self._sig_set = {u.signature for u in self._units if u.signature}
        # P2 — rebuild tag inverted index
        self._init_tag_index()
        self._tag_index = {}
        for u in self._units:
            self._tag_index_add(u)

    def _create_faiss_index(self, faiss):
        if self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = self.nprobe
            return index
        else:
            return faiss.IndexFlatIP(self.embedding_dim)

    def _rebuild_faiss_index(self, faiss=None) -> None:
        if faiss is None:
            faiss = _get_faiss()

        self._faiss_index = self._create_faiss_index(faiss)

        embs = []
        for unit in self._units:
            if unit.embedding is not None:
                emb = unit.embedding.astype(np.float32).copy()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                embs.append(emb)
            else:
                embs.append(np.zeros(self.embedding_dim, dtype=np.float32))

        if embs:
            emb_matrix = np.vstack(embs).astype(np.float32)
            if self.index_type == "ivfflat" and not self._faiss_index.is_trained:
                if emb_matrix.shape[0] < self.nlist:
                    logger.warning(
                        "ivfflat needs >= nlist(%d) training vectors, got %d; "
                        "falling back to a flat index.",
                        self.nlist, emb_matrix.shape[0],
                    )
                    self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    self._faiss_index.train(emb_matrix)
            self._faiss_index.add(emb_matrix)

        # Refresh the per-position fingerprint snapshot to match the index.
        self._indexed_emb_fp = [embedding_fingerprint(u.embedding) for u in self._units]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> int:
        faiss = _get_faiss()
        added = 0
        new_embs = []

        for unit in units:
            if unit.signature and unit.signature in self._sig_set:
                logger.debug(f"Duplicate signature {unit.signature}, skipping")
                continue

            pos = len(self._units)
            self._units.append(unit)
            self._id_index[unit.id] = pos
            if unit.signature:
                self._sig_set.add(unit.signature)
            self._tag_index_add(unit)              # P2
            if not hasattr(self, "_indexed_emb_fp"):
                self._indexed_emb_fp = [None] * pos
            self._indexed_emb_fp.append(embedding_fingerprint(unit.embedding))

            # Prepare embedding for FAISS
            if unit.embedding is not None:
                emb = unit.embedding.astype(np.float32).copy()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                new_embs.append(emb)
            else:
                new_embs.append(np.zeros(self.embedding_dim, dtype=np.float32))

            added += 1

        # Batch-add to FAISS
        if new_embs:
            emb_matrix = np.vstack(new_embs).astype(np.float32)
            if self.index_type == "ivfflat" and not self._faiss_index.is_trained:
                if emb_matrix.shape[0] < self.nlist:
                    logger.warning(
                        "ivfflat needs >= nlist(%d) training vectors, got %d; "
                        "falling back to a flat index.",
                        self.nlist, emb_matrix.shape[0],
                    )
                    self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    self._faiss_index.train(emb_matrix)
            self._faiss_index.add(emb_matrix)

        if added > 0:
            self.save()
        return added

    def update(self, unit: MemoryUnit) -> bool:
        pos = self._id_index.get(unit.id)
        if pos is None:
            return False

        old_unit = self._units[pos]
        old_sig = old_unit.signature
        if old_sig:
            self._sig_set.discard(old_sig)

        # P1 — invalidate stale embedding on content change
        self._maybe_invalidate_stale_embedding(old_unit, unit)
        # P2 — keep tag index in sync
        self._tag_index_update(old_unit, unit)
        # P3 — drop cached query embedding if source_task_query changed
        if (old_unit.source_task_query or "") != (unit.source_task_query or ""):
            self._query_emb_cache.pop(unit.id, None)

        # Detect whether the embedding changed VS WHAT THE INDEX HOLDS —
        # compare against the fingerprint snapshot taken when this position
        # was last indexed, NOT against old_unit.embedding: callers typically
        # mutate the object returned by get() in place (old_unit IS unit),
        # which made the object comparison always "unchanged" and persisted
        # stale FAISS vectors (35 real desyncs across 14 experiment stores).
        fps = getattr(self, "_indexed_emb_fp", None)
        if fps is None:
            fps = self._indexed_emb_fp = [None] * len(self._units)
        indexed_fp = fps[pos] if pos < len(fps) else None
        embedding_changed = embedding_fingerprint(unit.embedding) != indexed_fp

        self._units[pos] = unit
        if unit.signature:
            self._sig_set.add(unit.signature)

        if embedding_changed:
            self._rebuild_faiss_index()

        self.save()
        return True

    def delete(self, unit_id: str) -> bool:
        """P5 — default to soft-delete. Use ``hard_delete`` for eager removal."""
        pos = self._id_index.get(unit_id)
        if pos is None:
            return False
        self._units[pos].is_active = False
        self._pending_deletions.add(unit_id)
        self.save()
        return True

    def hard_delete(self, unit_id: str) -> bool:
        pos = self._id_index.get(unit_id)
        if pos is None:
            return False
        unit = self._units[pos]
        if unit.signature:
            self._sig_set.discard(unit.signature)
        self._tag_index_remove(unit_id)
        self._query_emb_cache.pop(unit_id, None)
        self._pending_deletions.discard(unit_id)
        self._units.pop(pos)
        self._rebuild_indices()
        self._rebuild_faiss_index()
        self.save()
        return True

    def compact(self) -> int:
        """P5 — physically remove all inactive units and rebuild FAISS.

        Scans ``is_active == False`` directly rather than relying on the
        in-memory ``_pending_deletions`` set (which is not persisted) so
        restart-surviving soft-deletes still get compacted.  Rebuilds the
        unit list from a filter to avoid stale-index hazards while popping.
        """
        to_remove = [u for u in self._units if not u.is_active]
        if not to_remove:
            self._pending_deletions.clear()
            return 0
        for u in to_remove:
            if u.signature:
                self._sig_set.discard(u.signature)
            self._tag_index_remove(u.id)
            self._query_emb_cache.pop(u.id, None)
        self._units = [u for u in self._units if u.is_active]
        self._rebuild_indices()
        self._rebuild_faiss_index()
        self._pending_deletions.clear()
        self.save()
        return len(to_remove)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        pos = self._id_index.get(unit_id)
        if pos is None:
            return None
        return self._units[pos]

    def get_all(
        self,
        active_only: bool = False,
        unit_type: Optional[MemoryUnitType] = None,
    ) -> List[MemoryUnit]:
        result = self._units
        if active_only:
            result = [u for u in result if u.is_active]
        if unit_type is not None:
            result = [u for u in result if u.type == unit_type]
        return result

    def count(self) -> int:
        return len(self._units)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def exists_signature(self, signature: str) -> bool:
        return signature in self._sig_set

    # ------------------------------------------------------------------
    # Embedding access
    # ------------------------------------------------------------------

    def get_embedding_index(
        self,
        active_only: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[MemoryUnit]]:
        if active_only:
            units = [u for u in self._units if u.is_active and u.embedding is not None]
        else:
            units = [u for u in self._units if u.embedding is not None]

        if not units:
            return None, []

        emb_matrix = np.vstack([u.embedding for u in units])
        return emb_matrix, units

    # ------------------------------------------------------------------
    # P3 query-side embedding index — inherited from BaseMemoryStorage.

    # ------------------------------------------------------------------
    # Hybrid search (structured filter + vector similarity)
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        active_only: bool = True,
        unit_type: Optional[MemoryUnitType] = None,
        min_confidence: float = 0.0,
        domains: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        Hybrid search: FAISS vector similarity with post-hoc structured filtering.

        The search retrieves a larger candidate set from FAISS, then applies
        metadata filters (type, active status, confidence, domains) to produce
        the final top-k results.

        Args:
            query_embedding: Query vector, shape (dim,).
            top_k: Number of results to return after filtering.
            active_only: Filter out inactive units.
            unit_type: Filter by MemoryUnitType.
            min_confidence: Minimum confidence threshold.
            domains: If set, unit must have at least one matching domain.

        Returns:
            List of (MemoryUnit, score) tuples sorted by descending similarity.
        """
        if self._faiss_index is None or self._faiss_index.ntotal == 0 or top_k <= 0:
            return []

        # Normalize query
        qe = query_embedding.astype(np.float32).copy().reshape(1, -1)
        norm = np.linalg.norm(qe)
        if norm > 0:
            qe /= norm

        # Expand the search window until we have top_k results that pass the
        # filters or the index is exhausted: a fixed 5x window let
        # high-scoring FILTERED-OUT vectors (inactive / wrong type / low
        # confidence) crowd out lower-scoring valid ones.
        domain_set = set(domains) if domains else None
        search_k = min(max(top_k * 5, 1), self._faiss_index.ntotal)
        while True:
            scores, indices = self._faiss_index.search(qe, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._units):
                    continue

                unit = self._units[idx]

                # Apply filters
                if active_only and not unit.is_active:
                    continue
                if unit_type is not None and unit.type != unit_type:
                    continue
                if unit.confidence < min_confidence:
                    continue
                if domain_set and not domain_set.intersection(unit.applicable_domains):
                    continue

                results.append((unit, float(score)))
                if len(results) >= top_k:
                    break

            if len(results) >= top_k or search_k >= self._faiss_index.ntotal:
                return results
            search_k = min(search_k * 2, self._faiss_index.ntotal)

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    _COLD_START_THRESHOLD = 20

    def get_health_report(self) -> StorageHealthReport:
        total = len(self._units)
        active = sum(1 for u in self._units if u.is_active)
        is_cold = total < self._COLD_START_THRESHOLD
        if self.index_type == "ivfflat" and is_cold:
            mode = "degraded"
        else:
            mode = "standard"
        if total == 0:
            quality = "none"
        elif is_cold:
            quality = "low"
        elif total < 50:
            quality = "medium"
        else:
            quality = "high"
        return StorageHealthReport(
            backend_type="hybrid",
            unit_count=total,
            active_unit_count=active,
            is_cold_start=is_cold,
            cold_start_threshold=self._COLD_START_THRESHOLD,
            retrieval_mode=mode,
            estimated_recall_quality=quality,
        )

    def filtered_get(
        self,
        active_only: bool = True,
        unit_type: Optional[MemoryUnitType] = None,
        min_confidence: float = 0.0,
        task_outcome: Optional[str] = None,
        domains: Optional[List[str]] = None,
    ) -> List[MemoryUnit]:
        """
        Structured metadata-only query without vector similarity.

        Useful for management operations (pruning, statistics, etc.)
        that need to filter by metadata fields.
        """
        result = self._units
        if active_only:
            result = [u for u in result if u.is_active]
        if unit_type is not None:
            result = [u for u in result if u.type == unit_type]
        if min_confidence > 0:
            result = [u for u in result if u.confidence >= min_confidence]
        if task_outcome is not None:
            result = [u for u in result if u.task_outcome == task_outcome]
        if domains:
            domain_set = set(domains)
            result = [
                u for u in result
                if domain_set.intersection(u.applicable_domains)
            ]
        return result
