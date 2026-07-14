"""
VectorStorage — FAISS-indexed vector storage backend for MemoryUnit.

Stores embeddings in a FAISS index for efficient similarity search,
with a JSON sidecar file for MemoryUnit metadata. This backend is optimized
for large-scale embedding-based retrieval.

Config:
    storage_dir: str          — Directory for index + metadata files
    embedding_dim: int        — Embedding dimension (default: 384 for all-MiniLM-L6-v2)
    index_type: str           — FAISS index type: "flat", "ivfflat" (default: "flat")
    nlist: int                — Number of Voronoi cells for IVFFlat (default: 100)
    nprobe: int               — Number of cells to probe at search time (default: 10)

Files created:
    <storage_dir>/faiss.index      — FAISS binary index
    <storage_dir>/metadata.json    — MemoryUnit metadata (everything except embedding)
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
    """Lazy import faiss to avoid hard dependency at module level."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "VectorStorage requires the 'faiss-cpu' or 'faiss-gpu' package. "
            "Install with: pip install faiss-cpu"
        )


class VectorStorage(BaseMemoryStorage):
    """
    FAISS-indexed vector storage.

    Embeddings are stored in a FAISS index for O(1) or O(log N) similarity
    search. Metadata (all MemoryUnit fields except embedding) is stored in
    a JSON sidecar file. A mapping from FAISS internal IDs to MemoryUnit IDs
    is maintained for consistent lookup.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.storage_dir: str = self.config.get(
            "storage_dir", "./storage/vector"
        )
        self.embedding_dim: int = self.config.get("embedding_dim", 384)
        self.index_type: str = self.config.get("index_type", "flat")
        self.nlist: int = self.config.get("nlist", 100)
        self.nprobe: int = self.config.get("nprobe", 10)

        self._index_path = os.path.join(self.storage_dir, "faiss.index")
        self._meta_path = os.path.join(self.storage_dir, "metadata.json")

        # In-memory state
        self._index = None                          # faiss.Index
        self._units: List[MemoryUnit] = []          # ordered by FAISS internal position
        self._id_to_pos: Dict[str, int] = {}        # unit.id -> position in _units
        self._sig_set: set = set()                   # signatures for dedup
        # P5 — soft-delete set; compact() physically removes
        self._pending_deletions: set = set()
        # P3 — query-side embedding cache
        self._query_emb_cache: Dict[str, Tuple[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            faiss = _get_faiss()
            os.makedirs(self.storage_dir, exist_ok=True)

            if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
                self._load(faiss)
            else:
                self._index = self._create_index(faiss)
                self._units = []
                self._id_to_pos = {}
                self._sig_set = set()
                self._indexed_emb_fp = []

            logger.info(
                f"VectorStorage initialized: {len(self._units)} units, "
                f"index_type={self.index_type}, dim={self.embedding_dim}"
            )
            return True
        except Exception as e:
            logger.error(f"VectorStorage initialization failed: {e}")
            return False

    def _create_index(self, faiss):
        """Create a new FAISS index based on config."""
        if self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = self.nprobe
            return index
        else:
            # Default: flat index with inner product (cosine sim on normalized vectors)
            return faiss.IndexFlatIP(self.embedding_dim)

    def save(self) -> None:
        faiss = _get_faiss()
        os.makedirs(self.storage_dir, exist_ok=True)

        # Atomic writes (tmp + os.replace): an interrupt could otherwise
        # leave a truncated metadata JSON or a torn index/metadata pair.
        tmp_index = self._index_path + ".tmp"
        faiss.write_index(self._index, tmp_index)
        os.replace(tmp_index, self._index_path)

        # Save metadata (MemoryUnits without embeddings to avoid duplication)
        meta_list = []
        for unit in self._units:
            d = unit.to_dict()
            d.pop("embedding", None)  # stored in FAISS, not in JSON
            meta_list.append(d)
        atomic_write_json(self._meta_path, meta_list)

    def _load(self, faiss) -> None:
        """Load existing index and metadata from disk."""
        self._index = faiss.read_index(self._index_path)

        with open(self._meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)

        if self._index.ntotal != len(meta_list):
            # Torn save detection: for the vector backend the vector truth
            # lives ONLY in FAISS, so a count mismatch cannot be self-healed.
            # Positions beyond the shorter side degrade to embedding=None
            # (reconstruct guard below) / are ignored by search.
            logger.warning(
                f"VectorStorage: metadata/index count mismatch "
                f"({len(meta_list)} units vs {self._index.ntotal} vectors) — "
                f"likely a torn save; degraded positions will have no embedding."
            )

        self._units = []
        self._id_to_pos = {}
        self._sig_set = set()

        for i, d in enumerate(meta_list):
            # Reconstruct embedding from FAISS index
            emb = np.zeros(self.embedding_dim, dtype=np.float32)
            try:
                self._index.reconstruct(i, emb)
                # Verify reconstruction succeeded (not still all-zeros)
                if np.any(emb != 0):
                    d["embedding"] = emb.tolist()
                else:
                    logger.warning(
                        f"FAISS reconstruct returned zero vector for position {i} "
                        f"(unit {d.get('id', '?')}), embedding will be None"
                    )
                    d["embedding"] = None
            except RuntimeError as e:
                logger.warning(
                    f"FAISS reconstruct failed for position {i} "
                    f"(unit {d.get('id', '?')}): {e}, embedding will be None"
                )
                d["embedding"] = None

            unit = MemoryUnit.from_dict(d)
            self._units.append(unit)
            self._id_to_pos[unit.id] = i
            if unit.signature:
                self._sig_set.add(unit.signature)
            self._tag_index_add(unit)              # P2 — rebuild tag index on load

        # Snapshot what the index holds per position (units' embeddings were
        # just reconstructed FROM the index, so they are in sync here).
        self._indexed_emb_fp = [embedding_fingerprint(u.embedding) for u in self._units]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> int:
        faiss = _get_faiss()
        added = 0

        vectors_to_add = []
        units_to_add = []

        for unit in units:
            if unit.signature and unit.signature in self._sig_set:
                logger.debug(f"Duplicate signature {unit.signature}, skipping")
                continue
            if unit.embedding is None:
                logger.warning(f"Unit {unit.id} has no embedding, skipping for VectorStorage")
                continue

            units_to_add.append(unit)
            # Normalize for cosine similarity via inner product
            emb = unit.embedding.astype(np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            vectors_to_add.append(emb)

        if not vectors_to_add:
            return 0

        emb_matrix = np.vstack(vectors_to_add).astype(np.float32)

        # Train IVF index if needed and not yet trained. faiss hard-fails
        # (RuntimeError) when the training set is smaller than nlist; fall
        # back to a flat index instead of crashing on small pools.
        if self.index_type == "ivfflat" and not self._index.is_trained:
            if emb_matrix.shape[0] < self.nlist:
                logger.warning(
                    "ivfflat needs >= nlist(%d) training vectors, got %d; "
                    "falling back to a flat index.",
                    self.nlist, emb_matrix.shape[0],
                )
                self._index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self._index.train(emb_matrix)

        self._index.add(emb_matrix)

        for unit in units_to_add:
            pos = len(self._units)
            self._units.append(unit)
            self._id_to_pos[unit.id] = pos
            if unit.signature:
                self._sig_set.add(unit.signature)
            self._tag_index_add(unit)              # P2
            self._indexed_emb_fp.append(embedding_fingerprint(unit.embedding))
            added += 1

        if added > 0:
            self.save()
        return added

    def update(self, unit: MemoryUnit) -> bool:
        pos = self._id_to_pos.get(unit.id)
        if pos is None:
            return False

        old_unit = self._units[pos]
        old_sig = old_unit.signature
        if old_sig:
            self._sig_set.discard(old_sig)

        # P1 — drop stale embedding if content changed but embedding didn't.
        self._maybe_invalidate_stale_embedding(old_unit, unit)
        # P2 — refresh tag index for diffs
        self._tag_index_update(old_unit, unit)
        # P3 — drop cached query embedding if source_task_query changed
        if (old_unit.source_task_query or "") != (unit.source_task_query or ""):
            self._query_emb_cache.pop(unit.id, None)

        # Detect whether the embedding changed VS WHAT THE INDEX HOLDS —
        # compare against the fingerprint snapshot taken when this position
        # was last indexed, NOT against old_unit.embedding: callers typically
        # mutate the object returned by get() in place (old_unit IS unit),
        # which made the object comparison always "unchanged" and left stale
        # vectors in FAISS (35 real desyncs across 14 experiment stores).
        fps = getattr(self, "_indexed_emb_fp", None)
        if fps is None:
            fps = self._indexed_emb_fp = [None] * len(self._units)
        indexed_fp = fps[pos] if pos < len(fps) else None
        embedding_changed = embedding_fingerprint(unit.embedding) != indexed_fp

        self._units[pos] = unit
        if unit.signature:
            self._sig_set.add(unit.signature)

        # Only rebuild FAISS when the embedding vector itself changed.
        # Pure metadata updates (confidence, decay_weight, etc.) skip rebuild.
        if embedding_changed:
            self._rebuild_faiss_index()

        self.save()
        return True

    def delete(self, unit_id: str) -> bool:
        """P5 — default delete now uses soft-delete + compact-threshold.

        Hard-rebuild rebuilds the entire FAISS index which is O(N) and blocks
        the search loop; we instead mark the unit soft-deleted (is_active=False
        + pending_deletions) and let ``compact`` clean up in a maintenance
        window.  Callers wanting the old behaviour can call ``hard_delete``.
        """
        return self._soft_delete_impl(unit_id)

    def hard_delete(self, unit_id: str) -> bool:
        pos = self._id_to_pos.get(unit_id)
        if pos is None:
            return False

        unit = self._units[pos]
        if unit.signature:
            self._sig_set.discard(unit.signature)
        self._tag_index_remove(unit_id)             # P2
        self._query_emb_cache.pop(unit_id, None)    # P3
        self._pending_deletions.discard(unit_id)

        self._units.pop(pos)
        self._id_to_pos = {u.id: i for i, u in enumerate(self._units)}
        self._rebuild_faiss_index()
        self.save()
        return True

    # P5 — soft delete: mark inactive + enqueue for compact
    def _soft_delete_impl(self, unit_id: str) -> bool:
        pos = self._id_to_pos.get(unit_id)
        if pos is None:
            return False
        u = self._units[pos]
        u.is_active = False
        self._pending_deletions.add(unit_id)
        self.save()
        return True

    def compact(self) -> int:
        """Physically remove all inactive units and rebuild FAISS.

        Uses ``is_active == False`` as the source of truth rather than the
        in-memory ``_pending_deletions`` set — the set is not persisted, so
        relying on it would leave soft-deleted units stranded across
        restarts.  Rebuilding the unit list from a filter also avoids the
        pop-by-stale-index hazard in the old implementation.
        """
        to_remove = [u for u in self._units if not u.is_active]
        if not to_remove:
            self._pending_deletions.clear()
            return 0
        # Tear down per-unit caches for the removed set
        for u in to_remove:
            if u.signature:
                self._sig_set.discard(u.signature)
            self._tag_index_remove(u.id)
            self._query_emb_cache.pop(u.id, None)
        # Rebuild unit list from active-only survivors
        self._units = [u for u in self._units if u.is_active]
        self._id_to_pos = {u.id: i for i, u in enumerate(self._units)}
        self._rebuild_faiss_index()
        self._pending_deletions.clear()
        self.save()
        return len(to_remove)

    def _rebuild_faiss_index(self) -> None:
        """Rebuild the entire FAISS index from current units."""
        faiss = _get_faiss()
        self._index = self._create_index(faiss)

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
            if self.index_type == "ivfflat" and not self._index.is_trained:
                if emb_matrix.shape[0] < self.nlist:
                    logger.warning(
                        "ivfflat needs >= nlist(%d) training vectors, got %d; "
                        "falling back to a flat index.",
                        self.nlist, emb_matrix.shape[0],
                    )
                    self._index = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    self._index.train(emb_matrix)
            self._index.add(emb_matrix)

        # Refresh the per-position fingerprint snapshot to match the index.
        self._indexed_emb_fp = [embedding_fingerprint(u.embedding) for u in self._units]

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        pos = self._id_to_pos.get(unit_id)
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

        # Return L2-normalized embeddings to match the FAISS index,
        # ensuring cosine similarity via numpy is consistent with FAISS search.
        raw = np.vstack([u.embedding for u in units]).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        emb_matrix = raw / norms
        return emb_matrix, units

    # ------------------------------------------------------------------
    # P3 query-side embedding index — inherited from BaseMemoryStorage.

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    # Below this threshold FAISS Flat search still works but the embedding
    # space is too sparse for reliable semantic matching.
    _COLD_START_THRESHOLD = 20

    def get_health_report(self) -> StorageHealthReport:
        total = len(self._units)
        active = sum(1 for u in self._units if u.is_active)
        is_cold = total < self._COLD_START_THRESHOLD
        # Distinguish index modes so the LLM knows accuracy level
        if self.index_type == "ivfflat" and is_cold:
            mode = "degraded"       # IVFFlat performs poorly below nlist
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
            backend_type="vector",
            unit_count=total,
            active_unit_count=active,
            is_cold_start=is_cold,
            cold_start_threshold=self._COLD_START_THRESHOLD,
            retrieval_mode=mode,
            estimated_recall_quality=quality,
        )

    # ------------------------------------------------------------------
    # FAISS-native search (bonus: direct similarity search)
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        active_only: bool = True,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        Perform FAISS-native similarity search.

        This is a convenience method that bypasses the retrieval layer for
        direct vector search. The retrieval layer can also use
        get_embedding_index() for its own search logic.

        Args:
            query_embedding: Query vector, shape (dim,).
            top_k: Number of results to return.
            active_only: If True, filter out inactive units from results.

        Returns:
            List of (MemoryUnit, score) tuples sorted by descending similarity.
        """
        if self._index is None or self._index.ntotal == 0 or top_k <= 0:
            return []

        # Normalize query
        qe = query_embedding.astype(np.float32).copy().reshape(1, -1)
        norm = np.linalg.norm(qe)
        if norm > 0:
            qe /= norm

        # Expand the search window until we have top_k ACTIVE results or the
        # index is exhausted: a fixed 3x window let high-scoring INACTIVE
        # (soft-deleted) vectors crowd out lower-scoring active ones, so
        # semantic search returned empty while active units existed.
        search_k = min(max(top_k * 3, 1), self._index.ntotal)
        while True:
            scores, indices = self._index.search(qe, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._units):
                    continue
                unit = self._units[idx]
                if active_only and not unit.is_active:
                    continue
                results.append((unit, float(score)))
                if len(results) >= top_k:
                    break

            if len(results) >= top_k or search_k >= self._index.ntotal:
                return results
            search_k = min(search_k * 2, self._index.ntotal)
