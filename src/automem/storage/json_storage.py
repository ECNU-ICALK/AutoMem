"""
JsonStorage — Pure JSON file storage backend for MemoryUnit.

Stores all MemoryUnit data (including embeddings as float lists) in a single
JSON file. Simple, human-readable, and zero-dependency beyond stdlib + numpy.

Config:
    db_path: str  — Path to the JSON file (default: ./storage/json/memory_db.json)

This is the simplest storage backend and mirrors the original storage approach
used by the modular provider's JSON backend.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..memory_schema import MemoryUnit, MemoryUnitType
from .base_storage import BaseMemoryStorage, StorageHealthReport, atomic_write_json

logger = logging.getLogger(__name__)


class JsonStorage(BaseMemoryStorage):
    """
    Pure JSON file storage.

    All MemoryUnits are kept in-memory as a list and serialized to a single
    JSON file on save(). Embeddings are stored as float arrays within each
    unit's JSON representation.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.db_path: str = self.config.get("db_path", "./storage/json/memory_db.json")
        self._units: List[MemoryUnit] = []
        self._id_index: Dict[str, int] = {}       # id -> list index
        self._sig_index: Dict[str, str] = {}       # signature -> id
        # P3 — query-side index: unit_id -> (query_text_hash, np.ndarray)
        self._query_emb_cache: Dict[str, Tuple[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._load()
            logger.info(
                f"JsonStorage initialized: {len(self._units)} units from {self.db_path}"
            )
            return True
        except Exception as e:
            logger.error(f"JsonStorage initialization failed: {e}")
            return False

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = [u.to_dict() for u in self._units]
        # Atomic write: direct overwrite truncates first, so an interrupt
        # mid-dump corrupted the whole pool (observed as a file containing
        # just "[").
        atomic_write_json(self.db_path, data)

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            self._units = []
            self._rebuild_indices()
            return
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._units = [MemoryUnit.from_dict(d) for d in data]
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._id_index = {u.id: i for i, u in enumerate(self._units)}
        self._sig_index = {u.signature: u.id for u in self._units if u.signature}
        # P2 — rebuild tag inverted index
        self._init_tag_index()
        self._tag_index = {}
        for u in self._units:
            self._tag_index_add(u)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> int:
        added = 0
        for unit in units:
            if unit.signature and unit.signature in self._sig_index:
                logger.debug(f"Duplicate signature {unit.signature}, skipping")
                continue
            idx = len(self._units)
            self._units.append(unit)
            self._id_index[unit.id] = idx
            if unit.signature:
                self._sig_index[unit.signature] = unit.id
            self._tag_index_add(unit)              # P2
            added += 1
        if added > 0:
            self.save()
        return added

    def update(self, unit: MemoryUnit) -> bool:
        idx = self._id_index.get(unit.id)
        if idx is None:
            return False
        old_unit = self._units[idx]
        # P1 — detect stale embedding on content change
        self._maybe_invalidate_stale_embedding(old_unit, unit)
        # P2 — refresh tag index for diffs
        self._tag_index_update(old_unit, unit)
        # P3 — drop cached query embedding if source_task_query changed
        if (old_unit.source_task_query or "") != (unit.source_task_query or ""):
            self._query_emb_cache.pop(unit.id, None)
        old_sig = old_unit.signature
        if old_sig and old_sig in self._sig_index:
            del self._sig_index[old_sig]
        self._units[idx] = unit
        if unit.signature:
            self._sig_index[unit.signature] = unit.id
        self.save()
        return True

    def delete(self, unit_id: str) -> bool:
        idx = self._id_index.get(unit_id)
        if idx is None:
            return False
        unit = self._units[idx]
        if unit.signature and unit.signature in self._sig_index:
            del self._sig_index[unit.signature]
        self._tag_index_remove(unit_id)             # P2
        self._query_emb_cache.pop(unit_id, None)    # P3
        self._units.pop(idx)
        self._rebuild_indices()
        self.save()
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        idx = self._id_index.get(unit_id)
        if idx is None:
            return None
        return self._units[idx]

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
        return signature in self._sig_index

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

    # P3 query-side embedding index — inherited from BaseMemoryStorage
    # (base impl L2-normalizes + iterates self.get_all()).

    # ------------------------------------------------------------------
    # Health report
    # ------------------------------------------------------------------

    # JsonStorage has no hard cold-start threshold — keyword retrieval works
    # even with a single unit. We use 5 as a soft threshold below which
    # TF-IDF vocabulary is too sparse for reliable matching.
    _COLD_START_THRESHOLD = 5

    def get_health_report(self) -> StorageHealthReport:
        total = len(self._units)
        active = sum(1 for u in self._units if u.is_active)
        is_cold = total < self._COLD_START_THRESHOLD
        if total == 0:
            quality = "none"
        elif is_cold:
            quality = "low"
        elif total < 20:
            quality = "medium"
        else:
            quality = "high"
        return StorageHealthReport(
            backend_type="json",
            unit_count=total,
            active_unit_count=active,
            is_cold_start=is_cold,
            cold_start_threshold=self._COLD_START_THRESHOLD,
            retrieval_mode="standard",
            estimated_recall_quality=quality,
        )
