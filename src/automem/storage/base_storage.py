"""
Abstract base class for memory storage backends.

The storage layer is responsible ONLY for persistence and basic CRUD operations
on MemoryUnit objects. It is orthogonal to:
  - Extraction: how MemoryUnits are created from trajectories
  - Retrieval:  how MemoryUnits are searched/ranked at query time
  - Management: how MemoryUnits are pruned, merged, or decayed over time
"""

import hashlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..memory_schema import MemoryUnit, MemoryUnitType


# --------------------------------------------------------------------
# Helpers shared by all backends
# --------------------------------------------------------------------

def atomic_write_json(path: str, obj: Any, indent: int = 2) -> None:
    """Write JSON via tempfile-in-same-dir + os.replace (atomic on POSIX).

    Direct ``open(path, "w")`` truncates first, so an interrupt mid-dump
    leaves a corrupted file (real failure mode: a store reduced to ``[``,
    making the whole pool unreadable on the next run). Same pattern as
    GraphStore._save_atomic.
    """
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def embedding_fingerprint(emb: Optional[np.ndarray]) -> Optional[bytes]:
    """Cheap stable fingerprint of an embedding vector (md5 of float32 bytes).

    Used by vector/hybrid stores to detect whether a unit's embedding
    changed relative to what the FAISS index holds. Comparing against the
    old unit OBJECT is unreliable: callers typically mutate the object
    returned by get() in place, so old and new are the same object and the
    comparison is always "unchanged".
    """
    if emb is None:
        return None
    arr = np.asarray(emb, dtype=np.float32)
    return hashlib.md5(arr.tobytes()).digest()

def content_fingerprint(unit: MemoryUnit) -> str:
    """Stable hash of a unit's semantic text.  Used to detect content-level
    changes (e.g. from case_rewrite) so stale embeddings can be invalidated.
    """
    try:
        text = unit.content_text()
    except Exception:
        text = json.dumps(unit.content or {}, sort_keys=True, default=str)
    return hashlib.md5((text or "").encode("utf-8", errors="ignore")).hexdigest()


def collect_tags(unit: MemoryUnit) -> Set[str]:
    """Collect all tag-like strings on a unit for the tag inverted index.

    Pulls from both ``applicable_task_types`` (runtime tags like tool_valid /
    promoted / case_rewritten) and ``content.task_type_tags`` (extraction-time
    domain tags).
    """
    tags: Set[str] = set()
    for t in getattr(unit, "applicable_task_types", []) or []:
        if isinstance(t, str) and t:
            tags.add(t)
    content = getattr(unit, "content", {}) or {}
    for t in content.get("task_type_tags", []) or []:
        if isinstance(t, str) and t:
            tags.add(t)
    return tags


@dataclass
class StorageHealthReport:
    """
    Runtime health snapshot for a storage backend.

    Produced by get_health_report() and consumed by:
      - architecture_selection prompt (to guide LLM decisions)
      - feedback_analysis prompt (to diagnose retrieval failures)

    Fields
    ------
    backend_type          : one of "json" | "vector" | "hybrid" | "graph" | "llm_graph"
    unit_count            : total MemoryUnits in storage (including inactive)
    active_unit_count     : units where is_active=True
    is_cold_start         : True when below cold_start_threshold
    cold_start_threshold  : backend-specific minimum for reliable retrieval
    retrieval_mode        : "standard" | "degraded" | "graph_sparse"
    estimated_recall_quality : "high" | "medium" | "low" | "none"
    graph_edge_count      : content-to-content edges (graph backends only)
    graph_avg_degree      : mean out-degree of content nodes (graph backends only)
    """
    backend_type: str
    unit_count: int
    active_unit_count: int
    is_cold_start: bool
    cold_start_threshold: int
    retrieval_mode: str                  # "standard" | "degraded" | "graph_sparse"
    estimated_recall_quality: str        # "high" | "medium" | "low" | "none"
    graph_edge_count: int = 0
    graph_avg_degree: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "backend_type": self.backend_type,
            "unit_count": self.unit_count,
            "active_unit_count": self.active_unit_count,
            "is_cold_start": self.is_cold_start,
            "cold_start_threshold": self.cold_start_threshold,
            "retrieval_mode": self.retrieval_mode,
            "estimated_recall_quality": self.estimated_recall_quality,
            "graph_edge_count": self.graph_edge_count,
            "graph_avg_degree": round(self.graph_avg_degree, 3),
        }


class BaseMemoryStorage(ABC):
    """
    Abstract interface for memory storage backends.

    All implementations must support:
      1. Persist and load MemoryUnits
      2. Basic CRUD by ID
      3. Signature-based deduplication check
      4. Provide embedding matrix for the retrieval layer
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """
        Set up storage backend (create dirs, load existing data, build indices).

        Returns:
            True if initialization succeeded.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Flush all in-memory state to persistent storage."""
        pass

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, units: List[MemoryUnit]) -> int:
        """
        Add new MemoryUnits to storage. Skips units whose signature already exists.

        Args:
            units: List of MemoryUnit objects to store.

        Returns:
            Number of units actually added (after dedup).
        """
        pass

    @abstractmethod
    def update(self, unit: MemoryUnit) -> bool:
        """
        Update an existing MemoryUnit (matched by id).

        Returns:
            True if the unit was found and updated.
        """
        pass

    @abstractmethod
    def delete(self, unit_id: str) -> bool:
        """
        Remove a MemoryUnit by id.

        Returns:
            True if the unit was found and removed.
        """
        pass

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        """Retrieve a single MemoryUnit by id, or None if not found."""
        pass

    @abstractmethod
    def get_all(
        self,
        active_only: bool = False,
        unit_type: Optional[MemoryUnitType] = None,
    ) -> List[MemoryUnit]:
        """
        Retrieve all stored MemoryUnits, optionally filtered.

        Args:
            active_only: If True, return only units with is_active=True.
            unit_type: If set, return only units of that type.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored MemoryUnits."""
        pass

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @abstractmethod
    def exists_signature(self, signature: str) -> bool:
        """Check whether a MemoryUnit with the given signature already exists."""
        pass

    # ------------------------------------------------------------------
    # Embedding access (for retrieval layer)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_embedding_index(
        self,
        active_only: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[MemoryUnit]]:
        """
        Return the embedding matrix and corresponding MemoryUnit list.

        This is the primary interface for the retrieval layer to perform
        similarity search without knowing storage internals.

        Args:
            active_only: If True, only include active units with embeddings.

        Returns:
            (embedding_matrix, units) where embedding_matrix is shape (N, dim)
            and units[i] corresponds to embedding_matrix[i].
            Returns (None, []) if no embeddings are available.
        """
        pass

    @abstractmethod
    def get_health_report(self) -> "StorageHealthReport":
        """
        Return a runtime health snapshot of this storage backend.

        Called by the optimization loop to inject storage state into the
        architecture_selection and feedback_analysis prompts, enabling the
        LLM to make informed decisions based on actual runtime conditions
        rather than static component descriptions.

        Returns:
            StorageHealthReport dataclass with current backend state.
        """
        pass

    # ------------------------------------------------------------------
    # P1/P2 shared: snapshot caches keyed on unit.id.
    #
    # Rationale (codex review finding): callers like CaseRewriteOp mutate
    # a MemoryUnit IN PLACE before calling ``update(unit)``.  Because the
    # store keeps the unit by reference (``self._units[idx]``), comparing
    # ``old_unit`` against ``new_unit`` yields equal tags/content (they are
    # the same object).  P1/P2 diffs against these snapshots instead.
    # ------------------------------------------------------------------

    def _init_snapshot_caches(self) -> None:
        """Subclasses must call this from ``initialize`` / ``_rebuild_indices``."""
        if not hasattr(self, "_tag_index"):
            self._tag_index: Dict[str, Set[str]] = {}
        if not hasattr(self, "_content_fp_cache"):
            self._content_fp_cache: Dict[str, str] = {}
        if not hasattr(self, "_tag_snapshot"):
            self._tag_snapshot: Dict[str, Set[str]] = {}
        if not hasattr(self, "_emb_fp_cache"):
            self._emb_fp_cache: Dict[str, Optional[bytes]] = {}

    # Backward-compat alias: existing code calls _init_tag_index.
    _init_tag_index = _init_snapshot_caches

    # ------------------------------------------------------------------
    # P1 — Stale embedding detection on update
    # ------------------------------------------------------------------

    def _maybe_invalidate_stale_embedding(
        self, old_unit: Optional[MemoryUnit], new_unit: MemoryUnit
    ) -> None:
        """If unit's content fingerprint changed since last add/update, treat
        embedding as stale.  Compares against the FINGERPRINT CACHE rather
        than ``old_unit`` — the snapshot cache is safe even when the caller
        mutates the unit in place.

        If a storage-level ``embedding_model`` was injected (see
        ``set_embedding_model``), re-encode immediately so retrievers never
        see a ``None`` embedding.  Otherwise fall back to setting embedding
        to ``None`` and rely on a later refresh op.
        """
        self._init_snapshot_caches()
        try:
            new_fp = content_fingerprint(new_unit)
        except Exception:
            return

        old_fp = self._content_fp_cache.get(new_unit.id)
        cur_emb_fp = embedding_fingerprint(new_unit.embedding)
        if old_fp is None or old_fp == new_fp:
            # First insert or no content change
            self._content_fp_cache[new_unit.id] = new_fp
            self._emb_fp_cache[new_unit.id] = cur_emb_fp
            return

        # Content changed. If the caller ALSO brought a new embedding (its
        # fingerprint differs from the last-seen one), trust it — e.g.
        # reflection_correction re-encodes before calling update(). The old
        # unconditional invalidation DISCARDED that freshly computed
        # embedding whenever no storage-level model was injected.
        prev_emb_fp = self._emb_fp_cache.get(new_unit.id)
        if cur_emb_fp != prev_emb_fp:
            self._content_fp_cache[new_unit.id] = new_fp
            self._emb_fp_cache[new_unit.id] = cur_emb_fp
            return

        # Content changed but the embedding did NOT — genuinely stale.
        # Try re-encode in-place if we have a model.
        model = getattr(self, "_embedding_model", None)
        if model is not None:
            try:
                text = new_unit.content_text() or ""
                if text:
                    new_emb = model.encode(text, convert_to_numpy=True,
                                           show_progress_bar=False)
                    new_unit.embedding = np.asarray(new_emb, dtype=np.float32)
                    self._content_fp_cache[new_unit.id] = new_fp
                    self._emb_fp_cache[new_unit.id] = embedding_fingerprint(
                        new_unit.embedding
                    )
                    return
            except Exception:
                pass  # fall through to blunt invalidation

        # No model: clear the embedding so the unit is at least not retrieved
        # with stale semantics.  A refresh_stale_embeddings op can re-encode
        # later.  (Silent hide, not ideal — prefer injecting an embedding_model.)
        new_unit.embedding = None
        self._content_fp_cache[new_unit.id] = new_fp
        self._emb_fp_cache[new_unit.id] = None

    def set_embedding_model(self, model) -> None:
        """Providers call this so the Store can re-encode on content updates
        without exposing embedding internals to callers.  Accepts any object
        with ``.encode(text, convert_to_numpy=True, show_progress_bar=...)``.
        """
        self._embedding_model = model

    # ------------------------------------------------------------------
    # P2 — Tag inverted index (default in-memory dict-of-sets)
    # ------------------------------------------------------------------

    def _tag_index_add(self, unit: MemoryUnit) -> None:
        self._init_snapshot_caches()
        tags = collect_tags(unit)
        for t in tags:
            self._tag_index.setdefault(t, set()).add(unit.id)
        self._tag_snapshot[unit.id] = set(tags)
        # Seed the content fingerprint cache so the NEXT update() can detect
        # a content diff even when the caller mutated the same object in
        # place (the common CaseRewriteOp pattern).
        try:
            self._content_fp_cache[unit.id] = content_fingerprint(unit)
            self._emb_fp_cache[unit.id] = embedding_fingerprint(unit.embedding)
        except Exception:
            pass

    def _tag_index_remove(self, unit_id: str) -> None:
        self._init_snapshot_caches()
        for ids in list(self._tag_index.values()):
            ids.discard(unit_id)
        self._tag_snapshot.pop(unit_id, None)
        self._content_fp_cache.pop(unit_id, None)
        self._emb_fp_cache.pop(unit_id, None)

    def _tag_index_update(
        self, old_unit: Optional[MemoryUnit], new_unit: MemoryUnit
    ) -> None:
        """Diff against the CACHED tag set, not against ``old_unit`` — the
        latter is unsafe under in-place mutation.
        """
        self._init_snapshot_caches()
        new_tags = collect_tags(new_unit)
        old_tags = self._tag_snapshot.get(new_unit.id, set())
        for t in old_tags - new_tags:
            if t in self._tag_index:
                self._tag_index[t].discard(new_unit.id)
        for t in new_tags - old_tags:
            self._tag_index.setdefault(t, set()).add(new_unit.id)
        self._tag_snapshot[new_unit.id] = set(new_tags)

    def get_units_by_tag(
        self, tag: str, active_only: bool = True,
    ) -> List[MemoryUnit]:
        """Look up units carrying a given tag (O(k) instead of O(N)).

        Default impl iterates the in-memory index; subclasses with richer
        structures (GraphStore) may override for better semantics.
        """
        self._init_tag_index()
        ids = self._tag_index.get(tag, set())
        out: List[MemoryUnit] = []
        for uid in ids:
            u = self.get(uid)
            if u is None:
                continue
            if active_only and not u.is_active:
                continue
            out.append(u)
        return out

    # ------------------------------------------------------------------
    # P3 — Query-side embedding index (for CBR)
    # ------------------------------------------------------------------

    def _init_query_cache(self) -> None:
        if not hasattr(self, "_query_emb_cache"):
            self._query_emb_cache: Dict[str, Tuple[str, np.ndarray]] = {}

    def set_query_embedding(
        self, unit_id: str, query_text: str, embedding: np.ndarray,
    ) -> None:
        """Cache a L2-normalized embedding for the unit's source_task_query.

        Callers pass any embedding; we normalize here so downstream cosine
        similarity is always correct regardless of caller's normalization.
        """
        self._init_query_cache()
        import hashlib
        h = hashlib.md5((query_text or "").encode("utf-8", errors="ignore")).hexdigest()
        v = np.asarray(embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        self._query_emb_cache[unit_id] = (h, v)

    def get_query_embedding_index(
        self, active_only: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[MemoryUnit]]:
        """Return a matrix of (L2-normalized) query embeddings + paired units.

        Empty cache → returns (None, []).  Retrievers MUST fall back to
        scanning when this returns None.
        """
        self._init_query_cache()
        units_with_emb: List[MemoryUnit] = []
        vectors: List[np.ndarray] = []
        for u in self.get_all(active_only=active_only):
            cached = self._query_emb_cache.get(u.id)
            if cached is None:
                continue
            vectors.append(cached[1])
            units_with_emb.append(u)
        if not units_with_emb:
            return None, []
        return np.vstack(vectors), units_with_emb

    # ------------------------------------------------------------------
    # P4 — Oversample / pool search API
    # ------------------------------------------------------------------

    def search_with_pool(
        self,
        query_embedding: Optional[np.ndarray],
        top_k: int = 5,
        pool_size: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """Return up to ``pool_size`` candidates (default ``top_k * 3``) so
        that the fixed context composer can inspect a larger pool before
        trimming to ``top_k``.

        Default impl delegates to ``get_embedding_index`` and does naive
        cosine ranking in numpy; backend-specific implementations (FAISS)
        may override.  Returns empty list if no embedding index exists.
        """
        pool_size = pool_size or max(top_k * 3, top_k)
        mat, units = self.get_embedding_index(active_only=True)
        if mat is None or not units or query_embedding is None:
            return []
        try:
            q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            # Normalize for cosine
            qn = q / (np.linalg.norm(q) + 1e-8)
            M = np.asarray(mat, dtype=np.float32)
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
            scores = Mn @ qn
        except Exception:
            return []

        # Optional simple filters
        candidates = list(zip(units, scores.tolist()))
        if filters:
            keep: List[Tuple[MemoryUnit, float]] = []
            t = filters.get("unit_type")
            min_conf = filters.get("min_confidence")
            require_tag = filters.get("require_tag")
            for u, s in candidates:
                if t and u.type != t:
                    continue
                if min_conf is not None and u.confidence < float(min_conf):
                    continue
                if require_tag and require_tag not in (u.applicable_task_types or []):
                    continue
                keep.append((u, s))
            candidates = keep

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:pool_size]

    # ------------------------------------------------------------------
    # P5 — Soft-delete + compact
    # ------------------------------------------------------------------

    def soft_delete(self, unit_id: str) -> bool:
        """Mark a unit inactive without rebuilding indices.

        Default impl fetches the unit, sets is_active=False, and calls
        ``update``.  Backends with heavy delete costs (FAISS) should prefer
        this path and run ``compact`` during a periodic maintenance window.
        """
        u = self.get(unit_id)
        if u is None:
            return False
        if not u.is_active:
            return True
        u.is_active = False
        return self.update(u)

    def compact(self) -> int:
        """Periodic maintenance: physically remove soft-deleted units and
        rebuild indices.  Returns number of units physically removed.

        Default no-op; backends that benefit from compaction override.
        """
        return 0
