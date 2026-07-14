"""SQLite-backed cache for web_search results.

Cache key components (hashed together):
  backend              — which search provider (serper / duckduckgo / jina_deepsearch)
  normalized_query     — query text after light normalization (strip + lower)
  serp_num (top_k)     — number of results requested
  filter_year          — search params that affect ranking
  extra_params_json    — any additional kwargs the caller wants to pin to

Each row stores BOTH raw and normalized payload:
  raw       — what the backend returned verbatim (best-effort JSON)
  normalized — the standardized list[dict] our agents consume

Errors and quota-fail responses are NOT cached (transient).
Empty result lists ARE cached (legitimate "no hits" answer for a static question).

Concurrency: SQLite WAL; one connection shared across threads guarded by a
process-local RLock. Safe under shared_memory_provider with concurrency >= 2.
"""
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CacheConfig

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


class SearchCache:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: Optional[float] = None,
        freeze: bool = False,
    ):
        cfg_dir = cache_dir if cache_dir is not None else CacheConfig.from_env().cache_dir
        self.root = Path(cfg_dir) / "search"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "search.db"
        self.ttl_seconds = ttl_seconds
        self.freeze = bool(freeze)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_cache (
                key             TEXT PRIMARY KEY,
                backend         TEXT NOT NULL,
                original_query  TEXT NOT NULL,
                normalized_query TEXT NOT NULL,
                serp_num        INTEGER NOT NULL,
                filter_year     INTEGER,
                extra_params    TEXT,
                raw             TEXT,
                normalized      TEXT NOT NULL,
                created_at      INTEGER NOT NULL,
                schema_version  INTEGER NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_search_created_at ON search_cache(created_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_search_backend ON search_cache(backend)")
        self._conn.commit()
        self._counters = {"hits": 0, "misses": 0, "writes": 0, "miss_frozen": 0}

    def _key(
        self,
        backend: str,
        normalized_query: str,
        serp_num: int,
        filter_year: Optional[int],
        extra_params_json: str,
    ) -> str:
        material = f"{backend}\0{normalized_query}\0{int(serp_num)}\0{filter_year}\0{extra_params_json}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def get(
        self,
        query: str,
        backend: str,
        serp_num: int = 5,
        filter_year: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        if not query or not query.strip():
            return None
        norm = _normalize_query(query)
        extra_json = json.dumps(extra_params or {}, sort_keys=True, ensure_ascii=False)
        key = self._key(backend, norm, serp_num, filter_year, extra_json)
        with self._lock:
            row = self._conn.execute(
                "SELECT normalized, created_at FROM search_cache WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            self._counters["misses"] += 1
            logger.info(
                f"[search_cache_miss] backend={backend} key={key[:8]} q={query[:80]!r}"
            )
            return None
        normalized_json, created_at = row
        if self.ttl_seconds is not None and (time.time() - float(created_at)) > self.ttl_seconds:
            self._counters["misses"] += 1
            logger.info(
                f"[search_cache_miss expired] backend={backend} key={key[:8]} q={query[:80]!r}"
            )
            return None
        try:
            data = json.loads(normalized_json)
        except Exception as e:
            logger.warning(f"[search_cache] corrupt entry {key[:8]}: {e}")
            self._counters["misses"] += 1
            return None
        self._counters["hits"] += 1
        logger.info(
            f"[search_cache_hit] backend={backend} key={key[:8]} q={query[:80]!r} n={len(data) if isinstance(data, list) else '?'}"
        )
        return data

    def set(
        self,
        query: str,
        backend: str,
        serp_num: int,
        filter_year: Optional[int],
        results_normalized: List[Dict[str, Any]],
        results_raw: Any = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not query or not query.strip():
            return
        norm = _normalize_query(query)
        extra_json = json.dumps(extra_params or {}, sort_keys=True, ensure_ascii=False)
        key = self._key(backend, norm, serp_num, filter_year, extra_json)
        try:
            normalized_payload = json.dumps(results_normalized, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[search_cache] cannot serialize normalized: {e}")
            return
        try:
            raw_payload = json.dumps(results_raw, ensure_ascii=False, default=str) if results_raw is not None else None
        except Exception as e:
            logger.debug(f"[search_cache] raw not JSON-serializable, dropping: {e}")
            raw_payload = None
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO search_cache
                (key, backend, original_query, normalized_query, serp_num,
                 filter_year, extra_params, raw, normalized, created_at, schema_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    key,
                    backend,
                    query,
                    norm,
                    int(serp_num),
                    filter_year,
                    extra_json,
                    raw_payload,
                    normalized_payload,
                    int(time.time()),
                    self.SCHEMA_VERSION,
                ),
            )
            self._conn.commit()
            self._counters["writes"] += 1

    def record_miss_frozen(self, query: str, backend: str) -> None:
        self._counters["miss_frozen"] += 1
        logger.warning(
            f"[search_cache_miss_frozen] backend={backend} q={query[:80]!r} "
            f"freeze_cache=true → returning empty results, no external API call"
        )

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]
            backends = dict(
                self._conn.execute(
                    "SELECT backend, COUNT(*) FROM search_cache GROUP BY backend"
                ).fetchall()
            )
        total_lookups = self._counters["hits"] + self._counters["misses"]
        return {
            "rows": n,
            "by_backend": backends,
            **self._counters,
            "hit_rate": (self._counters["hits"] / total_lookups) if total_lookups else 0.0,
            "db_path": str(self.db_path),
            "ttl_days": (self.ttl_seconds / 86400.0) if self.ttl_seconds else None,
            "freeze": self.freeze,
        }

    def purge_expired(self) -> int:
        if self.ttl_seconds is None:
            return 0
        cutoff = int(time.time() - self.ttl_seconds)
        with self._lock:
            cur = self._conn.execute("DELETE FROM search_cache WHERE created_at < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


_singleton: Optional[SearchCache] = None
_singleton_lock = threading.Lock()


def get_search_cache() -> SearchCache:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                cfg = CacheConfig.from_env()
                _singleton = SearchCache(
                    cache_dir=cfg.cache_dir,
                    ttl_seconds=cfg.search_ttl_seconds,
                    freeze=cfg.freeze_cache,
                )
    return _singleton
