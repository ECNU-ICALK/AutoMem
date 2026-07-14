"""Filesystem-backed cache for read_page() page contents.

Layout:
  .cache/pages/<reader>/<aa>/<sha256>.json

  - One file per (reader, normalized_url, extraction_mode) tuple.
  - Two-level subdirectory <aa>/ (first 2 hex chars of hash) keeps any
    single dir below ~10K files at typical scale.
  - Stored as JSON containing original_url, reader, mode, timestamp,
    raw and normalized content. We pick JSON over plaintext so we never
    lose provenance and can extend the schema later.

URL normalization:
  - lowercase scheme + host
  - strip trailing slash on path
  - drop fragment
  - preserve params + query (different ?id=... is different content)

Errors and quota-fail responses are not cached (heuristic check).
"""
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

from .config import CacheConfig

logger = logging.getLogger(__name__)


def _looks_like_error(content: str) -> bool:
    if not content:
        return True
    head = content[:300].lower()
    markers = (
        "error reading page",
        "terminate_quota",
        "rate limit exceeded",
        "no content extracted",
    )
    return any(m in head for m in markers)


def _normalize_url(url: str) -> str:
    u = urlparse((url or "").strip())
    scheme = (u.scheme or "https").lower()
    netloc = u.netloc.lower()
    path = u.path.rstrip("/")
    return urlunparse((scheme, netloc, path, u.params, u.query, ""))


class PageCache:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: Optional[float] = None,
        freeze: bool = False,
    ):
        cfg_dir = cache_dir if cache_dir is not None else CacheConfig.from_env().cache_dir
        self.root = Path(cfg_dir) / "pages"
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.freeze = bool(freeze)
        self._counters = {"hits": 0, "misses": 0, "writes": 0, "miss_frozen": 0}
        self._lock = threading.Lock()

    def _key(self, url: str, reader: str, mode: str) -> str:
        norm = _normalize_url(url)
        material = f"{reader}\0{mode}\0{norm}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path(self, key: str, reader: str) -> Path:
        return self.root / reader / key[:2] / f"{key}.json"

    def get(self, url: str, reader: str, mode: str = "default") -> Optional[str]:
        key = self._key(url, reader, mode)
        p = self._path(key, reader)
        if not p.exists():
            self._counters["misses"] += 1
            logger.info(
                f"[page_cache_miss] reader={reader} mode={mode} key={key[:8]} url={url[:120]}"
            )
            return None
        try:
            mtime = p.stat().st_mtime
        except Exception:
            self._counters["misses"] += 1
            return None
        if self.ttl_seconds is not None and (time.time() - mtime) > self.ttl_seconds:
            self._counters["misses"] += 1
            logger.info(
                f"[page_cache_miss expired] reader={reader} mode={mode} key={key[:8]} url={url[:120]}"
            )
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except Exception as e:
            logger.warning(f"[page_cache] read failed for {p}: {e}")
            self._counters["misses"] += 1
            return None
        content = blob.get("normalized") or blob.get("raw") or ""
        self._counters["hits"] += 1
        logger.info(
            f"[page_cache_hit] reader={reader} mode={mode} key={key[:8]} url={url[:120]} bytes={len(content)}"
        )
        return content

    def set(
        self,
        url: str,
        reader: str,
        mode: str,
        normalized_content: str,
        raw_content: Optional[str] = None,
    ) -> None:
        if not normalized_content or _looks_like_error(normalized_content):
            return
        key = self._key(url, reader, mode)
        p = self._path(key, reader)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "schema_version": self.SCHEMA_VERSION,
            "original_url": url,
            "normalized_url": _normalize_url(url),
            "reader": reader,
            "mode": mode,
            "created_at": int(time.time()),
            "raw": raw_content,
            "normalized": normalized_content,
        }
        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            with self._lock:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(blob, f, ensure_ascii=False)
                tmp.replace(p)
            self._counters["writes"] += 1
        except Exception as e:
            logger.warning(f"[page_cache] write failed for {p}: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def record_miss_frozen(self, url: str, reader: str, mode: str) -> None:
        self._counters["miss_frozen"] += 1
        logger.warning(
            f"[page_cache_miss_frozen] reader={reader} mode={mode} url={url[:120]} "
            f"freeze_cache=true → returning empty content, no external fetch"
        )

    def stats(self) -> Dict[str, Any]:
        try:
            files = sum(1 for _ in self.root.rglob("*.json"))
            size_bytes = sum(p.stat().st_size for p in self.root.rglob("*.json"))
        except Exception:
            files, size_bytes = 0, 0
        total_lookups = self._counters["hits"] + self._counters["misses"]
        return {
            "files": files,
            "size_mb": round(size_bytes / 1e6, 2),
            **self._counters,
            "hit_rate": (self._counters["hits"] / total_lookups) if total_lookups else 0.0,
            "root": str(self.root),
            "ttl_days": (self.ttl_seconds / 86400.0) if self.ttl_seconds else None,
            "freeze": self.freeze,
        }

    def purge_expired(self) -> int:
        if self.ttl_seconds is None:
            return 0
        cutoff = time.time() - self.ttl_seconds
        n = 0
        for p in self.root.rglob("*.json"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
            except Exception:
                continue
        return n


_singleton: Optional[PageCache] = None
_singleton_lock = threading.Lock()


def get_page_cache() -> PageCache:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                cfg = CacheConfig.from_env()
                _singleton = PageCache(
                    cache_dir=cfg.cache_dir,
                    ttl_seconds=cfg.page_ttl_seconds,
                    freeze=cfg.freeze_cache,
                )
    return _singleton
