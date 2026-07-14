"""Centralised env-var parsing for cache config.

Single source of truth so search_cache and page_cache agree on the
default cache_dir, ttl, freeze flag, and enable flags.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _default_cache_dir() -> Path:
    explicit = os.getenv("AUTOMEM_CACHE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    xdg_cache = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "automem"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "AutoMem"
    if os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        base = Path(local_app_data).expanduser() if local_app_data else Path.home()
        return base / "AutoMem" / "Cache"
    return Path.home() / ".cache" / "automem"


def _ttl_seconds(per_cache_var: str) -> Optional[float]:
    """Resolve TTL, returning None for 'no expiry'.

    Per-cache var (e.g. SEARCH_CACHE_TTL_DAYS) wins; otherwise CACHE_TTL_DAYS.
    Empty string or unset → no expiry.
    """
    raw = os.getenv(per_cache_var, os.getenv("CACHE_TTL_DAYS", "")).strip()
    if not raw:
        return None
    try:
        days = float(raw)
    except ValueError:
        return None
    if days <= 0:
        return None
    return days * 86400.0


@dataclass(frozen=True)
class CacheConfig:
    enable_search_cache: bool
    enable_page_cache: bool
    cache_dir: Path
    search_ttl_seconds: Optional[float]
    page_ttl_seconds: Optional[float]
    freeze_cache: bool

    @classmethod
    def from_env(cls) -> "CacheConfig":
        cache_dir = _default_cache_dir()
        return cls(
            enable_search_cache=env_bool("ENABLE_SEARCH_CACHE", True),
            enable_page_cache=env_bool("ENABLE_PAGE_CACHE", True),
            cache_dir=cache_dir,
            search_ttl_seconds=_ttl_seconds("SEARCH_CACHE_TTL_DAYS"),
            page_ttl_seconds=_ttl_seconds("PAGE_CACHE_TTL_DAYS"),
            freeze_cache=env_bool("FREEZE_CACHE", False),
        )
