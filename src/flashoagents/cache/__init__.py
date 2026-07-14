"""Persistent caches for web_search and read_page (in_memory branch).

Architecture-independent by design. The cache key for a search depends
ONLY on (backend, normalized_query, serp_num, filter_year, extra_params).
The cache key for a page depends ONLY on (reader, mode, normalized_url).
No run name, no architecture, no memory provider, no agent state ever
reaches the key. That means: a query/url cached during run X is replayed
for free during run Y — even if Y uses a totally different memory
architecture, retrieval strategy, or trigger configuration.

The point: Serper / Jina / Playwright costs are paid ONCE for a given
external request, and every later experiment re-uses the snapshot.

Keys include backend/reader IDs so different providers' results don't collide.
Each entry records: original input, backend, timestamp, raw + normalized payload.

Configuration (env vars, all optional):
  ENABLE_SEARCH_CACHE     bool, default true
  ENABLE_PAGE_CACHE       bool, default true
  AUTOMEM_CACHE_DIR       path,  default platform user cache/AutoMem
                          Set this to a SHARED location across runs to keep
                          one canonical snapshot.
  CACHE_TTL_DAYS          float, default unset (no expiry)
                          per-cache override: SEARCH_CACHE_TTL_DAYS / PAGE_CACHE_TTL_DAYS
  FREEZE_CACHE            bool, default false
                          when true: cache miss does NOT call external backend;
                          returns empty result + logs `cache_miss_frozen` so
                          experiments can pin to a fixed snapshot.

Constraints (enforced by code):
  - We cache search RESULTS and page CONTENT only.
  - We never cache: model outputs, judge decisions, final answers, GAIA labels,
    agent reasoning, memory units, or anything keyed on the run.
"""
from .search_cache import SearchCache, get_search_cache
from .page_cache import PageCache, get_page_cache
from .config import CacheConfig, env_bool

__all__ = [
    "SearchCache",
    "PageCache",
    "get_search_cache",
    "get_page_cache",
    "CacheConfig",
    "env_bool",
]
