#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import requests
import json
import time
import asyncio
import threading
from collections import deque
from .tools import Tool
from .models import OpenAIServerModel


# ---------------------------------------------------------------------------
# Serper 400-rate circuit breaker
#
# Serper returns HTTP 400 in two very different situations:
#   (a) malformed query — e.g. exact-phrase + site: + date combinations
#       that the API rejects on a per-query basis. The agent typically
#       recovers by reformulating. False-positive rate ~5-10%.
#   (b) account over quota — Serper switches to returning 400 for ALL
#       queries (non-standard; most APIs use 429/402). Saturates near
#       100% of misses.
#
# We can't tell them apart from a SINGLE response — they both have
# status=400 with no distinguishing body in our observations. The signal
# is in the RATE: if the trailing 400 fraction stays high over a window
# of recent calls, it's quota; otherwise it's per-query.
#
# Implementation: thread-safe sliding window over the last N Serper
# responses. Treat 400 as quota only when both (a) the window has at
# least MIN_CALLS samples (avoid early false positives) and (b) the
# 400-rate exceeds RATE_THRESHOLD. Otherwise return a soft per-query
# failure that lets the agent retry with a different formulation.
# ---------------------------------------------------------------------------
_SERPER_WINDOW_SIZE = 30
_SERPER_QUOTA_RATE_THRESHOLD = 0.40   # 40% of recent calls 400 → quota
_SERPER_QUOTA_MIN_CALLS = 8           # minimum samples before quota verdict
_serper_outcomes: deque = deque(maxlen=_SERPER_WINDOW_SIZE)
_serper_lock = threading.Lock()


def _record_serper_outcome(kind: str) -> Tuple[int, int]:
    """Append outcome ('ok' / '400' / 'other_err') to the rolling window
    and return (window_size, count_400)."""
    with _serper_lock:
        _serper_outcomes.append(kind)
        return len(_serper_outcomes), sum(1 for x in _serper_outcomes if x == "400")


def _is_400_quota_event() -> bool:
    """Decide whether the just-recorded 400 indicates account-level quota
    exhaustion (sustained high 400 rate) or a one-off malformed query."""
    with _serper_lock:
        n = len(_serper_outcomes)
        c400 = sum(1 for x in _serper_outcomes if x == "400")
    if n < _SERPER_QUOTA_MIN_CALLS:
        return False
    return (c400 / n) >= _SERPER_QUOTA_RATE_THRESHOLD


# ---------------------------------------------------------------------------
# Serper API-key pool + rotation (opt-in, for long deep-search / evolution
# runs that exceed a single key's quota). OPT-IN: only active when the env
# var SERPER_API_KEYS (comma-separated) is set with >1 key. When unset, the
# pool is just [SERPER_API_KEY] and rotation is a no-op -> behaviour is
# byte-for-byte identical to before (so GAIA / single-key runs are unaffected).
# On a quota signal the caller rotates to the next pool key and retries; only
# when ALL keys are exhausted does it raise TERMINATE_QUOTA.
# ---------------------------------------------------------------------------
_serper_key_rot_lock = threading.Lock()
_serper_key_idx = 0


def _serper_key_pool() -> list:
    multi = os.getenv("SERPER_API_KEYS", "")
    pool = [k.strip() for k in multi.split(",") if k.strip()] if multi.strip() else []
    single = os.getenv("SERPER_API_KEY", "") or ""
    if single and single not in pool:
        pool.insert(0, single)
    return pool


def _current_serper_key():
    pool = _serper_key_pool()
    if not pool:
        return os.getenv("SERPER_API_KEY")
    with _serper_key_rot_lock:
        return pool[_serper_key_idx % len(pool)]


def _rotate_serper_key() -> bool:
    """Advance to the next key in the pool. Returns True if a fresh (not-yet-
    tried) key is now active, False if the whole pool has been exhausted."""
    global _serper_key_idx
    pool = _serper_key_pool()
    if len(pool) <= 1:
        return False
    with _serper_key_rot_lock:
        if _serper_key_idx >= len(pool) - 1:
            return False  # already on / past the last key — pool exhausted
        _serper_key_idx += 1
        new_key = pool[_serper_key_idx]
    with _serper_lock:
        _serper_outcomes.clear()  # fresh 400-rate window for the new key
    logger.warning(
        "[serper] quota signal on key; rotated to pool key #%d (%s...)",
        _serper_key_idx, (new_key or "")[:8],
    )
    return True

logger = logging.getLogger(__name__)

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

def read_page_jina(url: str) -> str:
    """Read and return the content of a webpage using Jina reader."""
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Engine': 'browser',
        'X-Return-Format': 'markdown',
        "X-Remove-Selector": "header, .class, #id",
        "X-Retain-Images": "none",
        'X-Timeout': '10',
        'X-Token-Budget': '200000',
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=15)
        if response.status_code in (400, 401, 402, 403, 429):
            raise RuntimeError(
                f"TERMINATE_QUOTA: Jina API quota or rate limit exceeded (status={response.status_code})"
            )
        response.raise_for_status()
        # Some providers return 200 with quota message in body
        lower_body = (response.text or "").lower()
        if any(k in lower_body for k in ["quota", "rate limit", "payment required"]):
            raise RuntimeError("TERMINATE_QUOTA: Jina API quota or rate limit exceeded")
        return response.text
    except requests.RequestException as e:
        # If server returned a response with status
        status = getattr(getattr(e, 'response', None), 'status_code', None)
        if status in (400, 401, 402, 403, 429):
            raise RuntimeError(
                f"TERMINATE_QUOTA: Jina API quota or rate limit exceeded (status={status})"
            )
        return f"Error reading page: {str(e)}"

def read_page_crawl4ai(url: str) -> str:
    """Read and return the content of a webpage using crawl4ai."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig

        # Proxy support: egress network on this host is behind a transparent
        # filter; all public HTTPS must go through the HTTP CONNECT proxy.
        # playwright (crawl4ai backend) does NOT honour env HTTPS_PROXY on
        # its own - pass it explicitly via BrowserConfig(proxy=...).
        _proxy = (
            os.getenv("CRAWL_PROXY", "").strip()
            or os.getenv("HTTPS_PROXY", "").strip()
            or os.getenv("https_proxy", "").strip()
            or os.getenv("HTTP_PROXY", "").strip()
            or os.getenv("http_proxy", "").strip()
            or ""
        )
        _browser_cfg = BrowserConfig(proxy=_proxy) if _proxy else None

        async def _crawl():
            async with AsyncWebCrawler(config=_browser_cfg) as crawler:
                result = await crawler.arun(url=url)
                return result.markdown or ""
        
        # Run async function in sync context
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    markdown = loop.run_until_complete(_crawl())
                except ImportError:
                    # Fallback: create a new thread with new event loop
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _crawl())
                        markdown = future.result()
            else:
                markdown = loop.run_until_complete(_crawl())
        except RuntimeError:
            # No event loop, create a new one
            markdown = asyncio.run(_crawl())
        
        if not markdown:
            return f"Error reading page: No content extracted from {url}"
        return markdown
    except ImportError:
        return "Error: crawl4ai is not installed. Please install it with: pip install crawl4ai>=0.7.4"
    except Exception as e:
        return f"Error reading page with crawl4ai: {str(e)}"

def read_page(url: str, extraction_mode: str = "default") -> str:
    """
    Read and return the content of a webpage.

    Provider selection via WEB_ACCESS_PROVIDER environment variable:
    - 'jina' (default): Use Jina Reader API (requires JINA_API_KEY)
    - 'crawl4ai': Use crawl4ai library (free, open-source, no API key needed)

    Note: If both JINA_API_KEY and crawl4ai are configured, the provider
    specified in WEB_ACCESS_PROVIDER will be used. No automatic fallback.

    Cache (in_memory branch): wrapped by flashoagents.cache.PageCache when
    ENABLE_PAGE_CACHE=1 (default). Key = sha256(reader, mode, normalized_url).
    Set FREEZE_CACHE=1 to forbid network fetch on cache miss (returns empty).
    """
    provider = os.getenv("WEB_ACCESS_PROVIDER", "jina").lower()

    # Resolve which underlying reader will run; this is part of the cache key
    # because Jina vs crawl4ai may produce different markdown for the same URL.
    if provider == "crawl4ai":
        reader_id = "crawl4ai"
        fetch_fn = read_page_crawl4ai
    elif provider == "jina":
        reader_id = "jina"
        fetch_fn = read_page_jina
    else:
        import warnings
        warnings.warn(
            f"Invalid WEB_ACCESS_PROVIDER='{provider}'. "
            f"Valid values are 'jina' or 'crawl4ai'. Using 'jina' as default.",
            UserWarning,
        )
        reader_id = "jina"
        fetch_fn = read_page_jina

    from .cache import get_page_cache, env_bool

    use_cache = env_bool("ENABLE_PAGE_CACHE", True)
    cache = get_page_cache() if use_cache else None
    if cache is not None:
        cached = cache.get(url, reader=reader_id, mode=extraction_mode)
        if cached is not None:
            return cached
        if cache.freeze:
            cache.record_miss_frozen(url, reader=reader_id, mode=extraction_mode)
            return ""

    content = fetch_fn(url)

    if cache is not None and content:
        cache.set(url, reader=reader_id, mode=extraction_mode,
                  normalized_content=content, raw_content=None)
    return content

def web_search_google_serper(
    query: str, 
    filter_year: Optional[int] = None, 
    serp_num: int = 3, 
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API."""
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "location": "United States",
        "num": serp_num
    })
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }

    # while-loop (was: for attempt in range): a key-rotation on quota does NOT
    # consume a retry, so a long run can cycle through the whole SERPER_API_KEYS
    # pool before giving up. TERMINATE_QUOTA is raised only once every key is
    # exhausted. Single-key behaviour is unchanged (_rotate_serper_key()->False).
    attempt = 0
    while attempt < max_retries:
        headers['X-API-KEY'] = _current_serper_key()
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            sc = response.status_code
            # 401/402/403/429 are unambiguous account-level signals — quota
            # / billing / auth. Rotate to the next pool key; raise only if none.
            if sc in (401, 402, 403, 429):
                _record_serper_outcome("other_err")
                if _rotate_serper_key():
                    continue
                raise RuntimeError(
                    f"TERMINATE_QUOTA: Serper API quota or rate limit exceeded (status={sc}, all keys exhausted)"
                )
            # 400 is ambiguous — could be a malformed query (recoverable)
            # or sustained quota signal (Serper's non-standard quota code).
            # Use the rolling-window 400-rate to decide.
            if sc == 400:
                n, c400 = _record_serper_outcome("400")
                if _is_400_quota_event():
                    if _rotate_serper_key():
                        continue
                    raise RuntimeError(
                        f"TERMINATE_QUOTA: Serper API quota or rate limit "
                        f"exceeded (status=400, sustained {c400}/{n} recent "
                        f"calls = quota signal, all keys exhausted)"
                    )
                logger.info(
                    f"[serper] status=400 isolated (recent {c400}/{n} 400s); "
                    f"treating as malformed query, returning soft fail."
                )
                return [], (
                    "Bad query rejected by Serper (status=400). Try a simpler "
                    "reformulation without exact-phrase + site: combinations."
                )
            response.raise_for_status()
            _record_serper_outcome("ok")
            results = response.json()

            if isinstance(results, dict):
                err = results.get("error") or results.get("statusMessage") or ""
                if isinstance(err, str) and any(k in err.lower() for k in ["quota", "rate limit", "payment required"]):
                    if _rotate_serper_key():
                        continue
                    raise RuntimeError("TERMINATE_QUOTA: Serper API quota or rate limit exceeded (all keys exhausted)")

            if "organic" not in results or not results["organic"]:
                year_filter_msg = f" with year filter={filter_year}" if filter_year else ""
                return [], f"No results found for '{query}'{year_filter_msg}. Try a more general query."

            search_results = []
            for idx, page in enumerate(results["organic"], 1):
                search_results.append({
                    "idx": idx,
                    "title": page.get("title", "No title"),
                    "date": f"\nDate published: {page['date']}" if "date" in page else "",
                    "snippet": f"\n{page.get('snippet', 'No snippet')}",
                    "source": f"\nSource: {page.get('source', 'Unknown source')}",
                    "link": page.get('link', '#')
                })

            return search_results, ""

        except RuntimeError as e:
            if "TERMINATE_QUOTA" in str(e):
                raise
            attempt += 1
            if attempt >= max_retries:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
        except (requests.RequestException, json.JSONDecodeError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 402, 403, 429):
                _record_serper_outcome("other_err")
                if _rotate_serper_key():
                    continue
                raise RuntimeError(
                    f"TERMINATE_QUOTA: Serper API quota or rate limit exceeded (status={status}, all keys exhausted)"
                )
            if status == 400:
                n, c400 = _record_serper_outcome("400")
                if _is_400_quota_event():
                    if _rotate_serper_key():
                        continue
                    raise RuntimeError(
                        f"TERMINATE_QUOTA: Serper API quota or rate limit "
                        f"exceeded (status=400, sustained {c400}/{n} recent "
                        f"calls = quota signal, all keys exhausted)"
                    )
                return [], (
                    "Bad query rejected by Serper (status=400). Try a simpler "
                    "reformulation without exact-phrase + site: combinations."
                )
            attempt += 1
            if attempt >= max_retries:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)

    return [], "Unexpected error in web search"


def web_search_jina_deepsearch(
    query: str,
    filter_year: Optional[int] = None,
    serp_num: int = 5,
    max_retries: int = 2,
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Jina DeepSearch v1 via OpenAI-compatible API.

    Jina DeepSearch returns a synthesized answer with embedded source references.
    We parse the response into structured search results for compatibility with
    the existing WebSearchTool format.
    """
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."

    from openai import OpenAI
    import httpx

    api_key = os.getenv("JINA_API_KEY")
    api_base = os.getenv("JINA_API_BASE")
    if not api_key or not api_base:
        return [], "Jina DeepSearch requires both JINA_API_KEY and JINA_API_BASE"

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=30.0),
    )

    search_query = query
    if filter_year:
        search_query += f" {filter_year}"

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="jina-deepsearch-v1",
                messages=[{"role": "user", "content": search_query}],
                max_tokens=2000,
                stream=False,
            )
            content = resp.choices[0].message.content or ""

            # Return as a single comprehensive result
            search_results = [{
                "idx": 1,
                "title": f"DeepSearch: {query[:60]}",
                "date": "",
                "snippet": f"\n{content}",
                "source": "\nSource: Jina DeepSearch",
                "link": "#",
            }]
            return search_results, ""

        except Exception as e:
            err_str = str(e)
            if any(k in err_str.lower() for k in ["quota", "rate limit", "429", "402"]):
                raise RuntimeError(f"TERMINATE_QUOTA: Jina DeepSearch API quota exceeded: {err_str}")
            if attempt == max_retries - 1:
                return [], f"Jina DeepSearch failed after {max_retries} attempts: {err_str}"
            time.sleep(2)

    return [], "Unexpected error in Jina DeepSearch"


def web_search_duckduckgo(
    query: str,
    filter_year: Optional[int] = None,
    serp_num: int = 5,
    max_retries: int = 2,
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using DuckDuckGo (free, no API key needed)."""
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [], "duckduckgo_search package not installed. Run: pip install duckduckgo_search"

    for attempt in range(max_retries):
        try:
            results = DDGS().text(query, max_results=serp_num)
            if not results:
                return [], f"No results found for '{query}'."

            search_results = []
            for idx, r in enumerate(results, 1):
                search_results.append({
                    "idx": idx,
                    "title": r.get("title", "No title"),
                    "date": "",
                    "snippet": f"\n{r.get('body', 'No snippet')}",
                    "source": "\nSource: DuckDuckGo",
                    "link": r.get("href", "#"),
                })
            return search_results, ""

        except Exception as e:
            if attempt == max_retries - 1:
                return [], f"DuckDuckGo search failed: {str(e)}"
            time.sleep(1)

    return [], "Unexpected error in DuckDuckGo search"


# Search provider dispatcher
_SEARCH_PROVIDERS = {
    "serper": web_search_google_serper,
    "jina_deepsearch": web_search_jina_deepsearch,
    "duckduckgo": web_search_duckduckgo,
}


def web_search(
    query: str,
    filter_year: Optional[int] = None,
    serp_num: int = 5,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], str]:
    """Dispatch web search to the configured provider.

    Provider is selected by WEB_SEARCH_PROVIDER env var.
    Defaults to 'serper' for backward compatibility.

    Cache (in_memory branch): wrapped by flashoagents.cache.SearchCache when
    ENABLE_SEARCH_CACHE=1 (default). Key includes backend so different
    providers don't collide. Empty result lists are cached; errors are not.
    Set FREEZE_CACHE=1 to forbid external API on cache miss (returns []).
    """
    provider = os.getenv("WEB_SEARCH_PROVIDER", "serper").lower().strip()
    fn = _SEARCH_PROVIDERS.get(provider, web_search_google_serper)

    from .cache import get_search_cache, env_bool

    use_cache = env_bool("ENABLE_SEARCH_CACHE", True)
    cache = get_search_cache() if use_cache else None

    if cache is not None:
        cached = cache.get(
            query=query, backend=provider, serp_num=serp_num, filter_year=filter_year,
        )
        if cached is not None:
            return cached, ""
        if cache.freeze:
            cache.record_miss_frozen(query=query, backend=provider)
            return [], "FREEZE_CACHE: search cache miss while frozen"

    results, err = fn(query, filter_year=filter_year, serp_num=serp_num, max_retries=max_retries)

    # Cache only successful (no error) responses; quota/rate-limit errors are
    # transient and should not poison the cache. Empty result lists ARE cached
    # because "no hits" is a stable answer to a static GAIA-style query.
    if cache is not None and not err:
        cache.set(
            query=query,
            backend=provider,
            serp_num=serp_num,
            filter_year=filter_year,
            results_normalized=results,
        )
    return results, err


class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Retrieve relevant knowledge from Wikipedia and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "Provide a query string for the information you want to retrieve from Wikipedia."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "wiki_search"

    def forward(self, query: str) -> str:
        """Execute Wikipedia search and return formatted results."""
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
            "inprop": "url"
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                error_info = data['error']
                return f"Wikipedia API error: {error_info.get('code', 'unknown')} - {error_info.get('info', 'unknown')}"

            pages = data.get("query", {}).get("pages", {})
            results = []
            
            for page_id, page_info in pages.items():
                if int(page_id) < 0:  # Skip invalid pages
                    continue
                    
                title = page_info.get("title", "Unknown Title")
                extract = page_info.get("extract", "No extract available")
                page_url = page_info.get("fullurl", "No URL available")
                
                results.append(
                    f"[{title}]({page_url})\n"
                    f"Summary: {extract[:500]}{'...' if len(extract) > 500 else ''}"
                )

            return "\n\n".join(results) if results else f"No relevant information found for: {query}"
        
        except requests.Timeout:
            return "Request to Wikipedia API timed out. Please try again later."
        except requests.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

class WebSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query and return the search results."
    inputs = {
        "query": {
            "type": "string", 
            "description": "The web search query to perform."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.tool_name = "web_search"

    def forward(self, query: str) -> str:
        """Execute web search and return formatted results."""
        search_results, error_msg = web_search(query, serp_num=5)
        
        if error_msg:
            return error_msg
        
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"{result['idx']}. [{result['title']}]({result['link']})"
                f"{result['date']}{result['source']}\n"
                f"   {result['snippet'].strip()}"
            )
        
        return "\n\n".join(formatted_results) if formatted_results else "No search results found"

class CrawlPageTool(Tool):
    name = "crawl_page"
    description = "Access webpage using the provided URL and extract relevant content.  Please make full use of this tool to verify the accuracy of the searched content."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit."
        },
        "query": {
            "type": "string",
            "description": "The specific information to extract from the webpage."
        }
    }
    output_type = "string"
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__()
        self.tool_name = "crawl_page"
        self.model = model

    @staticmethod
    def truncate_text(text: str, max_length: int = 60000) -> str:
        """Truncate text to specified length."""
        return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

    def get_summary_prompt(self, query: str, url: str, content: str) -> str:
        """Generate prompt for content summarization."""
        return (
            f"Task: Extract all content from the web page that matches the search query.\n"
            f"Search Query: {query}\n\n"
            f"Web Page Content [url:{url}]:\n{content}\n\n"
            "Instructions:\n"
            "- Summarize all relevant content for the query (text, tables, lists) into concise points\n"
            "- If no relevant information exists, please straightly output 'No relevant information'\n"
            "- Keep the summary under 500 words"
        )

    def retry_predict(self, prompt: str, max_retries: int = 3) -> str:
        """Retry model prediction with exponential backoff."""
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.model(messages)
                if hasattr(response, 'content'):
                    content = response.content
                    return content.strip() if isinstance(content, str) else str(content)
                return str(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Content extraction failed: {str(e)}"
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        return "Content extraction failed after multiple attempts"

    def forward(self, url: str, query: str) -> str:
        """Crawl webpage and extract relevant content."""
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return "Invalid URL format. Must start with http:// or https://"
        
        page_content = read_page(url)
        if page_content.startswith("Error"):
            return page_content
        
        truncated_content = self.truncate_text(page_content)
        prompt = self.get_summary_prompt(query, url, truncated_content)
        
        return self.retry_predict(prompt)


class GitHubSearchTool(Tool):
    name = "github_search"
    description = (
        "Search GitHub issues, pull requests, or code using the GitHub REST API. "
        "Supports qualifiers like repo:owner/name, is:issue, is:closed, label:bug, sort:created, etc. "
        "Use this instead of web_search when you need to query GitHub issues, PRs, or code search."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "GitHub search query with qualifiers. "
                "Examples: 'repo:numpy/numpy is:issue is:closed label:Regression polynomial sort:created-asc', "
                "'repo:facebook/react is:pr is:merged sort:updated-desc'"
            ),
        },
        "search_type": {
            "type": "string",
            "description": "Type of search: 'issues' (issues & PRs), 'code', or 'repositories'. Default: 'issues'",
            "nullable": True,
        },
        "per_page": {
            "type": "integer",
            "description": "Number of results to return (1-30). Default: 10",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.token = os.environ.get("GITHUB_TOKEN", "")
        self.base_url = "https://api.github.com"

    def forward(self, query: str, search_type: str = "issues", per_page: int = 10) -> str:
        search_type = search_type or "issues"
        per_page = min(max(int(per_page or 10), 1), 30)

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        url = f"{self.base_url}/search/{search_type}"
        params = {"q": query, "per_page": per_page, "sort": "created", "order": "asc"}

        # Parse sort/order from query if specified
        if "sort:created-desc" in query:
            params["order"] = "desc"
            params["sort"] = "created"
        elif "sort:created-asc" in query:
            params["order"] = "asc"
            params["sort"] = "created"
        elif "sort:updated" in query:
            params["sort"] = "updated"

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 403:
                return "GitHub API rate limit exceeded. Set GITHUB_TOKEN in .env to increase limits."
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"GitHub API error: {e}"

        total = data.get("total_count", 0)
        items = data.get("items", [])

        if not items:
            return f"No results found (total_count={total}). Try adjusting your query qualifiers."

        results = [f"Found {total} results (showing {len(items)}):"]

        for i, item in enumerate(items, 1):
            if search_type == "issues":
                labels = ", ".join(l.get("name", "") for l in item.get("labels", []))
                state = item.get("state", "")
                created = item.get("created_at", "")[:10]
                updated = item.get("updated_at", "")[:10]
                results.append(
                    f"\n{i}. #{item.get('number','')} [{state}] {item.get('title','')}\n"
                    f"   URL: {item.get('html_url','')}\n"
                    f"   Labels: {labels or 'none'}\n"
                    f"   Created: {created} | Updated: {updated}\n"
                    f"   Body preview: {(item.get('body','') or '')[:200]}"
                )
            elif search_type == "code":
                results.append(
                    f"\n{i}. {item.get('path','')} in {item.get('repository',{}).get('full_name','')}\n"
                    f"   URL: {item.get('html_url','')}"
                )
            else:
                results.append(
                    f"\n{i}. {item.get('full_name','')} - {item.get('description','')[:100]}\n"
                    f"   URL: {item.get('html_url','')}"
                )

        return "\n".join(results)


class GitHubIssueTool(Tool):
    name = "github_issue_detail"
    description = (
        "Get detailed information about a specific GitHub issue or PR, "
        "including its timeline events (label additions, comments, etc.). "
        "Use this to find when a label was added to an issue."
    )
    inputs = {
        "repo": {
            "type": "string",
            "description": "Repository in 'owner/repo' format, e.g. 'numpy/numpy'",
        },
        "issue_number": {
            "type": "integer",
            "description": "The issue or PR number",
        },
        "include_timeline": {
            "type": "boolean",
            "description": "Whether to include timeline events (label additions, etc.). Default: true",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.token = os.environ.get("GITHUB_TOKEN", "")
        self.base_url = "https://api.github.com"

    def forward(self, repo: str, issue_number: int, include_timeline: bool = True) -> str:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        # Get issue details
        try:
            resp = requests.get(
                f"{self.base_url}/repos/{repo}/issues/{issue_number}",
                headers=headers, timeout=15,
            )
            if resp.status_code == 404:
                return f"Issue #{issue_number} not found in {repo}"
            resp.raise_for_status()
            issue = resp.json()
        except requests.RequestException as e:
            return f"GitHub API error: {e}"

        labels = ", ".join(l.get("name", "") for l in issue.get("labels", []))
        result = (
            f"Issue #{issue.get('number','')} [{issue.get('state','')}]: {issue.get('title','')}\n"
            f"URL: {issue.get('html_url','')}\n"
            f"Labels: {labels or 'none'}\n"
            f"Created: {issue.get('created_at','')}\n"
            f"Closed: {issue.get('closed_at','') or 'N/A'}\n"
            f"Updated: {issue.get('updated_at','')}\n"
            f"Body: {(issue.get('body','') or '')[:500]}"
        )

        # Get timeline events
        if include_timeline:
            try:
                headers_tl = dict(headers)
                headers_tl["Accept"] = "application/vnd.github.mockingbird-preview+json"
                resp_tl = requests.get(
                    f"{self.base_url}/repos/{repo}/issues/{issue_number}/timeline",
                    headers=headers_tl, params={"per_page": 100}, timeout=15,
                )
                resp_tl.raise_for_status()
                events = resp_tl.json()

                label_events = []
                for ev in events:
                    if ev.get("event") == "labeled":
                        label_name = ev.get("label", {}).get("name", "")
                        created = ev.get("created_at", "")
                        actor = ev.get("actor", {}).get("login", "")
                        label_events.append(f"  + Label '{label_name}' added on {created} by {actor}")
                    elif ev.get("event") == "unlabeled":
                        label_name = ev.get("label", {}).get("name", "")
                        created = ev.get("created_at", "")
                        label_events.append(f"  - Label '{label_name}' removed on {created}")

                if label_events:
                    result += "\n\nLabel Timeline:\n" + "\n".join(label_events)
                else:
                    result += "\n\nLabel Timeline: No label events found"

            except Exception as e:
                result += f"\n\nTimeline fetch failed: {e}"

        return result


__all__ = [
    "WikiSearchTool",
    "WebSearchTool",
    "CrawlPageTool",
    "GitHubSearchTool",
    "GitHubIssueTool",
]
