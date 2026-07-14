"""GAIA task-type taxonomy and classifier.

Produces an 8-category coarse classification used by:
  - Extraction prompts (route to file-type-aware tip categories)
  - Attribution analysis (per-task-type failure breakdown)
  - Stratified sampling (ensure batch covers all categories)

Categories (covers 100% of GAIA validation; based on 165-task scan):

  spreadsheet    8.5%   .xlsx / .csv tabular analysis
  image_qa       6.1%   .png / .jpg / .jpeg / .gif visual question answering
  video_qa       6.7%   YouTube video content extraction (text-only Q + URL)
  multimodal_file 11%   .pdf / .mp3 / .zip / .docx / .pptx / .txt / .jsonld / .pdb / .py
  numeric_compute 10%   "how many / how much / what is the (sum|average|...)"
  code_reasoning 2%     Python / Unlambda / code output prediction
  pop_culture    3%     movies / songs / actors / albums (no file)
  web_research   52%    everything else: arxiv / wikipedia / general web QA

The classifier is intentionally heuristic — it's a coarse signal, not
ground truth. Tasks with file_name take the file-extension branch
unconditionally (file always dominates intent).
"""

from __future__ import annotations

import os
from typing import Any, Dict

# Ordered list (priority matters: file extensions checked first, then
# regex over question text).
TASK_CATEGORIES = [
    "spreadsheet",
    "image_qa",
    "video_qa",
    "multimodal_file",
    "numeric_compute",
    "code_reasoning",
    "pop_culture",
    "web_research",
    "general",        # non-GAIA / non-English tasks the GAIA keyword lists don't cover
]

# File extensions that mark a category. Anything not listed under a
# specific category but present in a task falls into multimodal_file.
_EXT_SPREADSHEET = {".xlsx", ".xls", ".csv", ".tsv"}
_EXT_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
_EXT_OTHER_FILE = {  # Goes to multimodal_file
    ".pdf", ".mp3", ".wav", ".zip", ".docx", ".pptx",
    ".txt", ".jsonld", ".pdb", ".py", ".json", ".xml",
    ".html", ".md", ".tex", ".bib",
}

# Lowercase question keywords for content-based routing (file-less tasks)
_KW_VIDEO = ("youtube", "youtu.be", "video", "vimeo")
_KW_NUMERIC = (
    "how many", "how much", "what is the number", "what is the count",
    "what is the total", "what is the sum", "what is the average",
    "what is the percentage", "what fraction", "what percent",
)
_KW_CODE = ("unlambda", "python code", "what does the code", "the following code",
            "the output of", "what would print")
_KW_POPCULTURE = (
    "movie", "film", "actor", "actress", "singer", "song",
    "album", "musician", "band", "tv show", "tv series",
)


def classify_gaia_task(task: Dict[str, Any]) -> str:
    """Classify a GAIA task into one of TASK_CATEGORIES.

    Args:
        task: a dict with at least 'Question' (str) and optionally
              'file_name' (str). Compatible with metadata.jsonl entries.

    Returns:
        One of TASK_CATEGORIES.
    """
    fn = task.get("file_name") or ""
    q = (task.get("Question") or task.get("question") or "").lower()

    # 1. File-extension branch (file always dominates intent)
    if fn:
        ext = os.path.splitext(fn)[1].lower()
        if ext in _EXT_SPREADSHEET:
            return "spreadsheet"
        if ext in _EXT_IMAGE:
            return "image_qa"
        if ext in _EXT_OTHER_FILE:
            return "multimodal_file"
        # Unknown extension still suggests multimodal
        return "multimodal_file"

    # 2. Text-only branch — check by keyword priority
    if any(k in q for k in _KW_VIDEO):
        return "video_qa"
    if any(k in q for k in _KW_CODE):
        return "code_reasoning"
    # Numeric must come before pop_culture (e.g. "how many albums")
    if any(k in q for k in _KW_NUMERIC):
        return "numeric_compute"
    if any(k in q for k in _KW_POPCULTURE):
        return "pop_culture"

    # 3. Default: web research (English GAIA-style queries). For non-English /
    #    non-GAIA benchmarks the GAIA keyword lists above don't apply, so avoid
    #    forcing a misleading GAIA category — fall back to "general" when the
    #    query is clearly outside the English-keyword regime (e.g. CJK text).
    if any("一" <= ch <= "鿿" for ch in q):
        return "general"
    return "web_research"


def category_distribution(tasks) -> Dict[str, int]:
    """Aggregate counts by category for a list of tasks."""
    out: Dict[str, int] = {c: 0 for c in TASK_CATEGORIES}
    for t in tasks:
        out[classify_gaia_task(t)] += 1
    return out


def stratified_sample_by_category(
    tasks, n: int, seed: int = 42, level_aware: bool = True,
):
    """Sample ``n`` task indices preserving category (and optionally Level) shares.

    Largest-remainder allocation: each category gets
    round(n * cat_share). The leftover slots from rounding go to
    categories with the largest remainders.

    Args:
        tasks: list of task dicts (metadata.jsonl entries).
        n: target sample size.
        seed: RNG seed.
        level_aware: if True, also stratify within each category by Level.
    """
    import random
    rng = random.Random(seed)

    # Build buckets: category -> [indices], optionally further by level.
    buckets: Dict[str, Dict[str, list]] = {}
    for idx, t in enumerate(tasks):
        cat = classify_gaia_task(t)
        lvl = str(t.get("Level", "?"))
        buckets.setdefault(cat, {}).setdefault(lvl, []).append(idx)

    # Compute per-category quota
    total = len(tasks)
    if total == 0 or n <= 0:
        return []
    n = min(n, total)

    # Largest-remainder allocation
    raw = [(cat, n * sum(len(v) for v in by_lvl.values()) / total)
           for cat, by_lvl in buckets.items()]
    quotas = [(cat, int(q), q - int(q)) for cat, q in raw]
    allocated = sum(q for _, q, _ in quotas)
    leftover = n - allocated
    quotas.sort(key=lambda x: -x[2])  # largest remainder first
    cat_quotas = {}
    for i, (cat, base, _) in enumerate(quotas):
        cat_quotas[cat] = base + (1 if i < leftover else 0)

    # Sample within each category (level-aware if requested)
    sampled: list = []
    for cat, k in cat_quotas.items():
        if k <= 0 or cat not in buckets:
            continue
        by_lvl = buckets[cat]
        if level_aware:
            # Distribute k across levels by share within this category
            cat_total = sum(len(v) for v in by_lvl.values())
            for lvl, idxs in by_lvl.items():
                lvl_k = round(k * len(idxs) / cat_total) if cat_total else 0
                lvl_k = min(lvl_k, len(idxs))
                if lvl_k > 0:
                    sampled.extend(rng.sample(idxs, lvl_k))
            # Fill remaining with random picks from any remaining indices
            already = set(sampled)
            remaining = [i for v in by_lvl.values() for i in v if i not in already]
            while len(sampled) < k and remaining:
                pick = rng.choice(remaining)
                sampled.append(pick)
                remaining.remove(pick)
        else:
            flat = [i for v in by_lvl.values() for i in v]
            sampled.extend(rng.sample(flat, min(k, len(flat))))

    return sorted(set(sampled))[:n]
