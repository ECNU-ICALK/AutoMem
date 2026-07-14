"""
Tag Vocabulary — Structured tag system for memory classification and retrieval.

Maintains a multi-dimensional tag vocabulary that can grow dynamically
when the LLM encounters tasks outside the initial vocabulary.

Three orthogonal dimensions:
  - task_domain:      What kind of task (aligns with existing task_type_tags)
  - cognitive_skill:  What reasoning skill is needed
  - risk_pattern:     What failure modes are likely (aligns with insight.failure_pattern)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# ============================================================
# Initial vocabulary (frozen core)
# ============================================================

INITIAL_VOCABULARY: Dict[str, List[str]] = {
    "task_domain": [
        # General task types
        "geographic_spatial", "numeric_computation", "text_extraction",
        "date_time", "lookup_factual", "code_execution", "data_table",
        "multimedia_analysis", "web_search", "scientific_technical",
        "legal_regulatory", "literary_artistic", "sports_statistics",
        "format_conversion", "multi_step_reasoning",
        # GAIA-specific task types (must match tags used in tips/insights prompts)
        "citation_verification", "paywall_navigation", "cross_source_verification",
        "exact_answer_matching", "entity_disambiguation",
    ],
    "cognitive_skill": [
        "boundary_verification", "unit_conversion", "source_validation",
        "entity_decomposition", "temporal_filtering", "aggregation",
        "format_compliance", "multi_hop_search", "disambiguation",
        "comparison", "causal_reasoning", "constraint_satisfaction",
    ],
    "risk_pattern": [
        # General failure patterns
        "wrong_entity", "wrong_value", "format_error", "incomplete_search",
        "tool_misuse", "logic_error", "hallucination", "timeout", "scope_error",
        # GAIA-specific failure patterns (must match insight.failure_pattern enum)
        "paywall_blocked", "exact_match_failure", "disambiguation_error",
        "unit_conversion_error",
    ],
}


class TagVocabulary:
    """Manages a multi-dimensional, dynamically extensible tag vocabulary."""

    def __init__(self, vocab: Dict[str, List[str]] = None):
        self._vocab: Dict[str, List[str]] = {}
        for dim, tags in (vocab or INITIAL_VOCABULARY).items():
            self._vocab[dim] = list(tags)

    # ---- Access ----

    def all_tags(self, dimension: str) -> List[str]:
        return list(self._vocab.get(dimension, []))

    def dimensions(self) -> List[str]:
        return list(self._vocab.keys())

    def flatten(self) -> Set[str]:
        result = set()
        for tags in self._vocab.values():
            result.update(tags)
        return result

    def contains(self, dimension: str, tag: str) -> bool:
        return tag in self._vocab.get(dimension, [])

    # ---- Dynamic expansion ----

    def add_tag(self, dimension: str, tag: str) -> bool:
        """Add a new tag. Returns True if actually added (not duplicate)."""
        if dimension not in self._vocab:
            self._vocab[dimension] = []
        if tag not in self._vocab[dimension]:
            self._vocab[dimension].append(tag)
            logger.info(f"Tag vocabulary expanded: {dimension} += '{tag}'")
            return True
        return False

    # ---- Persistence ----

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TagVocabulary":
        try:
            with open(path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            return cls(vocab=vocab)
        except FileNotFoundError:
            logger.info(f"No vocabulary file at {path}, using defaults")
            return cls()

    def to_dict(self) -> Dict[str, List[str]]:
        return {dim: list(tags) for dim, tags in self._vocab.items()}
