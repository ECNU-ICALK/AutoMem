"""
Query Classifier — LLM-based query-to-tag mapping for structured retrieval.

Classifies a task query into structured tags across three dimensions
(task_domain, cognitive_skill, risk_pattern) using a lightweight LLM call.
Results are used by TagRetriever for structured matching against memory tags.
"""

import json
import logging
import re
from typing import Dict, List, Optional

from jinja2 import Template

from automem.resources import read_prompt_text

from .tag_vocabulary import TagVocabulary

logger = logging.getLogger(__name__)

def _load_prompt_template() -> str:
    return read_prompt_text("query_classify.txt")


def _parse_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response text, handling markdown fences."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class QueryClassifier:
    """Classifies task queries into structured tags using an LLM."""

    def __init__(self, model, vocabulary: TagVocabulary):
        """
        Args:
            model: An LLM model with __call__(messages=...) interface.
            vocabulary: TagVocabulary instance for tag validation and expansion.
        """
        self.model = model
        self.vocabulary = vocabulary
        self._template_str = _load_prompt_template()
        # Instance-level cache: query string → classification result dict
        # Avoids redundant LLM calls when the same query is classified multiple times.
        self._cache: Dict[str, Dict[str, List[str]]] = {}

    def classify(self, query: str) -> Dict[str, List[str]]:
        """
        Classify a task query into structured tags.

        Results are cached by query string to avoid redundant LLM calls when
        the same task query is encountered multiple times (e.g., during
        repeated provide_memory() calls within the same experiment run).

        Returns:
            Dict with keys: task_domain, cognitive_skill, risk_pattern.
            Each value is a list of tag strings.
        """
        # Check instance-level cache first (avoids repeated LLM calls per query)
        cached = self._cache.get(query)
        if cached is not None:
            return cached

        prompt = Template(self._template_str).render(
            query=query,
            task_domain_tags=", ".join(self.vocabulary.all_tags("task_domain")),
            cognitive_skill_tags=", ".join(self.vocabulary.all_tags("cognitive_skill")),
            risk_pattern_tags=", ".join(self.vocabulary.all_tags("risk_pattern")),
        )

        try:
            response = self.model(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
        except Exception as e:
            logger.warning(f"QueryClassifier LLM call failed: {e}")
            return {"task_domain": [], "cognitive_skill": [], "risk_pattern": []}

        # Extract text from response
        if hasattr(response, "content"):
            text = response.content
        elif isinstance(response, dict):
            text = response.get("content", "")
        else:
            text = str(response)

        result = _parse_json_from_text(text)
        if result is None:
            logger.warning(f"QueryClassifier failed to parse JSON from: {text[:200]}")
            return {"task_domain": [], "cognitive_skill": [], "risk_pattern": []}

        # Process new tags
        for new_tag_spec in result.get("new_tags", []):
            if ":" in str(new_tag_spec):
                dim, tag = str(new_tag_spec).split(":", 1)
                dim, tag = dim.strip(), tag.strip()
                if dim in ("task_domain", "cognitive_skill", "risk_pattern"):
                    self.vocabulary.add_tag(dim, tag)

        # Validate tags against vocabulary and return
        output = {}
        for dim in ("task_domain", "cognitive_skill", "risk_pattern"):
            raw_tags = result.get(dim, [])
            if not isinstance(raw_tags, list):
                raw_tags = [raw_tags] if raw_tags else []
            # Keep tags that are in vocabulary (including newly added ones)
            valid = [t for t in raw_tags if self.vocabulary.contains(dim, t)]
            output[dim] = valid

        # Store in cache before returning
        self._cache[query] = output
        return output
