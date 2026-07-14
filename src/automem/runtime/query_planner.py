"""Literal-preserving dual-query planning."""

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Optional

from .policy import DEFAULT_RUNTIME_POLICY, RuntimePolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryPlan:
    literal: str
    abstract: str = ""
    used_fallback: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class QueryPlanner:
    """Build a semantic focus without ever discarding the literal query."""

    _PROMPT = (
        "Extract the reusable retrieval intent from the task below. Preserve the "
        "operation class and action verbs, but do not repeat task-specific answers. "
        "Return JSON only as {{\"abstract_query\": \"...\"}}.\n\n"
        "Task:\n{query}\n\nCurrent progress (may be empty):\n{context}"
    )

    def __init__(self, policy: RuntimePolicy = DEFAULT_RUNTIME_POLICY):
        self.policy = policy

    @staticmethod
    def _content(response: Any) -> str:
        try:
            value = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            value = getattr(response, "content", "")
        if isinstance(value, list):
            return " ".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in value
            ).strip()
        return str(value or "").strip()

    @staticmethod
    def _complete(client: Any, model: str, messages: list[dict[str, str]]) -> Any:
        if hasattr(client, "chat"):
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=160,
                temperature=0.0,
            )
        if callable(client):
            return client(messages, max_tokens=160, temperature=0.0)
        raise TypeError("runtime model must be callable or OpenAI-compatible")

    @staticmethod
    def _usage(response: Any) -> tuple[int, int]:
        raw_response = getattr(response, "raw", None) or response
        usage = getattr(raw_response, "usage", None)
        if usage is None:
            return 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    @staticmethod
    def _parse(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return ""
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return ""
        if not isinstance(parsed, dict):
            return ""
        return str(parsed.get("abstract_query", "") or "").strip()

    def plan(
        self,
        literal_query: str,
        *,
        context: str = "",
        client: Optional[Any] = None,
        model: str = "",
        existing_abstract: str = "",
    ) -> QueryPlan:
        literal = str(literal_query or "").strip()
        if existing_abstract:
            return QueryPlan(literal=literal, abstract=existing_abstract.strip())
        if not literal or client is None or (not model and not callable(client)):
            return QueryPlan(literal=literal, used_fallback=True)
        prompt = self._PROMPT.format(
            query=literal[: self.policy.planner_max_query_chars],
            context=str(context or "")[-2000:],
        )
        try:
            response = self._complete(
                client,
                model,
                [{"role": "user", "content": prompt}],
            )
            abstract = self._parse(self._content(response))
            input_tokens, output_tokens = self._usage(response)
        except Exception as exc:
            logger.warning("query planner failed; using literal query: %s", exc)
            abstract = ""
            input_tokens = output_tokens = 0
        return QueryPlan(
            literal=literal,
            abstract=abstract,
            used_fallback=not bool(abstract),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
