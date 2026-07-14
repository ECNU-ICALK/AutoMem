"""Single-call relevance selection and evidence-grounded context rendering."""

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from .policy import DEFAULT_RUNTIME_POLICY, RuntimePolicy

logger = logging.getLogger(__name__)


@dataclass
class CompositionResult:
    kept_indices: List[int] = field(default_factory=list)
    guidance: str = ""
    no_guidance: bool = False
    used_fallback: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class MemoryContextComposer:
    """Judge candidates and compose one cited, tentative reference block."""

    _SYSTEM = (
        "You compose optional reference material for an autonomous agent from "
        "past-task memory. Drop misleading, entity-mismatched, or non-actionable "
        "items. Keep transferable operation-level experience. Never copy a prior "
        "task's answer, named entity, date, file name, or numeric result. Use "
        "tentative language and cite every statement with candidate labels such as "
        "[M0]. If nothing is useful, set no_guidance to true. Return JSON only: "
        "{\"keep_ids\":[0],\"guidance\":\"... [M0]\",\"no_guidance\":false}."
    )

    def __init__(self, policy: RuntimePolicy = DEFAULT_RUNTIME_POLICY):
        self.policy = policy

    @staticmethod
    def _response_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = getattr(response, "content", "")
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        return str(content or "").strip()

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
    def _complete(
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> Any:
        if hasattr(client, "chat"):
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        if callable(client):
            return client(messages, max_tokens=max_tokens, temperature=0.0)
        raise TypeError("runtime model must be callable or OpenAI-compatible")

    @staticmethod
    def _parse(text: str) -> Optional[Dict[str, Any]]:
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _fallback(
        candidates: Sequence[Dict[str, Any]],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> CompositionResult:
        if not candidates:
            return CompositionResult(
                no_guidance=True,
                used_fallback=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        top = candidates[0]
        body = str(top.get("text", "") or "").strip()[:1200]
        if not body:
            return CompositionResult(
                no_guidance=True,
                used_fallback=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return CompositionResult(
            kept_indices=[0],
            guidance=f"[Past experience, for reference only] {body} [M0]",
            used_fallback=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def compose(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        client: Optional[Any],
        model: str,
    ) -> CompositionResult:
        if not candidates:
            return CompositionResult(no_guidance=True)
        if client is None or (not model and not callable(client)):
            return self._fallback(candidates)

        lines = [f"Task: {query[:1500]}", "", "Candidate memories:"]
        for index, candidate in enumerate(candidates):
            text = str(candidate.get("text", "") or "")
            lines.append(
                f"[M{index}] score={float(candidate.get('score', 0.0)):.4f} "
                f"{text[: self.policy.composer_max_candidate_chars]}"
            )
        try:
            response = self._complete(
                client,
                model,
                [
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                self.policy.composer_max_output_tokens,
            )
        except Exception as exc:
            logger.warning("memory context composer failed; using top memory: %s", exc)
            return self._fallback(candidates)

        input_tokens, output_tokens = self._usage(response)
        parsed = self._parse(self._response_text(response))
        if parsed is None:
            return self._fallback(
                candidates,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        if bool(parsed.get("no_guidance", False)):
            return CompositionResult(
                no_guidance=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        raw_ids = parsed.get("keep_ids", [])
        if not isinstance(raw_ids, list):
            return self._fallback(
                candidates,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        kept = []
        for raw in raw_ids:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(candidates) and index not in kept:
                kept.append(index)
            if len(kept) >= self.policy.max_injected_units:
                break
        guidance = str(parsed.get("guidance", "") or "").strip()
        if not kept or not guidance:
            return CompositionResult(
                no_guidance=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        cited = {int(value) for value in re.findall(r"\[M(\d+)\]", guidance)}
        if cited != set(kept):
            return self._fallback(
                candidates,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return CompositionResult(
            kept_indices=kept,
            guidance="[Past experience, for reference only] " + guidance,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
