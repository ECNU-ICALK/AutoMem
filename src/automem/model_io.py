"""Small adapter for callable and OpenAI-compatible text models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextModelResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


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


def _response_usage(response: Any, client: Any) -> tuple[int, int]:
    raw_response = getattr(response, "raw", None) or response
    usage = getattr(raw_response, "usage", None)
    if usage is None:
        usage = getattr(response, "token_usage", None)
    if isinstance(usage, dict):
        input_tokens = int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        output_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
    elif usage is not None:
        input_tokens = int(
            getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0
        )
        output_tokens = int(
            getattr(
                usage,
                "completion_tokens",
                getattr(usage, "output_tokens", 0),
            )
            or 0
        )
    else:
        input_tokens = output_tokens = 0

    if input_tokens == 0 and output_tokens == 0:
        input_tokens = int(getattr(client, "last_input_token_count", 0) or 0)
        output_tokens = int(getattr(client, "last_output_token_count", 0) or 0)
    return input_tokens, output_tokens


def invoke_text_model(
    client: Any,
    *,
    model: str = "",
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> TextModelResult:
    """Invoke a configured model without consulting process environment."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(client, "chat"):
        if not model:
            raise ValueError("an OpenAI-compatible client requires an explicit model id")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif callable(client):
        response = client(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise TypeError("model must be callable or OpenAI-compatible")

    input_tokens, output_tokens = _response_usage(response, client)
    return TextModelResult(
        text=_response_text(response),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
