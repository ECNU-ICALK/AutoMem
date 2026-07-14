"""Fail-closed resolution for OpenAI-compatible API credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def resolve_openai_endpoint(
    role: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str]:
    """Resolve a key/base pair without sending a generic key to an implicit host.

    A role-specific base URL is accepted only with its matching role-specific
    key. Generic ``OPENAI_API_KEY`` falls back to the official endpoint unless
    the user explicitly supplies ``OPENAI_API_BASE`` or ``OPENAI_BASE_URL``.
    """

    env = os.environ if environ is None else environ
    prefix = str(role or "").strip().upper()
    role_key = env.get(f"{prefix}_API_KEY") if prefix else None
    role_base = env.get(f"{prefix}_API_BASE") if prefix else None
    if bool(role_base) != bool(role_key):
        missing = f"{prefix}_API_KEY" if role_base else f"{prefix}_API_BASE"
        raise ValueError(
            f"{prefix}_API_KEY and {prefix}_API_BASE must be configured together; "
            f"missing {missing}"
        )
    if role_base:
        return role_key, role_base

    generic_base = (
        env.get("OPENAI_API_BASE")
        or env.get("OPENAI_BASE_URL")
        or OPENAI_DEFAULT_BASE_URL
    )
    return env.get("OPENAI_API_KEY"), generic_base


__all__ = ["OPENAI_DEFAULT_BASE_URL", "resolve_openai_endpoint"]
