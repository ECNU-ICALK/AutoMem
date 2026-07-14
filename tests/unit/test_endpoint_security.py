from __future__ import annotations

import pytest

from automem.endpoints import OPENAI_DEFAULT_BASE_URL, resolve_openai_endpoint


def test_generic_openai_key_defaults_only_to_official_endpoint() -> None:
    key, base = resolve_openai_endpoint(environ={"OPENAI_API_KEY": "secret"})

    assert key == "secret"
    assert base == OPENAI_DEFAULT_BASE_URL
    assert "dashscope" not in base
    assert "dmxapi" not in base


def test_explicit_generic_pair_can_use_an_openai_compatible_endpoint() -> None:
    key, base = resolve_openai_endpoint(
        environ={
            "OPENAI_API_KEY": "compatible-key",
            "OPENAI_API_BASE": "https://llm.example.test/v1",
        }
    )

    assert (key, base) == ("compatible-key", "https://llm.example.test/v1")


def test_role_specific_base_cannot_capture_generic_key() -> None:
    with pytest.raises(ValueError, match="TASK_API_KEY"):
        resolve_openai_endpoint(
            "TASK",
            environ={
                "TASK_API_BASE": "https://third-party.example.test/v1",
                "OPENAI_API_KEY": "must-not-leak",
            },
        )


@pytest.mark.parametrize("role", ["JUDGE", "SEARCH", "DIAGNOSIS"])
def test_role_specific_key_cannot_use_a_generic_vendor_base(role: str) -> None:
    with pytest.raises(ValueError, match=rf"{role}_API_BASE"):
        resolve_openai_endpoint(
            role,
            environ={
                f"{role}_API_KEY": "role-secret",
                "OPENAI_API_KEY": "generic-key",
                "OPENAI_API_BASE": "https://task-vendor.example.test/v1",
            },
        )


def test_role_specific_pair_is_used_together() -> None:
    key, base = resolve_openai_endpoint(
        "JUDGE",
        environ={
            "JUDGE_API_KEY": "judge-key",
            "JUDGE_API_BASE": "https://judge.example.test/v1",
            "OPENAI_API_KEY": "generic-key",
        },
    )

    assert (key, base) == ("judge-key", "https://judge.example.test/v1")


def test_search_loader_rejects_role_base_without_role_key(monkeypatch) -> None:
    from argparse import Namespace

    from automem.search import engine

    monkeypatch.setenv("SEARCH_API_BASE", "https://search.example.test/v1")
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "generic-key-must-not-leak")

    with pytest.raises(ValueError, match="SEARCH_API_KEY"):
        engine.load_model(Namespace(model=None))
