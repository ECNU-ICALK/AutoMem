from __future__ import annotations

from pathlib import Path

from flashoagents.cache.config import CacheConfig


def test_cache_dir_uses_explicit_automem_override(monkeypatch, tmp_path):
    configured = tmp_path / "shared-cache"
    monkeypatch.setenv("AUTOMEM_CACHE_DIR", str(configured))
    monkeypatch.setenv("AUTOMAS_CACHE_DIR", str(tmp_path / "retired"))

    assert CacheConfig.from_env().cache_dir == configured


def test_cache_dir_defaults_to_user_cache_not_package_tree(monkeypatch, tmp_path):
    xdg_cache = tmp_path / "xdg-cache"
    monkeypatch.delenv("AUTOMEM_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))

    resolved = CacheConfig.from_env().cache_dir

    assert resolved == xdg_cache / "automem"
    assert "site-packages" not in resolved.parts
    assert resolved != Path(__file__).resolve().parents[2] / ".cache"
