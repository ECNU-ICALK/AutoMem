from __future__ import annotations

from types import SimpleNamespace

import pytest

from flashoagents.mm_tools import AudioInspectorTool, VisualInspectorTool


class _VisionModel:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content="model-provided description")


def test_visual_inspector_reuses_configured_task_model(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    image = tmp_path / "image.png"
    image.write_bytes(b"not-decoded-by-the-tool")
    model = _VisionModel()

    result = VisualInspectorTool(model, 1000, allowed_roots=[str(tmp_path)]).forward(
        str(image), question="What is visible?"
    )

    assert result == "model-provided description"
    assert len(model.calls) == 1
    messages, kwargs = model.calls[0]
    assert messages[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert kwargs == {"max_tokens": 2000, "temperature": 0.0}


def test_visual_inspector_rejects_remote_images(tmp_path):
    model = _VisionModel()

    with pytest.raises(ValueError, match="local files only"):
        VisualInspectorTool(model, 1000, allowed_roots=[str(tmp_path)]).forward(
            "https://images.example.test/sample.png",
            question="Describe it",
        )

    assert model.calls == []


def test_audio_inspector_includes_the_requested_question(tmp_path, monkeypatch):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"test audio")
    model = _VisionModel()
    inspector = AudioInspectorTool(model, 1000, allowed_roots=[str(tmp_path)])
    monkeypatch.setattr(inspector, "transcribe_audio", lambda _path: "transcript")

    result = inspector.forward(str(audio), question="Which speaker disagrees?")

    assert result == "model-provided description"
    messages, _kwargs = model.calls[0]
    prompt = messages[1]["content"][0]["text"]
    assert "Question: Which speaker disagrees?" in prompt
    assert "{question}" not in prompt
