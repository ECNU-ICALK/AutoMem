from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2.exceptions import SecurityError

from automem.benchmarks.gaia import resolve_attachment_path
from automem.benchmarks.gaia import runner as gaia_runner
from automem.benchmarks.webwalkerqa import runner as webwalker_runner
from automem.evaluation.aggregation import load_task_results
from automem.evaluation.utils import (
    completed_task_result_stems,
    dataset_file_sha256,
    load_completed_task_results,
    require_complete_task_run,
    save_task_result,
    task_result_validation_error,
    task_identity_digest,
)
from automem.benchmarks.xbench_deepsearch import runner as xbench_runner
from automem.llm_utils import render_prompt
from automem.providers.prompt_support import _render_prompt
from automem.search import engine
from flashoagents.mm_tools import AudioInspectorTool, TextInspectorTool
from flashoagents.mm_tools_utils import MarkdownConverter, safe_extract_zip


VALID_TASK_IDENTITY = "0" * 64


class _Model:
    def __call__(self, messages, **kwargs):
        return SimpleNamespace(content="ok")


@pytest.mark.parametrize("name", ["../secret.txt", "/etc/passwd"])
def test_gaia_attachment_rejects_paths_outside_input_root(tmp_path, name):
    metadata = tmp_path / "data" / "metadata.jsonl"
    metadata.parent.mkdir()
    metadata.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_attachment_path(name, metadata)


def test_gaia_attachment_rejects_symlink_escape(tmp_path):
    metadata = tmp_path / "data" / "metadata.jsonl"
    metadata.parent.mkdir()
    metadata.write_text("", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = metadata.parent / "attachment.txt"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes"):
        resolve_attachment_path(link.name, metadata)


def test_task_result_filename_cannot_escape_or_follow_symlink(tmp_path):
    run_dir = tmp_path / "tasks"
    run_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"original": true}', encoding="utf-8")
    target = run_dir / "1.json"
    target.symlink_to(outside)

    written = save_task_result({"item_index": 1}, str(run_dir), "1.json")

    assert Path(written).is_file()
    assert not Path(written).is_symlink()
    assert outside.read_text(encoding="utf-8") == '{"original": true}'
    with pytest.raises(ValueError, match="Unsafe"):
        save_task_result({}, str(run_dir), "../escaped.json")


def test_resume_skip_ignores_error_corrupt_and_symlink_results(tmp_path):
    run_dir = tmp_path / "tasks"
    run_dir.mkdir()
    save_task_result(
        {
            "item_index": 1,
            "task_score": 1.0,
            "status": "success",
            "judge_unjudged": False,
            "task_identity": VALID_TASK_IDENTITY,
        },
        str(run_dir),
        "1.json",
    )
    save_task_result(
        {"item_index": 2, "task_score": 0.0, "status": "error"},
        str(run_dir),
        "2.json",
    )
    (run_dir / "3.json").write_text("not json", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text(
        '{"item_index":4,"task_score":1.0,"status":"success",'
        '"judge_unjudged":false,"task_identity":"' + VALID_TASK_IDENTITY + '"}',
        encoding="utf-8",
    )
    (run_dir / "4.json").symlink_to(outside)

    assert completed_task_result_stems(str(run_dir)) == {"1"}
    assert [row["item_index"] for row in load_completed_task_results(str(run_dir))] == [1]


@pytest.mark.parametrize(
    "payload",
    [
        {"item_index": 1, "task_score": 0.0, "judge_unjudged": False},
        {"item_index": 1, "task_score": 0.0, "status": "success"},
        {
            "item_index": 1,
            "task_score": 0.0,
            "status": "success",
            "grader_explanation": "Judge Response error",
        },
    ],
)
def test_current_task_result_contract_rejects_legacy_judge_checkpoints(payload):
    assert task_result_validation_error(payload) is not None


def test_resume_requires_filename_to_match_payload_index(tmp_path):
    run_dir = tmp_path / "tasks"
    run_dir.mkdir()
    payload = {
        "item_index": 2,
        "task_score": 1.0,
        "status": "success",
        "judge_unjudged": False,
        "task_identity": VALID_TASK_IDENTITY,
    }
    save_task_result(payload, str(run_dir), "1.json")

    assert completed_task_result_stems(str(run_dir)) == set()
    assert load_completed_task_results(str(run_dir)) == []


def test_completion_guard_rejects_duplicate_item_indices():
    payload = {
        "item_index": 1,
        "task_score": 1.0,
        "status": "success",
        "judge_unjudged": False,
        "task_identity": VALID_TASK_IDENTITY,
    }

    with pytest.raises(RuntimeError, match="duplicate_indices=1"):
        require_complete_task_run("xBench", [payload, dict(payload)], 2, [])


def test_aggregation_rejects_error_and_symlink_checkpoints(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "1.json").write_text(
        json.dumps(
            {
                "item_index": 1,
                "task_identity": VALID_TASK_IDENTITY,
                "task_score": 1.0,
                "status": "error",
                "judge_unjudged": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit success"):
        load_task_results(str(tasks))

    (tasks / "1.json").unlink()
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "item_index": 1,
                "task_identity": VALID_TASK_IDENTITY,
                "task_score": 1.0,
                "status": "success",
                "judge_unjudged": False,
            }
        ),
        encoding="utf-8",
    )
    (tasks / "1.json").symlink_to(outside)
    with pytest.raises(ValueError, match="non-symlink"):
        load_task_results(str(tasks))


def test_engine_rejects_checkpoint_from_another_dataset(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"task_id":"first"}\n', encoding="utf-8")
    second.write_text('{"task_id":"second"}\n', encoding="utf-8")
    first_sha = dataset_file_sha256(first)
    second_sha = dataset_file_sha256(second)
    tasks = tmp_path / "tasks"
    save_task_result(
        {
            "item_index": 1,
            "task_identity": task_identity_digest(first_sha, 1),
            "task_score": 1.0,
            "status": "success",
            "judge_unjudged": False,
        },
        str(tasks),
        "1.json",
    )

    assert engine._scan_task_result_indices(tasks, first_sha) == ({0}, [], [])
    assert engine._scan_task_result_indices(tasks, second_sha) == (
        set(),
        [],
        ["1.json"],
    )


def test_webwalker_schema_rejected_before_model_initialization(tmp_path, monkeypatch):
    infile = tmp_path / "webwalker.json"
    infile.write_text(
        '[{"question":"", "answer":"answer", "root_url":"https://example.com"}]',
        encoding="utf-8",
    )
    model_initialized = False

    def fail_model(*_args, **_kwargs):
        nonlocal model_initialized
        model_initialized = True
        raise AssertionError("model must not be initialized")

    monkeypatch.setattr(webwalker_runner, "OpenAIServerModel", fail_model)
    with pytest.raises(ValueError, match="row 1.*question"):
        webwalker_runner.main(
            SimpleNamespace(
                infile=str(infile),
                outfile=str(tmp_path / "results.jsonl"),
                runtime_config_json=None,
                seed=42,
            )
        )
    assert not model_initialized


def test_gaia_canonicalizes_trusted_temp_root_before_zip_extract(tmp_path):
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    workspace = gaia_runner._canonical_temporary_workspace(
        SimpleNamespace(name=str(alias))
    )
    archive = tmp_path / "attachment.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("inside.txt", "safe")

    extracted = safe_extract_zip(str(archive), str(workspace / "unpacked"))

    assert workspace == physical.resolve()
    assert [Path(path).read_text(encoding="utf-8") for path in extracted] == ["safe"]


def test_xbench_checkpoint_uses_index_and_rejects_duplicate_external_ids():
    assert xbench_runner._checkpoint_filename(
        {"task_id": "same", "item_index": 2}
    ) == "2.json"
    with pytest.raises(ValueError, match="duplicate task ids"):
        xbench_runner._validate_xbench_task_ids(
            [{"id": "same"}, {"id": "same"}]
        )


@pytest.mark.parametrize("renderer", [render_prompt, _render_prompt])
def test_custom_jinja_prompt_cannot_access_python_globals(renderer, monkeypatch):
    monkeypatch.setenv("AUTOMEM_RCE_MARKER", "must-not-render")
    payload = (
        "{{ cycler.__init__.__globals__.os.environ['AUTOMEM_RCE_MARKER'] }}"
    )

    with pytest.raises(SecurityError):
        renderer(payload, {})


def test_file_inspector_is_default_deny_and_confined_to_allowed_root(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(PermissionError, match="no allowed root"):
        TextInspectorTool(_Model(), 1000).forward(str(inside))

    inspector = TextInspectorTool(_Model(), 1000, allowed_roots=[str(allowed)])
    assert inspector.forward("inside.txt") == "inside"
    with pytest.raises(PermissionError, match="outside"):
        inspector.forward(str(outside))


def test_document_converter_rejects_remote_and_file_urls():
    converter = MarkdownConverter()

    with pytest.raises(ValueError, match="disabled"):
        converter.convert("http://127.0.0.1/private")
    with pytest.raises(ValueError, match="disabled"):
        converter.convert("file:///etc/passwd")


def test_audio_transcription_requires_explicit_mtu_pair(tmp_path, monkeypatch):
    monkeypatch.delenv("MTU_API_KEY", raising=False)
    monkeypatch.delenv("MTU_BASE_URL", raising=False)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    inspector = AudioInspectorTool(
        _Model(), 1000, allowed_roots=[str(tmp_path)]
    )

    with pytest.raises(RuntimeError, match="both MTU_API_KEY and MTU_BASE_URL"):
        inspector.transcribe_audio(str(audio))


def test_xbench_judge_failure_is_unjudged_and_not_ingested(monkeypatch):
    class _TaskModel:
        def reset_total_counts(self):
            return None

        def get_total_counts(self):
            return {}

    class _AgentFn:
        def write_memory_to_messages(self, include_system_prompt=False):
            return []

    class _Agent:
        def __init__(self, *args, **kwargs):
            self.agent_fn = _AgentFn()

        def __call__(self, question):
            return {"agent_result": "wrong", "agent_trajectory": []}

    class _BrokenJudge:
        def __call__(self, messages):
            raise ConnectionError("judge unavailable")

    class _Provider:
        manager = None

        def __init__(self):
            self.ingestions = 0

        def reset_experiment_metrics(self):
            return None

        def take_in_memory(self, trajectory, extract_plan=None):
            self.ingestions += 1
            return True, "ok"

    monkeypatch.setattr(xbench_runner, "OpenAIServerModel", lambda **kwargs: _TaskModel())
    monkeypatch.setattr(xbench_runner, "MMSearchAgent", _Agent)
    provider = _Provider()

    result = xbench_runner.process_item(
        {"id": "one", "prompt": "question", "answer": "gold", "__item_index": 1},
        {},
        8,
        "default",
        4,
        enable_memory_evolution=True,
        judge_model=_BrokenJudge(),
        shared_memory_provider=provider,
    )

    assert result["status"] == "success"
    assert result["judge_unjudged"] is True
    assert provider.ingestions == 0

    no_judge_result = xbench_runner.process_item(
        {"id": "two", "prompt": "question", "answer": "gold", "__item_index": 2},
        {},
        8,
        "default",
        4,
        enable_memory_evolution=True,
        judge_model=None,
        shared_memory_provider=provider,
    )
    assert no_judge_result["judge_unjudged"] is True
    assert provider.ingestions == 0


@pytest.mark.parametrize("member_name", ["../escape.txt", "/absolute.txt"])
def test_safe_zip_rejects_path_escape(tmp_path, member_name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr(member_name, "escape")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(str(archive), str(tmp_path / "output"))


def test_safe_zip_rejects_symlinks_and_high_compression_ratio(tmp_path):
    symlink_archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink_archive, "w") as zip_ref:
        zip_ref.writestr(link, "target")

    with pytest.raises(ValueError, match="symlink"):
        safe_extract_zip(str(symlink_archive), str(tmp_path / "symlink-output"))

    bomb_archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(
        bomb_archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_ref:
        zip_ref.writestr("large.txt", b"0" * (2 * 1024 * 1024))

    with pytest.raises(ValueError, match="compression ratio"):
        safe_extract_zip(str(bomb_archive), str(tmp_path / "bomb-output"))


def test_safe_zip_extracts_regular_members_into_destination(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr("nested/file.txt", "content")

    extracted = safe_extract_zip(str(archive), str(tmp_path / "output"))

    assert [Path(path).relative_to(tmp_path / "output").as_posix() for path in extracted] == [
        "nested/file.txt"
    ]
    assert Path(extracted[0]).read_text(encoding="utf-8") == "content"


def test_safe_zip_rejects_existing_or_symlink_destination(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr("file.txt", "content")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="new, non-symlink"):
        safe_extract_zip(str(archive), str(existing))

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-output"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="new, non-symlink"):
        safe_extract_zip(str(archive), str(link))
    assert not (outside / "file.txt").exists()


def test_safe_zip_rejects_symlink_in_destination_parent(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr("file.txt", "content")

    outside = tmp_path / "outside-parent"
    outside.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="parent path must not contain symlinks"):
        safe_extract_zip(str(archive), str(parent_link / "new-output"))
    assert not (outside / "new-output").exists()
