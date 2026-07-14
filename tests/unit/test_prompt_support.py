from __future__ import annotations

from automem.memory_schema import MemoryUnit, MemoryUnitType
from automem.memory_types import TrajectoryData
from automem.providers.prompt_support import (
    _build_template_context,
    _parse_json_from_response,
    format_memory_unit,
)


def test_parse_json_response_supports_direct_and_fenced_payloads():
    assert _parse_json_from_response('{"kind": "tip"}') == {"kind": "tip"}
    assert _parse_json_from_response('result:\n```json\n[{"kind": "tip"}]\n```') == [
        {"kind": "tip"}
    ]


def test_template_context_preserves_chat_trajectory_and_attachment_tags():
    trajectory = TrajectoryData(
        query="Inspect the attached data",
        trajectory=[{"role": "assistant", "content": "Opened the workbook"}],
        result="done",
        metadata={
            "task_id": "task-1",
            "level": 2,
            "file_name": "outer.zip",
            "attachments": ["file_xlsx", "notes.csv", "file_xlsx"],
        },
    )

    context = _build_template_context(trajectory, is_correct=True)

    assert context["raw_trajectory"] == "[Msg 0 - assistant]\nOpened the workbook"
    assert context["task_files"] == ["file_zip", "file_xlsx", "file_csv"]
    assert context["has_files"] is True
    assert context["failure_reason"] is None


def test_format_memory_unit_retains_applicability_and_provenance():
    unit = MemoryUnit(
        id="tip-1",
        type=MemoryUnitType.TIP,
        content={
            "topic": "atomic writes",
            "principle": "replace files atomically",
            "micro_example": "write a temporary file before replace",
        },
        source_task_query="How should state be persisted?",
        use_when=["persistent state changes"],
        avoid_when=["read-only operations"],
    )

    rendered = format_memory_unit(unit, 0.42)

    assert rendered is not None
    assert "[TIP] atomic writes [Match: HIGH | raw=0.420]" in rendered
    assert "Apply when: persistent state changes" in rendered
    assert "Avoid when: read-only operations" in rendered
    assert "Source: How should state be persisted?" in rendered


def test_insight_failure_pattern_normalizes_to_controlled_vocabulary():
    from automem.memory_schema import (
        INSIGHT_FAILURE_PATTERNS,
        normalize_failure_pattern,
        split_extraction_output,
    )

    assert normalize_failure_pattern(" Wrong Entity ") == "wrong_entity"
    assert normalize_failure_pattern("tool-misuse") == "tool_misuse"
    assert normalize_failure_pattern("") == ""
    # Off-vocabulary values survive (information beats silence) but normalized.
    assert normalize_failure_pattern("Made Up Label") == "made_up_label"
    assert "made_up_label" not in INSIGHT_FAILURE_PATTERNS

    units = split_extraction_output(
        extraction_result={
            "failure_pattern": "Disambiguation Error",
            "root_cause_conclusion": "Committed to one candidate without cross-checking.",
            "corrective_strategy": "Cross-check candidates against an authoritative index.",
            "applicability": "queries with same-name candidates",
            "confidence_calibration": "medium",
        },
        unit_type=MemoryUnitType.INSIGHT,
        source_task_id="task-norm",
        source_task_query="who wrote it",
        task_outcome="failure",
    )
    assert len(units) == 1
    assert units[0].content["failure_pattern"] == "disambiguation_error"
    assert units[0].is_negative_example
