"""Prompt extraction, parsing, embedding, and rendering helpers.

This module contains implementation support for ``ModularMemoryProvider``.
It deliberately has no provider lifecycle or configuration entry points.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import jinja2
from jinja2.sandbox import SandboxedEnvironment

from ..memory_schema import MemoryUnit, MemoryUnitType
from ..memory_types import TrajectoryData

logger = logging.getLogger(__name__)


PROMPT_TO_UNIT_TYPE = {
    "insight": MemoryUnitType.INSIGHT,
    "tip": MemoryUnitType.TIP,
    "trajectory": MemoryUnitType.TRAJECTORY,
    "workflow": MemoryUnitType.WORKFLOW,
    "shortcut": MemoryUnitType.SHORTCUT,
}

# Exactly the five public Encode choices. Auxiliary extraction prompts
# (entity/relation/planning) were removed 2026-07-14: they had no call path —
# graph stores run their own embedded extraction prompts.
PROMPT_FILE_NAMES = {
    "insight": "insights_prompt.txt",
    "tip": "tips_prompt.txt",
    "trajectory": "trajectory_prompt.txt",
    "workflow": "workflow_prompt.txt",
    "shortcut": "shortcut_prompt.txt",
}


def _load_embedding_model(model_name: str, cache_dir: str):
    """Load SentenceTransformer with local cache fallback."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, model_name.replace("/", "_"))
    hf_cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    hf_repo_dir = os.path.join(
        hf_cache_root, f"models--{model_name.replace('/', '--')}"
    )
    device = (
        os.environ.get("MEMORY_EMBEDDING_DEVICE")
        or os.environ.get("SENTENCE_TRANSFORMERS_DEVICE")
        or None
    )

    if device is None:
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = "cuda:0"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    def _make_model(model_path: str):
        return SentenceTransformer(model_path, device=device)

    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            return _make_model(local_path)
    except Exception as exc:
        logger.warning("Failed to load local embedding model: %s", exc)

    try:
        snapshots_dir = os.path.join(hf_repo_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshot_names = sorted(os.listdir(snapshots_dir), reverse=True)
            for snapshot_name in snapshot_names:
                snapshot_path = os.path.join(snapshots_dir, snapshot_name)
                if os.path.isfile(os.path.join(snapshot_path, "modules.json")):
                    logger.info(
                        "Loading embedding model from local Hugging Face cache: %s",
                        snapshot_path,
                    )
                    return _make_model(snapshot_path)
    except Exception as exc:
        logger.warning("Failed to load Hugging Face cached embedding model: %s", exc)

    model = _make_model(model_name)
    model.save(local_path)
    return model


def _parse_json_from_response(text: str):
    """Extract JSON from a model response using bounded fallback strategies."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    logger.warning("Failed to parse JSON from LLM response: %s...", text[:300])
    return None


def _message_content_to_text(content: Any) -> str:
    """Flatten OpenAI-style string or typed-part message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        output = []
        for part in content:
            if isinstance(part, dict):
                output.append(str(part.get("text", part.get("content", ""))))
            else:
                output.append(str(part))
        return "\n".join(part for part in output if part)
    return str(content) if content else ""


def _format_chat_messages(trajectory: list) -> str:
    """Format a trajectory captured as OpenAI-style chat messages."""
    parts = []
    for index, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        role = step.get("role", "unknown")
        content = _message_content_to_text(step.get("content", ""))
        if not content:
            continue
        if len(content) > 2000:
            content = content[:2000] + "\n... [truncated]"
        parts.append(f"[Msg {index} - {role}]\n{content}")
    return "\n\n".join(parts)


def _format_trajectory_text(trajectory_data: TrajectoryData) -> str:
    """Format structured agent steps, falling back to chat messages."""
    parts = []
    trajectory = trajectory_data.trajectory or []

    for index, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        step_name = step.get("name", "unknown")

        if step_name == "plan":
            plan_text = step.get("value", "") or ""
            if len(plan_text) > 2000:
                plan_text = plan_text[:2000] + "\n... [truncated]"
            parts.append(f"[Step {index} - Plan]\n{plan_text}")
        elif step_name == "action":
            tool_calls = step.get("tool_calls", []) or []
            observation = step.get("obs", "") or ""
            thinking = step.get("think", "") or ""
            tool_texts = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("arguments", {})
                if isinstance(tool_args, dict):
                    tool_args_text = json.dumps(tool_args, ensure_ascii=False)
                else:
                    tool_args_text = str(tool_args)
                tool_texts.append(f"  - {tool_name}({tool_args_text})")

            action_text = f"[Step {index} - Action]"
            if thinking:
                thinking_text = (
                    thinking[:500] + "..." if len(thinking) > 500 else thinking
                )
                action_text += f"\nThinking: {thinking_text}"
            if tool_texts:
                action_text += "\nTool Calls:\n" + "\n".join(tool_texts)
            if observation:
                observation_text = (
                    observation[:1500] + "\n... [truncated]"
                    if len(observation) > 1500
                    else observation
                )
                action_text += f"\nObservations:\n{observation_text}"
            parts.append(action_text)
        elif step_name == "summary":
            summary_text = step.get("value", "") or ""
            if len(summary_text) > 1000:
                summary_text = summary_text[:1000] + "\n... [truncated]"
            parts.append(f"[Step {index} - Summary]\n{summary_text}")

    if not parts and trajectory:
        return _format_chat_messages(trajectory)
    return "\n\n".join(parts)


def _build_template_context(
    trajectory_data: TrajectoryData, is_correct: bool
) -> Dict[str, Any]:
    """Build the stable Jinja template context for memory extraction."""
    metadata = trajectory_data.metadata or {}
    raw_trajectory = _format_trajectory_text(trajectory_data)

    failure_reason = None
    if not is_correct:
        golden = metadata.get("golden_answer", "")
        if golden:
            failure_reason = (
                f"Answer mismatch: agent answered '{trajectory_data.result}', "
                f"expected '{golden}'"
            )
        else:
            failure_reason = metadata.get("error", "Unknown failure")

    from automem.task_complexity import task_complexity as _task_complexity

    raw_level = metadata.get("level") or metadata.get("Level") or "unknown"
    task_level = str(raw_level).strip() or "unknown"
    task_complexity_label = _task_complexity(
        task=metadata,
        trajectory=getattr(trajectory_data, "trajectory", None),
    )

    file_name = metadata.get("file_name") or ""
    task_files: List[str] = []
    if isinstance(file_name, str) and file_name.strip():
        extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if extension:
            task_files.append(f"file_{extension}")
    if isinstance(metadata.get("attachments"), list):
        for attachment in metadata["attachments"]:
            if not isinstance(attachment, str):
                continue
            tag: Optional[str] = None
            if attachment.startswith("file_") and "." not in attachment:
                tag = attachment.lower()
            elif "." in attachment:
                extension = attachment.rsplit(".", 1)[-1].lower()
                tag = f"file_{extension}"
            if tag and tag not in task_files:
                task_files.append(tag)

    return {
        "task_query": trajectory_data.query,
        "is_success": is_correct,
        "raw_trajectory": raw_trajectory,
        "final_result": str(trajectory_data.result) if trajectory_data.result else "",
        "golden_answer": str(metadata.get("golden_answer", "")),
        "task_id": metadata.get("task_id", ""),
        "task_order": metadata.get("task_order", 0),
        "memory_guidance": None,
        "failure_reason": failure_reason,
        "reference_trajectory": None,
        "memory_count_before": metadata.get("memory_count_before", 0),
        "task_complexity": task_complexity_label,
        "task_level": task_level,
        "task_files": task_files,
        "has_files": bool(task_files),
    }


def _render_prompt(template_str: str, context: dict) -> str:
    """Render a Jinja prompt template with permissive missing values."""
    environment = SandboxedEnvironment(
        undefined=jinja2.Undefined,
        autoescape=False,
    )
    template = environment.from_string(template_str)
    return template.render(**context)


def format_memory_unit(unit: MemoryUnit, score: float) -> Optional[str]:
    """Format one memory unit as agent-facing guidance."""
    content = unit.content
    type_tag = unit.type.value.upper()
    if score >= 0.40:
        band = "HIGH"
    elif score >= 0.10:
        band = "MEDIUM"
    else:
        band = "VERY_LOW — IGNORE unless Apply-when matches THIS exact task"
    score_text = f"[Match: {band} | raw={score:.3f}]"

    when_block = ""
    if unit.use_when:
        use_when = "; ".join(
            str(tag).strip() for tag in unit.use_when if str(tag).strip()
        )
        if use_when:
            when_block += f"\n  Apply when: {use_when}"
    if unit.avoid_when:
        avoid_when = "; ".join(
            str(tag).strip() for tag in unit.avoid_when if str(tag).strip()
        )
        if avoid_when:
            when_block += f"\n  Avoid when: {avoid_when}"
    source = (unit.source_task_query or "").strip()
    if source:
        when_block += f"\n  Source: {source[:120]}"

    if unit.type == MemoryUnitType.TIP:
        topic = content.get("topic", "")
        principle = content.get("principle", "")
        example = content.get("micro_example", "")
        return (
            f"[TIP] {topic} {score_text}\n"
            "  → Use: Apply this principle when planning if Apply-when matches.\n"
            f"  Principle: {principle}\n"
            f"  Example: {example}"
            f"{when_block}"
        )

    if unit.type == MemoryUnitType.WORKFLOW:
        chain_type = content.get("chain_type", "")
        chain_tag = f" [{chain_type}]" if chain_type else ""
        parts = [
            f"[WORKFLOW]{chain_tag} {score_text}",
            "  → Use: Reference step ordering and tool sequence. Adapt arguments to current task; do NOT copy parameters.",
        ]
        for workflow_key in ("agent_workflow", "search_workflow"):
            steps = content.get(workflow_key, [])
            if not steps:
                continue
            parts.append(f"  {workflow_key}:")
            for step in steps:
                step_number = step.get("step", "?")
                action = step.get("action", step.get("query_formulation", ""))
                parts.append(f"    Step {step_number}: {action}")
        format_check = content.get("final_format_check", "")
        if format_check:
            parts.append(f"  Final format check: {format_check}")
        return "\n".join(parts) + when_block

    if unit.type == MemoryUnitType.INSIGHT:
        root_cause = content.get("root_cause_conclusion", "")
        mismatch = content.get("state_mismatch_analysis", "")
        corrective = content.get("corrective_strategy", "")
        detection = content.get("detection_signal", "")
        failure_pattern = content.get("failure_pattern", "")
        extra_tag = f" [{failure_pattern}]" if failure_pattern else ""
        parts = [
            f"[INSIGHT]{extra_tag} {score_text}",
            "  → Use: AVOID this failure pattern. Apply the corrective_strategy proactively when the detection_signal appears.",
            f"  Root cause: {root_cause}",
            f"  Mismatch: {mismatch}",
        ]
        if corrective:
            parts.append(f"  Corrective: {corrective}")
        if detection:
            parts.append(f"  Detection signal: {detection}")
        return "\n".join(parts) + when_block

    if unit.type == MemoryUnitType.TRAJECTORY:
        steps = content.get("steps", [])
        key_decision = content.get("key_decision", "")
        critical_observation = content.get("critical_observation", "")
        is_negative = bool(getattr(unit, "is_negative_example", False))
        tag = "[TRAJECTORY ⚠ NEGATIVE EXAMPLE]" if is_negative else "[TRAJECTORY]"
        if is_negative:
            use_hint = (
                "AVOID this path. The prior task ENDED IN FAILURE; treat "
                "key_decision as a NEGATIVE example (what NOT to do)."
            )
        else:
            use_hint = (
                "Use key_decision + critical_observation as ABSTRACT inspiration "
                "only. Do NOT copy tool calls — the specific path is intentionally "
                "hidden to prevent verbatim replay."
            )
        parts = [
            f"{tag} ({len(steps)} steps abstracted) {score_text}",
            f"  → Use: {use_hint}",
        ]
        if key_decision:
            parts.append(f"  Key decision: {key_decision}")
        if critical_observation:
            parts.append(f"  Critical observation: {critical_observation}")
        return "\n".join(parts) + when_block

    if unit.type == MemoryUnitType.SHORTCUT:
        name = content.get("name", "unnamed")
        description = content.get("description", "")
        precondition = content.get("precondition", "")
        return (
            f"[SHORTCUT] {name} {score_text}\n"
            "  → Use: Invoke as a parameterized macro when Apply-when matches; "
            "substitute placeholders with current values.\n"
            f"  Description: {description}\n"
            f"  Precondition: {precondition}"
            f"{when_block}"
        )

    return f"[{type_tag}] {score_text}\n  {unit.content_text()[:200]}{when_block}"
