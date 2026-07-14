"""
ShortcutValidationOp — Tool-Manager generate-validate-register-reuse: VALIDATE stage.

Runs on on_insert. For each SHORTCUT unit being inserted, validates:
  - Has `input_schema`, `output_schema`, `test_invocation`, and a
    `action_sequence` where each step has a well-formed `tool_call`
  - All `tool_name`s come from a controlled vocabulary
  - Placeholders referenced in `action_sequence` are declared in `input_schema`
    (or are pre-defined role tokens like <COMPUTED_VALUE>, <RAW_OBSERVATION>)
  - Placeholders in `test_invocation` keys match `input_schema` names

Units passing validation get the tag `tool_valid` in `applicable_task_types`.
Units failing validation are kept (backward-compat) but without the tag; the
`shortcut_promotion` op only promotes units with `tool_valid`.

This is the "registration gate" — without the tag, a shortcut is treated as
plain memory; with the tag, it is first-class callable material.
"""

import re
import time
import logging
from typing import Any, Dict, List, Set

from ..base_op import BaseManageOp, OpResult, StorageCompatibility, TriggerType
from ...memory_schema import MemoryUnitType

logger = logging.getLogger(__name__)

# Controlled vocabulary of tool names that ever appear in our GAIA agent.
ALLOWED_TOOLS = {
    "web_search",
    "crawl_page",
    "inspect_file_as_text",
    "visual_inspector",
    "audio_inspector",
    "code_interpreter",
    "final_answer",
}

# Placeholders allowed WITHOUT being declared in input_schema (runtime-scope).
RUNTIME_PLACEHOLDERS = {
    "<COMPUTED_VALUE>", "<RAW_OBSERVATION>", "<SOURCE_URL>",
}

PLACEHOLDER_RE = re.compile(r"<[A-Z_][A-Z_0-9]*>")


def _collect_placeholders(text: str) -> Set[str]:
    return set(PLACEHOLDER_RE.findall(text or ""))


class ShortcutValidationOp(BaseManageOp):
    op_name = "shortcut_validation"
    op_group = "tool_manager"
    trigger_type = TriggerType.ON_INSERT
    storage_compatibility = StorageCompatibility.ALL
    requires_llm = False
    requires_embedding = False
    rl_action_id = 22

    _DEFAULT_CONFIG: Dict[str, Any] = {}

    def _validate_shortcut(self, unit) -> (bool, str):
        c = unit.content or {}
        if not isinstance(c, dict):
            return False, "content is not a dict"

        for k in ("name", "description", "action_sequence"):
            if k not in c:
                return False, f"missing required field {k!r}"

        # Input/output schema optional but strongly encouraged
        declared: Set[str] = set()
        schema = c.get("input_schema") or []
        if isinstance(schema, list):
            for s in schema:
                if isinstance(s, dict) and "name" in s:
                    declared.add(str(s["name"]))

        action_sequence = c.get("action_sequence") or []
        if not isinstance(action_sequence, list) or not action_sequence:
            return False, "action_sequence empty or wrong type"

        used_in_args: Set[str] = set()
        seen_intents = ""
        for i, step in enumerate(action_sequence):
            if not isinstance(step, dict):
                return False, f"step {i} not a dict"
            tc = step.get("tool_call")
            if not isinstance(tc, dict):
                # backward-compat: permit legacy executable_payload ONLY when
                # there's also no tool_call — but do NOT mark tool_valid
                if step.get("executable_payload"):
                    return False, f"step {i} uses legacy executable_payload (not tool_call)"
                return False, f"step {i} missing tool_call"
            tool_name = tc.get("tool_name", "")
            if tool_name not in ALLOWED_TOOLS:
                return False, f"step {i} tool_name {tool_name!r} not in allowed vocab"
            args = tc.get("args_pattern", "")
            if not isinstance(args, str):
                return False, f"step {i} args_pattern must be string"
            # Malformed <...> tokens: the collector regex only sees
            # <UPPER_SNAKE> names, so <Mixed_Case>/<file-path> used to sail
            # through completely unvalidated.
            for raw in re.findall(r"<[^<>\s]+>", args):
                if not PLACEHOLDER_RE.fullmatch(raw):
                    return False, (
                        f"step {i} malformed placeholder {raw!r} "
                        f"(must be <UPPER_SNAKE_CASE>)"
                    )
            # Placeholders in args must be declared or runtime-scope
            for ph in _collect_placeholders(args):
                if ph in RUNTIME_PLACEHOLDERS:
                    # Placeholder Invariant 7b: a runtime placeholder must be
                    # produced by an EARLIER step (mentioned in its intent) —
                    # consuming one before it exists makes the macro
                    # non-invocable downstream.
                    if ph not in seen_intents:
                        return False, (
                            f"step {i} consumes runtime placeholder {ph} not "
                            f"produced by any earlier step's intent"
                        )
                elif ph not in declared:
                    return False, f"step {i} placeholder {ph} not declared in input_schema"
                else:
                    used_in_args.add(ph)
            seen_intents += " " + str(step.get("intent", ""))

        # Invariant 7b, reverse direction: every declared input must be
        # referenced by at least one args_pattern (unused declared inputs
        # made macros advertise parameters they never consume).
        unused = declared - used_in_args - RUNTIME_PLACEHOLDERS
        if unused:
            return False, (
                f"declared inputs never used in any args_pattern: {sorted(unused)}"
            )

        # Placeholders in the trigger fields must also be declared (7b lists
        # precondition / use_when alongside args_pattern).
        for ph in _collect_placeholders(str(c.get("precondition") or "")):
            if ph not in declared and ph not in RUNTIME_PLACEHOLDERS:
                return False, f"precondition references undeclared placeholder {ph}"
        for trigger in (c.get("use_when") or []):
            for ph in _collect_placeholders(str(trigger)):
                if ph not in declared and ph not in RUNTIME_PLACEHOLDERS:
                    return False, f"use_when references undeclared placeholder {ph}"

        # test_invocation keys must EQUAL the declared inputs when present
        # (the old subset check let macros omit inputs and still validate).
        ti = c.get("test_invocation")
        if isinstance(ti, dict) and ti:
            ti_keys = set(str(k) for k in ti)
            if ti_keys != declared:
                return False, (
                    f"test_invocation keys {sorted(ti_keys)} != input_schema "
                    f"names {sorted(declared)}"
                )

        return True, "ok"

    def execute(self, context: Dict[str, Any]) -> OpResult:
        t0 = time.time()
        result = OpResult(op_name=self.op_name)

        try:
            all_units = self.store.get_all(active_only=False)
            shortcuts = [
                u for u in all_units if u.type == MemoryUnitType.SHORTCUT
            ]

            # Only (re)validate units that don't already carry the tool_valid tag
            candidates = [
                u for u in shortcuts
                if "tool_valid" not in u.applicable_task_types
                and "tool_invalid" not in u.applicable_task_types
            ]

            validated = 0
            rejected = 0
            reasons_sample: List[str] = []
            for u in candidates:
                ok, reason = self._validate_shortcut(u)
                if ok:
                    if "tool_valid" not in u.applicable_task_types:
                        u.applicable_task_types = list(u.applicable_task_types) + [
                            "tool_valid"
                        ]
                    self.store.update(u)
                    validated += 1
                else:
                    if "tool_invalid" not in u.applicable_task_types:
                        u.applicable_task_types = list(u.applicable_task_types) + [
                            "tool_invalid"
                        ]
                    self.store.update(u)
                    rejected += 1
                    if len(reasons_sample) < 3:
                        reasons_sample.append(reason)

            result.triggered = True
            result.units_modified = validated + rejected
            result.details = {
                "validated": validated,
                "rejected": rejected,
                "scanned": len(candidates),
                "reject_reasons_sample": reasons_sample,
            }
            result.duration_ms = (time.time() - t0) * 1000
            return result
        except Exception as e:
            logger.exception("ShortcutValidationOp failed: %s", e)
            result.error = str(e)
            result.duration_ms = (time.time() - t0) * 1000
            return result
