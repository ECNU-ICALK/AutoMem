"""Standalone LLM utility module.

Provides reusable helpers for prompt rendering, JSON parsing, and LLM
invocation so that individual providers / scripts do not have to duplicate
the same boilerplate.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

import jinja2
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def render_prompt(template_str: str, context: Dict[str, Any]) -> str:
    """Render a Jinja2 template string with the given context dict.

    Mirrors the logic in ``providers.prompt_support._render_prompt``:
    undefined variables are silently left empty rather than raising.
    """
    env = SandboxedEnvironment(undefined=jinja2.Undefined, autoescape=False)
    template = env.from_string(template_str)
    return template.render(**context)


def load_prompt(path: str) -> str:
    """Load a prompt template from a file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> Optional[Any]:
    """Multi-strategy JSON extraction from an LLM response.

    Strategies (tried in order):
      1. Direct ``json.loads`` on the stripped text.
      2. Extract content inside a ```json ... ``` fenced code block.
      3. Find the outermost ``[...]`` or ``{...}`` and parse that.

    Returns the parsed object, or ``None`` if all strategies fail.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: ```json ... ``` code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: first balanced JSON value via incremental decode (E9 fix)
    # Replaces the greedy regex `\{[\s\S]*\}` / `\[[\s\S]*\]` which captured
    # from the first `{` to the LAST `}`, swallowing trailing prose / sibling
    # JSON. JSONDecoder.raw_decode walks the text and returns the first
    # valid JSON value + its end position.
    #
    # Codex F-3 fix (2026-05-18): walk both '{' and '[' positions in
    # ORDER OF OCCURRENCE in the text, not "all { positions first then all [
    # positions". The earlier loop returned the first valid { even when an
    # outer [ array containing that { appeared earlier and was the intended
    # value — e.g. "Here are: [{...}, {...}]" was being parsed as the first
    # inner dict, dropping the list wrapper that `generate_candidates`
    # expected.
    decoder = json.JSONDecoder()
    n = len(text)
    idx = 0
    while idx < n:
        c = text[idx]
        if c == "{" or c == "[":
            try:
                obj, _end = decoder.raw_decode(text, idx)
                return obj
            except json.JSONDecodeError:
                pass
        idx += 1

    logger.warning("Failed to parse JSON from LLM response: %s...", text[:300])
    return None


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _prepare_prompt(prompt_path_or_text: str, template_vars: Dict[str, Any]) -> str:
    """Resolve the prompt source and render template variables."""
    if prompt_path_or_text.endswith(".txt") and os.path.isfile(prompt_path_or_text):
        template_str = load_prompt(prompt_path_or_text)
    else:
        template_str = prompt_path_or_text
    return render_prompt(template_str, template_vars)


def _invoke_model(model, filled_prompt: str, max_network_retries: int = 3) -> str:
    """Call the model and return the response text. Retries on network errors."""
    import time as _time
    messages = [
        {"role": "user", "content": [{"type": "text", "text": filled_prompt}]}
    ]
    for attempt in range(1, max_network_retries + 1):
        try:
            response = model(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            error_str = str(e).lower()
            is_network = any(k in error_str for k in [
                "connection", "timeout", "chunked read", "remote protocol",
                "peer closed", "reset by peer", "eof",
            ])
            if is_network and attempt < max_network_retries:
                wait = 5 * attempt
                logger.warning(f"Network error (attempt {attempt}/{max_network_retries}): {e}. Retrying in {wait}s...")
                _time.sleep(wait)
                continue
            raise


def call_llm_json(
    model,
    prompt_path_or_text: str,
    template_vars: Dict[str, Any],
    max_retries: int = 3,
    schema: Optional[Dict[str, Any]] = None,
    retry_with_feedback: bool = True,
) -> Dict[str, Any]:
    """Load prompt -> fill vars -> call LLM -> parse JSON -> return dict.

    If *prompt_path_or_text* ends with ``.txt`` and the file exists on disk,
    the file contents are loaded as the template.  Otherwise the string
    itself is treated as the template.

    Args:
        model: callable LLM (OpenAIServerModel or compatible).
        prompt_path_or_text: prompt file path (`.txt`) or raw template string.
        template_vars: Jinja2 context.
        max_retries: total attempts (incl. first).
        schema: optional schema dict. Keys are required field names; values are
            ``set`` (nested required keys), ``dict`` (recursive schema), or
            ``type`` (e.g. ``str``). When set, output is validated and a failure
            triggers a retry with feedback.
        retry_with_feedback: when True, retries append the previous (failed)
            output + error message to the prompt so the LLM can self-correct.

    Returns:
        Parsed JSON dict. On total failure, returns a stub dict
        ``{"_parse_failed": True, "_last_raw": ..., "_last_err": ...}``
        instead of raising — so callers can log + fall back without losing the
        raw LLM output.

    Fixes applied (2026-05-16):
        - B5: return stub dict instead of raising ValueError on exhaustion;
          preserves raw output for debugging.
        - B3/B4/B6: schema validation + feedback-driven retry.
        - E9: parse_json_response uses brace-balance walk (see parse_json_response).
    """
    filled = _prepare_prompt(prompt_path_or_text, template_vars)
    base_prompt = filled
    last_raw = ""
    last_err = ""

    for attempt in range(max_retries):
        last_raw = _invoke_model(model, filled)
        parsed = parse_json_response(last_raw)

        # Step 1: did parsing succeed?
        if parsed is None:
            last_err = "Output is not valid JSON (parse failed)."
        elif schema is not None:
            # Step 2: schema validation
            valid, err = _validate_schema(parsed, schema)
            if valid:
                return parsed
            last_err = err
        else:
            # No schema → accept any parsed dict / list
            return parsed

        # Decide whether to retry
        if attempt < max_retries - 1:
            logger.warning(
                "call_llm_json attempt %d/%d failed: %s",
                attempt + 1, max_retries, last_err,
            )
            if retry_with_feedback:
                filled = (
                    f"{base_prompt}\n\n"
                    f"═══ RETRY {attempt + 1} ═══\n"
                    f"Previous attempt failed validation:\n"
                    f"ERROR: {last_err}\n\n"
                    f"Previous output:\n{last_raw[:800]}\n\n"
                    f"Fix the issue. Output STRICTLY valid JSON matching the schema. "
                    f"NO prose, NO markdown code fences, NO comments."
                )

    # All retries exhausted — return stub dict (B5 fix: do NOT raise)
    logger.warning(
        "call_llm_json: all %d retries failed; last_err=%s; preserving raw output.",
        max_retries, last_err,
    )
    return {
        "_parse_failed": True,
        "_last_raw": last_raw[:1000],
        "_last_err": last_err,
    }


def _validate_schema(d: Any, schema: Dict[str, Any]) -> Tuple[bool, str]:
    """Lightweight JSON-schema validator used by ``call_llm_json``.

    Schema syntax:
        - {"key": str}            → require key, value is a non-empty string
        - {"key": int}            → require key, value is an int
        - {"key": dict}           → require key, value is a dict (any shape)
        - {"key": {...}}          → require key + recurse into nested schema
        - {"key": {"sub1", ...}}  → require key is a dict with these sub-keys

    Returns:
        (True, "") on success, (False, error_message) on first failure.
    """
    if not isinstance(d, dict):
        return False, f"top-level output is not a JSON object (got {type(d).__name__})"
    for k, expected in schema.items():
        if k not in d:
            return False, f"missing required key '{k}'"
        if isinstance(expected, dict):
            ok, err = _validate_schema(d[k], expected)
            if not ok:
                return False, f"'{k}': {err}"
        elif isinstance(expected, set):
            if not isinstance(d[k], dict):
                return False, f"'{k}' must be an object (got {type(d[k]).__name__})"
            missing = expected - set(d[k].keys())
            if missing:
                return False, f"'{k}' missing sub-keys: {sorted(missing)}"
        elif isinstance(expected, type):
            if not isinstance(d[k], expected):
                return False, f"'{k}' must be {expected.__name__} (got {type(d[k]).__name__})"
            if expected is str and not d[k].strip():
                return False, f"'{k}' must be a non-empty string"
    return True, ""


def call_llm_text(
    model,
    prompt_path_or_text: str,
    template_vars: Dict[str, Any],
    max_retries: int = 2,
) -> str:
    """Same workflow as :func:`call_llm_json` but returns the raw text.

    Useful for prompts whose output is plain text rather than structured JSON
    (e.g. the HyDE hypothesis generator).

    Retries up to *max_retries* times if the model returns an empty string.

    Raises:
        ValueError: If the response is empty after all retries.
    """
    filled = _prepare_prompt(prompt_path_or_text, template_vars)

    for attempt in range(1, max_retries + 1):
        raw = _invoke_model(model, filled)
        if raw and raw.strip():
            return raw.strip()
        logger.warning(
            "Empty LLM response (attempt %d/%d), retrying...", attempt, max_retries
        )

    raise ValueError(
        f"LLM returned empty text after {max_retries} retries."
    )


# ---------------------------------------------------------------------------
# Prompt contract validation
# ---------------------------------------------------------------------------

# Template vars required per prompt name
_REQUIRED_TEMPLATE_VARS = {
    "architecture_selection": [
        "benchmark_profile_json",
        "architecture_space_json",
        "allowed_edges_json",
        "hard_constraints_json",
        "history_summary_json",
        "resource_budget_json",
        "fitness_definition_json",
    ],
    "task_profiling": [
        "benchmark_name",
        "benchmark_description",
        "sample_tasks_json",
        "agent_capabilities",
        "architecture_space_summary",
    ],
    "feedback_analysis": [
        "benchmark_profile_json",
        "architecture_decision_json",
        "evaluation_report_json",
        "previous_prompt_text",
        "previous_revision_example_json",
        "improve_score",
    ],
}


def validate_prompt_contract(
    prompt_text: str,
    prompt_name: str,
) -> tuple:
    """Validate that a rewritten prompt preserves its structural contract.

    Checks:
      1. All required ``{{xxx}}`` template variables still present.
      2. IMMUTABLE_SCAFFOLD boundaries exist (if the prompt uses them).
      3. EDITABLE_POLICY boundaries exist (if the prompt uses them).
      4. Output JSON schema block is intact (contains ``selected_nodes`` etc.).

    Returns:
        (is_valid, errors): Tuple of bool and list of error strings.
    """
    errors = []

    # 1. Template variables
    required_vars = _REQUIRED_TEMPLATE_VARS.get(prompt_name, [])
    for var in required_vars:
        if "{{" + var + "}}" not in prompt_text:
            errors.append(f"Missing template variable: {{{{{var}}}}}")

    # 2. Scaffold boundaries
    has_scaffold_begin = "<!-- BEGIN_IMMUTABLE_SCAFFOLD -->" in prompt_text
    has_scaffold_end = "<!-- END_IMMUTABLE_SCAFFOLD -->" in prompt_text
    if has_scaffold_begin != has_scaffold_end:
        errors.append("IMMUTABLE_SCAFFOLD has mismatched BEGIN/END markers")

    # 3. Editable policy boundaries
    has_policy_begin = "<!-- BEGIN_EDITABLE_POLICY -->" in prompt_text
    has_policy_end = "<!-- END_EDITABLE_POLICY -->" in prompt_text
    if has_policy_begin != has_policy_end:
        errors.append("EDITABLE_POLICY has mismatched BEGIN/END markers")

    # 4. For architecture_selection, check schema block
    if prompt_name == "architecture_selection":
        for key in ["selected_nodes", "selected_edges", "architecture_config"]:
            if key not in prompt_text:
                errors.append(f"Output schema missing key: {key}")

    return len(errors) == 0, errors


def extract_editable_policy(prompt_text: str) -> str:
    """Extract the EDITABLE_POLICY block from a prompt."""
    import re as _re
    m = _re.search(
        r"<!-- BEGIN_EDITABLE_POLICY -->(.*?)<!-- END_EDITABLE_POLICY -->",
        prompt_text,
        _re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return ""


def replace_editable_policy(prompt_text: str, new_policy: str) -> str:
    """Replace only the EDITABLE_POLICY block in a prompt, preserving scaffold."""
    import re as _re
    pattern = r"(<!-- BEGIN_EDITABLE_POLICY -->)(.*?)(<!-- END_EDITABLE_POLICY -->)"
    replacement = r"\1\n" + new_policy + r"\n\3"
    result = _re.sub(pattern, replacement, prompt_text, flags=_re.DOTALL)
    return result
