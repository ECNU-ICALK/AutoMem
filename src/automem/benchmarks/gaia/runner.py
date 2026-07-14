#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import json
import logging
import shutil
import tempfile
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from flashoagents.mm_tools import (
    AudioInspectorTool,
    TextInspectorTool,
    VisualInspectorTool,
    get_single_file_description,
    get_zip_description,
)
from flashoagents.models import OpenAIServerModel
from flashoagents.base_agent import MMSearchAgent
from automem.benchmarks.gaia import resolve_attachment_path
from automem.evaluation.io import read_jsonl, write_jsonl
from automem.evaluation.judging import judge_equivalence
from automem.memory_types import (
    MemoryType,
    TrajectoryData,
    get_provider_class,
)
from automem.config import get_memory_config, load_runtime_config
from automem.endpoints import resolve_openai_endpoint
from automem.evaluation.utils import (
    TaskTimer, TokenCounter, dataset_file_sha256, task_identity_digest,
    load_completed_task_results, save_task_result,
    generate_unified_report, enrich_result_with_metrics, create_run_directory,
    capture_memory_metrics,
    require_complete_task_run,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()  # override=False: let shell env vars take priority over .env


def _normalize_answer_for_judge_fallback(ans):
    """Conservative normalization for the exact-match judge fallback.

    Only used when the LLM judge returned judgement="error" (infrastructure
    failure, not a verdict): if the normalized prediction equals the
    normalized golden answer, the task is provably correct and must not be
    scored 0. Real artifact of the bug: gaia_evo_gpt51_gate_260623 round_1
    candidate_1 tasks/87.json had agent_result="100" == golden_answer="100"
    scored 0 on a judge error.
    """
    if ans is None:
        return ""
    if isinstance(ans, dict):
        ans = ans.get("answer", "")
    import re as _re
    s = str(ans).strip().lower()
    s = _re.sub(r"\s+", " ", s)
    return s.strip(" .。'\"")


def parse_task_indices(indices_str):
    """Parse task indices with optional level and ignore syntax."""
    if not indices_str:
        return None
    
    indices_str = indices_str.strip()
    
    # Check if using new level or ignore syntax
    if '[level' not in indices_str.lower() and '[ignore]' not in indices_str.lower():
        # Legacy mode: simple indices without level filter
        indices = set()
        parts = indices_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                try:
                    start, end = part.split('-')
                    start, end = int(start.strip()), int(end.strip())
                    if start > end:
                        raise ValueError(f"Invalid range: {part} (start > end)")
                    indices.update(range(start, end + 1))
                except ValueError as e:
                    logger.error(f"Invalid range format: {part}. Error: {e}")
                    raise
            else:
                try:
                    indices.add(int(part))
                except ValueError:
                    logger.error(f"Invalid number format: {part}")
                    raise
        
        return indices
    
    # New level/ignore syntax mode
    import re
    select_specs = []
    ignore_specs = []
    
    # Strategy: Find tags and their content in order
    # Pattern matches: [levelX][ignore], [levelX], or [ignore] followed by optional content
    
    # First pass: identify all tags with positions
    tag_pattern = r'\[level(\d+)\]|\[ignore\]'
    tags = []
    
    for match in re.finditer(tag_pattern, indices_str, re.IGNORECASE):
        if match.group(1):  # [levelX]
            tags.append({
                'type': 'level',
                'level': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        else:  # [ignore]
            tags.append({
                'type': 'ignore',
                'level': None,
                'start': match.start(),
                'end': match.end()
            })
    
    if not tags:
        raise ValueError(f"Invalid syntax: {indices_str}")
    
    # Second pass: combine adjacent level+ignore and extract indices
    i = 0
    while i < len(tags):
        tag = tags[i]
        
        # Check if this is a [levelX] followed immediately by [ignore]
        if (tag['type'] == 'level' and 
            i + 1 < len(tags) and 
            tags[i + 1]['type'] == 'ignore'):
            
            # Check if they are truly adjacent (NO space or very minimal)
            between_text = indices_str[tag['end']:tags[i + 1]['start']]
            # Only treat as combined if there's NO space at all
            if len(between_text) == 0:  # Truly adjacent, no space
                # This is [levelX][ignore]
                level_num = tag['level']
                is_ignore = True
                end_pos = tags[i + 1]['end']
                i += 2  # Skip both tags
            else:
                # They are separate (has space)
                level_num = tag['level']
                is_ignore = False
                end_pos = tag['end']
                i += 1
        elif tag['type'] == 'level':
            level_num = tag['level']
            is_ignore = False
            end_pos = tag['end']
            i += 1
        else:  # tag['type'] == 'ignore'
            level_num = None
            is_ignore = True
            end_pos = tag['end']
            i += 1
        
        # Extract indices part (from end_pos to next tag or end of string)
        if i < len(tags):
            next_start = tags[i]['start']
            indices_part = indices_str[end_pos:next_start].strip()
        else:
            indices_part = indices_str[end_pos:].strip()
        
        level_num = level_num if level_num else None
        indices_part = indices_part.strip()
        
        if not indices_part:
            # No indices specified, means all tasks of this level
            spec = {"level": level_num, "indices": None}
            if is_ignore:
                ignore_specs.append(spec)
            else:
                select_specs.append(spec)
        else:
            # Parse the indices for this level
            indices = set()
            parts = indices_part.split(',')
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                if '-' in part:
                    try:
                        start, end = part.split('-')
                        start, end = int(start.strip()), int(end.strip())
                        if start > end:
                            raise ValueError(f"Invalid range: {part} (start > end)")
                        indices.update(range(start, end + 1))
                    except ValueError as e:
                        level_str = f"level{level_num}" if level_num else "global"
                        logger.error(f"Invalid range format in {level_str}: {part}. Error: {e}")
                        raise
                else:
                    try:
                        indices.add(int(part))
                    except ValueError:
                        level_str = f"level{level_num}" if level_num else "global"
                        logger.error(f"Invalid number format in {level_str}: {part}")
                        raise
            
            if indices:  # Only add if we have valid indices
                spec = {"level": level_num, "indices": indices}
                if is_ignore:
                    ignore_specs.append(spec)
                else:
                    select_specs.append(spec)
    
    return {"select": select_specs, "ignore": ignore_specs}


def load_memory_provider(memory_type_str, model=None, runtime_config_path=None):
    """Load and initialize memory provider from type string"""
    if not memory_type_str:
        return None
    
    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        logger.error(f"Invalid memory type: {memory_type_str}")
        return None
    
    try:
        provider_class = get_provider_class(memory_type)
        config = get_memory_config(memory_type, runtime_config_path)
        if model is not None:
            try:
                # many providers expect a 'model' in config
                config["model"] = model
            except Exception:
                pass
        provider = provider_class(config=config)
        
        if not provider.initialize():
            logger.error(f"Failed to initialize memory provider: {memory_type_str}")
            return None
        
        logger.info(f"Memory provider loaded: {memory_type_str}")
        return provider
    except Exception as e:
        logger.error(f"Failed to load memory provider {memory_type_str}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_retrieved_memory_context(agent_trajectory):
    """Collapse per-step memory guidance into a task-level retrieved context."""
    seen = set()
    parts = []
    for step in agent_trajectory or []:
        if not isinstance(step, dict):
            continue
        guidance = (step.get("memory_guidance") or "").strip()
        if not guidance or guidance in seen:
            continue
        seen.add(guidance)
        parts.append(guidance)
    return "\n\n".join(parts)


def _canonical_temporary_workspace(workspace) -> Path:
    """Return the physical path for a trusted TemporaryDirectory root."""

    return Path(workspace.name).resolve(strict=True)


def process_item(item, model_config, summary_interval, prompts_type, max_steps,
                 memory_type_str=None, item_index=None, enable_memory_evolution=True,
                 judge_model=None, shared_memory_provider=None, extract_plan=None,
                 runtime_config_path=None):
    """Process a single GAIA task with timing and metrics tracking"""
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()
    # Enforce the task-level token cap (TASK_TOKEN_CAP) on the TASK agent only,
    # never on the meta models (which accumulate across the whole run).
    task_model._enforce_token_cap = True
    
    memory_provider = shared_memory_provider
    if memory_provider is not None:
        try:
            memory_provider.reset_experiment_metrics()
        except Exception:
            pass
        try:
            memory_provider.model = task_model
            if getattr(memory_provider, "manager", None) is not None:
                memory_provider.manager.llm_client = task_model
        except Exception:
            pass
    elif memory_type_str:
        memory_provider = load_memory_provider(
            memory_type_str, task_model, runtime_config_path
        )
    
    attachment_workspace = None
    attachment_file = ""
    allowed_roots = []
    source_name = item.get("file_name")
    if source_name:
        source_root_raw = item.get("_attachment_root")
        if not source_root_raw:
            raise ValueError("GAIA attachment is missing its validated input root")
        source_root = Path(source_root_raw).resolve(strict=True)
        source_path = Path(source_name).resolve(strict=True)
        try:
            source_path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("GAIA attachment escaped its validated input root") from exc
        if not source_path.is_file():
            raise ValueError("GAIA attachment is not a regular file")
        if source_path.stat().st_size > 250 * 1024 * 1024:
            raise ValueError("GAIA attachment exceeds the 250 MiB task limit")

        attachment_workspace = tempfile.TemporaryDirectory(prefix="automem_gaia_")
        # macOS exposes TemporaryDirectory through /var, which is a symlink to
        # /private/var. Canonicalize this trusted root before the ZIP helper's
        # strict ancestor-symlink check.
        workspace_path = _canonical_temporary_workspace(attachment_workspace)
        attachment_dir = workspace_path / "attachments"
        attachment_dir.mkdir()
        copied_path = attachment_dir / source_path.name
        shutil.copy2(source_path, copied_path)
        attachment_file = str(copied_path)
        allowed_roots = [str(workspace_path)]

    visual_tool = VisualInspectorTool(
        task_model, 100000, allowed_roots=allowed_roots
    )
    text_tool = TextInspectorTool(task_model, 100000, allowed_roots=allowed_roots)
    audio_tool = AudioInspectorTool(
        task_model, 100000, allowed_roots=allowed_roots
    )

    timer = TaskTimer()
    timer.start()

    question = item["Question"]
    golden_answer = item["Final answer"]
    task_id = item.get("task_id")
    level = item.get("Level", "unknown")
    original_question = question

    if attachment_file:
        if Path(attachment_file).suffix.lower() == ".zip":
            question += "\n\nTo solve the task above, you will have to use these attached files:\n"
            attachment_text = get_zip_description(
                attachment_file,
                question,
                visual_tool,
                text_tool,
                audio_tool,
                extract_dir=str(Path(attachment_file).parent / "unpacked"),
            )
        else:
            question += "\n\nTo solve the task above, you will have to use this attached file:"
            attachment_text = get_single_file_description(
                attachment_file, question, visual_tool, text_tool, audio_tool,
            )
        # Keep the task prompt deterministic and give the agent a path relative
        # to its configured workspace instead of leaking a random temp prefix.
        question += attachment_text.replace(
            f"{workspace_path}{os.sep}", ""
        )

    search_agent = MMSearchAgent(
        task_model,
        summary_interval=summary_interval,
        prompts_type=prompts_type,
        max_steps=max_steps,
        memory_provider=memory_provider,
        allowed_roots=allowed_roots,
    )

    try:
        result = search_agent(question)
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(
                f"Task agent failed before producing an answer: {result!r}"
            )
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        retrieved_memory_context = extract_retrieved_memory_context(trajectory)
        
        is_correct = False
        judgement = None
        judge_unjudged = not bool(judge_model)
        judge_fallback = None
        if judge_model:
            try:
                eval_res = judge_equivalence(
                    original_question,
                    golden_answer,
                    result.get("agent_result", {}),
                    model=judge_model,
                )
                judgement = eval_res.get("judgement")
                judgement_str = (judgement or "").strip().lower()
                is_correct = (judgement_str == "correct")
                if judgement_str == "error":
                    # "error" is a judge-infrastructure failure, not a verdict.
                    # Rescue provably-correct answers via normalized exact
                    # match; otherwise mark the task unjudged so it is not
                    # distilled into failure memory below.
                    pred_norm = _normalize_answer_for_judge_fallback(result.get("agent_result"))
                    gold_norm = _normalize_answer_for_judge_fallback(golden_answer)
                    if pred_norm and pred_norm == gold_norm:
                        is_correct = True
                        judgement = "correct"
                        judge_fallback = "exact_match"
                        logger.warning(
                            f"Judge error rescued by exact-match fallback for task {task_id}"
                        )
                    else:
                        judge_unjudged = True
            except Exception as e:
                logger.warning(f"Judgement failed: {e}")
                judge_unjudged = True
        
        # Compute GAIA file metadata once — used by both extraction (below)
        # AND persisted into task_result so the H-plan subclassifier reading
        # task_result later sees correct task_files (Codex Round 3 R3-4).
        gaia_level = item.get("Level", level if level else "unknown")
        gaia_file_name = ""
        gaia_task_files: list = []
        fname_raw = attachment_file
        if isinstance(fname_raw, str) and fname_raw:
            import os as _os
            gaia_file_name = _os.path.basename(fname_raw)
            ext = gaia_file_name.rsplit(".", 1)[-1].lower() if "." in gaia_file_name else ""
            if ext == "zip":
                # Codex R3-5: peek inside the archive to recover real
                # attachment types. zip files are common in multi-file GAIA
                # tasks; without introspection task_files=[file_zip] would
                # train extraction on generic zip advice instead of the
                # real modality (pdf / xlsx / image / audio / code).
                try:
                    import zipfile as _zf
                    with _zf.ZipFile(fname_raw, "r") as zf:
                        for member in zf.namelist():
                            if member.endswith("/"):
                                continue
                            inner_ext = member.rsplit(".", 1)[-1].lower() if "." in member else ""
                            if inner_ext:
                                tag = f"file_{inner_ext}"
                                if tag not in gaia_task_files:
                                    gaia_task_files.append(tag)
                except Exception as _e:
                    logger.debug(f"zip introspection failed for {fname_raw}: {_e}")
                # Always keep file_zip as a hint that aggregation was needed.
                if "file_zip" not in gaia_task_files:
                    gaia_task_files.append("file_zip")
            elif ext:
                gaia_task_files.append(f"file_{ext}")

        if memory_provider and enable_memory_evolution and judge_unjudged:
            # An unjudged task has no ground-truth outcome; ingesting it with
            # is_correct=False would distill a possibly-successful trajectory
            # into failure "lessons" and poison the pool.
            logger.warning(
                f"Skipping memory ingestion for task {task_id}: judge returned no verdict"
            )
        elif memory_provider and enable_memory_evolution:
            try:
                # Codex Q14-2 fix (2026-04-28): use the EXECUTOR query
                # (which includes attachment description and strategy
                # prefix) so the provider's `_provided_units_by_query`
                # cache lookup at take_in_memory time matches the key
                # used by provide_memory. With original_question, the
                # cache miss falls back to shared `_last_provided_units`
                # which is racy under shared concurrency and can
                # attribute another task's memories as unused/used.
                # Keep original_question available via metadata.
                trajectory_data = TrajectoryData(
                    query=question,
                    # Use the structured agent_trajectory (plan/action/obs steps),
                    # NOT agent_messages (chat format): the trajectory extractor
                    # expects the {name, tool_calls, obs} schema. Passing
                    # agent_messages silently yielded empty trajectories ->
                    # answer-only stub memory.
                    trajectory=trajectory,
                    result=result.get("agent_result"),
                    metadata={
                        "task_id": task_id,
                        "status": "success",
                        "is_correct": is_correct,
                        "original_question": original_question,
                        "full_query": question,
                        # GAIA-specific metadata (Codex CR2-5 + R3-4 + R3-5)
                        "Level": str(gaia_level),
                        "level": str(gaia_level),
                        "file_name": gaia_file_name,
                        # task_files derived (incl. zip introspection) so
                        # _build_template_context can pick it directly via
                        # the attachments path even when file_name is a zip.
                        "attachments": list(gaia_task_files),
                    }
                )
                success, msg = memory_provider.take_in_memory(trajectory_data, extract_plan=extract_plan)
                if success:
                    logger.debug(f"Memory ingested: {msg}")
                else:
                    logger.warning(f"Memory ingestion failed: {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed: {e}")
        
        token_counter = TokenCounter.from_model(task_model)
        
        task_result = {
            "agent_result": result.get("agent_result"),
            "judgement": judgement,
            "judge_unjudged": judge_unjudged,
            "judge_fallback": judge_fallback,
            "task_score": 1.0 if is_correct else 0.0,
            "success": bool(is_correct),
            "task_id": task_id,
            "item_index": item_index,
            "task_identity": item.get("_task_identity"),
            "level": level,
            # Codex Round 3 R3-4: persist Level/file_name/task_files so the
            # H-plan llm_subclassify reads them later. Without these fields
            # multimodal/file failures get bucketed as true_reasoning_error.
            "Level": str(gaia_level),
            "file_name": gaia_file_name,
            "task_files": list(gaia_task_files),
            "question": original_question,
            "full_query": question,
            "golden_answer": golden_answer,
            "status": "success",
            "agent_trajectory": trajectory,
            "agent_messages": agent_messages,
            "retrieved_memory_context": retrieved_memory_context,
            "memory_metrics": capture_memory_metrics(memory_provider),
        }
        
        timer.stop()
        enriched = enrich_result_with_metrics(task_result, timer, token_counter)
        if attachment_workspace is not None:
            attachment_workspace.cleanup()
        return enriched
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Exception occurred while processing question: {question[:100]}...\nError: {error_msg}")
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        # Infrastructure failures have no trustworthy task outcome. Persist
        # the diagnostic result and fail the run, but never distill the
        # partial trajectory into negative memory for a later retry.
        
        task_result = {
            "agent_result": None,
            "judgement": None,
            "judge_unjudged": False,
            "judge_fallback": None,
            "task_score": 0.0,
            "success": False,
            "status": "error",
            "error": str(e),
            "error_traceback": error_msg,
            "task_id": task_id,
            "item_index": item_index,
            "task_identity": item.get("_task_identity"),
            "level": level,
            "question": original_question,
            "full_query": question,
            "golden_answer": golden_answer,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
            "retrieved_memory_context": extract_retrieved_memory_context([]),
            "memory_metrics": capture_memory_metrics(memory_provider),
        }
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        enriched = enrich_result_with_metrics(task_result, timer, token_counter)
        if attachment_workspace is not None:
            attachment_workspace.cleanup()
        return enriched


def main(args):
    infile = Path(args.infile).expanduser()
    if not infile.is_file():
        raise FileNotFoundError(f"GAIA input file not found: {infile}")
    args.infile = str(infile)
    outfile = Path(args.outfile).expanduser()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile = str(outfile)

    runtime_config_path = getattr(args, "runtime_config_json", None)
    runtime_config = (
        load_runtime_config(runtime_config_path) if runtime_config_path else None
    )
    extract_plan = runtime_config["extract_plan"] if runtime_config else None

    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    random.seed(args.seed)
    dataset_sha256 = dataset_file_sha256(args.infile)
    
    _task_api_key, _task_api_base = resolve_openai_endpoint("TASK")
    model_config = {
        "model_id": args.model or os.environ.get("DEFAULT_MODEL", "gpt-5"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": args.token_budget,
        "api_key": _task_api_key,
        "api_base": _task_api_base,
    }
    
    model = OpenAIServerModel(**model_config)

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r') as f:
            raw = json.load(f)
            data = []
            for idx, it in enumerate(raw):
                if isinstance(it, dict):
                    it = dict(it)
                    it["_global_index"] = idx + 1
                    it["_task_identity"] = task_identity_digest(
                        dataset_sha256, idx + 1
                    )
                data.append(it)
    else:
        raw = read_jsonl(args.infile)
        data = []
        for idx, it in enumerate(raw):
            if isinstance(it, dict):
                    it = dict(it)
                    it["_global_index"] = idx + 1
                    it["_task_identity"] = task_identity_digest(
                        dataset_sha256, idx + 1
                    )
                    data.append(it)

    attachment_root = str(Path(args.infile).resolve(strict=True).parent)
    for item in data:
        if not isinstance(item, dict) or not item.get("file_name"):
            continue
        item["file_name"] = resolve_attachment_path(item["file_name"], args.infile)
        item["_attachment_root"] = attachment_root

    if getattr(args, 'level', None) and not args.task_indices:
        level_str = str(args.level).strip()
        before = len(data)
        filtered = []
        for it in data:
            lv = it.get("Level") if isinstance(it, dict) else it.get("task")
            if lv is None:
                continue
            if str(lv).strip() == level_str:
                filtered.append(it)
        data = filtered
        logger.info(f"Level filter applied: level={level_str}, kept {len(data)}/{before}")

    if args.task_indices:
        try:
            parsed = parse_task_indices(args.task_indices)
            
            if isinstance(parsed, set):
                data = [data[i-1] for i in sorted(parsed) if 0 < i <= len(data)]
                logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
            elif isinstance(parsed, dict):
                select_specs = parsed.get("select", [])
                ignore_specs = parsed.get("ignore", [])
                
                selected_tasks = []
                if select_specs:
                    for spec in select_specs:
                        level_num = spec["level"]
                        indices = spec["indices"]
                        
                        level_filtered = []
                        for it in data:
                            lv = it.get("Level") if isinstance(it, dict) else None
                            if lv is not None and str(lv).strip() == level_num:
                                level_filtered.append(it)
                        
                        if indices is None:
                            selected_tasks.extend(level_filtered)
                            logger.info(f"Selected all {len(level_filtered)} tasks from level{level_num}")
                        else:
                            for idx in sorted(indices):
                                array_idx = idx - 1
                                if 0 <= array_idx < len(level_filtered):
                                    task = level_filtered[array_idx]
                                    selected_tasks.append(task)
                            
                            actual_count = len([i for i in indices if 0 < i <= len(level_filtered)])
                            global_indices = [level_filtered[i-1].get("_global_index") for i in sorted(indices) if 0 < i <= len(level_filtered)]
                            logger.info(f"Selected {actual_count} tasks from level{level_num} (level-relative indices: {sorted(indices)}, "
                                      f"global indices: {global_indices})")
                    
                    logger.info(f"Total selected tasks: {len(selected_tasks)}")
                else:
                    selected_tasks = data[:]
                    logger.info(f"No select specs, starting with all {len(selected_tasks)} tasks")
                
                if ignore_specs:
                    tasks_to_ignore = []
                    for spec in ignore_specs:
                        level_num = spec["level"]
                        indices = spec["indices"]
                        
                        if level_num is None:
                            if indices is None:
                                tasks_to_ignore.extend(selected_tasks)
                                logger.info("Ignoring all selected tasks")
                            else:
                                for it in selected_tasks:
                                    global_idx = it.get("_global_index")
                                    if global_idx and global_idx in indices:
                                        tasks_to_ignore.append(it)
                                
                                actual_count = len(tasks_to_ignore) - len([t for t in tasks_to_ignore if t not in selected_tasks or t.get("_global_index") not in indices])
                                ignored_global_indices = sorted([t.get("_global_index") for t in tasks_to_ignore if t.get("_global_index") in indices])
                                logger.info(f"Ignoring {len([t for t in selected_tasks if t.get('_global_index') in indices])} tasks with global indices: {sorted(indices)} "
                                          f"(found: {ignored_global_indices})")
                        else:
                            level_filtered = []
                            for it in selected_tasks:
                                lv = it.get("Level") if isinstance(it, dict) else None
                                if lv is not None and str(lv).strip() == level_num:
                                    level_filtered.append(it)
                            
                            if indices is None:
                                tasks_to_ignore.extend(level_filtered)
                                logger.info(f"Ignoring all {len(level_filtered)} tasks from level{level_num}")
                            else:
                                for idx in indices:
                                    array_idx = idx - 1
                                    if 0 <= array_idx < len(level_filtered):
                                        task = level_filtered[array_idx]
                                        tasks_to_ignore.append(task)
                                
                                actual_count = len([i for i in indices if 0 < i <= len(level_filtered)])
                                global_indices = [level_filtered[i-1].get("_global_index") for i in sorted(indices) if 0 < i <= len(level_filtered)]
                                logger.info(f"Ignoring {actual_count} tasks from level{level_num} (level-relative indices: {sorted(indices)}, "
                                          f"global indices: {global_indices})")
                    
                    tasks_to_ignore_set = set(id(t) for t in tasks_to_ignore)
                    data = [t for t in selected_tasks if id(t) not in tasks_to_ignore_set]
                    logger.info(f"After applying ignore specs: {len(data)} tasks remaining")
                else:
                    data = selected_tasks
            else:
                raise ValueError(f"Unexpected return type from parse_task_indices: {type(parsed)}")
                
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing task indices: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Invalid --task_indices: {args.task_indices!r}") from e
        if not data:
            raise ValueError(
                f"--task_indices selected no GAIA tasks: {args.task_indices!r}"
            )
    elif args.sample_num is not None:
        data = data[:args.sample_num]
    
    data_to_run = data
    logger.info(f"Total data to process: {len(data_to_run)}")

    # --- Resume support: skip tasks that already have result files ---
    if args.direct_output_dir and os.path.isdir(args.direct_output_dir):
        completed = {
            row["item_index"]: row["task_identity"]
            for row in load_completed_task_results(args.direct_output_dir)
        }
        if completed:
            before = len(data_to_run)
            data_to_run = [
                item for item in data_to_run
                if completed.get(item.get("_global_index"))
                != item.get("_task_identity")
            ]
            skipped = before - len(data_to_run)
            logger.info(f"Resume: skipping {skipped} already-completed tasks, {len(data_to_run)} remaining")

    memory_name = ""
    if args.memory_provider:
        try:
            memory_name = MemoryType(args.memory_provider).value + "_"
        except ValueError:
            pass
    
    if args.direct_output_dir:
        run_dir = args.direct_output_dir
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Using direct output directory: {run_dir}")
    else:
        out_dir = os.path.dirname(args.outfile) or "."
        base_name = os.path.splitext(os.path.basename(args.outfile))[0]
        run_dir = create_run_directory(out_dir, base_name, memory_name)
        logger.info(f"Run directory created: {run_dir}")

    results = []
    file_lock = threading.Lock()
    effective_concurrency = args.concurrency
    shared_memory_provider = None
    if args.shared_memory_provider and args.memory_provider:
        if effective_concurrency != 1:
            logger.warning(
                f"--shared_memory_provider with concurrency={effective_concurrency} "
                f"enabled.  Provider methods are serialized by an internal RLock; "
                f"memory-accumulation order across concurrent tasks is nondeterministic "
                f"(per-task memory_metrics are thread-local since 2026-07-11)."
            )
        shared_memory_provider = load_memory_provider(
            args.memory_provider, model, runtime_config_path
        )
        if shared_memory_provider is None:
            raise RuntimeError("Failed to initialize shared memory provider")
    elif (
        args.memory_provider
        and str(args.memory_provider).lower() not in ("none", "null")
        and effective_concurrency > 1
    ):
        logger.warning(
            "NON-shared stateful provider at concurrency %d: every worker "
            "instance loads and whole-file-saves the SAME default storage "
            "path, so concurrent saves are last-writer-wins and silently "
            "drop other workers' memories. Use --shared_memory_provider "
            "(or --concurrency 1) for any stateful provider.",
            effective_concurrency,
        )

    def safe_write(result):
        """Thread-safe result saving"""
        with file_lock:
            idx = result.get("item_index")
            filename = f"{idx}.json" if idx is not None else None
            save_task_result(result, run_dir, filename)

    if args.memory_provider:
        if shared_memory_provider is not None:
            logger.info(f"Memory provider enabled: {args.memory_provider} (shared provider, concurrency={effective_concurrency})")
        else:
            logger.info(f"Memory provider enabled: {args.memory_provider} (each thread creates independent instance, using {effective_concurrency} workers)")

    if not data_to_run:
        logger.info("All tasks already completed — nothing to run.")

    future_errors = []
    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        summary_interval = random.randint(args.summary_interval - 1, args.summary_interval + 1)

        futures = [
            executor.submit(
                process_item,
                item,
                model_config,
                summary_interval,
                args.prompts_type,
                args.max_steps,
                args.memory_provider,
                (item.get("_global_index") if isinstance(item, dict) else None),
                args.enable_memory_evolution,
                args.judge_model,
                shared_memory_provider,
                extract_plan,
                runtime_config_path,
            ) for item in data_to_run
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    safe_write(result)
                    
                    metrics = result.get("metrics", {})
                    if result.get("status") == "success":
                        logger.info(f"Task done [{len(results)}/{len(futures)}]: {result['question'][:80]}... "
                                  f"| Time: {metrics.get('elapsed_time', 0):.1f}s | Tokens: {metrics.get('total_tokens', 0)}")
                    elif result.get("status") == "error":
                        logger.warning(f"Task error [{len(results)}/{len(futures)}]: {result['question'][:80]}... | Error: {result.get('error', 'Unknown')}")
            except Exception as exc:
                import traceback
                logger.error(f"Failed to get result from future: {traceback.format_exc()}")
                future_errors.append(str(exc))

    logger.info(f"Processing completed. Completed this run: {len(results)}")
    require_complete_task_run("GAIA", results, len(futures), future_errors)
    all_results = [
        row
        for row in load_completed_task_results(run_dir)
        if row["task_identity"]
        == task_identity_digest(dataset_sha256, row["item_index"])
    ]

    write_jsonl(args.outfile, all_results)
    logger.info(f"Results saved to {args.outfile}")
    
    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        all_results,
        report_path, 
        dataset_name="GAIA",
        has_levels=True,
        level_key="level"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal data generation')

    parser.add_argument('--infile', type=str, required=True, help='GAIA JSON/JSONL input path')
    parser.add_argument('--outfile', type=str, default="runs/benchmarks/gaia/results.jsonl", help='output path')
    parser.add_argument('--model', type=str, default=None, help='Task model id override')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None, 
                        help='Task indices to run. Supports: '
                             '1) Simple: "5", "1-10", "1,3,5-8,10" (no level filter), '
                             '2) Level syntax: "[level1]35-53" (level1 indices 35-53), "[level2]" (all level2), '
                             '"[level1]1,3,5 [level2] [level3]10-20" (combined), '
                             '3) Ignore syntax: "[ignore] 1,2" (ignore by global _global_index), '
                             '"[level1][ignore] 1,2" (ignore level1-relative indices 1,2), '
                             '"[level1] [ignore] 3,5,9" (select all level1, ignore global indices 3,5,9)')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--token_budget', type=int, default=8000, help='Model max completion tokens')
    parser.add_argument('--judge_model', type=str,
                        default=(os.getenv('DEFAULT_JUDGE_MODEL') or os.getenv('DEFAULT_MODEL') or 'gpt-5'),
                        help='Model used for immediate judgement')
    parser.add_argument('--memory_provider', type=str,
                        choices=[MemoryType.MODULAR.value], default=None,
                        help='Enable the AutoMem modular provider')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory). Default: True')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false',
                        help='Disable memory system evolution (skip take_in_memory)')
    parser.add_argument('--level', type=str, choices=['1','2','3'], default=None, help='Filter GAIA tasks by level before applying indices')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Direct output directory (skips timestamped nesting)')
    parser.add_argument('--shared_memory_provider', action='store_true',
                        help='Reuse one memory provider across the full standalone run; '
                             'concurrency>1 is allowed but memory accumulation order is nondeterministic')
    parser.add_argument('--runtime_config_json', type=str, default=None,
                        help='Structured RuntimeConfig JSON emitted by AutoMem search')

    args = parser.parse_args()
    
    main(args)
    
