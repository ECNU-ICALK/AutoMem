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
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from flashoagents.models import OpenAIServerModel
from flashoagents.base_agent import SearchAgent
from automem.evaluation.io import read_jsonl, write_jsonl
from automem.memory_types import MemoryType, TrajectoryData, get_provider_class
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

load_dotenv(override=False)  # env (e.g. a sourced overlay) WINS over .env file; was override=True which clobbered inherited OPENAI_API_BASE back to .env's value


def _validate_webwalkerqa_items(data):
    """Enforce the public dataset contract before any model is initialized."""

    required_fields = ("question", "answer", "root_url")
    for row_number, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"WebWalkerQA row {row_number} must be a JSON object")
        invalid = [
            field
            for field in required_fields
            if not isinstance(item.get(field), str) or not item[field].strip()
        ]
        if invalid:
            raise ValueError(
                f"WebWalkerQA row {row_number} has missing or empty required "
                f"field(s): {', '.join(invalid)}"
            )
        info = item.get("info")
        if info is None:
            item["info"] = {}
        elif not isinstance(info, dict):
            raise ValueError(f"WebWalkerQA row {row_number} info must be an object")


def _normalize_answer_for_judge_fallback(ans):
    """Conservative normalization for the exact-match judge fallback.

    Mirrors run_flash_searcher_mm_gaia._normalize_answer_for_judge_fallback:
    used only when the judge returned judgement="error" (infrastructure
    failure, not a verdict) to rescue provably-correct answers from being
    scored 0.
    """
    if ans is None:
        return ""
    if isinstance(ans, dict):
        ans = ans.get("answer", "")
    import re as _re
    s = str(ans).strip().lower()
    s = _re.sub(r"\s+", " ", s)
    return s.strip(" .。'\"")


def judge_webwalkerqa_answer(question, golden_answer, pred_answer, model="gpt-5"):
    """Judge if the predicted answer is correct for WebWalkerQA tasks."""
    from openai import OpenAI
    
    try:
        if isinstance(pred_answer, dict):
            pred_answer = pred_answer.get("answer", pred_answer)
    except Exception:
        pass
    
    if not pred_answer or (isinstance(pred_answer, str) and pred_answer.strip() == ''):
        return {
            "question": question,
            "judgement": "incorrect",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }
    
    prompt = f"""You are a general AI assistant. Based on the [Correct Answer] provided below, determine whether the [Response] to the [Original Question] is correct.

[Original Question]: {question}

[Correct Answer]: {golden_answer}

[Response]: {pred_answer}

Your judgment must follow this standard:
- Focus only on whether there are substantial differences between the [Response] and the [Correct Answer]
- Do not comment on the background of the question
- Do not attempt to resolve the problem again
- Only focus on judging whether the answers are consistent
- If the [Response] is consistent with the [Correct Answer], or within an acceptable small margin of error for numerical questions, judge as "correct"
- Otherwise (i.e., in cases of any inconsistency, ambiguity, non-equivalence, or incorrectly extracted answer), judge as "incorrect"

Output JSON format:
{{
  "judgement": "correct" or "incorrect"
}}"""

    import time as _time
    from automem.evaluation.judging import _parse_judge_response

    try:
        # Judge endpoint is independent of the task endpoint: prefer JUDGE_API_* so the
        # answer-grader (e.g. Qwen) can run on a different backend than the task agent
        # (e.g. gpt-5.4). Falls back to OPENAI_API_* when JUDGE_API_* is unset (legacy
        # single-endpoint behaviour), matching the xBench / LongMemEval runners.
        judge_api_key, judge_api_base = resolve_openai_endpoint("JUDGE")
        client = OpenAI(api_key=judge_api_key, base_url=judge_api_base)
    except Exception as e:
        logger.error(f"Judge client init failed: {e}")
        return {
            "question": question,
            "judgement": "error",
            "golden_answer": golden_answer,
            "pred_answer": pred_answer,
        }

    last_error = None
    for _attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fair judge for web navigation tasks. Focus on core answer correctness, not formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            result_text = response.choices[0].message.content.strip()

            verdict = _parse_judge_response(result_text)
            if verdict is not None:
                return {
                    "question": question,
                    "judgement": verdict,
                    "golden_answer": golden_answer,
                    "pred_answer": pred_answer,
                }
            # NO coercion to 'incorrect' here: an unusable judge output is a
            # judge-infrastructure failure, not a verdict. The old code
            # silently converted such failures into confident wrong scores.
            last_error = f"unparseable judge response: {result_text[:200]!r}"
        except Exception as e:
            last_error = str(e)
        if _attempt < 2:
            _time.sleep(1.5 * (_attempt + 1))

    logger.error(f"Error judging answer after 3 attempts: {last_error}")
    return {
        "question": question,
        "judgement": "error",
        "golden_answer": golden_answer,
        "pred_answer": pred_answer,
    }


WEBWALKERQA_PROMPT_TEMPLATE = """You are tasked with answering a question that requires navigating through a website to find the information.

Question: {question}

Starting URL: {root_url}

Please:
1. Start from the provided root URL
2. Navigate through the website to find the information needed
3. Use web search and page crawling tools to explore the site
4. Provide a clear and accurate answer based on what you find

Important: You MUST begin by accessing {root_url}
"""


def parse_task_indices(indices_str):
    """Parse index string like "5", "1-10" or "1,3,5-8,10" into a 1-based index set."""
    if not indices_str:
        return None
    
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


def process_item(
    item,
    model_config,
    summary_interval,
    prompts_type,
    max_steps,
    memory_type_str=None,
    item_index=None,
    enable_memory_evolution=True,
    judge_model=None,
    shared_memory_provider=None,
    extract_plan=None,
    runtime_config_path=None,
):
    """Process a single WebWalkerQA task with timing and metrics tracking"""
    _validate_webwalkerqa_items([item])
    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()
    
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
    
    timer = TaskTimer()
    timer.start()
    
    search_agent = SearchAgent(
        task_model, 
        summary_interval=summary_interval, 
        prompts_type=prompts_type, 
        max_steps=max_steps,
        memory_provider=memory_provider
    )

    question = item["question"]
    golden_answer = item["answer"]
    root_url = item["root_url"]
    info = item.get("info", {})
    
    domain = info.get("domain", "")
    difficulty = info.get("difficulty_level", "")
    lang = info.get("lang", "en")
    question_type = info.get("type", "")
    source_websites = info.get("source_website", [])
    golden_path = info.get("golden_path", [])
    
    enhanced_question = WEBWALKERQA_PROMPT_TEMPLATE.format(
        question=question,
        root_url=root_url
    )
    
    try:
        result = search_agent(enhanced_question)
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(
                f"Task agent failed before producing an answer: {result!r}"
            )
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        trajectory = result.get("agent_trajectory", [])
        
        is_correct = False
        judgement = None
        judge_unjudged = not bool(judge_model)
        judge_fallback = None
        if judge_model:
            try:
                eval_res = judge_webwalkerqa_answer(
                    question,
                    golden_answer,
                    result.get("agent_result", {}),
                    model=judge_model,
                )
                judgement = eval_res.get("judgement")
                judgement_str = (judgement or "").strip().lower()
                is_correct = (judgement_str == "correct")
                if judgement_str == "error":
                    # Judge infrastructure failed (not a verdict). Rescue
                    # provably-correct answers via normalized exact match;
                    # otherwise mark the task unjudged so it is not distilled
                    # into failure memory below.
                    pred_norm = _normalize_answer_for_judge_fallback(result.get("agent_result"))
                    gold_norm = _normalize_answer_for_judge_fallback(golden_answer)
                    if pred_norm and pred_norm == gold_norm:
                        is_correct = True
                        judgement = "correct"
                        judge_fallback = "exact_match"
                        logger.warning(
                            f"Judge error rescued by exact-match fallback for item {item_index}"
                        )
                    else:
                        judge_unjudged = True
            except Exception as e:
                logger.warning(f"Judgement failed: {e}")
                judge_unjudged = True

        if memory_provider and enable_memory_evolution and judge_unjudged:
            # An unjudged task has no ground-truth outcome; ingesting it with
            # is_correct=False would distill a possibly-successful trajectory
            # into failure "lessons" and poison the pool.
            logger.warning(
                f"Skipping memory ingestion for item {item_index}: judge returned no verdict"
            )
        elif memory_provider and enable_memory_evolution:
            try:
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
                        "item_index": item_index,
                        "status": "success",
                        "is_correct": is_correct,
                        "domain": domain,
                        "difficulty": difficulty,
                        "full_query": enhanced_question,
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
            "item_index": item_index,
            "task_identity": item.get("_task_identity"),
            "question": question,
            "enhanced_question": enhanced_question,
            "golden_answer": golden_answer,
            "root_url": root_url,
            "domain": domain,
            "difficulty": difficulty,
            "language": lang,
            "type": question_type,
            "source_websites": source_websites,
            "golden_path": golden_path,
            "status": "success",
            "agent_trajectory": trajectory,
            "agent_messages": agent_messages,
            "memory_metrics": capture_memory_metrics(memory_provider),
        }
        
        timer.stop()
        return enrich_result_with_metrics(task_result, timer, token_counter)
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Exception occurred while processing question: {question[:100]}...\nError: {error_msg}")
        
        try:
            agent_messages = search_agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
        except Exception:
            agent_messages = []
        
        # Do not turn API/tool/worker failures into task-level negative
        # examples. The invalid result remains diagnosable and retryable.
        
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
            "item_index": item_index,
            "task_identity": item.get("_task_identity"),
            "question": question,
            "enhanced_question": enhanced_question,
            "golden_answer": golden_answer,
            "root_url": root_url,
            "domain": domain,
            "difficulty": difficulty,
            "language": lang,
            "type": question_type,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
            "memory_metrics": capture_memory_metrics(memory_provider),
        }
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)


def main(args):
    infile = Path(args.infile).expanduser()
    if not infile.is_file():
        raise FileNotFoundError(f"WebWalkerQA input file not found: {infile}")
    args.infile = str(infile)
    outfile = Path(args.outfile).expanduser()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile = str(outfile)

    runtime_config_path = getattr(args, "runtime_config_json", None)
    runtime_config = (
        load_runtime_config(runtime_config_path) if runtime_config_path else None
    )
    extract_plan = runtime_config["extract_plan"] if runtime_config else None

    random.seed(args.seed)
    dataset_sha256 = dataset_file_sha256(args.infile)

    if args.infile.lower().endswith('.json'):
        with open(args.infile, 'r', encoding='utf-8') as f:
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

    _validate_webwalkerqa_items(data)
    logger.info(f"Loaded {len(data)} items from {args.infile}")

    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    task_api_key, task_api_base = resolve_openai_endpoint()
    model_config = {
        "model_id": args.model or os.environ.get("DEFAULT_MODEL", "gpt-5"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": args.token_budget,
        "api_key": task_api_key,
        "api_base": task_api_base,
    }
    model = OpenAIServerModel(**model_config)

    if args.difficulty:
        difficulty_filter = args.difficulty.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("difficulty_level", "").lower() == difficulty_filter
        ]
        logger.info(f"Difficulty filter applied: difficulty={difficulty_filter}, kept {len(data)}/{before}")

    if args.lang:
        lang_filter = args.lang.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("lang", "").lower() == lang_filter
        ]
        logger.info(f"Language filter applied: lang={lang_filter}, kept {len(data)}/{before}")

    if args.domain:
        domain_filter = args.domain.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("domain", "").lower() == domain_filter
        ]
        logger.info(f"Domain filter applied: domain={domain_filter}, kept {len(data)}/{before}")

    if args.question_type:
        type_filter = args.question_type.lower()
        before = len(data)
        data = [
            it for it in data 
            if isinstance(it, dict) and it.get("info", {}).get("type", "").lower() == type_filter
        ]
        logger.info(f"Type filter applied: type={type_filter}, kept {len(data)}/{before}")

    if args.task_indices:
        try:
            selected_indices = parse_task_indices(args.task_indices)
            data = [data[i-1] for i in sorted(selected_indices) if 0 < i <= len(data)]
            logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing task indices: {e}")
            raise ValueError(f"Invalid --task_indices: {args.task_indices!r}") from e
        if not data:
            raise ValueError(
                f"--task_indices selected no WebWalkerQA tasks: {args.task_indices!r}"
            )
    elif args.sample_num is not None:
        data = data[:args.sample_num]
        logger.info(f"Limited to first {args.sample_num} tasks")
    
    data_to_run = data
    logger.info(f"Total data to process: {len(data_to_run)}")

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
            # Relaxed from a hard error to a warning, matching the xBench runner:
            # the modular provider tolerates concurrent access (lock-guarded), so
            # the architecture-search evaluation can run candidates at concurrency>1
            # for speed, accepting some memory-accumulation ordering nondeterminism.
            logger.warning(
                "--shared_memory_provider with --concurrency %d (>1): shared pool is "
                "accessed concurrently; results may be mildly nondeterministic.",
                effective_concurrency,
            )
        shared_memory_provider = load_memory_provider(
            args.memory_provider, model, runtime_config_path
        )
        if shared_memory_provider is None:
            raise RuntimeError("Failed to initialize shared memory provider")

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
            if (
                effective_concurrency > 1
                and str(args.memory_provider).lower() not in ("none", "null")
            ):
                logger.warning(
                    "NON-shared stateful provider at concurrency %d: every worker "
                    "instance loads and whole-file-saves the SAME default storage "
                    "path, so concurrent saves are last-writer-wins and silently "
                    "drop other workers' memories. Use --shared_memory_provider "
                    "(or --concurrency 1) for any stateful provider.",
                    effective_concurrency,
                )
    
    # Skip-on-exist: drop tasks whose result json already exists in run_dir so a
    # RESUMED run (e.g. an architecture-search candidate re-launched after a pause)
    # CONTINUES from where it stopped instead of re-running the whole batch from
    # scratch. The shared memory pool is persisted to disk, so accumulation across
    # the already-done tasks is preserved and the remaining tasks build on it.
    _n_before = len(data_to_run)
    _completed_identities = {
        row["item_index"]: row["task_identity"]
        for row in load_completed_task_results(run_dir)
    }
    data_to_run = [
        it for it in data_to_run
        if not (
            isinstance(it, dict)
            and it.get("_global_index") is not None
            and _completed_identities.get(it["_global_index"])
            == it.get("_task_identity")
        )
    ]
    if len(data_to_run) < _n_before:
        logger.info(
            f"Skip-on-exist: {_n_before - len(data_to_run)} already-done task(s) in "
            f"{run_dir}; running {len(data_to_run)} remaining."
        )

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
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing WebWalkerQA"):
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

    logger.info(f"Processing completed. Total results: {len(results)}")
    require_complete_task_run(
        "WebWalkerQA", results, len(futures), future_errors
    )
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
        dataset_name="WebWalkerQA",
        has_levels=True,
        level_key="difficulty"
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebWalkerQA evaluation with Flash Searcher')

    parser.add_argument('--infile', type=str, required=True,
                        help='WebWalkerQA JSON/JSONL input path')
    parser.add_argument('--outfile', type=str,
                        default="runs/benchmarks/webwalkerqa/results.jsonl",
                        help='Output path for results')
    parser.add_argument('--model', type=str, default=None, help='Task model id override')
    parser.add_argument('--sample_num', type=int, default=None, 
                        help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None, 
                        help='Task indices to run, supports: single number (e.g., "5"), range (e.g., "1-10"), or mixed (e.g., "1,3,5-10,20")')
    parser.add_argument('--summary_interval', type=int, default=8, 
                        help='Summary interval for agent')
    parser.add_argument('--prompts_type', type=str, default="default", 
                        help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=1, 
                        help='Number of concurrent tasks (default=1 to avoid memory provider concurrency issues)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_steps', type=int, default=40, 
                        help='Maximum number of steps for agent')
    parser.add_argument('--token_budget', type=int, default=32768, help='Model max completion tokens')
    parser.add_argument('--judge_model', type=str, default=(os.getenv('DEFAULT_JUDGE_MODEL') or 'gpt-5'),
                        help='Model used for answer judgement')
    parser.add_argument('--memory_provider', type=str,
                        choices=[MemoryType.MODULAR.value], default=None,
                        help='Enable the AutoMem modular provider')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory). Default: True')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false',
                        help='Disable memory system evolution (skip take_in_memory)')
    
    # WebWalkerQA specific filters
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'], default=None,
                        help='Filter tasks by difficulty level')
    parser.add_argument('--lang', type=str, choices=['en', 'zh'], default=None,
                        help='Filter tasks by language')
    parser.add_argument('--domain', type=str, default=None,
                        help='Filter tasks by domain (e.g., conference, game, organization)')
    parser.add_argument('--question_type', type=str, choices=['single_source', 'multi_source'], default=None,
                        help='Filter tasks by question type')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Direct output directory (skips timestamped nesting)')
    parser.add_argument('--runtime_config_json', type=str, default=None,
                        help='Structured RuntimeConfig JSON emitted by AutoMem search')
    parser.add_argument('--shared_memory_provider', action='store_true',
                        help='Reuse one memory provider across the full standalone run; '
                             'concurrency>1 is allowed but memory accumulation order is nondeterministic')

    args = parser.parse_args()
    
    main(args)
