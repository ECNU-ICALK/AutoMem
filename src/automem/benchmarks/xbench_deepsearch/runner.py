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
import csv
import base64
import re
import unicodedata
from tqdm import tqdm
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from flashoagents.models import OpenAIServerModel
from flashoagents.base_agent import MMSearchAgent
from automem.evaluation.io import write_jsonl
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


def xor_decrypt(data, key):
    """XOR decrypt data with a key"""
    key_bytes = key.encode('utf-8')
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


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


LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。

请确保最后一行严格为 `结论: 正确` 或 `结论: 错误`（只含这两个词之一）。
""".strip()


def _normalize_answer(s):
    """Normalize for robust comparison: full-width->half-width (NFKC, so
    full-width '：' and digits collapse to ASCII), strip surrounding
    whitespace/quotes/punctuation, lowercase. Fixes the brittle exact-match
    that scored correct answers as wrong on Chinese formatting."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.strip()
    s = re.sub(r"^[\s:'\"`,.，。；;]+|[\s'\"`,.，。；;]+$", "", s)
    return s.lower()


def _extract_labeled(text, label):
    """Return the text after a `label`（最终答案/解释/结论）marker, tolerant of
    both ASCII ':' and full-width '：' and a missing colon. None if absent."""
    if not text:
        return None
    m = re.search(rf'{label}\s*[:：]?\s*(.+)', text, re.DOTALL)
    return m.group(1).strip() if m else None


class JudgeInfrastructureError(RuntimeError):
    """The judge produced no usable verdict, so the task must not be scored."""


def _validate_xbench_task_ids(data):
    """Reject ids that cannot identify rows unambiguously in reports."""

    task_ids = [str(item.get("id") or "").strip() for item in data]
    if any(not task_id for task_id in task_ids):
        raise ValueError("xBench input contains an empty task id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("xBench input contains duplicate task ids")


def _checkpoint_filename(result):
    """Return the checkpoint name bound to the internal one-based row index."""

    item_index = result.get("item_index")
    if type(item_index) is not int or item_index < 1:
        raise ValueError("xBench result is missing a valid one-based item_index")
    return f"{item_index}.json"


def grade_question(question_text, correct_answer, llm_response, judge_model):
    if llm_response is None:
        return 0, "", ""

    # Fast path: normalized exact match. Take the answer after a 最终答案/Final
    # Answer label if present, else the (already-clean) response itself.
    candidate = _extract_labeled(llm_response, r'(?:最终答案|final answer)') or llm_response
    if _normalize_answer(candidate) == _normalize_answer(correct_answer):
        return 1, candidate.strip(), "normalized exact match (no LLM judge)"

    # Otherwise: LLM-as-judge for semantic equivalence (robust to formatting,
    # units, paraphrase, numeric tolerance).
    judge_prompt = LLM_JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=llm_response,
    )

    def _call_judge():
        msg = judge_model([{"role": "user", "content": judge_prompt}])
        return msg.content if msg else None

    try:
        judge_response = _call_judge()
    except Exception as e:
        logger.warning(f"Judge model call failed: {e}")
        raise JudgeInfrastructureError("judge model call failed") from e

    if not isinstance(judge_response, str):
        raise JudgeInfrastructureError("judge response is not a string")

    # Echo guard (2026-07 fix). A flaky judge endpoint sometimes regurgitates the
    # prompt template instead of judging. The template carries literal placeholder
    # tokens ([回答]/[正确答案]/[最终答案]) and the instruction line "结论：正确/错误",
    # so the old FIRST-match parser scored those echoes as a spurious 正确 and
    # inflated accuracy (this is what produced the fake xBench 90.9% run: e.g.
    # model answered -1905 for gold -1999 yet was graded 正确). Detect the echo,
    # retry once, and fail closed to 0 rather than silently pass.
    def _looks_like_echo(resp):
        return any(tok in resp for tok in ("[回答]", "[正确答案]", "[最终答案]"))

    if _looks_like_echo(judge_response):
        logger.warning("Judge echoed the prompt template (no real verdict); retrying once.")
        try:
            retry = _call_judge()
        except Exception as e:
            logger.warning(f"Judge retry failed: {e}")
            raise JudgeInfrastructureError("judge echo retry failed") from e
        if isinstance(retry, str):
            judge_response = retry
        if _looks_like_echo(judge_response):
            raise JudgeInfrastructureError("judge echoed the prompt after retry")

    # Robust verdict parse: the prompt requires the verdict on the LAST line, so
    # take the LAST 结论→正确/错误 match. Taking the FIRST (old behaviour) matched
    # the template's own "结论：正确" example whenever a response was verbose or
    # echoed the instructions. Tolerates ':'/'：'/whitespace/连接词.
    verdicts = re.findall(r'结论[\s\S]{0,15}?(正确|错误)', judge_response)
    if not verdicts:
        raise JudgeInfrastructureError("judge verdict not found")
    score = 1 if (verdicts and verdicts[-1] == "正确") else 0
    extracted = _extract_labeled(judge_response, '最终答案') or ""
    explanation = _extract_labeled(judge_response, '解释') or ""

    return score, extracted, explanation


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


def process_item(
    item,
    model_config,
    summary_interval,
    prompts_type,
    max_steps,
    memory_type_str=None,
    enable_memory_evolution=True,
    judge_model=None,
    shared_memory_provider=None,
    extract_plan=None,
    runtime_config_path=None,
):
    """Process a single XBench task with timing and metrics tracking"""
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
    
    search_agent = MMSearchAgent(
        task_model, 
        summary_interval=summary_interval, 
        prompts_type=prompts_type, 
        max_steps=max_steps,
        memory_provider=memory_provider
    )

    question = item["prompt"]
    golden_answer = item["answer"]
    task_id = item.get("id")
    
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
        
        is_correct = False
        score = 0
        extracted_answer = ""
        grader_explanation = ""
        judge_unjudged = not bool(judge_model)
        if judge_model:
            try:
                agent_result = result.get("agent_result", "")
                if agent_result is None:
                    agent_response = ""
                elif isinstance(agent_result, dict):
                    agent_response = agent_result.get("answer", "")
                    if not agent_response:
                        agent_response = json.dumps(agent_result, ensure_ascii=False)
                else:
                    agent_response = str(agent_result) if agent_result else ""
                
                if not isinstance(agent_response, str):
                    agent_response = str(agent_response) if agent_response else ""
                
                score, extracted_answer, grader_explanation = grade_question(
                    question,
                    golden_answer,
                    agent_response,
                    judge_model
                )
                is_correct = (score == 1)
            except Exception as e:
                logger.warning(f"Judgement failed: {e}")
                score = 0
                extracted_answer = ""
                grader_explanation = f"Error: {str(e)}"
                judge_unjudged = True
        
        if memory_provider and enable_memory_evolution and not judge_unjudged:
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
                        "task_id": task_id,
                        "status": "success",
                        "is_correct": is_correct,
                        "full_query": question,
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
            "task_id": task_id,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "item_index": item.get("__item_index"),
            "task_identity": item.get("_task_identity"),
            "status": "success",
            "judge_unjudged": judge_unjudged,
            "task_score": float(score),
            "success": bool(is_correct),
            "agent_messages": agent_messages,
            "score": score,
            "extracted_answer": extracted_answer,
            "grader_explanation": grader_explanation,
            "memory_metrics": capture_memory_metrics(memory_provider),
            **result,
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
        
        # Infrastructure failures are not valid negative examples. Keep the
        # diagnostic JSON, fail the run, and let the next invocation retry it.
        
        task_result = {
            "task_id": task_id,
            "question": question,
            "full_query": question,
            "golden_answer": golden_answer,
            "item_index": item.get("__item_index"),
            "task_identity": item.get("_task_identity"),
            "status": "error",
            "task_score": 0.0,
            "success": False,
            "error": str(e),
            "error_traceback": error_msg,
            "agent_result": None,
            "agent_trajectory": [],
            "agent_messages": agent_messages,
            "score": 0,
            "extracted_answer": "",
            "grader_explanation": f"Error: {str(e)}",
            "memory_metrics": capture_memory_metrics(memory_provider),
        }
        
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)


def main(args):
    infile = Path(args.infile).expanduser()
    if not infile.is_file():
        raise FileNotFoundError(f"xBench DeepSearch input CSV not found: {infile}")
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
    
    task_api_key, task_api_base = resolve_openai_endpoint()
    model_config = {
        "model_id": args.model or os.environ.get("DEFAULT_MODEL", "gpt-5"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": args.token_budget,
        "api_key": task_api_key,
        "api_base": task_api_base,
    }
    
    model = OpenAIServerModel(**model_config)

    judge_api_key, judge_api_base = resolve_openai_endpoint("JUDGE")
    judge_model = OpenAIServerModel(
        args.judge_model,
        custom_role_conversions=custom_role_conversions,
        api_key=judge_api_key,
        api_base=judge_api_base,
        max_completion_tokens=4096,
    )

    data = []
    dataset_sha256 = dataset_file_sha256(args.infile)
    with open(args.infile, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for question in reader:
            key = question["canary"]
            question["prompt"] = xor_decrypt(base64.b64decode(question["prompt"]), key).decode('utf-8')
            question["answer"] = xor_decrypt(base64.b64decode(question["answer"]), key).decode('utf-8')
            data.append(question)

    _validate_xbench_task_ids(data)
    for item_index, item in enumerate(data, start=1):
        item["_task_identity"] = task_identity_digest(
            dataset_sha256, item_index
        )

    if args.task_indices:
        try:
            selected_indices = parse_task_indices(args.task_indices)
            picked = []
            for i in sorted(selected_indices):
                if 0 < i <= len(data):
                    item = data[i - 1]
                    # 1-based index, matches --task_indices and the search loop's
                    # split index mapping; emitted as item_index so the loop can
                    # attribute per-task scores back to the search batch.
                    item["__item_index"] = i
                    picked.append(item)
            data = picked
            logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing task indices: {e}")
            raise ValueError(f"Invalid --task_indices: {args.task_indices!r}") from e
        if not data:
            raise ValueError(
                f"--task_indices selected no xBench tasks: {args.task_indices!r}"
            )
    elif args.sample_num is not None:
        data = data[:args.sample_num]
        for i, item in enumerate(data, 1):
            item["__item_index"] = i
    else:
        for i, item in enumerate(data, 1):
            item["__item_index"] = i

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

    # Checkpoints are bound to the internal one-based row index, not the
    # external dataset id. This prevents duplicate or filename-shaped ids from
    # overwriting another task's result.
    _done_identities = {
        row["item_index"]: row["task_identity"]
        for row in load_completed_task_results(run_dir)
    }
    if _done_identities:
        _before = len(data_to_run)
        data_to_run = [
            item
            for item in data_to_run
            if _done_identities.get(item.get("__item_index"))
            != item.get("_task_identity")
        ]
        logger.info(
            f"Resume: {len(_done_identities)} task(s) already done; "
            f"{len(data_to_run)}/{_before} remaining."
        )

    results = []
    file_lock = threading.Lock()
    effective_concurrency = args.concurrency
    shared_memory_provider = None
    if args.shared_memory_provider and args.memory_provider:
        if effective_concurrency != 1:
            # The provider serializes take_in_memory / provide_memory with an
            # internal RLock and was explicitly designed for shared-memory +
            # concurrency>1. Since 2026-07-11 per-task memory_metrics are
            # thread-local (truly per-task); the residual nondeterminism is
            # only the memory-accumulation ORDER across concurrent tasks.
            logger.warning(
                f"--shared_memory_provider with concurrency={effective_concurrency} "
                f"enabled. Provider methods are serialized by an internal RLock; "
                f"memory-accumulation order across concurrent tasks is nondeterministic."
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
            result_to_save = {k: v for k, v in result.items() 
                             if k not in ["agent_messages", "grader_explanation"]}
            
            ordered_result = {}
            if "score" in result_to_save:
                ordered_result["score"] = result_to_save["score"]
            if "extracted_answer" in result_to_save:
                ordered_result["extracted_answer"] = result_to_save["extracted_answer"]
            for k, v in result_to_save.items():
                if k not in ["score", "extracted_answer"]:
                    ordered_result[k] = v
            
            filename = _checkpoint_filename(ordered_result)
            if not args.skip_summary:
                write_jsonl(args.outfile, [ordered_result], "a")

            save_task_result(ordered_result, run_dir, filename)

    if args.memory_provider:
        if shared_memory_provider is not None:
            logger.info(f"Memory provider enabled: {args.memory_provider} (shared provider, concurrency={effective_concurrency})")
        else:
            logger.info(f"Memory provider enabled: {args.memory_provider} (each thread creates independent instance, using {effective_concurrency} workers)")

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
                args.enable_memory_evolution,
                judge_model,
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
                                  f"| Time: {metrics.get('elapsed_time', 0):.1f}s | Tokens: {metrics.get('total_tokens', 0)} | Score: {result.get('score', 'N/A')}")
                    elif result.get("status") == "error":
                        logger.warning(f"Task error [{len(results)}/{len(futures)}]: {result['question'][:80]}... | Error: {result.get('error', 'Unknown')}")
            except Exception as exc:
                import traceback
                logger.error(f"Failed to get result from future: {traceback.format_exc()}")
                future_errors.append(str(exc))

    logger.info(f"Processing completed. Completed this run: {len(results)}")
    require_complete_task_run("xBench", results, len(futures), future_errors)
    all_results = [
        row
        for row in load_completed_task_results(run_dir)
        if row["task_identity"]
        == task_identity_digest(dataset_sha256, row["item_index"])
    ]
    if not args.skip_summary:
        write_jsonl(args.outfile, all_results)

    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        all_results,
        report_path,
        dataset_name="XBench",
        has_levels=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal data generation')

    parser.add_argument('--infile', type=str, required=True,
                        help='xBench DeepSearch CSV input path')
    parser.add_argument('--outfile', type=str,
                        default="runs/benchmarks/xbench_deepsearch/results.jsonl",
                        help='output path')
    parser.add_argument('--model', type=str, default=None, help='Task model id override')
    parser.add_argument('--sample_num', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--task_indices', type=str, default=None, 
                        help='Task indices to run, supports: single number (e.g., "5"), range (e.g., "23-165"), or mixed (e.g., "1,3,5-10,20")')
    parser.add_argument('--summary_interval', type=int, default=8, help='Summary interval')
    parser.add_argument('--prompts_type', type=str, default="default", help='Type of prompts to use')
    parser.add_argument('--concurrency', type=int, default=1, help='Number of concurrency (default=1 to avoid memory provider concurrency issues)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_steps', type=int, default=40, help='Maximum number of steps')
    parser.add_argument('--token_budget', type=int, default=32768, help='Model max completion tokens')
    parser.add_argument('--memory_provider', type=str,
                        choices=[MemoryType.MODULAR.value], default=None,
                        help='Enable the AutoMem modular provider')
    parser.add_argument('--enable_memory_evolution', action='store_true', default=True,
                        help='Enable memory system evolution (take_in_memory). Default: True')
    parser.add_argument('--disable_memory_evolution', dest='enable_memory_evolution', action='store_false',
                        help='Disable memory system evolution (skip take_in_memory)')
    parser.add_argument('--skip_summary', action='store_true', help='Only save per-task json files, skip appending to summary outfile')
    parser.add_argument('--judge_model', type=str,
                        default=(os.getenv('DEFAULT_JUDGE_MODEL') or 'gpt-5'),
                        help='Judge model id')
    parser.add_argument('--runtime_config_json', type=str, default=None,
                        help='Structured RuntimeConfig JSON emitted by AutoMem search')
    parser.add_argument('--direct_output_dir', type=str, default=None, help='Direct output directory (skips timestamped nesting)')
    parser.add_argument('--shared_memory_provider', action='store_true',
                        help='Reuse one memory provider across the full standalone run; '
                             'concurrency>1 is allowed but memory accumulation order is nondeterministic')

    args = parser.parse_args()
    
    main(args)
    
