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
import re
import time
import json
import argparse
import json_repair
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from automem.endpoints import resolve_openai_endpoint

load_dotenv()

# Judge uses separate JUDGE_API_* env vars if set, otherwise falls back to OPENAI_API_*
import httpx

_client = None


def _get_client():
    """Lazily construct the judge client on first use.

    The old module-level construction made `import lasj` fail outright in any
    environment without OPENAI_API_KEY (clean CI, tests that only need
    _parse_judge_response).
    """
    global _client
    if _client is None:
        api_key, api_base = resolve_openai_endpoint("JUDGE")
        _client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=600.0),
        )
    return _client

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json_repair.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error：line {line_num} - {e}")
    return data

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def _parse_judge_response(response_text):
    """Extract a correct/incorrect verdict from a judge LLM response.

    Judges (esp. qwen-family) often wrap the JSON in ```json fences or emit
    slightly malformed JSON; the old strict json.loads scored those tasks as
    judgement="error" -> 0 points even when the answer was right. Candidates
    are the raw text followed by every fenced block in order, each tried
    with strict json.loads then json_repair; the LAST valid verdict wins
    (xBench echo-bug lesson: a judge may echo the prompt template — which
    contains a verdict-shaped snippet — before emitting its real verdict).
    Returns 'correct' / 'incorrect', or None if no valid verdict exists.
    """
    if not response_text:
        return None
    candidates = [response_text]
    candidates.extend(
        m.strip() for m in re.findall(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
    )
    verdict = None
    for cand in candidates:
        for loader in (json.loads, json_repair.loads):
            try:
                obj = loader(cand)
            except Exception:
                continue
            if isinstance(obj, dict):
                v = str(obj.get("judgement", "")).strip().lower()
                if v in ("correct", "incorrect"):
                    verdict = v
                    break  # first successful loader per candidate is enough
    return verdict


def judge_equivalence(question, gt_answer, pred_answer, model="qwen3-max"):
    try:
        pred_answer = pred_answer["answer"]
    except Exception:
        pass
    prompt = f"""
    Please determine if the predicted answer is equivalent to the labeled answer. 
    Question:  {question} 
    Labeled Answer:  {gt_answer} 
    Predicted Answer: {pred_answer}  
    Are these answers equivalent? 
    The output should in the following json format: 
    {{  
    "rationale": "your rationale for the judgement, as a text", 
    "judgement": "your judgement result, can only be 'correct' or 'incorrect'" 
    }}
    """
    if pred_answer is None or pred_answer == '':
        # No answer produced (empty or None pred) is a plain wrong answer,
        # not a judge-infrastructure error.
        return {
            "question": question,
            "judgement": "incorrect",
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
        }
    last_error = None
    for _attempt in range(3):
        try:
            client = _get_client()
            use_stream = os.environ.get("FORCE_STREAM", "").strip().lower() in ("1", "true", "yes")
            if use_stream:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a fair judge evaluating if two answers to a question are equivalent."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    stream=True,
                )
                content_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_parts.append(chunk.choices[0].delta.content)
                response_text = "".join(content_parts).strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a fair judge evaluating if two answers to a question are equivalent."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content.strip()

            verdict = _parse_judge_response(response_text)
            if verdict is not None:
                return {
                    "question": question,
                    "judgement": verdict,
                    "gt_answer": gt_answer,
                    "pred_answer": pred_answer,
                }
            last_error = f"unparseable judge response: {response_text[:200]!r}"
        except Exception as e:
            last_error = str(e)
        if _attempt < 2:
            time.sleep(1.5 * (_attempt + 1))

    print(f"Error judging equivalence after 3 attempts: {last_error}")
    return {
        "question": question,
        "judgement": "error",
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
    }

def process_batch(items, model, max_workers=5):

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, item in enumerate(items):

            question = item.get('question', '') if item is not None else ''
            gt_answer = item.get('golden_answer', '') if item is not None else ''
            pred_answer = item.get('agent_result', {}) if item is not None else {}
            
            futures[executor.submit(
                judge_equivalence,
                question,
                gt_answer,
                pred_answer,
                model
            )] = idx
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging answers"):
            results.append(future.result())
    return results

def calculate_accuracy(results):
    total = len(results)
    if total == 0:
        return 0.0
        
    correct = sum(1 for r in results if r['judgement'] == 'correct')
    incorrect = sum(1 for r in results if r['judgement'] == 'incorrect')
    errors = sum(1 for r in results if r['judgement'] == 'error')
    
    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
    
    print(f"Total items: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{correct + incorrect})")
    
    return accuracy

def main(args):
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} items")
    
    if args.sample_size and args.sample_size < len(data):
        data = data[:args.sample_size]
        print(f"Processing first {args.sample_size} items")
    
    results = process_batch(data, args.model, args.max_workers)
    
    save_results(results, args.output_file)
    print(f"Results saved to {args.output_file}")
    calculate_accuracy(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge equivalence between predicted and labeled answers using OpenAI API")
    
    parser.add_argument("--input_file", default="./data/<example.json>", help="Path to input JSONL file containing questions, answers and agent results")
    parser.add_argument("--output_file", default="./output/<example.jsonl>", help="Path to save judgement results (JSONL)")
    parser.add_argument("--model", default="qwen3-max", help="OpenAI model to use for judging")
    parser.add_argument("--sample_size", type=int, help="Number of items to process (optional)")
    parser.add_argument("--max_workers", type=int, default=20, help="Number of parallel workers for API calls")
    
    args = parser.parse_args()
    main(args)
