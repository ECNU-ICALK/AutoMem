# Reproduction

[简体中文](reproduction_CN.md)

## Offline checks (no credentials)

```bash
python -m pip install -e ".[dev,benchmarks]"

automem space
automem smoke
pytest -m "not online"
```

Two complete synthetic evolution rounds — split creation, candidates, canonical
handoff, Pareto updates, and M3 runoff — run offline with:

```bash
python -m automem.search.engine \
  --run_name evolution-smoke --output_dir "$(mktemp -d)" \
  --infile examples/smoke_tasks.jsonl \
  --max_rounds 2 --num_candidates 3 \
  --warmup_n 1 --search_n 4 --batch_size 2 --validation_n 1 --test_n 1 \
  --dry_run --no_ledger
```

Synthetic metrics validate control flow only; they are not benchmark results.

## Datasets

Obtain each benchmark from its official source under its own license; AutoMem
never downloads data. Required input fields:

| Runner | Input | Required fields |
| --- | --- | --- |
| GAIA | JSON array or JSONL | `Question`, `Final answer`, `file_name`, `task_id`, `Level` |
| WebWalkerQA | JSON array or JSONL | non-empty `question`, `answer`, `root_url` |
| xBench-DeepSearch | UTF-8 CSV | `id`, `prompt`, `answer`, `reference_steps`, `canary` (base64 + XOR-canary encoding; do not publish decrypted rows) |

GAIA `file_name` must be empty or a relative path confined to the input file's
directory; ZIP attachments are extracted under strict path/type/size limits.
`--task_indices` and persisted `item_index` are one-based for all runners.

## Online runs

Copy `.env.example` to `.env` and set only the variables you use — key/base
pairs must be configured together, and populated credentials must never be
committed. Then either run one benchmark directly:

```bash
python -m automem.benchmarks.gaia.runner \
  --infile data/gaia/metadata.jsonl --model TASK_MODEL --judge_model JUDGE_MODEL \
  --memory_provider modular --enable_memory_evolution
```

or launch a full architecture search (GAIA defaults shown; xBench-DeepSearch
defaults to a 10/70/10/10 split over its 100 rows):

```bash
python -m automem.search.engine \
  --run_name gaia-search --output_dir runs/search \
  --infile data/gaia/metadata.jsonl --benchmark GAIA \
  --model TASK_MODEL --search_model SEARCH_MODEL \
  --judge_model JUDGE_MODEL --diagnosis_model DIAGNOSIS_MODEL \
  --max_rounds 8 --num_candidates 3 \
  --warmup_n 19 --search_n 100 --batch_size 50 --validation_n 30 --test_n 15 \
  --final_validation
```

`--model` drives the task agent only; `--search_model`, `--diagnosis_model`, and
`--judge_model` control their named roles. `python -m automem.search.engine
--help` and each runner's `--help` list every argument.
