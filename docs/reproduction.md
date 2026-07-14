# Reproduction

[简体中文](reproduction_CN.md)

## Release Boundary

This repository reproduces the AutoMem software and its fixed search protocol,
using the paper-aligned architecture contract: Encode selects a non-empty subset
of the five extraction types, so the manuscript's multi-Encode discovered routes
(for example `tip+trajectory+workflow` on GAIA) are expressible as strict
`automem-esrm-v1` `ArchitectureSpec` documents. The repository intentionally
excludes benchmark data, model weights, credentials, external baseline
repositories, raw rollouts, memory pools, and reported-result artifacts.

It is still not, by itself, a numerical reproduction package for
`AutoMAS/paper/main.tex`: the manuscript's runs used their own data snapshots,
split/final-report protocol, model revisions, and prompt states. Do not label
metrics produced by this release as reproductions of the manuscript tables unless
those inputs are separately reconciled and released.

The July graph-adaptive operations are part of the sole
`graph_consolidate` manager; there is no second `graph_adaptive` architecture option.

## Offline Verification

The full offline test suite imports the optional multimedia runners, so install both
development and benchmark dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,benchmarks]"

automem space
automem smoke
python -m compileall -q src
ruff check src tests
pytest -m "not online"
```

`automem space` reports the public 5/5/6/4 registry and compatible combinations.
`automem smoke` checks the architecture, storage, retrieval, context-composition,
and refresh contracts without network access.

To exercise two complete synthetic evolution rounds, including split creation,
candidate search, canonical handoff, Pareto updates, and M3 runoff:

```bash
SMOKE_ROOT="$(mktemp -d)"
python -m automem.search.engine \
  --run_name evolution-smoke \
  --output_dir "$SMOKE_ROOT" \
  --infile examples/smoke_tasks.jsonl \
  --max_rounds 2 \
  --num_candidates 3 \
  --warmup_n 1 \
  --search_n 4 \
  --batch_size 2 \
  --validation_n 1 \
  --test_n 1 \
  --dry_run \
  --no_ledger
```

Synthetic metrics only validate control flow. They are not benchmark results.

## Dataset Contracts

Obtain datasets from their official sources and comply with their licenses. AutoMem
does not download them.

| Runner | Input | Required fields |
| --- | --- | --- |
| GAIA | JSON array or JSONL | `Question`, `Final answer`, `file_name`, `task_id`, `Level` |
| WebWalkerQA | JSON array or JSONL | non-empty `question`, `answer`, `root_url`; optional object-valued `info` metadata |
| xBench-DeepSearch | UTF-8 CSV | `id`, `prompt`, `answer`, `reference_steps`, `canary` |

GAIA `file_name` must be empty or a relative path confined to the input file's
directory. Absolute paths, `..` escapes, and symlink escapes are rejected. ZIP
attachments are extracted into a per-task temporary directory with path, type,
member-count, expanded-size, and compression-ratio limits.

xBench `prompt` and `answer` are expected to use its base64 plus per-row XOR-canary
encoding. Do not publish decrypted benchmark rows. `--task_indices` uses one-based
indices for all runners; persisted `item_index` is also one-based.

## Online Search

Copy `.env.example` to a private environment file or export only the variables you
need. Never commit populated credentials. Role-specific endpoint variables must be
configured as key/base pairs.

Inspect the exact installed CLIs first:

```bash
python -m automem.benchmarks.gaia.runner --help
python -m automem.benchmarks.webwalkerqa.runner --help
python -m automem.benchmarks.xbench_deepsearch.runner --help
python -m automem.search.engine --help
```

An explicit GAIA search example for the current release is:

```bash
python -m automem.search.engine \
  --run_name gaia-current-v1 \
  --output_dir runs/search \
  --infile data/gaia/metadata.jsonl \
  --benchmark GAIA \
  --model TASK_MODEL_ID \
  --search_model SEARCH_MODEL_ID \
  --judge_model JUDGE_MODEL_ID \
  --diagnosis_model DIAGNOSIS_MODEL_ID \
  --max_rounds 8 \
  --num_candidates 3 \
  --warmup_n 19 \
  --search_n 100 \
  --batch_size 50 \
  --validation_n 30 \
  --test_n 15 \
  --max_steps 40 \
  --token_budget 8192 \
  --concurrency 1 \
  --final_validation
```

`--model` controls only the benchmark task agent; `--search_model` controls the
architecture proposer, while `--diagnosis_model` and `--judge_model` keep their
named roles. These are release defaults, not a claim about the older paper budget.

The current xBench release default is 10 warmup, 70 search, 10 validation, and 10
held-out tasks, covering its 100 rows. An explicit equivalent command is:

```bash
python -m automem.search.engine \
  --run_name xbench-current-v1 \
  --output_dir runs/search \
  --infile data/xbench/deepsearch.csv \
  --benchmark xBench-DeepSearch \
  --model TASK_MODEL_ID \
  --search_model SEARCH_MODEL_ID \
  --judge_model JUDGE_MODEL_ID \
  --diagnosis_model DIAGNOSIS_MODEL_ID \
  --warmup_n 10 \
  --search_n 70 \
  --batch_size 50 \
  --validation_n 10 \
  --test_n 10 \
  --concurrency 1 \
  --final_validation
```

These xBench sizes are the current software protocol, not a reconstruction of an
older paper split. When no `--data_split` or split-size argument is supplied, the
search engine applies the same 10/70/10/10 xBench default automatically. Keep
`--concurrency 1` for evolution runs (the search coordinator enforces it): shared memory operations are
thread-safe, but task completion order at higher concurrency changes which memory is
available to later tasks. Standalone runners permit higher shared-memory concurrency,
but explicitly report that resulting memory accumulation order as nondeterministic.

Resume with the identical command plus `--resume`. The run records a protocol digest
covering task and referenced-attachment bytes, split, runner/package source, prompts,
resolved models/endpoints, Web providers/cache policy, and behavior-changing
arguments. A mismatch invalidates protocol-dependent checkpoints. Incomplete
stateful candidates are replayed as a full batch from their persisted round-start
canonical snapshot; AutoMem never fills only the missing tasks into an already
evolved candidate store. Warmup, M3 contenders, and held-out memory validation use
the same exact-or-full-replay rule. M3 additionally binds every contender to one
manifest-signed `runoff_start_state` snapshot. Use a new `--run_name` when
intentionally changing the experiment.

Online evidence is not immutable merely because the protocol digest matches. For a
controlled rerun, use a dedicated `AUTOMEM_CACHE_DIR`, populate the search/page cache
once, retain a hash or archive of it, and set `FREEZE_CACHE=true`. Otherwise record
live Web/provider drift as a limitation. Cache contents themselves are deliberately
not folded into the run digest because they grow during a normal run.

## Outputs And Failure Semantics

Run data is written beneath the chosen `--output_dir` and is ignored by Git. The
search records the split, protocol signature, baseline, canonical pool, per-candidate
task results, Pareto history, champion state, M3 runoff, and optional held-out report.

Runner worker failures, task infrastructure errors, missing judge verdicts, missing
or extra task files, and corrupt outputs cause a non-zero run instead of being scored
as ordinary wrong answers. Invalid result files are not resume-skipped; the affected
candidate is rebuilt from its round-start snapshot before a full-batch retry. Review
`run.log` and per-task error JSON before retrying.

Every scoreable runner checkpoint follows one current schema: its filename is
`<item_index>.json`, the one-based integer `item_index` matches that filename,
`status` is exactly `"success"`, `judge_unjudged` is explicitly `false`, and
`task_score` is finite in `[0, 1]`. It also carries a `task_identity` SHA-256 derived
from the exact input-file SHA-256 and one-based row index. Runner resume and engine
aggregation both compare that identity with the current dataset. Custom runners must
emit the same contract. Old, mismatched, duplicate, or half-structured checkpoints
are retried rather than reused.
Audio inspection additionally requires system `ffmpeg` plus an explicit
`MTU_API_KEY`/`MTU_BASE_URL` pair.

## Reproducibility Record

Record the Git commit and dirty status, Python and dependency versions, serialized
architecture and fingerprint, protocol/runtime digests, exact dataset revision and
task IDs, model/provider revisions, endpoint roles, seeds, all CLI arguments, and
hashes of released output artifacts. Unknown model revisions or mutable datasets must
be reported as limitations.

## Baselines

The search automatically evaluates its own no-memory control. External baseline
implementations are intentionally not vendored. Obtain each baseline from its
official source under its license and record its upstream URL, exact revision,
backbone, judge, task subset, and scoring rules. `--baseline_from` only reuses an
AutoMem no-memory baseline when task indices, model, and baseline protocol digest
match exactly.
