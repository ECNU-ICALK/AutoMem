# Configuration

## Public Contract

The public architecture document is a strict `ArchitectureSpec` with exactly five
fields. `encode` is a non-empty list of unique extraction types (a single string
is accepted as shorthand for a one-type subset and serializes back as a list);
the other four fields are strings:

```json
{
  "schema_version": "1",
  "encode": ["tip", "trajectory", "workflow"],
  "store": "graph",
  "retrieve": "contrastive",
  "manage": "lightweight"
}
```

Each architecture selects a non-empty Encode subset and one value from each of
the other coordinates in `automem-esrm-v1`. All selected encode types are
persisted to the one selected store. Unknown fields, missing fields, wrongly
typed values, duplicate encode entries, unsupported schema versions, unknown
options, and incompatible combinations are errors.

The implemented Python interface is:

```python
from automem.architecture.models import ArchitectureSpec

spec = ArchitectureSpec.from_dict(
    {
        "schema_version": "1",
        "encode": ["tip", "trajectory", "workflow"],
        "store": "graph",
        "retrieve": "contrastive",
        "manage": "lightweight",
    }
)
print(spec.fingerprint)
```

`fingerprint` is the SHA-256 digest of canonical JSON; the encode subset is
canonicalized to the fixed menu order (`tip`, `insight`, `trajectory`,
`workflow`, `shortcut`) first, so fingerprints do not depend on the order the
subset was written in. Use `automem space` to inspect the machine-readable
option registry and compatibility counts.

## Compatibility Rules

- `retrieve: "graph"` requires `store: "graph"` or `store: "llm_graph"`.
- `manage: "graph_consolidate"` requires graph-family storage and
  `retrieve: "graph"` because it consumes weighted edge traces.

The other management names are `lightweight`, `json_full`, and `tool_manager`.
There are no additional public management aliases.

## What Is Not Configurable

`automem-runtime-v1` is code-defined and is not part of `ArchitectureSpec`:

- G2 uses one cited context-composition call and a top-one offline fallback.
- G3 allows `BEGIN` plus at most one explicit refresh boundary.
- G4 always preserves the literal query and uses an abstract query only as a
  semantic supplement.

Do not add query-transform, injection-renderer, phase, graph-adaptation, threshold,
or fallback fields to an architecture document. Such fields are rejected as
unknown, and exposing them would change the experimental contract.

## Operational Inputs

Dataset paths, output directories, model IDs, seeds, task budgets, and concurrency
are runner or search-process arguments. Credentials and compatible service
endpoints are supplied separately through the environment variables listed in
`.env.example`. They are operational inputs, not architecture coordinates.

Search model roles are explicit: `--model` is the benchmark task agent,
`--search_model` is the architecture proposer, `--diagnosis_model` is the diagnosis
and ledger role, and `--judge_model` is the evaluator. Their resolved identities and
effective public endpoints are signed separately into the protocol digest.

The search engine writes a structured `runtime_config.json` when launching a
benchmark worker. That file is an internal compiler-to-worker transport containing
the selected implementation settings. It is generated output, not a second public
configuration schema and not a place for G2/G3/G4 options.

An optional `--data_split` file uses zero-based task indices:

```json
{
  "profile_indices": [0],
  "optimization_indices": [1, 2, 3, 4],
  "validation_indices": [5],
  "final_test_indices": [6]
}
```

Indices must be exact integers (booleans are rejected), non-negative, unique within
each split, in range for the current input, and disjoint across splits. The resolved
split is persisted inside the run and included in the evaluation-protocol digest.
Changing it requires a new run rather than mixing checkpoints.

Each scoreable task checkpoint is also bound to the exact benchmark file and its
one-based global row through `task_identity`. Custom runners must emit this SHA-256
field together with explicit success/judge status; otherwise search aggregation fails
closed instead of treating the file as a scored task.

The GAIA release defaults are warmup/search/validation/final-test
`19/100/30/15`. xBench has 100 rows, so when no custom split and no explicit split
size is supplied its current release default is `10/70/10/10`. These are software
protocol defaults, not claims about older manuscript budgets. Any explicit split
argument or `--data_split` remains authoritative.

The search coordinator requires `--concurrency 1` for real evolution because all
tasks in a candidate share a causally evolving memory pool. Limits such as
`--max_steps`, `--token_budget`, model IDs, dataset bytes, runner source, and endpoint
roles are operational inputs, but they are signed into the run protocol so resume
cannot silently reuse incompatible measurements.
Standalone runners allow a shared provider with concurrency above one, but task
completion order then changes memory accumulation. Such runs are intentionally
reported as nondeterministic and are not substitutes for the coordinator's
single-worker evolution protocol.

Web evidence also depends on `WEB_SEARCH_PROVIDER`, `WEB_ACCESS_PROVIDER`, proxy,
streaming, and cache controls in `.env.example`. Their resolved non-secret settings
are included in the protocol digest. Cached response *contents* and live web pages
remain external mutable inputs: for a controlled rerun, pin a dedicated
`AUTOMEM_CACHE_DIR`, populate it once, then run with `FREEZE_CACHE=true` and retain a
hash/archive of that cache alongside the result record.

AutoMem currently does not provide a public YAML resolver or commands named
`config resolve`, `data prepare`, or `results verify`. Documentation and automation
must not rely on those interfaces.
