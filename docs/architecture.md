# Architecture

## System Boundary

AutoMem separates the architecture that is searched from the runtime that evaluates
it:

```text
task trajectory
      |
      v
 Encode -> Store -> Retrieve -> Manage
                         |
                         v
              fixed memory-use runtime

search control: propose -> validate -> evaluate -> diagnose -> select
```

The four architecture coordinates choose memory representations, persistence,
retrieval, and lifecycle management. The fixed runtime controls when retrieval is
allowed, how a query is formed, and how retrieved evidence is rendered for the
agent. Search logic may propose only the four architecture coordinates.

## Public Space

`automem-esrm-v1` contains:

- Encode: a non-empty subset of `tip`, `insight`, `trajectory`, `workflow`,
  `shortcut` (31 possible selections)
- Store: `json`, `vector`, `hybrid`, `graph`, `llm_graph`
- Retrieve: `hybrid`, `contrastive`, `cbr_rerank`, `graph`, `hyde`, `mmr`
- Manage: `lightweight`, `json_full`, `tool_manager`, `graph_consolidate`

The pre-validation count is `31 * 5 * 6 * 4 = 3720`. Every selected encode type
is persisted to the single selected store; per-type mixed routing is not part of
the public space. Compatibility validation currently accepts 2573 combinations:

- `graph` retrieval requires `graph` or `llm_graph` storage.
- `graph_consolidate` requires `graph` or `llm_graph` storage and the `graph`
  retriever.

`graph_consolidate` is the only public graph-management name. It runs content
consolidation and records success/failure evidence for traversed `SIMILAR` edges,
then updates their effective weights. It does not create a separate adaptive
coordinate, and it does not include shortcut promotion; shortcut management belongs
to `tool_manager`. Runtime configs and search repair reject retired preset names;
the release has no hidden compatibility aliases.

## Fixed Memory-Use Runtime

`src/automem/runtime/policy.py` defines `automem-runtime-v1`. The same immutable
policy is used for all candidates.

### G2: Context Composition

The provider retrieves candidate units and makes one model call that jointly
selects relevant units and writes tentative guidance. Every retained statement must
cite a candidate, and the provider maps those citations to memory-unit IDs. The
policy injects at most three units. If no model is available, the call fails, or the
response is malformed, the composer falls back to the first retrieved unit. An
empty candidate list produces no guidance.

### G3: Phase And Duplicate Control

A task session starts at `BEGIN`. The agent may request one additional `IN` lookup
only at an explicit summary or replan boundary. The attempt is consumed even when
retrieval or composition returns nothing, so repeated empty lookups cannot bypass
the limit. Unit IDs and guidance fingerprints prevent reinjection within the same
task.

### G4: Literal-Preserving Query Planning

The planner keeps the complete literal task query. When a model can derive a
reusable abstract intent, that text contributes only to semantic retrieval; the
literal query remains the retrieval query. Planner failure therefore degrades to
literal-only retrieval rather than changing task meaning.

G2, G3, and G4 have no public switches. Their limits are code constants covered by
offline tests, not runtime options embedded in YAML, JSON, environment variables,
or proposer output.

## Source Ownership

```text
src/automem/architecture/   Strict public spec and internal compilation
src/automem/providers/      Extraction and memory lifecycle integration
src/automem/storage/        Persistence implementations
src/automem/retrieval/      Candidate selection and graph edge traces
src/automem/management/     Presets and lifecycle operations
src/automem/runtime/        Fixed planner, composer, session policy
src/automem/search/         Search control and evaluation orchestration
src/automem/evaluation/     Metrics and artifact aggregation
src/automem/benchmarks/     Dataset-specific runner modules
src/flashoagents/           Agent runtime used by the runners
```

Installed code resolves prompts through package resources. It must not depend on a
developer checkout path or an external baseline repository.

## Persistence And Artifacts

Stores persist memory units under a user-selected runtime directory. Search runs,
task traces, model responses, vector indexes, downloaded datasets, and memory pools
are generated artifacts and are excluded from Git. Publishing any of them requires
separate privacy, license, and secret review.

The repository contains benchmark runner code but no benchmark payloads or baseline
implementations. Baseline reproduction remains an external responsibility and must
pin its own upstream source, revision, license, models, and evaluation protocol.
