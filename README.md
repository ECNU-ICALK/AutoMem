# AutoMem

English | [简体中文](README_CN.md)

AutoMem is a research framework for task-adaptive long-term memory in LLM
agents. It searches one explicit memory architecture with four coordinates:
**Encode**, **Store**, **Retrieve**, and **Manage**. The repository includes the
architecture model, storage and retrieval implementations, management pipelines,
the fixed memory-use runtime, the search loop, and benchmark runners. It does not
vendor baseline repositories, benchmark datasets, model weights, or result
artifacts.

The architecture contract matches the paper: Encode selects a non-empty subset
of the five extraction types (multi-Encode routes such as
`tip+trajectory+workflow` are first-class), while Store, Retrieve, and Manage
each select one value. This repository is still not a standalone numerical
reproduction package — datasets, credentials, and result artifacts are
intentionally excluded. See [Reproduction](docs/reproduction.md) before
comparing results.

## Public Architecture Space

The sole public space is `automem-esrm-v1`:

| Coordinate | Options |
| --- | --- |
| Encode (non-empty subset of 5 → 31 selections) | `tip`, `insight`, `trajectory`, `workflow`, `shortcut` |
| Store (5) | `json`, `vector`, `hybrid`, `graph`, `llm_graph` |
| Retrieve (6) | `hybrid`, `contrastive`, `cbr_rerank`, `graph`, `hyde`, `mmr` |
| Manage (4) | `lightweight`, `json_full`, `tool_manager`, `graph_consolidate` |

This is a 3720-point space before compatibility validation (31 encode subsets x
5 x 6 x 4). Every selected encode type is persisted to the one selected store.
Graph retrieval requires `graph` or `llm_graph` storage. `graph_consolidate`
combines graph content consolidation with success-aware edge adaptation and
therefore requires graph-family storage together with the `graph` retriever.
The current validator accepts 2573 combinations.

An `ArchitectureSpec` selects a non-empty subset for Encode (a single string is
accepted as shorthand for a one-type subset) and exactly one value for each of
the other coordinates. Missing, unknown, duplicate, and incompatible values are
rejected. Execution behavior is not a fifth search coordinate.

## Fixed Runtime

Every architecture runs under the code-defined `automem-runtime-v1` policy:

- **G2 context composition:** retrieved candidates go through one relevance and
  composition call that produces tentative, cited guidance. If the model is
  unavailable or its output is unusable, the top retrieved item is used as the
  offline fallback.
- **G3 refresh lifecycle:** memory is considered at task `BEGIN`, followed by at
  most one explicitly requested refresh at a summary or replan boundary. Ordinary
  intermediate steps cannot refresh memory.
- **G4 query planning:** the literal task query is always preserved. An abstract
  query may supplement the semantic representation, but never replaces the literal
  query.

These behaviors and their limits are implemented in `src/automem/runtime/`. They
are deliberately absent from `ArchitectureSpec`, configuration files, environment
variables, and search proposals.

## Repository Layout

```text
src/automem/architecture/   Public schema, compatibility rules, compiler
src/automem/providers/      Memory extraction and provider lifecycle
src/automem/storage/        JSON, vector, hybrid, and graph-family stores
src/automem/retrieval/      Retrieval implementations
src/automem/management/     Lifecycle operations and four public presets
src/automem/runtime/        Fixed G2/G3/G4 execution policy
src/automem/search/         Architecture search, diagnosis, and selection
src/automem/evaluation/     Offline aggregation and benchmark utilities
src/automem/benchmarks/     GAIA, WebWalkerQA, and xBench runner modules
src/automem/prompts/        Installed prompt resources
src/flashoagents/           Modified agent runtime used by benchmark runners
tests/                      Offline unit, integration, and smoke tests
docs/                       Architecture, configuration, reproduction guides
configs/                    Policy for version-controlled architecture inputs
examples/                   Synthetic inputs for offline control-flow smoke tests
```

Runtime data belongs in ignored directories such as `data/`, `runs/`, `storage/`,
or an external artifact store.

A strict, ready-to-validate input is provided at
[`configs/example.architecture.json`](configs/example.architecture.json).

## Install And Check

AutoMem requires Python 3.10 or newer. The default checks are offline and require
no API credentials:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,benchmarks]"

automem space
automem smoke
pytest -m "not online"
```

The `benchmarks` group installs the vector, web, and document/media dependencies
needed to execute the three benchmark runners and their offline tests. The lighter `web`
group covers web service adapters, while `vector` enables vector backends for
library-only use:

```bash
python -m pip install -e ".[benchmarks]"
```

See [Reproduction](docs/reproduction.md) for the implemented runner and search
inspection commands. Online runs require separately obtained datasets, service
credentials, and an explicit review of cost and data handling.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Example architecture](configs/example.architecture.json)
- [Reproduction](docs/reproduction.md)
- [复现说明（中文）](docs/reproduction_CN.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## Citation And License

Software citation metadata is in [CITATION.cff](CITATION.cff). AutoMem is licensed
under Apache License 2.0. Modified third-party source retains its file-level
headers; provenance limitations are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
