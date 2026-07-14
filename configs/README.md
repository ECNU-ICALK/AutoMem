# Configurations

This directory is reserved for small, version-controlled public architecture
inputs. It must never contain credentials, datasets, generated worker
`runtime_config.json` files, memory pools, task traces, indexes, or results.

## Architecture Document

AutoMem exposes one JSON shape:

```json
{
  "schema_version": "1",
  "encode": ["tip", "trajectory", "workflow"],
  "store": "graph",
  "retrieve": "contrastive",
  "manage": "lightweight"
}
```

The checked-in [`example.architecture.json`](example.architecture.json) is this
valid multi-Encode document (the shape of the paper's discovered GAIA route) and
can be used as the starting point for a run record.

The option registry is `automem-esrm-v1`: five encoders, five stores, six
retrievers, and four managers. A document selects a non-empty subset of encoders
(a single string is accepted as shorthand for a one-type subset) and exactly one
value for each other coordinate; all selected encoders share the one selected
store. The four manager names are `lightweight`, `json_full`, `tool_manager`, and
`graph_consolidate`.

Files should be validated through `ArchitectureSpec.from_dict`. There is no public
YAML configuration resolver or configuration CLI in the current package.

## Rules

- Keep the strict five-field schema; unknown fields are errors.
- Do not add G2 context composition, G3 refresh, or G4 query-planning options.
  Those behaviors are fixed by `automem-runtime-v1`.
- `graph` retrieval requires graph-family storage.
- `graph_consolidate` requires graph-family storage and the `graph` retriever.
- Keep credentials and service endpoints in local secret management; use
  `.env.example` only as a name reference.
- Do not commit host-specific absolute paths, private endpoints, or backup copies.
- Treat a released architecture file as immutable; add a new named file for a new
  selection.

See [Configuration](../docs/configuration.md) for the exact contract and
[Reproduction](../docs/reproduction.md) for run records.
