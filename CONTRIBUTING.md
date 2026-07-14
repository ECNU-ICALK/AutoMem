# Contributing

## Scope

Contributions should preserve the separation between the canonical architecture
space, fixed runtime policy, benchmark adapters, and generated artifacts. Changes
to public behavior require tests and an explicit schema, space-ID, or runtime-policy
version assessment.

## Development Workflow

1. Create a focused branch from the current default branch.
2. Add or update offline tests before changing behavior.
3. Run compilation, CLI smoke checks, unit tests, and relevant integration tests.
4. Verify that no credentials, datasets, model weights, caches, or run outputs are
   staged.
5. Explain public API, configuration, and reproducibility effects in the pull
   request.

The required local checks are:

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src
automem space
automem smoke
pytest -m "not online"
```

## Architecture-Space Changes

`automem-esrm-v1` is immutable. Do not add an option to it in place. A changed
Encode/Store/Retrieve/Manage menu requires a new `space_id`, updated validation,
documentation, and a clear compatibility note.

G2 context composition, G3 refresh control, and G4 query planning are fixed by
`automem-runtime-v1`; they are not search coordinates or configuration options.
Success-aware graph edge adaptation is part of the `graph_consolidate` manager, not
a fifth coordinate or a separate management preset.

## Tests

- Unit tests must be deterministic and offline.
- Regression tests should encode a confirmed failure, not an implementation detail.
- Integration tests may use synthetic stores and fake model clients.
- Network and paid-service tests must carry the `online` marker and remain opt-in.

## Third-Party Code

Do not copy external code, prompts, datasets, or figures without recording its
source, exact revision, license, copyright notice, and modification history. Update
`THIRD_PARTY_NOTICES.md` in the same pull request. When legacy provenance is
incomplete, state that limitation explicitly instead of inventing a revision.
