# Third-Party Notices

This repository includes modified agent-runtime source under
`src/flashoagents/` and related benchmark/evaluation integration files. It does not
include external baseline implementations.

## FlashOAgents Runtime Snapshot

The included snapshot was migrated from the pre-existing AutoMAS project tree while
creating the clean AutoMem repository. The source files themselves identify the
following copyright and license lineage:

- files including `agents.py`, `models.py`, `memory.py`, `tools.py`, `utils.py`,
  `monitoring.py`, `agent_types.py`, and `_function_type_hints_utils.py` carry
  2024/2025 Hugging Face copyright notices and Apache License 2.0 headers;
- several of those files state that portions were modified by the OPPO PersonalAI
  Team under Apache License 2.0; and
- `base_agent.py`, `search_tools.py`, `mm_tools.py`, `mm_tools_utils.py`, the
  benchmark runners, and selected evaluation helpers carry 2025 OPPO PersonalAI
  copyright notices and Apache License 2.0 headers.

The per-file copyright and Apache headers have been preserved. The repository's
root `LICENSE` contains the Apache License 2.0 text.

The migrated snapshot did not contain a reliable machine-readable record of the
original external repository URL and exact upstream revision. This notice therefore
does not claim an upstream commit pin or byte-for-byte correspondence with a named
release. The YAML prompt resources shipped with `src/flashoagents/` have the same
snapshot-level provenance limitation. Establishing and recording an exact upstream
revision remains required before claiming exact upstream reproducibility.

AutoMem changes to the snapshot include package/import relocation, installed-resource
path handling, structured search-to-runner configuration transport, integration with
the AutoMem provider lifecycle, and the fixed query-planning, context-composition,
and refresh behavior described in `docs/architecture.md`.

## Baselines

AutoMem does not vendor baseline repositories. A baseline must be obtained from its
official source and used under its own license. Any reported comparison should
record that source, its exact revision, and the evaluation protocol separately.

## Python Dependencies

Packages declared in `pyproject.toml` are installed as separately distributed
dependencies; their source is not redistributed by this repository. Their own
licenses and notices apply. A release lock file or software bill of materials should
be generated for a tagged, deployable environment.
