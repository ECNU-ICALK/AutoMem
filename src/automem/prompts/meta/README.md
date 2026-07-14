# Meta-Level Prompts

Prompts that drive the outer architecture-search loop. They are NOT used
during agent task execution — the task-time prompts live one directory up
(`prompts/*_prompt.txt`) and inside the runtime modules.

Every file in this directory is hashed into the evaluation-protocol digest,
so editing any of them invalidates `--resume` for in-flight runs.

## Files (in call order within one search round)

| File | Purpose | Called from |
|------|---------|-------------|
| `architecture_search.txt` | Proposer: generate this round's candidate architectures from the Pareto front, pool stats, ledger, and last round's diagnosis | `search/engine.py` (every round) |
| `layer_diagnosis_fixed.txt` | Per-candidate 4-layer LLM diagnosis (Encode/Store/Retrieve/Manage) producing a `priority_action` inside the public search space | `search/attribution.py` |
| `diagnosis_synthesizer.txt` | Per-candidate synthesis of 6 diagnostic signals into one verdict for the next-round proposer | `search/diagnosis_synthesizer.py` |
| `differential_diagnosis.txt` | Best-vs-worst candidate comparison for the round | `search/attribution.py` via `search/engine.py` |
| `round_level_synthesizer.txt` | Round-level verdict (explore/exploit stance, next-round focus, stop recommendation) | `search/diagnosis_synthesizer.py` |
| `ledger_update.txt` | Experience-ledger delta: new principles, evidence updates, dead ends, open questions | `search/experience_ledger.py` |

All six prompts operate on the same public space `automem-esrm-v1`:
Encode is a non-empty subset of {tip, insight, trajectory, workflow,
shortcut}; Store/Retrieve/Manage are single choices; every selected encode
type is routed to the one selected store.
