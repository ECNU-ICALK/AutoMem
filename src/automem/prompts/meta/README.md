# Meta-Level Prompts

Prompts for the prompt-driven memory architecture optimization loop.
These are NOT used during agent task execution — they drive the outer
optimization loop that selects and improves the memory architecture.

## Files

| File | Purpose | Used By |
|------|---------|---------|
| `entity_extraction.txt` | Extract entities from agent trajectory | Phase 1 Step B |
| `relation_extraction.txt` | Build relation graph from extracted memories + entities | Phase 1 Step C |
| `task_profiling.txt` | Characterize a benchmark/task set for architecture selection | Optimization init |
| `feedback_analysis.txt` | Analyze evaluation results and generate improvement feedback | Optimization Step C |
