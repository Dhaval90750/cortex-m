# Cortex M

Cortex M studies whether a compact, explicit working gist and provenance-aware
memory improve multi-turn AI responses. It treats M2M sparse routing as a
separate, gated research track. The archived MIMO experiments are historical
evidence, not evidence of conversation-memory performance.

## Current milestone

1. Deterministic memory and retrieval prototype.
2. Local language-model benchmark against current-turn, full-history, and
   gist-only baselines.
3. Stress tests for conflicts, unsupported claims, and memory failures.

See `docs/architecture.md`, `docs/e2e_plan.md`, and `HANDOFF.md` for design and
the exact current status. Historical material is preserved in
`../Old MIMO/workspace_snapshot_2026-10-23/`.

## Quick start

```powershell
.\scripts\setup.ps1
.\scripts\python.ps1 -m unittest discover -s tests -v
.\scripts\python.ps1 data\generate.py
.\scripts\python.ps1 benchmarks\run_core_eval.py --split dev
.\scripts\python.ps1 benchmarks\run_core_eval.py --split holdout
```

The deterministic memory, retrieval, planner, and verifier do not use a local
AI model or network access. The llama.cpp model backend is optional and was not
started because free system RAM was below the project preflight threshold.
The core evaluation uses gold annotations and is not evidence of an LLM gain.
Model files and generated runs are not committed.
