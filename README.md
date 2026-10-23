# Cortex M

Cortex-M is a model-free memory assistant built from explicit phrase rules,
SQLite provenance, conflict-aware retrieval, and fixed response templates. It
uses no local model, external AI API, or network call. The archived MIMO
experiments are historical research, separate from this system.

## Current milestone

1. Deterministic memory and retrieval.
2. Model-free phrase extraction and fixed response rendering.
3. Paired comparison against current-turn, full-transcript, and gist-only
   rule-based baselines.
4. Stress tests for conflicts, unsupported values, and memory failures.

See `docs/architecture.md`, `docs/e2e_plan.md`, and `HANDOFF.md` for design and
the exact current status. Historical material is preserved in
`../Old MIMO/workspace_snapshot_2026-10-23/`.

## Quick start

```powershell
.\scripts\setup.ps1
.\scripts\python.ps1 -m unittest discover -s tests -v
.\scripts\python.ps1 data\generate.py
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split dev
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split holdout
.\scripts\python.ps1 scripts\demo.py
```

`RuleBasedCortex` recognizes only configured explicit phrases and values. It
stores provenance, resolves corrections, identifies conflicts, and abstains on
unknown facts. The benchmark's labels are used for validation/scoring, not for
extraction or response generation. Read `docs/research_paper.md` for results
and limitations.
