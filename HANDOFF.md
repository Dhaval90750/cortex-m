# Cortex-M handoff

## Objective

Study compact factual memory, correction/conflict handling, provenance-aware
retrieval, and safe replies using a model-free, deterministic system. No AI
model or external API is part of the current design.

## Workspace and environment

- Historical material: `E:\Research\Old MIMO\workspace_snapshot_2026-10-23`.
- Fresh root venv: `E:\Research\.venv`, Python 3.12.10; Cortex-M uses it.
- Runtime dependencies: Python standard library only. No torch, model weights,
  or network calls.
- The archived venv deletion was requested, but automatic safety review
  rejected recursive deletion. It remains present; its archive manifest is
  unchanged.

## Phase status

- Phases 0-5: complete for the model-free research scope.
- Phase 6: partial; see `docs/e2e_plan.md` for tested failure classes and gaps.
- Phase 7: M2M is not integrated. Archived route evidence does not meet the
  repeatable causal gate; no new M2M run was performed.

## Main experiment

Version 2 synthetic corpus, seed 271828: 80 dev and 240 holdout cases, four
categories, eight irrelevant turns per case. Corpus hashes are in
`data/generated_v2/manifest.json`.

Latest holdout run: `results/rule-20261023T174812718078Z/`.

- Current-turn rule system: 25.0% exact task success.
- Full-transcript rule system: 100.0%.
- Latest-value-only gist: 75.0%, with 25.0% unsupported-answer rate; it misses
  conflict cases.
- Cortex-M: 100.0%, 0% unsupported answers, using 72.9% fewer approximate
  context words than full transcript.
- Paired Cortex-M minus full-transcript success: 0.0 percentage points,
  percentile bootstrap 95% CI [0, 0], n=240.
- Cortex-M p95 latency is about 2.16x full-transcript p95 in this small
  microbenchmark; the absolute times are below 1 ms and susceptible to timer
  noise. This relative latency criterion did not pass.

This is a controlled phrase/lexicon benchmark, not general language
understanding and not evidence of an AI performance gain. The consolidated
methods and findings are in `docs/research_paper.md`.

## Reproduce

From `E:\Research\Cortex-M`:

```powershell
.\scripts\setup.ps1
.\scripts\python.ps1 -m unittest discover -s tests -v
.\scripts\python.ps1 data\generate.py
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split dev
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split holdout
.\scripts\python.ps1 scripts\demo.py
```

Each benchmark run saves configuration, runtime, corpus manifest, all case
results, summary, stdout, and interpretation under `results/rule-*/`.

## Single next action

Expand Phase 6 with paraphrase, quoted speech, topic shift, and long-session
stress cases while preserving the current v2 holdout unchanged.
