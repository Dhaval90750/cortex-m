# Cortex-M handoff

## Objective

Research a unified Cortex-M architecture combining evidence-linked gist/memory with MIMO sparse relational routing. The executable remains a deterministic model-free memory prototype, not a general language or coding assistant. See `docs/unified_research_plan.md` for phases and gates.

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
- Phase 7: the archived M2M-35D champion has now been reproduced on its
  original N=8 synthetic benchmark (96.66% mean exact graph, 10/10 seeds above
  95%; archive reference 96.20%). Per-seed checkpoints are saved under
  `results/mimo-35d-full-checkpointed/checkpoints/`. A structured MIMO task can
  be dispatched from the same visualizer prompt. A separate gist-conditioned
  model now consumes a mode fact retrieved from Cortex-M; correction changes
  routes and output. Its synthetic benchmark has one weak seed (58.3%) and
  supports only a fixed two-mode schema. Variable demonstrations and broad
  gist/task routing remain open. No general language/coding capability is
  established.

## Main experiment

Version 2 synthetic corpus, seed 271828: 80 dev and 240 holdout cases, four
categories, eight irrelevant turns per case. Corpus hashes are in
`data/generated_v2/manifest.json`.

Latest fresh holdout run: `results/rule-20261023T181742571252Z/`.

- Current-turn rule system: 25.0% exact task success.
- Full-transcript rule system: 100.0%.
- Latest-value-only gist: 75.0%, with 25.0% unsupported-answer rate; it misses
  conflict cases.
- Cortex-M: 100.0%, 0% unsupported answers, using 72.9% fewer approximate
  context words than full transcript.
- Paired Cortex-M minus full-transcript success: 0.0 percentage points,
  percentile bootstrap 95% CI [0, 0], n=240.
- Cortex-M p95 latency was about 2.055x full-transcript p95 in this small
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
results, summary, stdout, and interpretation under `results/rule-*/`. The
walkthrough exposes each pipeline stage. The acceptance checker runs ten
functional scenarios in fresh temporary SQLite stores, reads the latest saved
benchmark summary, evaluates the gates, and saves a report under
`results/acceptance-*/`. Latest run passed 10/10 functional cases; aggregate
acceptance is partial because the latest holdout p95 latency ratio is 2.055x
against the <=2.0x gate. Detailed report: `results/acceptance-20261023T183231Z/`.

Interactive visualizer: run `scripts/visualizer.py` and open
`http://127.0.0.1:8765`. It runs the actual Cortex-M methods and presents a
recorded stage trace, response, and persisted fact history. Demo memory is in
`data/visualizer_memory.sqlite3` (gitignored); use the UI's clear action to
remove it.

## MIMO and dynamic gist experiments

The unmodified archived champion source is
`E:\Research\Old MIMO\workspace_snapshot_2026-10-23\M2M_SparseRoute_v1.py`.
Its copied benchmark is `benchmarks/mimo/m2m_sparse_route_v1.py`; the source
model/training portion matched the archive SHA-256 before artifact-saving code
was appended. Two full CPU runs produced byte-identical per-seed CSVs. The
checkpointed run records the environment, full config, means/medians, source
hash, and limitations in `results/mimo-35d-full-checkpointed/summary.json`.
Ten seed checkpoints total about 834 KB. CPU-only PyTorch is in
`E:\Research\.venv-mimo`; Cortex-M's standard-library `.venv` was not altered.

The original MIMO-only prompt route can still be launched with `--mimo-checkpoint`.
For the actually gist-conditioned prototype, launch with:

```powershell
.\scripts\mimo-python.ps1 scripts\visualizer.py --port 8767 --mimo-gist-checkpoint results\mimo-gist-conditioned\checkpoints\seed-101.pt
```

First store a mode fact through the same prompt, for example `My mode is alpha.`
Then send JSON with `task=mimo_gist_n8`, the eight numeric values, and an
eight-ID `role_ids` permutation. On each task call, Cortex-M retrieves the
current mode fact and passes its vector into the trained MIMO router. A
correction to beta changes routes and output. This experiment is a bounded
two-mode synthetic task; use `docs/dynamic_gist_methods.md` for the broader
gist-method roadmap and variable-support-set next phase.

Validation on 2026-10-24: 46/46 core/UI tests pass in `.venv-mimo` (the two
MIMO-specific visualizer tests skip in dependency-free `.venv`), and 10/10
optional MIMO tests pass. A live port-8767 run stored alpha, solved, corrected
to beta, solved again, observed changed conditioning/routes/outputs, and then
cleared both temporary facts; evidence is in
`results/mimo-gist-conditioned/live-e2e.json`. The acceptance run passed 10/10
functional cases; aggregate acceptance remains partial because the saved
synthetic holdout p95 ratio is 2.055x against the <=2.0x gate. Broad gist
generalization and causal proof for the sparse route mechanism remain open.

The unified research program and phase gates are in `docs/unified_research_plan.md`. Next, replace the two-value mode gist with a variable-size support-set gist, test on held-out rule graphs and support sizes, and compare against the existing non-neural support router. Keep broader language/code capability gated on a separately benchmarked language solver.

Latest acceptance report: `results/acceptance-20261024T005211Z/` (10/10 functional pass; aggregate partial). Latest reproduced MIMO result: `results/mimo-35d-full-checkpointed/`. Gist-conditioned MIMO and its live correction proof are in `results/mimo-gist-conditioned/`. Earlier consolidated scale and system audits remain in `results/phase-check-20261023T183413Z/`, `results/system-audit-20261023T182500Z/`, `results/scale-probe-20261023T183231Z/`, and `results/mimo93-audit-20261023T183441Z/`.

The numeric OLS proof run remains in
`results/learning/linear-regression-20261023T180514260451Z/`.
