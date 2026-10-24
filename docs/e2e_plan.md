# Cortex-M execution phases

## Phase 0 — Preserve and organize

Prior research is in `../Old MIMO/workspace_snapshot_2026-10-23/`. Its file
inventory is in `../Old MIMO/archive_manifest.csv`. The archived venv deletion
was requested but blocked by the app's automatic safety review; it remains.

**Status:** archived and verified, with the user-requested deletion limitation
recorded in `HANDOFF.md`.

## Phase 1 — Reproducible runtime

Create the fresh root environment at `E:\Research\.venv`. Cortex-M uses Python
standard-library modules and does not need torch, llama.cpp, model weights, or
network access. `scripts/setup.ps1` creates/verifies the environment and
`scripts/python.ps1` runs commands in it.

**Status:** complete. Test and setup commands are recorded in `README.md`.

## Phase 2 — Memory, retrieval, and safe actions

Implement SQLite facts with source text, explicit-only storage, correction and
conflict transitions, lexical retrieval, bounded gist, response plans, a
template renderer, and fail-closed behavior for unavailable or corrupt storage.

**Status:** 44 total tests pass for transitions, retrieval, source preservation,
ordering, serialization, clearing, conflict/correction, context budget,
unsupported values, unknown queries, and outage/corruption handling.

## Phase 3 — Controlled data

Generate version 2 with seed 271828: 80 development and 240 held-out cases,
20/60 per category across recall, correction, conflict, and unknown/outage.
Each case has eight irrelevant turns after its fact statements. Dev/holdout
fact combinations and primary phrasing pools are disjoint. The rule extractor
must match the fact annotations before benchmark scoring starts.

**Status:** complete. Hashes are in `data/generated_v2/manifest.json`.

## Phase 4 — Model-free extraction and response

Use a bounded, configured key/value lexicon, exact phrase rules, SQLite memory,
and fixed response templates. Unsupported phrases remain unknown. No model or
API is called.

**Status:** implemented in `src/cortex_m/rules.py`, `assistant.py`, and
`render.py`; the corpus validator confirms extracted facts match annotations.
The visualizer now has one input and automatically routes supported statements
and questions. This UI convenience does not add general language understanding.

## Phase 5 — Paired deterministic evaluation

Compare four conditions on identical case ids: current-turn rules,
full-transcript rules, a gist that keeps only the latest value, and full
Cortex-M with correction/conflict state. Report exact success, unsafe answer
rate, approximate context words, paired bootstrap interval, and latency.

**Status:** complete on dev and holdout. Results are saved under
`results/rule-*/`. The holdout run is the primary result. See
`docs/research_paper.md` for analysis and limits.

## Phase 6 — Stress and failure tests

Cover long histories, topic shifts, ambiguous references, contradictions,
inferred-only details, adversarial/irrelevant memory, corruption, outage,
context limits, unsupported requests, and unknown lexicon entries.

**Status:** partial. Current deterministic tests include 100 unrelated facts,
adversarial evidence, unrelated conflict, unknown query, corrupt database,
outage, context budget, and unsupported value. Expand tests for more varied
phrasing and long-lived database sessions before production use.

## Phase 7 — Unified MIMO × gist research

MIMO sparse routing and compact gist/memory are one Cortex-M research program,
but remain separately measured until a valid integration experiment is run.
See `unified_research_plan.md` for the shared hypothesis, component baselines,
corrected benchmark, causal routing gate, and matched gist × MIMO experiment.
Archived M2M-91/92/93 interventions are mixed and do not establish causal
benefit; historical MIMO/SLM proxy concerns are in `system_audit.md`.

**Status (2026-10-24):** the archived M2M-35D champion has been fully rerun on
its original N=8 benchmark (96.66% mean exact graph over ten seeds; matching
CSV hash across two runs), and checkpoints are saved. A trained gist-conditioned
variant now receives a mode fact retrieved from Cortex-M on every task request.
True, missing and flipped-gist controls show a large bounded synthetic-task
effect, but one of ten seeds is weak and the runtime schema remains alpha/beta.
Broad dynamic gist and sparse-routing mechanism gates remain open. See
`unified_research_plan.md` and `dynamic_gist_methods.md` for next steps.

## Artifact contract

Each benchmark output directory includes config, environment, dataset manifest,
case-level JSONL, summary, stdout log, and interpretation. Current runs are
retained locally and the research paper consolidates the primary results.
