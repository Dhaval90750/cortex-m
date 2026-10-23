# End-to-end phases and gates

## Phase 0 — Preserve and organize

**Status:** archived and verified. The user later requested removal of the
archived M2M venv. Automatic safety review rejected recursive deletion, so the
venv remains and its manifest rows remain accurate.

## Phase 1 — Reproducible project and runtime

Implement the source tree, local Python runner, project-local tests, run
manifests, and handoff. A fresh Python venv is at `../.venv` in the Research
root; the Cortex-M runner uses it. Local model setup is blocked at preflight:
about 1.5 GB free RAM is below the 2 GB reserve, so no model was downloaded.

**Gate:** core tests pass; endpoint/model gate remains blocked until adequate
memory is available. No network inference calls were made.

## Phase 2 — Memory and gist API

Implement SQLite event storage, explicit-only facts, source evidence,
correction/conflict rules, clearing, lexical retrieval, 512-token gist,
response planning, claim verification, and safe fallbacks.

**Gate:** unit tests cover provenance, corrections, conflicts, inference
rejection, clearing, ordering, serialization, database reopen, and outage.

**Execution status:** complete; 16 unit/stress tests pass, including the listed
state transitions, database reopen, record serialization round-trip, ordering,
and fail-closed outage/corruption behavior.

## Phase 3 — Controlled data and baselines

Generate 80 development and 240 held-out conversations, 60 each across recall,
correction/change, ambiguity/conflict, and unknown/inference/outage cases. Keep
templates and fact combinations disjoint across splits. Freeze prompts after
development-set iteration. Store JSONL plus generator seed and SHA-256.

**Gate:** all expected answers/actions match the annotations; no held-out data
is used for tuning; all four conditions see the same case ids.

**Execution status:** versioned corpus generated (seed 271828, 80 dev and 240
holdout) and deterministic Cortex-M core checked at 80/80 and 240/240. This
uses annotations as input and is not the four-system language comparison.

## Phase 4 — Local language integration

Implement model-assisted fact extraction with structured output and source-span
validation, then local generation through llama.cpp. Test current-turn-only,
full transcript, gist-only, and full Cortex M. Use identical model, context,
decoding, and output budget; capture per-case prompt and result metadata.

**Gate:** extraction scores are reported separately from memory/retrieval
scores; local model generation succeeds and stays within the preflight budget.

## Phase 5 — Paired evaluation

Run all four systems on the held-out split. Compute exact task success, fact
recall, unsupported-claim rate, correction/conflict handling, abstention and
retrieval precision/recall, token counts, and latency. Produce paired bootstrap
95% intervals and per-category tables, plus every case-level response.

**Gate:** meet the thresholds in `decisions.md`; otherwise report the failed
criterion and preserve the held-out set untouched.

## Phase 6 — Stress and failure tests

Cover long histories, topic shifts, ambiguous references, contradiction after
retrieval, inferred-only details, adversarial/irrelevant memory, corruption,
retrieval timeout, database outage, context pressure, and unsupported requests.

**Gate:** each failure produces the specified clarify/abstain/fallback behavior
and never becomes a confident unsupported fact.

**Execution status:** partial deterministic stress coverage exists for long
irrelevant histories, adversarial evidence, irrelevant conflicts, ambiguous
queries, corrupted SQLite, context budget, unsupported values, and outage.
Remaining stress cases need expansion before this phase is complete.

## Phase 7 — M2M revalidation and possible integration

Repair support/query task alignment and route targets; compare Sparsemax-K with
matched dense, fixed, random, and oracle controls. Screen five seeds, confirm
ten. Run paired route perturbation and ablation tests. Integrate only if the
M2M gate in `decisions.md` is met; otherwise retain it as a separate research
result.

**Gate:** repeatable held-out compositional gain, causal route evidence, and
robustness to the prescribed perturbations.

**Execution status:** reviewed archived M2M-91/92/93 summary CSVs. The route
integration gate is not met; see `docs/evidence_ledger.md`. This was not a
fresh corrected benchmark, so M2M remains separate and no new causal claim is
made.

## Run artifact contract

Each `results/<run-id>/` contains `config.json`, `environment.json`,
`dataset_manifest.json`, `case_results.jsonl`, `summary.json`,
`stdout.log`, and `interpretation.md`. `HANDOFF.md` is updated after each phase
with the next single action.
