# Cortex M handoff

## Objective

Test whether explicit compact memory, provenance-aware retrieval, response
preparation, and verification improve multi-turn fact continuity over a
matched language-model baseline. Keep all model calls local. M2M routing stays
separate until its benchmark and causal evidence pass the integration gate.

## Workspace history

The previous Research workspace is under
`E:\Research\Old MIMO\workspace_snapshot_2026-10-23`; its SHA-256 inventory is
`E:\Research\Old MIMO\archive_manifest.csv`. The user asked to delete the
archived `M2M_Network\.venv`, but automatic safety review rejected recursive
deletion. It remains, and the inventory still describes current contents.

## Current phase and next action

Phases 0-2 are complete. The Phase 3 corpus and annotated-core checks are
complete; its four-system baseline comparison is pending. A fresh Python
3.12.10 environment is at `E:\Research\.venv`; the project runner uses it.
Phase 4 model evaluation is hardware-blocked at preflight: about 1.5 GB RAM
was free, below the required 2 GB reserve. No model was downloaded or started.
Phase 6 has partial deterministic stress coverage. Existing M2M route evidence
does not pass the integration gate. Next: wait for enough free RAM to run the
local model preflight; LLM comparisons remain blocked until then.

## Environment

- Fresh Cortex-M venv: `E:\Research\.venv`, Python 3.12.10.
- Cortex-M core uses only Python standard-library modules; no PyTorch or model
  packages were installed.
- Around 7.7 GB total RAM and about 1.5 GB free at preflight; no NVIDIA GPU.
- LLM backend is local llama.cpp only. No hosted inference calls are allowed.

## Commands

From `E:\Research\Cortex-M`:

```powershell
.\scripts\python.ps1 -m unittest discover -s tests -v
.\scripts\python.ps1 data\generate.py
.\scripts\python.ps1 benchmarks\run_core_eval.py --split dev
.\scripts\python.ps1 benchmarks\run_core_eval.py --split holdout
```

Generated data is under `data/generated/`; run artifacts are under
`results/core-<UTC timestamp>/`. The deterministic core evaluation uses
annotated facts. It is not an LLM or baseline comparison. Do not load a local
model until memory meets the limits in `docs/decisions.md`.

## Decisions and evidence

- Target: multi-turn factual continuity, correction, conflict handling, and
  safe abstention.
- Unit suite: 16/16 passed on the fresh root environment.
- Deterministic core check: dev 80/80; holdout 240/240. This checks annotated
  storage, retrieval, action selection, and outage behavior only; it does not
  demonstrate language-model gains.
- Dataset seed: 271828. Split hashes are stored in
  `data/generated/manifest.json`.
- The held-out split has not been used for prompt tuning.
- Previous MIMO results and their limits are recorded in
  `docs/evidence_ledger.md`.

## Handoff update protocol

After each completed phase, update this file with status, command,
runtime/model version, results path, failures, and the single next action.
