# Decisions and assumptions

| Decision | Choice | Reason |
|---|---|---|
| Archive boundary | Snapshot every current Research item under `Old MIMO` | User explicitly requested all existing files be moved, including the old Cortex-M v0. |
| First research task | Multi-turn factual continuity | Directly tests gist, recall, correction, conflict, and abstention together. |
| Model execution | Local only | User selected local-first and rejected hardware uncertainty as a reason to use cloud; try a smaller local model instead. |
| First model | Qwen3-1.7B GGUF Q4_K_M; fallback Qwen3-0.6B GGUF Q8_0 | Fits a CPU-first experiment more plausibly than larger models; require measured preflight. |
| Memory persistence | SQLite, one database per experiment/session | Transactional, inspectable, standard-library interface; supports provenance and state changes. |
| User fact policy | Persist explicit facts only; preserve source evidence; inferred facts are not promoted | Avoid converting a model guess into a remembered user statement. |
| Evaluation data | Synthetic conversations with machine-readable ground truth | Allows paired controls without personal data and prevents accidental leakage. |
| Baselines | Current turn, full transcript, gist-only, full Cortex M | Separates ordinary context, compact state, retrieval, planning, and verification effects. |
| M2M placement | Separate research track, integration only after corrected causal evaluation | Existing route labels and support-task validity need repair. |
| Hosted inference | Out of scope | No prompts or user data are sent to external inference services. |

## Preflight thresholds

Accept a local model only if it loads without paging, peak process RSS is at
most 4 GB, at least 2 GB system memory remains available, and a 10-prompt
short-response pilot reaches at least 5 generated tokens/second. Try the
smaller model if the first candidate fails. If both fail, continue deterministic
memory work and record model evaluation as hardware-blocked.

## Main benchmark acceptance thresholds

On held-out examples, require full Cortex M to improve exact task success by at
least 10 percentage points over full transcript, with paired bootstrap 95% CI
above zero; unsupported-claim rate at most 2% and no worse than baseline;
correction and conflict handling each at least 95%; prompt tokens at least 30%
below full transcript; and p95 latency no more than twice the full-transcript
condition. Report failures as results; do not tune on held-out cases.
