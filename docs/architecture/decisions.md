# Decisions and assumptions

| Decision | Choice | Reason |
|---|---|---|
| Core/runtime split | Dependency-free memory core plus optional local CPU MIMO experiment; no hosted API | Keep fact memory usable without model dependencies while allowing the user-authorized trainable specialist research path. |
| Language handling | Explicit lexicon and deterministic phrase rules | Predictable, inspectable behavior; unknown facts are not guessed. |
| Output generation | Fixed evidence-bound templates | Every answer can be traced to a stored fact; clarification and abstention are deterministic. |
| First research task | Multi-turn factual continuity | Exercises recall, correction, conflict handling, and abstention. |
| Memory persistence | SQLite, one database per experiment/session | Transactional, inspectable, standard-library interface. |
| User fact policy | Persist only lexicon-matched explicit assertions; preserve source evidence | Avoid promoting inference into a remembered user fact. |
| Evaluation data | Versioned synthetic conversations with machine-readable labels | Enables paired controls without personal data. Labels are used for scoring and corpus validation, not runtime extraction. |
| Baselines | Current-turn rules, full-transcript rules, gist-only, Cortex-M | Isolates history access, conflict state, and provenance behavior without AI. |
| M2M placement | Optional task specialist behind explicit task routing; not a general assistant | Archived champion passes its narrow N=8 benchmark. A gist-conditioned variant has bounded causal evidence, but seed robustness and wider task coverage remain open. |
| Language-model evaluation | Not yet selected or benchmarked | Small MIMO experiments are not language models; broad chat and coding capability cannot be inferred from these results. |

## Runtime boundary

The Cortex-M fact-memory core uses Python standard-library modules only. The
optional research environment uses CPU PyTorch for the M2M specialist and loads
small trained checkpoints. Neither path calls a hosted AI service. Memory
extraction recognizes configured key/value phrases. Unsupported, missing, or
conflicted gist should stop solver dispatch and produce abstention or
clarification. Adding more fixed labels expands only that schema; it does not
add open-ended language understanding. Dynamic support-example gist is the
planned path beyond fixed mode values.

## Benchmark thresholds and reporting

Report exact task success, unsupported-answer rate, paired differences,
approximate context words, and latency. The context count is whitespace-based,
not a tokenizer measurement. The design goal is no loss against full transcript
on the controlled task with at least 30% less context. Any failure of that goal
must be stated; do not claim broad language ability or an AI performance gain.

## Archived M2M evidence

The archived M2M-35D champion is available as an explicitly invoked numeric
specialist. The new retrieved-gist variant demonstrates a conditional effect
on the synthetic two-mode task, but is not enabled for arbitrary prompts. Keep
general/default routing gated on robustness, variable-support-set evaluation,
and broader matched controls. The saved M2M-91/92/93 tables remain separate and
do not establish a causal routing benefit; see `docs/evidence_ledger.md`.
