# Decisions and assumptions

| Decision | Choice | Reason |
|---|---|---|
| Mainline system | No model, no AI inference, no hosted API | Must run on the available machine without weights or external calls. |
| Language handling | Explicit lexicon and deterministic phrase rules | Predictable, inspectable behavior; unknown facts are not guessed. |
| Output generation | Fixed evidence-bound templates | Every answer can be traced to a stored fact; clarification and abstention are deterministic. |
| First research task | Multi-turn factual continuity | Exercises recall, correction, conflict handling, and abstention. |
| Memory persistence | SQLite, one database per experiment/session | Transactional, inspectable, standard-library interface. |
| User fact policy | Persist only lexicon-matched explicit assertions; preserve source evidence | Avoid promoting inference into a remembered user fact. |
| Evaluation data | Versioned synthetic conversations with machine-readable labels | Enables paired controls without personal data. Labels are used for scoring and corpus validation, not runtime extraction. |
| Baselines | Current-turn rules, full-transcript rules, gist-only, Cortex-M | Isolates history access, conflict state, and provenance behavior without AI. |
| M2M placement | Separate, not integrated | Existing routing summaries do not pass the corrected multi-seed causal gate. |
| LLM evaluation | Out of current scope | The current objective is model-free software and benchmarking. |

## Model-free system boundary

The program uses Python standard-library modules only. It does not load model
weights, call an AI service, or need network access. Extraction recognizes
configured key/value phrases. Unsupported or ambiguous wording should produce
no stored fact, followed by abstention or clarification. Expanding the lexicon
changes coverage; it does not add general language understanding.

## Benchmark thresholds and reporting

Report exact task success, unsupported-answer rate, paired differences,
approximate context words, and latency. The context count is whitespace-based,
not a tokenizer measurement. The design goal is no loss against full transcript
on the controlled task with at least 30% less context. Any failure of that goal
must be stated; do not claim broad language ability or an AI performance gain.

## Archived M2M evidence

Keep M2M out of Cortex-M response path unless a corrected benchmark provides
repeatable compositional gain, matched controls, and causal route evidence.
The saved M2M-91/92/93 tables do not meet that gate; see
`docs/evidence_ledger.md`.
