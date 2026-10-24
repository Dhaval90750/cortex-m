# Cortex-M unified research and implementation plan

**Updated:** 2026-10-24  
**Purpose:** Carry the original MIMO sparse-routing research and the Cortex-M gist/memory prototype as one research program. Keep component experiments separate until a matched integrated experiment passes its gate.

## Goal and current boundary

The long-term goal is a general-purpose assistant that can learn from supplied material, retain useful state, reason across requests, and use tools with verifiable results. “Universal” is an aspiration, not a currently testable guarantee; no finite benchmark establishes human-level generality.

The default core is a deterministic, model-free memory/retrieval prototype. An optional local PyTorch checkpoint now runs one archived N=8 relational task through the same single-prompt visualizer path. Neither component understands arbitrary language, learns coding from examples, or generates code. Archived MIMO/M2M networks solve constructed numeric relational-routing tasks, not general language tasks. Under a strict no-model/no-API constraint, the finite rule system cannot become a broad conversational and coding assistant.

## Unified working hypothesis

Use a compact evidence-linked gist as a candidate working-state interface:

1. Observe input/tool output and identify candidate facts, goals, constraints, and questions.
2. Ground each candidate with source, time/turn, explicitness, and confidence; preserve corrections and unresolved conflicts.
3. Retrieve bounded relevant memories and current context.
4. Prepare a task state: goal, evidence, missing information, and action (`answer`, `clarify`, `abstain`, or `tool`).
5. Compute with the selected solver. MIMO sparse routing is one candidate for relational/numeric computation, not a universal reasoner.
6. Verify claims against evidence and tool output; test artifacts before reporting success.
7. Update memory only with provenance and an auditable transition.

Gist/memory and MIMO meet at the selection/compute boundary: gist could condition routing, while routing could select task-specific relational operations. Existing experiments have **not** shown that this integration improves a language model or general assistant. Validate components independently, then test interaction against matched baselines.

## Archived evidence and limits

### MIMO/M2M

The historical master record reports M2M-35D Sparsemax-K at 96.20% mean exact-graph accuracy over ten seeds on a constructed N=8 hidden-rule task and 94.50% over five seeds at N=16; N=32 is unresolved. It reports 34.37% for independent sigmoid gating and 84.00% for Softmax-K on the N=8 setup. These are historical bounded results, not general-reasoning evidence.

Later route interventions are mixed:

- M2M-91: 100% routing-score perturbation reduced top-half overlap to ~0.493 while MSE improved ~0.176% (reported interval half-width ~0.078 percentage points).
- M2M-92: hard-top1 and hard-worst raised MSE ~1.04% and ~2.91%; zero-top1 raised it ~13.07%.
- M2M-93: rank-16 perturbation raised MSE ~2.91%; smaller interventions were mostly below 1%.
- `MIMO-M2M-final` has a questionable pair-routing target; the archived audit finds it unsuitable for routing-accuracy claims.
- SLM-01 is a small synthetic Transformer proxy with a support/query target mismatch, not a language-model result.

Do not conclude either “routing is causal” or “routing does nothing.” Reproduce and repair the task and metrics before integration.

### Cortex-M memory

The current core stores a narrow class of explicit facts in SQLite, tracks provenance/corrections/conflicts, retrieves lexically, and uses bounded evidence-grounded templates. Its 240-case holdout is a developer-generated grammar: 100% matched the full-transcript rule system with ~72.9% fewer approximate context words, but does not test open-language understanding. The automated suite has 44 passing tests; 10 functional acceptance cases pass. A synthetic 10,000-record SQLite probe survives reopen and exact-key retrieval but shows ~0.92 s full-scan retrieval. Latest acceptance is partial: p95 ratio 2.055x against a <=2.0x threshold.

See [system audit](system_audit.md), [research paper](research_paper.md), [evidence ledger](evidence_ledger.md), and `../results/system-audit-20261023T182500Z/`.

## Phases and advancement gates

### A — Freeze evidence and map archive (complete; maintenance only)

Preserve `E:\Research\Old MIMO`; do not relocate/delete it. Link claims to master record, scripts, result tables, checkpoints, and validity status. Keep historic venv separate from root `.venv`.

**Gate:** every headline result labeled reproduced, historical/unreproduced, invalid proxy, or unresolved.

### B — Reproducible baselines (memory v0 and archived M2M-35D reproduction complete)

Keep the automated suite and 10-case acceptance run reproducible. Re-run N=8 routing conditions with common data generation, fixed splits, matched parameter/training budgets, and at least ten declared seeds; report graph accuracy, numeric error, compute, and failures. Keep N=16 as a replication/scaling point; N=32 remains exploratory until multiple seeds finish.

**Gate:** Cortex-M's controlled memory baseline and archived N=8 M2M-35D champion have reproducible commands, source hash, declared seeds, result CSV, checkpoints, and summaries. The archived champion was run twice with identical per-seed CSV output. This closes reproduction for that benchmark only.

### C — Correct MIMO benchmark and causal routing (champion reproduced; causal validation remains open)

1. Generate support/query examples from the same latent rule and output transform.
2. Verify route labels from the generator, not hidden inference-time values.
3. Compare learned routes with fixed, uniform, random, oracle, and matched dense/soft alternatives under equal budgets.
4. Run paired route interventions on identical inputs/checkpoints. Measure route change separately from output error; add restoration controls and multiple seeds.
5. Test unseen permutations/distributions/compositions, more modes/relations, and controlled N scaling.

**Gate:** still open. The reproduced model clears its historical score (96.66% vs 96.20% mean exact graph) on the same fixed synthetic task, but the rerun has no matched dense/soft/random/no-context retraining baselines and does not establish causal routing. The diagnostic context classifier is near chance (53.84%). Keep MIMO as a bounded experimental specialist, not a general reasoner.

### D — Memory/gist coverage and durability (next Cortex-M work)

Create independently annotated, held-out cases for multi-session recall, paraphrases, temporal updates, multi-hop relations, ambiguity, stale facts, duplicates/conflicts, corrections, forgetting, long irrelevant history, multilingual input, and unknown requests. Add migration, backup/restore, soak, crash, and multiprocess testing; index retrieval instead of re-tokenizing all rows.

Measure extraction/retrieval precision and recall, supported-answer accuracy, temporal/correction accuracy, abstention precision/recall, unsupported-claim rate, recovery, latency, and memory. Publish category errors and intervals. A test generator and matching rules are not independent validation.

**Gate:** tested memory transitions and held-out improvement over no-memory and simple transcript/latest-value baselines without more unsupported answers.

### E — Solver-resource decision

- **Strict no-model/no-API:** continue as transparent memory, numeric, and deterministic tools; define tested task families and reject unsupported tasks. Do not claim general-assistant ability.
- **Broad language/code goal:** benchmark candidate language models on this machine before selection. The optional local environment currently contains CPU PyTorch for small MIMO experiments, but no pretrained language model has been installed or benchmarked. Inventory: 7.7 GiB RAM, i5-1135G7, Iris Xe, MX350. Compare task quality, resources, latency, and privacy before choosing a language solver. Keep hosted APIs out unless separately chosen.

**Gate:** record inference-resource policy, hardware/latency budget, privacy boundary, and tasks; measure candidate before integration. Broad assistant capability requires a learned language solver plus state, retrieval, tool routing, safe execution, and verification.

### F — Unified gist × MIMO experiment (orchestration prototype exists; evidence gate open)

On identical held-out tasks and one frozen solver/model, compare: no persistent gist; full transcript; structured gist/retrieval; gist + MIMO; gist + matched non-MIMO specialist. Match data, prompts, model, decoding, tool access, and compute/token budget. Ablate gist fields, provenance, retrieval, routing, and verification. Score task success, evidence support, abstention, intervention effects, tool success, latency, memory, and failures.

**Gate:** open. A trained gist-conditioned MIMO variant consumes a mode fact retrieved from Cortex-M memory; changing/correcting that fact changes the conditioning vector, routes, and outputs. On a fixed synthetic two-mode N=8 task, correct gist averaged 95.83% exact graph (9/10 seeds at 100%, one at 58.33%); no-gist averaged 0.017% and flipped gist 0%. This is evidence that the supplied mode signal affects the model on that task. A variable support-set gist path is also integrated as a transparent non-neural comparator over 168 predefined relations and passed a four-relation/32-example test. A learned variable-size gist encoder, free-form task parsing, and general routing remain open. A separate direct-route neural variant was unstable (74.95% average) and remains a preserved failed experiment.

### G — Tools, coding, safety, release

After solver selection, add allowlisted tools and a disposable workspace sandbox. Coding loop: specification, patch, formatter/linter, immutable tests, regressions, diff review, honest report. Test prompt injection in documents, unsafe requests, secrets, file/network limits, resource limits, crashes, and timeouts. Require independent red-team and usability work before release.

Fallbacks: missing evidence -> ask/abstain; conflict -> show both with provenance; storage failure -> say memory unavailable and never claim save; model unavailable -> explicit deterministic/offline mode; tool failure -> show error and preserve artifacts; failed sandbox/test -> never claim code succeeded; unsupported task -> say so and offer a tested action.

## Artifact contract

For every experiment save hypothesis, phase/gate, code revision/hash, environment/dependencies, seeds, data/split hashes, config, per-example outputs, metrics/intervals, command/stdout/stderr, failures, interpretation, and limitations. Separate measurement from explanation; retain failures and create timestamped reruns. Keep personal UI memory out of benchmark data.

## Immediate sequence

1. Automated checks: 46/46 core/UI tests and 10/10 optional MIMO integration tests passed on 2026-10-24; functional acceptance remains partial at 10/10 because a separate latency gate failed. `git diff --check` also passed.
2. Reproduce matched dense, soft, random, and no-context MIMO baselines with equal training budgets and declared seeds.
3. Run paired causal routing interventions and unseen-distribution/permutation tests against the champion.
4. Design a valid test where gist evidence actually conditions a task solver; compare against transcript, gist-only, and matched specialist controls.
5. Expand independent memory/gist coverage and improve 10K-query scaling.
6. Keep broad-language capability gated on a separately benchmarked learned language solver; the current MIMO specialist is not that solver.
