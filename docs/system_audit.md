# Cortex-M system audit

**Audit date:** 2026-10-23  
**Audited build:** model-free Cortex-M memory prototype, single-prompt visualizer,
and a separate classical numeric-regression workbench.  
**Overall verdict:** the bounded memory prototype is reproducible and the
single-prompt front end now routes supported statements and questions. It is
not a general assistant, coding model, or ChatGPT competitor. Capabilities not
implemented are called out below rather than counted as passing tests.

## Scope and method

The audit uses the implementation and saved outputs in this repository, a
44-test standard-library suite, the generated v2 development and holdout
benchmarks, a 10,000-record SQLite scale probe, and live HTTP tests against the
visualizer API. The browser-facing interface was also opened and exercised with
a code-generation request. The synthetic benchmark is developer-authored and
its grammar is bounded; it does not represent general user prompts.

External research informs the missing evaluation coverage. LongMemEval tests
information extraction, multi-session and temporal reasoning, knowledge
updates, and abstention across 500 questions and scalable histories. LoCoMo
adds long conversations (about 300 turns / 9K tokens on average), event
summaries, temporal and causal questions, and multimodal dialogue. Cortex-M's
current corpus covers a much smaller subset: explicit fact recall, correction,
conflict, and unknown cases. [LongMemEval paper](https://arxiv.org/abs/2410.10813),
[LoCoMo paper](https://arxiv.org/abs/2402.17753).

For coding, SWE-bench Pro measures long-horizon repository engineering across
multiple files and requires running task-specific tests. Cortex-M currently
has no code-generation model, repository agent, patch loop, or execution
sandbox, so passing its memory tests says nothing about coding ability.
[SWE-bench Pro paper](https://arxiv.org/abs/2509.16941).

NIST's AI RMF and Generative AI Profile call for lifecycle risk management and
repeatable test, evaluation, verification, and validation. NIST's 2026 ARIA
evaluation manual distinguishes model testing, red teaming, and user testing.
This audit covers code-level functional tests and limited API red teaming; it
does not substitute for independent user studies or deployment monitoring.
[NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework),
[NIST GenAI Profile](https://doi.org/10.6028/NIST.AI.600-1),
[NIST ARIA Evaluation Planning Manual](https://doi.org/10.6028/NIST.AI.200-3).

## Phase-by-phase implementation audit

| Phase / area | Built and directly verified | Missing or limit | Audit status |
|---|---|---|---|
| Runtime and dependencies | Python 3.12.10, standard-library runtime; no model client, hosted API, or required third-party package | No deployment packaging, installer, or supported-device matrix | Implemented for research prototype |
| Explicit fact capture | Configured keys/values plus a narrow open-value personal-name pattern; uncertainty and negation filters; source text and turn provenance | No broad language understanding; arbitrary documents and paraphrases are not reliably extracted | Partial |
| Memory persistence | SQLite facts, corrections, conflicts, reopen persistence, clear operation, audit events | One local database/profile; no encryption, retention controls, or secure-erasure guarantee; stored evidence is plaintext | Partial |
| Retrieval and response | BM25-like lexical scoring, exact-key preference, conflict clarification, evidence-bound fixed templates, abstention and storage fallback | Ranking rebuilds token data from every fact on each query; no semantic search, temporal reasoning, session summaries, or general reasoning | Partial |
| User interface | One prompt field; automatic statement/question routing; exact same application path exercised over localhost HTTP | Not yet a conversational transcript UI; no streaming, attachments, voice, tools, or general intent classification | Single-prompt interaction implemented; chat product incomplete |
| Classical training | OLS regression from numeric CSV, deterministic split, train-only scaling, saved JSON model, held-out metrics and prediction CLI | This trains a numeric regression model only; it does not train Cortex-M to understand language or write code | One task family complete as v1 |
| Code assistance | Unsupported requests are explicitly rejected in the one-prompt UI/API | No code generation, repository browsing/editing agent, compiler loop, test repair, or sandbox | Not implemented |
| Safety | Rule path stores explicit supported statements; conflict asks for clarification; unsupported code task is labeled unsupported; template responses are evidence-bound | No broad harmful-content, jailbreak, prompt-injection, or high-impact-domain red-team corpus; no generative model safety testing | Limited to deterministic behaviors |
| Security | Local-only bind enforced; loopback Host validation, same-origin write checks, JSON-only POST, 20 KB request cap, HTML escaping, corrupt-database fail-closed behavior | No authentication or encrypted transport; local machine users can access the data; no independent penetration test | Basic localhost hardening; not production security |
| Fairness and language coverage | Personal name capture allows unlisted values and Unicode letters by implementation | No controlled evaluation across languages, dialects, demographic groups, or name distributions; most fact rules are English-only | Not evaluated |
| Privacy | No telemetry/API transmission in this build; clear removes current fact rows | SQLite data is plaintext; normal DELETE is not a secure wipe; audit rows remain; no privacy review or user data lifecycle controls | Partial; local-only is not equivalent to private/secure |
| Reliability | Unit checks cover correction/conflict, persistence, clear, outage, corruption, unsupported evidence, and retrieval under irrelevant facts | No long-running soak, backup/restore, crash-injection campaign, multi-process contention, or migration testing | Partial |
| Performance and scale | 10,000 synthetic facts stored, reopened, and exact-key recalled; measured local write/load/retrieval | Retrieval was ~0.92 s at 10K because it scans/re-tokenizes the corpus; one run is not a capacity guarantee; latency benchmark gate is unstable and missed | Measured, optimization needed |
| Human evaluation and operations | Local trace exposes extraction, evidence, plan, and response; user can inspect and clear current memory | No independent user study, monitoring dashboard, release rollback, incident response, or feedback workflow | Not implemented |
| Competitive capability | Reproducible limited memory benchmark and numeric regression demo | No foundation model or broad learned representation; no ChatGPT comparison or general capability benchmark | Not a general AI assistant |

## Test results

| Test class | Result | What it establishes |
|---|---:|---|
| Unit, integration, and localhost API suite | **44/44 pass** | The explicitly implemented deterministic cases and new one-prompt/security routing behave as tested |
| Functional acceptance | **10/10 pass** | Recall, correction, conflict, abstention, persistence, deletion, name recall, outage, and corrupt database in fresh test stores |
| Generated v2 development benchmark | **80 cases** | Cortex-M and full-transcript rules both scored 100%; gist-only 75%; current-turn 25%; controlled phrase grammar only |
| Generated v2 holdout benchmark | **240 cases** | Cortex-M and full-transcript rules both scored 100%; gist-only 75%; current-turn 25%; 72.9% approximate context-word reduction vs transcript |
| Saved benchmark gates | **4/5 pass** | The holdout p95 latency ratio was 2.055× vs a <=2.0× target. Sub-millisecond timing is noisy; target is not claimed as met |
| SQLite capacity probe | **10,000/10,000 pass** | Directly inserted synthetic records survived reopen and the target was retrieved; write 0.982 s, reopen/load 0.195 s, retrieval 0.919 s, database 5,586,944 bytes, traced Python peak 31,591,485 bytes |
| Code task probe | **Correctly unsupported** | “Write a Python function that computes factorial” returns an explicit unsupported response; no code is generated or run |

The 10K probe inserts structured records directly into SQLite. It does **not**
prove ingestion or reasoning over 10,000 arbitrary pages, long natural-language
history, or attachments. It also exposes a scaling issue: retrieval currently
loads and tokenizes every fact for each query.

## Requirements for a serious general-assistant path

“Training data on Cortex-M” can mean either saving it as retrievable memory or
updating a model's learned parameters. This repository only implements the
first for a narrow fact grammar and the second for numeric OLS regression.
Neither teaches general language or coding ability.

A credible coding-assistant milestone needs a capable pretrained language
model, a single conversational history, repository/file tools, a restricted
execution sandbox, and an evaluation loop that checks patches against hidden
tests and regressions. A general ChatGPT competitor additionally needs broad
knowledge, instruction following, calibration, multilingual and multimodal
capabilities, safety evaluation, and ongoing operations. A memory store and
rule engine can complement such a model; they cannot replace its learned
language and code capabilities.

The current machine has 7.7 GiB RAM, an 11th Gen Intel Core i5-1135G7, Intel
Iris Xe graphics, and an NVIDIA GeForce MX350. No local language model was
benchmarked in this audit. This hardware inventory is not evidence that a
candidate model will fit or perform acceptably; a model choice requires a
separate measured installation and latency/memory experiment.

## Prioritized next phases

1. Freeze this report and benchmark artifacts as the verified baseline.
2. Decide whether the no-model/no-API constraint still holds. If it does, scope
   Cortex-M as a transparent memory-and-tools product and stop describing it as
   a ChatGPT competitor. If general language and coding are required, evaluate
   a model-backed architecture; first benchmark one candidate on this exact
   computer before selecting it.
3. Expand memory evaluation using LongMemEval/LoCoMo-style cases: multiple
   sessions, temporal changes, multi-hop questions, long irrelevant history,
   summaries, multilingual phrasing, and calibrated abstention. Keep a hidden
   holdout and publish exact task-level results.
4. For coding, add an isolated workspace agent only after selecting the model
   path; evaluate on repository tasks with immutable tests, patch review,
   execution limits, and explicit failure analysis.
5. Add privacy lifecycle controls, migration/restore and soak tests, independent
   red teaming, and usability testing before any public release.

## Reproduction commands

```powershell
.\scripts\python.ps1 -m unittest discover -s tests -v
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split dev
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split holdout
.\scripts\python.ps1 scripts\check_success.py
.\scripts\python.ps1 scripts\run_scale_probe.py
.\scripts\python.ps1 scripts\visualizer.py
```

The exact test log, environment inventory, copied benchmark summaries, acceptance
report, scale report, and live UI smoke sequence are in
`results/system-audit-20261023T182500Z/`. See also the timestamped reports under `results/system-audit-*`,
`results/acceptance-*`, `results/rule-*`, and `results/scale-probe-*`.

## Unified research follow-through

The MIMO sparse-routing and Cortex-M gist/memory work are organized as one research program, while retaining separate evidence and advancement gates. See [the unified plan](unified_research_plan.md). A read-only audit of archived M2M-93 raw rows found saved arithmetic internally consistent (1,512/1,512 derived-field checks across 72 rows and all reported means/interval formulas match). The 72 observations are nested across three trained seeds; the pooled interval does not account for possible within-seed dependence. This audit neither reproduces the experiment nor validates its causal interpretation.

Automated reruns on 2026-10-24: 44/44 unit/integration tests passed; functional acceptance 10/10 passed, but aggregate benchmark status remains partial because p95 ratio is 2.055× against a ≤2.0× gate; 10,000 synthetic fact rows persisted/reopened and the key was recovered, with ~0.94 s retrieval. Reports are under `results/acceptance-20261023T183231Z/`, `results/scale-probe-20261023T183231Z/`, and `results/mimo93-audit-20261023T183441Z/`.
