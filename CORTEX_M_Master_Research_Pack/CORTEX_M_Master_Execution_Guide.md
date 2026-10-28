---
title: "CORTEX-M Master Execution Guide"
subtitle: "Core Architecture • Engineering Platform • Scientific Discovery (CORTEX-M-SD)"
author: "CORTEX-M Research Plan"
date: "29 October 2026"
---

# Document control

| Field | Value |
|---|---|
| Version | 1.0 — initial execution plan |
| Scope | CORTEX-M Core, CORTEX-M Engineering, CORTEX-M Scientific Discovery (CORTEX-M-SD), shared data and evaluation infrastructure |
| Compute assumption | CPU-only for the first research cycle; inventory actual RAM, CPU, disk and OS before choosing model sizes |
| Intended use | Research planning and experiment execution; not evidence that any proposed capability has already been achieved |
| Companion files | `CORTEX_M_Data_Registry.csv` and `CORTEX_M_Experiment_Preregistration_Template.json` |

> **Critical realism about compute:** On an ordinary CPU-only workstation, do not plan to pretrain a capable general-purpose LLM from scratch. Begin with small trainable CORTEX-M modules, synthetic tasks, retrieval, symbolic tools, deterministic verification and a frozen or lightweight local language-model backend if feasible. Scale only after module-level benefits are measurable. All performance outcomes described below are expected artifacts or hypotheses, not observed results.

# Executive summary

This guide turns the existing CORTEX-M training-data roadmap and Scientific Discovery proposal into an executable research programme. The objective is to develop and test a modular architecture that uses explicit pre-output gist formation, associative/episodic memory, adaptive routing, continual learning and verification. CORTEX-M-SD extends the system with an abductive hypothesis engine, a counterfactual/world-model interface, active experiment selection and an evidence ledger.

The central research question is:

> **Does a CORTEX-M configuration with inspectable gist, provenance-aware memory, adaptive computation and a hypothesis-testing loop improve held-out generalisation or reduce the experimental budget required to identify hidden rules, compared with strong simpler baselines?**

That question is falsifiable. The system may find that one component helps, another adds no value, and the integrated design performs worse than a simpler model. Those are legitimate outcomes. An architecture is not validated by its name, design elegance, data volume or ability to produce plausible explanations. It is validated by repeatable measurements and comparisons that isolate what caused any improvement.

## Outcomes expected from the first research cycle

1. **Reproducible baseline:** the previous MIMO/M2M experiments are audited, with code/config/data/metric versions reconciled or prior claims explicitly marked unreproduced.
2. **Governed data ecosystem:** every data source has a stable ID, canonical link, version, license-review status, intended role, split policy, quality notes and acquisition instructions.
3. **Modular prototype:** gist, memory, routing, continual-learning, verifier, code-tool and scientific-discovery mechanisms can be tested and disabled independently.
4. **Comparable baselines:** constant/linear, MLP, retrieval-only, static/dense routing, relevant legacy MIMO and symbolic-regression baselines run in the same evaluation harness.
5. **Scientific discovery benchmark:** hidden-rule environments expose observations, allow bounded interventions, keep the true rule hidden from the model and evaluate predictions on withheld conditions.
6. **Evidence-backed integration decision:** only components supported by ablation results are integrated into the default architecture.

## Recommended research tracks

| Track | What it owns | What it shares |
|---|---|---|
| **CORTEX-M Core** | Gist state, associative memory, adaptive routing, general task coordinator and continual learning | Data provenance, run logging, configuration, evaluator and verification APIs |
| **CORTEX-M Engineering** | Repository search, code editing, tool schemas, isolated execution, test feedback and repair | Episodic memory, router, safe action loop, run/trace format |
| **CORTEX-M-SD** | Scientific hypothesis generation, candidate models, world/simulation interface, experiment selection, scientific verification and claim status | Gist, evidence memory, tool execution, experiment registry and provenance |

Do not force all tracks into one monolithic neural network. Share interfaces and data/evaluation discipline first. Share parameters only when there is evidence that joint training is beneficial.

# Part I — Research charter and evidence standards

## 1. Central hypothesis and its alternatives

**Primary hypothesis (H1):** Compared with matched simpler baselines, explicit gist, episodic memory, adaptive routing and a controlled abductive experiment loop can improve generalisation, sample efficiency, or the number of experiments required to identify a hidden rule on at least some controlled tasks.

**Null/competing hypothesis (H0):** Improvements are absent, unstable, explained by extra parameters or compute, caused by data leakage, or recoverable by a simpler method such as retrieval, an MLP, regression or symbolic search. H0 must remain a valid outcome.

The first milestone is not “invent relativity.” It is to demonstrate a narrower claim: CORTEX-M can infer a rule withheld from it, predict unseen cases, choose an informative test, update its explanatory state after a result, and reject an explanation when it fails.

## 2. Claims ladder

| Evidence level | Meaning | Permitted claim |
|---|---|---|
| 0 — end-to-end prototype | Modules execute and log outputs | “The prototype runs.” |
| 1 — beats trivial baseline | Better than a constant/random/simple baseline on a fixed task | “Promising on this task and split.” |
| 2 — ablation-supported | Gain reduces when the proposed active component is removed | “Evidence that this component contributes.” |
| 3 — replicated | Result repeats over multiple seeds and a frozen confirmation split | “Repeatable improvement within this benchmark.” |
| 4 — transfer | Gain holds on an excluded rule family or another task domain | “Evidence of limited generalisation.” |
| 5 — external scientific validation | Predictions survive independent real-world observations/experiments and expert review | “Potential scientific result,” with limitations and domain review |

A generated equation that fits visible observations is not automatically a discovery. A simulator is not independent evidence if its own hidden answer leaks to the model or the model only learns the assumptions already embedded in it.

## 3. Non-negotiable research principles

- **One main question per experiment.** Change one major factor at a time where possible.
- **Keep a strong simple baseline.** The benchmark should not be chosen to favour the architecture.
- **Freeze test sets.** Final test data must not be used for training, prompt design, threshold tuning or repeated informal inspection.
- **Store raw outputs.** Save per-example predictions, retrieved memories, route scores, candidate hypotheses, experiment choices and verifier results—not just an aggregate score.
- **Record lineage.** Every run needs code commit, dataset IDs/hashes, config, seed, package versions, hardware and command.
- **Preserve negative results.** Failed or neutral experiments inform the architecture just as much as successful ones.
- **Keep claims proportional.** A synthetic-world result supports a claim about that synthetic task, not automatically about real science or human-like reasoning.
- **Abstain when evidence is insufficient.** “Unknown” is a valid result when the verifier or data cannot support a confident conclusion.

## 4. Baseline and statistical rules

For early screening, target **at least five fixed seeds**. For promising confirmatory comparisons, target **ten seeds when feasible**. On CPU constraints, use fewer only with the limitation stated and all available seed-level results shown. Choose one primary metric and a minimum effect of interest before confirmatory runs. A provisional 5% relative improvement can be a planning reference, but only use it if justified for the task and committed before results are seen. Report effect size, seed-level distribution, confidence interval where feasible, runtime and memory.

Use the same dataset version, splits, task budget and tuning budget across variants. If parameter count or compute differs, measure and report it. Do not describe a routing system as faster until wall-clock profiling demonstrates a real end-to-end benefit.

# Part II — Architecture and repo foundation

## 5. Target architecture

The proposed loop is:

**Input/observation → evidence retrieval → gist construction/update → candidate generation → routing/allocation → prediction/action → independent verification → feedback and memory update → stop, retry or abstain.**

For CORTEX-M-SD, the central extension is a distinct *hypothesis–prediction–experiment* loop. Memory and gist are supportive but do not replace a proposal mechanism capable of generating competing explanations or a world-model interface capable of producing predictions under counterfactual interventions.

| Module | First implementation | Later extension | Minimum test |
|---|---|---|---|
| Data/provenance | Manifest, hashes, source and split tracking | Knowledge graph and automated source status checks | Exact source/version is recoverable |
| Base model/encoder | Small task model or frozen backend | Adapter/continued training only if compute supports it | Baseline accuracy and resource profile |
| Gist buffer | Typed, inspectable fields | Optional learned latent representation | Coverage, contradiction and revision checks |
| Associative memory | Exact lookup then cosine top-k | Reranker, utility and aging policy | Downstream lift; conflict and stale-memory tests |
| Adaptive router | Rule-based then soft router | Sparse top-k experts | Task score, load balance, route stability, latency |
| Hypothesis engine | Candidate grammar + symbolic regression | Learned proposal generator | Candidate validity, diversity and held-out predictions |
| World model | Independent hidden-rule simulator | Learned surrogate model | Held-out transition prediction and uncertainty |
| Experiment planner | Random/fixed/disagreement heuristics | Learned action policy | Experiments-to-identification and budget curve |
| Verifier | Deterministic schema/unit/numeric checks | Independent learned critique as additional signal | False acceptance/rejection rates |
| Continual learning | Task sequence + replay baseline | Regularization, modular adapters | Retention/forgetting and storage cost |
| Evaluation | Shared scripted runner | Dedicated harness if necessary | Clean rerun reproduces expected values |

## 6. Recommended repository structure

```text
cortex-m/
  README.md  pyproject.toml  .gitignore
  configs/                   # versioned YAML/JSON configs
  data/
    registry/datasets.csv
    raw/                      # local only; ignored by Git
    interim/                  # parsed but not approved
    processed/                # frozen splits/manifests
    synthetic/                # model-visible data; ground truth separate
  src/cortexm/
    encoders/ gist/ memory/ routing/ continual_learning/
    engineering/ scientific_discovery/
    evaluation/ provenance/ verification/
  experiments/                # one immutable folder per experiment ID
  tests/unit/ tests/integration/ tests/regression/
  scripts/                    # validate_registry, inspect_data, run, score
  reports/experiment_cards/ reports/figures/ reports/release_notes/
  artifacts/                  # local checkpoints/logs; ignored by Git
  environment/                # lockfile and system_info.md
```

### Setup sequence

1. Inventory CPU model/core count, RAM, free disk, OS, Python version and any GPU. Store this in `environment/system_info.md`.
2. Use a supported Python version (for a first project, Python 3.11 or 3.12 is a sensible starting point; verify package compatibility before committing).
3. Create and activate a project virtual environment; use `python -m pip` rather than relying on a bare `pip` command on Windows.
4. Install the minimum stack: NumPy, pandas or Polars, scikit-learn, pytest, PyYAML, JSON Schema validation and plotting. Install PyTorch from its official selector: <https://pytorch.org/get-started/locally/>
5. Run a tensor smoke test, a test-discovery smoke test and a one-batch forward/backward test before any full training run.
6. Pin known-working package versions. Record `python --version`, `python -m pip freeze`, OS details and all setup commands.
7. Keep `data/raw/` and `artifacts/` out of Git. Version manifests, code, transformations and compact metadata instead.

Example Windows PowerShell setup, after installing a compatible Python version:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas scikit-learn pytest pyyaml jsonschema matplotlib
# Install PyTorch using the official selector linked above.
python -m pytest -q
```

Do not make TensorFlow, GPU availability, a vector database service or a large orchestration framework prerequisites for the first experiment. Fix the environment before blaming the model. If dependency installation fails, keep the exact error and known-good environment in `reports/environment_setup.md`.

## 7. Experiment identity and artifacts

Each run receives a unique ID such as `CM-001`, `CM-020-R03` or `CM-SCI-01-R02`. A run folder must include:

- frozen config and hash;
- code commit and working-tree status;
- dataset IDs, source versions, split IDs and hashes;
- seed, Python/package lock, OS and hardware details;
- exact command and start/end time;
- model/checkpoint path or hash;
- raw predictions and per-example outcomes;
- route/memory/hypothesis/experiment logs where relevant;
- summary metrics, errors/warnings and an interpretation note.

Never overwrite the only copy of an experiment. A corrected metric, new split or changed model is a new run and must be labelled as such.

# Part III — Data sourcing, governance and preparation

## 8. Dataset acquisition tiers

The data catalogue is a menu, not an instruction to download everything. Pick a source only when it answers a defined experimental question. Prioritise by ground-truth availability, relevance, rights clarity, tractability, storage and expected information gain.

| Tier | Contents | Why this tier exists | Promotion gate |
|---|---|---|---|
| **Tier 0: synthetic** | Hidden-rule equations, toy causal systems, generated trajectories, synthetic episodic-memory records | Small, cheap and known ground truth; ideal for controlled falsification | Generator tests pass; ground truth is isolated from the model; splits are locked |
| **Tier 1: compact benchmarks** | Selected math/code problems, one small UCI dataset, a small retrieval/memory task | Validates loaders and metrics without huge downloads | Official splits and leakage checks verified; runtime recorded |
| **Tier 2: targeted scientific data** | One NIST measurement dataset, selected arXiv/open-access metadata and papers, one small NASA/NOAA/chemistry dataset | Adds real evidence and a bridge to transfer | Source terms, units, methods, citations and data quality are documented |
| **Tier 3: large corpora** | FineWeb/Dolma/DCLM, large code archives and broad scientific archives | Possible later scaling; not a starting requirement | Compute/storage budget approved; rights, privacy and contamination pipeline audited |

## 9. Registry fields required before training

| Field | Required value |
|---|---|
| `dataset_id` | Stable ID, e.g. `D032`; changing the underlying source/version materially creates a new version/record. |
| `name`, `canonical_url` | Provider name and URL, not only a third-party mirror. |
| `version_or_snapshot` | Release tag, revision, dump date, DOI/version or API retrieval date. |
| `intended_role` | Training, retrieval, synthetic environment, development benchmark, sealed test, metadata/reference only. |
| `license_status` | Not reviewed / review required / reviewed for a stated purpose; include reviewer/date and exact terms. |
| `scope_and_size` | Rows/files/tokens, compressed and extracted bytes, schema, and expected processing overhead. Mark unknowns. |
| `domain_and_units` | Subject, language, measurement units, experimental conditions and data dictionary. |
| `provenance` | Original publisher, paper/DOI, repository, instrument, simulator version or upstream source ID. |
| `quality_flags` | Missingness, duplicates, label uncertainty, contradictory records, retraction/correction status. |
| `split_policy` | Random/group/time/subject/trajectory/repository/hidden-rule split policy and frozen split IDs. |
| `hashes` | SHA-256 for source artifacts and processed split manifests. |
| `decision` | Accepted, restricted, retrieval-only, eval-only or rejected, with rationale. |

## 10. Dataset ingestion — exact sequence

1. **Read the canonical source page.** Review its current dataset card, documentation, license, citation requirements, data schema and restrictions. A public download is not the same as permission for every use.
2. **Download metadata first.** Inspect available versions/configurations, file list, record count and approximate size before requesting large data.
3. **Acquire the smallest useful sample.** Parse a sample and inspect records. Check encoding, expected fields, units, types, missing values and malformed data.
4. **Classify intended use.** Mark each resource as candidate training, retrieval/reference, simulator, development benchmark or sealed test. Do not put every resource into the training pool.
5. **Review terms and privacy.** For mixed corpora, review components individually. Screen for personal information, credentials/secrets and controlled-access data. Exclude ambiguous or prohibited sources until cleared.
6. **Record provenance and hashes.** Save URL, version, retrieval timestamp, file sizes, record count and SHA-256; keep raw data immutable.
7. **Profile quality.** Measure duplicates, near-duplicates, missing fields, label distribution, outliers, inconsistent units, contradictory labels and parsing failure rate. Keep a report of what was filtered.
8. **Deduplicate at the right level.** Exact and near-duplicate document/question/code/trajectory detection matters. Group related records together before splits.
9. **Freeze splits before model tuning.** Split math by source/problem family where possible; code by repository/project; science by paper, experiment, site or time; synthetic rules by entire equation family/operator structure.
10. **Transform without destroying lineage.** Preserve raw-to-processed mapping. Store transform script/version and a processed-data manifest.
11. **Validate loaders and scorers.** Write tests for counts, dimensions, units, labels and no-overlap between train/dev/test.
12. **Promote accepted data.** Move only approved outputs into `data/processed/`. Keep rejected/ambiguous sources out of training and explain why.

### Scientific evidence taxonomy

| Evidence type | Meaning | What the system may conclude |
|---|---|---|
| Observed measurement | Instrument/experiment data, with method and conditions | Evidence about measured quantities, including uncertainty and limitations |
| Derived value | Calculated from measurements under explicit assumptions | Valid only if inputs, equations, transformations and units are valid |
| Computational prediction | Model/simulator output | Conditional on the computation method and its assumptions |
| Literature claim | Statement attributed to a publication | A claim to assess and cite, not automatically a settled fact |
| Hypothesis | Proposed model not adequately tested | Candidate for test, not a validated conclusion |
| Correction/retraction | Notice changing a source's status or interpretation | Update the evidence record and preserve historical provenance |

## 11. Leakage and contamination controls

- Keep train/dev/test manifests separate; the training process should not load the sealed-test directory by default.
- Fingerprint math questions, code repositories, papers and trajectories to detect exact and approximate overlap.
- In synthetic scientific discovery, hold out whole equation families or operator structures—not just randomly selected datapoints from equations seen in training.
- For real science, hold out papers/projects/experimental series or chronological segments. Randomly splitting rows from the same experiment can make the task artificially easy.
- If using a pretrained language model, acknowledge that its training corpus may be partly unknown. Private generated hidden-rule tests help provide a cleaner evaluation layer.
- When a benchmark question is repeatedly viewed during prompt design or tuning, reclassify it as development material and stop treating it as a sealed test.
- Never put the hidden rule, hidden parameters or ground-truth labels into the model-visible prompt, logs consumed by the model, retrieval store or training context.

# Part IV — Phase-by-phase execution guide

The following phases are deliberately sequential at the gate level. Some engineering can overlap, but a later phase must not rely on an unresolved result from an earlier phase. The calendar later in this document is indicative only.

## Phase 0 — Inventory, scope and archive

**Goal:** know what exists, what is missing, and what hardware is available before rewriting the architecture.

**Step-by-step**

1. Create the repository and add the proposed folder layout from Part II.
2. Record OS, CPU, RAM, free disk, Python version, network constraints and available accelerator. Use a short `system_info.md`, not assumptions from prior chats.
3. Locate original MIMO scripts, notebooks, dataset generators, configs, checkpoints, training logs and result exports. Archive copies before making edits.
4. Create an artifact inventory: path/name, size, modified date, hash, version, purpose and whether it is believed to relate to M2M-60/M2M-61.
5. Mark all earlier metric claims as either backed by artifacts, partially backed or unverified.
6. Freeze the first research question and write a one-paragraph definition of the smallest result that would falsify the current hypothesis.

**Expected artifacts:** `README.md`, `environment/system_info.md`, `reports/artifact_inventory.csv`, and a one-page research question note.

**Acceptance gate:** another person (or a clean future session) can identify the project source of each prior result, or the missing evidence is explicitly recorded. No new result should depend on an undocumented previous run.

## Phase 1 — Forensic MIMO/M2M audit (CM-001)

**Why this is a blocker:** existing notes have a configuration discrepancy. Prior notes mention a 32-input/32-output setting, whereas an initial static audit of `MIMO_M2M_FINAL.py` indicated an 8-input/4-output configuration, support examples and multiple routers. These could be different experiment versions. Do not merge metrics until the relationship is explained.

**Step-by-step**

1. Make the original source tree read-only and compute SHA-256 hashes.
2. Compare input/output dimensionality, training sample counts, support-set size, model components, losses, optimizer, seed, update count, evaluation data and metric implementation.
3. Identify exactly which file generated each M2M-60/M2M-61 value. Link the run log and source revision.
4. Unit-test metric implementations with hand-calculated toy examples. Verify whether purported pair precision, recall and F1 are genuinely distinct measures rather than copies of exact-match accuracy.
5. Inspect intervention code. Confirm the shuffled-support predictions and routing diagnostics come from the same forward pass. Verify zeroed-relation tests remove only the intended representation rather than wiping all output context unless that is the intentional intervention.
6. Rerun the baseline using the original environment/config if possible. If not, list every missing dependency and distinguish exact replication from a fresh reimplementation.
7. Compare saved predictions/routes/intermediate values when available; quantify differences instead of describing them only qualitatively.
8. Write `reports/CM-001_audit.md`: reproduced / partially reproduced / not reproducible, with evidence and next action.

**Expected result:** a clear source of truth for earlier experiments, or a formal statement that old numbers cannot yet support an architecture claim.

**Gate:** metric tests pass and run lineage is reconciled. If it fails, keep MIMO as an interesting exploratory baseline and proceed with a clean new harness; do not silently reuse questionable metrics.

## Phase 2 — Data registry and rights review

**Goal:** turn all proposed source links into an auditable acquisition plan.

**Step-by-step**

1. Add the companion CSV registry to `data/registry/datasets.csv` and preserve stable IDs.
2. For every P1 item, inspect the canonical page and fill in version, size, license status, intended role, split policy and source citations.
3. Classify each source as train, retrieval/reference, environment, development benchmark or sealed test. Keep training and evaluation roles separate even when a public dataset supports both.
4. Start with Tier 0 synthetic tasks plus a few compact Tier 1 sources. Do not bulk-download FineWeb, Dolma, The Stack v2 or large scientific data to “prepare for later.”
5. Record unresolved terms as “review required.” Do not infer that public availability means unrestricted commercial use or redistribution.
6. Add a registry validator that fails if any accepted record is missing a URL, version, purpose, license status or split/usage policy.
7. Add a storage budget column after measuring a small sample end-to-end. Preserve the measured bytes and extrapolation method.

**Expected artifacts:** approved P1 source list, `datasets.csv`, a license/terms decision log and a data-acquisition order.

**Gate:** every source used in training has a documented source/version and appropriate-use decision. If a source is ambiguous, defer it or keep it out of model training.

## Phase 3 — Environment, tests and logging

**Goal:** guarantee that results can be repeated before complexity increases.

**Step-by-step**

1. Build the virtual environment and install only the packages needed for the first baseline.
2. Add a small config schema and reject invalid/missing settings before a run begins.
3. Create seeded data generation and a test that changing the seed changes the generated sample while keeping the same seed reproduces it.
4. Create unit tests for data shapes, label ranges, metric calculations, intervention semantics and hidden-label isolation.
5. Add one command that runs a small training job and writes metrics, predictions, logs and config snapshot to an experiment folder.
6. Add a clean-checkout smoke test and package lockfile.
7. Run one complete baseline from a fresh checkout to confirm that the workflow is not dependent on an interactive notebook state.

**Expected artifacts:** working test suite, stable config schema, experiment runner and an initial raw result bundle.

**Gate:** a clean run completes and produces the same deterministic results where determinism is expected. If it fails, fix the pipeline rather than extending the architecture.

## Phase 4 — Baseline harness (CM-010)

**Goal:** establish an honest reference for every later module.

**Step-by-step**

1. Choose one small deterministic numeric task and one simple classification or structured-rule task.
2. Create train/dev/test splits before training; save their hashes.
3. Implement constant/heuristic, linear/polynomial and compact MLP baselines.
4. Add retrieval-only as a separate baseline where examples can be reused.
5. Add symbolic regression (for the rule-recovery task) with an explicitly restricted operator set and fixed search budget.
6. Run at least five seeds for screening, saving individual outputs and runtime.
7. Test the scorer on edge cases and manually verify a small handful of predictions.
8. Freeze the baseline report before adding gist/memory/routing changes.

**Expected result:** a reliable baseline score range and known failure modes. The task should be easy enough to validate but nontrivial enough that a constant predictor is not already perfect.

**Gate:** baseline variants use identical splits and the same scoring implementation; all metric tests pass. Do not progress to a claim about an improvement until this is true.

## Phase 5 — Structured gist buffer (CM-020)

**Goal:** test whether a dedicated, updateable pre-output state reduces omissions, contradictions or sample needs.

**Initial schema fields:** `objective`, `core_claim`, `evidence`, `assumptions`, `constraints`, `predictions`, `uncertainty`, `next_action`, `contradictions`, `open_questions`, `provenance_and_time`, and a revision history.

**Step-by-step**

1. Implement the gist as a typed structure (dataclass or validated JSON), not only a latent vector.
2. Construct gist from input plus selected evidence using explicit rules first.
3. Define annotation criteria for key facts and constraints, and manually review a small fixed set of examples.
4. Compare four variants: no gist, structured gist, optional learned latent gist, and structured-plus-latent gist.
5. Add oracle-gist and corrupted/irrelevant-gist controls. The oracle condition is only an upper-bound diagnostic, not a fair production model.
6. Test gist revision after new evidence arrives; preserve prior claim and revision reason instead of overwriting history.
7. Measure fact/constraint coverage, omission, contradiction, unsupported claim rate, gist size and downstream task score.
8. Inspect cases where gist improves scores but hides uncertainty or overcompresses important details.

**Expected result:** a quantified answer to whether structured gist improves performance or interpretability after controlling for additional context/parameters.

**Gate:** keep gist in the default path only if its effect is measurable and the system retains an effective no-gist fallback. If gains are absent, simplify the schema or leave it optional.

## Phase 6 — Episodic and associative memory (CM-030)

**Goal:** determine whether memory improves relevant recall and downstream decisions rather than just retrieving semantically similar text.

**Step-by-step**

1. Start with exact key-value lookup, then cosine nearest-neighbour retrieval, then top-k. Avoid a complex database in the first iteration.
2. Each record should have memory ID, task context, observation, hypothesis, action, expected outcome, actual outcome, source/episode IDs, timestamp, confidence/evidence class and invalidation conditions.
3. Define relevance thresholding; allow “no trustworthy memory found.”
4. Add synthetic tests for relevant, irrelevant, duplicated, corrupted, stale and contradictory memories.
5. Compare no-memory, exact lookup, cosine top-k and provenance-aware retrieval variants under matched settings.
6. Evaluate downstream score improvement, retrieval precision/recall, incorrect-use rate, provenance accuracy, latency and storage.
7. Add eviction policies (FIFO, recency, relevance, utility) only when correctness tests work.
8. Test a memory update where a hypothesis is disproved; ensure the record is marked rejected/overridden, not silently deleted or still treated as certain.

**Expected result:** clear positive/negative utility by memory type and a tested conflict/age policy.

**Gate:** memory must improve a target task or provide a justified auditability benefit without unbounded stale-memory errors. If it harms the task, use thresholds, provenance, expiration or no-memory fallback.

## Phase 7 — Adaptive routing (CM-040)

**Goal:** test whether context-dependent module selection improves quality, robustness or measured resource usage.

**Step-by-step**

1. Implement a deterministic task-to-module router as the baseline.
2. Add a small soft router; log probabilities, selected route, outputs and failures.
3. Add top-k/sparse routing only after the soft route works and profiling shows why sparsity might help.
4. Test task performance against static/dense routes under matched data, parameter budget and inference conditions.
5. Perturb inputs with paraphrases, irrelevant context, low-quality memory and out-of-domain tasks; measure route stability and fallback behavior.
6. Measure expert utilization, route entropy, load balance, latency and memory—not only routing overlap.
7. Add a safe dense/static fallback if router confidence is low.
8. Verify that support examples influence routes in an expected way and that route changes improve downstream predictions, not simply look different.

**Expected result:** a measured quality/compute trade-off, including the possibility that a simpler router is best.

**Gate:** only keep adaptive/sparse routing as default if it gives a predeclared advantage or meets a clearly justified flexibility need. Do not claim faster execution without timing it.

## Phase 8 — Continual learning (CM-050)

**Goal:** evaluate learning from sequential tasks while retaining previous skills.

**Step-by-step**

1. Build a sequence of small task/rule families A → B → C → D with controlled overlaps.
2. Measure each task before training and immediately after its learning phase.
3. Continue sequential training; measure old tasks after each new task rather than only at the end.
4. Compare naive fine-tuning, replay buffer, regularization (e.g. EWC if appropriate), modular task-specific heads, and external-memory-only update.
5. Fix sample budgets and task ordering across variants; report storage/update costs.
6. Evaluate a task sequence without explicit task labels if deployment would not supply them.
7. Hold out one family to measure transfer rather than memorization.
8. Record any trade-off where new-task score rises while old-task performance falls.

**Metrics:** current-task score, earlier-task retention, forgetting, forward/backward transfer, replay storage, update time and inference latency.

**Gate:** select a continual-learning strategy based on a declared retention/compute trade-off. Do not claim “no forgetting”; report actual observed forgetting.

## Phase 9 — Verifier and bounded repair

**Goal:** create independent checks that catch errors and prevent the system from turning plausible output into claimed truth.

**Step-by-step**

1. Implement deterministic checks first: schema/type, tensor dimensions, numeric range, unit/dimensional consistency, algebraic substitution, code tests and simulator prediction error.
2. Make verifier outputs structured: `pass`, `fail` or `unknown`; failed invariant; evidence; scorer version; and next action.
3. Set a small repair budget initially (for example, at most two attempts) and stop/abstain if checks still fail.
4. Tune thresholds on development data only.
5. Inject known faults: wrong units, invalid shapes, inconsistent constraints, corrupted measurements and code regressions.
6. Measure false acceptance, false rejection, detection by fault type, repair attempts and compute cost.
7. Keep evaluator code and hidden ground truth outside the model's writable/readable workspace.

**Gate:** verifier catches the predeclared class of errors at the required rate and does not reject too many valid cases. Otherwise narrow the verifier's responsibilities or require human review.

## Phase 10 — CORTEX-M Engineering platform

**Goal:** build an agent that solves small software tasks from repository context and uses actual test results to guide repairs.

**Recommended trajectory format:** `issue/task → repository snapshot → retrieved files → proposed patch → patch applied in sandbox → test command → stdout/stderr + return code → diagnosis → bounded repair → final regression status`.

**Step-by-step**

1. Create a tiny local test repository with one failing test and one known correct fix. Use this as the first end-to-end task.
2. Implement repository search and file retrieval with path allowlists, size limits and binary-file filters.
3. Define a strict patch schema: proposed files, unified diff, rationale, expected test and resource budget. Reject edits outside the task workspace.
4. Execute code only in a sandbox/container with time, memory, filesystem and network restrictions. If secure isolation is not available, keep initial testing static/synthetic and do not execute untrusted output on the host.
5. Capture command, return code, duration, test names, failures, traceback, environment, repository commit and patch hash.
6. Build a small dataset of successful and unsuccessful trajectories; label based on observed test outcomes, not the model's assertion that its patch is correct.
7. Add timeouts, retry caps, schema validation, restricted file access and human confirmation before destructive actions.
8. Run local held-out tasks before attempting SWE-bench, browser automation or broad repository repair.

**Initial data order:** custom synthetic tasks → small MBPP/CodeSearchNet subset → a narrow SWE-bench development subset if isolation works → broader code corpora only after source-level license/contamination reviews. Treat HumanEval and other public tests as evaluation-first, not data to repeatedly tune on. For code corpora, split at repository/project level where possible.

**Gate:** safe execution is enforced, every action/result is logged, acceptance is based on independent tests, and the system stops honestly after repeated failures.

# Part V — CORTEX-M Scientific Discovery (CORTEX-M-SD)

## 12. Scientific discovery loop

CORTEX-M-SD must separate observation, hypothesis, derivation and validation. The hypothesis engine proposes candidate explanations; the world model predicts their consequences; the experiment planner selects a discriminating intervention; the verifier scores outcomes; and an evidence ledger records what was actually supported.

| Stage | Input | Operation | Output / log |
|---|---|---|---|
| 1. Observe | Measurements, prior records and provenance | Validate quality, units, noise and conditions | Normalized observations + uncertainty |
| 2. Represent | Task, evidence and constraints | Build/update the structured gist and open questions | Current explanatory state |
| 3. Propose | Evidence and permitted priors | Generate several candidate explanations | Hypothesis set, assumptions and complexity |
| 4. Predict | Candidate + potential interventions | Compute outcomes or distributions | Predictions with uncertainty |
| 5. Select | Candidate disagreement, cost and safety | Choose the most discriminating affordable experiment | Experiment choice and expected information gain |
| 6. Execute | Experiment configuration | Query independent simulator or approved experiment | Raw result, seed and conditions |
| 7. Score | Prediction and new observation | Compute error and formal checks | Updated candidate scores/rejections |
| 8. Revise | Scores and contradictions | Update beliefs, gist and episodic memory | Versioned hypothesis history |
| 9. Stop/abstain | Budget, uncertainty and predeclared stop rule | Stop if evidence is adequate, budget exhausted or ambiguity remains | Claim status and next recommended test |

## 13. Hypothesis engine: build a constrained proposal system first

Do not begin with an unconstrained language model inventing equations in prose. First implement an explicit candidate grammar whose members can be evaluated by code. A language model may later propose new candidate structures, but each proposal must be parsed, checked, executable or formally represented, and independently scored.

### Candidate-generation ladder

1. **Equation/rule library:** linear, polynomial, exponential, sinusoidal, piecewise, simple conservation/invariance forms and relational rules.
2. **Symbolic regression baseline:** use a tool such as PySR. Record allowed operators, search budget, complexity penalty and seed.
3. **Model comparison:** compare training fit, held-out prediction, complexity and residual structure with a predeclared selection criterion.
4. **Language-assisted proposal:** request candidate equations or experiments in a strict JSON schema; validate syntax and permitted operators before execution.
5. **Candidate diversity:** retain multiple plausible models when evidence does not distinguish them; do not collapse to the best training fit immediately.
6. **Novelty accounting:** record whether the proposed candidate is new relative to the allowed library, an algebraic re-expression, or already known in the model's allowed prior knowledge.

Each hypothesis record should include: ID; equation or executable representation; variables and units; applicability scope; assumptions; source/generating process; fitted parameters; training fit; held-out predictions; complexity; uncertainty/calibration; contradictions; proposed discriminating experiment; status; and full revision history.

## 14. World model: protect ground truth

Start with an external ground-truth simulator that is known to the experiment designer but hidden from the model. The model may query it through a constrained observation/action API. It must not read the hidden equation, hidden parameters, evaluator source or labels. Later, CORTEX-M may learn a surrogate world model from observed trajectories, but the surrogate must be evaluated separately against ground-truth transitions and uncertainty.

### World progression

- **World 1:** deterministic 1-D dynamics with several candidate hidden rule families.
- **World 2:** controlled noise, missing observations and measurement uncertainty.
- **World 3:** multi-variable relationships where observations alone may not identify causal structure; interventions are available.
- **World 4:** a held-out operator family or different representation (transfer test).
- **World 5:** one carefully selected real dataset with provenance, units, uncertainty and a scientifically defined question.

Each simulator should implement an interface such as `reset(seed)`, `observe(conditions)`, `get_budget()` and `record_result()`. Store model-visible observations and evaluator-only ground truth separately. Add tests that try to leak labels through record IDs, debug strings, file paths, logs or serialization order.

## 15. Active experiment selection

An experiment is useful when it distinguishes candidate explanations. Repeating an observation on which all candidates agree has low value. Start with a disagreement heuristic and compare it with random and fixed-sequence policies; only later consider learning an experiment policy.

1. Enumerate a finite set of allowable interventions (for example, change initial velocity or measurement times).
2. Calculate each candidate's predicted output under every intervention.
3. Score each experiment using a measure such as prediction variance/pairwise distance divided by estimated cost.
4. Choose a permitted, safe experiment with the highest score; record candidate predictions *before* execution.
5. Execute the experiment, record the raw result and update scores.
6. Repeat to a predeclared budget/stopping rule.
7. Compare random, fixed, disagreement-based and oracle policies under the same initial observations and budget. Oracle selection is an upper-bound diagnostic only.

**Metrics:** identification success at fixed budget, held-out prediction error, experiments-to-identification, simulator calls, wall-clock cost, robustness under noise and uncertainty calibration.

## 16. Scientific verifier and claim ledger

The first verifier should be deterministic wherever possible:

- numeric predictions on observations withheld from fitting;
- dimensional/unit checks where units are known;
- boundary/limit checks under declared physical constraints;
- counterfactual checks after changing initial conditions;
- complexity assessment, without treating simplicity as proof of truth;
- active search for a condition where the candidate fails;
- independence checks to ensure validation observations were not used to fit/tune;
- a claim-status check that distinguishes observed, hypothesised, derived and validated.

Verifier output should be structured: `pass/fail/unknown`, failed invariant, evidence used, scorer version and next action. Use a bounded repair budget (two attempts is a reasonable initial test setting, not a universal rule) and abstain if checks still fail. The hypothesis-generating process must not be able to edit the hidden evaluator or modify the observation record after seeing the answer.

## 17. Synthetic discovery episode schema

A discovery record should capture the entire decision process, not just input/output:

```json
{
  "episode_id": "WORLD1-SEED11-EP0001",
  "world_id": "hidden_dynamics_v1",
  "world_version": "1.0.0",
  "seed": 11,
  "visible_context": {"variables": ["t", "x", "v0"], "units": {"t": "s", "x": "m"}},
  "observations": [],
  "candidate_hypotheses": [],
  "selected_experiment": {},
  "predictions_before_experiment": [],
  "observed_result": {},
  "prediction_errors": [],
  "verifier_results": [],
  "hypothesis_status_updates": [],
  "budget_before_after": {},
  "provenance": {"generator_commit": "", "config_hash": ""}
}
```

Store `hidden_rule_id`, true parameters and the equation in a separate evaluator-only record, never in the model-visible JSON. Include tests that validate the separation. The data generator—not a language model writing a convincing story—must generate observations and ground-truth labels.

## 18. CM-SCI experiment series (with expected outcomes and gates)

### CM-SCI-01 — Hidden-rule recovery

**Question:** Can the system infer a rule from sparse observations and predict unseen cases?

**Procedure:**
1. Generate multiple rule families with varied parameters and initial conditions; store a private ground-truth map.
2. Expose only limited observations per episode. Reserve new initial conditions and entire equation families for testing.
3. Run constant/linear/polynomial, MLP and symbolic-regression baselines.
4. Run CORTEX-M without memory, then with memory/gist as separate variants.
5. Score candidate rules on held-out trajectories; check algebraic equivalence and unit consistency.
6. Inspect overfitting, leakage and failures by rule family.

**Expected artifacts:** world generator, unit/leak tests, frozen split manifest, config snapshots, per-seed predictions and comparison report.

**Primary metric:** held-out prediction error or rule-family identification, selected before final test. Secondary metrics: samples required, candidate complexity, runtime and robustness under noise.

**Gate:** proceed only if the generator hides ground truth correctly and the method beats the relevant simple baseline reproducibly. A curve fitting the observed data alone is not sufficient evidence of discovery.

**Failure handling:** simplify the grammar/noise, audit leakage, add diagnostic cases and check that symbolic-regression search budget/operator set is fair. Keep the negative result if it persists.

### CM-SCI-02 — Counterfactual prediction

**Question:** Does the candidate predict outcomes under interventions rather than merely interpolate observed conditions?

**Procedure:** fit candidates on designated conditions; vary initial conditions or other controllable variables; compare predictions to hidden simulator outputs; benchmark against interpolation and MLP; include corrupted-assumption and mismatched-simulator controls.

**Expected artifacts:** intervention generator and score report by intervention type/extrapolation distance.

**Primary metric:** held-out counterfactual prediction error. Secondary: calibration, error versus distance from training support and false confidence.

**Gate:** a transfer claim requires improvement beyond simple baselines on the predeclared interventions without label/rule leakage. If it only interpolates, narrow the claim accordingly.

### CM-SCI-03 — Competing explanations

**Question:** Can CORTEX-M keep multiple explanations alive and choose an observation that distinguishes them?

**Procedure:** construct cases where multiple hypotheses fit initial evidence; require structured candidate objects; enumerate permitted interventions; compare disagreement-based selection with random/fixed selection; run the selected intervention; update candidate rankings and preserve rejection reasons.

**Expected artifacts:** hypothesis schema, per-episode prediction/action/result trace and ranking changes.

**Primary metric:** fraction of cases where the ground-truth model is identified within a fixed experiment budget. Secondary: candidate diversity, experiments-to-identification and calibration.

**Gate:** experiment selection must reduce the budget versus random/fixed on held-out cases. If all candidates make indistinguishable predictions, the task is not identifiable; redesign the environment rather than blaming the learner.

### CM-SCI-04 — Active experiment selection

**Question:** Does information-guided experiment choice improve discovery efficiency?

**Procedure:** implement random, fixed, disagreement-based and optional oracle policies; keep initial conditions and budgets matched; vary noise, candidate count and experiment cost; compare error/success at equal budgets and total cost to target error.

**Expected artifacts:** one policy API, action logs and budget-performance curves.

**Primary metric:** discovery success or prediction error at a fixed budget. Secondary: query cost, wasted experiments and noise robustness.

**Gate:** retain active selection only if it provides a meaningful predeclared benefit over random and fixed policies across seeds. Otherwise keep it as a heuristic or optional module.

### CM-SCI-05 — Compact unifying principle

**Question:** Can CORTEX-M find one compact model that generalises across related phenomena and also reject forced unification when evidence does not support it?

**Procedure:** generate worlds where multiple effects share a latent rule and negative-control worlds where they do not; fit unified and independent models; assess unseen combinations and a predeclared complexity measure; inspect false-unification claims.

**Expected artifacts:** multi-phenomenon generator, complexity metric and report of successes and appropriate rejections.

**Primary metric:** generalisation to unseen combinations with complexity controlled. Secondary: description length/number of parameters and false-unification rate.

**Gate:** a method must perform well in unifiable worlds and not invent a unified law in negative controls. Shorter expressions alone are not proof of truth.

### CM-SCI-06 — Novel-rule transfer

**Question:** Can CORTEX-M discover a rule family excluded from training, development, examples and prompt design?

**Procedure:** predeclare held-out rule/operator families; change representation or variable encoding; check prompts and corpus for leakage; freeze baselines; score on the sealed transfer set over multiple seeds.

**Expected artifacts:** sealed split generator, hashes, configs, per-seed results and independent prediction checks.

**Primary metric:** held-out-family identification/prediction score. Secondary: sample efficiency, calibration and in-family/held-out gap.

**Gate:** only claim transfer if it exceeds appropriate baselines and survives sealed evaluation. If not, describe the capability as recovery within known families.

## 19. Bridge to real scientific data

Do not jump directly from a toy world to claims about new physics. Use this ladder: **synthetic hidden laws → rediscovery of known laws from restricted observations → one real dataset with a defined prediction target → a different domain → independent review and real-world validation where possible**.

1. Choose a tractable dataset with good documentation: a reference measurement series, basic physical measurements, a bounded astronomy task or a compact materials/chemistry dataset.
2. Write a protocol before modeling: question, candidate explanations, known confounders, uncertainty, allowed transformations and independent holdout.
3. Prevent literature retrieval from leaking the expected rule into the supposed discovery task.
4. Compare with domain-relevant statistical and scientific baselines.
5. Ask a knowledgeable reviewer whether the proposed question and split are scientifically meaningful.
6. Label generated models as candidates requiring expert review. Do not call them new discoveries without independent evidence.

# Part VI — Training and evaluation protocol

## 20. Train modules first; do not start with giant end-to-end training

The recommended first implementation is hybrid. Train compact task-specific modules for architecture claims and use deterministic tools wherever they provide a stronger check. If a suitable pretrained language model is available within the hardware budget, keep it frozen at first as a proposal/interpretation backend. This isolates whether CORTEX-M mechanisms help without requiring foundation-model pretraining.

Do not conflate these activities:

- **Pretraining from random initialization:** learning broad language or representation capabilities from a very large corpus; out of scope for an ordinary CPU-first project.
- **Continued pretraining:** adapting a model to a domain with additional raw text; still compute intensive and potentially licensing-sensitive.
- **Supervised fine-tuning:** learning outputs from labelled input/target examples; requires trustworthy labels and separation from evaluation data.
- **Preference/reward optimization:** changes behavior based on preferences or a reward signal; must be separately designed and evaluated.
- **Training small modules:** learning gist encoders, routers, regressors, retrieval weights or hypothesis-ranking components; the practical starting point.
- **Retrieval/indexing:** organizing information for lookup; does not necessarily change model parameters.
- **Interactive experience generation:** the system chooses actions in an environment and receives outcomes; particularly important for active scientific discovery.

### Baseline ladder

| Baseline | Implementation | Why it is required |
|---|---|---|
| B0 — trivial | Constant predictor, majority class or simple heuristic | Detects broken data and establishes whether a task is nontrivial |
| B1 — linear/symbolic | Linear regression, polynomial regression, simple equation search | Strong low-complexity floor; essential for rule recovery |
| B2 — compact MLP | Small feed-forward network, tuned only on dev | Measures standard neural learning without CORTEX-M mechanisms |
| B3 — retrieval-only | Exact/nearest-neighbour/top-k examples; no learned router/gist | Separates retrieval benefits from other modules |
| B4 — static/dense routing | Fixed mapping or dense expert mixture | Tests whether dynamic routing is necessary |
| B5 — audited MIMO baseline | Recovered, version-matched original implementation | Connects to previous research after lineage and metric audit |
| B6 — component ablations | Full model minus gist, memory, routing, verifier or hypothesis loop | Attributes effects to modules |
| B7 — scientific discovery baseline | Symbolic regression, candidate grammar and/or simple model comparison | Compares against methods designed for equation/rule discovery |

## 21. Shared training loop — exact procedure

1. Load a versioned configuration and validate required fields before touching the data.
2. Load only approved dataset IDs and verify hashes/split manifests.
3. Set all random seeds and record hardware/software details.
4. Instantiate the model and optimizer from configuration; record parameter counts and trainable/frozen status.
5. Run one small-batch smoke test: forward pass, loss, backward pass if training, and one evaluation pass.
6. Run a short pilot to estimate time/RAM and check that loss/metrics behave plausibly.
7. Train baseline and treatment variants with the same declared budget.
8. Save checkpoints and per-epoch data; do not save only a final aggregate.
9. Use the development split only for hyperparameter selection, early stopping and thresholds.
10. Freeze configuration and launch planned confirmatory runs on the sealed test.
11. Generate the result report from raw outputs using the versioned scoring code.
12. Archive the run folder and make a go/no-go decision against the preregistered gate.

## 22. CPU-first operating guidance

| Stage | Use CPU for | Avoid at first |
|---|---|---|
| Prototype | Synthetic worlds, small MLPs, gist/router modules, retrieval, schema/unit tests | Full web-scale pretraining, giant batches, simulations without budgets |
| Early experiments | Compact datasets, sequential seeds, CPU profiling, data-quality checks | Many expensive variants in parallel without memory/runtime monitoring |
| Language component | Frozen small model if it loads in RAM; tool/rule fallback if not | Assuming any multi-billion-parameter model can be efficiently trained on an arbitrary CPU workstation |
| Scale decision | Measure CPU time, RAM, disk and identify the actual bottleneck | Adding more compute or complex infrastructure before knowing what requires it |

If language-model adaptation is eventually needed, compare a frozen model plus retrieval/tools against fine-tuning and adapter options. Feasibility depends on the exact base model, parameter count, quantization, RAM, software support and task. Do not promise fine-tuning throughput before measuring it. The first successful project milestone can be architecture evaluation using small trainable components even when the language model remains frozen.

## 23. Core metrics

| Subsystem | Primary measures | Diagnostic measures |
|---|---|---|
| General reasoning | Held-out accuracy/loss | Calibration, errors by type, sample efficiency, inference cost |
| Gist | Key-fact/constraint coverage, contradiction rate | Omission, unsupported claims, size, revision latency |
| Memory | Downstream performance lift with memory on vs off | Precision@k, recall@k, stale/conflicting retrieval, latency, storage |
| Routing | Task quality at matched resource budget | Route overlap, entropy, load balance, fallback rate, wall time |
| Continual learning | Score on new and previous tasks after sequential learning | Forgetting, retention, forward/backward transfer, replay storage |
| Engineering | Tests passed / task resolved | Regression count, sandbox timeouts, retries and execution cost |
| Scientific discovery | Held-out prediction; rule identification; success at fixed experiment budget | Experiments-to-identification, complexity, uncertainty calibration and false-unification rate |
| Verifier | False-accept and false-reject rate | Detection by fault type, repair count, abstention and cost |
| System | Quality at fixed resource budget | RAM, disk, CPU/wall time; energy only if measured/reliably estimated |

## 24. Minimum ablation matrix

Every confirmatory integrated result should include these variants (some can be postponed during a tiny pilot):

| Variant | Gist | Memory | Routing | Hypothesis/experiment loop | Interpretation |
|---|---|---|---|---|---|
| V0 | Off | Off | Static | Off | Simple baseline |
| V1 | On | Off | Static | Off | Gist contribution |
| V2 | Off | On | Static | Off | Memory contribution |
| V3 | On | On | Static | Off | Combined gist + memory |
| V4 | On | On | Adaptive | Off | Routing contribution |
| V5 | On | On | Adaptive | On | Full scientific loop |
| V6 | Corrupted | On | Adaptive | On | Reliance on gist quality |
| V7 | On | Irrelevant/conflicting | Adaptive | On | Robustness to bad retrieval |
| V8 | Oracle diagnostic | Oracle where applicable | Oracle where applicable | Oracle experiment choice | Upper-bound diagnostic only, not a fair deployment comparison |

Use matched parameter count and compute controls where possible. When not possible, report the gap and conduct a control that offers a similar amount of extra capacity/compute without the proposed mechanism.

## 25. Gist training metrics and decision logic

The gist buffer should store: task objective, core claim, evidence and source IDs, assumptions, constraints, predictions, uncertainty, next action, contradictions, open questions and versioned history. Start with a structured object and explicit rules; add a learned latent vector only after structured gist is working.

Compare no-gist, structured-gist, latent-gist and structured-plus-latent variants. Add oracle, corrupted and irrelevant gist controls. Score key-fact/constraint coverage, omission rate, contradiction rate, unsupported-claim rate, gist size, revision correctness and downstream task metric. Oracle gist is a diagnostic upper bound. Keep gist in the default path only if its effect is measurable or if it adds a separately justified auditability benefit without unacceptable performance cost.

## 26. Memory metrics and decision logic

Memory records should include observation, task context, hypothesis, action, prediction, observed outcome, provenance, timestamp, reliability/evidence class and invalidation conditions. Compare no memory, exact lookup, cosine top-k and provenance-aware retrieval under the same task budget. Measure both retrieval quality and downstream performance.

Test: relevant memories; irrelevant memories; duplicate records; corrupted data; stale knowledge; contradictory records; previously accepted hypotheses later disproved. Do not overwrite contradictions silently. Mark invalidated memories and preserve provenance. Keep memory only if it improves the task or offers a measured auditability benefit and has a working no-memory fallback.

## 27. Routing metrics and decision logic

Start with deterministic routing, then soft routing; add sparse top-k only if there is a reason. Measure task quality, route stability under paraphrases/irrelevant context, expert utilization/load balance, fallback frequency, wall-clock latency and memory use. A changing route is not automatically a useful route, and sparse routing can be slower at small scale. Keep a dense/static fallback when confidence is low.

## 28. Continual-learning metrics and decision logic

Use a fixed sequence of rule/task families, such as A → B → C → D. Score each task before learning, immediately after learning it and after subsequent tasks. Compare naive fine-tuning, replay, regularization such as EWC when appropriate, modular/task-specific parameters and external-memory-only updates. Track new-task performance together with old-task retention, forgetting, transfer, storage and update time. Do not claim “no catastrophic forgetting”; report the observed trade-off.

## 29. Engineering platform — tool/action/result learning

The engineering branch should learn from repository-level tasks and actual test outcomes rather than source snippets alone.

1. Start with a tiny local repository with one failing test and one known fix.
2. Implement repository search and file retrieval with path allowlists, size limits and binary-file exclusion.
3. Define a strict patch schema: files, diff, rationale, tests and resource budget. Reject modifications outside the task workspace.
4. Execute generated code only in a sandbox/container with time, memory, filesystem and network limits. If secure isolation is unavailable, use static/synthetic tasks only.
5. Capture command, return code, duration, test names, stdout/stderr, traceback, environment, repository commit and patch hash.
6. Build a trajectory corpus from both successful and failed attempts, labelled using actual observed outcomes.
7. Add timeouts, a retry cap, strict tool schemas and human confirmation for destructive actions.
8. Evaluate on held-out tasks/repositories before expanding to SWE-bench or browser automation.

**Initial code data order:** custom synthetic tasks; a small MBPP or CodeSearchNet subset; a narrow SWE-bench development subset after sandboxing is stable; broad The Stack v2/StarCoder2 data only after source-level licensing and contamination review. Treat HumanEval and similar public sets as evaluation-first. Repository-level splits are preferred to random snippet splits.

**Engineering gate:** test success is independently measured; action traces are complete; generated code is sandboxed; timeout/failure behavior is safe; the agent does not claim a task passed unless logs verify it.

# Part VII — Indicative execution calendar

This is a planning baseline for one primary researcher, CPU-first work and limited parallelization. It is not a guarantee. If a phase gate fails, stop and investigate rather than advancing just to maintain the calendar.

| Weeks | Primary work | Concrete deliverable | Gate / exit evidence |
|---|---|---|---|
| 1 | Hardware inventory, repository, environment and source collection | `system_info.md`, clean repo, artifact inventory | Smoke tests pass; old artifacts located or evidence gap documented |
| 2 | MIMO forensic audit and data registry | CM-001 audit, initial P1 source decisions | Earlier metrics reconciled or excluded from claims |
| 3–4 | Baseline task, loaders, scorer tests, logging and seeds | CM-010 baseline harness | Same command reproduces metrics; splits frozen |
| 4–6 | Structured gist buffer and controls | CM-020 gist report | Coverage/contradiction metrics work; gist is independently switchable |
| 5–7 | Episodic memory and retrieval controls | CM-030 memory comparison | Memory utility established or retained as optional |
| 6–8 | Routing variants and profiling | CM-040 routing report | Quality/resource gain measured or simpler route retained |
| 8–10 | Sequential-task/continual learning | CM-050 retention report | New-task gains compared with forgetting and storage cost |
| 9–12 | Safe engineering tool interface | Engineering v0 with local test-fixing tasks | Sandbox and action/result logs pass safety tests |
| 10–12 | Synthetic hidden-rule world and CM-SCI-01 | World v1 and rule-recovery report | Ground truth isolated; baseline and held-out evaluation work |
| 12–15 | Counterfactuals, competing hypotheses, active experiments | CM-SCI-02/03/04 reports | Compare with random and fixed policies under matched budgets |
| 15–17 | Unified-principle and negative-control tasks | CM-SCI-05 report | Can reject unsupported unification in negative controls |
| 17–19 | Sealed novel-rule transfer | CM-SCI-06 report | At least one held-out-family transfer test completed |
| 16–20 | One targeted real scientific dataset (can overlap) | Data protocol with provenance/units/baseline | Rights, data splits, units and baseline reviewed |
| 20–22 | Integrate selected modules and run ablations | Integrated-model report | Each included component has component-level evidence |
| 23 | Confirmatory runs/reproducibility audit | Clean rerun package, metrics and limitations | Rerun matches within expected variation |
| 24 | Research decision/release preparation | Technical report and next-phase proposal | Claims match evidence level; licensing/safety checks pass |

## 30. Weekly cadence

- **Start of week:** declare one primary question, baseline, expected artifacts and acceptance gate.
- **During implementation:** test the smallest component first; keep commands and errors in the run record.
- **Midweek:** manually inspect a small sample; verify scorers with independent calculations/unit tests.
- **End of week:** run matched comparisons; record actual runtime/RAM; update the experiment register; write result and decision.
- **Every two weeks:** review data provenance, contamination controls, checkpoint storage and technical debt.

# Part VIII — Full risk register and failure playbook

| Risk | Early warning | Mitigation / fallback | Decision |
|---|---|---|---|
| Gist provides no gain | No improvement over no-gist; omissions or contradiction rise | Compare structured/oracle/corrupted/no gist; simplify slots or bypass | Optional component or remove |
| Memory is stale/irrelevant | Wrong retrievals raise errors | Provenance, freshness threshold, conflict handling and no-memory fallback | Restrict scope or improve retrieval |
| Router collapse | One expert dominates; routes change under harmless paraphrases | Soft routing, load balancing or static/dense fallback | Keep best-performing simple route |
| Continual forgetting | Old task scores fall after new task | Replay, regularization, modular parameters, external memory | Choose with retention trade-off visible |
| Hypothesis engine imitates known forms | Strong in-family result, weak unseen-rule transfer | Hold out rule families/operator structures; compare symbolic regression | Limit claim to known-family recovery |
| Simulator leakage | Near-perfect early performance or hidden labels in logs | Separate evaluator process; audit serializer/file names; leak tests | Invalidate and rerun |
| World model validates itself | Good prediction only in one simulator | Alternative simulator/real data; held-out measured data; uncertainty | Narrow claim and add independent evidence |
| License ambiguity | Missing/mixed rights or disallowed intended use | Exclude, retrieval-only or obtain review | No ingestion until cleared |
| Benchmark contamination | Suspiciously high score or duplicates | Group split, dedupe, private synthetic test | Rebuild split; old score exploratory only |
| CPU bottleneck | Swapping, slow jobs, RAM pressure | Smaller batches/tasks, sequential seeds, streaming, frozen model | Scale only with a measured bottleneck |
| Metric bug | Unrelated metrics identical or implausible | Hand-calculated test cases and independent scorer | Invalidate affected report |
| Verifier overconfidence | Invalid models pass; good models fail frequently | Deterministic checks, dev calibration, abstention | Reduce verifier scope |
| Scope creep | Many modules before baselines stabilize | Phase gates; one main question per run | Defer additional modules |
| Inflated scientific claim | Wording exceeds held-out evidence | Claims ladder and external review | Block publication/release until corrected |

## 31. Troubleshooting checklist

| Symptom | Inspect first | Next action |
|---|---|---|
| Package installation fails | Python version/architecture, venv, proxy/network, exact error | Recreate clean env, isolate package, pin known-good requirements |
| Metrics vary unexpectedly | Seeds, data order, split hash, nondeterministic operations and config | Record variation; test deterministic paths; report spread |
| Shuffled-support effect appears identical | Whether examples truly changed; route and prediction diagnostics from same forward pass | Assert changed input hash and compare route/prediction tensors |
| Precision/recall/F1 equal accuracy | Metric implementation and class semantics | Hand-test confusion-matrix cases and recompute independently |
| Ablation has no effect | Whether component truly disconnected; caches or bypass paths | Add intervention tests on the relevant tensor/API path |
| Training score high but test poor | Leakage, overfit, split logic, preprocessing mismatch, novel rule family | Inspect residuals and split; reduce complexity; record exact failure |
| Sparse route slower than dense | Kernel/data movement overhead and small batch size | Profile end-to-end; keep sparse routing only if measured benefit exists |
| Scientific result has no independent test | All data from one simulator or one paper/series | Add held-out condition/domain or expert review; reduce evidence level |
| Result cannot be reproduced | Missing config/code hash/source version/checkpoint | Mark incomplete; repair logging before next expensive run |

# Part IX — Acceptance gates and release procedure

## 32. Project gates

| Gate | Minimum pass condition | If it fails |
|---|---|---|
| G0 — environment | Clean setup; imports/tests run; system facts recorded | Pause and fix environment |
| G1 — prior audit | MIMO lineage reconciled or old metrics explicitly excluded | Do not rely on old figures |
| G2 — data governance | Source/version/rights status/hashes/splits and leak checks recorded | Keep unreviewed data out of training |
| G3 — baseline harness | Scorers tested and runs reproducible | Fix harness before architecture research |
| G4 — module correctness | Gist, memory, routing and verifier unit/integration tests pass | Keep module off or use simple fallback |
| G5 — science loop | Hidden rule, prediction, experiment, outcome and update logged end-to-end | Return to a smaller hidden-rule world |
| G6 — ablation evidence | Full model and controls tested under matched settings | No component-attribution claim |
| G7 — transfer evidence | Sealed family/domain test with appropriate baselines | Limit scope to in-family performance |
| G8 — reproducibility | Clean rerun and artifact audit pass | Do not claim reproducibility |
| G9 — release | Claims, source terms, limitations and execution safety reviewed | Restrict release or remove material |

## 33. Final deliverables

- Architecture spec and stable module interfaces.
- Unit, integration and regression tests.
- Dataset registry with IDs, links, versions, usage decisions and split manifests.
- Immutable experiment configs, code IDs, package lock and seed records.
- Baseline/ablation result tables with all seed-level outcomes and interval/effect-size reporting where possible.
- Synthetic-world generator plus separate evaluator-only ground truth, model-visible episodes and leak tests.
- Engineering sandbox, tool interface, test logs and safe-failure behavior.
- Scientific claim ledger with provenance and statuses.
- Experiment pre-registration form and report template.
- Technical report including failed results, compute budget, limits and next hypotheses.

## 34. Release-readiness checklist

- [ ] Dataset redistribution rights reviewed; when uncertain, release scripts/manifests rather than the data.
- [ ] No private data, credentials, secrets or restricted material in the release.
- [ ] Hidden test answers and evaluator-only ground truth are not in model-visible training bundles.
- [ ] Every conclusion is labelled as established result, preliminary evidence, hypothesis or future goal.
- [ ] Benchmark versions, prompts, splits, seeds, scorer code and test-time compute are recorded.
- [ ] Failures, fallback behavior, supported environments and known limitations are documented.
- [ ] Clean checkout and core tests pass.
- [ ] Primary result can be reproduced from documented instructions.

# Part X — Data priorities, source index and training order

## 35. First 10 sources/tools to prioritise

| Priority | Source/tool | Initial use | Why now |
|---|---|---|---|
| 1 | Custom synthetic hidden-rule generator | Train/evaluate rule recovery | Controlled ground truth and simple leak tests |
| 2 | FineMath | Small math-training sample | Explanatory examples; validate quality before use |
| 3 | GSM8K | Loader/scorer and math baseline | Compact benchmark with documented splits |
| 4 | UCI dataset | One small numeric task | Rapid end-to-end metrics and plotting check |
| 5 | PySR | Symbolic-regression baseline | Direct baseline for equation recovery |
| 6 | ScienceWorld | Interactive scientific tasks | Tests actions and outcomes beyond static text |
| 7 | NIST Data | One reference series | Provenance, units and measurement handling |
| 8 | OpenAlex | Metadata and research graph | Paper/source relationships without bulk full text |
| 9 | Synthetic memory episodes; LoCoMo for evaluation | Memory retrieval and updates | Custom train records while preserving a public benchmark as evaluation-first |
| 10 | MBPP or CodeSearchNet subset | Engineering smoke test | Small programming domain before repository repair |

## 36. Keep data roles separate

- **Training data** updates model parameters or trains a component.
- **Retrieval corpus** is looked up at inference time; it does not necessarily update parameters.
- **Interactive environment** generates experience through actions and observations.
- **Development benchmark** guides algorithm and threshold selection and can become overfit.
- **Sealed test** is for planned final evaluation only.
- **Reference/evidence source** helps validate claims/units but is not automatically appropriate for training.
- **Metadata/knowledge graph** supports discovery/provenance but is not ground truth by itself.

## 37. Storage and compute budgeting

1. Measure compressed size, extracted size, parsed size, cache and checkpoint footprint on a small sample.
2. Estimate RAM from decoded batches, not compressed size alone.
3. Use streaming or provider subsets for large corpora; do not load full archives into memory.
4. Keep raw files and artifacts out of Git; version only manifest, script and non-sensitive summary.
5. If constrained, reduce dataset/model/task scope first. Do not optimize the wrong bottleneck.
6. Never begin full-corpus download without confirming disk space and intended use.

## 38. Dataset priority key

- **P1:** most useful for the first reproducible prototype; inspect and integrate only after source/split/rights checks.
- **P2:** useful once the basic harness works; add to a specific experiment with a written reason.
- **P3:** large, expensive, optional, advanced or later transfer source; defer unless a defined experiment needs it.

The companion `CORTEX_M_Data_Registry.csv` is the operational register. It contains canonical links and a suggested use/priority for each row. Re-check live terms, dataset versions and sizes during acquisition.

# Part XI — Recommended experiment IDs

| ID | Experiment | Main question |
|---|---|---|
| CM-001 | Prior MIMO/M2M audit | Which old metrics can be reproduced and trusted? |
| CM-010 | Baseline harness | Are loaders, splits, scorers and runs deterministic/reproducible? |
| CM-020 | Gist ablation | Does structured gist improve task outcomes/coverage after controlling for capacity? |
| CM-030 | Memory ablation | Does provenance-aware episodic retrieval improve downstream decisions? |
| CM-040 | Routing ablation | Does adaptive routing improve quality or resource use versus static/dense routing? |
| CM-050 | Continual learning | What retention/forgetting trade-off do replay, regularization and modules offer? |
| CM-060 | Verifier fault injection | Which deterministic checks detect errors and at what false-accept/reject rate? |
| CM-070 | Engineering tool loop | Can the system fix local known bugs using actual test results safely? |
| CM-SCI-01 | Hidden-rule recovery | Can it recover withheld rules from sparse observations? |
| CM-SCI-02 | Counterfactual prediction | Does the model predict unseen interventions? |
| CM-SCI-03 | Competing explanations | Can it distinguish candidate models with targeted observations? |
| CM-SCI-04 | Active experiment selection | Does it reduce query/experiment budget versus random/fixed policies? |
| CM-SCI-05 | Compact unification | Can it find and validate shared principles without false unification? |
| CM-SCI-06 | Novel-rule transfer | Can it transfer to excluded rule families/representations? |
| CM-INT-01 | Integrated ablation | Do individually useful modules remain useful together? |
| CM-REL-01 | Reproducibility/release | Can the primary result be rerun from a clean environment? |

# Appendix A — Experiment pre-registration/report template

The companion JSON file is a machine-readable starting template. For each experiment, fill in:

1. Experiment ID/title, owner and date.
2. Research question and falsifiable hypothesis.
3. Null/competing hypothesis.
4. Primary metric, direction, minimum effect of interest and stopping rule.
5. Secondary metrics, named baselines and ablations.
6. Dataset IDs, versions/hashes, split version and contamination controls.
7. Seeds, training configuration, evaluation configuration and hardware budget.
8. Expected artifacts, acceptance gate and failure interpretation.
9. Limitations and actual results after the run. Never enter expected outcomes into the results field as if observed.

# Appendix B — Interpreting outcomes correctly

- If the full model wins but removing the proposed component changes nothing, do not attribute the win to that component.
- If only one seed wins, classify the result as unstable until investigated.
- If the target equation, hidden label or expected answer leaks into input/training data, the discovery experiment is invalid.
- If active selection saves experiments but worsens the endpoint beyond its predeclared tolerance, report the trade-off rather than a win.
- If a verifier catches injected errors but rejects valid results often, report both false-accept and false-reject behavior.
- If discovery works only for rule families included in the grammar, state that capability boundary explicitly.
- If a public benchmark is repeatedly tuned on, reclassify it as development and evaluate on a new sealed set.
- If large-model pretraining is infeasible, the research is not a failure: first isolate the architecture mechanisms that can be tested under available resources.

# Appendix C — Open research questions

1. Does structured gist reduce important omissions or contradictions after controlling for added context length?
2. Does provenance-aware memory outperform a matched retrieval baseline when stale/conflicting cases are included?
3. Does adaptive routing improve the best task metric at a fixed compute budget compared with static/dense routing?
4. Can CORTEX-M recover withheld rule families from sparse observations and predict interventions not seen during fitting?
5. Does active experiment selection reach a target error with fewer simulator calls than random/fixed policies?
6. Can it infer compact common principles and also reject false unification in negative controls?
7. Does continual learning retain old tasks at acceptable compute and storage cost?
8. Do gains replicate across seeds and transfer to a sealed task family?

> **Closing operating principle:** Build the smallest experiment that could disprove the next architectural claim. Keep the baseline alive, preserve the evidence, and let measured results—not the architecture narrative—decide what becomes part of CORTEX-M.
