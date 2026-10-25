# CORTEX-M Original Industrial Ecosystem — Implementation Master Plan

**Purpose:** turn the original CORTEX-M ecosystem plan into a staged engineering programme with a concrete monorepo layout, shared-core boundaries, per-platform workstreams, testing gates, and evidence requirements.

**Planning baseline:** the original programme describes eight product lines: CORTEX-IDE, CORTEX-Personal, CORTEX-Fin, CORTEX-OS, CORTEX-Robotics, CORTEX-Mesh, CORTEX-Health, and CORTEX-Space. The original architecture also names CORTEX-Synapse, CORTEX-Router, CORTEX-Gist, CORTEX-Trace, CORTEX-Shield, CORTEX-Blackboard, CORTEX-Swarm, CORTEX-Bridge, and CortexLM. These names are preserved here. The uploaded plan's performance numbers are treated as targets or reported results to reproduce—not as independently verified properties.

## 1. Executive decision

Build **one modular CORTEX-M foundation and eight separately governed product lines**. Do not build eight independent AI stacks, and do not assume that every product should put the same neural substrate directly in its critical control loop.

Recommended order:

1. Audit and reproduce the existing CORTEX-M repository and benchmarks.
2. Stabilize shared contracts, runtime, trace/evaluation, configuration, storage, and security.
3. Deliver CORTEX-IDE and CORTEX-Personal as the first practical products.
4. Validate CORTEX-Synapse, Router, Gist, Shield, and Trace against ordinary baselines as separate research workstreams.
5. Build CORTEX-Fin as offline fraud/risk analytics and replay tooling before any live transaction or trading integration.
6. Build CORTEX-OS as a research/user-space environment on a mature kernel or formally verified kernel base before attempting to replace the OS kernel.
7. Build CORTEX-Robotics and CORTEX-Mesh in simulation first, with deterministic safety/control systems separated from learned inference. Focus Mesh on civilian coordination, logistics, environmental sensing, and search-and-rescue simulations; do not implement combat, targeting, or detection-evasion behavior.
8. Treat CORTEX-Health/BCI and CORTEX-Space as qualification-driven programmes with domain experts, hardware partners, formal verification where applicable, and independent validation.

The first release should be useful without requiring the full long-term research programme to succeed. The research programme should be able to continue without destabilizing the released product.

## 2. Portfolio and intended scope

### 2.1 Shared foundation: CORTEX-M

Responsibilities:
- Shared contracts and event formats across Python, Rust, C/C++, and client applications.
- Model gateway and inference adapters.
- CORTEX-Synapse associative memory and experimental learning/unlearning.
- CORTEX-Router, CORTEX-Gist, and CORTEX-Bridge.
- CORTEX-Trace and experiment/evaluation harnesses.
- CORTEX-Shield as evidence-aware uncertainty and abstention—not a claim of zero hallucination.
- Tool registry, policy checks, task orchestration, bounded resource budgets, and audit events.
- Common release/versioning and hardware profiling methods.

Non-goal: forcing every platform to use a Hopfield/attractor computation for every operation. Filesystem transactions, access control, financial ledgers, safety interlocks, and medical device state need exact and deterministic semantics.

### 2.2 CORTEX-IDE / CortexCode

Initial product candidate. Build a local-first software engineering workspace with repository understanding, code search, AST-aware edits, patch generation, tests, linting, Git diff/status, bounded repair loops, and independently checked completion reports.

Do not give the agent unrestricted access to the host. Use workspace scoping, isolated command execution, resource/time limits, controlled network egress, and approval for destructive or externally visible actions.

### 2.3 CORTEX-Personal

A personal assistant for desktop/mobile with conversations, offline model support, approved personal memory, document lookup, and selected local tools. Separate offline capability from cloud capability in the UI and configuration. “100% offline” can only be claimed for a profile whose enabled model and tools do not contact a network.

Start with a desktop/local prototype. Then measure and optimize for a clearly named phone/device profile. A compact routing kernel or tiny model is not automatically a general-purpose language model.

### 2.4 CORTEX-Fin

Start with historical transaction replay, anomaly/fraud scoring research, explainable risk features, evaluation reports, and a shadow-mode service that does not authorize or decline real transactions. Financial records require durable, auditable storage; zero-database memory is not a replacement for a ledger or source-of-record system.

Any later integration with payment authorization, order placement, or high-frequency trading requires domain-specific risk controls, compliance review, operational limits, independent validation, and a kill switch. Do not let a neural memory or language model directly make irreversible financial decisions without deterministic controls and a responsible system owner.

### 2.5 CORTEX-OS

Begin as a research environment: a userspace cognitive service, command/API layer, and experiments around associative search and task scheduling. Keep a conventional filesystem, permissions, process isolation, and recovery mechanisms. Do not claim that semantic attractor search replaces a filesystem or that energy thresholds stop exploits without evidence.

Only investigate a kernel or microkernel integration after the userspace prototype has clear requirements. A base such as seL4 can be studied for its documented verification properties and supported configurations, but verified properties apply to specified configurations and do not transfer automatically to new CORTEX code.

### 2.6 CORTEX-Robotics

Begin in simulation using recorded/synthetic sensor streams, state estimation, command proposals, and safe supervisory behaviors. Any hardware control must retain an independent deterministic controller and hardware safety interlocks. The CORTEX model must not be the sole emergency-stop, collision-avoidance, or stability mechanism.

For industrial robot contexts, assess applicable safety requirements including ISO 10218-1:2025 and ISO 10218-2:2025; other robot categories require their own applicable standards and hazard analyses. Validate on target hardware using worst-case timing, not average latency alone.

### 2.7 CORTEX-Mesh

Use this as a civilian multi-node coordination platform for warehouse robots, environmental sensing, disaster-response simulation, or cooperative mapping with lawful, transparent communications. Build a simulator, node health and identity, message validation, topology changes, fault handling, degraded-mode behavior, and auditable coordination first.

The original document describes radio-silent tactical swarm behavior and avoiding detection. This implementation plan does **not** include combat, target engagement, evasion of detection, or military swarm tactics. If developing civilian field systems, keep communications and emergency behavior transparent, compliant, and independently safety-reviewed.

### 2.8 CORTEX-Health / BCI

Begin with offline datasets, synthetic data, signal-processing experiments, and non-clinical demonstrations. Clearly label outputs as research results, not diagnosis or treatment. Do not connect an experimental model to an implant, stimulation path, or prosthetic actuator that can harm someone without a qualified clinical/device team, formal hazard controls, and required regulatory approvals.

A medical-device software lifecycle and clinical programme must be established before any intended clinical product. FDA material includes specific guidance for implanted BCI non-clinical testing and clinical considerations; IEC 62304 covers medical-device software lifecycle processes. Determine the applicable jurisdiction and current requirements with specialist counsel and clinical/device partners.

### 2.9 CORTEX-Space

Start with ground-based telemetry parsing, sensor-data classification, anomaly detection, and high-fidelity simulation. Build fault injection, recovery, deterministic interfaces, timing analysis, version control, requirements-to-test traceability, and end-to-end mission scenarios before any flight use.

NASA software engineering guidance treats flight-software qualification as a planned lifecycle involving configuration management, verification and validation, reviews, regression strategy, and acceptance evidence. Use such practices as a reference model, while tailoring the actual process to the mission/customer and applicable standards.

## 3. Target architecture

```text
                         PRODUCT INTERFACES
          Studio | CLI | IDE | Personal | SDK | API | Edge UI
                               |
                    CAPABILITY / WORKFLOW LAYER
     Chat | Coding | Research | Finance analytics | Robotics sim
      Mesh simulation | OS tools | Health research | Space telemetry
                               |
                       CORTEX ORCHESTRATOR
       task state | planner | routing | tool loop | budgets | approvals
                               |
             +-----------------+------------------+
             |                                    |
        MODEL PLANE                           TOOL PLANE
 CortexLM / local LLM / APIs            files / Git / browser / Python
 embeddings / rerankers / vision         simulator / APIs / data pipelines
             |                                    |
             +-----------------+------------------+
                               |
                  CONTEXT, MEMORY & KNOWLEDGE
  working state | user memory | Synapse | document index | provenance
                               |
                   TRUST / SAFETY / POLICY PLANE
 permissions | sandbox | deterministic interlocks | approval | audit
                               |
                  PERSISTENCE & JOB INFRASTRUCTURE
 metadata DB | files/artifacts | event records | job queue | checkpoints
                               |
                    OBSERVABILITY AND EVALUATION
 unit/integration/system tests | traces | benchmarks | HIL/simulation
                               |
                         TARGET ADAPTERS
 Python | Rust | C/C++ | MCU/RTOS | mobile runtime | target hardware
```

Treat this as logical architecture, not a mandate to deploy each box as a microservice. Start as a modular monolith with process isolation for untrusted execution and separate workers for heavy training or hardware-in-the-loop jobs.

## 4. Engineering principles

1. **One source of truth:** source files, financial records, document revisions, device settings, and approvals must have authoritative records. Synaptic memory is an additional representation, not the canonical ledger for exact facts.
2. **Research code is not flight/clinical/financial production code:** isolate experiments from release artifacts.
3. **Hard real-time control is deterministic:** a language model or associative retrieval engine may propose/score/supervise; it must not replace independent safety mechanisms by default.
4. **Every performance claim names its workload:** hardware, input size, precision, batch size, warm/cold state, statistical distribution, p50/p95/p99/worst case, and measurement boundary.
5. **Every task has limits:** step count, execution time, memory/compute budget, network policy, cancellation and recovery semantics.
6. **Tool output is data, not policy:** web pages, source comments, documents, model output and other agent outputs cannot grant permissions.
7. **Use baselines:** compare each CORTEX mechanism with a simpler, conventional method on the same held-out workload.
8. **Modular monolith first:** separate services when scale, security boundary, or hardware requires it.
9. **Hardware deployment is earned:** simulation, unit tests, target-hardware tests, environmental testing and domain review precede field use.
10. **No absolute “zero leakage,” “zero hallucination,” “exploit immunity,” or medical/flight safety claims without appropriately scoped evidence.**

## 5. Recommended repository structure

Use a monorepo initially. Keep shared Python code separate from low-level Rust/C/C++ platform code and communicate across languages through versioned schemas and narrow bindings. Do not make firmware depend on Python at runtime.

```text
cortex-m/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
├── uv.lock                         # or one chosen lockfile; avoid competing lockfiles
├── Cargo.toml                      # Rust workspace, if Rust components are adopted
├── CMakeLists.txt                  # native components, if/when needed
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml
│
├── apps/
│   ├── studio/
│   │   ├── frontend/               # UI source and tests
│   │   ├── backend/                # thin API adapter only
│   │   └── README.md
│   ├── cli/
│   │   ├── commands/
│   │   └── README.md
│   ├── api/
│   │   ├── routes/
│   │   ├── middleware/
│   │   └── README.md
│   └── desktop/                    # later; do not scaffold until needed
│
├── packages/
│   ├── cortex-contracts/           # typed shared contracts and errors
│   ├── cortex-runtime/             # conversation/runtime orchestration
│   ├── cortex-models/              # gateway and model adapters
│   ├── cortex-memory/              # memory interfaces and lifecycle
│   ├── cortex-knowledge/            # ingestion, retrieval and citations
│   ├── cortex-tools/               # tool registry and tool interfaces
│   ├── cortex-policy/              # authZ, action policy, approval gates
│   ├── cortex-observability/       # events, traces, metrics, audit events
│   └── cortex-eval/                # evaluation harness and reports
│
├── research/
│   ├── synapse/
│   │   ├── src/
│   │   ├── experiments/
│   │   ├── configs/
│   │   └── README.md
│   ├── router/
│   ├── gist/
│   ├── shield/
│   ├── trace-attribution/
│   ├── papers/
│   └── notebooks/                  # exploratory only; production code lives in packages/
│
├── models/
│   ├── cortex_lm/
│   │   ├── architecture/
│   │   ├── tokenizer/
│   │   ├── training/
│   │   ├── finetuning/
│   │   ├── inference/
│   │   └── evaluation/
│   ├── adapters/                   # provider-specific adapters
│   ├── serving/
│   └── model_cards/
│
├── agents/
│   ├── coordinator/
│   ├── planner/
│   ├── critic/
│   ├── verifier/
│   └── workflows/
│
├── tools/
│   ├── filesystem/
│   ├── shell/
│   ├── python_sandbox/
│   ├── git/
│   ├── browser/
│   ├── search/
│   ├── data_analysis/
│   └── integrations/
│
├── platforms/
│   ├── ide/
│   │   ├── app/
│   │   ├── repository_index/
│   │   ├── patching/
│   │   ├── test_runner/
│   │   ├── git_integration/
│   │   ├── sandbox/
│   │   ├── evals/
│   │   └── docs/
│   ├── personal/
│   │   ├── desktop/
│   │   ├── mobile/
│   │   ├── local_storage/
│   │   ├── offline_profiles/
│   │   ├── sync/                    # optional, only with explicit privacy design
│   │   └── evals/
│   ├── fin/
│   │   ├── ingestion/
│   │   ├── feature_pipeline/
│   │   ├── fraud_models/
│   │   ├── replay/
│   │   ├── shadow_mode/
│   │   ├── risk_controls/
│   │   └── evals/
│   ├── os/
│   │   ├── userland/
│   │   ├── service_manager/
│   │   ├── filesystem_adapter/
│   │   ├── scheduler_experiments/
│   │   ├── kernel_experiments/      # isolated from ordinary app runtime
│   │   ├── targets/
│   │   └── evals/
│   ├── robotics/
│   │   ├── simulation/
│   │   ├── sensor_adapters/
│   │   ├── state_estimation/
│   │   ├── supervisory_policy/
│   │   ├── deterministic_control_interfaces/
│   │   ├── safety_monitor/
│   │   ├── hardware_in_loop/
│   │   └── evals/
│   ├── mesh/
│   │   ├── simulation/
│   │   ├── node_identity/
│   │   ├── message_schema/
│   │   ├── coordination/
│   │   ├── fault_handling/
│   │   ├── observability/
│   │   └── evals/
│   ├── health/
│   │   ├── datasets/                # manifests only; sensitive data stays out of Git
│   │   ├── signal_processing/
│   │   ├── offline_experiments/
│   │   ├── clinical_interfaces/     # disabled until formal programme exists
│   │   ├── risk_management/
│   │   └── evals/
│   └── space/
│       ├── telemetry/
│       ├── simulation/
│       ├── autonomy_experiments/
│       ├── fault_injection/
│       ├── target_builds/
│       ├── requirements_traceability/
│       └── qualification_evidence/
│
├── embedded/
│   ├── common/                      # language-neutral protocol definitions
│   ├── rust/
│   ├── cpp/
│   ├── firmware/
│   ├── toolchains/
│   └── target_profiles/
│
├── schemas/
│   ├── events/
│   ├── tool_calls/
│   ├── device_telemetry/
│   ├── model_manifests/
│   └── versioning.md
│
├── storage/
│   ├── migrations/
│   ├── repository_interfaces/
│   ├── sqlite/
│   ├── postgres/
│   ├── object_store/
│   └── retention/
│
├── datasets/
│   ├── manifests/
│   ├── schemas/
│   └── README.md                    # raw datasets and secrets must not be committed
│
├── artifacts/                       # local output; ignored by Git
│   ├── models/
│   ├── checkpoints/
│   ├── eval_reports/
│   ├── profiles/
│   └── builds/
│
├── configs/
│   ├── local.example.toml
│   ├── dev.toml
│   ├── test.toml
│   ├── profiles/
│   └── policies/
│
├── infra/
│   ├── docker/
│   ├── devcontainer/
│   ├── ci/
│   ├── packaging/
│   ├── deployment/
│   └── provisioning/
│
├── docs/
│   ├── architecture/
│   ├── decisions/                   # ADRs: architecture decision records
│   ├── requirements/
│   ├── threat_models/
│   ├── hazard_analyses/
│   ├── verification_plans/
│   ├── operations/
│   ├── user_guides/
│   └── release_notes/
│
└── tests/
    ├── unit/
    ├── integration/
    ├── end_to_end/
    ├── security/
    ├── performance/
    ├── property_based/
    ├── simulation/
    ├── hardware_in_loop/
    └── fixtures/
```

### Folder rules

- `packages/` contains reusable production code. `platforms/` contains product-specific workflows and adapters. `research/` holds experimental implementations and papers. Do not import research notebooks into production runtime.
- `schemas/` owns language-neutral payload definitions. Rust/C++/firmware components consume generated or hand-maintained bindings with compatibility tests; they must not import Python modules.
- `platforms/<name>/evals/` holds platform-specific metrics and datasets/manifests. Common harness code belongs in `packages/cortex-eval/`.
- `artifacts/`, raw data, customer data, secrets, generated binaries and large checkpoints are excluded from Git. Use a controlled artifact/model store and dataset manifests.
- Each platform should include `README.md`, `requirements/`, `design.md`, `risk_register.md`, `tests/`, `evals/`, and a release checklist before hardware or customer use.
- Keep a single canonical `pyproject.toml` and one lock strategy for Python dependencies. Add Rust/C++ build files only for real native code, not as empty placeholders.
- Avoid creating a microservice or repository for every folder. Start with a monorepo and split only when ownership, release cadence, deployment isolation or hardware toolchains demand it.

## 6. Shared package responsibilities

### `cortex-contracts`

Typed request/response models, errors, capability declarations, event schema versions, task IDs, artifact references and result statuses. No model or platform-specific business logic.

### `cortex-runtime`

Conversation lifecycle, context construction, model call loop, task orchestration, cancellation, bounded retry policy, streaming event production and completion checks.

### `cortex-models`

Provider adapters, local inference adapters, model capability discovery, model selection and normalized usage/error metadata. It must not contain authorization policy or platform-specific workflow logic.

### `cortex-memory`

Memory interfaces, identity/provenance, retrieval result schemas, scoped storage, update/delete semantics, snapshots and adapters to CORTEX-Synapse. It must distinguish persistent authoritative records from approximate associative memory.

### `cortex-knowledge`

Document ingestion, parser interfaces, indexing, search, source references, version handling, filtering and deletion propagation.

### `cortex-tools`

Tool schemas, tool registration, execution context, result normalization and sandbox interfaces. High-risk tools call `cortex-policy` before execution and independently enforce limits in their executor.

### `cortex-policy`

Identity, permissions, per-tool policy, approval requirements, risk tiers, audit policy and tenant scope. Do not delegate access decisions to the language model.

### `cortex-observability`

Structured logs, traces, metrics and provenance records. Redact sensitive fields and define retention/access rules.

### `cortex-eval`

Versioned evaluation definitions, baselines, result schemas, score calculators, statistical summaries and report generation.

## 7. Per-platform implementation checklists and exit gates

### 7.1 CORTEX-IDE — first product

1. Workspace discovery and Git status.
2. Text, symbol and dependency search.
3. Context assembly within a token budget.
4. Plan and acceptance criteria for each task.
5. Patch generation with diff preview.
6. Isolated test/lint/build execution.
7. Bounded repair loop; stop after budget or repeated failures.
8. Independent verification against actual test output.
9. User approval for file changes, commit, push, install, deploy or deletion as appropriate.
10. Task/resume history and reproducible task benchmark.

**Exit gate:** successful patch plus evidence, no unexplained file modifications, task-level comparison to a single-model baseline, sandbox and permission tests passed.

### 7.2 CORTEX-Personal — second product

1. Local chat and history.
2. Explicit backend status: local/offline versus remote/cloud.
3. Opt-in scoped memory and delete/inspect controls.
4. File and document import.
5. Local search and supported local tools.
6. Portable settings/export and safe update process.
7. Desktop build first; mobile profile only after size, latency, memory and battery measurements.

**Exit gate:** offline profile proven with network disabled; claims state exactly which functions work offline; local data deletion and backup/restore tested.

### 7.3 CORTEX-Fin — offline analytics first

1. Use historical/synthetic data only initially.
2. Define the ground-truth labels and cost of false positives/negatives.
3. Build feature extraction and baseline scoring.
4. Use time-based training/test splits to avoid leakage.
5. Report precision/recall, calibration, false decline rate, drift and subgroup performance where lawful and appropriate.
6. Replay historical streams and load-test with clearly defined timing boundaries.
7. Add shadow mode; compare outputs to incumbent rules without changing live decisions.
8. Require independent risk/compliance review and explicit rollback/kill switch before any production decision integration.

**Exit gate:** repeatable historical replay, calibrated risk reporting, robust drift handling, audited controls. No claim of 85 microsecond total decision latency until measured end-to-end on named hardware under load.

### 7.4 CORTEX-OS — userspace before kernel

1. Build a CLI/userspace service with normal filesystem semantics.
2. Test associative search as a supplemental retrieval/index layer.
3. Benchmark search latency and correctness against conventional indexes.
4. Add process/service supervision and explicit permission boundaries.
5. Prototype task scheduling separately in simulation.
6. Create kernel-interface requirements and threat model.
7. Evaluate an established kernel/microkernel base before designing a custom kernel.
8. Only proceed to privileged code with a dedicated review process, hardware/virtual test matrix, fault injection and recovery plan.

**Exit gate:** stable userspace prototype and comparative evidence. Do not claim a semantic VFS replacement or exploit immunity from a Lyapunov score alone.

### 7.5 CORTEX-Robotics — simulation and supervisory control

1. Select a non-hazardous robot/simulator and clear success criteria.
2. Define sensor units, clocks, frames, stale-data behavior and message schemas.
3. Replay logged sensor sequences and test perception/state-estimation outputs.
4. Keep learned inference outside the deterministic safety loop initially.
5. Add a supervisory proposal interface to a conventional controller.
6. Simulate slip, sensor loss, delayed messages, actuator saturation and power loss.
7. Perform software-in-the-loop and then hardware-in-the-loop testing.
8. Measure worst-case execution time and jitter on target hardware.
9. Conduct hazard analysis and independent safety review before controlled physical trials.

**Exit gate:** safe simulated behavior over defined scenarios; target timing and hardware tests; a deterministic safe-state and independent emergency stop. A claimed 5 kHz loop corresponds to a 200 microsecond period; if 203 microseconds is the full end-to-end control latency, it already exceeds that period, so define exactly what each number measures.

### 7.6 CORTEX-Mesh — civilian multi-node coordination

1. Build a simulator with nodes, message loss, latency, clock skew and node failures.
2. Define authenticated node identity, versioned message schemas and replay protection.
3. Implement benign cooperative tasks such as inventory, environmental measurement, search-and-rescue simulation or shared mapping.
4. Define graceful degraded behavior when communications fail.
5. Test partial node loss and recovery using benign scenarios.
6. Keep safety boundaries local and independently enforced at each node.
7. Test network partition, stale state, malicious/invalid messages and recovery.
8. Validate the exact radio/network hardware and legal operating conditions before field trials.

**Exit gate:** reproducible civilian simulations, reliable identity/message validation, safe degraded mode and auditable decisions. This plan excludes combat, targeting, or detection-evasion behavior.

### 7.7 CORTEX-Health / BCI — research-only first

1. Define one research question and lawful data sources.
2. Build offline parsers and signal-processing baselines.
3. Keep patient data protected and out of Git; use approved access and retention controls.
4. Evaluate decoding quality on held-out participants/sessions where applicable, not only randomly split samples.
5. Test signal drift, electrode/channel loss, artifacts and confidence calibration.
6. Use `INSUFFICIENT_EVIDENCE` as an explicit result state, not a promise that errors cannot occur.
7. Establish hazard analysis, traceability, risk controls, device quality system and independent validation with clinical/device experts before any clinical use.
8. Do not connect the research system to implants, stimulation, or actuators until the appropriate formal programme is authorized.

**Exit gate:** validated non-clinical results plus a reviewed regulatory, clinical and device-safety plan before any clinical product claim.

### 7.8 CORTEX-Space — simulation and telemetry first

1. Define a notional mission and operational constraints.
2. Build telemetry parsers and deterministic sensor-data pipelines.
3. Create high-fidelity simulation for nominal and off-nominal scenarios.
4. Inject corrupted telemetry, sensor faults, timing issues, resets and resource exhaustion.
5. Define requirements with bidirectional traceability to tests.
6. Control source versions, compiler/toolchain and build artifacts.
7. Test on target-like hardware or an appropriately validated simulator.
8. Conduct independent verification, regression, fault-recovery and acceptance review.
9. Add radiation/environmental effects and target hardware only with qualified hardware partners.

**Exit gate:** requirements traceability, validated simulations, fault-injection evidence, reproducible builds, independent review and customer/mission acceptance. Do not claim compatibility with RAD750 or radiation tolerance based only on a small RAM footprint.

## 8. Wave rollout and gating

The original timeline proposes:
- Wave 1: CORTEX-Personal and CORTEX-IDE.
- Wave 2: CORTEX-Fin and CORTEX-OS.
- Wave 3: CORTEX-Robotics and CORTEX-Mesh.
- Wave 4: CORTEX-Health and CORTEX-Space.

Retain this as a portfolio ordering, not a fixed promise. The common foundation and verification work must precede each platform. More compute, hardware, external domain expertise, regulatory obligations and independent validation make the later waves much less predictable than an ordinary software release.

### Wave 0 — Foundation and evidence
- Inventory implementation, reproduce tests, define contracts, build trace/eval harness.
- Exit: known-good baseline and release criteria.

### Wave 1 — Developer/product validation
- CORTEX-IDE, CORTEX-Personal, core local deployment.
- Exit: useful end-to-end workflows, security baseline, user feedback and measured value.

### Wave 2 — Reliable systems and analytics
- CORTEX-Fin offline replay/shadow mode; CORTEX-OS userspace experiments; improved memory and persistence.
- Exit: durable state, auditable results, reproducible performance and no direct autonomous financial authority.

### Wave 3 — Simulated physical systems
- CORTEX-Robotics and civilian CORTEX-Mesh simulation.
- Exit: scenario coverage, fault injection, safe degraded operation and domain review before hardware pilots.

### Wave 4 — Qualification-driven research
- CORTEX-Health/BCI and CORTEX-Space simulation/telemetry.
- Exit: specialist-led lifecycle, risk management, traceable verification and qualified hardware/environment testing before any operational use.

## 9. CI/CD and development quality

For every pull request:
- Formatting and lint checks.
- Type checking for shared Python interfaces.
- Unit tests and schema compatibility checks.
- Secret/dependency scanning.
- Security tests for policy and tool boundaries.
- Integration tests using local/mock model providers.
- Small deterministic benchmark smoke tests.
- Build/package verification for affected platform.

Nightly or scheduled:
- Larger regression benchmarks.
- Retrieval and memory-capacity experiments.
- Concurrency and resource profiling.
- End-to-end coding/research tasks.
- Simulation suites and fault injection.
- Hardware-in-the-loop only on controlled runners.

Release candidates:
- Re-run pinned dependencies and versioned datasets.
- Generate model/software bills of materials where appropriate.
- Sign artifacts and verify checksums.
- Test upgrade, rollback, backup and restore.
- Publish known limitations and evidence report.

Do not let nondeterministic model tests block all code changes without diagnosis. Separate deterministic regression tests from sampled quality evaluations and track the latter statistically.

## 10. Data, artefacts and Git discipline

- Commit source code, schemas, small fixtures, test definitions, dataset manifests and documentation.
- Do not commit API keys, patient-level data, private customer documents, production logs, large model weights, raw datasets without rights, build directories or checkpoint files.
- Store model weights/checkpoints in an artifact store with versioned manifests and controlled access.
- Store research results with code commit hash, dependency lockfile, config, seed, dataset version, hardware and timestamp.
- Give every generated artifact a provenance record.
- Use migrations for persistent metadata changes.
- Define deletion behavior for original records and all derived indexes/caches.
- Keep an architecture decision record (ADR) when a major dependency, data representation, or boundary changes.

## 11. Performance-claim audit

The uploaded original plan lists values such as 203 microseconds, 20.8 KB, below 15 MB RAM, 85 microseconds, 5 kHz, zero cross-tenant leakage, and 500 nodes. Do not put them into product guarantees until each has a reproducible test and precise scope.

For each claim record:
- Exact code path measured.
- Hardware, OS/RTOS, compiler and model version.
- Input/output shapes and precision.
- Warm-up and cold-start behavior.
- p50/p95/p99 plus worst-case timing where relevant.
- Memory baseline, peak memory and artifact size.
- Concurrent load and failure behavior.
- Baseline comparison.
- Independent reproduction status.

Specific checks:
- A 5 kHz loop has a 200 microsecond period. A 203 microsecond *whole-loop* latency cannot meet every 200 microsecond cycle; verify whether the reported figures refer to different work segments.
- A 20.8 KB binary might describe a small kernel/model component, not a full general-purpose language model. Define the artifact's contents and capability.
- “0.00% leakage” should be expressed as “no leakage observed in a specified test suite of N attempts,” with threat model and coverage, not as a universal guarantee.
- Sparse weights do not automatically mean proportional runtime RAM savings; measure actual data structure and allocator behavior.
- Low average latency is not a hard real-time guarantee. Physical systems need worst-case timing analysis and target-hardware tests.

## 12. Staffing and external expertise

Minimum practical responsibilities, even if one person initially covers multiple roles:
- Core/runtime engineer.
- ML/model and evaluation engineer.
- Security/operations owner.
- Product/interface engineer.
- Platform specialist per active workstream.
- Independent safety/domain reviewers for robotics, medical/BCI, finance and space before consequential deployment.

Do not attempt all eight specialized platforms at production maturity with a single undifferentiated backlog. One founder can build the common core and prototypes, but clinical, flight, safety-critical robotics, and live financial systems require qualified domain collaboration and formal evidence.

## 13. First 30 days — concrete checklist

### Week 1
- Freeze a baseline commit and document local setup.
- Map each README claim to code/tests/benchmark.
- Run all tests and save machine-readable results.
- Inventory dependencies and model/tool entry points.
- Add `docs/architecture/`, `docs/decisions/`, and `docs/requirements/`.

### Week 2
- Define contracts for model gateway, context, memory, tools, policy decisions and task status.
- Add stable request/task IDs, structured errors and trace events.
- Create the evaluation harness interface and baseline report schema.
- Do not move every file; add adapters around the existing implementation first.

### Week 3
- Route Studio chat and CLI through one runtime entry point.
- Implement consistent streaming, cancellation and context limits.
- Connect one low-risk tool and one sandboxed tool.
- Add end-to-end test for the complete request path.

### Week 4
- Implement the first CortexCode vertical slice: repository inspect → search/read → patch → test → diff/report.
- Evaluate against a single-model baseline.
- Draft separate risk registers for IDE, Personal, Fin, OS, Robotics, Mesh, Health and Space.
- Choose one near-term product owner and freeze new side quests until the first product workflow is stable.

## 14. Definition of done for the overall architecture

CORTEX-M is ready to be considered a mature platform only when:
- Every platform uses stable shared contracts rather than duplicated runtimes.
- The model can be switched without rewriting capability workflows.
- Memory and knowledge are inspectable, scoped, versioned and deletable.
- Tools are permission-checked, bounded and observable.
- Every long task has state, cancellation, retry limits and failure behavior.
- Every product has its own tests, risk register, benchmarks and acceptance criteria.
- Performance and safety claims are reproducible on named hardware and workloads.
- Critical physical/medical/financial/space applications have independent domain validation and appropriate approval before operational use.
- A release can be built, tested, packaged, upgraded, rolled back, monitored and audited reproducibly.

## 15. Reference material for specialist tracks

- NASA, *Qualification of Flight Software*: https://swehb.nasa.gov/spaces/7150/pages/16449844/7.12%2B-%2BQualification%2Bof%2BFlight%2BSoftware
- NASA, *Software Engineering Procedural Requirements, Standards, and Related Resources*: https://www.nasa.gov/intelligent-systems-division/software-management-office/nasa-software-engineering-procedural-requirements-standards-and-related-resources/
- seL4 documentation: https://docs.sel4.systems/projects/sel4/index.html
- ISO 10218-1:2025, industrial robot safety requirements (scope is limited; determine the applicable standard for each robot type): https://www.iso.org/standard/73933.html
- FDA, implanted BCI devices—non-clinical testing and clinical considerations: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/implanted-brain-computer-interface-bci-devices-patients-paralysis-or-amputation-non-clinical-testing
- IEC 62304, medical-device software lifecycle processes: https://webstore.iec.ch/en/publication/6792
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework

These references are starting points, not a substitute for checking the current jurisdiction, standard edition, contractual requirements and specialist advice that apply to a real deployment.
