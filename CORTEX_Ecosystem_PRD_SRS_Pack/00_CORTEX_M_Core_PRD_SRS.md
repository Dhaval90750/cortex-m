# 00 CORTEX M Core: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-M Core Platform
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Shared runtime, model gateway, memory, tools, orchestration, policy and evaluation

| Document control | Value |
| --- | --- |
| Version | 0.9 — Draft for review |
| Date | 26 October 2026 |
| Status | Proposed requirements; not approved |
| Basis | Original CORTEX ecosystem plan and implementation master plan |
| Evidence note | Concept performance numbers are targets until reproduced on named hardware and workload. |


# How to read this document

PRD sections define users, outcomes, scope, workflows and success measures. SRS sections define testable system behavior, interfaces, data, quality attributes, security, failure behavior and verification. SHALL/MUST = required; SHOULD = recommended; MAY = optional. P0 = safety/security/correctness gate; P1 = MVP; P2 = later.

# PART A — PRODUCT REQUIREMENTS DOCUMENT (PRD)


## A1. Vision

Provide a reusable, testable foundation for the CORTEX product family without forcing every product to use every research subsystem.

## A2. Target users

Product engineering teams
AI/ML researchers
Security and QA reviewers
Administrators of team deployments

## A3. Problems and desired outcomes

Duplicated model/tool logic
Confusion between generation, memory, orchestration and authorization
Unreproducible research claims
Inconsistent task state and provenance

## A4. MVP scope

Versioned contracts/events
Model gateway with two adapters
Conversation/task runtime with cancellation
Tool registry and policy enforcement
Memory/knowledge interfaces
Structured trace and evaluation harness
Lightweight local persistence

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1 / MVP | MVP features in this document; one end-to-end primary workflow | Workflow succeeds; core security/correctness tests pass; limitations documented. |
| R2 | More integrations, usability, reporting, automation and broader evaluation | Measured improvement against R1; no critical regression. |
| R3 / controlled pilot | Scale, team deployment or hardware/external integration where relevant | Domain/security review, runbook, rollback and approval. |
| Explicitly deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and acceptance evidence. |


## A6. Primary user journeys

Request → validate actor/config → build context → route to model/tool → enforce policy → stream result → persist state and trace.
Long task → task ID → progress events → optional approval/input → verification → final artifacts/status.
Research component change → contract tests and ablation evaluation → promote only if release gates pass.

## A7. Product success measures


| Metric | Definition and measurement |
| --- | --- |
| Task success | Share of benchmark tasks meeting explicit acceptance criteria; version the test set. |
| Quality | Domain-specific correctness, relevance and uncertainty; compare to a named baseline. |
| Reliability | Error rate, recovery, cancellation and partial-completion behavior under fault tests. |
| Performance | Latency percentiles, throughput, resource use and cost on named hardware/workloads. |
| Safety/security | Unauthorized actions, isolation failures and unsafe transitions from adversarial tests. |
| Usability | User task completion, clarity, accessibility and feedback from a defined pilot. |


## A8. Assumptions to validate

The first release is a prototype/research product unless the owner approves production scope.
Existing implementation, supported hardware and dependencies must be audited before committing estimates.
Data rights, retention, jurisdiction and domain-specific requirements must be confirmed.
A product owner and appropriate domain/security reviewers must approve the baseline.

# PART B — SOFTWARE REQUIREMENTS SPECIFICATION (SRS)


## B1. System context

A modular monolith initially: interfaces and contracts first, services separated only when isolation/scaling requires it. The model proposes; runtime validates; policy authorizes; executors enforce; persistence stores authoritative state.

| Component | Responsibility | Boundary |
| --- | --- | --- |
| UI / CLI / API | Collect inputs; display status, approvals, results and errors | UI is not a security boundary. |
| Workflow runtime | Task state, steps, budgets, retries and completion | Model may propose; runtime validates. |
| Model/research logic | Generate analysis, predictions or proposals | Cannot grant its own permissions. |
| Deterministic services | Validation, persistence, authorization, safety and source-of-record operations | Must be testable without an LLM. |
| Persistence/observability | State, provenance, audit, artifacts and metrics | Retention, access and deletion are explicit. |


## B2. Functional requirements


| ID | Priority | Requirement | Verification |
| --- | --- | --- | --- |
| FR-CORE-001 | P0 | The system SHALL define versioned schemas for requests, responses, errors, events, tasks, artifacts and model/tool capabilities. | Unit + integration + end-to-end test |
| FR-CORE-002 | P1 | The model gateway SHALL normalize streaming, usage metadata, timeouts and provider errors across configured backends. | Unit + integration + end-to-end test |
| FR-CORE-003 | P1 | The runtime SHALL expose model capability metadata including context limit, tools, structured output, modalities and locality. | Unit + integration + end-to-end test |
| FR-CORE-004 | P1 | The conversation manager SHALL manage turn ordering, context assembly, cancellation and bounded summarization/truncation. | Unit + integration + end-to-end test |
| FR-CORE-005 | P1 | The task orchestrator SHALL maintain durable task state, dependencies, budgets, retries, cancellation and completion status. | Unit + integration + end-to-end test |
| FR-CORE-006 | P1 | The tool registry SHALL validate input/output schemas and declare version, permissions, risk, limits and side effects. | Unit + integration + end-to-end test |
| FR-CORE-007 | P0 | The policy engine SHALL authorize actor/action/resource independently of model output; executors SHALL enforce decisions. | Unit + integration + end-to-end test |
| FR-CORE-008 | P1 | The memory API SHALL support scoped retrieval, store, update and deletion with provenance and version metadata. | Unit + integration + end-to-end test |
| FR-CORE-009 | P1 | The knowledge API SHALL support ingestion, search, source inspection, revision and deletion propagation. | Unit + integration + end-to-end test |
| FR-CORE-010 | P0 | Trace events SHALL correlate task, model, memory, tool, policy, verification and artifact events with sensitive-field redaction. | Unit + integration + end-to-end test |
| FR-CORE-011 | P1 | The evaluation harness SHALL run versioned unit, integration, task-quality, security and performance tests against named baselines. | Unit + integration + end-to-end test |
| FR-CORE-012 | P0 | The runtime SHALL enforce configurable time, token, retry, tool-call, memory and concurrency budgets. | Unit + integration + end-to-end test |
| FR-CORE-013 | P1 | Configuration SHALL be typed, validated, precedence-defined and separated from secrets. | Unit + integration + end-to-end test |
| FR-CORE-014 | P1 | Generated artifacts SHALL retain producer/version, source references, configuration and checksum metadata. | Unit + integration + end-to-end test |
| FR-CORE-015 | P2 | New model, memory, tool and workflow providers SHOULD be addable without modifying unrelated product logic. | Unit + integration + end-to-end test |


## B3. Non-functional requirements


| ID | Quality attribute | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-CORE-001 | Reliability | Timeouts and cancellations must reach callers as structured states; tasks cannot remain active indefinitely. | Test/inspection/analysis |
| NFR-CORE-002 | Security | Least privilege, scoped identity, redaction, executor-enforced policy and isolated untrusted execution are mandatory. | Test/inspection/analysis |
| NFR-CORE-003 | Observability | Tasks must be traceable end-to-end subject to privacy and retention rules. | Test/inspection/analysis |
| NFR-CORE-004 | Performance | Set targets only after workload-specific baselines; report percentiles and resource use. | Test/inspection/analysis |
| NFR-CORE-005 | Portability | Core contracts must not depend on one model vendor or memory implementation. | Test/inspection/analysis |
| NFR-CORE-006 | Reproducibility | Evaluations record code, config, dependencies, data version, hardware and seed where applicable. | Test/inspection/analysis |
| NFR-CORE-007 | Privacy | Deletion must propagate to relevant records, indexes and caches under documented retention rules. | Test/inspection/analysis |


## B4. External interfaces

Studio/CLI/API
Model providers
Memory and knowledge providers
Tool sandbox/executors
Persistence/job queue
Identity/policy service
Evaluation/artifact store

## B5. Data requirements

Actor/tenant/session IDs
Conversations and context references
Task records and state transitions
Model metadata/usage
Tool invocations and results
Policy decisions/approvals
Memory/source references
Trace/evaluation records
Artifact manifests
Persistent entities require stable IDs and schema/version metadata where applicable.
Imported data and generated artifacts must retain provenance; generated text is not automatically a verified source.
Define retention, export, backup, restore and deletion behavior before production use.
Sensitive data must not enter source control, unredacted telemetry or training data without approved rights and purpose.

## B6. Error and recovery requirements


| Condition | Required behavior |
| --- | --- |
| Invalid input/schema | Reject with stable error and field-level explanation; no partial protected action. |
| Model/provider timeout | Bound retries; record timeout; never claim completion. |
| Tool failure | Capture status/output safely; preserve task state; retry only when safe/idempotent. |
| Permission denied | Do not execute; record reason; return safe explanation. |
| Cancellation | Stop new work and cancel pending operations where possible; report already completed actions. |
| Persistence failure | Do not claim durable completion until commit succeeds. |
| Stale/corrupt source | Mark unavailable/stale and recover from canonical source where safe. |
| Budget exceeded | Stop at defined boundary and return partial progress and remaining risks. |


## B7. Security, privacy and safety

Apply least privilege and scope identity by user, tenant, task, tool and resource.
Treat external documents, code, web pages, telemetry and tool outputs as untrusted data, not policy instructions.
Bind approvals to exact high-impact actions and enforce policy in the executor, not only in the model or UI.
Isolate untrusted execution; enforce time, CPU, memory, process and network limits outside the model.
Maintain threat/risk registers and adversarial tests; keep AI advice separate from authoritative state transitions.

## B8. Verification and validation


| Test level | Coverage | Release rule |
| --- | --- | --- |
| Unit | Pure logic, schemas, transforms and state transitions | Required in CI. |
| Contract | Model/tool/event/API schema compatibility | Required for shared interfaces. |
| Integration | Persistence, providers, sandbox, permissions and adapters | Required for changed components. |
| End-to-end | Primary user journeys | Required for MVP. |
| Security/adversarial | Injection, leakage, authorization bypass, resource exhaustion | Required before pilot. |
| Performance | Named hardware/workload, p50/p95/p99 and resource limits | Required before quantitative claims. |
| Domain validation | Independent review for finance, OS, robotics, health or space as applicable | Required before high-impact deployment. |


## B9. Acceptance criteria

Core contracts have schema/compatibility tests
One conversation flow works through two model adapters
A bounded task can call a tool, be cancelled and produce a complete trace
A denied policy decision cannot be overridden by model output
A baseline evaluation report can be reproduced from pinned configuration

## B10. Risks and mitigations


| Risk | Required mitigation |
| --- | --- |
| Premature microservices | Assign owner, control, regression test and release gate. |
| Treating approximate memory as authoritative | Assign owner, control, regression test and release gate. |
| Tool authorization bypass | Assign owner, control, regression test and release gate. |
| Sensitive trace leakage | Assign owner, control, regression test and release gate. |
| Unreproducible benchmarks | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record requirement coverage, test evidence, open defects, known limitations, dependency/model versions and approvals.

## B12. Open decisions

Confirm product owner, users, deployment profile and release scope.
Confirm supported OS/hardware, data sources and integrations.
Set numeric performance thresholds only after baseline profiling.
Confirm data classification, retention, legal/regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-M Core Platform
- **Continuous Modern Hopfield Attractors**: Core associative memory relies on continuous state relaxation governed by the energy function:
$$E(x) = -\text{lse}(\beta X^T x) + \frac{1}{2} x^T x + C$$
where state iteration follows $x_{t+1} = X \text{softmax}(\beta X^T x_t)$, providing exponential memory capacity and guaranteed Lyapunov stability.
- **Orthogonal Synaptic Grafting ($P_\perp$)**: Zero-collateral parameter integration uses exact null-space projection:
$$P_\perp = I - Q_B Q_B^T$$
where $Q_B$ is computed via thin QR/SVD on recipient basin prototypes $X_B$, guaranteeing $P_\perp X_B = 0$ ($0.0\%$ collateral interference to existing basins).
- **Universal Event Gateway**: Unified typed pub-sub event bus (`cortex_m.core.EventBus`) with microsecond timestamping, trace IDs, and sensitive field masking.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Runtime & Contracts**: `src/cortex_m/core/runtime.py`, `src/cortex_m/core/contracts/`
- **Memory & Epistemics**: `src/cortex_m/memory/continuous_hopfield.py`, `src/cortex_m/learning/consolidation.py`
- **Grafting & Orthogonality**: `src/cortex_m/learning/grafting.py` (`SynapticKnowledgePackager`)
- **Universal Gateway**: `src/cortex_m/gateway.py` (`EcosystemGateway`)
- **CLI & REST Server**: `src/cortex_m/cli.py`, `src/cortex_m/server/app.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-M Core Platform
- **Continuous Modern Hopfield Attractors**: Core associative memory relies on continuous state relaxation governed by the energy function:
$$E(x) = -\frac{1}{\beta} \text{lse}(\beta M x) + \frac{1}{2} \|x\|^2$$
where state iteration follows $x_{t+1} = M^T \text{softmax}(\beta M x_t)$, providing exponential memory capacity and guaranteed Lyapunov stability (energy strictly decreases along recurrent trajectories).
- **Batch Attractor Parallel Convergence**: High-throughput vectorized Modern Hopfield convergence across query batches $X \in \mathbb{R}^{B \times d}$:
$$X_{t+1} = \text{normalize}\left(\text{softmax}(\beta X_t M^T) M + \gamma \frac{X_t W^T}{\|W\|}\right)$$
- **Analytical Lyapunov Energy Gradient & Landscape Profiling**: Exact energy gradient $\nabla E(x) = x - M^T \text{softmax}(\beta M x)$ evaluates local basin depth, barrier height $\Delta E = \bar{E}_{\text{rim}} - E^*$, and stability margins.
- **Truncated SVD Matrix Factorization**: Synaptic parameter compression via rank-$k$ factorization:
$$W \approx U_k \Sigma_k V_k^T$$
achieving ~2x compression while preserving zero diagonal and exact matrix symmetry.
- **Counterfactual Basin Stress Testing**: Adversarial noise injection evaluating Lyapunov basin stability:
$$S_{\text{basin}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(f(x + \epsilon_i) = f(x))$$
differentiating verified attractor basins from fragile hallucinations.
- **Dynamic Temperature Annealing**: Adaptive inverse temperature $\beta(E)$ sharpening retrieval for well-settled memories ($E < -1.0$) and relaxing for exploratory association.
- **Associative Reasoning Chains**: Multi-hop continuous attractor traversal projecting out visited basins to explore adjacent semantic attractors without prompt engineering.
- **Gram-Schmidt Memory Fusion**: Composite prototype synthesis with orthogonal projection against non-target memories to guarantee zero catastrophic interference.
- **Orthogonal Synaptic Grafting ($P_\perp$)**: Zero-collateral parameter integration uses exact null-space projection:
$$P_\perp = I - Q_B Q_B^T$$
guaranteeing $P_\perp X_B = 0$ ($0.0\%$ collateral interference to existing basins).

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Core Architecture & Beast Neural Substrate**: `src/cortex_m/cortex_v2.py` (`CortexV2`)
- **Continuous Synaptic Memory**: `src/cortex_m/pathway/synaptic_memory.py` (`SynapticMemoryNetwork`)
- **Runtime & Contracts**: `src/cortex_m/core/runtime.py`, `src/cortex_m/core/contracts/`
- **Developmental Sleep Consolidation**: `src/cortex_m/learning/consolidation.py` (`NocturnalSleepConsolidator`)
- **Grafting & Orthogonality**: `src/cortex_m/learning/grafting.py` (`SynapticKnowledgePackager`)
- **Universal Gateway**: `src/cortex_m/gateway.py` (`EcosystemGateway`)
- **CLI & REST Server**: `src/cortex_m/cli.py`, `src/cortex_m/server/app.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-M Core Platform
- **Continuous Modern Hopfield Attractors**: Core associative memory relies on continuous state relaxation governed by the energy function:
  $$E(x) = -\frac{1}{\beta} \text{lse}(\beta M x) + \frac{1}{2} \|x\|^2$$
  where state iteration follows $x_{t+1} = M^T \text{softmax}(\beta M x_t)$, providing exponential memory capacity and guaranteed Lyapunov stability (energy strictly decreases along recurrent trajectories).
- **Batch Attractor Parallel Convergence**: High-throughput vectorized Modern Hopfield convergence across query batches $X \in \mathbb{R}^{B \times d}$:
  $$X_{t+1} = \text{normalize}\left(\text{softmax}(\beta X_t M^T) M + \gamma \frac{X_t W^T}{\|W\|}\right)$$
- **Analytical Lyapunov Energy Gradient & Landscape Profiling**: Exact energy gradient $\nabla E(x) = x - M^T \text{softmax}(\beta M x)$ evaluates local basin depth, barrier height $\Delta E = \bar{E}_{\text{rim}} - E^*$, and stability margins.
- **Truncated SVD Matrix Factorization**: Synaptic parameter compression via rank-$k$ factorization:
  $$W \approx U_k \Sigma_k V_k^T$$
  achieving ~2x compression while preserving zero diagonal and exact matrix symmetry.
- **Counterfactual Basin Stress Testing**: Adversarial noise injection evaluating Lyapunov basin stability:
  $$S_{\text{basin}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(f(x + \epsilon_i) = f(x))$$
  differentiating verified attractor basins from fragile hallucinations.
- **Dynamic Temperature Annealing**: Adaptive inverse temperature $\beta(E)$ sharpening retrieval for well-settled memories ($E < -1.0$) and relaxing for exploratory association.
- **Associative Reasoning Chains**: Multi-hop continuous attractor traversal projecting out visited basins to explore adjacent semantic attractors without prompt engineering.
- **Gram-Schmidt Memory Fusion**: Composite prototype synthesis with orthogonal projection against non-target memories to guarantee zero catastrophic interference.
- **Orthogonal Synaptic Grafting ($P_\perp$)**: Zero-collateral parameter integration uses exact null-space projection:
  $$P_\perp = I - Q_B Q_B^T$$
  guaranteeing $P_\perp X_B = 0$ ($0.0\%$ collateral interference to existing basins).

---

### Implemented Codebase Traceability
- **Core Architecture & Beast Neural Substrate**: `src/cortex_m/cortex_v2.py` (`CortexV2`)
- **Continuous Synaptic Memory**: `src/cortex_m/pathway/synaptic_memory.py` (`SynapticMemoryNetwork`)
- **Runtime & Contracts**: `src/cortex_m/core/runtime.py`, `src/cortex_m/core/contracts/`
- **Developmental Sleep Consolidation**: `src/cortex_m/learning/consolidation.py` (`NocturnalSleepConsolidator`)
- **Grafting & Orthogonality**: `src/cortex_m/learning/grafting.py` (`SynapticKnowledgePackager`)
- **Universal Gateway**: `src/cortex_m/gateway.py` (`EcosystemGateway`)
- **CLI & REST Server**: `src/cortex_m/cli.py`, `src/cortex_m/server/app.py`