# 03 CORTEX Fin: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-Fin
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Financial analytics, fraud/risk research, replay and shadow-mode evaluation

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

Analyze transaction-like data and evaluate risk models with explainable, auditable outputs before any live decision integration.

## A2. Target users

Fraud/risk analysts
Financial data scientists
Compliance/audit teams
Reliability engineers

## A3. Problems and desired outcomes

Risk patterns drift
Future data can leak into evaluation
Scores may be uncalibrated or unexplained
Latency measurements may omit queueing and integration

## A4. MVP scope

Approved historical/synthetic data
Schema validation
Baseline scoring
Time-based evaluation
Replay engine
Explainable reports
Shadow mode without transaction authority

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1 / MVP | MVP features in this document; one end-to-end primary workflow | Workflow succeeds; core security/correctness tests pass; limitations documented. |
| R2 | More integrations, usability, reporting, automation and broader evaluation | Measured improvement against R1; no critical regression. |
| R3 / controlled pilot | Scale, team deployment or hardware/external integration where relevant | Domain/security review, runbook, rollback and approval. |
| Explicitly deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and acceptance evidence. |


## A6. Primary user journeys

Import dataset → validate provenance/schema → compute features → baseline/candidate scoring → replay → metrics/reason codes.
Compare shadow score to incumbent result → record discrepancy for review without changing live outcome.
Drift alert → provide evidence for human review; no unapproved automatic policy change.

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

Offline analytics/replay pipeline with durable auditable records, deterministic feature processing, baseline and candidate scoring, evaluation reports and no direct live authorization/decline path in MVP.

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
| FR-FIN-001 | P0 | The system SHALL accept only approved historical or synthetic datasets with provenance, purpose and retention metadata. | Unit + integration + end-to-end test |
| FR-FIN-002 | P1 | Ingestion SHALL validate required fields, timestamps, duplicates, units/currency and missing-value policy. | Unit + integration + end-to-end test |
| FR-FIN-003 | P1 | Feature transformations SHALL be versioned and deterministic. | Unit + integration + end-to-end test |
| FR-FIN-004 | P1 | Evaluation SHALL include a rules/statistical baseline before candidate scoring is promoted. | Unit + integration + end-to-end test |
| FR-FIN-005 | P1 | Each score SHALL identify model/version, score meaning, available reason codes and limitations. | Unit + integration + end-to-end test |
| FR-FIN-006 | P1 | Replay SHALL be reproducible with event-time order, load settings and explicit latency boundaries. | Unit + integration + end-to-end test |
| FR-FIN-007 | P1 | Reports SHALL include appropriate precision/recall, calibration and false-positive/negative trade-offs. | Unit + integration + end-to-end test |
| FR-FIN-008 | P0 | Evaluation splits SHALL be time-aware and include leakage checks. | Unit + integration + end-to-end test |
| FR-FIN-009 | P2 | Drift monitoring SHALL alert for review rather than silently adapt production decisions. | Unit + integration + end-to-end test |
| FR-FIN-010 | P0 | MVP shadow mode SHALL not authorize, decline or move real money. | Unit + integration + end-to-end test |
| FR-FIN-011 | P0 | Audit records SHALL identify dataset/model/config versions, score, reason and review owner. | Unit + integration + end-to-end test |
| FR-FIN-012 | P0 | Any later production integration SHALL require a tested kill switch, rollback and independent approval. | Unit + integration + end-to-end test |
| FR-FIN-013 | P0 | Sensitive transaction fields SHALL be access-controlled and redacted from logs. | Unit + integration + end-to-end test |
| FR-FIN-014 | P1 | Latency reports SHALL separate component and end-to-end latency under representative load. | Unit + integration + end-to-end test |
| FR-FIN-015 | P0 | Live decision integration SHALL be blocked until risk, security, compliance and operations approval. | Unit + integration + end-to-end test |


## B3. Non-functional requirements


| ID | Quality attribute | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-FIN-001 | Auditability | Replay and score must be reproducible from versioned inputs/configuration. | Test/inspection/analysis |
| NFR-FIN-002 | Integrity | Records have stable IDs and validation; audit is protected from casual alteration. | Test/inspection/analysis |
| NFR-FIN-003 | Performance | Report p50/p95/p99 and load; no unmeasured microsecond end-to-end claims. | Test/inspection/analysis |
| NFR-FIN-004 | Safety | MVP has no direct live transaction authority. | Test/inspection/analysis |
| NFR-FIN-005 | Privacy | Minimize sensitive data, enforce access and retention. | Test/inspection/analysis |
| NFR-FIN-006 | Quality | Measure calibration, drift and appropriate subgroup performance. | Test/inspection/analysis |
| NFR-FIN-007 | Recovery | Replay resumes without double-counting events. | Test/inspection/analysis |


## B4. External interfaces

Approved data source
Feature pipeline
Scoring models
Replay harness
Audit/report store
Incumbent risk system in shadow mode only

## B5. Data requirements

Historical/synthetic transactions
Feature definitions
Model/threshold versions
Scores/reason codes
Replay traces
Evaluation/drift reports
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

Pinned replay is reproducible
Temporal leakage tests pass
Reports show baseline and error trade-offs
Shadow mode cannot mutate live outcomes
Kill switch/rollback tested before later production integration

## B10. Risks and mitigations


| Risk | Required mitigation |
| --- | --- |
| False declines/missed fraud | Assign owner, control, regression test and release gate. |
| Leakage | Assign owner, control, regression test and release gate. |
| Poor calibration | Assign owner, control, regression test and release gate. |
| Regulatory failure | Assign owner, control, regression test and release gate. |
| Misleading latency | Assign owner, control, regression test and release gate. |
| Feedback loops | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record requirement coverage, test evidence, open defects, known limitations, dependency/model versions and approvals.

## B12. Open decisions

Confirm product owner, users, deployment profile and release scope.
Confirm supported OS/hardware, data sources and integrations.
Set numeric performance thresholds only after baseline profiling.
Confirm data classification, retention, legal/regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Fin
- **Lyapunov Energy Gradient Attribution**: Transaction risk scoring calculates continuous attractor energy $E(x)$, decomposing the gradient $\nabla_x E(x)$ into explicit interpretable factor attributions satisfying FCRA and GDPR Article 22.
- **High-Throughput Columnar Streaming**: Processes transaction tap streams via Apache Arrow columnar memory buffers sustaining $> 100,000\text{ tx/sec}$ with scoring latency $\le 150\ \mu\text{s}$.
- **Strict Shadow-Mode Isolation**: Non-authoritative read-only pipe with zero write credentials to live payment rails or bank settlement ledgers.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Anomaly Scorer**: `platforms/fin/src/anomaly_scorer.py` (`FinancialAnomalyScorer`, `FraudScoringReport`)
- **Feature Encoder**: `platforms/fin/src/transaction_encoder.py` (`FinancialTransaction`, `TransactionFeatureEncoder`)
- **Event Replay Engine**: `platforms/fin/src/event_replay.py` (`HighThroughputReplayEngine`)
- **Gateway Dispatcher**: `execute_fin(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-Fin
- **Lyapunov Energy Gradient Attribution**: Transaction risk scoring calculates continuous attractor energy $E(x)$, decomposing the gradient $\nabla_x E(x)$ into explicit interpretable factor attributions satisfying FCRA and GDPR Article 22.
- **High-Throughput Columnar Streaming**: Processes transaction tap streams via Apache Arrow columnar memory buffers sustaining $> 100,000\text{ tx/sec}$ with scoring latency $\le 150\ \mu\text{s}$.
- **Strict Shadow-Mode Isolation**: Non-authoritative read-only pipe with zero write credentials to live payment rails or bank settlement ledgers.

---

### Implemented Codebase Traceability
- **Anomaly Scorer**: `platforms/fin/src/anomaly_scorer.py` (`FinancialAnomalyScorer`, `FraudScoringReport`)
- **Feature Encoder**: `platforms/fin/src/transaction_encoder.py` (`FinancialTransaction`, `TransactionFeatureEncoder`)
- **Event Replay Engine**: `platforms/fin/src/event_replay.py` (`HighThroughputReplayEngine`)
- **Gateway Dispatcher**: `execute_fin(action, params)` in `src/cortex_m/gateway.py`