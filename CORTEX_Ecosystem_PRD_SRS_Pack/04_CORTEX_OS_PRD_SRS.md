# 04 CORTEX OS: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-OS
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Research-oriented cognitive operating environment, beginning in userspace

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

Explore semantic search, task scheduling and cognitive services while retaining conventional filesystem, process and permission semantics.

## A2. Target users

Systems researchers
OS/runtime engineers
Developer-tool researchers
Security reviewers

## A3. Problems and desired outcomes

Semantic retrieval may be confused with authoritative filesystem behavior
Privileged OS work is high-risk
Scheduling claims need baseline comparisons
Kernel claims need independent evidence

## A4. MVP scope

Userspace service/CLI
Conventional filesystem semantics
Supplemental semantic index
Scheduler simulator
Process supervision
Permission/recovery tests

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1 / MVP | MVP features in this document; one end-to-end primary workflow | Workflow succeeds; core security/correctness tests pass; limitations documented. |
| R2 | More integrations, usability, reporting, automation and broader evaluation | Measured improvement against R1; no critical regression. |
| R3 / controlled pilot | Scale, team deployment or hardware/external integration where relevant | Domain/security review, runbook, rollback and approval. |
| Explicitly deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and acceptance evidence. |


## A6. Primary user journeys

Userspace service starts → standard filesystem remains authoritative → semantic index returns candidate paths → source freshness checked.
Versioned scheduler workload runs → baseline/candidate compared → latency, fairness and failure behavior reported.
Privileged/kernel feature is blocked until threat model, requirements, design review and target test plan are approved.

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

Start as userspace services on a mature OS. Semantic retrieval is supplemental; canonical file identity, permissions, process isolation and recovery remain authoritative. Kernel work is a separate gated research programme.

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
| FR-OS-001 | P1 | Initial release SHALL run as a userspace service on a mature OS without a custom kernel dependency. | Unit + integration + end-to-end test |
| FR-OS-002 | P0 | File operations SHALL preserve conventional identity, permissions, atomicity and error semantics. | Unit + integration + end-to-end test |
| FR-OS-003 | P1 | Semantic search MAY return candidate paths but SHALL resolve them to canonical current sources. | Unit + integration + end-to-end test |
| FR-OS-004 | P1 | The index SHALL track source changes, invalidation, reindexing and crash recovery. | Unit + integration + end-to-end test |
| FR-OS-005 | P2 | Scheduler policies SHALL be prototyped in a simulator with versioned workloads and fairness metrics. | Unit + integration + end-to-end test |
| FR-OS-006 | P1 | Process supervision SHALL enforce resource limits and explicit restart policies. | Unit + integration + end-to-end test |
| FR-OS-007 | P0 | User/service/admin capabilities SHALL be separated and privilege escalation denied by default. | Unit + integration + end-to-end test |
| FR-OS-008 | P1 | Service lifecycle, policy and scheduler outcomes SHALL be auditable with sensitive data redacted. | Unit + integration + end-to-end test |
| FR-OS-009 | P1 | Benchmarks SHALL compare semantic search/scheduling against conventional implementations. | Unit + integration + end-to-end test |
| FR-OS-010 | P0 | Kernel/privileged work SHALL require a threat model, target, test matrix and specialist review. | Unit + integration + end-to-end test |
| FR-OS-011 | P1 | Interrupted operations and corrupted indexes SHALL have defined recovery behavior. | Unit + integration + end-to-end test |
| FR-OS-012 | P1 | Supported OS, filesystem assumptions and toolchains SHALL be documented. | Unit + integration + end-to-end test |


## B3. Non-functional requirements


| ID | Quality attribute | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-OS-001 | Isolation | Ordinary functions do not require elevated privileges. | Test/inspection/analysis |
| NFR-OS-002 | Correctness | Semantic results never replace canonical file operations. | Test/inspection/analysis |
| NFR-OS-003 | Security | No exploit-immunity claim from an energy/learned score; OS mechanisms enforce boundaries. | Test/inspection/analysis |
| NFR-OS-004 | Performance | Measure correctness, p50/p95 latency, CPU/RAM and index maintenance. | Test/inspection/analysis |
| NFR-OS-005 | Recovery | Interrupted operations leave consistent recoverable state. | Test/inspection/analysis |
| NFR-OS-006 | Portability | Supported OS/toolchain combinations are declared and tested. | Test/inspection/analysis |


## B4. External interfaces

Host OS/filesystem
CORTEX-M runtime/memory adapter
CLI/API
Scheduler simulator
Metrics/evaluation harness

## B5. Data requirements

Index metadata
Canonical path references
Scheduler workloads
Process/service state
Benchmark traces
Config/audit events
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

Userspace service works without elevated privileges
Semantic results resolve to current canonical paths
Index recovers after forced termination
Search/scheduler compared to conventional baselines
Kernel work remains gated pending independent review

## B10. Risks and mitigations


| Risk | Required mitigation |
| --- | --- |
| Stale semantic paths | Assign owner, control, regression test and release gate. |
| Privilege escalation | Assign owner, control, regression test and release gate. |
| Overclaimed kernel properties | Assign owner, control, regression test and release gate. |
| Filesystem behavior breakage | Assign owner, control, regression test and release gate. |
| Toolchain fragility | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record requirement coverage, test evidence, open defects, known limitations, dependency/model versions and approvals.

## B12. Open decisions

Confirm product owner, users, deployment profile and release scope.
Confirm supported OS/hardware, data sources and integrations.
Set numeric performance thresholds only after baseline profiling.
Confirm data classification, retention, legal/regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-OS
- **Non-Privileged Userspace Indexing**: Filesystem notifications monitored via cross-platform debounced watchers (500ms debounce) running entirely under unprivileged user permissions (`UID != 0`).
- **Dynamic Cognitive Affinity Graph**: Semantic document clustering via modern Hopfield associative affinity:
$$A_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$$
clustering related research and code artifacts without altering canonical disk hierarchies.
- **Sub-Millisecond Spotlight Omnibar**: Sub-millisecond continuous attractor resonance search over 64D gist embeddings.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **File Watcher**: `platforms/os/src/file_watcher.py` (`NonPrivilegedFileWatcher`)
- **Affinity Clustering**: `platforms/os/src/affinity_cluster.py` (`CognitiveAffinityGraph`)
- **Omnibar Engine**: `platforms/os/src/omnibar_search.py` (`SpotlightOmnibarEngine`)
- **System Monitor**: `platforms/os/src/system_monitor.py` (`SystemResourceMonitor`)
- **Gateway Dispatcher**: `execute_os(action, params)` in `src/cortex_m/gateway.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-OS
- **Non-Privileged Userspace Indexing**: Filesystem notifications monitored via cross-platform debounced watchers (500ms debounce) running entirely under unprivileged user permissions (`UID != 0`).
- **Dynamic Cognitive Affinity Graph**: Semantic document clustering via modern Hopfield associative affinity:
$$A_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$$
clustering related research and code artifacts without altering canonical disk hierarchies.
- **Newman-Girvan Community Modularity**: Quantifies partitioning quality via:
$$Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right) \delta(c_i, c_j)$$
- **Centroid Cosine Resonance Membership**: Predicts incoming document cluster assignment via cosine projection onto cluster centroid vectors.
- **Sub-Millisecond Spotlight Omnibar**: Sub-millisecond continuous attractor resonance search over 64D gist embeddings.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **File Watcher**: `platforms/os/src/file_watcher.py` (`NonPrivilegedFileWatcher`, `FileDocument`)
- **Affinity Clustering**: `platforms/os/src/affinity_cluster.py` (`CognitiveAffinityGraph`, `ProjectCluster`)
- **Omnibar Engine**: `platforms/os/src/omnibar_search.py` (`SpotlightOmnibarEngine`)
- **System Monitor**: `platforms/os/src/system_monitor.py` (`SystemResourceMonitor`)
- **Gateway Dispatcher**: `execute_os(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-OS
- **Non-Privileged Userspace Indexing**: Filesystem notifications monitored via cross-platform debounced watchers (500ms debounce) running entirely under unprivileged user permissions (`UID != 0`).
- **Dynamic Cognitive Affinity Graph**: Semantic document clustering via modern Hopfield associative affinity:
  $$A_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$$
  clustering related research and code artifacts without altering canonical disk hierarchies.
- **Newman-Girvan Community Modularity**: Quantifies partitioning quality via:
  $$Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right) \delta(c_i, c_j)$$
- **Centroid Cosine Resonance Membership**: Predicts incoming document cluster assignment via cosine projection onto cluster centroid vectors.
- **Sub-Millisecond Spotlight Omnibar**: Sub-millisecond continuous attractor resonance search over 64D gist embeddings.

---

### Implemented Codebase Traceability
- **File Watcher**: `platforms/os/src/file_watcher.py` (`NonPrivilegedFileWatcher`, `FileDocument`)
- **Affinity Clustering**: `platforms/os/src/affinity_cluster.py` (`CognitiveAffinityGraph`, `ProjectCluster`)
- **Omnibar Engine**: `platforms/os/src/omnibar_search.py` (`SpotlightOmnibarEngine`)
- **System Monitor**: `platforms/os/src/system_monitor.py` (`SystemResourceMonitor`)
- **Gateway Dispatcher**: `execute_os(action, params)` in `src/cortex_m/gateway.py`