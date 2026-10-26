# 06 CORTEX Mesh: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-Mesh
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Civilian distributed coordination for logistics, sensing, mapping and disaster-response simulation

| Document control | Value |
| --- | --- |
| Version | 0.9 — Draft for review |
| Date | 26 October 2026 |
| Status | Proposed requirements; not approved |
| Basis | Original CORTEX ecosystem plan and implementation master plan |
| Evidence note | Concept performance numbers remain targets until reproduced on named hardware/workloads. |


# How to read this document

PRD defines user needs, outcomes, scope and success measures. SRS defines testable behavior, interfaces, data, quality attributes, security, failure behavior and verification. SHALL/MUST = required; SHOULD = recommended; MAY = optional. P0 = safety/security/correctness gate; P1 = MVP; P2 = later.

# PART A — PRODUCT REQUIREMENTS DOCUMENT (PRD)


## A1. Vision

Coordinate authorized nodes with transparent communications, validated messages, fault handling and safe degraded operation.

## A2. Target users

Distributed-systems engineers
Warehouse/logistics operators
Environmental monitoring teams
Disaster-response researchers

## A3. Problems and desired outcomes

Nodes disconnect/rejoin
Messages are delayed, duplicated or malformed
Topology changes disrupt coordination
Distributed state diverges

## A4. MVP scope

Node identity/health
Versioned message schemas
Topology simulator
Civilian task coordination
Fault injection
Audit log
Degraded-mode rules

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1/MVP | MVP capabilities and one complete primary workflow | Workflow works; correctness/security tests pass; limitations documented. |
| R2 | Usability, reporting, integrations and wider evaluations | Measured improvement against R1 without critical regressions. |
| R3/controlled pilot | Scale, external systems or hardware only where applicable | Domain/security review, runbook, rollback and approval. |
| Deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and evidence required. |


## A6. Primary user journeys

Node enrolls → reports health → message is authenticated/validated → task assignment recorded → acknowledgement/failure tracked.
Partition occurs → stale state marked → documented degraded behavior → reconciliation after reconnect.
Unauthenticated or malformed message → reject and record security event.

## A7. Product success measures


| Metric | Measurement |
| --- | --- |
| Task success | Versioned tasks meeting acceptance criteria. |
| Quality | Domain correctness/utility compared with named baseline. |
| Reliability | Error, recovery, cancellation and partial-completion behavior. |
| Performance | Latency percentiles, throughput and resource use on named workload. |
| Safety/security | Unauthorized actions, leakage and unsafe transitions. |
| Usability | User completion, clarity, accessibility and pilot feedback. |


## A8. Assumptions to validate

Prototype/research scope applies until production use is explicitly approved.
Audit current implementation and hardware before setting delivery or performance promises.
Confirm data rights, retention, jurisdiction and domain requirements.
Obtain approval from product owner and relevant domain/security reviewers.

# PART B — SOFTWARE REQUIREMENTS SPECIFICATION (SRS)


## B1. System context

Civilian distributed service with authenticated nodes, versioned messages, topology state, task coordination, reconciliation and an operator-visible audit. Simulation precedes field trials.

| Component | Responsibility | Boundary |
| --- | --- | --- |
| UI/CLI/API | Input, status, approvals, results and errors | Not a security boundary. |
| Workflow runtime | Task state, budgets, retries and completion | Model proposes; runtime validates. |
| Model/research logic | Analysis, predictions or proposals | Cannot authorize its own actions. |
| Deterministic services | Validation, persistence, safety and source-of-record operations | Must be testable without a model. |
| Persistence/observability | State, provenance, artifacts and metrics | Retention/access/deletion explicit. |


## B2. Functional requirements


| ID | Priority | Requirement | Verification |
| --- | --- | --- | --- |
| FR-MESH-001 | P0 | Each node SHALL have unique identity and authenticated credentials with enrollment and revocation. | Unit + integration + end-to-end |
| FR-MESH-002 | P1 | Track node availability, software version, resource health and last-seen time. | Unit + integration + end-to-end |
| FR-MESH-003 | P0 | Messages SHALL validate schema/version, sender, timestamp, size and duplicate/replay handling. | Unit + integration + end-to-end |
| FR-MESH-004 | P1 | Represent changing connectivity and retain topology history for diagnosis. | Unit + integration + end-to-end |
| FR-MESH-005 | P1 | Support civilian logistics, environmental sensing, cooperative mapping and disaster-response scenarios. | Unit + integration + end-to-end |
| FR-MESH-006 | P1 | Define task ownership, idempotency, conflict resolution and state reconciliation. | Unit + integration + end-to-end |
| FR-MESH-007 | P1 | Simulate node failure, partition, clock skew, loss, duplication and recovery. | Unit + integration + end-to-end |
| FR-MESH-008 | P0 | Define safe degraded behavior when coordination or connectivity is unavailable. | Unit + integration + end-to-end |
| FR-MESH-009 | P0 | Record task assignment, node status, message outcome and operator override. | Unit + integration + end-to-end |
| FR-MESH-010 | P0 | Authenticate nodes, authorize message types, rotate/revoke credentials and limit exposed services. | Unit + integration + end-to-end |
| FR-MESH-011 | P2 | Dashboard SHALL show node health, connectivity, task assignments and stale data. | Unit + integration + end-to-end |
| FR-MESH-012 | P0 | Field trials require communications, privacy, safety and operational review. | Unit + integration + end-to-end |


## B3. Non-functional requirements


| ID | Quality | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-MESH-001 | Resilience | Detect stale state and define behavior during partitions. | Test/inspection/analysis |
| NFR-MESH-002 | Security | Only authenticated/authorized nodes can issue accepted coordination messages. | Test/inspection/analysis |
| NFR-MESH-003 | Auditability | Task assignments and overrides are traceable. | Test/inspection/analysis |
| NFR-MESH-004 | Interoperability | Versioned schemas reject incompatible messages safely. | Test/inspection/analysis |
| NFR-MESH-005 | Scope | Civilian coordination only; no combat, targeting, detection evasion or military swarm behavior. | Test/inspection/analysis |
| NFR-MESH-006 | Performance | Measure delivery, convergence, resource use and recovery under defined network conditions. | Test/inspection/analysis |


## B4. External interfaces

Node agents
Network transport adapters
Identity service
Topology simulator
CORTEX-M coordinator
Operator dashboard
Audit store

## B5. Data requirements

Node identities/capabilities
Validated messages
Topology snapshots
Task assignments
Health telemetry
Fault results
Audit events
Persistent entities require stable IDs and schema/version metadata where applicable.
Record provenance for imported data and generated artifacts.
Define retention, export, backup, restore and deletion behavior before production use.
Keep sensitive data out of source control, unredacted telemetry and training datasets without approved rights and purpose.

## B6. Error and recovery behavior


| Condition | Required behavior |
| --- | --- |
| Invalid input | Reject with stable error; do not partially execute protected actions. |
| Model timeout | Bound retries and record timeout; never claim completion. |
| Tool failure | Capture status safely; retry only when safe/idempotent. |
| Permission denied | Do not execute; record reason. |
| Cancellation | Stop new work and report completed actions. |
| Persistence failure | Do not claim durable completion before commit. |
| Stale/corrupt data | Mark stale and recover from canonical source where safe. |
| Budget exceeded | Stop and return partial progress and risks. |


## B7. Security, privacy and safety

Use least privilege and scope identity by user, tenant, task, tool and resource.
Treat external documents, telemetry and tool outputs as untrusted data, not policy instructions.
Bind approvals to exact high-impact actions and enforce in executors.
Isolate untrusted execution and enforce resource limits outside the model.
Keep advisory AI output separate from authoritative state transitions; maintain threat/risk reviews.

## B8. Verification and validation


| Level | Coverage | Release rule |
| --- | --- | --- |
| Unit | Logic, schemas, transforms, state transitions | Required in CI |
| Contract | API/event/tool/model compatibility | Required for shared interfaces |
| Integration | Persistence, providers, sandbox and permissions | Required for changed components |
| End-to-end | Primary user journeys | Required for MVP |
| Security/adversarial | Injection, leakage, bypass, resource exhaustion | Required before pilot |
| Performance | Named hardware/workload and percentiles | Before quantitative claims |
| Domain validation | Independent domain review as applicable | Before high-impact deployment |


## B9. Acceptance criteria

Malformed/unauthenticated messages rejected
Partition tests produce documented safe behavior
Revoked node cannot issue accepted actions
Task history is reconstructable from audit
Field deployment gated by review

## B10. Risks and mitigations


| Risk | Mitigation |
| --- | --- |
| Spoofing/replay | Assign owner, control, regression test and release gate. |
| Partition conflicts | Assign owner, control, regression test and release gate. |
| Stale data treated as current | Assign owner, control, regression test and release gate. |
| Unsafe degraded behavior | Assign owner, control, regression test and release gate. |
| Location/privacy leakage | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record coverage, evidence, open defects, limitations, versions and approvals.

## B12. Open decisions

Confirm owner, users, deployment profile and release scope.
Confirm supported hardware, data sources and integrations.
Set performance thresholds after baseline profiling.
Confirm data classification, retention, regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Mesh
- **Practical Byzantine Fault Tolerant (PBFT) Consensus**: 3-phase consensus state machine (Pre-Prepare, Prepare, Commit) tolerating up to $f < n/3$ adversarial or disconnected nodes with Ed25519 cryptographic digest signatures.
- **Causal MIMO Sparse Swarm Routing**: Energy-aware and distance-weighted auction routing allocating logistics tasks across autonomous rovers and sensor nodes with Lipschitz continuity cushions.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **BFT Consensus Engine**: `platforms/mesh/src/consensus.py` (`BFTConsensusEngine`, `ConsensusProposal`, `ConsensusVote`)
- **Swarm Router**: `platforms/mesh/src/swarm_router.py` (`CortexSwarmRouter`, `SwarmTaskAllocation`)
- **Mesh Node**: `platforms/mesh/src/node.py` (`MeshNode`, `NodeStatus`)
- **Gateway Dispatcher**: `execute_mesh(action, params)` in `src/cortex_m/gateway.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Mesh
- **Practical Byzantine Fault Tolerant (PBFT) Consensus**: Complete 3-phase consensus state machine (Pre-Prepare, Prepare, Commit) with View-Change protocol and signature verification, tolerating up to $f < n/3$ adversarial or partitioned nodes with $2f+1$ quorum threshold verification.
- **Lamport/Mattern Vector Clock Causal Tracking**: Causal message delivery tracking across decentralized nodes:
$$VC_i[j] = \max(VC_i[j], VC_{\text{incoming}}[j])$$
guaranteeing causal consistency and preventing out-of-order execution across mesh partitions.
- **Causal MIMO Sparse Swarm Routing**: Energy-aware and distance-weighted auction routing allocating logistics tasks across autonomous rovers and sensor nodes with Lipschitz continuity cushions.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **BFT Consensus Engine**: `platforms/mesh/src/consensus.py` (`BFTConsensusEngine`, `ConsensusProposal`, `ConsensusVote`, `ConsensusPhase`)
- **Swarm Router**: `platforms/mesh/src/swarm_router.py` (`CortexSwarmRouter`, `SwarmTaskAllocation`)
- **Mesh Node**: `platforms/mesh/src/node.py` (`MeshNode`, `NodeStatus`)
- **Gateway Dispatcher**: `execute_mesh(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-Mesh
- **Practical Byzantine Fault Tolerant (PBFT) Consensus**: Complete 3-phase consensus state machine (Pre-Prepare, Prepare, Commit) with View-Change protocol and signature verification, tolerating up to $f < n/3$ adversarial or partitioned nodes with $2f+1$ quorum threshold verification.
- **Lamport/Mattern Vector Clock Causal Tracking**: Causal message delivery tracking across decentralized nodes:
  $$VC_i[j] = \max(VC_i[j], VC_{\text{incoming}}[j])$$
  guaranteeing causal consistency and preventing out-of-order execution across mesh partitions.
- **Causal MIMO Sparse Swarm Routing**: Energy-aware and distance-weighted auction routing allocating logistics tasks across autonomous rovers and sensor nodes with Lipschitz continuity cushions.

---

### Implemented Codebase Traceability
- **BFT Consensus Engine**: `platforms/mesh/src/consensus.py` (`BFTConsensusEngine`, `ConsensusProposal`, `ConsensusVote`, `ConsensusPhase`)
- **Swarm Router**: `platforms/mesh/src/swarm_router.py` (`CortexSwarmRouter`, `SwarmTaskAllocation`)
- **Mesh Node**: `platforms/mesh/src/node.py` (`MeshNode`, `NodeStatus`)
- **Gateway Dispatcher**: `execute_mesh(action, params)` in `src/cortex_m/gateway.py`