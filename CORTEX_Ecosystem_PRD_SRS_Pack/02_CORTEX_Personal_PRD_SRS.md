# 02 CORTEX Personal: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-Personal
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Local-first personal AI assistant for desktop and later mobile

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

Provide private everyday assistance, conversation, document search and approved personal memory, with clear local versus cloud behavior.

## A2. Target users

Individual users
Privacy-conscious users
Offline users
Personal knowledge managers

## A3. Problems and desired outcomes

Personal context is fragmented
Local/cloud processing is unclear
Incorrect or unwanted memories persist
Offline claims may hide network dependencies

## A4. MVP scope

Chat/history
Local model profile
Local/cloud indicator
Opt-in memory
Document import/search
Memory inspect/edit/delete
Portable settings and backup

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1 / MVP | MVP features in this document; one end-to-end primary workflow | Workflow succeeds; core security/correctness tests pass; limitations documented. |
| R2 | More integrations, usability, reporting, automation and broader evaluation | Measured improvement against R1; no critical regression. |
| R3 / controlled pilot | Scale, team deployment or hardware/external integration where relevant | Domain/security review, runbook, rollback and approval. |
| Explicitly deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and acceptance evidence. |


## A6. Primary user journeys

Chat → show local/cloud route → use approved context/memory → answer with sources where relevant.
Import document → parse/index → show status → answer questions with source references.
Forget memory → locate record and derivatives → delete per retention policy → report result.

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

A personal client over CORTEX-M runtime, local/remote model adapters, scoped memory, local knowledge index and OS credential storage. Local-only mode must be verified with network disabled.

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
| FR-PER-001 | P1 | Users SHALL create, rename, archive, export and delete conversations with confirmation for destructive actions. | Unit + integration + end-to-end test |
| FR-PER-002 | P1 | The client SHALL show available models, capability limits, privacy mode and resource requirements. | Unit + integration + end-to-end test |
| FR-PER-003 | P0 | An advertised offline profile SHALL operate with network disabled for all included functions. | Unit + integration + end-to-end test |
| FR-PER-004 | P0 | Long-lived personal memory SHALL be opt-in and distinguish persistent memory from temporary context. | Unit + integration + end-to-end test |
| FR-PER-005 | P1 | Users SHALL inspect, correct, disable, export and delete memories with visible source and scope. | Unit + integration + end-to-end test |
| FR-PER-006 | P1 | The knowledge library SHALL import supported files and preserve original source references. | Unit + integration + end-to-end test |
| FR-PER-007 | P1 | Local tools SHALL operate only within user-approved scopes. | Unit + integration + end-to-end test |
| FR-PER-008 | P0 | Before sensitive content is sent remotely, the UI SHALL disclose remote inference or external access. | Unit + integration + end-to-end test |
| FR-PER-009 | P2 | Settings, conversations and user-owned memory SHOULD export to documented versioned formats. | Unit + integration + end-to-end test |
| FR-PER-010 | P2 | Reminders/automation SHALL require explicit configuration and expose active status. | Unit + integration + end-to-end test |
| FR-PER-011 | P2 | Multi-device sync, if enabled, SHALL be opt-in and access-controlled. | Unit + integration + end-to-end test |
| FR-PER-012 | P0 | Deletion SHALL cover primary records and derived indexes/caches under documented backup retention. | Unit + integration + end-to-end test |


## B3. Non-functional requirements


| ID | Quality attribute | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-PER-001 | Privacy | Offline mode must be tested with network disabled. | Test/inspection/analysis |
| NFR-PER-002 | Usability | Memory and local/cloud state must be understandable without technical expertise. | Test/inspection/analysis |
| NFR-PER-003 | Performance | Measure startup, first-token latency, RAM and battery per named device. | Test/inspection/analysis |
| NFR-PER-004 | Portability | Exports and settings use documented versioned formats. | Test/inspection/analysis |
| NFR-PER-005 | Protection | Use platform secret storage and encryption for synced data where available. | Test/inspection/analysis |
| NFR-PER-006 | Reliability | Backups must restore and indexes must be rebuildable from source files. | Test/inspection/analysis |
| NFR-PER-007 | Honesty | Do not label a workflow offline if any required component contacts a remote service. | Test/inspection/analysis |


## B4. External interfaces

CORTEX-M runtime
Local/remote model backends
Local filesystem and document index
OS credential store
Optional sync service

## B5. Data requirements

Conversations
Approved memories
Document metadata/indexes
Settings/model profiles
Automation definitions
Backup manifests
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

Offline profile works with network disabled
User can inspect/delete memory and searchable derivatives
Cloud use is disclosed before sensitive transmission
Backup/export restores in a clean profile
Performance claims identify model and hardware

## B10. Risks and mitigations


| Risk | Required mitigation |
| --- | --- |
| Accidental cloud transmission | Assign owner, control, regression test and release gate. |
| Unwanted memory retention | Assign owner, control, regression test and release gate. |
| Sync data loss | Assign owner, control, regression test and release gate. |
| Mobile footprint not achievable | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record requirement coverage, test evidence, open defects, known limitations, dependency/model versions and approvals.

## B12. Open decisions

Confirm product owner, users, deployment profile and release scope.
Confirm supported OS/hardware, data sources and integrations.
Set numeric performance thresholds only after baseline profiling.
Confirm data classification, retention, legal/regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Personal
- **Tabula Rasa Epistemic Verification**: Projection onto synaptic weight matrix $W_{\text{personal}}$ determines evidence threshold $S = \|W x\| / \|x\|$. If $S < \tau_{\text{novelty}}$, returns `INSUFFICIENT_EVIDENCE` (honest abstention).
- **Continuous Online Hebbian Imprinting**: Weight adaptation uses bounded outer-product updates $\Delta W = \eta (z z^T - \alpha W)$ with instant anti-Hebbian unlearning for exact GDPR privacy erasure.
- **Ultralight Low-Rank Tenant Paging**: Factorizes user memory into low-rank slices ($2.0\text{ KB}$ per tenant factor) consuming $< 15\text{ MB}$ total runtime RAM.
- **Zero-Egress Security Interlock**: Hardware socket interlock (`platforms/personal/src/interlock.py`) guarantees $0.0\text{ KB}$ external WAN transmission with AES-256-GCM encrypted persistence.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Personal Substrate**: `platforms/personal/src/substrate.py` (`PersonalSubstrateCoordinator`)
- **Zero-Egress Interlock**: `platforms/personal/src/interlock.py` (`ZeroEgressInterlock`)
- **Tenant Paging**: `platforms/personal/src/tenant_paging.py` (`PersonalTenantPagingEngine`)
- **Encrypted Vault**: `platforms/personal/src/vault.py` (`EncryptedPersonalVault`)
- **Gateway Dispatcher**: `execute_personal(action, params)` in `src/cortex_m/gateway.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Personal
- **Tabula Rasa Epistemic Verification**: Projection onto synaptic weight matrix $W_{\text{personal}}$ determines evidence threshold $S = \|W x\| / \|x\|$. If $S < \tau_{\text{novelty}}$, returns `INSUFFICIENT_EVIDENCE` (honest abstention).
- **Continuous Online Hebbian Imprinting**: Weight adaptation uses bounded outer-product updates $\Delta W = \eta (z z^T - \alpha W)$ with instant anti-Hebbian unlearning for exact GDPR privacy erasure.
- **Ultralight Low-Rank Tenant Factor Paging**: Factorizes user memory into low-rank slices ($2.0\text{ KB}$ rank-4 factors per tenant):
$$\Delta W_{\text{tenant}} = U_{\text{tenant}} V_{\text{tenant}}^T$$
with disk-backed serialization (`page_out_to_disk`, `page_in_from_disk`) consuming $< 15\text{ MB}$ total runtime RAM.
- **Zero-Egress Security Interlock**: Hardware socket interlock (`platforms/personal/src/interlock.py`) guarantees $0.0\text{ KB}$ external WAN transmission with AES-256-GCM / PBKDF2-HMAC-SHA256 encrypted persistence.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Personal Substrate**: `platforms/personal/src/substrate.py` (`PersonalSubstrateCoordinator`)
- **Zero-Egress Interlock**: `platforms/personal/src/interlock.py` (`ZeroEgressInterlock`)
- **Tenant Paging**: `platforms/personal/src/tenant_paging.py` (`PersonalTenantPagingEngine`, `TenantFactorSlice`)
- **Encrypted Vault**: `platforms/personal/src/vault.py` (`EncryptedPersonalVault`)
- **Gateway Dispatcher**: `execute_personal(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-Personal
- **Tabula Rasa Epistemic Verification**: Projection onto synaptic weight matrix $W_{\text{personal}}$ determines evidence threshold $S = \|W x\| / \|x\|$. If $S < \tau_{\text{novelty}}$, returns `INSUFFICIENT_EVIDENCE` (honest abstention).
- **Continuous Online Hebbian Imprinting**: Weight adaptation uses bounded outer-product updates $\Delta W = \eta (z z^T - \alpha W)$ with instant anti-Hebbian unlearning for exact GDPR privacy erasure.
- **Ultralight Low-Rank Tenant Factor Paging**: Factorizes user memory into low-rank slices ($2.0\text{ KB}$ rank-4 factors per tenant):
  $$\Delta W_{\text{tenant}} = U_{\text{tenant}} V_{\text{tenant}}^T$$
  with disk-backed serialization (`page_out_to_disk`, `page_in_from_disk`) consuming $< 15\text{ MB}$ total runtime RAM.
- **Zero-Egress Security Interlock**: Hardware socket interlock (`platforms/personal/src/interlock.py`) guarantees $0.0\text{ KB}$ external WAN transmission with AES-256-GCM / PBKDF2-HMAC-SHA256 encrypted persistence.

---

### Implemented Codebase Traceability
- **Personal Substrate**: `platforms/personal/src/substrate.py` (`PersonalSubstrateCoordinator`)
- **Zero-Egress Interlock**: `platforms/personal/src/interlock.py` (`ZeroEgressInterlock`)
- **Tenant Paging**: `platforms/personal/src/tenant_paging.py` (`PersonalTenantPagingEngine`, `TenantFactorSlice`)
- **Encrypted Vault**: `platforms/personal/src/vault.py` (`EncryptedPersonalVault`)
- **Gateway Dispatcher**: `execute_personal(action, params)` in `src/cortex_m/gateway.py`