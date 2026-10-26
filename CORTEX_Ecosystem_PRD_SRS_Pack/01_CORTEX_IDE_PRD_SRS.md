# 01 CORTEX IDE: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-IDE / CortexCode
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Repository-aware coding agent and developer workspace

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

Help developers understand, modify, test, debug and review code while keeping them in control of changes and external actions.

## A2. Target users

Individual developers
Software teams
QA engineers
Technical leads/reviewers

## A3. Problems and desired outcomes

Large repositories are hard to navigate
Generated changes may be untested or too broad
Unsafe commands and unrelated file changes
False claims that tests passed

## A4. MVP scope

Workspace/Git discovery
Text and symbol search
Task plan and acceptance criteria
Patch edits and diff review
Sandboxed test/lint/build
Bounded repair loop
Verification report and task history

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1 / MVP | MVP features in this document; one end-to-end primary workflow | Workflow succeeds; core security/correctness tests pass; limitations documented. |
| R2 | More integrations, usability, reporting, automation and broader evaluation | Measured improvement against R1; no critical regression. |
| R3 / controlled pilot | Scale, team deployment or hardware/external integration where relevant | Domain/security review, runbook, rollback and approval. |
| Explicitly deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and acceptance evidence. |


## A6. Primary user journeys

Select repository → inspect Git state → index relevant code → plan task → show acceptance criteria → patch → sandbox tests → diff/evidence → accept or reject.
Test fails → analyze captured output → bounded repair → rerun → report actual passed/failed/skipped/not-run results.
Privileged/external action requested → policy gate → approval bound to exact action → execute or deny.

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

A product workflow over CORTEX-M runtime, model gateway, repository index, patch engine, isolated command executor and policy/approval services. Repository text and issue content are untrusted input.

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
| FR-IDE-001 | P1 | The system SHALL identify workspace root, Git branch, dirty state and detected languages without uploading source by default. | Unit + integration + end-to-end test |
| FR-IDE-002 | P1 | The indexer SHALL index configured files/symbols and exclude ignored paths, generated artifacts and configured secrets. | Unit + integration + end-to-end test |
| FR-IDE-003 | P1 | Search SHALL support filename, exact text, symbol and optional semantic queries with file/line references. | Unit + integration + end-to-end test |
| FR-IDE-004 | P1 | Each task plan SHALL list steps, assumptions, likely files and observable acceptance criteria editable by the user. | Unit + integration + end-to-end test |
| FR-IDE-005 | P1 | Context assembly SHALL respect a configurable budget and preserve source path/line metadata. | Unit + integration + end-to-end test |
| FR-IDE-006 | P1 | Edits SHALL be applied as reviewable patches and detect concurrent workspace modifications. | Unit + integration + end-to-end test |
| FR-IDE-007 | P1 | The UI SHALL show a diff and flag changes outside the planned scope before acceptance. | Unit + integration + end-to-end test |
| FR-IDE-008 | P0 | Commands SHALL run in an isolated workspace with CPU, memory, process, time and network limits. | Unit + integration + end-to-end test |
| FR-IDE-009 | P1 | Test/lint/build results SHALL record command, exit code, duration and captured output/artifact references. | Unit + integration + end-to-end test |
| FR-IDE-010 | P1 | Repair loops SHALL have attempt/budget limits and stop after repeated failure or unsafe state. | Unit + integration + end-to-end test |
| FR-IDE-011 | P0 | Commit, push, deployment, deletion and remote changes SHALL require configured authorization/approval. | Unit + integration + end-to-end test |
| FR-IDE-012 | P0 | Completion reports SHALL derive test status from actual execution evidence, not model-generated claims. | Unit + integration + end-to-end test |
| FR-IDE-013 | P2 | Task history MAY resume after revalidating repository state and current diff. | Unit + integration + end-to-end test |
| FR-IDE-014 | P2 | Editor, issue tracker and CI integrations SHALL use scoped credentials and explicit permissions. | Unit + integration + end-to-end test |
| FR-IDE-015 | P0 | Repository content and issue text SHALL not be able to change tool permissions or override policy. | Unit + integration + end-to-end test |


## B3. Non-functional requirements


| ID | Quality attribute | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-IDE-001 | Workspace safety | Writes outside the approved workspace are denied; symlink/path traversal is tested. | Test/inspection/analysis |
| NFR-IDE-002 | Reproducibility | Record repository revision, model/config, tools and verification results. | Test/inspection/analysis |
| NFR-IDE-003 | Usability | Users can inspect changes before accepting them and see why tasks stopped. | Test/inspection/analysis |
| NFR-IDE-004 | Performance | Measure index/search on representative repositories; incremental updates where possible. | Test/inspection/analysis |
| NFR-IDE-005 | Reliability | Crash/cancel must not produce false completion. | Test/inspection/analysis |
| NFR-IDE-006 | Security | No host credentials in untrusted execution; network disabled by default. | Test/inspection/analysis |
| NFR-IDE-007 | Compatibility | Document supported language/build tools and repository profiles. | Test/inspection/analysis |


## B4. External interfaces

CORTEX-M runtime/model gateway
Filesystem and code index
Git
Sandbox executor
Test/lint/build tools
Optional editor/issue/CI integrations

## B5. Data requirements

Workspace metadata
Index and source locations
Plans and acceptance criteria
Patches/diffs
Commands/test results
Approvals and traces
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

Benchmark task produces a minimal patch and reviewable diff
Reported test status matches captured command output
Denied shell/network/file actions are blocked by executor
No unexplained files change in benchmark tasks
Cancellation/restart does not falsely complete or duplicate actions

## B10. Risks and mitigations


| Risk | Required mitigation |
| --- | --- |
| Destructive edits | Assign owner, control, regression test and release gate. |
| Secret exposure | Assign owner, control, regression test and release gate. |
| Prompt injection | Assign owner, control, regression test and release gate. |
| False test claims | Assign owner, control, regression test and release gate. |
| Stale index | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record requirement coverage, test evidence, open defects, known limitations, dependency/model versions and approvals.

## B12. Open decisions

Confirm product owner, users, deployment profile and release scope.
Confirm supported OS/hardware, data sources and integrations.
Set numeric performance thresholds only after baseline profiling.
Confirm data classification, retention, legal/regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-IDE / CortexCode
- **AST Symbol Resolution**: Abstract Syntax Tree parsing (`platforms/ide/src/ast_parser.py`) constructs complete symbol call graphs and cross-file reference DAGs without token truncation.
- **Syntactic Diff Integrity**: AST Diff Engine (`platforms/ide/src/diff_engine.py`) enforces syntax-preserving AST chunk replacements with comment preservation and regression prevention.
- **Isolated Subprocess Sandbox**: Deterministic subprocess execution (`platforms/ide/src/sandbox.py`) enforcing hard timeout ceilings ($5.0\text{ s}$ default), stripped environment variables, and strict memory limits.
- **Multi-Agent Swarm Protocol**: 5W1H DAG execution across Architect, Coder, Critic, and Auditor personas with cryptographic approval gates prior to git mutation.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **AST Parser**: `platforms/ide/src/ast_parser.py` (`ASTSymbolTableParser`)
- **Diff Engine**: `platforms/ide/src/diff_engine.py` (`ASTDiffEngine`)
- **Sandbox Runner**: `platforms/ide/src/sandbox.py` (`SubprocessSandboxRunner`)
- **Swarm Orchestrator**: `platforms/ide/src/swarm.py` (`CortexCodeSwarm`)
- **Gateway Dispatcher**: `execute_ide(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-IDE / CortexCode
- **AST Symbol Resolution**: Abstract Syntax Tree parsing (`platforms/ide/src/ast_parser.py`) constructs complete symbol call graphs and cross-file reference DAGs without token truncation.
- **Syntactic Diff Integrity**: AST Diff Engine (`platforms/ide/src/diff_engine.py`) enforces syntax-preserving AST chunk replacements with comment preservation and regression prevention.
- **Isolated Subprocess Sandbox**: Deterministic subprocess execution (`platforms/ide/src/sandbox.py`) enforcing hard timeout ceilings ($5.0\text{ s}$ default), stripped environment variables, and strict memory limits.
- **Multi-Agent Swarm Protocol**: 5W1H DAG execution across Architect, Coder, Critic, and Auditor personas with cryptographic approval gates prior to git mutation.

---

### Implemented Codebase Traceability
- **AST Parser**: `platforms/ide/src/ast_parser.py` (`ASTSymbolTableParser`)
- **Diff Engine**: `platforms/ide/src/diff_engine.py` (`ASTDiffEngine`)
- **Sandbox Runner**: `platforms/ide/src/sandbox.py` (`SubprocessSandboxRunner`)
- **Swarm Orchestrator**: `platforms/ide/src/swarm.py` (`CortexCodeSwarm`)
- **Gateway Dispatcher**: `execute_ide(action, params)` in `src/cortex_m/gateway.py`