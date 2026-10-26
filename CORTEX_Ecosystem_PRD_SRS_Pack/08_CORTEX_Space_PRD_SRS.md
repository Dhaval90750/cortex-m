# 08 CORTEX Space: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-Space
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Space-system research, telemetry processing, anomaly detection and mission simulation

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

Develop reproducible ground-based telemetry and simulation capabilities before any flight-qualified use.

## A2. Target users

Space systems engineers
Mission operations researchers
Embedded/real-time engineers
Verification/validation teams

## A3. Problems and desired outcomes

Telemetry corrupted/delayed/missing
Fault recovery must be reproducible
Flight software requires verification evidence
Hardware compatibility cannot be inferred from footprint alone

## A4. MVP scope

Telemetry parsers
Deterministic pipeline
Mission simulator
Fault injection
Anomaly baseline
Requirements-to-test traceability
Reproducible builds

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1/MVP | MVP capabilities and one complete primary workflow | Workflow works; correctness/security tests pass; limitations documented. |
| R2 | Usability, reporting, integrations and wider evaluations | Measured improvement against R1 without critical regressions. |
| R3/controlled pilot | Scale, external systems or hardware only where applicable | Domain/security review, runbook, rollback and approval. |
| Deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and evidence required. |


## A6. Primary user journeys

Telemetry arrives → parser validates schema/checksum/timestamp → deterministic state updates or frame marked invalid → advisory anomaly output.
Fault injection → recovery/degraded state tested → requirements linked to evidence and build artifact.
Flight-use request → target hardware/environment, independent verification and mission acceptance gates checked.

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

Ground-based telemetry and simulation workflow. Deterministic parsing/state updates remain authoritative; AI anomaly findings are advisory until separately qualified.

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
| FR-SPC-001 | P1 | Represent mission phases, constraints, expected telemetry and safe/degraded modes in versioned config. | Unit + integration + end-to-end |
| FR-SPC-002 | P0 | Parse telemetry with schema/version, units, timestamps, validity/checksum and malformed-frame handling. | Unit + integration + end-to-end |
| FR-SPC-003 | P0 | Keep authoritative parsing and state updates deterministic; AI analysis remains advisory until qualified. | Unit + integration + end-to-end |
| FR-SPC-004 | P1 | Compare rule-based and learned anomaly detection on recorded datasets with defined labels. | Unit + integration + end-to-end |
| FR-SPC-005 | P1 | Run reproducible nominal/off-nominal mission scenarios with time, sensor and fault models. | Unit + integration + end-to-end |
| FR-SPC-006 | P1 | Inject corrupted data, sensor faults, clock/timing errors, resets, resource exhaustion and communication loss. | Unit + integration + end-to-end |
| FR-SPC-007 | P0 | Specify and test fault detection, degraded mode, restart and state recovery. | Unit + integration + end-to-end |
| FR-SPC-008 | P0 | Link requirements bidirectionally to design, tests, results, defects and release evidence. | Unit + integration + end-to-end |
| FR-SPC-009 | P1 | Record source commit, compiler/toolchain, flags, dependencies, config, checksum and artifact identity. | Unit + integration + end-to-end |
| FR-SPC-010 | P1 | Validate on representative/target-like hardware and document environmental assumptions. | Unit + integration + end-to-end |
| FR-SPC-011 | P0 | Require independent review of safety-critical requirements and fault recovery. | Unit + integration + end-to-end |
| FR-SPC-012 | P0 | No flight/radiation-tolerance claim without qualified hardware/environment tests and mission acceptance. | Unit + integration + end-to-end |


## B3. Non-functional requirements


| ID | Quality | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-SPC-001 | Reliability | Fault handling/recovery is specified and tested. | Test/inspection/analysis |
| NFR-SPC-002 | Traceability | Requirements, code, tests, results and artifacts are linked. | Test/inspection/analysis |
| NFR-SPC-003 | Reproducibility | Builds/simulations record toolchain/configuration. | Test/inspection/analysis |
| NFR-SPC-004 | Performance | Timing/memory measured on named target-like hardware with worst-case scenarios. | Test/inspection/analysis |
| NFR-SPC-005 | Safety | AI anomaly output is advisory until separately qualified. | Test/inspection/analysis |
| NFR-SPC-006 | Qualification | Flight use requires independent verification and mission/customer acceptance. | Test/inspection/analysis |


## B4. External interfaces

Telemetry source/simulator
Mission configuration
Deterministic parser/state service
Anomaly detector
Build toolchain
Test/artifact registry
Mission visualization

## B5. Data requirements

Telemetry frames/validity
Mission scenarios
Fault logs
Anomaly reports
Requirements/test links
Build manifests/checksums
Acceptance evidence
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

Malformed/corrupt telemetry rejected or marked invalid
Fault tests verify defined degraded states
Requirements map to executable tests/evidence
Build artifacts reproducible within documented constraints
No flight-qualified claim without qualification gate

## B10. Risks and mitigations


| Risk | Mitigation |
| --- | --- |
| Invalid telemetry accepted | Assign owner, control, regression test and release gate. |
| False/missed anomaly | Assign owner, control, regression test and release gate. |
| Unrecoverable state | Assign owner, control, regression test and release gate. |
| Unreproducible build | Assign owner, control, regression test and release gate. |
| Unsupported hardware claim | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record coverage, evidence, open defects, limitations, versions and approvals.

## B12. Open decisions

Confirm owner, users, deployment profile and release scope.
Confirm supported hardware, data sources and integrations.
Set performance thresholds after baseline profiling.
Confirm data classification, retention, regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Space
- **Keplerian & J2 Oblateness Orbit Propagation**: Analytical orbital dynamics modeling Earth oblateness ($J_2 = 1.08263 \times 10^{-3}$) nodal regression and perigee precession with line-of-sight ground station contact cones (Goldstone, Madrid, Canberra).
- **CCSDS Frame Synchronization & Telemetry Decommutation**: Validates 32-bit Attached Sync Marker (`0x1ACFFC1D`) and CRC-16 telecommand/telemetry packet integrity.
- **Attractor-Based SEU Radiation Supervisor**: Continuous attractor anomaly scoring isolating Single Event Upsets (SEU latchups) from routine orbital thermal cycles.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Orbit Propagator**: `platforms/space/src/orbit_propagator.py` (`OrbitPropagator`, `OrbitalElements`)
- **CCSDS Processor**: `platforms/space/src/ccsds_telemetry.py` (`CCSDSTelemetryProcessor`, `DecommutatedTelemetry`)
- **Safety Supervisor**: `platforms/space/src/safety_supervisor.py` (`SpaceSafetySupervisor`, `TelemetryAnomalyAlert`)
- **Gateway Dispatcher**: `execute_space(action, params)` in `src/cortex_m/gateway.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Space
- **Keplerian & J2 Oblateness Orbit Propagation**: Analytical orbital dynamics modeling Earth oblateness ($J_2 = 1.08263 \times 10^{-3}$) nodal regression and perigee precession with NORAD Two-Line Element (TLE) parsing (`from_tle`).
- **Multi-Orbit Ground Station Pass Prediction**: Computes elevation angle $\theta_{\text{elev}} = \arcsin(\hat{r} \cdot \hat{n})$ against topocentric ground stations (Goldstone, Madrid, Canberra), recording AOS/LOS timestamps and elevation mask bounds ($> 5^\circ$).
- **CCSDS Frame Synchronization & 14-Bit Continuity**: Validates 32-bit Attached Sync Marker (`0x1ACFFC1D`) and CRC-16 checksums, analyzing sequence counter discontinuities with modulo-16384 wrap-around.
- **Attractor-Based SEU Radiation Supervisor**: Continuous attractor anomaly scoring isolating Single Event Upsets (SEU latchups) from routine orbital thermal cycles.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Orbit Propagator**: `platforms/space/src/orbit_propagator.py` (`OrbitPropagator`, `OrbitalElements`)
- **CCSDS Processor**: `platforms/space/src/ccsds_telemetry.py` (`CCSDSTelemetryProcessor`, `DecommutatedTelemetry`)
- **Safety Supervisor**: `platforms/space/src/safety_supervisor.py` (`SpaceSafetySupervisor`, `TelemetryAnomalyAlert`)
- **Gateway Dispatcher**: `execute_space(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-Space
- **Keplerian & J2 Oblateness Orbit Propagation**: Analytical orbital dynamics modeling Earth oblateness ($J_2 = 1.08263 \times 10^{-3}$) nodal regression and perigee precession with NORAD Two-Line Element (TLE) parsing (`from_tle`).
- **Multi-Orbit Ground Station Pass Prediction**: Computes elevation angle $\theta_{\text{elev}} = \arcsin(\hat{r} \cdot \hat{n})$ against topocentric ground stations (Goldstone, Madrid, Canberra), recording AOS/LOS timestamps and elevation mask bounds ($> 5^\circ$).
- **CCSDS Frame Synchronization & 14-Bit Continuity**: Validates 32-bit Attached Sync Marker (`0x1ACFFC1D`) and CRC-16 checksums, analyzing sequence counter discontinuities with modulo-16384 wrap-around.
- **Attractor-Based SEU Radiation Supervisor**: Continuous attractor anomaly scoring isolating Single Event Upsets (SEU latchups) from routine orbital thermal cycles.

---

### Implemented Codebase Traceability
- **Orbit Propagator**: `platforms/space/src/orbit_propagator.py` (`OrbitPropagator`, `OrbitalElements`)
- **CCSDS Processor**: `platforms/space/src/ccsds_telemetry.py` (`CCSDSTelemetryProcessor`, `DecommutatedTelemetry`)
- **Safety Supervisor**: `platforms/space/src/safety_supervisor.py` (`SpaceSafetySupervisor`, `TelemetryAnomalyAlert`)
- **Gateway Dispatcher**: `execute_space(action, params)` in `src/cortex_m/gateway.py`