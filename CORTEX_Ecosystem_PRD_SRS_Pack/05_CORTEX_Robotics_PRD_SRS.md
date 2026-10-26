# 05 CORTEX Robotics: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-Robotics
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Robotics research, sensor processing and supervised control

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

Develop and evaluate robotics perception and supervisory behaviors in simulation first, with deterministic control and independent safety interlocks.

## A2. Target users

Robotics researchers
Controls engineers
Simulation engineers
Safety/test engineers

## A3. Problems and desired outcomes

Noisy/time-dependent sensor data
Learned policies fail outside training scenarios
Average latency hides deadline misses
AI must not be the only safety mechanism

## A4. MVP scope

Simulation harness
Recorded/synthetic sensors
State-estimation baseline
AI command proposals
Deterministic safety supervisor
Fault injection

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1/MVP | MVP capabilities and one complete primary workflow | Workflow works; correctness/security tests pass; limitations documented. |
| R2 | Usability, reporting, integrations and wider evaluations | Measured improvement against R1 without critical regressions. |
| R3/controlled pilot | Scale, external systems or hardware only where applicable | Domain/security review, runbook, rollback and approval. |
| Deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and evidence required. |


## A6. Primary user journeys

Simulation emits timestamped sensors → validate/estimate state → AI proposes action → supervisor checks → simulator executes or rejects → trace outcome.
Sensor/model fails → controller enters defined safe/degraded state → operator sees fault and can stop/override.
Hardware trial request → safety, hardware-in-loop and domain review gates checked before deployment.

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

Simulation-first workflow: sensor validation and state estimation → AI proposes high-level action → deterministic supervisor checks constraints → simulator executes/rejects. Hardware control is separately gated.

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
| FR-ROB-001 | P1 | Define versioned simulation scenarios, sensor models, initial state, constraints and expected safety outcomes. | Unit + integration + end-to-end |
| FR-ROB-002 | P1 | Validate sensor timestamps, ordering, units, ranges and missing-data behavior. | Unit + integration + end-to-end |
| FR-ROB-003 | P1 | Provide baseline state estimation and quality indicators; distinguish estimates from raw observations. | Unit + integration + end-to-end |
| FR-ROB-004 | P0 | AI may propose high-level goals/actions but cannot bypass deterministic controller constraints. | Unit + integration + end-to-end |
| FR-ROB-005 | P0 | A deterministic safety supervisor SHALL validate workspace, speed, collision and operational limits before dispatch. | Unit + integration + end-to-end |
| FR-ROB-006 | P0 | Emergency stop, stability and protective functions SHALL be independent of LLM and learned memory. | Unit + integration + end-to-end |
| FR-ROB-007 | P1 | Fault injection SHALL cover sensor loss, delay, actuator saturation, drift, clock error and resource exhaustion. | Unit + integration + end-to-end |
| FR-ROB-008 | P1 | Timing reports SHALL include worst-case/percentile latency, jitter, deadline misses and named hardware. | Unit + integration + end-to-end |
| FR-ROB-009 | P2 | Simulation and hardware drivers SHALL be separated behind explicit interfaces. | Unit + integration + end-to-end |
| FR-ROB-010 | P1 | Scenario regression SHALL run after model, controller or configuration changes. | Unit + integration + end-to-end |
| FR-ROB-011 | P1 | Operator interface SHALL expose mode, health, fault, manual override and safe-stop status. | Unit + integration + end-to-end |
| FR-ROB-012 | P0 | Maintain hazard analysis, requirement-to-test links, review records and release constraints. | Unit + integration + end-to-end |


## B3. Non-functional requirements


| ID | Quality | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-ROB-001 | Safety | AI is never the sole emergency-stop, collision-avoidance or stability mechanism. | Test/inspection/analysis |
| NFR-ROB-002 | Timing | Use worst-case timing and deadline analysis, not average latency. | Test/inspection/analysis |
| NFR-ROB-003 | Determinism | Safety supervisor behavior must be deterministic and independently testable. | Test/inspection/analysis |
| NFR-ROB-004 | Traceability | Every proposal links to sensor snapshot, model/config and supervisor decision. | Test/inspection/analysis |
| NFR-ROB-005 | Fault tolerance | Model/network/sensor loss leads to a defined safe degraded state. | Test/inspection/analysis |
| NFR-ROB-006 | Qualification | Hardware requires safety review and hardware-in-loop testing. | Test/inspection/analysis |


## B4. External interfaces

Simulator
Sensor drivers
State estimator
CORTEX-M runtime
Deterministic controller
Safety supervisor
Operator UI/telemetry

## B5. Data requirements

Sensor samples
Estimated states
Command proposals
Supervisor decisions
Robot telemetry
Scenario/test results
Hazard and verification records
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

AI proposals cannot bypass supervisor
Fault injection reaches defined safe states
Worst-case timing is reported on named target
Regression suite is reproducible
No hardware trial without documented approval

## B10. Risks and mitigations


| Risk | Mitigation |
| --- | --- |
| Unsafe command | Assign owner, control, regression test and release gate. |
| Sensor drift/loss | Assign owner, control, regression test and release gate. |
| Timing overrun | Assign owner, control, regression test and release gate. |
| Simulation-to-reality gap | Assign owner, control, regression test and release gate. |
| Operator misunderstanding | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record coverage, evidence, open defects, limitations, versions and approvals.

## B12. Open decisions

Confirm owner, users, deployment profile and release scope.
Confirm supported hardware, data sources and integrations.
Set performance thresholds after baseline profiling.
Confirm data classification, retention, regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Robotics
- **6-DOF Forward/Inverse Kinematics**: Analytical and numerical DH kinematic solver with singularity-robust Jacobian pseudo-inverse:
$$J^+ = J^T (J J^T + \lambda^2 I)^{-1}$$
- **Continuous Attractor Trajectory Generation**: Quintic polynomial spline interpolation minimizing jerk ($C^2$ continuity) bounded by energy basins.
- **Deterministic ISO 10218-1/2 Safety Interlock**: Speed and Separation Monitoring (SSM) dynamic envelope:
$$v_{\text{allowable}} = \sqrt{2 a_{\text{max}} (d_{\text{human}} - d_{\text{stop}})}$$
with hard real-time emergency stop ($< 100\text{ ms}$).

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Manipulator Kinematics**: `platforms/robotics/src/kinematics.py` (`Manipulator6DOF`, `RobotPose`)
- **Path Planner**: `platforms/robotics/src/path_planner.py` (`AttractorPathPlanner`)
- **Safety Interlock**: `platforms/robotics/src/safety_interlock.py` (`DeterministicSafetyInterlock`, `SafetyStatusReport`)
- **Gateway Dispatcher**: `execute_robotics(action, params)` in `src/cortex_m/gateway.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Robotics
- **6-DOF Denavit-Hartenberg (DH) Kinematics**: Computes exact homogeneous $4 \times 4$ transformation matrices $T_i^{i-1}$ from base to end-effector across standard industrial articulating geometries.
- **Damped Least Squares (Levenberg-Marquardt) Inverse Kinematics**: Singularity-robust numerical IK:
$$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} (x_{\text{target}} - x_{\text{current}})$$
with step clamping and mechanical joint limit bounding.
- **Yoshikawa Manipulability Index**: Evaluates kinematic dexterity and distance from singularities:
$$w = \sqrt{\det(J J^T)}$$
- **Deterministic ISO 10218-1/2 Safety Interlock**: Speed and Separation Monitoring (SSM) dynamic envelope:
$$d_{\text{stop}} = v \cdot t_{\text{react}} + \frac{v^2}{2 a_{\text{max}}}$$
$$v_{\text{allowable}} = \sqrt{2 a_{\text{max}} (d_{\text{human}} - d_{\text{stop}})}$$
with 3D workspace bounding box barriers (floor penetration prevention) and hard real-time emergency stop ($< 100\text{ ms}$).

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Manipulator Kinematics**: `platforms/robotics/src/kinematics.py` (`Manipulator6DOF`, `RobotPose`)
- **Path Planner**: `platforms/robotics/src/path_planner.py` (`AttractorPathPlanner`)
- **Safety Interlock**: `platforms/robotics/src/safety_interlock.py` (`DeterministicSafetyInterlock`, `SafetyStatusReport`)
- **Gateway Dispatcher**: `execute_robotics(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-Robotics
- **6-DOF Denavit-Hartenberg (DH) Kinematics**: Computes exact homogeneous $4 \times 4$ transformation matrices $T_i^{i-1}$ from base to end-effector across standard industrial articulating geometries.
- **Damped Least Squares (Levenberg-Marquardt) Inverse Kinematics**: Singularity-robust numerical IK:
  $$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} (x_{\text{target}} - x_{\text{current}})$$
  with step clamping and mechanical joint limit bounding.
- **Yoshikawa Manipulability Index**: Evaluates kinematic dexterity and distance from singularities:
  $$w = \sqrt{\det(J J^T)}$$
- **Deterministic ISO 10218-1/2 Safety Interlock**: Speed and Separation Monitoring (SSM) dynamic envelope:
  $$d_{\text{stop}} = v \cdot t_{\text{react}} + \frac{v^2}{2 a_{\text{max}}}$$
  $$v_{\text{allowable}} = \sqrt{2 a_{\text{max}} (d_{\text{human}} - d_{\text{stop}})}$$
  with 3D workspace bounding box barriers (floor penetration prevention) and hard real-time emergency stop ($< 100\text{ ms}$).

---

### Implemented Codebase Traceability
- **Manipulator Kinematics**: `platforms/robotics/src/kinematics.py` (`Manipulator6DOF`, `RobotPose`)
- **Path Planner**: `platforms/robotics/src/path_planner.py` (`AttractorPathPlanner`)
- **Safety Interlock**: `platforms/robotics/src/safety_interlock.py` (`DeterministicSafetyInterlock`, `SafetyStatusReport`)
- **Gateway Dispatcher**: `execute_robotics(action, params)` in `src/cortex_m/gateway.py`