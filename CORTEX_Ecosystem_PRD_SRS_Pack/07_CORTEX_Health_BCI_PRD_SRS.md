# 07 CORTEX Health BCI: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CORTEX-Health / BCI Research
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Non-clinical neural-signal research, offline analysis and BCI experimentation

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

Support offline signal analysis with data protection, rigorous evaluation and a clear boundary from diagnosis, treatment or actuation.

## A2. Target users

Neuroscience researchers
Signal-processing engineers
Clinical/device research teams
Data governance reviewers

## A3. Problems and desired outcomes

Noisy participant-specific signals
Data leakage inflates results
Models drift across sessions/devices
Clinical use requires governance and validation

## A4. MVP scope

Approved offline datasets
Signal preprocessing
Baseline pipeline
Participant/session-aware splits
Artifact/drift analysis
Research-only reports
Access/deletion controls

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1/MVP | MVP capabilities and one complete primary workflow | Workflow works; correctness/security tests pass; limitations documented. |
| R2 | Usability, reporting, integrations and wider evaluations | Measured improvement against R1 without critical regressions. |
| R3/controlled pilot | Scale, external systems or hardware only where applicable | Domain/security review, runbook, rollback and approval. |
| Deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and evidence required. |


## A6. Primary user journeys

Approved dataset selected → access/purpose checks → versioned preprocessing → held-out evaluation → report with uncertainty/limitations.
Input quality insufficient → explicit insufficient-evidence result.
Clinical/device integration requested → blocked pending qualified governance, hazard controls and required approvals.

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

Offline research pipeline with controlled data access, versioned preprocessing, participant/session-aware evaluation and no clinical/stimulation/actuator pathway in the research release.

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
| FR-HLT-001 | P0 | Record data source, authorization/consent basis, purpose, protocol, device metadata and retention. | Unit + integration + end-to-end |
| FR-HLT-002 | P0 | Keep participant data out of source control and general logs; enforce role/purpose access. | Unit + integration + end-to-end |
| FR-HLT-003 | P1 | Validate sample rate, channels, units, timestamps, missing channels and file format. | Unit + integration + end-to-end |
| FR-HLT-004 | P1 | Version filtering, artifact handling, normalization and segmentation operations. | Unit + integration + end-to-end |
| FR-HLT-005 | P1 | Provide reproducible signal-processing/statistical baselines before advanced learned models. | Unit + integration + end-to-end |
| FR-HLT-006 | P0 | Use participant/session-aware held-out evaluation where applicable and document leakage controls. | Unit + integration + end-to-end |
| FR-HLT-007 | P1 | Test channel loss, artifacts, drift, noise, device changes and calibration. | Unit + integration + end-to-end |
| FR-HLT-008 | P0 | Label outputs research-only and display uncertainty/unsupported-use warnings. | Unit + integration + end-to-end |
| FR-HLT-009 | P0 | Return insufficient-evidence when input quality or documented confidence criteria fail. | Unit + integration + end-to-end |
| FR-HLT-010 | P0 | Apply retention/deletion to source data, derived features, indexes and caches. | Unit + integration + end-to-end |
| FR-HLT-011 | P1 | Record code, dataset version, split, configuration, seed, model and report. | Unit + integration + end-to-end |
| FR-HLT-012 | P0 | No implant, stimulation, diagnosis or actuator connection without qualified governance and required approvals. | Unit + integration + end-to-end |


## B3. Non-functional requirements


| ID | Quality | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-HLT-001 | Privacy | Sensitive data is minimized, access-controlled and excluded from ordinary telemetry. | Test/inspection/analysis |
| NFR-HLT-002 | Validity | Use appropriate held-out participants/sessions and report uncertainty/limitations. | Test/inspection/analysis |
| NFR-HLT-003 | Safety | No clinical diagnosis/treatment claim or direct stimulation/actuation in research release. | Test/inspection/analysis |
| NFR-HLT-004 | Traceability | Results trace to authorization, source, preprocessing and model/config. | Test/inspection/analysis |
| NFR-HLT-005 | Integrity | Dataset/preprocessing changes are versioned and detectable. | Test/inspection/analysis |
| NFR-HLT-006 | Regulatory readiness | Clinical use requires jurisdiction-specific review, quality process, qualified partners and independent validation. | Test/inspection/analysis |


## B4. External interfaces

Approved datasets
Signal acquisition formats
Preprocessing pipeline
Offline model gateway
Evaluation/reporting
Protected storage/access control

## B5. Data requirements

Raw/derived signals
Participant/session metadata
Consent/purpose metadata
Preprocessing configs
Predictions/uncertainty
Evaluation reports
Retention/deletion records
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

Dataset access and provenance recorded
Held-out split documented/reproducible
Outputs labeled research-only
Deletion/retention tested
Research build has no clinical/actuator pathway

## B10. Risks and mitigations


| Risk | Mitigation |
| --- | --- |
| Privacy breach | Assign owner, control, regression test and release gate. |
| Train/test leakage | Assign owner, control, regression test and release gate. |
| Clinical misuse | Assign owner, control, regression test and release gate. |
| Data drift | Assign owner, control, regression test and release gate. |
| Unapproved actuator connection | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record coverage, evidence, open defects, limitations, versions and approvals.

## B12. Open decisions

Confirm owner, users, deployment profile and release scope.
Confirm supported hardware, data sources and integrations.
Set performance thresholds after baseline profiling.
Confirm data classification, retention, regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Health / BCI
- **Deterministic Biosignal Conditioning**: 4th-order Butterworth bandpass (0.5–100.0 Hz) and 50/60 Hz notch filtering with FastICA ocular artifact suppression.
- **Morlet Continuous Wavelet Decomposition**: Time-frequency spectral analysis computing instantaneous power across Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–50 Hz) bands.
- **State-Space Attractor Decoding**: Continuous Hopfield cognitive state classification (Resting, Focused, Cognitive Workload, Drowsy) with strict non-clinical disclaimers.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Biosignal Filter**: `platforms/health/src/signal_filter.py` (`BiosignalFilter`, `FilteredSignal`)
- **Wavelet Decomposer**: `platforms/health/src/wavelet_analysis.py` (`WaveletFrequencyDecomposer`, `NeuralBandpower`)
- **BCI Decoder**: `platforms/health/src/bci_decoder.py` (`ContinuousBCIDecoder`, `BCIDecodingResult`)
- **Gateway Dispatcher**: `execute_health(action, params)` in `src/cortex_m/gateway.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CORTEX-Health / BCI
- **Deterministic Biosignal Conditioning**: DC offset detrending, IIR 50/60 Hz notch filtering, and artifact clipping ($\pm 500\ \mu\text{V}$).
- **Spectral Signal-to-Noise Ratio (SNR) & Lead Impedance**: Evaluates empirical SNR in dB and audits electrode contact integrity, flagging flatline disconnections (variance $< 10^{-4}$) and saturated railed leads.
- **Welch Power Spectral Density**: Smoothed spectral estimation via 50% overlapping Hanning-windowed periodograms:
$$P_{xx}(f) = \frac{1}{K} \sum_{k=1}^K I_k(f)$$
- **Exponential Moving Average (EMA) BCI Stabilization**: Temporal smoothing across continuous sliding prediction windows:
$$P_t = \alpha P_{\text{inst}} + (1 - \alpha) P_{t-1}$$
preventing classification jitter on discrete cognitive states (Rest, Motor Left/Right, Focus, Drowsy).

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Biosignal Filter**: `platforms/health/src/signal_filter.py` (`BiosignalFilter`, `FilteredSignal`)
- **Wavelet Decomposer**: `platforms/health/src/wavelet_analysis.py` (`WaveletFrequencyDecomposer`, `NeuralBandpower`, `FrequencyBand`)
- **BCI Decoder**: `platforms/health/src/bci_decoder.py` (`ContinuousBCIDecoder`, `BCIDecodingResult`, `CognitiveState`)
- **Gateway Dispatcher**: `execute_health(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CORTEX-Health / BCI
- **Deterministic Biosignal Conditioning**: DC offset detrending, IIR 50/60 Hz notch filtering, and artifact clipping ($\pm 500\ \mu\text{V}$).
- **Spectral Signal-to-Noise Ratio (SNR) & Lead Impedance**: Evaluates empirical SNR in dB and audits electrode contact integrity, flagging flatline disconnections (variance $< 10^{-4}$) and saturated railed leads.
- **Welch Power Spectral Density**: Smoothed spectral estimation via 50% overlapping Hanning-windowed periodograms:
  $$P_{xx}(f) = \frac{1}{K} \sum_{k=1}^K I_k(f)$$
- **Exponential Moving Average (EMA) BCI Stabilization**: Temporal smoothing across continuous sliding prediction windows:
  $$P_t = \alpha P_{\text{inst}} + (1 - \alpha) P_{t-1}$$
  preventing classification jitter on discrete cognitive states (Rest, Motor Left/Right, Focus, Drowsy).

---

### Implemented Codebase Traceability
- **Biosignal Filter**: `platforms/health/src/signal_filter.py` (`BiosignalFilter`, `FilteredSignal`)
- **Wavelet Decomposer**: `platforms/health/src/wavelet_analysis.py` (`WaveletFrequencyDecomposer`, `NeuralBandpower`, `FrequencyBand`)
- **BCI Decoder**: `platforms/health/src/bci_decoder.py` (`ContinuousBCIDecoder`, `BCIDecodingResult`, `CognitiveState`)
- **Gateway Dispatcher**: `execute_health(action, params)` in `src/cortex_m/gateway.py`