# 09 CortexLM: Master PRD & SRS Specification
**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline

CortexLM Model Family
PRODUCT REQUIREMENTS DOCUMENT (PRD) + SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
Native language-model research, fine-tuning, evaluation, packaging and serving

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

Develop a measurable model family while allowing CORTEX-M to remain useful with external models during research.

## A2. Target users

ML researchers
Data engineers
Model trainers
Inference/serving engineers
Product teams

## A3. Problems and desired outcomes

Training claims need reproducibility
Quality can regress across tasks
Data rights and contamination need control
Native research model may not match pretrained baselines

## A4. MVP scope

Dataset/model registry
Small training baseline
Fine-tuning path
Evaluation harness
Checkpoint packaging
Model gateway adapter

## A5. Release scope and non-goals


| Release | Scope | Exit gate |
| --- | --- | --- |
| R1/MVP | MVP capabilities and one complete primary workflow | Workflow works; correctness/security tests pass; limitations documented. |
| R2 | Usability, reporting, integrations and wider evaluations | Measured improvement against R1 without critical regressions. |
| R3/controlled pilot | Scale, external systems or hardware only where applicable | Domain/security review, runbook, rollback and approval. |
| Deferred | Unvalidated high-impact automation and unsupported performance claims | Separate requirements, risk review and evidence required. |


## A6. Primary user journeys

Register dataset/rights → version config → bounded training → checkpoint/metrics → held-out evaluation → promote or reject.
Register model → validate model card/capabilities → adapter smoke test → staging evaluation → approved promotion.
Compare baseline, retrieval, Synapse, Router, Shield and combined setup on identical tasks and report gains/cost/regressions.

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

Parallel tracks: a small train-from-scratch baseline to validate the pipeline, and fine-tuning of a suitable licensed pretrained model for useful capability. Native CORTEX research is compared with baselines through ablation tests.

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
| FR-LM-001 | P1 | Each model SHALL have a card for architecture, tokenizer, intended use, limits, license, training-data summary, evaluation and serving requirements. | Unit + integration + end-to-end |
| FR-LM-002 | P0 | Datasets SHALL be versioned with provenance, rights, filtering, deduplication, splits and quality checks. | Unit + integration + end-to-end |
| FR-LM-003 | P1 | Training config SHALL record hyperparameters, code, dependencies, seed, hardware, precision and distributed settings. | Unit + integration + end-to-end |
| FR-LM-004 | P1 | Training jobs SHALL support bounded execution, checkpoints, resume, resource limits and failure reports. | Unit + integration + end-to-end |
| FR-LM-005 | P1 | A small autoregressive baseline SHALL validate tokenization, loss, checkpointing and generation end-to-end. | Unit + integration + end-to-end |
| FR-LM-006 | P1 | Fine-tuning SHALL support an approved supervised workflow and parameter-efficient methods where appropriate. | Unit + integration + end-to-end |
| FR-LM-007 | P1 | Tool-use datasets/evaluations SHALL cover valid calls, result interpretation, correction and stop behavior. | Unit + integration + end-to-end |
| FR-LM-008 | P0 | Evaluation SHALL use held-out language, instruction, coding, tool, safety and task-specific tests against named baselines. | Unit + integration + end-to-end |
| FR-LM-009 | P1 | Ablations SHALL compare base, retrieval, Synapse, routing, abstention and combined configurations on the same tasks. | Unit + integration + end-to-end |
| FR-LM-010 | P1 | Registry SHALL version weights, adapters, tokenizer, config, report, checksums and deployment status. | Unit + integration + end-to-end |
| FR-LM-011 | P1 | Serving adapter SHALL declare capabilities and normalized errors through the CORTEX model gateway. | Unit + integration + end-to-end |
| FR-LM-012 | P0 | Production promotion SHALL require evaluation and safety gates; previous versions remain available for rollback. | Unit + integration + end-to-end |
| FR-LM-013 | P0 | Data lacking appropriate rights SHALL be excluded; sensitive data and synthetic data quality require governance. | Unit + integration + end-to-end |
| FR-LM-014 | P1 | Profile tokens/sec, memory, convergence and validation gain on small runs before scaling compute. | Unit + integration + end-to-end |
| FR-LM-015 | P1 | Serving rollback SHALL restore a known-good model and compatible tokenizer/config. | Unit + integration + end-to-end |


## B3. Non-functional requirements


| ID | Quality | Requirement | Verification |
| --- | --- | --- | --- |
| NFR-LM-001 | Reproducibility | Every run records code, dataset, config, dependencies, hardware, seed and checkpoint lineage. | Test/inspection/analysis |
| NFR-LM-002 | Governance | Dataset rights/provenance are mandatory release metadata. | Test/inspection/analysis |
| NFR-LM-003 | Evaluation | Train/evaluation sets are separated and contamination checks documented. | Test/inspection/analysis |
| NFR-LM-004 | Reliability | Training supports checkpoint recovery and artifact-integrity checks. | Test/inspection/analysis |
| NFR-LM-005 | Serving | Capabilities and limits are exposed to the router and callers. | Test/inspection/analysis |
| NFR-LM-006 | Safety | Model cannot bypass tool authorization or runtime policy. | Test/inspection/analysis |
| NFR-LM-007 | Resources | Training/inference jobs have budgets and cancellation behavior. | Test/inspection/analysis |


## B4. External interfaces

CORTEX-M model gateway
Dataset/artifact store
Training runner
Experiment tracker
Evaluation harness
Inference engine
Model registry

## B5. Data requirements

Dataset manifests
Tokenizer/model config
Training metrics/logs
Checkpoints/adapters
Evaluation results
Model cards
Serving manifests
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

Small run reproducible from recorded config
Fine-tuned model compared to base on held-out tasks
Checkpoint/tokenizer/adapter load from clean environment
Release includes model card and limitations
Tool calls remain subject to runtime policy

## B10. Risks and mitigations


| Risk | Mitigation |
| --- | --- |
| Data rights/contamination | Assign owner, control, regression test and release gate. |
| Overfitting/regression | Assign owner, control, regression test and release gate. |
| Non-reproducible training | Assign owner, control, regression test and release gate. |
| Compute limits | Assign owner, control, regression test and release gate. |
| Misleading comparisons | Assign owner, control, regression test and release gate. |
| Unsafe tool outputs | Assign owner, control, regression test and release gate. |


## B11. Traceability and release control

Every PRD goal SHALL map to SRS IDs; each requirement SHALL map to a test or justified inspection/analysis. Each release candidate SHALL record coverage, evidence, open defects, limitations, versions and approvals.

## B12. Open decisions

Confirm owner, users, deployment profile and release scope.
Confirm supported hardware, data sources and integrations.
Set performance thresholds after baseline profiling.
Confirm data classification, retention, regulatory requirements and reviewers.

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CortexLM Model Family
- **Native Autoregressive Language Architecture**: Compact causal transformer with rotary position embeddings (RoPE), SwiGLU activation, and RMSNorm.
- **External Gateway Interoperability**: Seamless model gateway abstraction allowing execution of native CortexLM weights alongside external research providers.
- **Developmental Sleep Consolidation**: Nocturnal consolidation cycles applying Hebbian replay, weight decay, and synaptic pruning ($\|W_{ij}\| < \epsilon$) to stabilize long-term memory without catastrophic forgetting.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Model Architecture**: `src/cortex_m/model/cortex_lm.py` (`CortexLM`, `CortexLMOutput`)
- **Trainer**: `src/cortex_m/model/real_data_trainer.py`, `src/cortex_m/database/db_trainer.py`
- **Metacognition & Introspection**: `src/cortex_m/model/metacognition.py`
- **Knowledge Base**: `src/cortex_m/model/knowledge_base.py`
- **Gateway Dispatcher**: `execute_learning(action, params)` in `src/cortex_m/gateway.py`

# Technical Architecture & Mathematical Formulations

### Technical Architecture & Mathematical Foundations: CortexLM Model Family
- **Native Autoregressive Language Architecture**: Compact causal transformer with rotary position embeddings (RoPE), SwiGLU activation, and RMSNorm.
- **External Gateway Interoperability**: Seamless model gateway abstraction allowing execution of native CortexLM weights alongside external research providers.
- **Developmental Sleep Consolidation**: Nocturnal consolidation cycles applying Hebbian replay, weight decay, and synaptic pruning ($\|W_{ij}\| < \epsilon$) to stabilize long-term memory without catastrophic forgetting.

# Implemented Codebase Traceability

### Implemented Codebase Traceability
- **Model Architecture**: `src/cortex_m/model/cortex_lm.py` (`CortexLM`, `CortexLMOutput`)
- **Trainer**: `src/cortex_m/model/real_data_trainer.py`, `src/cortex_m/database/db_trainer.py`
- **Metacognition & Introspection**: `src/cortex_m/model/metacognition.py`
- **Knowledge Base**: `src/cortex_m/model/knowledge_base.py`
- **Gateway Dispatcher**: `execute_learning(action, params)` in `src/cortex_m/gateway.py`

---

### Technical Architecture & Mathematical Foundations: CortexLM Model Family
- **Native Autoregressive Language Architecture**: Compact causal transformer with rotary position embeddings (RoPE), SwiGLU activation, and RMSNorm.
- **External Gateway Interoperability**: Seamless model gateway abstraction allowing execution of native CortexLM weights alongside external research providers.
- **Developmental Sleep Consolidation**: Nocturnal consolidation cycles applying Hebbian replay, weight decay, and synaptic pruning ($\|W_{ij}\| < \epsilon$) to stabilize long-term memory without catastrophic forgetting.

---

### Implemented Codebase Traceability
- **Model Architecture**: `src/cortex_m/model/cortex_lm.py` (`CortexLM`, `CortexLMOutput`)
- **Trainer**: `src/cortex_m/model/real_data_trainer.py`, `src/cortex_m/database/db_trainer.py`
- **Metacognition & Introspection**: `src/cortex_m/model/metacognition.py`
- **Knowledge Base**: `src/cortex_m/model/knowledge_base.py`
- **Gateway Dispatcher**: `execute_learning(action, params)` in `src/cortex_m/gateway.py`