# CORTEX Ecosystem PRD/SRS Pack
**Master Document Baseline:** 26 October 2026

Ten unified system specifications defining the functional, architectural, mathematical, and verification requirements of the CORTEX-M multi-platform cognitive intelligence ecosystem.

## Document Inventory

| System | Document | Scope & Architectural Focus |
|---|---|---|
| **CORTEX-M Core Platform** | `00_CORTEX_M_Core_PRD_SRS.docx + .md` | Shared runtime, Hopfield attractor memory, orthogonal grafting, event gateway |
| **CORTEX-IDE / CortexCode** | `01_CORTEX_IDE_PRD_SRS.docx + .md` | AST call graphs, diff validation, 5.0s subprocess sandbox, multi-agent swarm |
| **CORTEX-Personal** | `02_CORTEX_Personal_PRD_SRS.docx + .md` | Tabula rasa Hebbian learning, tenant factor paging (2.0 KB), zero-egress interlock |
| **CORTEX-Fin** | `03_CORTEX_Fin_PRD_SRS.docx + .md` | Columnar streaming (>100k tx/s), Lyapunov energy risk gradient, shadow mode |
| **CORTEX-OS** | `04_CORTEX_OS_PRD_SRS.docx + .md` | Non-privileged userspace watcher, cognitive affinity clustering, spotlight omnibar |
| **CORTEX-Robotics** | `05_CORTEX_Robotics_PRD_SRS.docx + .md` | 6-DOF kinematics, quintic jerk-bounded planner, ISO 10218-1/2 safety interlock |
| **CORTEX-Mesh** | `06_CORTEX_Mesh_PRD_SRS.docx + .md` | Practical BFT 3-phase consensus, causal MIMO sparse routing, swarm logistics |
| **CORTEX-Health / BCI** | `07_CORTEX_Health_BCI_PRD_SRS.docx + .md` | Butterworth/notch conditioning, Morlet continuous wavelet decomposition, BCI state decode |
| **CORTEX-Space** | `08_CORTEX_Space_PRD_SRS.docx + .md` | Keplerian + J2 orbit propagation, CCSDS ASM/CRC16 sync, SEU attractor tracking |
| **CortexLM Model Family** | `09_CortexLM_PRD_SRS.docx + .md` | Causal transformer architecture, model gateway adapter, nocturnal sleep consolidation |

## Approval Workflow & Governance
1. **Product Owner Review**: Approves target personas, MVP boundaries, success measures, and non-goals.
2. **Engineering Review**: Audits mathematical formulations, interfaces, state machines, and data models.
3. **Security & Privacy Review**: Verifies zero-egress interlocks, tenant isolation, and cryptographic auditability.
4. **Domain Specialist Review**: Evaluates high-impact safety requirements (ISO 10218, NASA NPR 7150.2, FDA non-clinical boundaries, FCRA risk attribution).
5. **QA & Traceability Verification**: Bidirectionally maps all requirement IDs (`FR-*`, `NFR-*`) to automated unit/integration test suites.
6. **Release Baseline Sign-Off**: Formally approves versioned baselines with regression testing and rollback runbooks.

## Disclaimers & Safety Principles
- **Ground-Based Simulation**: Robotics, space, and medical platforms operate strictly in software digital twin simulators and non-clinical research environments.
- **Deterministic Separation**: Neural attractor inference acts in a supervisory and advisory capacity; physical emergency stops, bank ledgers, and orbit command verification remain deterministic.
