# CORTEX-M Industrial Platform Workstreams
### Architectural Specifications & Research Roadmap Governed by the Master Ecosystem Implementation Plan

> [!NOTE]
> **Research Roadmap & Specification Scope**: The workstreams documented in this directory represent our architectural design targets, simulation requirements, and domain safety boundaries. They are planned extensions that build upon the core CORTEX-M neural substrate and are **not** claimed as completed physical deployments.

This directory houses the architectural specifications, safety boundaries, and simulation protocols for the 8 specialized CORTEX-M product lines as governed by the [Master Ecosystem Implementation Plan](../docs/master_plan/cortex_m_original_ecosystem_implementation_plan.md).

---

## The 8 Planned Workstreams

| Platform | Workspace | Master PRD & SRS Specification | Target Domain & Architectural Scope | Qualification & Safety Standard |
|---|---|---|---|---|
| **CORTEX-IDE** | [`platforms/ide/`](ide/) | [Detailed PRD & SRS](ide/PRD_SRS.md) | Autonomous repository engineering & patch validation | Sandboxed Execution & Human Approval Gate |
| **CORTEX-Personal** | [`platforms/personal/`](personal/) | [Detailed PRD & SRS](personal/PRD_SRS.md) | Edge & desktop personal cognitive assistant | Private Tenant Isolation & Zero Cloud Egress |
| **CORTEX-Fin** | [`platforms/fin/`](fin/) | [Detailed PRD & SRS](fin/PRD_SRS.md) | High-throughput transaction replay & fraud research | Shadow-Mode Isolation & Deterministic Ledgers |
| **CORTEX-OS** | [`platforms/os/`](os/) | [Detailed PRD & SRS](os/PRD_SRS.md) | Userspace process scheduling & associative search | Userspace Sandbox & Semantic Affinity Clusters |
| **CORTEX-Robotics** | [`platforms/robotics/`](robotics/) | [Detailed PRD & SRS](robotics/PRD_SRS.md) | Digital twin kinematic simulation & safety envelopes | ISO 10218-1/2:2025 Robotic Safety Standards |
| **CORTEX-Mesh** | [`platforms/mesh/`](mesh/) | [Detailed PRD & SRS](mesh/PRD_SRS.md) | Cooperative logistics swarm & decentralized routing | Civilian Transparent Protocols & BFT Consensus |
| **CORTEX-Health** | [`platforms/health/`](health/) | [Detailed PRD & SRS](health/PRD_SRS.md) | Non-clinical neural signal processing & BCI research | FDA Non-Clinical Guidance & IEC 62304 |
| **CORTEX-Space** | [`platforms/space/`](space/) | [Detailed PRD & SRS](space/PRD_SRS.md) | Orbital satellite telemetry parsing & anomaly tracking | NASA Flight Software Qualification (NPR 7150.2) |

---

## Architectural Principles
1. **Deterministic Safety Separation**: Neural associative inference supervises and proposes; hardware safety interlocks, financial ledgers, and medical state transitions remain deterministic.
2. **Simulation Before Hardware**: Robotics, Mesh, Health, and Space operate in high-fidelity software simulators before any physical hardware validation.
3. **Common Core Contracts**: All platforms communicate with the underlying runtime via [`cortex_m.core.contracts`](../src/cortex_m/core/contracts/).
4. **Purpose-Driven Visual Identity**: Each workstream maintains a tailored chromatic identity governed by the [Ecosystem Design System](../docs/architecture/design_system.md), completely avoiding generic blue-and-cyan defaults (e.g., Fern for IDE, Olive for Personal, Emerald for Finance, Safety Orange for Robotics, Cosmic Burgundy for Space).

