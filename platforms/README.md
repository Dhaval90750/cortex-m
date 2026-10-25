# CORTEX-M Industrial Platform Workstreams

This directory houses the product surface implementations and simulation environments for the 8 specialized CORTEX-M product lines as governed by the [Master Ecosystem Implementation Plan](file:///e:/Research/Cortex-M/docs/master_plan/cortex_m_original_ecosystem_implementation_plan.md).

---

## The 8 Specialized Product Lines

| Platform | Workspace | Description | Safety & Qualification Standard |
|---|---|---|---|
| **CORTEX-IDE** | [`platforms/ide/`](file:///e:/Research/Cortex-M/platforms/ide/) | Repository-aware autonomous coding agent with AST diffs and sandboxed test execution. | Sandboxed Subprocess & Human Approval Gate |
| **CORTEX-Personal** | [`platforms/personal/`](file:///e:/Research/Cortex-M/platforms/personal/) | 100% offline desktop & edge assistant with private continuous memory. | Scoped Tenant Isolation & Edge Paging |
| **CORTEX-Fin** | [`platforms/fin/`](file:///e:/Research/Cortex-M/platforms/fin/) | High-throughput transaction replay, fraud scoring research, and shadow-mode risk models. | Deterministic Ledgers & Shadow-Mode Isolation |
| **CORTEX-OS** | [`platforms/os/`](file:///e:/Research/Cortex-M/platforms/os/) | Cognitive user-space scheduler, semantic associative file indexing, and process supervisor. | Userspace Memory Isolation & Capability Gates |
| **CORTEX-Robotics** | [`platforms/robotics/`](file:///e:/Research/Cortex-M/platforms/robotics/) | Digital twin simulation, trajectory supervisor, and deterministic safety interlocks. | ISO 10218-1/2:2025 Robotic Safety Standards |
| **CORTEX-Mesh** | [`platforms/mesh/`](file:///e:/Research/Cortex-M/platforms/mesh/) | Decentralized cooperative logistics swarm, node consensus, and fault tolerance. | Civilian Transparent Protocols & BFT Simulation |
| **CORTEX-Health** | [`platforms/health/`](file:///e:/Research/Cortex-M/platforms/health/) | Non-clinical neural signal processing, EEG/ECoG waveforms, and BCI research. | FDA Non-Clinical BCI Guidance & IEC 62304 |
| **CORTEX-Space** | [`platforms/space/`](file:///e:/Research/Cortex-M/platforms/space/) | Satellite orbital telemetry parsing, radiation anomaly tracking, and mission simulation. | NASA Software Engineering (NPR 7150.2 / SP-8000) |

---

## Architectural Rules
1. **Deterministic Safety Separation:** Neural associative inference supervises and proposes; hardware safety interlocks, financial ledgers, and medical state transitions remain deterministic.
2. **Simulation Before Hardware:** Robotics, Mesh, Health, and Space operate in high-fidelity software simulators before any physical hardware validation.
3. **Common Core Contracts:** All platforms communicate with the underlying runtime via [`cortex_m.core.contracts`](file:///e:/Research/Cortex-M/src/cortex_m/core/contracts/).
