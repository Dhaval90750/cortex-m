# CORTEX-M: Context-Oriented Routing and Trace Exchange for MIMO
### Autonomous Neural Cognition, Continuous Attractor Memory & Industrial AI Ecosystem

<p align="center">
  <img src="assets/cortex_m_logo.png" alt="CORTEX-M Logo" width="220" />
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-brightgreen.svg" alt="Python 3.12" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg" alt="PyTorch 2.1+" /></a>
  <a href="tests_mimo/"><img src="https://img.shields.io/badge/Tests-129%2F129%20Passing-success.svg" alt="Tests: 129/129 Passing" /></a>
  <a href="benchmarks/run_comparison_benchmark.py"><img src="https://img.shields.io/badge/Benchmarks-Comparative%20Suite%20Verified-success.svg" alt="Comparative Benchmarks Verified" /></a>
  <a href="#scientific-comparative-benchmarks"><img src="https://img.shields.io/badge/Database-Zero%20External%20DBs-purple.svg" alt="Zero External DBs" /></a>
</p>

---

## 🌟 Executive Summary & Master Vision

**CORTEX-M** (**C**ontext-**O**riented **R**outing and **T**race **EX**change for **M**IMO) is a bio-inspired, memory-centric cognitive AI runtime. Rather than serving as a generic wrapper around external cloud APIs or delegating memory to external relational/vector databases, CORTEX-M operates over an in-memory physical synaptic substrate powered by **Continuous Hopfield Attractor Basins**, **Causal MIMO Sparse Routing**, and **Calibrated Epistemic Evidence Verification**.

### 👶 The Newborn Substrate Principle (Continuous Learning)
> CORTEX-M begins like a child on its first breath: it possesses a complete neural architecture and the fundamental reflex to breathe (continuous Hopfield dynamics, synaptic plasticity, causal routing, and epistemic verification). It has **zero hardcoded domain facts**, **zero canned answers**, and **zero pre-programmed coding heuristics**. All knowledge is acquired continuously on the go, in direct interaction with the user through online Hebbian potentiation ($\Delta W = \eta z z^T$) and anti-Hebbian depression.

CORTEX-M unites six core cognitive systems under one modular monolith runtime:
1. **CORTEX-Synapse**: Continuous Hopfield associative memory substrate with instant Hebbian potentiation and zero-collateral anti-Hebbian unlearning.
2. **CORTEX-Shield**: Evidence-aware uncertainty and calibrated epistemic verification (`VERIFIED`, `PARTIALLY_SUPPORTED`, `CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE`).
3. **CORTEX-Router**: Causal MIMO sparse router executing $O(k \log N)$ hierarchical routing with Lipschitz fallback cushions.
4. **CORTEX-Gist**: Factored pre-verbal latent bottleneck separating semantic intent across orthogonal functional heads.
5. **CortexCode**: Repository-aware autonomous software engineering agent with Git integration, AST symbol indexing, unified diffs, 5.0s sandboxed execution, and human approval gates.
6. **CortexLM**: Continuous attractor neural language model operating natively on PyTorch or acting as a grounded gateway for local SLMs (Ollama/GGUF) and external models.

---

## 🏷️ Canonical Taxonomy: The CORTEX-M Family

| Subsystem Name | Canonical Term | Scientific Role & Technical Responsibility |
|---|---|---|
| **Core Runtime** | **CORTEX-M Core** | Unified coordinator linking model, memory, tools, events, and security policy. |
| **Language Model** | **CortexLM** | Continuous neural language model operating over Hopfield attractor basins with Lyapunov shielding. |
| **Trace Engine** | **CORTEX-Trace** | Pathway activation trajectory logger delivering verifiable causal provenance ($\mathcal{R}$). |
| **Context Router** | **CORTEX-Router** | Causal MIMO sparse router executing $O(k \log N)$ hierarchical routing across $N=16, 32, 1024$. |
| **Synaptic Memory** | **CORTEX-Synapse** | Continuous Hopfield substrate executing instant Hebbian plasticity ($\Delta W = \eta z z^T$). |
| **Evidence Verifier** | **CORTEX-Shield** | Epistemic verification pipeline preventing confabulation via calibrated cue projection. |
| **Pre-Verbal Gist** | **CORTEX-Gist** | 128-D factored invariant latent bottleneck separating intent across 4 orthogonal heads. |
| **Multi-Agent State** | **CORTEX-Blackboard** | Shared associative working state synchronizing agent coordination through attractor energy minima. |
| **Autonomous Swarm** | **CORTEX-Swarm** | Multi-agent task completer coordinating Coordinator, Planner, Specialist, Critic, and Verifier. |
| **Software Engineer** | **CortexCode** | Repository-aware coding agent with AST diffs, 5-second test sandbox, and human approval gates. |
| **Model Gateway** | **CORTEX-Bridge** | Epistemic prompt augmentation and hallucination scoring bridge for external LLMs / local SLMs. |
| **Headless REST API** | **CORTEX-Server** | Pure headless JSON API exposing status, settings, memory recall, chat, and tool execution. |

---

## 🏛️ The 10-Layer Target Architecture

CORTEX-M organizes all cognitive responsibilities across ten clearly defined architectural layers:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 10 — Product Surfaces: CLI · Python SDK · Headless REST API       │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 9  — Capability Workflows: Chat · Coding · Research · Finance   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 8  — CORTEX Orchestrator: State Machine · Plan · Bounded Budgets │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 7  — Model Plane: CortexLM · Local SLM (Ollama) · Hosted APIs    │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 6  — Tool Plane: Filesystem · Python Sandbox · Git · Calculator   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 5  — Memory Plane: Synapse Attractors · Working State · Vault   │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4  — Trust & Verification: Policy Engine · Approvals · Evidence  │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3  — Persistence: Config Profiles · Task State · Snapshots       │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2  — Observability & Evaluation: CORTEX-Trace · Benchmarks · Logs │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 1  — Runtime Infrastructure: Process Manager · Limits · Recovery │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Scientific Comparative Benchmarks

To establish rigorous scientific validation, CORTEX-M was evaluated against standard production baselines on identical held-out test sets ([`benchmarks/run_comparison_benchmark.py`](benchmarks/run_comparison_benchmark.py)):

### Benchmark Evaluation Matrix

| Metric | BM25 (Lexical) | Vector RAG (Dense Cosine) | CORTEX-M (Continuous Hopfield) |
|---|---|---|---|
| **In-Distribution Recall Accuracy** | $100.0\%$ | $100.0\%$ | **$100.0\%$** |
| **Out-of-Distribution Abstention** | $0.0\%$ *(100% False Grounding)* | $0.0\%$ *(100% False Grounding)* | **$80.0\%$** *(Calibrated Abstention)* |
| **Mean Retrieval Latency** | $0.031\text{ ms}$ | $0.046\text{ ms}$ | **$0.487\text{ ms}$** *(Sub-millisecond)* |
| **Unlearning / Exact Deletion** | ❌ Re-indexing required | ❌ Index rebuild required | **$100.0\%$** *(Anti-Hebbian Depression)* |
| **Unlearning Collateral Damage** | N/A | N/A | **$0.00\%$** *(19/19 unrelated facts intact)* |
| **External Database Dependencies** | Inverted Index / SQLite | FAISS / Pinecone / pgvector | **Zero External Databases** |

> **Key Finding on Epistemic Abstention**: When queried with out-of-distribution prompts (e.g. *"Emperor Penguins in Antarctica"*), conventional vector RAG forces a retrieval on the nearest cosine neighbor, causing 100% false grounding. CORTEX-M projects the initial cue $\xi^{(0)}$ against the synaptic memory matrix $\mathbf{M}$ *before* attractor convergence, accurately detecting query novelty ($S=0.109$) and triggering epistemic abstention (`INSUFFICIENT_EVIDENCE`).

---

## 🗺️ Research Roadmap & Target Architectural Specifications

The 8 specialized platform workstreams below represent our architectural specifications and research roadmap as governed by the [Master Ecosystem Implementation Plan](docs/master_plan/cortex_m_original_ecosystem_implementation_plan.md). They establish the target operational domains and safety boundaries for future phases:

> [!NOTE]
> **Ground-Truth Scope**: The workstreams below are architectural design targets and future research roadmaps, **not** claimed as completed physical deployments. The active repository implements and verifies the underlying continuous neural substrate, synaptic memory, causal routing, and evidence verification runtime upon which these platforms are being engineered.

```text
cortex-m/platforms/
├── ide/         # CORTEX-IDE: Autonomous coding agent with AST diffs & sandboxed test runner
├── personal/    # CORTEX-Personal: 100% offline edge assistant with private memory isolation
├── fin/         # CORTEX-Fin: Transaction replay, fraud scoring research & shadow-mode analytics
├── os/          # CORTEX-OS: Cognitive user-space scheduler & semantic associative file browser
├── robotics/    # CORTEX-Robotics: ISO 10218-1/2 digital twin simulation & safety envelopes
├── mesh/        # CORTEX-Mesh: Civilian cooperative logistics swarm & decentralized consensus
├── health/      # CORTEX-Health: Non-clinical neural signal processing, EEG/ECoG & BCI research
└── space/       # CORTEX-Space: Orbital satellite telemetry, anomaly tracking & NASA SP-8000
```

### Architectural Specifications Matrix

| Platform Workstream | Architectural Role & Target Domain | Qualification & Safety Boundary |
|---|---|---|
| **CORTEX-IDE** | Autonomous repository engineer with AST diffs & test runner | 5.0s subprocess sandbox with human approval gates |
| **CORTEX-Personal** | 100% private local cognitive assistant | Zero cloud egress with tenant synaptic isolation |
| **CORTEX-Fin** | High-throughput transaction replay & fraud scoring research | Read-only shadow-mode execution & deterministic ledgers |
| **CORTEX-OS** | Cognitive userspace process scheduler & semantic indexing | Non-privileged userspace sandbox & semantic clustering |
| **CORTEX-Robotics** | Digital twin trajectory generation & safety envelope simulation | ISO 10218-1/2:2025 deterministic hardware safety interlocks |
| **CORTEX-Mesh** | Cooperative logistics swarm & decentralized routing | Transparent civilian protocols & BFT consensus routing |
| **CORTEX-Health** | Non-clinical neural signal processing & BCI research | FDA Non-Clinical Guidance & IEC 62304 research standards |
| **CORTEX-Space** | Orbital satellite telemetry parsing & anomaly tracking | NASA Flight Software Qualification (NPR 7150.2) specifications |

---

## 🚀 Quickstart & Interactive Usage

### Option A: Universal Terminal CLI & Runner

```bash
# 1. Interactive conversational terminal (learns on the go)
python -m apps.cli.main chat

# 2. Start headless JSON REST API server (port 8888)
python -m apps.api.server

# 3. Run full scientific comparative benchmark suite
python benchmarks/run_comparison_benchmark.py
```

### Option B: Python Library Usage (Continuous On-The-Go Learning)

Demonstrating the **Newborn Substrate Principle**:
```python
from cortex_m.core import CortexRuntime, CortexConfig, DeploymentProfile

# Initialize runtime with fresh, uncommitted synaptic substrate
runtime = CortexRuntime(config=CortexConfig(profile=DeploymentProfile.DEVELOPER_WORKSTATION))

# 1. Ask about an unknown concept — substrate abstains with calibrated epistemic uncertainty
initial = runtime.ask("What is the primary power source of the orbital probe?")
print(f"Status: {initial['evidence_status']}")  # 'INSUFFICIENT_EVIDENCE' (Calibrated Abstention)
print(f"Grounded: {initial['is_grounded']}")    # False

# 2. Imprint new knowledge dynamically into synaptic weights on the go (Hebbian learning)
runtime.learn("The orbital probe utilizes dual triple-junction gallium-arsenide solar arrays.")

# 3. Query again — the energy basin converges to the newly potentiated attractor
recall = runtime.ask("What is the primary power source of the orbital probe?")
print(f"Status: {recall['evidence_status']}")    # 'VERIFIED'
print(f"Grounded: {recall['is_grounded']}")      # True
print(f"Answer: {recall['text']}")
```

---

## 📂 Repository Layout

```text
cortex-m/
├── apps/                        # Headless Application Entry Points
│   ├── cli/main.py              # Universal Terminal CLI
│   └── api/server.py            # Headless JSON REST API Server
│
├── src/cortex_m/                # Core Python Package
│   ├── core/                    # Core Infrastructure & Contracts
│   │   ├── runtime.py           # Unified CortexRuntime Coordinator
│   │   ├── config.py            # Typed Config & Deployment Profiles
│   │   ├── events.py            # Standardized Event Bus (task.started, etc.)
│   │   ├── errors.py            # Domain Exception Hierarchy
│   │   └── contracts/           # Interfaces: ModelGateway, MemoryProvider, PolicyEngine
│   ├── pathway/                 # Continuous Neural Substrate
│   │   ├── synaptic_memory.py   # Continuous Hopfield Network
│   │   ├── causal_mimo.py       # Causal MIMO Sparse Router
│   │   ├── gist_bottleneck.py   # 128-D Factored Gist
│   │   └── tracer.py            # CORTEX-Trace Trajectory Logger
│   ├── security/                # Trust & Risk Plane
│   │   ├── evidence.py          # Calibrated Evidence Verifier
│   │   └── critic.py            # Neuromodulatory Plasticity Gate
│   ├── agents/                  # Autonomous Engineering Swarm
│   │   ├── git_context.py       # Git Branch & AST Symbol Table Parser
│   │   ├── patch_sandbox.py     # Unified Diff Generator & 5.0s Subprocess Runner
│   │   └── task_completer.py    # Multi-Agent Coordination Pipeline
│   └── server/                  # Headless REST API Engine
│
├── platforms/                   # 8 Specialized Platform Architectural Scaffolds
├── configs/                     # Deployment Profiles (default, edge, workstation)
├── benchmarks/                  # Empirical Benchmark Suites
├── docs/                        # Complete Documentation Suite
│   ├── master_plan/             # Master Engineering Blueprints
│   ├── evaluations/             # Technical Audits & Benchmark Reports
│   ├── papers/                  # Research Manuscripts & Working Drafts (Pre-Publication)
│   └── architecture/            # Architecture Decision Records (ADRs)
└── tests/                       # 129 Automated Unit Tests (100% Pass Rate)
```

---

## 🧪 Running the Test Suites

CORTEX-M maintains a strict regression policy with **129/129 tests passing**:

```bash
# Core Runtime, Evidence Verifier & Agent Tests (49 tests)
python -m unittest discover -s tests

# MIMO Pathway, Synaptic Memory & Routing Tests (80 tests)
python -m unittest discover -s tests_mimo

# Run Comparative Benchmark (Continuous Hopfield vs Vector RAG vs BM25)
python benchmarks/run_comparison_benchmark.py
```

---

## 📄 License & Research Status

CORTEX-M is open-source software licensed under the **[MIT License](LICENSE)**.

> [!NOTE]
> CORTEX-M is an active open research and systems engineering initiative. Technical reports and manuscripts in `docs/` are internal working drafts and preprints in progress, not yet published or peer-reviewed in academic journals.

```bibtex
@software{cortex_m_runtime,
  title  = {CORTEX-M: A Continuous Synaptic Substrate for Associative Memory, Causal MIMO Routing, and Grounded Cognition},
  author = {CORTEX-M Research Team},
  year   = {2026},
  url    = {https://github.com/Dhaval90750/cortex-m}
}
```
