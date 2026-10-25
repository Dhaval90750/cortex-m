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

CORTEX-M unites six core cognitive systems under one modular monolith runtime:
1. **CORTEX-Synapse**: Continuous Hopfield associative memory substrate with instant Hebbian potentiation and zero-collateral anti-Hebbian unlearning.
2. **CORTEX-Shield**: Evidence-aware uncertainty and calibrated epistemic verification (`VERIFIED`, `PARTIALLY_SUPPORTED`, `CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE`).
3. **CORTEX-Router**: Causal MIMO sparse router executing $O(k \log N)$ hierarchical routing with Lipschitz fallback cushions.
4. **CORTEX-Gist**: Factored pre-verbal latent bottleneck separating semantic intent across orthogonal functional heads.
5. **CortexCode**: Repository-aware autonomous software engineering agent with Git integration, AST symbol indexing, unified diffs, 5.0s sandboxed execution, and human approval gates.
6. **CortexLM**: Continuous attractor neural language model operating natively on PyTorch or acting as a grounded gateway for local SLMs (Ollama/GGUF) and external models.

<p align="center">
  <img src="assets/ui_previews/cortex_brand_grid.png" alt="CORTEX-M Ecosystem UI Suite" width="100%" />
</p>

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
| **Universal Studio** | **Cortex Studio** | Turnkey browser workbench with real-time 3D attractor energy visualization and evidence badges. |

---

## 🏛️ The 10-Layer Target Architecture

CORTEX-M organizes all cognitive responsibilities across ten clearly defined architectural layers:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 10 — Product Surfaces: Studio Web UI · CLI · SDK · REST API      │
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

## 🏭 The 8 Industrial Platform Ecosystem

Governed by the [Master Ecosystem Implementation Plan](docs/master_plan/cortex_m_original_ecosystem_implementation_plan.md), CORTEX-M provides dedicated platforms with strict safety separation between neural inference and deterministic control:

```
cortex-m/platforms/
├── ide/         # CORTEX-IDE: Autonomous coding agent with AST diffs & sandboxed test runner
├── personal/    # CORTEX-Personal: 100% offline desktop & edge assistant with private memory
├── fin/         # CORTEX-Fin: Transaction replay, fraud scoring research & shadow-mode analytics
├── os/          # CORTEX-OS: Cognitive user-space scheduler & semantic associative file browser
├── robotics/    # CORTEX-Robotics: ISO 10218-1/2 digital twin simulation & safety envelopes
├── mesh/        # CORTEX-Mesh: Civilian cooperative logistics swarm & decentralized consensus
├── health/      # CORTEX-Health: Non-clinical neural signal processing, EEG/ECoG & BCI research
└── space/       # CORTEX-Space: Orbital satellite telemetry, anomaly tracking & NASA SP-8000
```

### Visual Platform Gallery

| Platform | Interface Preview & Architecture | Key Specialization |
|---|---|---|
| **Cortex Studio Hub** | [![Cortex Studio](assets/ui_previews/cortex_studio_dashboard.png)](platforms/personal/) | 3D Hopfield Attractor Energy Basin & Epistemic Verification Badges |
| **CortexCode IDE** | [![CortexCode](assets/ui_previews/cortex_code_ide.png)](platforms/ide/) | Side-by-Side Unified Diff Reviewer & Sandboxed Subprocess Test Runner |
| **CortexResearch** | [![CortexResearch](assets/ui_previews/cortex_research_dashboard.png)](docs/master_plan/cortex_m_master_engineering_and_product_blueprint_full.md#8-layer-5--knowledge-ingestion-and-retrieval) | Citation Provenance Graph & Multi-Source Contradiction Matrix |
| **CortexLab** | [![CortexLab](assets/ui_previews/cortex_lab_dashboard.png)](docs/master_plan/cortex_m_master_engineering_and_product_blueprint_full.md#11-the-full-language-model-training-programme) | Pre-training & LoRA Fine-Tuning Loss Curves & Attractor Basin Depth |
| **CORTEX-Fin** | [![CORTEX-Fin](assets/ui_previews/cortex_fin_dashboard.png)](platforms/fin/) | >140k tx/sec Transaction Replay & Explainable Attractor Waterfall |
| **CORTEX-Robotics** | [![CORTEX-Robotics](assets/ui_previews/cortex_robotics_console.png)](platforms/robotics/) | 6-DOF Manipulator Digital Twin & ISO 10218-1/2 Safety Interlocks |
| **CORTEX-Mesh** | [![CORTEX-Mesh](assets/ui_previews/cortex_mesh_command.png)](platforms/mesh/) | Decentralized Swarm Topology & BFT Consensus Routing Stream |
| **CORTEX-Space** | [![CORTEX-Space](assets/ui_previews/cortex_space_mission.png)](platforms/space/) | 3D Orbital Ground Track & NASA SP-8000 Telemetry Anomaly Console |
| **CORTEX-Health** | [![CORTEX-Health](assets/ui_previews/cortex_health_bci.png)](platforms/health/) | 16-Channel Microvolt Neural Oscilloscope & Wavelet Spectrogram |
| **CORTEX-OS** | [![CORTEX-OS](assets/ui_previews/cortex_os_dashboard.png)](platforms/os/) | Associative Omnibar Search & Dynamic Cognitive Affinity Clusters |

---

## 🚀 Quickstart & Installation

### Option A: Universal Node / npx Runner (Recommended)
No manual Python environment configuration required if you have Node.js installed:

```bash
# 1. Launch Cortex Studio in browser
npx cortex-m studio

# 2. Interactive conversational chat in terminal
npx cortex-m chat

# 3. Autonomous repository software engineer (CortexCode)
npx cortex-m code "Implement LRU cache in utils.py"

# 4. Run full comparative benchmark suite
npx cortex-m bench
```

### Option B: 1-Click Desktop Scripts
* **Windows**: Double-click `run.bat` or run `.\run.ps1`
* **Linux / macOS**: Run `./run.sh`

### Option C: Python Library Usage
```python
from cortex_m.core import CortexRuntime, CortexConfig, DeploymentProfile

# Initialize runtime with developer workstation profile
runtime = CortexRuntime(config=CortexConfig(profile=DeploymentProfile.DEVELOPER_WORKSTATION))

# 1. Learn new facts into physical synaptic connection weights
runtime.learn("CORTEX-M replaces external vector databases with continuous Hopfield attractors.")

# 2. Query with calibrated epistemic evidence verification
result = runtime.ask("What does CORTEX-M replace vector databases with?")
print(f"Status: {result['evidence_status']}")  # 'VERIFIED'
print(f"Grounded: {result['is_grounded']}")    # True
print(f"Answer: {result['text']}")
```

---

## 📂 Repository Layout

```text
cortex-m/
├── apps/                        # Turnkey Application Surfaces
│   ├── studio/run.py            # Cortex Studio Web Server
│   ├── cli/main.py              # Universal Terminal CLI
│   └── api/                     # Headless REST API
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
│   └── server/                  # Web Workbench Engine
│
├── platforms/                   # 8 Specialized Platform Scaffolds
├── configs/                     # Deployment Profiles (default, edge, workstation)
├── benchmarks/                  # Empirical Benchmark Suites
├── docs/                        # Complete Documentation Suite
│   ├── master_plan/             # Master Engineering Blueprints
│   ├── evaluations/             # Technical Audits & Benchmark Reports
│   ├── papers/                  # IEEE Academic Papers (Markdown, LaTeX, PDF)
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

## 📄 License & Academic Citation

CORTEX-M is open-source software licensed under the **[MIT License](LICENSE)**.

```bibtex
@article{cortex_m_2026,
  title   = {CORTEX-M: Context-Oriented Routing and Trace Exchange for MIMO via Continuous Hopfield Attractors},
  author  = {CORTEX-M Research Team},
  journal = {IEEE Transactions on Neural Networks and Learning Systems (Preprint)},
  year    = {2026}
}
```
