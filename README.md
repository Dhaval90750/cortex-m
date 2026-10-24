# Cortex-M: Bio-Inspired Neural Substrate & Cognitive AI Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-98%2F98%20Passing-success.svg)](tests_mimo/)
[![Empirical Benchmarks](https://img.shields.io/badge/Benchmarks-9%2F9%20Passing-success.svg)](results/cortex_v2_research_eval.json)
[![Zero External DBs](https://img.shields.io/badge/Database-Zero%20External%20DBs-purple.svg)](#architecture-pillars)

**Cortex-M** is a bio-inspired neural architecture and cognitive AI platform that replaces external relational/vector databases with a continuous synaptic memory substrate. It unifies **Continuous Modern Hopfield associative retrieval**, **causal MIMO sparse routing**, **128-D factored pre-verbal gist**, and **biological sleep consolidation** with a turnkey developer SDK and foundation model augmentation layer.

---

## 🏛️ Architecture Pillars

```
                                  [ User Input / Prompt ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          [ Dorsal Syntactic Stream ]                 [ Ventral Semantic Stream ]
      (Verbatim Code ASTs, Numbers, Strings)     (Hypersphere Encoding & 128-D Gist)
                       │                                           │
                       │                        ┌──────────────────┴──────────────────┐
                       │                        ▼                                     ▼
                       │            [ Neuromodulatory Critic ]             [ Continuous Hopfield ]
                       │          (Plasticity Gate & Write-Lock)          (Synaptic Energy Descent)
                       │                        │                                     │
                       │                        ▼                                     ▼
                       │            [ Causal MIMO Sparse Router ]         [ Pathway Resonance ]
                       │             (ε-Causal Fallback Cushion)          (Episodic Attribution)
                       │                        │                                     │
                       └────────────────────────┼─────────────────────────────────────┘
                                                ▼
                                    [ Grounded Realization ]
                                 (Direct Output or LLM Prompt)
```

1. **Pure Synaptic Memory (Zero External Databases):**
   - Memories reside directly within physical synaptic connection weights ($\mathbf{W}$) and energy attractor basins.
   - **Hebbian Learning:** Rapid potentiation $\Delta \mathbf{W} = \eta \cdot \mathcal{V}(x) \cdot (\mathbf{z} \mathbf{z}^T)$.
   - **Anti-Hebbian Unlearning:** Targeted synaptic depression $\Delta \mathbf{W} = -\eta \cdot (\mathbf{v} \mathbf{v}^T)$ without retraining.
   - **Pathway Resonance:** Records neural activation trajectories to provide verifiable provenance and explainability.

2. **Causal MIMO Sparse Routing & Factored Gist:**
   - Enforces genuine causal necessity between sparse routes and memory attractors ($>14,000\times$ clean-to-scrambled causality ratio).
   - **128-D Factored Gist:** Deconstructs pre-verbal intent across 4 orthogonal heads (`action`, `entity`, `modality`, `constraint`), dropping semantic collision risk by $10^{-12}$.
   - **$\varepsilon$-Causal Fallback Cushion:** Ensures graceful Lipschitz degradation under adversarial perturbations.

3. **Hierarchical Routing ($N=16, 32, 1024$):**
   - Evaluates routes in $O(k \log N)$ complexity, breaking the combinatorial routing wall in $<5$ ms on consumer CPUs.

4. **Biological Sleep Consolidation & Synaptic Homeostasis:**
   - Sleep-replay cycle prunes parasitic connections to **86.7%+ sparsity**.
   - **Stochastic Synaptic Sprouting:** Reactivates dormant synapses via low-amplitude Brownian noise, preventing dead-synapse starvation.
   - **Quarantine Buffer:** Filters adversarial patterns to eliminate "Nightmare" consolidation injection.

5. **Multi-Tenant Ephemeral Synaptic Paging:**
   - Decomposes weights into a shared base matrix plus per-user rank-4 factors: $\mathbf{W}_{\text{effective}}^{(u)} = \mathbf{W}_{\text{shared}} + \mathbf{U}^{(u)} (\mathbf{V}^{(u)})^T$.
   - **2 KB footprint per tenant** with **0.00% cross-tenant data leakage**, allowing 500,000 isolated concurrent users per GB of RAM.

---

## 🚀 Quick Start

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Dhaval90750/cortex-m.git
cd cortex-m
pip install -r requirements-mimo.txt
```

### Python SDK (`CortexEngine`)

```python
from cortex_m import CortexEngine

# 1. Initialize engine
engine = CortexEngine()

# 2. Epistemic Learning (No Vector DB needed!)
engine.remember(
    "The quantum communication protocol operates on 1550nm fiber.",
    memory_id="quantum_comm",
    source="lab_specs",
)

# 3. Grounded Recall with Provenance
res = engine.recall("What wavelength does the quantum protocol use?")
print(res.text)     # The quantum communication protocol operates on 1550nm fiber.
print(res.explain()) # Output derived from synaptic attractor 'quantum_comm' (Resonance: 0.700)

# 4. Multi-Tenant Ephemeral Isolation (2 KB RAM per tenant)
with engine.tenant("tenant_enterprise_a") as session:
    session.remember("Confidential internal budget is $5M.")
    print(session.recall("What is the budget?").text)

# 5. Cognitive Substrate Health Audit
print(engine.audit())
```

---

## 🤖 Modern LLM Augmentation (`CortexLLMBridge`)

Plug Cortex-M directly into foundation models (Llama, Gemini, Claude, OpenAI, or local PyTorch models) to eliminate vector databases, prevent hallucinations, and maintain explainability:

```python
from cortex_m import CortexEngine

engine = CortexEngine()
bridge = engine.get_llm_bridge()

# Learn factual knowledge
engine.remember(
    "The primary gRPC streaming service operates on Port 9443 with mTLS.",
    memory_id="grpc_port_spec",
    source="devops_manifest",
)

# Augment prompt and assess hallucination risk
augmented = bridge.augment_prompt(
    user_query="Which port and encryption protocol should I connect to?",
    system_prompt="You are a DevOps assistant.",
)

print(augmented["augmented_system_prompt"])
# -> Injects [Grounded Synaptic Context (Source: devops_manifest, Pathway Resonance: 70%)]
print(augmented["hallucination_assessment"]["hallucination_risk_score"])  # e.g., 0.04 (Reliable)
```

### Wrapping Any LLM Callable

```python
def call_my_llm(system_prompt: str, user_prompt: str) -> str:
    # Call your local LLM or API here
    return "Connect to Port 9443 using mTLS."

# Wrap LLM with turnkey grounded memory and working memory tracking
wrapped_llm = bridge.wrap_llm(call_my_llm)
result = wrapped_llm("Which port should I use?")
print(result["response"])
print(result["cortex_diagnostics"])
```

---

## 💻 Command Line Interface (CLI)

```bash
# Learn a fact into synaptic weights
python scripts/cortex_cli.py learn --text "Cluster runs on port 9443" --id "net_spec"

# Recall with provenance
python scripts/cortex_cli.py recall --query "What port does the cluster use?"

# Augment prompt for LLM consumption
python scripts/cortex_cli.py augment --query "Which port to connect to?"

# Analyze 128-D Factored Gist & Dorsal syntax
python scripts/cortex_cli.py analyze --prompt "Synthesize SHA-256 function under 50 lines"

# Audit synaptic substrate health
python scripts/cortex_cli.py audit

# Run biological sleep consolidation
python scripts/cortex_cli.py sleep
```

---

## 📊 Empirical Research Benchmarks

Evaluated via `scripts/run_cortex_v2_eval.py` across 100 research iterations (M2M-1 to M2M-100):

| Benchmark | Capability Measured | Target Metric | Empirical Result | Status |
|---|---|---|---|---|
| **B1** | Semantic Recall & Denoising | QA Sim $\ge 0.70$, Noise $60\%$ | Sim: $0.719$, Denoised: $1.000$ | **PASS** |
| **B2** | Anti-Hebbian Synaptic Unlearning | Targeted memory erased | Targeted erased, rest intact | **PASS** |
| **B3** | Pathway Resonance & Attribution | Resonance $\ge 0.70$, Acc $\ge 80\%$ | $\mathcal{R} = 0.700$, Acc: $80.0\%$ | **PASS** |
| **B4** | Gist Bottleneck Generalization | Compression $2.0\times$ | $64 \to 32 \to 64$, Mode: $0.777$ | **PASS** |
| **B5** | Causality Crisis Resolution | Causality Ratio $\ge 1.50\times$ | Clean: $0.000$, Ratio: **$>14,000\times$** | **PASS** |
| **B6** | Hierarchical Scaling ($N=32$) | Latency $< 10$ ms | Latency: **$4.00$ ms** on CPU | **PASS** |
| **B7** | Sleep Consolidation & Homeostasis | Sparsity $\ge 80\%$, Norm Stable | **$86.52\%$ Sparsity**, $3,653$ pruned | **PASS** |
| **B8** | Prefrontal Working Memory | Multi-turn coreference | Resolved ellipsis: $1.000$ match | **PASS** |
| **B9** | M2M-94..100 MIMO-Gist Suite | Rate-distortion, Invertibility | $\beta=1.25$, $\mathcal{L}_{\text{inv}}=0.078$ | **PASS** |

Run benchmarks locally:
```bash
python scripts/run_cortex_v2_eval.py
```

Run test suite (98 tests):
```bash
python -m unittest discover -s tests_mimo
python -m unittest discover -s tests
```

---

## 📄 Publications & Documentation

- [IEEE Research Paper (PDF)](docs/cortex_m_ieee_research_paper.pdf)
- [IEEE Extended Edition with Graphs (PDF)](docs/cortex_m_ieee_extended_paper.pdf)
- [Solutions & Defense Plan](docs/cortex_m_solutions_and_defense_plan.md)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
