# CORTEX-M: Master Engineering & Product Blueprint
## Architectural Standards, 10-Layer Stack, and End-to-End Implementation Roadmap

---

## 1. The Master Vision: What CORTEX-M Should Become

Preserve the central identity of CORTEX-M rather than turning it into a generic wrapper around existing models. The CORTEX-M research identity centers on:
- **CORTEX-Synapse** (Associative continuous Hopfield memory)
- **CORTEX-Router** (Causal MIMO sparse router)
- **CORTEX-Gist** (Factored pre-verbal bottleneck)
- **CORTEX-Trace** (Structured provenance and causal activation trajectory)
- **CORTEX-Shield** (Evidence-aware abstention and risk estimation)
- **CORTEX-Bridge** (Model gateway connecting memory to external models)
- **CORTEX-Swarm / CortexCode** (Autonomous multi-agent engineering)
- **CortexLM** (Continuous attractor language model research family)
- **Cortex Studio & CLI** (Turnkey unified interfaces)

### The Final Product Family
```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORTEX-M UNIFIED RUNTIME                        │
│ (Common runtime, context system, model router, memory, tasks, policy)  │
├───────────────────┬───────────────────┬────────────────────────────────┤
│    CortexLM       │    CortexCode     │        CortexResearch          │
│ Language model    │ Repository-aware  │ Web research, source-backed    │
│ research family & │ coding agent with │ answers, document analysis,    │
│ continuous states │ sandbox execution │ and claim verification         │
├───────────────────┼───────────────────┼────────────────────────────────┤
│    CortexLab      │ Cortex Studio/CLI │        Cortex-Mobile           │
│ Model fine-tuning,│ Unified web/CLI   │ Compact edge inference profile │
│ datasets, evals & │ interfaces to the │ for constrained devices        │
│ benchmark gates   │ same runtime      │ and microcontrollers           │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

**Guiding Principle:** One runtime, multiple model backends, specialized tools, different interfaces, and one shared evaluation framework.

---

## 2. The 5 Core Engineering Rules

1. **Rule 1 — Build a Modular Monolith First**:
   - One Python application, one repository, clearly separated packages.
   - Decoupled interfaces; split into network services only when GPU scaling, training isolation, or sandbox security require it.
2. **Rule 2 — Separate Model, Memory, and Runtime**:
   - The LLM generates and interprets language.
   - Memory retrieves information.
   - The Orchestrator manages task progression.
   - Tools perform deterministic operations.
   - The Policy Layer authorizes actions. Never make one component responsible for all five.
3. **Rule 3 — Preserve Zero-Database Research Without Dogma**:
   - Keep pure synaptic memory ($W$) as a first-class research mode.
   - Use simple durable storage (SQLite / PostgreSQL / files) for configuration, session metadata, file revisions, audit logs, and task recovery.
4. **Rule 4 — Make Every Major Subsystem Replaceable**:
   - Strict interface contracts for ModelGateway, MemoryProvider, SearchProvider, ToolRegistry, and Orchestrator.
5. **Rule 5 — Require Measurable Value for Complex Features**:
   - CORTEX-Gist, CORTEX-Router, CORTEX-Trace, and CORTEX-Shield must have dedicated evaluation suites measuring task improvement, latency, and cost over baseline alternatives.

---

## 3. The 10 Architectural Layers

```
Layer 10: Product Surfaces       [ Studio UI | CLI | SDK | REST/SSE APIs | Mobile ]
Layer 09: Capability Workflows   [ Chat | Coding | Research | Data Analysis | Tutoring ]
Layer 08: CORTEX Orchestrator    [ Task State Machine | Planning | Step DAG | Budgets ]
Layer 07: Model Plane            [ CortexLM | Local SLMs (Ollama) | Hosted APIs | Gateway ]
Layer 06: Tool Plane             [ Filesystem | Shell | Python Sandbox | Git | Web ]
Layer 05: Memory & Knowledge     [ Working Memory | Synaptic Memory | Document Library ]
Layer 04: Trust & Verification   [ Central Policy Engine | Approvals | Sandboxing ]
Layer 03: Persistence & Jobs     [ Structured Metadata | Snapshots | Task Event Logs ]
Layer 02: Observability & Eval   [ CORTEX-Trace | CORTEX-Eval Benchmark Suite | Logs ]
Layer 01: Runtime Infrastructure [ Configuration System | Task Lifecycle | Profiles ]
```

---

## 4. Deep-Dive by Layer

### Layer 1: Runtime Infrastructure
- **Typed Configuration**: Precedence: Admin Policy > Project Config > User Config > Defaults. Clear validation on startup; secrets kept in secure environment stores.
- **Process & Task Lifecycle**: Background tasks managed with explicit states, cancellation tokens, resource limits, and crash recovery.
- **Deployment Profiles**: `Local Research`, `Local Assistant`, `Developer Workstation`, `Team Server`, `Enterprise Private`, and `Edge`.

### Layer 2: Observability & Evaluation
- **CORTEX-Trace**: Structured provenance recording Request ID, Session ID, Model Version, Sources, Synaptic Energy, Router Decisions, Tool Args, and Resource Timings.
- **CORTEX-Eval**: Versioned benchmarks evaluating end-to-end task outcomes alongside mathematical unit metrics.
- **Evaluation Gates**:
  - *CORTEX-Synapse*: Retrieval recall, capacity scaling, interference, and deletion collateral damage.
  - *CORTEX-Router*: Route selection accuracy on unseen out-of-distribution inputs.
  - *CORTEX-Shield*: Calibrated false-acceptance vs. false-rejection error curves.
  - *CortexCode*: Real repository bug fix rates, test pass percentage, and regression rates.

### Layer 3: Persistence & Storage Separation
Do not store conflicting copies or force all state into memory weights:
1. **Working Memory**: Active task context and recent tool outputs.
2. **Conversation History**: Explicit transcript turns and user requests.
3. **User Memory**: Persistent user-approved preferences and facts.
4. **Knowledge Library**: Exact source documents, code ASTs, and versioned texts.
5. **Synaptic Memory**: Continuous Hopfield associative attractors and online weights.

### Layer 4: Trust, Security & Sandboxing
- **Central Policy Engine**: Evaluates `(actor, session, resource, action)` $\rightarrow$ returns `allow`, `deny`, or `require_approval`.
- **Tool Risk Classification**: Low (read/calc) $\rightarrow$ Moderate (write to workspace) $\rightarrow$ High (execute code, network egress) $\rightarrow$ Critical (delete datasets, deploy, publish).
- **Execution Sandboxing**: Disposable subprocess/container environments with memory limits, 5-second timeouts, restricted network egress, and cleanup.
- **Untrusted External Content**: Web pages, PDF text, and tool outputs treated strictly as data, never as executable system prompts.

### Layer 5: Knowledge Ingestion & Hybrid Retrieval
- **Unified Ingestion**: PDF, DOCX, Markdown, HTML, Code, CSV, JSON parsed with line/page references preserved.
- **Hybrid Retrieval**:
  $$\text{Query} \longrightarrow \text{Lexical Search (BM25)} + \text{Semantic Embedding} + \text{CORTEX-Synapse Associative Attractor} \longrightarrow \text{Reranker}$$

### Layer 6: Universal Tool Plane
- Standard tool contract: Input/Output JSON schema, risk tier, execution environment, timeout, and structured error format.
- Interoperability with Model Context Protocol (MCP) as an optional external adapter.

### Layer 7: Model Plane & CortexLM Roadmap
- **Track A (Serving)**: Gateway to existing open-weights (Ollama, vLLM) and cloud APIs.
- **Track B (CortexLM Family)**:
  - *CortexLM-Research*: Continuous attractor state dynamics.
  - *CortexLM-Base*: Small controlled autoregressive baseline.
  - *CortexLM-Instruct*: Fine-tuned for instruction following.
  - *CortexLM-Code*: Repository context, patch generation, and test repair.
  - *CortexLM-Local*: Quantized profile for edge hardware.

### Layer 8: CORTEX Orchestrator
- **Bounded State Machine**: `created` $\rightarrow$ `planning` $\rightarrow$ `running` $\rightarrow$ `waiting_for_approval` $\rightarrow$ `verifying` $\rightarrow$ `completed` / `failed`.
- **Failure Resilience**: Explicit handling for model timeouts, tool errors, partial outputs, and budget exhaustion.
- **Multi-Agent Coordination**: Explicit task contracts between Coordinator, Planner, Coder, Critic, Auditor, and Policy Executor.

### Layer 9: Capability Workflows
- **CortexCode**: Repo scan $\rightarrow$ bounded plan $\rightarrow$ surgical unified diff $\rightarrow$ sandboxed test execution $\rightarrow$ critic review $\rightarrow$ human approval gate.
- **CortexResearch**: Web search $\rightarrow$ page retrieval $\rightarrow$ claim extraction $\rightarrow$ cross-source reconciliation $\rightarrow$ report with verified citations.
- **Data Analysis**: Isolated Python execution, deterministic calculations, automated chart synthesis.
- **Tutor & Analyst**: Socratic instruction, document comparison, revision diffing.

### Layer 10: Product Surfaces (Studio, CLI, SDK)
- **Cortex Studio**: Workspaces for Chat, Coding, Research, Knowledge, Memory Lab, and Model Lab.
- **Unified CLI**: `cortex chat`, `cortex code`, `cortex research`, `cortex knowledge`, `cortex memory`, `cortex eval`.
- **Public SDK / API**: Versioned REST/SSE endpoints for conversations, tasks, tools, memory, and models.

---

## 5. Target Project Directory Structure

```
cortex-m/
├── apps/
│   ├── studio/                 # Studio frontend application
│   ├── cli/                    # Unified CLI command suite
│   └── api/                    # REST / SSE server endpoints
├── cortex_m/
│   ├── core/                   # Runtime, config, errors, events, contracts
│   ├── conversations/          # Sessions, history, context builder
│   ├── models/                 # Gateway, router, backends, prompt templates
│   ├── pathway/                # Synaptic memory, causal MIMO, gist, paging
│   ├── memory/                 # Working, episodic, user memory, lifecycle
│   ├── knowledge/              # Ingestion, parsers, indexing, citations
│   ├── tools/                  # Registry, filesystem, shell, python, git
│   ├── security/               # Auth, policies, approvals, audit, sandbox
│   ├── orchestration/          # Coordinator, tasks, state machine, budgets
│   ├── agents/                 # Coding, research, analysis, verification
│   ├── training/               # Datasets, configs, jobs, checkpoints
│   └── evaluation/             # Test harness, benchmark suites, reporting
├── storage/                    # Migrations, schemas, snapshots
├── tests/                      # Unit, integration, e2e, adversarial, perf
├── configs/                    # Profiles: local, workstation, team, edge
├── docs/                       # Specifications, research papers, plans
└── pyproject.toml              # Packaging and dependency declarations
```

---

## 6. The 11-Phase Progressive Development Roadmap

- **Phase 0 — Baseline Audit**: Inventory existing files, run tests, audit headline claims, verify environment.
- **Phase 1 — Core Runtime**: Package boundaries, typed config, model gateway, context builder, tool contract.
- **Phase 2 — Complete Chatbot**: Multi-turn history, session management, streaming, tool activity, memory controls.
- **Phase 3 — Tools & Unified CLI**: File tools, sandboxed Python, command runner, permissions, CLI commands.
- **Phase 4 — CortexCode Flagship**: Repo discovery, unified diffs, sandboxed test execution, approval gate.
- **Phase 5 — Knowledge Library**: Unified parser pipeline, hybrid retrieval, source citations, access filters.
- **Phase 6 — CortexResearch**: Search providers, page extraction, cross-source verification, injection defense.
- **Phase 7 — Memory Hardening**: Interference benchmarks, unlearning collateral probe, calibrated abstention.
- **Phase 8 — CortexLab & Model Adaptation**: Dataset management, fine-tuning harness, evaluation registry.
- **Phase 9 — Broader Capabilities**: Data analysis, tutoring, long-running background tasks.
- **Phase 10 — Scale & Multi-Profile Release**: Enterprise auth, tenant isolation, signed releases, edge profile.

---

## 7. The Six Release Gates

| Gate | Title | Measurable Acceptance Criteria |
|---|---|---|
| **Gate A** | **Core Platform** | Runtime supports multiple model backends, conversations, streaming, controlled tools, and scoped state. |
| **Gate B** | **Useful Assistant** | Chat, CLI, knowledge retrieval, and source-backed answers work end to end with repeatable evaluations. |
| **Gate C** | **Autonomous Task Execution** | CortexCode completes repository changes with verified patches, test runs, and bounded repairs. |
| **Gate D** | **Reliable Memory & Security** | Unlearning verified with zero collateral damage; tenant isolation and adversarial injection defended. |
| **Gate E** | **Owned Model Capability** | CortexLM training and fine-tuning pipelines produce reproducible improvement on held-out tasks. |
| **Gate F** | **Production-Grade Platform** | Documented deployment profiles (Workstation, Team, Edge) pass recovery, auditability, and load tests. |

---

## 8. Resolution of High-Risk Claims

1. **"Zero Hallucination"**: Replaced by calibrated evidence verification (`VERIFIED`, `PARTIAL`, `CONFLICTING`, `ABSTAIN`).
2. **"14,000× Causal Ratio"**: Report raw numerator/denominator, absolute deltas, variance across seeds, and evaluate on unseen distributions.
3. **"86.7% Pruning"**: Reconciled with empirical measurements; evaluate pre- and post-pruning retrieval degradation.
4. **"500,000 Users/GB"**: Clarified as dormant rank-4 tensor footprint (~2 KB); live concurrent requests measured separately.
5. **"20.8 KB Edge Kernel"**: Define exact scope (routing/attractor kernel vs. language model); benchmark device latency and RAM.
6. **"Zero External Databases"**: Preserved as a core research mode while supporting durable storage for enterprise metadata and audit trails.
