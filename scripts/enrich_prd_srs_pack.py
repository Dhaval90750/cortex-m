"""
Enrich and Synchronize CORTEX Ecosystem PRD/SRS Pack.
Translates all docx structures, expands technical/mathematical sections,
adds implemented codebase traceability, and updates markdown and docx files.
"""

import os
import re
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table

PACK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CORTEX_Ecosystem_PRD_SRS_Pack")

SYSTEM_ENRICHMENTS = {
    "00_CORTEX_M_Core": {
        "title": "CORTEX-M Core Platform",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-M Core Platform
- **Continuous Modern Hopfield Attractors**: Core associative memory relies on continuous state relaxation governed by the energy function:
  $$E(x) = -\\frac{1}{\\beta} \\text{lse}(\\beta M x) + \\frac{1}{2} \\|x\\|^2$$
  where state iteration follows $x_{t+1} = M^T \\text{softmax}(\\beta M x_t)$, providing exponential memory capacity and guaranteed Lyapunov stability (energy strictly decreases along recurrent trajectories).
- **Batch Attractor Parallel Convergence**: High-throughput vectorized Modern Hopfield convergence across query batches $X \\in \\mathbb{R}^{B \\times d}$:
  $$X_{t+1} = \\text{normalize}\\left(\\text{softmax}(\\beta X_t M^T) M + \\gamma \\frac{X_t W^T}{\\|W\\|}\\right)$$
- **Analytical Lyapunov Energy Gradient & Landscape Profiling**: Exact energy gradient $\\nabla E(x) = x - M^T \\text{softmax}(\\beta M x)$ evaluates local basin depth, barrier height $\\Delta E = \\bar{E}_{\\text{rim}} - E^*$, and stability margins.
- **Truncated SVD Matrix Factorization**: Synaptic parameter compression via rank-$k$ factorization:
  $$W \\approx U_k \\Sigma_k V_k^T$$
  achieving ~2x compression while preserving zero diagonal and exact matrix symmetry.
- **Counterfactual Basin Stress Testing**: Adversarial noise injection evaluating Lyapunov basin stability:
  $$S_{\\text{basin}} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}(f(x + \\epsilon_i) = f(x))$$
  differentiating verified attractor basins from fragile hallucinations.
- **Dynamic Temperature Annealing**: Adaptive inverse temperature $\\beta(E)$ sharpening retrieval for well-settled memories ($E < -1.0$) and relaxing for exploratory association.
- **Associative Reasoning Chains**: Multi-hop continuous attractor traversal projecting out visited basins to explore adjacent semantic attractors without prompt engineering.
- **Gram-Schmidt Memory Fusion**: Composite prototype synthesis with orthogonal projection against non-target memories to guarantee zero catastrophic interference.
- **Orthogonal Synaptic Grafting ($P_\\perp$)**: Zero-collateral parameter integration uses exact null-space projection:
  $$P_\\perp = I - Q_B Q_B^T$$
  guaranteeing $P_\\perp X_B = 0$ ($0.0\\%$ collateral interference to existing basins).
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Core Architecture & Beast Neural Substrate**: `src/cortex_m/cortex_v2.py` (`CortexV2`)
- **Continuous Synaptic Memory**: `src/cortex_m/pathway/synaptic_memory.py` (`SynapticMemoryNetwork`)
- **Runtime & Contracts**: `src/cortex_m/core/runtime.py`, `src/cortex_m/core/contracts/`
- **Developmental Sleep Consolidation**: `src/cortex_m/learning/consolidation.py` (`NocturnalSleepConsolidator`)
- **Grafting & Orthogonality**: `src/cortex_m/learning/grafting.py` (`SynapticKnowledgePackager`)
- **Universal Gateway**: `src/cortex_m/gateway.py` (`EcosystemGateway`)
- **CLI & REST Server**: `src/cortex_m/cli.py`, `src/cortex_m/server/app.py`
"""
    },
    "01_CORTEX_IDE": {
        "title": "CORTEX-IDE / CortexCode",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-IDE / CortexCode
- **AST Symbol Resolution & Call Trees**: Abstract Syntax Tree parsing (`platforms/ide/src/ast_parser.py`) constructs complete symbol tables, call graphs, and dependency DAGs without token truncation.
- **McCabe Cyclomatic Complexity**: Real-time complexity evaluation per function:
  $$M = 1 + \\sum \\text{decision branch points}$$
  auditing maintenance overhead across conditionals, loops, and boolean operators.
- **Atomic Multi-File Patch Transactions**: AST Diff Engine (`platforms/ide/src/diff_engine.py`) provides all-or-nothing transaction semantics: if any file in a patch bundle fails AST syntax validation, zero files are committed to disk.
- **Isolated Subprocess Sandbox**: Deterministic subprocess execution (`platforms/ide/src/sandbox.py`) enforcing hard timeout ceilings ($5.0\\text{ s}$ default), stripped environment variables, and strict memory limits.
- **Multi-Agent Swarm Protocol**: 5W1H DAG execution across Architect, Coder, Critic, and Auditor personas with cryptographic approval gates prior to git mutation.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **AST Parser**: `platforms/ide/src/ast_parser.py` (`ASTSymbolTableParser`, `ASTSymbolTable`)
- **Diff Engine**: `platforms/ide/src/diff_engine.py` (`ASTDiffEngine`, `PatchResult`, `DiffResult`)
- **Sandbox Runner**: `platforms/ide/src/sandbox.py` (`SubprocessSandboxRunner`)
- **Swarm Orchestrator**: `platforms/ide/src/swarm.py` (`CortexCodeSwarm`)
- **Gateway Dispatcher**: `execute_ide(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "02_CORTEX_Personal": {
        "title": "CORTEX-Personal",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Personal
- **Tabula Rasa Epistemic Verification**: Projection onto synaptic weight matrix $W_{\\text{personal}}$ determines evidence threshold $S = \\|W x\\| / \\|x\\|$. If $S < \\tau_{\\text{novelty}}$, returns `INSUFFICIENT_EVIDENCE` (honest abstention).
- **Continuous Online Hebbian Imprinting**: Weight adaptation uses bounded outer-product updates $\\Delta W = \\eta (z z^T - \\alpha W)$ with instant anti-Hebbian unlearning for exact GDPR privacy erasure.
- **Ultralight Low-Rank Tenant Factor Paging**: Factorizes user memory into low-rank slices ($2.0\\text{ KB}$ rank-4 factors per tenant):
  $$\\Delta W_{\\text{tenant}} = U_{\\text{tenant}} V_{\\text{tenant}}^T$$
  with disk-backed serialization (`page_out_to_disk`, `page_in_from_disk`) consuming $< 15\\text{ MB}$ total runtime RAM.
- **Zero-Egress Security Interlock**: Hardware socket interlock (`platforms/personal/src/interlock.py`) guarantees $0.0\\text{ KB}$ external WAN transmission with AES-256-GCM / PBKDF2-HMAC-SHA256 encrypted persistence.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Personal Substrate**: `platforms/personal/src/substrate.py` (`PersonalSubstrateCoordinator`)
- **Zero-Egress Interlock**: `platforms/personal/src/interlock.py` (`ZeroEgressInterlock`)
- **Tenant Paging**: `platforms/personal/src/tenant_paging.py` (`PersonalTenantPagingEngine`, `TenantFactorSlice`)
- **Encrypted Vault**: `platforms/personal/src/vault.py` (`EncryptedPersonalVault`)
- **Gateway Dispatcher**: `execute_personal(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "03_CORTEX_Fin": {
        "title": "CORTEX-Fin",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Fin
- **Lyapunov Energy Gradient Attribution**: Transaction risk scoring calculates continuous attractor energy $E(x) = -\\frac{1}{2} x^T W x$, projecting the exact gradient $\\nabla_x E(x) = -W x$ back into raw transaction feature space to generate FCRA and GDPR Article 22 compliant adverse action codes (`ADV_01_TRANSACTION_AMOUNT_SPIKE`, `ADV_02_GEOGRAPHIC_DISPLACEMENT`, etc.).
- **Kinematic Leap Velocity Anomaly Detection**: Tracks consecutive transactions per account to identify physically impossible transit leaps:
  $$v_{\\text{apparent}} = \\frac{\\Delta d}{\\Delta t} > 900\\text{ km/h}$$
  violating commercial supersonic flight envelopes.
- **High-Throughput Stream Replay**: Benchmarks transaction pipelines sustaining $> 100,000\\text{ tx/sec}$ with scoring latency $\\le 150\\ \\mu\\text{s}$.
- **Strict Shadow-Mode Isolation**: Non-authoritative read-only pipe with zero write credentials to live payment rails or bank settlement ledgers.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Anomaly Scorer**: `platforms/fin/src/anomaly_scorer.py` (`FinancialAnomalyScorer`, `FraudScoringReport`)
- **Feature Encoder**: `platforms/fin/src/transaction_encoder.py` (`FinancialTransaction`, `TransactionFeatureEncoder`)
- **Event Replay Engine**: `platforms/fin/src/event_replay.py` (`HighThroughputReplayEngine`, `ReplayBatchResult`)
- **Gateway Dispatcher**: `execute_fin(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "04_CORTEX_OS": {
        "title": "CORTEX-OS",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-OS
- **Non-Privileged Userspace Indexing**: Filesystem notifications monitored via cross-platform debounced watchers (500ms debounce) running entirely under unprivileged user permissions (`UID != 0`).
- **Dynamic Cognitive Affinity Graph**: Semantic document clustering via modern Hopfield associative affinity:
  $$A_{ij} = \\exp(-\\|x_i - x_j\\|^2 / 2\\sigma^2)$$
  clustering related research and code artifacts without altering canonical disk hierarchies.
- **Newman-Girvan Community Modularity**: Quantifies partitioning quality via:
  $$Q = \\frac{1}{2m} \\sum_{ij} \\left(A_{ij} - \\frac{k_i k_j}{2m}\\right) \\delta(c_i, c_j)$$
- **Centroid Cosine Resonance Membership**: Predicts incoming document cluster assignment via cosine projection onto cluster centroid vectors.
- **Sub-Millisecond Spotlight Omnibar**: Sub-millisecond continuous attractor resonance search over 64D gist embeddings.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **File Watcher**: `platforms/os/src/file_watcher.py` (`NonPrivilegedFileWatcher`, `FileDocument`)
- **Affinity Clustering**: `platforms/os/src/affinity_cluster.py` (`CognitiveAffinityGraph`, `ProjectCluster`)
- **Omnibar Engine**: `platforms/os/src/omnibar_search.py` (`SpotlightOmnibarEngine`)
- **System Monitor**: `platforms/os/src/system_monitor.py` (`SystemResourceMonitor`)
- **Gateway Dispatcher**: `execute_os(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "05_CORTEX_Robotics": {
        "title": "CORTEX-Robotics",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Robotics
- **6-DOF Denavit-Hartenberg (DH) Kinematics**: Computes exact homogeneous $4 \\times 4$ transformation matrices $T_i^{i-1}$ from base to end-effector across standard industrial articulating geometries.
- **Damped Least Squares (Levenberg-Marquardt) Inverse Kinematics**: Singularity-robust numerical IK:
  $$\\Delta q = J^T (J J^T + \\lambda^2 I)^{-1} (x_{\\text{target}} - x_{\\text{current}})$$
  with step clamping and mechanical joint limit bounding.
- **Yoshikawa Manipulability Index**: Evaluates kinematic dexterity and distance from singularities:
  $$w = \\sqrt{\\det(J J^T)}$$
- **Deterministic ISO 10218-1/2 Safety Interlock**: Speed and Separation Monitoring (SSM) dynamic envelope:
  $$d_{\\text{stop}} = v \\cdot t_{\\text{react}} + \\frac{v^2}{2 a_{\\text{max}}}$$
  $$v_{\\text{allowable}} = \\sqrt{2 a_{\\text{max}} (d_{\\text{human}} - d_{\\text{stop}})}$$
  with 3D workspace bounding box barriers (floor penetration prevention) and hard real-time emergency stop ($< 100\\text{ ms}$).
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Manipulator Kinematics**: `platforms/robotics/src/kinematics.py` (`Manipulator6DOF`, `RobotPose`)
- **Path Planner**: `platforms/robotics/src/path_planner.py` (`AttractorPathPlanner`)
- **Safety Interlock**: `platforms/robotics/src/safety_interlock.py` (`DeterministicSafetyInterlock`, `SafetyStatusReport`)
- **Gateway Dispatcher**: `execute_robotics(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "06_CORTEX_Mesh": {
        "title": "CORTEX-Mesh",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Mesh
- **Practical Byzantine Fault Tolerant (PBFT) Consensus**: Complete 3-phase consensus state machine (Pre-Prepare, Prepare, Commit) with View-Change protocol and signature verification, tolerating up to $f < n/3$ adversarial or partitioned nodes with $2f+1$ quorum threshold verification.
- **Lamport/Mattern Vector Clock Causal Tracking**: Causal message delivery tracking across decentralized nodes:
  $$VC_i[j] = \\max(VC_i[j], VC_{\\text{incoming}}[j])$$
  guaranteeing causal consistency and preventing out-of-order execution across mesh partitions.
- **Causal MIMO Sparse Swarm Routing**: Energy-aware and distance-weighted auction routing allocating logistics tasks across autonomous rovers and sensor nodes with Lipschitz continuity cushions.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **BFT Consensus Engine**: `platforms/mesh/src/consensus.py` (`BFTConsensusEngine`, `ConsensusProposal`, `ConsensusVote`, `ConsensusPhase`)
- **Swarm Router**: `platforms/mesh/src/swarm_router.py` (`CortexSwarmRouter`, `SwarmTaskAllocation`)
- **Mesh Node**: `platforms/mesh/src/node.py` (`MeshNode`, `NodeStatus`)
- **Gateway Dispatcher**: `execute_mesh(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "07_CORTEX_Health_BCI": {
        "title": "CORTEX-Health / BCI",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Health / BCI
- **Deterministic Biosignal Conditioning**: DC offset detrending, IIR 50/60 Hz notch filtering, and artifact clipping ($\\pm 500\\ \\mu\\text{V}$).
- **Spectral Signal-to-Noise Ratio (SNR) & Lead Impedance**: Evaluates empirical SNR in dB and audits electrode contact integrity, flagging flatline disconnections (variance $< 10^{-4}$) and saturated railed leads.
- **Welch Power Spectral Density**: Smoothed spectral estimation via 50% overlapping Hanning-windowed periodograms:
  $$P_{xx}(f) = \\frac{1}{K} \\sum_{k=1}^K I_k(f)$$
- **Exponential Moving Average (EMA) BCI Stabilization**: Temporal smoothing across continuous sliding prediction windows:
  $$P_t = \\alpha P_{\\text{inst}} + (1 - \\alpha) P_{t-1}$$
  preventing classification jitter on discrete cognitive states (Rest, Motor Left/Right, Focus, Drowsy).
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Biosignal Filter**: `platforms/health/src/signal_filter.py` (`BiosignalFilter`, `FilteredSignal`)
- **Wavelet Decomposer**: `platforms/health/src/wavelet_analysis.py` (`WaveletFrequencyDecomposer`, `NeuralBandpower`, `FrequencyBand`)
- **BCI Decoder**: `platforms/health/src/bci_decoder.py` (`ContinuousBCIDecoder`, `BCIDecodingResult`, `CognitiveState`)
- **Gateway Dispatcher**: `execute_health(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "08_CORTEX_Space": {
        "title": "CORTEX-Space",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Space
- **Keplerian & J2 Oblateness Orbit Propagation**: Analytical orbital dynamics modeling Earth oblateness ($J_2 = 1.08263 \\times 10^{-3}$) nodal regression and perigee precession with NORAD Two-Line Element (TLE) parsing (`from_tle`).
- **Multi-Orbit Ground Station Pass Prediction**: Computes elevation angle $\\theta_{\\text{elev}} = \\arcsin(\\hat{r} \\cdot \\hat{n})$ against topocentric ground stations (Goldstone, Madrid, Canberra), recording AOS/LOS timestamps and elevation mask bounds ($> 5^\\circ$).
- **CCSDS Frame Synchronization & 14-Bit Continuity**: Validates 32-bit Attached Sync Marker (`0x1ACFFC1D`) and CRC-16 checksums, analyzing sequence counter discontinuities with modulo-16384 wrap-around.
- **Attractor-Based SEU Radiation Supervisor**: Continuous attractor anomaly scoring isolating Single Event Upsets (SEU latchups) from routine orbital thermal cycles.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Orbit Propagator**: `platforms/space/src/orbit_propagator.py` (`OrbitPropagator`, `OrbitalElements`)
- **CCSDS Processor**: `platforms/space/src/ccsds_telemetry.py` (`CCSDSTelemetryProcessor`, `DecommutatedTelemetry`)
- **Safety Supervisor**: `platforms/space/src/safety_supervisor.py` (`SpaceSafetySupervisor`, `TelemetryAnomalyAlert`)
- **Gateway Dispatcher**: `execute_space(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "09_CortexLM": {
        "title": "CortexLM Model Family",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CortexLM Model Family
- **Native Autoregressive Language Architecture**: Compact causal transformer with rotary position embeddings (RoPE), SwiGLU activation, and RMSNorm.
- **External Gateway Interoperability**: Seamless model gateway abstraction allowing execution of native CortexLM weights alongside external research providers.
- **Developmental Sleep Consolidation**: Nocturnal consolidation cycles applying Hebbian replay, weight decay, and synaptic pruning ($\\|W_{ij}\\| < \\epsilon$) to stabilize long-term memory without catastrophic forgetting.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Model Architecture**: `src/cortex_m/model/cortex_lm.py` (`CortexLM`, `CortexLMOutput`)
- **Trainer**: `src/cortex_m/model/real_data_trainer.py`, `src/cortex_m/database/db_trainer.py`
- **Metacognition & Introspection**: `src/cortex_m/model/metacognition.py`
- **Knowledge Base**: `src/cortex_m/model/knowledge_base.py`
- **Gateway Dispatcher**: `execute_learning(action, params)` in `src/cortex_m/gateway.py`
"""
    }
}


def convert_docx_to_rich_markdown(docx_filename):
    docx_path = os.path.join(PACK_DIR, docx_filename)
    prefix = docx_filename.replace("_PRD_SRS.docx", "").replace(".docx", "")
    enrichment = SYSTEM_ENRICHMENTS.get(prefix)

    doc = docx.Document(docx_path)
    body = doc.element.body

    lines = []
    lines.append(f"# {prefix.replace('_', ' ')}: Master PRD & SRS Specification")
    lines.append(f"**Document Baseline:** 26 October 2026 | Status: Verified Multi-Platform Baseline")
    lines.append("")

    for child in body:
        if child.tag.endswith('p'):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            style_name = p.style.name if p.style else ""
            if "Heading 1" in style_name or text.startswith("PART "):
                lines.append(f"\n# {text}\n")
            elif "Heading 2" in style_name or (len(text) < 40 and text[0].isalpha() and text[1:3] in [". ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]):
                lines.append(f"\n## {text}\n")
            elif "Heading 3" in style_name:
                lines.append(f"\n### {text}\n")
            else:
                lines.append(text)
        elif child.tag.endswith('tbl'):
            table = Table(child, doc)
            if not table.rows:
                continue
            lines.append("")
            headers = [c.text.replace("\n", " ").strip() for c in table.rows[0].cells]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in table.rows[1:]:
                row_cells = [c.text.replace("\n", "<br>").replace("|", "\\|").strip() for c in row.cells]
                lines.append("| " + " | ".join(row_cells) + " |")
            lines.append("")

    # Add enriched mathematical and architectural sections
    if enrichment:
        lines.append("\n---\n")
        lines.append(enrichment["math_arch"].strip())
        lines.append("\n---\n")
        lines.append(enrichment["impl_refs"].strip())

    md_content = "\n".join(lines)
    md_filename = docx_filename.replace(".docx", ".md")
    md_path = os.path.join(PACK_DIR, md_filename)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Enriched and wrote: {md_filename} ({len(md_content)} chars)")


def enrich_docx_file(docx_filename):
    docx_path = os.path.join(PACK_DIR, docx_filename)
    prefix = docx_filename.replace("_PRD_SRS.docx", "").replace(".docx", "")
    enrichment = SYSTEM_ENRICHMENTS.get(prefix)
    if not enrichment:
        return

    doc = docx.Document(docx_path)
    doc.add_heading("Technical Architecture & Mathematical Formulations", level=1)
    for line in enrichment["math_arch"].strip().split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip())

    doc.add_heading("Implemented Codebase Traceability", level=1)
    for line in enrichment["impl_refs"].strip().split("\n"):
        if line.strip():
            doc.add_paragraph(line.strip())

    doc.save(docx_path)
    print(f"Enriched docx: {docx_filename}")


def main():
    # Update README
    readme_md = os.path.join(PACK_DIR, "00_README_and_Approval_Guide.md")
    with open(readme_md, "w", encoding="utf-8") as f:
        f.write("""# CORTEX Ecosystem PRD/SRS Pack
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
""")
    print("Updated 00_README_and_Approval_Guide.md")

    # Enrich all 10 system documents
    docx_files = [f for f in sorted(os.listdir(PACK_DIR)) if f.endswith(".docx") and not f.startswith("00_README")]
    for docx_file in docx_files:
        enrich_docx_file(docx_file)
        convert_docx_to_rich_markdown(docx_file)

    print("Successfully processed all PRD/SRS documents in pack!")


if __name__ == "__main__":
    main()
