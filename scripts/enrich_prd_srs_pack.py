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
  $$E(x) = -\\text{lse}(\\beta X^T x) + \\frac{1}{2} x^T x + C$$
  where state iteration follows $x_{t+1} = X \\text{softmax}(\\beta X^T x_t)$, providing exponential memory capacity and guaranteed Lyapunov stability.
- **Orthogonal Synaptic Grafting ($P_\\perp$)**: Zero-collateral parameter integration uses exact null-space projection:
  $$P_\\perp = I - Q_B Q_B^T$$
  where $Q_B$ is computed via thin QR/SVD on recipient basin prototypes $X_B$, guaranteeing $P_\\perp X_B = 0$ ($0.0\\%$ collateral interference to existing basins).
- **Universal Event Gateway**: Unified typed pub-sub event bus (`cortex_m.core.EventBus`) with microsecond timestamping, trace IDs, and sensitive field masking.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Runtime & Contracts**: `src/cortex_m/core/runtime.py`, `src/cortex_m/core/contracts/`
- **Memory & Epistemics**: `src/cortex_m/memory/continuous_hopfield.py`, `src/cortex_m/learning/consolidation.py`
- **Grafting & Orthogonality**: `src/cortex_m/learning/grafting.py` (`SynapticKnowledgePackager`)
- **Universal Gateway**: `src/cortex_m/gateway.py` (`EcosystemGateway`)
- **CLI & REST Server**: `src/cortex_m/cli.py`, `src/cortex_m/server/app.py`
"""
    },
    "01_CORTEX_IDE": {
        "title": "CORTEX-IDE / CortexCode",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-IDE / CortexCode
- **AST Symbol Resolution**: Abstract Syntax Tree parsing (`platforms/ide/src/ast_parser.py`) constructs complete symbol call graphs and cross-file reference DAGs without token truncation.
- **Syntactic Diff Integrity**: AST Diff Engine (`platforms/ide/src/diff_engine.py`) enforces syntax-preserving AST chunk replacements with comment preservation and regression prevention.
- **Isolated Subprocess Sandbox**: Deterministic subprocess execution (`platforms/ide/src/sandbox.py`) enforcing hard timeout ceilings ($5.0\\text{ s}$ default), stripped environment variables, and strict memory limits.
- **Multi-Agent Swarm Protocol**: 5W1H DAG execution across Architect, Coder, Critic, and Auditor personas with cryptographic approval gates prior to git mutation.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **AST Parser**: `platforms/ide/src/ast_parser.py` (`ASTSymbolTableParser`)
- **Diff Engine**: `platforms/ide/src/diff_engine.py` (`ASTDiffEngine`)
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
- **Ultralight Low-Rank Tenant Paging**: Factorizes user memory into low-rank slices ($2.0\\text{ KB}$ per tenant factor) consuming $< 15\\text{ MB}$ total runtime RAM.
- **Zero-Egress Security Interlock**: Hardware socket interlock (`platforms/personal/src/interlock.py`) guarantees $0.0\\text{ KB}$ external WAN transmission with AES-256-GCM encrypted persistence.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Personal Substrate**: `platforms/personal/src/substrate.py` (`PersonalSubstrateCoordinator`)
- **Zero-Egress Interlock**: `platforms/personal/src/interlock.py` (`ZeroEgressInterlock`)
- **Tenant Paging**: `platforms/personal/src/tenant_paging.py` (`PersonalTenantPagingEngine`)
- **Encrypted Vault**: `platforms/personal/src/vault.py` (`EncryptedPersonalVault`)
- **Gateway Dispatcher**: `execute_personal(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "03_CORTEX_Fin": {
        "title": "CORTEX-Fin",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Fin
- **Lyapunov Energy Gradient Attribution**: Transaction risk scoring calculates continuous attractor energy $E(x)$, decomposing the gradient $\\nabla_x E(x)$ into explicit interpretable factor attributions satisfying FCRA and GDPR Article 22.
- **High-Throughput Columnar Streaming**: Processes transaction tap streams via Apache Arrow columnar memory buffers sustaining $> 100,000\\text{ tx/sec}$ with scoring latency $\\le 150\\ \\mu\\text{s}$.
- **Strict Shadow-Mode Isolation**: Non-authoritative read-only pipe with zero write credentials to live payment rails or bank settlement ledgers.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Anomaly Scorer**: `platforms/fin/src/anomaly_scorer.py` (`FinancialAnomalyScorer`, `FraudScoringReport`)
- **Feature Encoder**: `platforms/fin/src/transaction_encoder.py` (`FinancialTransaction`, `TransactionFeatureEncoder`)
- **Event Replay Engine**: `platforms/fin/src/event_replay.py` (`HighThroughputReplayEngine`)
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
- **Sub-Millisecond Spotlight Omnibar**: Sub-millisecond continuous attractor resonance search over 64D gist embeddings.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **File Watcher**: `platforms/os/src/file_watcher.py` (`NonPrivilegedFileWatcher`)
- **Affinity Clustering**: `platforms/os/src/affinity_cluster.py` (`CognitiveAffinityGraph`)
- **Omnibar Engine**: `platforms/os/src/omnibar_search.py` (`SpotlightOmnibarEngine`)
- **System Monitor**: `platforms/os/src/system_monitor.py` (`SystemResourceMonitor`)
- **Gateway Dispatcher**: `execute_os(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "05_CORTEX_Robotics": {
        "title": "CORTEX-Robotics",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Robotics
- **6-DOF Forward/Inverse Kinematics**: Analytical and numerical DH kinematic solver with singularity-robust Jacobian pseudo-inverse:
  $$J^+ = J^T (J J^T + \\lambda^2 I)^{-1}$$
- **Continuous Attractor Trajectory Generation**: Quintic polynomial spline interpolation minimizing jerk ($C^2$ continuity) bounded by energy basins.
- **Deterministic ISO 10218-1/2 Safety Interlock**: Speed and Separation Monitoring (SSM) dynamic envelope:
  $$v_{\\text{allowable}} = \\sqrt{2 a_{\\text{max}} (d_{\\text{human}} - d_{\\text{stop}})}$$
  with hard real-time emergency stop ($< 100\\text{ ms}$).
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
- **Practical Byzantine Fault Tolerant (PBFT) Consensus**: 3-phase consensus state machine (Pre-Prepare, Prepare, Commit) tolerating up to $f < n/3$ adversarial or disconnected nodes with Ed25519 cryptographic digest signatures.
- **Causal MIMO Sparse Swarm Routing**: Energy-aware and distance-weighted auction routing allocating logistics tasks across autonomous rovers and sensor nodes with Lipschitz continuity cushions.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **BFT Consensus Engine**: `platforms/mesh/src/consensus.py` (`BFTConsensusEngine`, `ConsensusProposal`, `ConsensusVote`)
- **Swarm Router**: `platforms/mesh/src/swarm_router.py` (`CortexSwarmRouter`, `SwarmTaskAllocation`)
- **Mesh Node**: `platforms/mesh/src/node.py` (`MeshNode`, `NodeStatus`)
- **Gateway Dispatcher**: `execute_mesh(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "07_CORTEX_Health_BCI": {
        "title": "CORTEX-Health / BCI",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Health / BCI
- **Deterministic Biosignal Conditioning**: 4th-order Butterworth bandpass (0.5–100.0 Hz) and 50/60 Hz notch filtering with FastICA ocular artifact suppression.
- **Morlet Continuous Wavelet Decomposition**: Time-frequency spectral analysis computing instantaneous power across Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–50 Hz) bands.
- **State-Space Attractor Decoding**: Continuous Hopfield cognitive state classification (Resting, Focused, Cognitive Workload, Drowsy) with strict non-clinical disclaimers.
""",
        "impl_refs": """
### Implemented Codebase Traceability
- **Biosignal Filter**: `platforms/health/src/signal_filter.py` (`BiosignalFilter`, `FilteredSignal`)
- **Wavelet Decomposer**: `platforms/health/src/wavelet_analysis.py` (`WaveletFrequencyDecomposer`, `NeuralBandpower`)
- **BCI Decoder**: `platforms/health/src/bci_decoder.py` (`ContinuousBCIDecoder`, `BCIDecodingResult`)
- **Gateway Dispatcher**: `execute_health(action, params)` in `src/cortex_m/gateway.py`
"""
    },
    "08_CORTEX_Space": {
        "title": "CORTEX-Space",
        "math_arch": """
### Technical Architecture & Mathematical Foundations: CORTEX-Space
- **Keplerian & J2 Oblateness Orbit Propagation**: Analytical orbital dynamics modeling Earth oblateness ($J_2 = 1.08263 \\times 10^{-3}$) nodal regression and perigee precession with line-of-sight ground station contact cones (Goldstone, Madrid, Canberra).
- **CCSDS Frame Synchronization & Telemetry Decommutation**: Validates 32-bit Attached Sync Marker (`0x1ACFFC1D`) and CRC-16 telecommand/telemetry packet integrity.
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
