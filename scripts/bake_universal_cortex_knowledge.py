"""Universal CORTEX-M Master Memory Baker.

Encodes comprehensive scientific, mathematical, computational, algorithmic,
and platform-domain invariants across all 10 CORTEX-M platforms into binary PyTorch (.pt)
Hopfield attractors and orthogonal null-space projectors.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure src in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from cortex_m.memory.episodic_store import EpisodicMemoryStore, EpisodicRecord
from cortex_m.learning.continual import ContinualLearningSequence
from cortex_m.data.synthetic import generate_discovery_batch


UNIVERSAL_KNOWLEDGE_PACKS: Dict[str, Dict[str, Any]] = {
    "cortex_core": {
        "domain": "universal_cognition_logic_reasoning",
        "facts": [
            ("first_principles", "Deconstruct problems into irreducible axioms and synthesize conclusions deductively", "Derived"),
            ("counterfactual_evaluation", "Evaluate action pathways against counterfactual outcomes under altered conditions", "Derived"),
            ("epistemic_abstention", "If query cue cosine similarity falls below 0.28, abstain gracefully rather than hallucinate", "Derived"),
            ("orthogonal_nullspace", "Continual parameter updates must project onto nullspace P_perp = I - Z^T(ZZ^T)^-1 Z to eliminate catastrophic forgetting", "Derived"),
            ("structured_gist", "Maintain active invariant constraints in memory buffer while shedding stale conversational noise", "Observed"),
            ("godel_incompleteness", "Any consistent formal axiomatic system capable of arithmetic contains undecidable true statements", "Literature"),
            ("turing_halting", "No general algorithm can decide whether an arbitrary Turing machine program halts on input", "Literature"),
            ("shannon_entropy", "Information entropy H(X) = -sum p(x) log2 p(x) defines the fundamental limit of lossless compression", "Derived"),
            ("kolmogorov_complexity", "Algorithmic complexity is the length of the shortest computer program that outputs the given object", "Literature"),
            ("p_vs_np_barrier", "NP problems have solutions verifiable in polynomial time; P versus NP remains the central open complexity question", "Literature"),
            ("bayesian_inference", "Posterior probability P(H|E) = P(E|H) * P(H) / P(E) balances prior belief against empirical evidence", "Derived"),
            ("occam_razor", "Among competing hypotheses explaining observations equally well, select the one with fewest assumptions", "Derived"),
            ("popper_falsification", "A scientific theory is valid only if it generates empirically testable and falsifiable hypotheses", "Derived"),
            ("curry_howard", "Constructive proofs correspond directly to executable functional computer programs", "Literature"),
            ("bellman_optimality", "An optimal policy has the property that whatever initial state and decision, remaining decisions must be optimal", "Derived"),
            ("markov_property", "Future state depends only on the present state and action, conditionally independent of historical path", "Derived"),
            ("cap_theorem", "A distributed data store can simultaneously provide at most two of Consistency, Availability, and Partition tolerance", "Literature"),
            ("byzantine_consensus", "Reaching distributed agreement under Byzantine adversarial nodes requires at least 3f + 1 total replicas", "Literature"),
            ("flp_impossibility", "No deterministic asynchronous consensus protocol can guarantee agreement in the presence of unannounced node crashes", "Literature"),
            ("vector_clocks", "Vector timestamping establishes a strict partial ordering of causal events across distributed processes", "Observed"),
            ("lyapunov_stability", "Continuous Hopfield energy descent dE/dt <= 0 guarantees convergence to stable local attractor states", "Derived"),
            ("hebbian_plasticity", "Synaptic connection strength updates via Delta W = eta * z * z^T strengthen co-active neural pathways", "Observed"),
            ("anti_hebbian_depression", "Targeted memory invalidation applies Delta W = -eta * v * v^T to selectively dissolve attractor basins", "Observed"),
            ("system_1_system_2", "Fast associative intuitive retrieval (Hopfield) coupled with slow deliberate symbolic reasoning (AST/verification)", "Derived"),
        ],
    },
    "cortex_sd": {
        "domain": "universal_scientific_discovery",
        "facts": [
            ("conservation_energy", "Total energy in an isolated thermodynamic system remains strictly conserved (dE = dQ - dW)", "Observed"),
            ("conservation_momentum", "Linear momentum remains constant in any closed inertial frame with zero net external force", "Observed"),
            ("speed_of_light", "Vacuum speed of light c = 299792458 m/s defines universal invariant causal speed limit", "Observed"),
            ("planck_constant", "Quantum action quanta h = 6.62607015e-34 J*s governs photon energy E = h * nu", "Observed"),
            ("gravitational_constant", "Newtonian constant of gravitation G = 6.67430e-11 m^3/(kg*s^2) determines metric spacetime curvature", "Observed"),
            ("boltzmann_constant", "Boltzmann constant k_B = 1.380649e-23 J/K connects microscopic states to macroscopic entropy S = k_B * ln(Omega)", "Observed"),
            ("schrodinger_equation", "Time-dependent quantum state evolution obeys i*hbar * d|psi>/dt = H_hat * |psi>", "Observed"),
            ("heisenberg_uncertainty", "Canonical conjugate observables obey fundamental quantum measurement limit delta_x * delta_p >= hbar / 2", "Observed"),
            ("pauli_exclusion", "Identical fermions cannot occupy identical quantum states simultaneously, explaining atomic shell structure", "Observed"),
            ("maxwell_equations", "Electrodynamics governed by div D = rho, div B = 0, curl E = -dB/dt, curl H = J + dD/dt", "Observed"),
            ("lorentz_force", "Electromagnetic force on charged particle obeys F = q * (E + v x B)", "Observed"),
            ("einstein_field_equations", "Spacetime geometry couples to energy-momentum tensor via G_mu_nu + Lambda*g_mu_nu = (8*pi*G/c^4) * T_mu_nu", "Observed"),
            ("special_relativity_energy", "Relativistic energy-momentum invariant satisfies E^2 = (p*c)^2 + (m0 * c^2)^2", "Observed"),
            ("second_law_thermodynamics", "Isolated thermodynamic system entropy never decreases over time: dS_total >= 0", "Observed"),
            ("carnot_efficiency", "Maximum theoretical thermodynamic heat engine efficiency is bounded by eta = 1 - T_cold / T_hot", "Derived"),
            ("dna_central_dogma", "Genetic information flows from DNA transcription to messenger RNA, translated into functional protein chains", "Observed"),
            ("evolution_natural_selection", "Heritable variation combined with differential reproductive fitness drives evolutionary adaptation", "Observed"),
            ("crispr_mechanism", "Cas9 endonuclease guided by synthetic single guide RNA creates targeted double-strand DNA breaks", "Observed"),
            ("periodic_law", "Chemical and physical properties of chemical elements recur periodically according to atomic number Z", "Observed"),
            ("chemical_equilibrium", "Dynamic chemical equilibrium reached when forward and reverse reaction rates balance: Delta_G = 0", "Observed"),
            ("arrhenius_kinetics", "Chemical reaction rate constant scales exponentially with activation energy k = A * exp(-E_a / (R*T))", "Observed"),
            ("nernst_potential", "Electrochemical equilibrium cell reduction potential obeys E = E0 - (R*T / (n*F)) * ln(Q)", "Derived"),
            ("kepler_laws", "Planetary orbits form ellipses with Sun at focus; equal areas swept in equal times; T^2 is proportional to a^3", "Observed"),
            ("hubbles_law", "Extragalactic recessional velocity scales linearly with proper distance: v = H0 * d", "Observed"),
        ],
    },
    "cortex_ide": {
        "domain": "software_engineering_ide",
        "facts": [
            ("ast_grammar", "Tree-sitter concrete syntax trees provide lossless byte-exact AST representation for symbol parsing", "Observed"),
            ("symbol_table_resolution", "Lexical scoping rules resolve variable declarations, shadowed identifiers, and imported module namespaces", "Derived"),
            ("unified_diff_hunks", "Unified diff hunk headers @@ -l,s +l,s @@ specify strict contextual before/after edit boundaries", "Observed"),
            ("test_sandboxing", "Subprocess execution must enforce hard execution timeouts, memory limits, and environment sanitization", "Observed"),
            ("type_inference", "Hindley-Milner type inference reconstructs principal types without explicit annotation in pure expressions", "Literature"),
            ("static_analysis_cfg", "Control flow graph construction models basic blocks, edge jumps, branch conditions, and dead code reachability", "Derived"),
            ("dataflow_taint_analysis", "Taint analysis tracks untrusted user inputs flowing into critical security sink functions", "Observed"),
            ("refactoring_soundness", "Semantic-preserving transformations maintain AST equivalence under variable renaming and method extraction", "Derived"),
            ("lsp_protocol", "Language Server Protocol standardizes communication between editor clients and language tooling servers", "Literature"),
            ("cyclomatic_complexity", "Cyclomatic complexity M = E - N + 2P bounds the number of independent execution paths through a method", "Derived"),
            ("mutation_testing", "Mutation testing evaluates test suite quality by injecting synthetic syntax defects and checking test failure", "Observed"),
            ("git_rebase_invariants", "Atomic rebase replays commit deltas sequentially, preserving cryptographic parent commit trees", "Observed"),
        ],
    },
    "cortex_personal": {
        "domain": "personal_intelligence_privacy",
        "facts": [
            ("zero_cloud_egress", "Strict network firewall rules prevent personal conversation, token, or context egress beyond localhost", "Observed"),
            ("hardware_enclave_encryption", "Episodic records stored on disk encrypted via AES-256-GCM using hardware TPM enclave keys", "Observed"),
            ("differential_privacy", "Personal queries apply calibrated Laplacian noise to ensure (epsilon, delta)-differential privacy", "Derived"),
            ("recency_attractor_bias", "Continuous Hopfield temperature scaling dynamically boosts recent personal preferences over stale facts", "Observed"),
            ("local_identity_partitioning", "Distinct multi-user identities reside in completely isolated synaptic attractor namespaces", "Observed"),
            ("calendar_conflict_resolution", "Local calendar synchronization detects interval overlaps via interval tree query algorithms", "Derived"),
            ("ephemeral_scratch_pruning", "Temporary scratchpads and transient context sessions automatically wiped from memory upon task completion", "Observed"),
            ("local_quantized_inference", "Quantized local neural checkpoints run entirely in system RAM without external API reliance", "Observed"),
        ],
    },
    "cortex_fin": {
        "domain": "high_throughput_quantitative_finance",
        "facts": [
            ("black_scholes_pde", "Option price V obeys dV/dt + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + r*S*dV/dS - r*V = 0 under no-arbitrage", "Derived"),
            ("option_greeks", "Delta=dV/dS, Gamma=d^2V/dS^2, Vega=dV/d_sigma, Theta=dV/dt, Rho=dV/dr quantify risk sensitivities", "Derived"),
            ("value_at_risk", "Parametric VaR at confidence 1-alpha calculates maximum expected portfolio loss over horizon delta_t", "Observed"),
            ("expected_shortfall", "Conditional VaR (CVaR) computes expected loss given that loss exceeds the VaR threshold", "Derived"),
            ("limit_order_book_fifo", "Order matching engine enforces strict price-time priority execution for bid and ask queues", "Observed"),
            ("cointegration_pairs", "Engle-Granger two-step method identifies stationary linear combinations of non-stationary time series", "Derived"),
            ("markowitz_frontier", "Mean-variance optimization solves min w^T * Sigma * w subject to target return w^T * mu = R", "Derived"),
            ("market_microstructure_spread", "Bid-ask spread reflects inventory holding costs, order processing fees, and adverse selection risk", "Observed"),
            ("risk_neutral_pricing", "Asset pricing under equivalent martingale measure Q equates discounted asset price to a martingale", "Derived"),
            ("order_flow_imbalance", "Order flow imbalance between aggressive buy and sell market orders predicts short-horizon price delta", "Observed"),
        ],
    },
    "cortex_os": {
        "domain": "operating_system_kernel_scheduling",
        "facts": [
            ("cfs_scheduler", "Linux Completely Fair Scheduler maintains a red-black tree indexed by thread virtual runtime vruntime", "Observed"),
            ("virtual_memory_paging", "Multi-level page tables translate virtual addresses to physical pages with hardware TLB caching", "Observed"),
            ("page_fault_resolution", "Demand paging handles major page faults via disk swap in and minor page faults via memory mapping", "Observed"),
            ("deadlock_coffman_conditions", "Deadlock occurs only if Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait hold simultaneously", "Literature"),
            ("bankers_algorithm", "Dijkstra Bankers algorithm ensures resource allocation never transitions system into an unsafe state", "Derived"),
            ("synchronization_primitives", "Atomic compare-and-swap (CAS) underpins lock-free ring buffers, spinlocks, and mutexes", "Observed"),
            ("read_copy_update", "RCU allows concurrent readers zero-lock overhead while writers publish updates via atomic pointers and grace periods", "Observed"),
            ("ebpf_programmable_kernel", "Extended BPF runs sandboxed bytecode in OS kernel space for zero-overhead tracing and packet filtering", "Observed"),
            ("io_uring_async", "io_uring provides true asynchronous zero-copy system call submission and completion ring buffers", "Observed"),
            ("interrupt_handling_halves", "Top-half hardware interrupt handlers execute time-critical ops; bottom-half softirqs handle deferred tasks", "Observed"),
        ],
    },
    "cortex_robotics": {
        "domain": "embodied_kinematics_and_robotics",
        "facts": [
            ("denavit_hartenberg", "DH parameters (link length a, twist alpha, offset d, joint angle theta) parameterize serial robot kinematics", "Observed"),
            ("forward_kinematics", "Successive homogeneous transformation matrices compute end-effector pose T = T1 * T2 * ... * Tn", "Derived"),
            ("inverse_kinematics_jacobian", "Manipulator Jacobian relates joint velocities to Cartesian velocities: v = J(q) * q_dot", "Derived"),
            ("damped_least_squares_ik", "Singularity-robust IK solves q_dot = J^T * (J * J^T + lambda^2 * I)^-1 * v to avoid unbounded joint rates", "Derived"),
            ("quaternion_rotations", "Unit quaternions q = (w, x, y, z) represent 3D spatial rotations avoiding gimbal lock singularities", "Literature"),
            ("pid_closed_loop", "Control signal u(t) = K_p * e(t) + K_i * int e(tau)dtau + K_d * de/dt balances response speed and steady-state error", "Observed"),
            ("computed_torque_control", "Feedback linearization cancels nonlinear Coriolis and gravity forces: tau = M(q)*u + C(q,q_dot)*q_dot + g(q)", "Derived"),
            ("ekf_slam", "Extended Kalman Filter simultaneously estimates robot pose state and landmark positions with covariance updates", "Observed"),
            ("rrt_star_planning", "Rapidly-exploring Random Tree Star converges asymptotically to the optimal collision-free trajectory", "Derived"),
            ("se3_lie_algebra", "Rigid body motion resides in Lie group SE(3), with tangent space Lie algebra se(3) defining twists", "Literature"),
        ],
    },
    "cortex_mesh": {
        "domain": "distributed_mesh_and_p2p_networks",
        "facts": [
            ("kademlia_dht_xor", "Kademlia distributed hash tables calculate node distance via bitwise XOR metric d(x, y) = x XOR y", "Observed"),
            ("raft_consensus", "Raft decomposes distributed consensus into leader election, log replication, and commitment safety", "Observed"),
            ("gossip_swim_protocol", "SWIM gossip protocol provides weakly consistent, O(1) message overhead process group membership", "Observed"),
            ("crdt_eventual_consistency", "Conflict-free Replicated Data Types guarantee eventual consistency via commutative, idempotent join operations", "Derived"),
            ("p2p_nat_hole_punching", "STUN/TURN/ICE protocols establish direct UDP peer sessions across symmetric and cone NAT gateways", "Observed"),
            ("graph_neural_message_passing", "Spatial GNNs update node representations via permutation-invariant neighborhood feature aggregation", "Derived"),
            ("dijkstra_shortest_path", "Dijkstra algorithm finds single-source shortest paths in weighted graphs with non-negative edge costs", "Derived"),
            ("byzantine_fault_tolerance", "PBFT algorithm guarantees state machine replication safety across networks with up to (N-1)/3 malicious nodes", "Literature"),
            ("epidemic_data_dissemination", "Rumor mongering protocols disseminate state updates with high probability in logarithmic time O(log N)", "Derived"),
        ],
    },
    "cortex_health": {
        "domain": "neurotechnology_biosignals_and_bci",
        "facts": [
            ("eeg_international_10_20", "Electrode placements at 10% and 20% intervals across skull reference nasion, inion, and preauricular points", "Observed"),
            ("eeg_delta_waves", "Delta frequency band (0.5 to 4.0 Hz) indicates slow-wave non-REM stage 3/4 deep restorative sleep", "Observed"),
            ("eeg_theta_waves", "Theta frequency band (4.0 to 8.0 Hz) corresponds to drowsiness, memory encoding, and meditative states", "Observed"),
            ("eeg_alpha_waves", "Alpha rhythm (8.0 to 13.0 Hz) emerges from occipital cortex during relaxed wakefulness with closed eyes (Berger effect)", "Observed"),
            ("eeg_beta_waves", "Beta frequency band (13.0 to 30.0 Hz) indicates active cognitive concentration, problem solving, and motor execution", "Observed"),
            ("eeg_gamma_waves", "Gamma frequency band (30.0 to 100.0 Hz) reflects cross-modal sensory binding and conscious cognitive integration", "Observed"),
            ("common_spatial_patterns", "CSP filters maximize variance for one motor imagery class while simultaneously minimizing it for another", "Derived"),
            ("independent_component_analysis", "ICA blinds source separation decomposes multichannel EEG into statistically independent ocular/cardiac artifacts", "Derived"),
            ("p300_evoked_potential", "P300 positive voltage deflection occurs ~300ms after rare, task-relevant oddball sensory stimulus", "Observed"),
            ("ssvep_resonance", "Steady-State Visually Evoked Potentials elicit neural oscillations at exact visual strobe flicker frequencies", "Observed"),
            ("notch_filter_artifact_removal", "Zero-phase IIR notch filter removes 50Hz/60Hz AC mains powerline interference from biosignals", "Observed"),
        ],
    },
    "cortex_space": {
        "domain": "astrodynamics_and_orbital_telemetry",
        "facts": [
            ("keplerian_elements", "Six orbital elements (semi-major axis a, eccentricity e, inclination i, RAAN Omega, arg periapsis omega, true anomaly nu) fix orbit in space", "Observed"),
            ("vis_viva_equation", "Orbital speed obeys v^2 = mu * (2/r - 1/a), where mu = G*M is gravitational parameter", "Derived"),
            ("hohmann_transfer", "Two-impulse orbital transfer connects coplanar circular orbits via semi-ellipse with minimal delta-V budget", "Derived"),
            ("tsiolkovsky_rocket_equation", "Ideal velocity change Delta_v = I_sp * g0 * ln(m_initial / m_final) bounds propellant mass fraction", "Derived"),
            ("lagrange_points", "Five equilibrium points L1-L5 in restricted three-body problem balance gravitational forces with centrifugal force", "Derived"),
            ("j2_oblateness_perturbation", "Earth quadrupole moment J2 causes nodal precession of orbital plane and rotation of line of apsides", "Observed"),
            ("sun_synchronous_orbit", "Retrograde inclination matched to J2 nodal precession causes orbit plane to track mean solar motion", "Derived"),
            ("adcs_reaction_wheels", "Attitude Determination and Control Systems exchange angular momentum via 3-axis reaction wheels and desaturating magnetorquers", "Observed"),
            ("star_tracker_quaternion", "Star tracker algorithms identify star triangles against astronomical catalogs to yield arcsecond attitude quaternions", "Observed"),
            ("solar_radiation_pressure", "Photon momentum flux exerts perturbation force F_srp = (P_sun / c) * A * (1 + R_spec) * cos(theta)", "Observed"),
        ],
    },
}


def bake_universal_checkpoints():
    models_dir = repo_root / "models" / "checkpoints"
    os.makedirs(models_dir, exist_ok=True)

    print(f"[INFO] Initializing Universal Knowledge Baker across all 10 CORTEX-M platforms...")
    print(f"[INFO] Target checkpoint directory: {models_dir}")

    total_attractors = 0

    for platform_name, pack in UNIVERSAL_KNOWLEDGE_PACKS.items():
        domain = pack["domain"]
        facts = pack["facts"]
        dim = 64
        store = EpisodicMemoryStore(dim=dim)
        prototypes = []

        print(f"\n[BAKING] Platform: {platform_name} | Domain: {domain} | Fact Count: {len(facts)}")

        for fact_id, claim_text, ev_type in facts:
            record = EpisodicRecord(
                record_id=f"{platform_name}_{fact_id}",
                timestamp=time.time(),
                source=f"cortex_m.{platform_name}",
                claim=claim_text,
                evidence_type=ev_type,
                confidence=0.99,
                context={"platform": platform_name, "rule": fact_id},
                tags=[platform_name, "universal_foundational_invariant"],
            )
            store.store(record)
            vec = store._record_vectors.get(record.record_id)
            if vec is not None:
                prototypes.append(vec.detach().cpu().numpy().reshape(1, -1))

        stacked_protos = np.vstack(prototypes)
        P_perp = ContinualLearningSequence.orthogonal_projector(stacked_protos)

        payload = {
            "platform": platform_name,
            "domain": domain,
            "dimension": dim,
            "num_foundational_attractors": len(facts),
            "hopfield_weights": store.hopfield.synaptic_weights.clone(),
            "attractor_matrix": torch.from_numpy(stacked_protos).float(),
            "nullspace_projector": torch.from_numpy(P_perp).float(),
            "epistemic_threshold": 0.28,
            "inverse_temp_beta": store.hopfield.beta,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        output_file = models_dir / f"{platform_name}_baseline.pt"
        torch.save(payload, output_file)
        size_kb = output_file.stat().st_size / 1024.0
        total_attractors += len(facts)
        print(f"  -> Successfully Baked: {output_file.name} ({size_kb:.1f} KB, {len(facts)} attractors, P_perp shape {P_perp.shape})")

    print(f"\n[UNIVERSAL SUCCESS] Baked all 10 CORTEX-M platforms!")
    print(f"  - Total Invariant Attractors Imprinted: {total_attractors}")
    print(f"  - Zero JSON files generated. Pure binary tensor .pt checkpoints created.")


if __name__ == "__main__":
    bake_universal_checkpoints()
