"""Standalone Memory Baker for CORTEX-M Platforms.

Generates pre-trained, binary PyTorch (.pt) continuous Hopfield attractor states
for each of the 8 CORTEX-M industrial platforms.
Zero human-readable JSON files are exposed; weights and projections are serialized
directly as binary tensors.
"""

from __future__ import annotations

import os
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F

from cortex_m.memory.episodic_store import EpisodicMemoryStore, EpisodicRecord
from cortex_m.learning.continual import ContinualLearningSequence


PLATFORM_DEFINITIONS = {
    "cortex_ide": {
        "domain": "software_engineering",
        "description": "AST grammar, symbol table resolution, unified diff application, test sandboxing",
        "dimension": 64,
        "seed_facts": [
            ("ast_traversal", "Tree-sitter concrete syntax trees guarantee deterministic node offsets"),
            ("diff_unification", "Hunk headers define @@ -l,s +l,s @@ zero-index replacement boundaries"),
            ("sandbox_boundary", "Subprocess execution must enforce 5.0s hard timeout and strip secrets"),
            ("approval_gate", "Irreversible git operations require explicit human approval barrier"),
            ("type_integrity", "Static type checking via MyPy enforces interface signature invariants"),
        ],
    },
    "cortex_personal": {
        "domain": "personal_assistant",
        "description": "Zero cloud egress, private episodic recall, local user preference timeline",
        "dimension": 64,
        "seed_facts": [
            ("privacy_guarantee", "Zero telemetry or network egress allowed outside localhost boundary"),
            ("local_storage", "Episodic records encrypted locally via hardware enclave keys"),
            ("temporal_decay", "Recent user preferences take priority over stale episodic facts"),
            ("calendar_sync", "Local schedule events indexed with deterministic conflict detection"),
            ("context_isolation", "Personal context streams separated across strict user identities"),
        ],
    },
    "cortex_fin": {
        "domain": "high_throughput_finance",
        "description": "High-throughput transaction replay, fraud anomaly detection, shadow ledgers",
        "dimension": 64,
        "seed_facts": [
            ("ledger_invariance", "Double-entry bookkeeping mandates total debits equal total credits"),
            ("fraud_anomaly", "Hopfield energy spikes above Lyapunov threshold indicate anomalous flow"),
            ("shadow_isolation", "Simulated transactions never commit directly to production settlement"),
            ("event_determinism", "Microsecond order streams must maintain causal timestamp ordering"),
            ("audit_trail", "Every risk scoring decision retains cryptographic provenance hash"),
        ],
    },
    "cortex_os": {
        "domain": "operating_system_scheduling",
        "description": "Cognitive task affinity, userspace process management, associative search",
        "dimension": 64,
        "seed_facts": [
            ("privilege_boundary", "OS system monitor runs in userspace with non-privileged verification"),
            ("semantic_affinity", "Threads with high cosine embedding affinity co-scheduled on same cluster"),
            ("content_addressable", "File recall uses continuous vector signatures instead of rigid paths"),
            ("energy_budget", "CPU throttling balances associative retrieval against thermal throttling"),
            ("memory_guard", "Process working set strictly bounded to prevent swapping thrash"),
        ],
    },
    "cortex_robotics": {
        "domain": "embodied_kinematics",
        "description": "ISO 10218 safety envelope, digital twin co-simulation, velocity limiters",
        "dimension": 64,
        "seed_facts": [
            ("iso_10218_safety", "End-effector velocity must clamp strictly below 250 mm/s near humans"),
            ("kinematic_envelope", "Joint torque limits hard-clamped via deterministic physical interlocks"),
            ("event_sensor_loop", "Neuromorphic Dynamic Vision Sensors trigger microsecond reflexes"),
            ("digital_twin", "Actuator trajectories verified in high-fidelity simulator before hardware"),
            ("singularity_avoidance", "Jacobian determinant monitored to prevent kinematic lockup"),
        ],
    },
    "cortex_mesh": {
        "domain": "decentralized_swarm",
        "description": "BFT consensus, intermittent link routing, shared associative state",
        "dimension": 64,
        "seed_facts": [
            ("bft_consensus", "Byzantine Fault Tolerant consensus tolerates up to f < n/3 faulty peers"),
            ("mesh_gossip", "Attractor energy vectors gossiped over low-bandwidth P2P radio packets"),
            ("intermittent_links", "Opportunistic store-and-forward routing preserves causal sequence"),
            ("distributed_sync", "Associative state converges to global Lyapunov energy minimum"),
            ("peer_discovery", "Zero centralized directory; nodes discover via mDNS cryptographic beacons"),
        ],
    },
    "cortex_health": {
        "domain": "neural_signal_processing",
        "description": "Non-clinical BCI research, EEG telemetry filtering, IEC 62304 boundaries",
        "dimension": 64,
        "seed_facts": [
            ("non_clinical_boundary", "System provides research telemetry only; zero direct medical actuation"),
            ("iec_62304_standard", "Medical software life cycle controls enforce fail-safe defaults"),
            ("signal_filtering", "Artifact suppression removes ocular and muscular microvolt noise"),
            ("spectral_bands", "Continuous bandpass separates Delta, Theta, Alpha, and Beta frequencies"),
            ("deterministic_gates", "Zero probabilistic actuation permitted on biological telemetry"),
        ],
    },
    "cortex_space": {
        "domain": "orbital_telemetry",
        "description": "NASA NPR 7150.2 compliance, radiation fault tolerance, telemetry anomaly isolation",
        "dimension": 64,
        "seed_facts": [
            ("nasa_npr_7150_2", "Flight software compliance mandates zero unhandled panics and static heap"),
            ("radiation_tolerance", "Single Event Upset (SEU) bit-flips detected via triple modular check"),
            ("ccsds_packet_sync", "Telemetry packets parsed according to CCSDS space data standards"),
            ("blackout_resilience", "Orbital occultation communication blackouts handled autonomously"),
            ("drift_detection", "Reaction wheel telemetry drift isolated via Lyapunov attractor basins"),
        ],
    },
}


def bake_platform_memory(platform_name: str, config: Dict[str, Any], output_dir: Path) -> Path:
    """Bakes continuous Hopfield attractor states and orthogonal projectors into .pt binary."""
    dim = config["dimension"]
    seed_facts = config["seed_facts"]

    store = EpisodicMemoryStore(dim=dim)

    # Imprint foundational platform seed facts
    prototypes = []
    for fact_key, fact_text in seed_facts:
        record = EpisodicRecord(
            record_id=f"{platform_name}_{fact_key}",
            timestamp=time.time(),
            source=f"cortex_m.platforms.{platform_name}",
            claim=fact_text,
            evidence_type="Observed",
            confidence=0.99,
            context={"platform": platform_name, "rule_key": fact_key},
            tags=[platform_name, "foundational_rule"],
        )
        store.store(record)
        vec = store._record_vectors.get(record.record_id)
        if vec is not None:
            prototypes.append(vec.detach().cpu().numpy().reshape(1, -1))

    # Compute orthogonal null-space projector (P_perp)
    stacked_protos = np.vstack(prototypes)
    P_perp = ContinualLearningSequence.orthogonal_projector(stacked_protos)

    # Extract Hopfield weight matrix and attractor patterns
    hopfield_W = store.hopfield.synaptic_weights.clone()
    attractor_matrix = torch.from_numpy(stacked_protos).float()
    nullspace_projector = torch.from_numpy(P_perp).float()

    # Compile the binary payload
    payload = {
        "platform": platform_name,
        "domain": config["domain"],
        "dimension": dim,
        "num_foundational_attractors": len(seed_facts),
        "hopfield_weights": hopfield_W,
        "attractor_matrix": attractor_matrix,
        "nullspace_projector": nullspace_projector,
        "epistemic_threshold": 0.28,
        "inverse_temp_beta": store.hopfield.beta,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output_file = output_dir / f"{platform_name}_baseline.pt"
    torch.save(payload, output_file)
    return output_file


def main():
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models" / "checkpoints"
    os.makedirs(models_dir, exist_ok=True)

    print(f"[INFO] Baking standalone binary .pt memories for all 8 CORTEX-M platforms...")
    for platform_name, cfg in PLATFORM_DEFINITIONS.items():
        out_path = bake_platform_memory(platform_name, cfg, models_dir)
        size_kb = out_path.stat().st_size / 1024.0
        print(f"  - Baked: {out_path.name} ({size_kb:.1f} KB) -> Domain: {cfg['domain']}")

    print(f"\n[SUCCESS] All 8 platform memory checkpoints successfully baked into: {models_dir}")
    print("[SECURITY] Zero JSON files generated. Model weights and attractors stored strictly as binary .pt tensors.")


if __name__ == "__main__":
    main()

