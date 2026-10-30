"""CORTEX-M & CORTEX-M-SD Baseline Memory Baker.

Imprints synthetic discovery episodes (CM-SCI-001, CM-GIST-001), formal mathematical/physical
invariants (FineMath, NIST constants), and orthogonal null-space projectors into binary .pt models.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure src is in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from cortex_m.memory.episodic_store import EpisodicMemoryStore, EpisodicRecord
from cortex_m.learning.continual import ContinualLearningSequence
from cortex_m.data.synthetic import generate_discovery_batch


FOUNDATIONAL_SCIENTIFIC_LAWS = [
    ("conservation_energy", "Total energy in an isolated thermodynamic system remains strictly conserved (dE = dQ - dW)"),
    ("conservation_momentum", "Linear momentum remains constant in any closed inertial frame with zero net external force"),
    ("speed_of_light", "Vacuum speed of light c = 299792458 m/s defines universal invariant causal horizon"),
    ("planck_constant", "Quantum action quanta h = 6.62607015e-34 J*s governs photon energy E = h*nu"),
    ("gravitational_constant", "Newtonian constant of gravitation G = 6.67430e-11 m^3/(kg*s^2) determines metric curvature"),
    ("boltzmann_constant", "Boltzmann constant k_B = 1.380649e-23 J/K connects microscopic entropy to temperature S = k_B * ln(Omega)"),
    ("falsification_criterion", "A scientific hypothesis is epistemically valid only if it makes empirically testable and falsifiable predictions"),
    ("null_hypothesis_testing", "Observed experimental deviations require rejection of null hypothesis at p < 0.01 threshold"),
]

FOUNDATIONAL_CORE_REASONING = [
    ("first_principles", "Deconstruct problems into foundational irreducible axioms and synthesize conclusions deductively"),
    ("counterfactual_evaluation", "Evaluate action pathways against counterfactual outcomes under altered historical conditions"),
    ("epistemic_abstention", "If query cue cosine similarity falls below 0.28, abstain gracefully rather than hallucinate"),
    ("orthogonal_protection", "Gradient updates must project onto null-space of prior attractors: Delta W = Delta W * P_perp"),
    ("structured_gist", "Maintain active invariant constraints in memory buffer while shedding stale conversational noise"),
]


def bake_memory_checkpoint(
    target_name: str,
    domain: str,
    records_list: list[tuple[str, str, str, float]],
    output_path: Path,
    dim: int = 64,
) -> Path:
    store = EpisodicMemoryStore(dim=dim)
    prototypes = []

    for rec_id, claim_text, evidence_type, conf in records_list:
        rec = EpisodicRecord(
            record_id=rec_id,
            timestamp=time.time(),
            source=f"cortex_m.{target_name}",
            claim=claim_text,
            evidence_type=evidence_type,
            confidence=conf,
            context={"target": target_name},
            tags=[target_name, "foundational_memory"],
        )
        store.store(rec)
        vec = store._record_vectors.get(rec.record_id)
        if vec is not None:
            prototypes.append(vec.detach().cpu().numpy().reshape(1, -1))

    stacked_protos = np.vstack(prototypes)
    P_perp = ContinualLearningSequence.orthogonal_projector(stacked_protos)

    payload = {
        "model_id": target_name,
        "domain": domain,
        "dimension": dim,
        "num_foundational_attractors": len(records_list),
        "hopfield_weights": store.hopfield.synaptic_weights.clone(),
        "attractor_matrix": torch.from_numpy(stacked_protos).float(),
        "nullspace_projector": torch.from_numpy(P_perp).float(),
        "epistemic_threshold": 0.28,
        "inverse_temp_beta": store.hopfield.beta,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    torch.save(payload, output_path)
    return output_path


def main():
    models_dir = repo_root / "models" / "checkpoints"
    os.makedirs(models_dir, exist_ok=True)

    print("[INFO] Generating synthetic discovery & gist episode suite...")
    sci_records, gist_records = generate_discovery_batch(num_episodes=25)
    print(f"  - Generated {len(sci_records)} CM-SCI discovery episodes")
    print(f"  - Generated {len(gist_records)} CM-GIST state-drift episodes")

    # 1. Compile CORTEX-M-SD (Scientific Discovery) Memory Checkpoint
    sd_items = []
    for law_id, law_text in FOUNDATIONAL_SCIENTIFIC_LAWS:
        sd_items.append((f"LAW_{law_id}", law_text, "Observed", 0.99))
    for ep in sci_records:
        sd_items.append((ep["record_id"], ep["claim"], ep["evidence_type"], ep["confidence"]))

    sd_path = models_dir / "cortex_sd_baseline.pt"
    bake_memory_checkpoint("cortex_sd", "scientific_discovery", sd_items, sd_path)
    print(f"[SUCCESS] Baked CORTEX-M-SD checkpoint: {sd_path.name} ({sd_path.stat().st_size / 1024.0:.1f} KB, {len(sd_items)} attractors)")

    # 2. Compile CORTEX-Core (Shared Core Reasoning & Memory) Checkpoint
    core_items = []
    for core_id, core_text in FOUNDATIONAL_CORE_REASONING:
        core_items.append((f"CORE_{core_id}", core_text, "Derived", 0.99))
    for g_ep in gist_records:
        core_items.append((g_ep["record_id"], g_ep["claim"], g_ep["evidence_type"], g_ep["confidence"]))

    core_path = models_dir / "cortex_core_baseline.pt"
    bake_memory_checkpoint("cortex_core", "shared_core_reasoning", core_items, core_path)
    print(f"[SUCCESS] Baked CORTEX-Core checkpoint: {core_path.name} ({core_path.stat().st_size / 1024.0:.1f} KB, {len(core_items)} attractors)")


if __name__ == "__main__":
    main()

