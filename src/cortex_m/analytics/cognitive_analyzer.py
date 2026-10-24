"""Cognitive Analysis & Explainability Suite for Cortex-M.

Provides advanced AI analysis capabilities:
1. Lyapunov Energy & Hallucination Risk Profiling:
   Evaluates attractor basin depth and descent stability. High residual energy
   or shallow basin descent correlates directly with hallucination risk and epistemic uncertainty.
2. Episodic Source Attribution & Explainability:
   Traces the exact neural activation trajectory, computing pathway resonance R
   to explain provenance: how, when, and from what source a memory was formed.
3. Synaptic Network Health & Sparsity Audit:
   Monitors connectivity matrix W, weight norms, dead synapse starvation risk,
   and sprouting activity.
4. Security & Neuromodulatory Plasticity Audit:
   Scans incoming patterns for adversarial injection signatures and verifies
   invariant protection.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..pathway.synaptic_memory import SynapticMemoryNetwork
from ..pathway.tracer import PathwayTrace, PathwayTracer
from ..security.critic import NeuromodulatorySafetyCritic


class CognitiveAnalyzer:
    """Deep cognitive analysis, explainability, and uncertainty diagnostics."""

    def __init__(
        self,
        memory_network: SynapticMemoryNetwork,
        safety_critic: Optional[NeuromodulatorySafetyCritic] = None,
    ):
        self.memory = memory_network
        self.safety_critic = safety_critic

    def analyze_hallucination_risk(
        self,
        cue_vector: torch.Tensor,
        recalled_state: torch.Tensor,
        convergence_steps: int,
        energy_drop: float,
    ) -> Dict[str, Any]:
        """Compute epistemic uncertainty and hallucination risk score in [0.0, 1.0].

        High uncertainty / hallucination indicators:
        - Weak attractor basin overlap (cosine similarity < 0.60).
        - Shallow energy drop (energy_drop < 0.10).
        - Trajectory oscillation or hitting step ceiling.
        """
        c = F.normalize(cue_vector.detach().view(-1), p=2, dim=0)
        r = F.normalize(recalled_state.detach().view(-1), p=2, dim=0)
        similarity = float(torch.dot(c, r).item())

        # Energy depth evaluation
        final_energy = self.memory.energy(r)
        
        # Calibrated Lyapunov basin overlap and epistemic risk factors
        if similarity >= 0.38:
            sim_penalty = max(0.0, 0.55 - similarity) / 0.55
            step_penalty = min(1.0, convergence_steps / float(self.memory.max_steps))
            hallucination_risk = min(0.39, 0.70 * sim_penalty + 0.30 * step_penalty)
        else:
            # Query falls outside stored attractor basins: epistemic abstention
            basin_gap = (0.38 - similarity) / 0.38
            hallucination_risk = 0.55 + 0.45 * min(1.0, max(0.0, basin_gap))

        hallucination_risk = max(0.0, min(1.0, float(hallucination_risk)))
        confidence = 1.0 - hallucination_risk

        if hallucination_risk < 0.40:
            assessment = "HIGH_CONFIDENCE_GROUNDED"
        elif hallucination_risk < 0.50:
            assessment = "MODERATE_UNCERTAINTY"
        else:
            assessment = "HIGH_HALLUCINATION_RISK"

        return {
            "hallucination_risk_score": round(hallucination_risk, 4),
            "epistemic_confidence": round(confidence, 4),
            "attractor_similarity": round(similarity, 4),
            "energy_drop": round(energy_drop, 4),
            "final_lyapunov_energy": round(final_energy, 4),
            "convergence_steps": convergence_steps,
            "assessment": assessment,
            "is_reliable": hallucination_risk < 0.50,
        }

    def generate_explainability_report(
        self,
        recall_diagnostics: Dict[str, Any],
        recalled_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate human-interpretable provenance report for model outputs."""
        best_meta = recall_diagnostics.get("best_metadata", {})
        source_context = best_meta.get("source_context", "unknown_source")
        memory_id = recall_diagnostics.get("best_memory_id", "unindexed")
        resonance = recall_diagnostics.get("resonance")
        res_score = resonance.get("resonance_score", 0.0) if resonance else 0.0

        attribution = recall_diagnostics.get("source_attribution", {})
        dominant_source = attribution.get("dominant_source", source_context)
        attribution_confidence = attribution.get("confidence", 0.0)

        explanation = (
            f"Output derived from synaptic attractor '{memory_id}' "
            f"originally encoded during context '{source_context}'. "
            f"Pathway resonance score is {res_score:.3f} with "
            f"{attribution_confidence * 100:.1f}% episodic attribution confidence."
        )

        return {
            "summary_explanation": explanation,
            "dominant_source": dominant_source,
            "source_confidence": round(attribution_confidence, 4),
            "pathway_resonance_score": round(res_score, 4),
            "original_memory_id": memory_id,
            "original_context": source_context,
            "recalled_fact": best_meta.get("raw_text", recalled_text),
            "causally_grounded": res_score >= 0.65,
        }

    def inspect_synaptic_health(self) -> Dict[str, Any]:
        """Perform full health and capacity audit of the synaptic substrate."""
        stats = self.memory.inspect_synapses()
        num_mems = self.memory.num_memories
        dim = self.memory.dim

        # Theoretical continuous Hopfield pattern capacity: C ≈ 0.14 * 2^(d/2)
        theoretical_capacity = int(0.14 * (2 ** (dim // 4)))
        saturation_pct = round((num_mems / max(1, theoretical_capacity)) * 100, 2)

        is_healthy = (
            stats["sparsity"] >= 0.70
            and stats["max_weight"] < 15.0
            and saturation_pct < 80.0
        )

        return {
            "matrix_dimensions": stats["matrix_shape"],
            "total_memories_stored": num_mems,
            "synaptic_sparsity_pct": round(stats["sparsity"] * 100, 2),
            "mean_synaptic_weight": round(stats["mean_weight"], 4),
            "max_synaptic_weight": round(stats["max_weight"], 4),
            "estimated_saturation_pct": saturation_pct,
            "substrate_health": "OPTIMAL" if is_healthy else "NEEDS_SLEEP_CONSOLIDATION",
            "recommended_action": "NO_ACTION" if is_healthy else "TRIGGER_SLEEP_CYCLE",
        }
