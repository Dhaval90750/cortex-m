"""Neural pathway tracing and synaptic trajectory recording.

Just like biological episodic memory traces neural firing sequences through
hippocampal-cortical loops, this module records:
1. The exact activation trajectory (neurons that fired at each recurrent step).
2. The synaptic weight change (Delta W) that occurred when the memory was learned.
3. Pathway resonance: comparing recall trajectories against original learning paths.
4. Episodic source attribution: identifying which learning episode generated a recall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class PathwayStep:
    """A single step along a neural activation path."""
    step_index: int
    energy: float
    state_norm: float
    top_active_neurons: list[int]
    top_activations: list[float]


@dataclass
class PathwayTrace:
    """Complete episodic record of a neural path traversal."""
    trace_id: str
    phase: str  # "learn", "recall", or "unlearn"
    timestamp: float
    initial_energy: float
    final_energy: float
    steps: list[PathwayStep] = field(default_factory=list)
    synaptic_delta_norm: float = 0.0
    synaptic_delta: Optional[torch.Tensor] = None  # [dim, dim] trace
    final_state: Optional[torch.Tensor] = None
    convergence_step: int = 0
    source_context: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "phase": self.phase,
            "num_steps": len(self.steps),
            "initial_energy": round(self.initial_energy, 4),
            "final_energy": round(self.final_energy, 4),
            "energy_delta": round(self.final_energy - self.initial_energy, 4),
            "synaptic_delta_norm": round(self.synaptic_delta_norm, 4),
            "convergence_step": self.convergence_step,
            "source_context": self.source_context,
            "active_path_summary": [
                {"step": s.step_index, "energy": round(s.energy, 4), "neurons": s.top_active_neurons[:5]}
                for s in self.steps
            ],
        }


class PathwayTracer:
    """Monitors, records, and compares neural pathways through synaptic space."""

    def __init__(self, top_k_neurons: int = 8):
        self.top_k_neurons = top_k_neurons
        self.history: list[PathwayTrace] = []

    def record_step(
        self,
        step_index: int,
        state: torch.Tensor,
        energy: float,
    ) -> PathwayStep:
        """Record the state of active neurons and energy at step t."""
        flat = state.detach().view(-1)
        k = min(self.top_k_neurons, flat.size(0))
        topk = torch.topk(flat.abs(), k=k)
        return PathwayStep(
            step_index=step_index,
            energy=float(energy),
            state_norm=float(flat.norm().item()),
            top_active_neurons=topk.indices.tolist(),
            top_activations=[round(float(v), 4) for v in topk.values],
        )

    def start_trace(
        self,
        trace_id: str,
        phase: str,
        initial_energy: float,
        source_context: Optional[str] = None,
    ) -> PathwayTrace:
        trace = PathwayTrace(
            trace_id=trace_id,
            phase=phase,
            timestamp=time.time(),
            initial_energy=float(initial_energy),
            final_energy=float(initial_energy),
            source_context=source_context,
        )
        return trace

    def finish_trace(
        self,
        trace: PathwayTrace,
        final_state: torch.Tensor,
        final_energy: float,
        synaptic_delta: Optional[torch.Tensor] = None,
        convergence_step: int = 0,
    ) -> PathwayTrace:
        trace.final_state = final_state.detach().clone()
        trace.final_energy = float(final_energy)
        trace.convergence_step = convergence_step
        if synaptic_delta is not None:
            trace.synaptic_delta = synaptic_delta.detach().clone()
            trace.synaptic_delta_norm = float(synaptic_delta.norm().item())
        self.history.append(trace)
        return trace

    @staticmethod
    def compute_resonance(learn_trace: PathwayTrace, recall_trace: PathwayTrace) -> dict[str, Any]:
        """Compute pathway resonance: does the recall path re-traverse the learning path?
        
        Evaluates:
        1. Attractor state alignment (cosine similarity of final converged states).
        2. Neural ensemble overlap (Jaccard similarity of firing neuron indices).
        3. Synaptic pathway projection (alignment with synaptic delta tensor).
        """
        if learn_trace.final_state is None or recall_trace.final_state is None:
            raise ValueError("Both traces must have final_state recorded to compute resonance")

        # 1. Attractor state alignment
        s_learn = F.normalize(learn_trace.final_state.view(-1), p=2, dim=0)
        s_recall = F.normalize(recall_trace.final_state.view(-1), p=2, dim=0)
        state_similarity = float(torch.dot(s_learn, s_recall).item())

        # 2. Neural ensemble overlap across steps
        learn_active_sets = [set(s.top_active_neurons) for s in learn_trace.steps]
        recall_active_sets = [set(s.top_active_neurons) for s in recall_trace.steps]

        all_learn_neurons = set().union(*learn_active_sets) if learn_active_sets else set()
        all_recall_neurons = set().union(*recall_active_sets) if recall_active_sets else set()

        jaccard = (
            len(all_learn_neurons & all_recall_neurons) / len(all_learn_neurons | all_recall_neurons)
            if (all_learn_neurons | all_recall_neurons)
            else 0.0
        )

        # 3. Synaptic pathway alignment: did recall project along Delta W?
        synaptic_alignment = 0.0
        if learn_trace.synaptic_delta is not None:
            dW = learn_trace.synaptic_delta
            proj = torch.matmul(s_recall.unsqueeze(0), torch.matmul(dW, s_recall.unsqueeze(1)))
            # Normalize projection by delta norm
            d_norm = learn_trace.synaptic_delta_norm if learn_trace.synaptic_delta_norm > 0 else 1.0
            synaptic_alignment = float(proj.item() / d_norm)

        # Biological lateral gating: pathway resonance requires convergence into the same attractor basin
        state_gate = max(0.0, state_similarity)
        ensemble_factor = 0.50 + 0.30 * jaccard + 0.20 * max(0.0, min(1.0, synaptic_alignment))
        resonance_score = state_gate * ensemble_factor

        return {
            "resonance_score": round(resonance_score, 4),
            "state_similarity": round(state_similarity, 4),
            "neural_ensemble_overlap": round(jaccard, 4),
            "synaptic_alignment": round(synaptic_alignment, 4),
            "is_pathway_matched": bool(resonance_score >= 0.65),
            "learn_trace_id": learn_trace.trace_id,
            "recall_trace_id": recall_trace.trace_id,
            "source_context": learn_trace.source_context,
        }

    @staticmethod
    def attribute_source(
        recall_trace: PathwayTrace,
        episodic_traces: Dict[str, PathwayTrace],
    ) -> dict[str, Any]:
        """Given a recall trace, attribute WHICH episodic learning event it originated from."""
        if not episodic_traces:
            return {"attributed": False, "best_source": None, "confidence": 0.0, "rankings": []}

        rankings: list[dict[str, Any]] = []
        for mem_id, learn_trace in episodic_traces.items():
            if learn_trace.phase != "learn" or learn_trace.final_state is None:
                continue
            res = PathwayTracer.compute_resonance(learn_trace, recall_trace)
            rankings.append({
                "memory_id": mem_id,
                "source_context": learn_trace.source_context,
                "resonance_score": res["resonance_score"],
                "state_similarity": res["state_similarity"],
            })

        rankings.sort(key=lambda x: x["resonance_score"], reverse=True)
        if not rankings:
            return {"attributed": False, "best_source": None, "confidence": 0.0, "rankings": []}

        best = rankings[0]
        is_attributed = best["resonance_score"] >= 0.65
        return {
            "attributed": is_attributed,
            "best_memory_id": best["memory_id"],
            "best_source_context": best["source_context"],
            "confidence": best["resonance_score"],
            "rankings": rankings[:5],
        }
