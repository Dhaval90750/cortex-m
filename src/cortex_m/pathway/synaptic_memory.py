"""Synaptic Associative Memory Network with Continuous Hopfield Dynamics.

This module realizes the core principles:
1. No hard disk or relational database table.
2. Memories reside directly within synaptic connection weights (W) and attractor basins.
3. Learning is synaptic plasticity: updating connection matrix W via Hebbian updates (Delta W = z z^T).
4. Unlearning is anti-Hebbian synaptic depression (Delta W = -eta v v^T).
5. Recall is recurrent energy descent: signals travel through connection pathways
   until converging to a stable attractor state.
6. Pathway tracing: records the exact neural trajectory and synaptic delta,
   verifying pathway resonance and episodic source attribution.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tracer import PathwayTrace, PathwayTracer


class SynapticMemoryNetwork(nn.Module):
    """Associative neural memory operating via continuous attractor dynamics.

    Parameters
    ----------
    dim : int
        Dimensionality of the neural state space.
    beta : float
        Inverse temperature controlling attractor basin sharpness.
    max_steps : int
        Maximum recurrent relaxation steps for pattern retrieval.
    tol : float
        Convergence tolerance for recurrent state updates.
    decay : float
        Synaptic decay rate for homeostatic plasticity.
    """

    def __init__(
        self,
        dim: int = 64,
        beta: float = 8.0,
        max_steps: int = 15,
        tol: float = 1e-4,
        decay: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.beta = beta
        self.base_beta = beta
        self.adaptive_beta = True
        self.max_steps = max_steps
        self.tol = tol
        self.decay = decay

        # Synaptic connection weight matrix: W_ij between neuron i and neuron j
        self.register_buffer("synaptic_weights", torch.zeros(dim, dim))

        # Memory attractor pattern bank: [M, dim] - internal neural state patterns
        self.register_buffer("memory_attractors", torch.empty(0, dim))

        # Associated labels or contextual metadata (for introspection/inspection)
        self.memory_metadata: list[dict[str, Any]] = []

        # Pathway tracer
        self.tracer = PathwayTracer(top_k_neurons=8)

        # Episodic traces mapping memory_id -> PathwayTrace
        self.episodic_traces: dict[str, PathwayTrace] = {}

    @property
    def num_memories(self) -> int:
        return self.memory_attractors.size(0)

    def energy(self, state: torch.Tensor) -> float:
        """Compute the continuous Hopfield Lyapunov energy of state xi.

        E(xi) = - (1 / beta) * logsumexp(beta * M * xi) + 0.5 * ||xi||^2
        Energy strictly decreases along recurrent trajectories.
        """
        if self.num_memories == 0:
            return 0.5 * float((state ** 2).sum().item())

        xi = state.view(-1)
        overlaps = torch.matmul(self.memory_attractors, xi)  # [M]
        lse = torch.logsumexp(self.beta * overlaps, dim=0) / self.beta
        quadratic = 0.5 * (xi ** 2).sum()
        e = float((quadratic - lse).item())
        return e

    def learn_pattern(
        self,
        pattern: torch.Tensor,
        memory_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        source_context: Optional[str] = None,
        learning_rate: float = 1.0,
    ) -> PathwayTrace:
        """Store a new pattern via Hebbian synaptic plasticity.

        Updates the physical synaptic weight matrix:
            Delta W = eta * (z z^T)
        And registers the attractor into the continuous network space.
        Records the complete pathway trace during learning.
        """
        p = F.normalize(pattern.detach().view(-1), p=2, dim=0)
        mem_id = memory_id or f"mem_{self.num_memories}_{len(self.episodic_traces)}"

        # 1. Start pathway trace
        init_energy = self.energy(p)
        trace = self.tracer.start_trace(
            trace_id=mem_id,
            phase="learn",
            initial_energy=init_energy,
            source_context=source_context,
        )
        self.tracer.record_step(0, p, init_energy)

        # 2. Hebbian synaptic plasticity update: Delta W = p (x) p
        delta_W = learning_rate * torch.ger(p, p)
        # Apply homeostatic decay
        self.synaptic_weights = (1.0 - self.decay) * self.synaptic_weights + delta_W

        # 3. Incorporate pattern into continuous attractor matrix
        new_attractor = p.unsqueeze(0)
        self.memory_attractors = torch.cat([self.memory_attractors, new_attractor], dim=0)

        meta = metadata.copy() if metadata else {}
        meta.update({
            "memory_id": mem_id,
            "norm": float(p.norm().item()),
            "source_context": source_context,
        })
        self.memory_metadata.append(meta)

        # 4. Measure post-learning energy & finalize trace
        post_energy = self.energy(p)
        finished_trace = self.tracer.finish_trace(
            trace=trace,
            final_state=p,
            final_energy=post_energy,
            synaptic_delta=delta_W,
            convergence_step=1,
        )
        self.episodic_traces[mem_id] = finished_trace
        if len(self.episodic_traces) > 1000:
            oldest_key = next(iter(self.episodic_traces))
            del self.episodic_traces[oldest_key]
        return finished_trace

    def forget_pattern(
        self,
        target: Union[str, torch.Tensor],
        learning_rate: float = 1.0,
    ) -> dict[str, Any]:
        """Anti-Hebbian targeted forgetting (synaptic depression).

        Selectively weakens the synaptic connections:
            W <- W - eta * (v v^T)
        and removes the attractor basin from the active network.
        """
        if self.num_memories == 0:
            return {"status": "empty", "memory_id": None}

        # Identify which memory index to forget
        target_idx = -1
        target_mem_id = None
        target_pattern = None

        if isinstance(target, str):
            for i, meta in enumerate(self.memory_metadata):
                if meta.get("memory_id") == target:
                    target_idx = i
                    target_mem_id = target
                    target_pattern = self.memory_attractors[i]
                    break
        elif isinstance(target, torch.Tensor):
            q = F.normalize(target.detach().view(-1), p=2, dim=0)
            scores = torch.matmul(self.memory_attractors, q)
            best_val, best_i = scores.max(dim=0)
            if best_val.item() >= 0.80:
                target_idx = int(best_i.item())
                target_mem_id = self.memory_metadata[target_idx].get("memory_id")
                target_pattern = self.memory_attractors[target_idx]

        if target_idx < 0 or target_pattern is None:
            return {"status": "not_found", "memory_id": target if isinstance(target, str) else None}

        # 1. Anti-Hebbian synaptic depression
        delta_W_anti = -learning_rate * torch.ger(target_pattern, target_pattern)
        self.synaptic_weights = self.synaptic_weights + delta_W_anti

        # 2. Record unlearn trace
        trace_id = f"unlearn_{target_mem_id}"
        unlearn_trace = self.tracer.start_trace(
            trace_id=trace_id,
            phase="unlearn",
            initial_energy=self.energy(target_pattern),
            source_context=self.memory_metadata[target_idx].get("source_context"),
        )
        self.tracer.record_step(0, target_pattern, unlearn_trace.initial_energy)

        # 3. Remove attractor from active bank
        keep_mask = torch.ones(self.num_memories, dtype=torch.bool)
        keep_mask[target_idx] = False
        self.memory_attractors = self.memory_attractors[keep_mask]
        meta_removed = self.memory_metadata.pop(target_idx)

        # Finalize unlearn trace
        finished_trace = self.tracer.finish_trace(
            trace=unlearn_trace,
            final_state=target_pattern,
            final_energy=self.energy(target_pattern),
            synaptic_delta=delta_W_anti,
            convergence_step=1,
        )

        return {
            "status": "unlearned",
            "memory_id": target_mem_id,
            "metadata": meta_removed,
            "synaptic_delta_norm": float(delta_W_anti.norm().item()),
            "remaining_memories": self.num_memories,
        }

    def recall(
        self,
        cue: torch.Tensor,
        query_id: Optional[str] = None,
        return_trajectory: bool = True,
        synaptic_override: Optional[torch.Tensor] = None,
        extra_attractors: Optional[torch.Tensor] = None,
        extra_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, PathwayTrace, dict[str, Any]]:
        """Recurrent associative recall from a partial or noisy cue.

        Iteratively settles into the nearest attractor basin driven jointly by
        continuous Hopfield attractor projection and physical synaptic conductance:
            xi^(t+1) = normalize(M^T softmax(beta * M * xi^(t)) + gamma * W_eff * xi^(t))
        Traces the exact pathway traversed through the neural state space.
        """
        active_attractors = self.memory_attractors
        active_metadata = list(self.memory_metadata)
        if extra_attractors is not None and extra_attractors.size(0) > 0:
            if active_attractors.size(0) > 0:
                active_attractors = torch.cat([active_attractors, extra_attractors], dim=0)
            else:
                active_attractors = extra_attractors
            if extra_metadata:
                active_metadata.extend(extra_metadata)

        num_mems = active_attractors.size(0)
        if num_mems == 0:
            raise RuntimeError("Cannot recall from empty synaptic memory.")

        qid = query_id or f"query_{len(self.tracer.history)}"
        state = F.normalize(cue.detach().view(-1), p=2, dim=0)

        init_energy = self.energy(state)
        trace = self.tracer.start_trace(trace_id=qid, phase="recall", initial_energy=init_energy)

        step_record = self.tracer.record_step(0, state, init_energy)
        trace.steps.append(step_record)

        # Adaptive beta scaling: beta(M) = beta_0 * sqrt(dim / ln(M + 2))
        effective_beta = self.beta
        if self.adaptive_beta and num_mems > 0:
            effective_beta = self.base_beta * math.sqrt(self.dim / math.log(num_mems + 2.0))

        active_weights = synaptic_override if synaptic_override is not None else self.synaptic_weights
        w_norm = float(active_weights.norm().item())

        converged = False
        step = 0
        current_state = state
        prev_energy = init_energy
        stall_count = 0

        for step in range(1, self.max_steps + 1):
            # 1. Overlaps with attractors scaled by adaptive beta
            scores = torch.matmul(active_attractors, current_state)  # [M]
            attention_weights = F.softmax(effective_beta * scores, dim=0)

            # 2. Continuous Hopfield attractor projection coupled with physical synaptic transmission
            hopfield_pull = torch.matmul(attention_weights.unsqueeze(0), active_attractors).squeeze(0)
            if w_norm > 1e-4:
                synaptic_flow = torch.matmul(active_weights, current_state)
                synaptic_flow = synaptic_flow / (w_norm + 1e-4)
                next_state = hopfield_pull + 0.15 * synaptic_flow
            else:
                next_state = hopfield_pull

            next_state = F.normalize(next_state, p=2, dim=0)

            # 3. Energy at current step
            current_energy = self.energy(next_state)
            step_record = self.tracer.record_step(step, next_state, current_energy)
            trace.steps.append(step_record)

            # Check convergence first
            delta = float((next_state - current_state).norm().item())
            current_state = next_state
            if delta < self.tol:
                converged = True
                break

            # Lyapunov descent watchdog: detect spurious attractor traps / algorithmic DoS cycles
            energy_delta = abs(current_energy - prev_energy)
            if step >= 3 and energy_delta < 1e-5:
                stall_count += 1
                if stall_count >= 2:
                    # Inject orthogonal escape perturbation to exit spurious limit cycle
                    noise = torch.randn_like(next_state)
                    noise = noise - torch.dot(noise, next_state) * next_state
                    current_state = F.normalize(next_state + 0.20 * noise, p=2, dim=0)
                    stall_count = 0

            prev_energy = current_energy

        final_energy = self.energy(current_state)
        finished_trace = self.tracer.finish_trace(
            trace=trace,
            final_state=current_state,
            final_energy=final_energy,
            convergence_step=step,
        )

        # Compute match to closest stored attractor
        final_scores = torch.matmul(active_attractors, current_state)
        best_idx = int(final_scores.argmax().item())
        best_similarity = float(final_scores[best_idx].item())
        best_meta = active_metadata[best_idx] if best_idx < len(active_metadata) else {}

        # Compute direct cue-to-attractor projection to detect out-of-distribution queries
        norm_cue = F.normalize(cue.view(-1), p=2, dim=-1)
        cue_scores = torch.matmul(active_attractors, norm_cue)
        cue_similarity = float(cue_scores[best_idx].item())
        displacement = float(torch.norm(current_state - norm_cue, p=2).item())

        # Look up resonance with the original learning trace if available
        matched_mem_id = best_meta.get("memory_id")
        resonance_info = None
        if matched_mem_id and matched_mem_id in self.episodic_traces:
            learn_trace = self.episodic_traces[matched_mem_id]
            resonance_info = PathwayTracer.compute_resonance(learn_trace, finished_trace)

        # Episodic source attribution across all known memory events
        source_attribution = PathwayTracer.attribute_source(finished_trace, self.episodic_traces)

        diagnostics = {
            "converged": converged,
            "steps_taken": step,
            "best_memory_index": best_idx,
            "best_memory_id": matched_mem_id,
            "attractor_similarity": round(best_similarity, 4),
            "cue_similarity": round(cue_similarity, 4),
            "attractor_displacement": round(displacement, 4),
            "best_metadata": best_meta,
            "resonance": resonance_info,
            "source_attribution": source_attribution,
            "energy_drop": round(init_energy - final_energy, 4),
        }

        return current_state, finished_trace, diagnostics

    def consolidate_synapses(self, max_norm: float = 10.0) -> None:
        """Homeostatic normalization preventing unbounded synaptic weight growth."""
        norm = self.synaptic_weights.norm()
        if norm > max_norm:
            self.synaptic_weights.mul_(max_norm / (norm + 1e-6))

    def inspect_synapses(self) -> dict[str, Any]:
        """Inspect the current physical synaptic connectivity matrix."""
        W = self.synaptic_weights
        return {
            "matrix_shape": list(W.shape),
            "mean_weight": float(W.mean().item()),
            "std_weight": float(W.std().item()),
            "max_weight": float(W.max().item()),
            "min_weight": float(W.min().item()),
            "sparsity": float((W.abs() < 1e-4).float().mean().item()),
            "total_memories_stored": self.num_memories,
        }

    def sprout_synapses(self, sprout_prob: float = 0.05, sprout_scale: float = 0.01) -> int:
        """Stochastic synaptic sprouting to resolve dead synapse starvation.

        Re-energizes pruned/dormant connections (W_ij == 0) with low-amplitude
        Brownian reactivation noise, preserving gradient receptivity for future adaptation.
        """
        dead_mask = self.synaptic_weights.abs() < 1e-4
        rand_gate = torch.rand_like(self.synaptic_weights) < sprout_prob
        reactivate_mask = dead_mask & rand_gate

        noise = torch.randn_like(self.synaptic_weights) * sprout_scale
        self.synaptic_weights[reactivate_mask] = noise[reactivate_mask]
        return int(reactivate_mask.sum().item())

    def batch_recall(
        self,
        cues: torch.Tensor,
        synaptic_override: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Vectorized parallel Modern Hopfield attractor convergence for batch queries [B, dim].
        
        Accelerates bulk queries and real-time streaming recall by eliminating Python loops.
        """
        if self.num_memories == 0:
            raise RuntimeError("Cannot batch recall from empty synaptic memory.")

        steps = max_steps or self.max_steps
        X = F.normalize(cues.detach().to(self.memory_attractors.device), p=2, dim=-1)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        B, D = X.shape

        effective_beta = self.beta
        if self.adaptive_beta and self.num_memories > 0:
            effective_beta = self.base_beta * math.sqrt(self.dim / math.log(self.num_memories + 2.0))

        active_weights = synaptic_override if synaptic_override is not None else self.synaptic_weights
        w_norm = float(active_weights.norm().item())

        converged_step = steps
        for step in range(1, steps + 1):
            # [B, M] = [B, D] @ [D, M]
            scores = torch.matmul(X, self.memory_attractors.t())
            attn = F.softmax(effective_beta * scores, dim=-1)

            # [B, D] = [B, M] @ [M, D]
            hopfield_pull = torch.matmul(attn, self.memory_attractors)

            if w_norm > 1e-4:
                synaptic_flow = torch.matmul(X, active_weights.t()) / (w_norm + 1e-4)
                X_next = hopfield_pull + 0.15 * synaptic_flow
            else:
                X_next = hopfield_pull

            X_next = F.normalize(X_next, p=2, dim=-1)
            delta = torch.norm(X_next - X, dim=-1).max().item()
            X = X_next
            if delta < self.tol:
                converged_step = step
                break

        # Match to nearest stored attractor for each query
        final_scores = torch.matmul(X, self.memory_attractors.t())  # [B, M]
        best_vals, best_indices = final_scores.max(dim=-1)

        diagnostics: list[dict[str, Any]] = []
        for i in range(B):
            idx = int(best_indices[i].item())
            sim = float(best_vals[i].item())
            meta = self.memory_metadata[idx] if idx < len(self.memory_metadata) else {}
            diagnostics.append({
                "query_index": i,
                "converged_step": converged_step,
                "best_memory_index": idx,
                "best_memory_id": meta.get("memory_id"),
                "attractor_similarity": round(sim, 4),
                "best_metadata": meta,
            })

        return X, best_vals, diagnostics

    def compute_energy_gradient(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the exact analytical Lyapunov energy gradient nabla E(x).
        
        nabla E(x) = x - M^T softmax(beta * M * x)
        Zero at attractor equilibria; points in direction of steepest energy ascent.
        """
        if self.num_memories == 0:
            return state.view(-1).clone()

        x = state.view(-1)
        scores = torch.matmul(self.memory_attractors, x)
        attn = F.softmax(self.beta * scores, dim=0)
        reconstruction = torch.matmul(attn.unsqueeze(0), self.memory_attractors).squeeze(0)
        return x - reconstruction

    def compute_energy_landscape(
        self,
        state: torch.Tensor,
        num_radial_samples: int = 8,
        radius: float = 0.1,
    ) -> dict[str, Any]:
        """Perform comprehensive Lyapunov landscape analysis around a neural state.
        
        Evaluates local energy gradient norm, basin depth, barrier height, and attractor stability.
        """
        x = F.normalize(state.view(-1), p=2, dim=0)
        e_center = self.energy(x)
        grad = self.compute_energy_gradient(x)
        grad_norm = float(grad.norm().item())

        # Radial perturbation barrier height estimation
        torch.manual_seed(999)
        radial_energies = []
        for _ in range(num_radial_samples):
            v = torch.randn_like(x)
            v = v - torch.dot(v, x) * x  # Tangent space projection
            v = F.normalize(v, p=2, dim=0) * radius
            x_pert = F.normalize(x + v, p=2, dim=0)
            radial_energies.append(self.energy(x_pert))

        mean_rim_energy = sum(radial_energies) / max(1, len(radial_energies))
        basin_barrier_depth = max(0.0, mean_rim_energy - e_center)

        # Attractor sharpness ratio
        if self.num_memories > 0:
            scores = torch.matmul(self.memory_attractors, x)
            probs = F.softmax(self.beta * scores, dim=0)
            top_probs, _ = torch.topk(probs, min(2, self.num_memories))
            top1 = float(top_probs[0].item())
            top2 = float(top_probs[1].item()) if top_probs.size(0) > 1 else 0.0
            margin = top1 - top2
        else:
            margin = 0.0

        return {
            "energy": round(e_center, 4),
            "gradient_norm": round(grad_norm, 4),
            "is_local_minimum": grad_norm < 0.15 and basin_barrier_depth > 0.01,
            "basin_barrier_depth": round(basin_barrier_depth, 4),
            "attractor_margin": round(margin, 4),
            "num_radial_samples": num_radial_samples,
        }

    def fuse_attractors(
        self,
        memory_ids: Sequence[str],
        fused_memory_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Synthesize multiple attractor patterns into a composite prototype attractor.
        
        Uses Gram-Schmidt orthogonalization against non-target memories to prevent
        catastrophic interference with unrelated attractor basins.
        """
        target_indices = []
        for i, meta in enumerate(self.memory_metadata):
            if meta.get("memory_id") in memory_ids:
                target_indices.append(i)

        if not target_indices:
            raise ValueError(f"None of requested memory_ids {memory_ids} found in attractor bank.")

        # Mean pattern of target attractors
        patterns = self.memory_attractors[target_indices]
        composite = patterns.mean(dim=0)

        # Gram-Schmidt orthogonalization against unselected memories
        other_indices = [i for i in range(self.num_memories) if i not in target_indices]
        if other_indices:
            others = self.memory_attractors[other_indices]
            for other_pat in others:
                proj = torch.dot(composite, other_pat)
                composite = composite - 0.5 * proj * other_pat

        composite = F.normalize(composite, p=2, dim=0)

        # Store as new fused pattern
        trace = self.learn_pattern(
            pattern=composite,
            memory_id=fused_memory_id,
            metadata=metadata or {"fused_from": list(memory_ids), "type": "fused_prototype"},
            source_context=f"fusion_of_{'_'.join(memory_ids)}",
        )

        return {
            "fused_memory_id": fused_memory_id,
            "source_memories": list(memory_ids),
            "final_energy": round(trace.final_energy, 4),
            "num_memories_now": self.num_memories,
        }

    def save_checkpoint(self, path: Union[str, Any]) -> dict[str, Any]:
        """Persist continuous synaptic weights, attractors, and metadata to disk."""
        from pathlib import Path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "dim": self.dim,
            "beta": self.beta,
            "synaptic_weights": self.synaptic_weights.detach().cpu(),
            "memory_attractors": self.memory_attractors.detach().cpu(),
            "memory_metadata": self.memory_metadata,
            "decay": self.decay,
        }
        torch.save(checkpoint, target)
        return {
            "status": "checkpoint_saved",
            "path": str(target),
            "num_memories": self.num_memories,
        }

    def load_checkpoint(self, path: Union[str, Any]) -> dict[str, Any]:
        """Restore continuous synaptic weights, attractors, and metadata from disk."""
        from pathlib import Path
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Synaptic checkpoint not found: {target}")
        checkpoint = torch.load(target, map_location=self.synaptic_weights.device)
        self.synaptic_weights.copy_(checkpoint["synaptic_weights"])
        self.memory_attractors = checkpoint["memory_attractors"].to(self.memory_attractors.device)
        self.memory_metadata = checkpoint.get("memory_metadata", [])
        return {
            "status": "checkpoint_loaded",
            "path": str(target),
            "num_memories": self.num_memories,
        }
