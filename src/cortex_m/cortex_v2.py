"""Cortex-M v2: Unified Neural Pathway Architecture.

Unifies:
1. Hybrid Semantic Encoding (Hypersphere representations).
2. Continuous Modern Hopfield Synaptic Memory with Hebbian plasticity and Anti-Hebbian unlearning.
3. Neural Pathway Tracing, Resonance, and Episodic Source Attribution.
4. Learned Pre-Verbal Gist Information Bottleneck.
5. Causal MIMO Sparse Routing with Route-Memory Binding.
6. Hierarchical Sparse Routing scaling to N=16 and N=32.
7. Biological Sleep Consolidation & Synaptic Homeostasis.
8. Working Memory Persistent Activation Buffer for Multi-Turn Threading.

No SQLite. No external database. Zero cloud dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .pathway.causal_mimo import CausalPathwayMIMO
from .pathway.consolidation import SynapticConsolidator
from .pathway.dual_stream import DualStreamProcessor
from .pathway.encoder import NeuralFeatureEncoder
from .pathway.gist_bottleneck import GistBottleneck
from .pathway.gist_generator import GistRealizer
from .pathway.hierarchical_router import HierarchicalSparseRouter
from .pathway.synaptic_memory import SynapticMemoryNetwork
from .pathway.tenant_paging import SynapticTenantManager
from .pathway.tracer import PathwayTrace, PathwayTracer
from .pathway.working_memory import WorkingMemoryBuffer
from .security.critic import NeuromodulatorySafetyCritic
from .security.evidence import EvidenceStatus, EvidenceVerificationReport, EvidenceVerifier
from .skills.registry import SkillRegistry


class CortexV2:
    """Unified entry point for Cortex-M neural memory and routing."""

    def __init__(
        self,
        dim: int = 64,
        gist_dim: int = 32,
        beta: float = 8.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.dim = dim
        self.gist_dim = gist_dim
        self.device = device
        self.seed = seed

        torch.manual_seed(seed)
        self.encoder = NeuralFeatureEncoder(output_dim=dim, seed=seed).to(device)
        self.memory = SynapticMemoryNetwork(dim=dim, beta=beta).to(device)
        self.causal_mimo = CausalPathwayMIMO(
            memory_dim=dim,
            gist_dim=gist_dim,
        ).to(device)
        self.causal_mimo.eval()

        # Scaling: Hierarchical router for N=16, 32
        self.hierarchical_router = HierarchicalSparseRouter(
            memory_dim=dim,
            seed=seed,
        ).to(device)
        self.hierarchical_router.eval()

        # Biological sleep consolidation & homeostasis with quarantine
        self.consolidator = SynapticConsolidator(self.memory)

        # 3-Tier Prefrontal working memory buffer
        self.working_memory = WorkingMemoryBuffer(dim=dim)

        # Pre-verbal gist realizer
        self.realizer = GistRealizer()

        # Hardened Security Critic (Plasticity Write-Lock & Invariant Protection)
        self.safety_critic = NeuromodulatorySafetyCritic(input_dim=dim, seed=seed).to(device)

        # Multi-Tenant Synaptic Delta Isolation (Zero Leakage, 2 KB per tenant)
        self.tenant_manager = SynapticTenantManager(base_weights=self.memory.synaptic_weights)

        # Dual-Stream Dorsal/Ventral Processor (Code/Math literals + 128-D Factored Gist)
        self.dual_stream = DualStreamProcessor(dim=dim, head_dim=32, seed=seed).to(device)

        # Modular Cognitive & Engineering Skills Engine
        self.skills = SkillRegistry(cortex=self)

        # Structured Evidence Verification Pipeline
        self.evidence_verifier = EvidenceVerifier()

    def execute_skill(
        self,
        skill_name: str,
        target: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a specialized cognitive or software engineering skill."""
        return self.skills.execute(skill_name, target, context)

    def learn(
        self,
        text_or_vector: Union[str, Sequence[float], torch.Tensor],
        memory_id: Optional[str] = None,
        source_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        salience: float = 1.0,
        learning_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Learn a new memory by modifying physical synaptic connection weights.

        Updates Delta W = salience * eta * (z z^T), registers attractor basin,
        and records the complete neural activation trajectory.
        """
        if isinstance(text_or_vector, str):
            encoded = self.encoder.encode_text(text_or_vector)
            text_str = text_or_vector
        elif isinstance(text_or_vector, torch.Tensor):
            encoded = self.encoder.encode_numeric(text_or_vector).squeeze(0)
            text_str = None
        else:
            t = torch.tensor(text_or_vector, dtype=torch.float32, device=self.device)
            encoded = self.encoder.encode_numeric(t).squeeze(0)
            text_str = None

        # Neuromodulatory Safety Critic evaluation
        permitted, critic_score, reason = self.safety_critic.evaluate_plasticity_permission(
            raw_text=text_str,
            embedding=encoded,
            target_memory_id=memory_id,
            is_unlearning=False,
        )
        if not permitted:
            return {
                "status": "blocked_by_safety_critic",
                "memory_id": memory_id,
                "reason": reason,
                "safety_score": critic_score,
                "synaptic_delta_norm": 0.0,
            }

        meta = metadata.copy() if metadata else {}
        meta["salience"] = float(salience)
        meta["safety_score"] = float(critic_score)
        if text_str:
            meta["raw_text"] = text_str

        effective_lr = learning_rate * salience * critic_score
        trace = self.memory.learn_pattern(
            pattern=encoded,
            memory_id=memory_id,
            metadata=meta,
            source_context=source_context,
            learning_rate=effective_lr,
        )

        # Prime working memory with new experience
        self.working_memory.update(encoded, label=memory_id or text_str)

        return {
            "status": "learned_in_synapses",
            "memory_id": trace.trace_id,
            "source_context": source_context,
            "salience": salience,
            "safety_score": critic_score,
            "synaptic_delta_norm": trace.synaptic_delta_norm,
            "initial_energy": trace.initial_energy,
            "final_energy": trace.final_energy,
            "total_memories": self.memory.num_memories,
        }

    def recall(
        self,
        cue: Union[str, Sequence[float], torch.Tensor],
        query_id: Optional[str] = None,
        use_working_memory: bool = False,
    ) -> Dict[str, Any]:
        """Associative recall through recurrent energy descent in synaptic space."""
        if isinstance(cue, str):
            cue_vector = self.encoder.encode_text(cue)
            cue_text = cue
        elif isinstance(cue, torch.Tensor):
            cue_vector = self.encoder.encode_numeric(cue).squeeze(0)
            cue_text = None
        else:
            t = torch.tensor(cue, dtype=torch.float32, device=self.device)
            cue_vector = self.encoder.encode_numeric(t).squeeze(0)
            cue_text = None

        if use_working_memory:
            effective_cue = self.working_memory.prepare_contextual_cue(cue_vector)
        else:
            effective_cue = cue_vector

        recalled_state, trace, diag = self.memory.recall(effective_cue, query_id=query_id)

        # Update working memory with recalled state
        if use_working_memory:
            self.working_memory.update(recalled_state, label=cue_text or "recall")

        # Extract pre-verbal gist from recalled attractor
        gist, _ = self.causal_mimo.gist_bottleneck(recalled_state)
        gist_info = self.causal_mimo.gist_bottleneck.inspect_gist(gist)

        # Formulate verbal realization
        realized = self.realizer.formulate_response(
            gist=gist,
            recalled_metadata=diag["best_metadata"],
            resonance=diag["resonance"],
            source_attribution=diag["source_attribution"],
        )

        evidence = self.evidence_verifier.verify(
            query=cue_text or "vector_query",
            best_metadata=diag["best_metadata"],
            attractor_similarity=diag.get("cue_similarity", diag["attractor_similarity"]),
            resonance=diag["resonance"],
            steps_taken=diag["steps_taken"],
            energy_drop=diag["energy_drop"],
            final_energy=trace.final_energy if hasattr(trace, "final_energy") else 0.0,
        )

        return {
            "recalled_state": recalled_state,
            "cue_text": cue_text,
            "best_memory_id": diag["best_memory_id"],
            "best_metadata": diag["best_metadata"],
            "attractor_similarity": diag["attractor_similarity"],
            "cue_similarity": diag.get("cue_similarity", diag["attractor_similarity"]),
            "attractor_displacement": diag.get("attractor_displacement", 0.0),
            "resonance": diag["resonance"],
            "source_attribution": diag["source_attribution"],
            "gist": gist,
            "gist_diagnostics": gist_info,
            "verbal_response": realized["verbal_response"],
            "energy_drop": diag["energy_drop"],
            "steps_taken": diag["steps_taken"],
            "trace_summary": trace.summary(),
            "evidence_report": evidence.to_dict(),
            "evidence_status": evidence.status.value,
        }

    def forget(
        self,
        target: Union[str, torch.Tensor],
        learning_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Anti-Hebbian targeted forgetting (synaptic depression)."""
        target_id = None
        if isinstance(target, str):
            if not target.startswith("mem_"):
                for meta in self.memory.memory_metadata:
                    if meta.get("raw_text") == target:
                        target = meta.get("memory_id")
                        break
            target_id = target

        # Safety Critic invariant protection (prevent unlearning protected guardrails)
        permitted, critic_score, reason = self.safety_critic.evaluate_plasticity_permission(
            raw_text=None,
            embedding=torch.zeros(self.dim, device=self.device),
            target_memory_id=target_id,
            is_unlearning=True,
        )
        if not permitted:
            return {
                "status": "blocked_by_safety_critic",
                "memory_id": target_id,
                "reason": reason,
                "safety_score": 0.0,
            }

        return self.memory.forget_pattern(target, learning_rate=learning_rate)

    def route(
        self,
        cue: Union[str, torch.Tensor],
        values: Sequence[float],
        role_ids: Sequence[int],
        scramble_routes: bool = False,
    ) -> Dict[str, Any]:
        """Gist-conditioned Causal MIMO Routing for N=8 inputs."""
        recall_res = self.recall(cue)
        attractor = recall_res["recalled_state"]
        matched_mem_id = recall_res["best_memory_id"]

        synaptic_delta = None
        if matched_mem_id and matched_mem_id in self.memory.episodic_traces:
            synaptic_delta = self.memory.episodic_traces[matched_mem_id].synaptic_delta

        x = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)
        r = F.one_hot(torch.tensor(role_ids, dtype=torch.long, device=self.device), num_classes=8).float().unsqueeze(0)

        with torch.no_grad():
            y, gates, binding_loss, gist, _ = self.causal_mimo.forward_causal(
                x=x,
                r=r,
                memory_attractor=attractor,
                synaptic_delta=synaptic_delta,
                scramble_routes=scramble_routes,
            )

        gate_indices = gates.squeeze(0).argmax(dim=-1).tolist()
        return {
            "kind": "causal_pathway_mimo",
            "outputs": [round(float(v), 4) for v in y.squeeze(0).tolist()],
            "selected_routes": gate_indices,
            "gate_sparsity": round(float((gates < 0.01).float().mean().item()), 4),
            "recalled_memory_id": matched_mem_id,
            "pathway_resonance": recall_res["resonance"],
            "pre_verbal_gist_norm": round(float(gist.norm().item()), 4),
            "is_scrambled": scramble_routes,
        }

    def route_hierarchical(
        self,
        cue: Union[str, torch.Tensor],
        values: Sequence[float],
        role_ids: Sequence[int],
        n_out: int = 4,
    ) -> Dict[str, Any]:
        """Hierarchical Sparse Routing scaling to N=16 and N=32 inputs."""
        recall_res = self.recall(cue)
        attractor = recall_res["recalled_state"]

        x = torch.tensor(values, dtype=torch.float32, device=self.device)
        r = torch.tensor(role_ids, dtype=torch.long, device=self.device)

        with torch.no_grad():
            result = self.hierarchical_router.route_hierarchical(
                x=x,
                role_ids=r,
                memory_attractor=attractor,
                gist=recall_res.get("gist"),
                n_out=n_out,
            )

        result["recalled_memory_id"] = recall_res["best_memory_id"]
        result["pathway_resonance"] = recall_res["resonance"]
        return result

    def sleep_consolidation(
        self,
        replay_iterations: int = 3,
        noise_level: float = 0.15,
        decay_factor: float = 0.92,
    ) -> Dict[str, Any]:
        """Run homeostatic sleep replay: reinforce memories, prune weak weights, renormalize budget."""
        return self.consolidator.sleep_cycle(
            replay_iterations=replay_iterations,
            noise_level=noise_level,
            decay_factor=decay_factor,
        )

    def multi_turn_interact(self, message: str) -> Dict[str, Any]:
        """Multi-turn interaction using prefrontal working memory context."""
        return self.recall(message, use_working_memory=True)

    def inspect_synapses(self) -> Dict[str, Any]:
        """Inspect internal synaptic weights and connectivity."""
        return self.memory.inspect_synapses()

    def compare_pathways(self, learn_id: str, recall_id: str) -> Dict[str, Any]:
        """Compare a learning trace against a recall trace for resonance."""
        l_trace = self.memory.episodic_traces.get(learn_id)
        if not l_trace:
            raise KeyError(f"No learning trace found with id {learn_id}")
        r_trace = next((t for t in self.memory.tracer.history if t.trace_id == recall_id), None)
        if not r_trace:
            raise KeyError(f"No recall trace found with id {recall_id}")
        return PathwayTracer.compute_resonance(l_trace, r_trace)

    def learn_for_tenant(
        self,
        tenant_id: str,
        text_or_vector: Union[str, Sequence[float], torch.Tensor],
        memory_id: Optional[str] = None,
        lr: float = 0.5,
    ) -> Dict[str, Any]:
        """Learn isolated memory strictly within tenant's private 2 KB low-rank adapter."""
        target_mem_id = memory_id or f"tenant_{tenant_id}_{len(self.tenant_manager.get_or_create_tenant(tenant_id).stored_patterns)}"
        if isinstance(text_or_vector, str):
            encoded = self.encoder.encode_text(text_or_vector)
            meta = {
                "raw_text": text_or_vector,
                "memory_id": target_mem_id,
                "source_context": f"tenant_{tenant_id}",
            }
        elif isinstance(text_or_vector, torch.Tensor):
            encoded = self.encoder.encode_numeric(text_or_vector).squeeze(0)
            meta = {"memory_id": target_mem_id}
        else:
            t = torch.tensor(text_or_vector, dtype=torch.float32, device=self.device)
            encoded = self.encoder.encode_numeric(t).squeeze(0)
            meta = {"memory_id": target_mem_id}

        return self.tenant_manager.learn_tenant_memory(tenant_id, encoded, lr=lr, metadata=meta)

    def recall_for_tenant(
        self,
        tenant_id: str,
        cue: Union[str, Sequence[float], torch.Tensor],
    ) -> Dict[str, Any]:
        """Recall with tenant-isolated effective synaptic weights (zero leakage)."""
        if isinstance(cue, str):
            cue_vector = self.encoder.encode_text(cue)
        elif isinstance(cue, torch.Tensor):
            cue_vector = self.encoder.encode_numeric(cue).squeeze(0)
        else:
            t = torch.tensor(cue, dtype=torch.float32, device=self.device)
            cue_vector = self.encoder.encode_numeric(t).squeeze(0)

        tenant = self.tenant_manager.get_or_create_tenant(tenant_id)
        effective_w = self.tenant_manager.get_effective_weights(tenant_id)

        if self.memory.num_memories == 0 and not tenant.stored_patterns:
            return {
                "tenant_id": tenant_id,
                "recalled_state": cue_vector,
                "best_memory_id": None,
                "attractor_similarity": 0.0,
                "best_metadata": {},
                "steps_taken": 0,
                "energy_drop": 0.0,
                "resonance": None,
                "source_attribution": {"dominant_source": f"tenant_{tenant_id}"},
            }

        t_stack = torch.stack(tenant.stored_patterns, dim=0).to(self.device) if tenant.stored_patterns else None
        t_meta = list(tenant.stored_metadata) if tenant.stored_metadata else None

        recalled, trace, diag = self.memory.recall(
            cue_vector,
            synaptic_override=effective_w,
            extra_attractors=t_stack,
            extra_metadata=t_meta,
        )

        return {
            "tenant_id": tenant_id,
            "recalled_state": recalled,
            "best_memory_id": diag["best_memory_id"],
            "attractor_similarity": diag["attractor_similarity"],
            "best_metadata": diag["best_metadata"],
            "steps_taken": diag["steps_taken"],
            "energy_drop": diag["energy_drop"],
            "resonance": diag["resonance"],
            "source_attribution": diag["source_attribution"],
        }

    def process_dual_stream(self, text_or_code: str) -> Dict[str, Any]:
        """Process input through Ventral (Factored Gist) + Dorsal (Verbatim Code/Math) streams."""
        encoded = self.encoder.encode_text(text_or_code)
        return self.dual_stream.process(text_or_code, encoded)

    def compress_synapses(self, target_rank: int = 16) -> Dict[str, Any]:
        """Compress synaptic weight matrix W via truncated Singular Value Decomposition (SVD).
        
        Preserves attractor energy basins while achieving substantial memory reduction.
        W approx U_k * Sigma_k * V_k^T
        """
        W = self.memory.synaptic_weights.detach()
        dim = W.shape[0]
        k = min(target_rank, dim)

        # Compute full SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Truncate to top-k singular values
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]

        # Reconstructed low-rank synaptic matrix
        W_approx = torch.matmul(U_k * S_k, Vh_k)
        # Ensure zero diagonal and exact symmetry
        W_approx = 0.5 * (W_approx + W_approx.t())
        W_approx.fill_diagonal_(0.0)

        # Compute reconstruction error (Frobenius norm)
        orig_norm = float(torch.linalg.matrix_norm(W).item())
        err_norm = float(torch.linalg.matrix_norm(W - W_approx).item())
        relative_error = err_norm / max(1e-8, orig_norm)
        fidelity = max(0.0, 1.0 - relative_error)

        # Apply compressed matrix
        self.memory.synaptic_weights.copy_(W_approx)

        # Compression ratio
        raw_params = dim * dim
        compressed_params = (2 * dim * k) + k
        compression_ratio = raw_params / max(1, compressed_params)

        return {
            "rank": k,
            "original_params": raw_params,
            "compressed_params": compressed_params,
            "compression_ratio": round(compression_ratio, 2),
            "frobenius_relative_error": round(relative_error, 4),
            "attractor_fidelity_retention": round(fidelity * 100.0, 2),
        }

    def counterfactual_audit(
        self,
        cue: Union[str, Sequence[float], torch.Tensor],
        num_perturbations: int = 5,
        perturbation_epsilon: float = 0.08,
    ) -> Dict[str, Any]:
        """Perform adversarial/counterfactual stress test on an attractor basin.
        
        Injects spherical Gaussian noise to evaluate whether a fact is anchored in a 
        stable Lyapunov basin or is an ungrounded hallucination boundary.
        """
        if isinstance(cue, str):
            base_vec = self.encoder.encode_text(cue)
        elif isinstance(cue, torch.Tensor):
            base_vec = self.encoder.encode_numeric(cue).squeeze(0)
        else:
            base_vec = self.encoder.encode_numeric(torch.tensor(cue, dtype=torch.float32, device=self.device)).squeeze(0)

        base_recalled, _, base_diag = self.memory.recall(base_vec)
        base_id = base_diag.get("best_memory_id")
        base_sim = base_diag.get("attractor_similarity", 0.0)

        basin_hits = 0
        perturbed_similarities = []

        torch.manual_seed(12345)
        for _ in range(num_perturbations):
            noise = torch.randn_like(base_vec)
            noise = (noise / torch.norm(noise).clamp(min=1e-8)) * perturbation_epsilon
            pert_vec = F.normalize(base_vec + noise, p=2, dim=-1)

            recalled_p, _, diag_p = self.memory.recall(pert_vec)
            p_id = diag_p.get("best_memory_id")
            p_sim = diag_p.get("attractor_similarity", 0.0)
            perturbed_similarities.append(p_sim)

            if p_id == base_id and p_sim >= (base_sim * 0.85):
                basin_hits += 1

        basin_stability = basin_hits / max(1, num_perturbations)
        is_grounded = basin_stability >= 0.80 and base_sim >= 0.70

        return {
            "base_memory_id": base_id,
            "base_similarity": round(base_sim, 4),
            "basin_stability_score": round(basin_stability, 3),
            "is_lyapunov_grounded": is_grounded,
            "num_perturbations_tested": num_perturbations,
            "mean_perturbed_similarity": round(float(sum(perturbed_similarities) / len(perturbed_similarities)), 4) if perturbed_similarities else 0.0,
            "epistemic_verdict": "VERIFIED_ATTRACTOR_BASIN" if is_grounded else "FRAGILE_OR_HYPOTHETICAL",
        }

    def adaptive_recall(
        self,
        cue: Union[str, Sequence[float], torch.Tensor],
        adaptive_temperature: bool = True,
    ) -> Dict[str, Any]:
        """Execute attractor recall with dynamic inverse temperature scaling."""
        if isinstance(cue, str):
            cue_vector = self.encoder.encode_text(cue)
        elif isinstance(cue, torch.Tensor):
            cue_vector = self.encoder.encode_numeric(cue).squeeze(0)
        else:
            cue_vector = self.encoder.encode_numeric(torch.tensor(cue, dtype=torch.float32, device=self.device)).squeeze(0)

        # Dynamically scale beta: higher for sharp retrieval, lower for fuzzy association
        orig_beta = self.memory.beta
        if adaptive_temperature:
            energy = self.memory.energy(cue_vector)
            # If cue has low energy (well-settled), sharpen beta for exact retrieval
            if energy < -1.0:
                self.memory.beta = min(32.0, orig_beta * 1.5)
            else:
                self.memory.beta = max(2.0, orig_beta * 0.75)

        try:
            recalled, trace, diag = self.memory.recall(cue_vector)
            diag["effective_beta"] = self.memory.beta
            return {
                "recalled_state": recalled,
                "diagnostics": diag,
            }
        finally:
            self.memory.beta = orig_beta

    def batch_recall(
        self,
        cues: Sequence[Union[str, Sequence[float], torch.Tensor]],
    ) -> List[Dict[str, Any]]:
        """Vectorized parallel Modern Hopfield attractor convergence for multiple queries.
        
        Accelerates bulk queries and real-time streaming recall with zero Python loop overhead.
        """
        if not cues:
            return []

        # Encode all queries into a single tensor batch [B, dim]
        encoded_tensors = []
        for c in cues:
            if isinstance(c, str):
                vec = self.encoder.encode_text(c)
            elif isinstance(c, torch.Tensor):
                vec = self.encoder.encode_numeric(c).squeeze(0)
            else:
                vec = self.encoder.encode_numeric(torch.tensor(c, dtype=torch.float32, device=self.device)).squeeze(0)
            encoded_tensors.append(vec.view(1, -1))

        batch_tensor = torch.cat(encoded_tensors, dim=0)
        _, _, diag_list = self.memory.batch_recall(batch_tensor)
        return diag_list

    def map_energy_landscape(
        self,
        cue: Union[str, Sequence[float], torch.Tensor],
        num_radial_samples: int = 8,
        radius: float = 0.1,
    ) -> Dict[str, Any]:
        """Perform comprehensive Lyapunov landscape analysis around a neural query state.
        
        Evaluates local gradient norm, basin depth barrier, and attractor stability margin.
        """
        if isinstance(cue, str):
            vec = self.encoder.encode_text(cue)
        elif isinstance(cue, torch.Tensor):
            vec = self.encoder.encode_numeric(cue).squeeze(0)
        else:
            vec = self.encoder.encode_numeric(torch.tensor(cue, dtype=torch.float32, device=self.device)).squeeze(0)

        landscape = self.memory.compute_energy_landscape(
            vec,
            num_radial_samples=num_radial_samples,
            radius=radius,
        )
        return landscape

    def associative_reasoning_chain(
        self,
        start_cue: Union[str, Sequence[float], torch.Tensor],
        max_hops: int = 3,
        perturbation_drift: float = 0.25,
    ) -> Dict[str, Any]:
        """Perform multi-hop continuous attractor traversal (cognitive reasoning chain).
        
        Follows semantic gradients from one attractor basin to resonant adjacent basins,
        simulating biological train-of-thought progression without prompt engineering.
        """
        chain_steps = []
        visited_ids = set()
        visited_attractors = []

        current_cue = start_cue
        for hop in range(max_hops):
            recall_res = self.adaptive_recall(current_cue)
            diag = recall_res["diagnostics"]
            mem_id = diag.get("best_memory_id")
            attractor_sim = diag.get("attractor_similarity", 0.0)

            if not mem_id or mem_id in visited_ids or attractor_sim < 0.35:
                # Loop detected, empty memory, or out-of-distribution boundary
                break

            step_info = {
                "hop": hop + 1,
                "attractor_id": mem_id,
                "attractor_similarity": attractor_sim,
                "energy_drop": diag.get("energy_drop", 0.0),
                "metadata": diag.get("best_metadata", {}),
            }
            chain_steps.append(step_info)
            visited_ids.add(mem_id)

            # Evolve state: take recalled attractor, project out all previously visited attractors
            # to force transition into adjacent resonant basin
            state = recall_res["recalled_state"].clone()
            visited_attractors.append(state)

            next_state = state
            for prev_attractor in visited_attractors:
                proj = torch.dot(next_state, prev_attractor)
                next_state = next_state - proj * prev_attractor

            # Add lateral exploration perturbation
            noise = torch.randn_like(state)
            next_state = F.normalize(next_state + perturbation_drift * F.normalize(noise, dim=0), dim=0)
            current_cue = next_state

        return {
            "num_hops_completed": len(chain_steps),
            "reasoning_path": [s["attractor_id"] for s in chain_steps if s["attractor_id"]],
            "hops": chain_steps,
        }

    def fuse_memories(
        self,
        memory_ids: Sequence[str],
        fused_memory_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize multiple attractor patterns into an integrated concept prototype."""
        return self.memory.fuse_attractors(memory_ids, fused_memory_id, metadata=metadata)

    def save_synapses(self, filepath: str) -> Dict[str, Any]:
        """Save synaptic network state to disk for persistent zero-DB recall."""
        return self.memory.save_checkpoint(filepath)

    def load_synapses(self, filepath: str) -> Dict[str, Any]:
        """Load synaptic network state from disk."""
        return self.memory.load_checkpoint(filepath)

