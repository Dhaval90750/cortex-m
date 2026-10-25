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
        lr: float = 0.5,
    ) -> Dict[str, Any]:
        """Learn isolated memory strictly within tenant's private 2 KB low-rank adapter."""
        if isinstance(text_or_vector, str):
            encoded = self.encoder.encode_text(text_or_vector)
            meta = {
                "raw_text": text_or_vector,
                "memory_id": f"tenant_{tenant_id}_{len(self.tenant_manager.get_or_create_tenant(tenant_id).stored_patterns)}",
                "source_context": f"tenant_{tenant_id}",
            }
        elif isinstance(text_or_vector, torch.Tensor):
            encoded = self.encoder.encode_numeric(text_or_vector).squeeze(0)
            meta = {"memory_id": f"tenant_{tenant_id}_{len(self.tenant_manager.get_or_create_tenant(tenant_id).stored_patterns)}"}
        else:
            t = torch.tensor(text_or_vector, dtype=torch.float32, device=self.device)
            encoded = self.encoder.encode_numeric(t).squeeze(0)
            meta = {"memory_id": f"tenant_{tenant_id}_{len(self.tenant_manager.get_or_create_tenant(tenant_id).stored_patterns)}"}

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

    def save_synapses(self, filepath: str) -> Dict[str, Any]:
        """Save synaptic network state to disk for persistent zero-DB recall."""
        return self.memory.save_checkpoint(filepath)

    def load_synapses(self, filepath: str) -> Dict[str, Any]:
        """Load synaptic network state from disk."""
        return self.memory.load_checkpoint(filepath)
