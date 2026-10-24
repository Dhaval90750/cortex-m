"""MIMO Sparse Routing conditioned on Synaptic Pathway Memory.

Connects the multi-input multi-output relational routing mechanism directly to
the biological synaptic memory substrate:
1. Experiences/memories modify synaptic weights W and register attractors.
2. A task query cue traverses recurrent synaptic pathways to an attractor.
3. The active pathway and attractor modulate the MIMO router's sparse gates G.
4. Output is produced while providing full introspection of both the memory
   traversal path and the physical routing path.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mimo.m2m_sparse_route import (
    HIDDEN,
    K,
    N_IN,
    N_OUT,
    ROLE_DIM,
    TAU_MIN,
    M2MSparseRoute,
    sparsemax,
)
from .encoder import NeuralFeatureEncoder
from .synaptic_memory import SynapticMemoryNetwork
from .tracer import PathwayTrace


class SynapticConditionedMIMO(M2MSparseRoute):
    """M2M Sparse Router modulated by synaptic memory attractor states."""

    def __init__(self, memory_dim: int = 64):
        super().__init__()
        self.memory_dim = memory_dim
        # Projects memory attractor vector into router context dimension
        self.memory_context_proj = nn.Sequential(
            nn.Linear(memory_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )

    def forward_with_memory(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        memory_attractor: torch.Tensor,
        tau: float = TAU_MIN,
    ):
        """Forward pass where memory attractor directly conditions routing context."""
        B = x.size(0)
        h = self.node(torch.cat([x.unsqueeze(-1), r], dim=-1))
        key = self.context_key(torch.cat([h, r], dim=-1))
        global_role = r.mean(dim=1)
        query = self.context_query(global_role)
        value = self.context_value(h)

        attention_scores = (key * query[:, None]).sum(dim=-1) / (HIDDEN ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_state = (attention_weights[:, :, None] * value).sum(dim=1)

        # Modulate router context with the recalled synaptic memory attractor
        mem_proj = self.memory_context_proj(memory_attractor)
        if mem_proj.ndim == 1:
            mem_proj = mem_proj.unsqueeze(0).expand(B, -1)
        c = self.context_pool(context_state) + mem_proj

        context_logits = self.context_classifier(c)
        out_roles = self.out_role(torch.eye(N_OUT, ROLE_DIM, device=x.device))
        hi = h[:, None].expand(B, N_OUT, N_IN, HIDDEN)
        ho = out_roles[None, :, None].expand(B, N_OUT, N_IN, HIDDEN)
        hc = c[:, None, None].expand(B, N_OUT, N_IN, HIDDEN)

        pair = hi * ho
        z = torch.cat([hi, ho, pair, hc, pair * hc], dim=-1)
        scores = self.router(z).squeeze(-1)
        gates = K * sparsemax(scores / tau, dim=2)

        rin = r[:, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        rout = torch.eye(N_OUT, ROLE_DIM, device=x.device)[None, :, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        weights = self.synapse(torch.cat([rin, rout], dim=-1)).squeeze(-1)
        y = (gates * weights * x[:, None]).sum(dim=2) + self.bias

        return y, gates, weights, context_logits, c, attention_weights


class MIMOPathwaySystem:
    """Unified system coordinating Synaptic Memory, Pathway Tracing, and MIMO Routing.

    Operates without hard disk or SQLite:
    - `learn(input_data, metadata)`: Updates synaptic weights, records learning pathway.
    - `recall(cue)`: Traverses synaptic connections to attractor, records recall pathway.
    - `route(task_cue, values, role_ids)`: Recalls memory attractor, sets MIMO pathways, computes output.
    """

    def __init__(
        self,
        dim: int = 64,
        mimo_model: Optional[SynapticConditionedMIMO] = None,
        device: str = "cpu",
    ):
        self.dim = dim
        self.device = device
        self.encoder = NeuralFeatureEncoder(output_dim=dim).to(device)
        self.memory = SynapticMemoryNetwork(dim=dim).to(device)
        self.mimo = (mimo_model or SynapticConditionedMIMO(memory_dim=dim)).to(device)
        self.mimo.eval()

    def learn(
        self,
        experience: Union[str, torch.Tensor],
        memory_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Learn an experience by updating synaptic weights and storing training path trace."""
        if isinstance(experience, str):
            vec = self.encoder.encode_text(experience)
            meta = metadata or {"raw_text": experience}
        else:
            vec = self.encoder.encode_numeric(experience).squeeze(0)
            meta = metadata or {"raw_tensor": experience.tolist()}

        trace = self.memory.learn_pattern(vec, memory_id=memory_id, metadata=meta)
        return {
            "status": "learned_in_synapses",
            "memory_id": trace.trace_id,
            "trace_summary": trace.summary(),
            "synaptic_stats": self.memory.inspect_synapses(),
        }

    def recall(self, cue: Union[str, torch.Tensor]) -> dict[str, Any]:
        """Recall an attractor state from a partial or noisy cue via pathway traversal."""
        if isinstance(cue, str):
            cue_vec = self.encoder.encode_text(cue)
        else:
            cue_vec = self.encoder.encode_numeric(cue).squeeze(0)

        attractor, trace, diagnostics = self.memory.recall(cue_vec)
        return {
            "recalled_attractor_norm": float(attractor.norm().item()),
            "trace_summary": trace.summary(),
            "diagnostics": diagnostics,
        }

    def route_task(
        self,
        task_cue: Union[str, torch.Tensor],
        values: Sequence[float],
        role_ids: Sequence[int],
    ) -> dict[str, Any]:
        """Execute a MIMO task conditioned on synaptic memory recall and pathway trace."""
        # 1. Recall memory attractor using task cue
        if isinstance(task_cue, str):
            cue_vec = self.encoder.encode_text(task_cue)
        else:
            cue_vec = self.encoder.encode_numeric(torch.as_tensor(task_cue, dtype=torch.float32)).squeeze(0)

        attractor, recall_trace, diagnostics = self.memory.recall(cue_vec)

        # 2. Format MIMO inputs
        x = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        roles = torch.as_tensor(role_ids, dtype=torch.long, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if roles.ndim == 1:
            roles = roles.unsqueeze(0)

        if x.shape[1] != N_IN:
            raise ValueError(f"Values must have {N_IN} inputs, got {x.shape[1]}")
        r = F.one_hot(roles, ROLE_DIM).to(dtype=x.dtype)

        # 3. Forward pass through MIMO modulated by attractor
        with torch.inference_mode():
            outputs, gates, weights, context_logits, c, attn = self.mimo.forward_with_memory(
                x, r, attractor.unsqueeze(0), tau=TAU_MIN
            )
            selected_pos = gates.topk(K, dim=2).indices

        return {
            "mimo_outputs": outputs.cpu().tolist(),
            "selected_physical_positions": selected_pos.cpu().tolist(),
            "gates": gates.cpu().tolist(),
            "memory_traversal": {
                "steps_taken": diagnostics["steps_taken"],
                "attractor_similarity": diagnostics["attractor_similarity"],
                "best_memory_id": diagnostics["best_memory_id"],
                "resonance": diagnostics["resonance"],
                "pathway_trace": recall_trace.summary(),
            },
            "synaptic_stats": self.memory.inspect_synapses(),
        }
