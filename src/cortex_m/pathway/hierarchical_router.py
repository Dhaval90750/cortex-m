"""Hierarchical Sparse Router for Scaling Beyond N=8 to N=16 and N=32.

Breaks the N=32 Combinatorial Wall discovered in Old MIMO (M2M-35D collapsed from
96.66% at N=8 down to 0.08% at N=32 because pairwise candidates exploded to 496).

Architecture:
1. Multi-scale Cortical Organization:
   - For N <= 8: Direct competitive sparse routing.
   - For N > 8 (e.g. N=16, 32): Hierarchical routing.
     Level 1 (Local Circuits): Group N inputs into clusters of size 4-8.
     Level 2 (Higher Cortical Areas): Route across cluster summary vectors.
2. Synaptic Pathway Prior:
   Memory attractor pattern provides an initialization bias for router scores,
   converting cold-start combinatorial search into memory-guided warm refinement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mimo.m2m_sparse_route import HIDDEN, K, ROLE_DIM, TAU_MIN, sparsemax


class HierarchicalSparseRouter(nn.Module):
    """Hierarchical sparse routing engine scaling up to N=32 and N=64 inputs."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN,
        role_dim: int = ROLE_DIM,
        cluster_size: int = 8,
        memory_dim: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.role_dim = role_dim
        self.cluster_size = cluster_size
        self.memory_dim = memory_dim

        torch.manual_seed(seed)
        # Node encoder for continuous input + role
        self.node_proj = nn.Sequential(
            nn.Linear(1 + role_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Level 1 Intra-cluster router: scores within each cluster
        self.local_router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Cluster pooling projection
        self.cluster_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Level 2 Inter-cluster router: scores across cluster representatives
        self.inter_router = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Synaptic memory prior projection: memory attractor -> prior bias over inputs
        self.memory_prior_proj = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # M2M-98: Gist-Gated Multi-Scale Modulation (macro-gist -> Level 2 inter-cluster gate)
        self.gist_cluster_gate = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output synaptic transformation
        self.output_synapse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def route_hierarchical(
        self,
        x: torch.Tensor,
        role_ids: torch.Tensor,
        memory_attractor: Optional[torch.Tensor] = None,
        gist: Optional[torch.Tensor] = None,
        n_out: int = 4,
        k_local: int = 2,
        k_global: int = 2,
    ) -> Dict[str, Any]:
        """Hierarchically route N inputs (N=8, 16, 32) into n_out targets with optional gist modulation.

        x: [B, N] or [N]
        role_ids: [B, N] or [N]
        memory_attractor: [memory_dim]
        gist: [32] (M2M-98 macro-gist vector)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if role_ids.ndim == 1:
            role_ids = role_ids.unsqueeze(0)

        B, N = x.shape
        device = x.device

        # One-hot role embeddings
        r_onehot = F.one_hot(role_ids % self.role_dim, num_classes=self.role_dim).float()
        node_in = torch.cat([x.unsqueeze(-1), r_onehot], dim=-1)
        h = self.node_proj(node_in)  # [B, N, hidden_dim]

        # Memory prior bias
        prior_bias = torch.zeros(B, N, device=device)
        if memory_attractor is not None:
            att_flat = memory_attractor.view(-1)
            mem_emb = self.memory_prior_proj(att_flat).unsqueeze(0).expand(B, -1)  # [B, hidden_dim]
            # Prior affinity: dot product between memory embedding and node representation
            prior_bias = (h * mem_emb.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)

        if N <= self.cluster_size:
            # Direct routing for small N
            # Candidate scores per output
            out_tokens = h[:, :n_out] if n_out <= N else h[:, :1].repeat(1, n_out, 1)
            pair_h = h.unsqueeze(1).expand(B, n_out, N, self.hidden_dim)
            pair_out = out_tokens.unsqueeze(2).expand(B, n_out, N, self.hidden_dim)
            scores = self.local_router(torch.cat([pair_h, pair_out], dim=-1)).squeeze(-1)  # [B, n_out, N]
            scores = scores + 0.3 * prior_bias.unsqueeze(1)

            # Sparsemax gating
            score_std = scores.std(dim=-1, keepdim=True).clamp(min=1e-3)
            scaled_scores = (scores - scores.mean(dim=-1, keepdim=True)) / score_std
            gates = k_local * sparsemax(scaled_scores / 0.8, dim=-1)

            # Gated synaptic outputs
            node_vals = self.output_synapse(h).squeeze(-1)  # [B, N]
            y = (gates * node_vals.unsqueeze(1)).sum(dim=-1)

            return {
                "outputs": y.squeeze(0).tolist(),
                "gates": gates.squeeze(0),
                "levels": 1,
                "selected_inputs": gates.squeeze(0).argmax(dim=-1).tolist(),
                "sparsity": float((gates == 0).float().mean().item()),
                "hierarchical": False,
            }

        # Multi-cluster hierarchical routing for N=16, 32, etc.
        num_clusters = math.ceil(N / self.cluster_size)
        cluster_summaries: list[torch.Tensor] = []
        cluster_gates_list: list[torch.Tensor] = []
        cluster_indices_list: list[list[int]] = []

        for c_idx in range(num_clusters):
            start = c_idx * self.cluster_size
            end = min(N, start + self.cluster_size)
            h_cluster = h[:, start:end]  # [B, c_len, hidden_dim]
            c_len = end - start
            cluster_indices_list.append(list(range(start, end)))

            # Level 1: Intra-cluster local routing
            center = h_cluster.mean(dim=1, keepdim=True)
            local_scores = self.local_router(
                torch.cat([h_cluster, center.expand(-1, c_len, -1)], dim=-1)
            ).squeeze(-1)
            local_scores = local_scores + 0.3 * prior_bias[:, start:end]

            l_std = local_scores.std(dim=-1, keepdim=True).clamp(min=1e-3)
            l_scaled = (local_scores - local_scores.mean(dim=-1, keepdim=True)) / l_std
            l_gates = k_local * sparsemax(l_scaled / 0.8, dim=-1)
            cluster_gates_list.append(l_gates)

            # Summarize cluster via gated pool
            summary = (l_gates.unsqueeze(-1) * h_cluster).sum(dim=1)  # [B, hidden_dim]
            cluster_summaries.append(self.cluster_pool(summary))

        # Level 2: Inter-cluster routing across cluster summaries
        summaries_tensor = torch.stack(cluster_summaries, dim=1)  # [B, num_clusters, hidden_dim]
        global_summary = summaries_tensor.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]

        # Route between clusters to produce each target output
        final_outputs: list[float] = []
        inter_gates_list: list[torch.Tensor] = []
        global_selected_inputs: list[int] = []

        for o in range(n_out):
            out_query = summaries_tensor[:, o % num_clusters : o % num_clusters + 1]
            inter_in = torch.cat(
                [summaries_tensor, out_query.expand(-1, num_clusters, -1), global_summary.expand(-1, num_clusters, -1)],
                dim=-1,
            )
            inter_scores = self.inter_router(inter_in).squeeze(-1)  # [B, num_clusters]
            if gist is not None:
                g_flat = gist.view(-1)
                g_emb = self.gist_cluster_gate(g_flat).unsqueeze(0).expand(B, -1)
                g_score = (summaries_tensor * g_emb.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)
                inter_scores = inter_scores + 0.35 * g_score
            i_std = inter_scores.std(dim=-1, keepdim=True).clamp(min=1e-3)
            i_scaled = (inter_scores - inter_scores.mean(dim=-1, keepdim=True)) / i_std
            inter_gates = k_global * sparsemax(i_scaled / 0.8, dim=-1)  # [B, num_clusters]
            inter_gates_list.append(inter_gates)

            best_cluster = int(inter_gates.squeeze(0).argmax().item())
            local_best = int(cluster_gates_list[best_cluster].squeeze(0).argmax().item())
            chosen_global_input = cluster_indices_list[best_cluster][local_best]
            global_selected_inputs.append(chosen_global_input)

            # Synthesize output from winning cluster summary
            val = float(self.output_synapse(summaries_tensor[:, best_cluster]).item())
            final_outputs.append(val)

        return {
            "outputs": [round(v, 4) for v in final_outputs],
            "num_clusters": num_clusters,
            "levels": 2,
            "selected_inputs": global_selected_inputs,
            "cluster_gates": [g.squeeze(0).tolist() for g in cluster_gates_list],
            "inter_gates": [g.squeeze(0).tolist() for g in inter_gates_list],
            "hierarchical": True,
        }
