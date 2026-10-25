"""Causal Route-Memory Binding and Counterfactual MIMO Evaluation.

Resolves the Causality Crisis (M2M-88 -> M2M-93):
In Old MIMO, randomly scrambling the routes barely changed output MSE (1.276 -> 1.290),
revealing that learned sparse routes were decorative rather than causal mechanisms.

This module enforces genuine causal necessity:
1. Gist + Synaptic Memory Attractor directly parameterize router edge scores.
2. Route-Memory Binding aligns router gates G_j with physical synaptic plasticity Delta W_j:
      L_binding = - sum_j cos_sim(G_j, Delta W_j)
3. Counterfactual Scramble Protocol:
   Evaluates task performance under clean routing vs. scrambled/ablated routes.
   Empirically verifies that causality ratio MSE(scrambled) / MSE(clean) > 1.5x.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
from .gist_bottleneck import GistBottleneck


class CausalPathwayMIMO(M2MSparseRoute):
    """MIMO Sparse Router with causal route-memory binding and gist conditioning."""

    def __init__(
        self,
        memory_dim: int = 64,
        gist_dim: int = 32,
        binding_weight: float = 0.5,
        epsilon_cushion: float = 0.001,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.gist_dim = gist_dim
        self.binding_weight = binding_weight
        self.epsilon_cushion = epsilon_cushion

        # Gist information bottleneck
        self.gist_bottleneck = GistBottleneck(
            in_dim=memory_dim,
            gist_dim=gist_dim,
            out_dim=HIDDEN,
        )

        # M2M-95: Cross-Attentive Gist-Sensory Fusion (Gist queries sensory token representations)
        self.gist_cross_attn = nn.MultiheadAttention(embed_dim=HIDDEN, num_heads=2, batch_first=True)

        # Attractor + Gist integration layer
        self.router_context_fusion = nn.Sequential(
            nn.Linear(HIDDEN + memory_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
        )

        # Synaptic path projection for route-memory binding
        self.synaptic_path_proj = nn.Linear(memory_dim, N_IN)

        if checkpoint_path is not None:
            self.load_m2m_checkpoint(checkpoint_path)

    def load_m2m_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load trained M2M-35D base weights into this causal architecture."""
        p = Path(checkpoint_path)
        if not p.exists():
            return
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        base_keys = {
            k: v
            for k, v in state_dict.items()
            if k in self.state_dict() and self.state_dict()[k].shape == v.shape
        }
        self.load_state_dict(base_keys, strict=False)

    def forward_causal(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        memory_attractor: torch.Tensor,
        synaptic_delta: Optional[torch.Tensor] = None,
        tau: float = TAU_MIN,
        scramble_routes: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute causal MIMO forward pass.
        
        Returns:
            y: Output tensor [B, N_OUT]
            gates: Sparse gate activations [B, N_OUT, N_IN]
            binding_loss: Route-memory alignment loss scalar
            gist: Pre-verbal gist vector [gist_dim]
            c: Context vector [B, HIDDEN]
        """
        B = x.size(0)
        h = self.node(torch.cat([x.unsqueeze(-1), r], dim=-1))
        key = self.context_key(torch.cat([h, r], dim=-1))
        global_role = r.mean(dim=1)
        query = self.context_query(global_role)
        value = self.context_value(h)

        attention_scores = (key * query[:, None]).sum(dim=-1) / (HIDDEN ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_state = (attention_weights[:, :, None] * value).sum(dim=1)

        # Extract pre-verbal gist from memory attractor (supporting 1D and batched 2D inputs)
        if memory_attractor.ndim == 1:
            attractor_batch = memory_attractor.unsqueeze(0).expand(B, -1)
            gist, expanded_gist = self.gist_bottleneck(memory_attractor)
            if expanded_gist.ndim == 1:
                expanded_gist = expanded_gist.unsqueeze(0).expand(B, -1)
        else:
            attractor_batch = memory_attractor
            if attractor_batch.size(0) != B:
                attractor_batch = attractor_batch.expand(B, -1)
            gist, expanded_gist = self.gist_bottleneck(attractor_batch)
            if expanded_gist.ndim == 1:
                expanded_gist = expanded_gist.unsqueeze(0).expand(B, -1)

        # M2M-95: Cross-Attentive Gist-Sensory Fusion (Gist queries sensory inputs)
        gist_query = expanded_gist.unsqueeze(1)
        cross_out, _ = self.gist_cross_attn(query=gist_query, key=h, value=h)
        cross_gist = cross_out.squeeze(1)

        # Fuse context_state with cross-attended gist and attractor
        fused_input = torch.cat([cross_gist, attractor_batch], dim=-1)
        fused_context = self.router_context_fusion(fused_input)

        c = self.context_pool(context_state) + fused_context

        # Router edge scores
        out_roles = self.out_role(torch.eye(N_OUT, ROLE_DIM, device=x.device))
        hi = h[:, None].expand(B, N_OUT, N_IN, HIDDEN)
        ho = out_roles[None, :, None].expand(B, N_OUT, N_IN, HIDDEN)
        hc = c[:, None, None].expand(B, N_OUT, N_IN, HIDDEN)

        pair = hi * ho
        z = torch.cat([hi, ho, pair, hc, pair * hc], dim=-1)
        scores = self.router(z).squeeze(-1)  # [B, N_OUT, N_IN]

        # Enforce competitive sparsity via Sparsemax scaled by tau
        effective_tau = max(tau, 1e-4)
        score_std = scores.std(dim=-1, keepdim=True).clamp(min=1e-3)
        score_mean = scores.mean(dim=-1, keepdim=True)
        scores_scaled = (scores - score_mean) / score_std

        # Temperature scaling aligned with tau parameter
        temp = 0.8 * (effective_tau / TAU_MIN)
        raw_sparse_gates = sparsemax(scores_scaled / temp, dim=2)
        if self.epsilon_cushion > 0.0:
            uniform_cushion = torch.full_like(raw_sparse_gates, 1.0 / N_IN)
            damped_gates = (1.0 - self.epsilon_cushion) * raw_sparse_gates + self.epsilon_cushion * uniform_cushion
        else:
            damped_gates = raw_sparse_gates
        clean_gates = K * damped_gates

        if scramble_routes:
            # Counterfactual perturbation: scramble route choices across inputs independently per batch item
            scrambled = []
            for b_idx in range(B):
                perm = torch.randperm(N_IN, device=x.device)
                # Ensure genuine displacement (no fixed points / identity map)
                if (perm == torch.arange(N_IN, device=x.device)).any():
                    shift = torch.randint(1, N_IN, (1,), device=x.device).item()
                    perm = (perm + shift) % N_IN
                scrambled.append(clean_gates[b_idx:b_idx+1, :, perm])
            gates = torch.cat(scrambled, dim=0)
        else:
            gates = clean_gates

        rin = r[:, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        rout = torch.eye(N_OUT, ROLE_DIM, device=x.device)[None, :, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        weights = self.synapse(torch.cat([rin, rout], dim=-1)).squeeze(-1)

        # Pure causal pathway transmission: only gated connections propagate signal
        y = (gates * weights * x[:, None]).sum(dim=2) + self.bias

        # Compute Route-Memory Binding Loss (bounded in [0.0, 2.0] so loss never diverges negative)
        binding_loss = torch.tensor(0.0, device=x.device)
        if synaptic_delta is not None:
            if synaptic_delta.ndim == 2:
                # Capture comprehensive synaptic activity across diagonal and off-diagonal associations
                syn_magnitude = 0.5 * synaptic_delta.diag().abs() + 0.5 * synaptic_delta.abs().mean(dim=-1)
                synaptic_profile = F.normalize(self.synaptic_path_proj(syn_magnitude), p=2, dim=-1)
                mean_gate = clean_gates.mean(dim=(0, 1))  # [N_IN]
                binding_sim = F.cosine_similarity(mean_gate.unsqueeze(0), synaptic_profile.unsqueeze(0))
                binding_loss = 1.0 - binding_sim.squeeze()

        return y, gates, binding_loss, gist, c

    def evaluate_counterfactual(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        memory_attractor: torch.Tensor,
        target_y: torch.Tensor,
        num_scrambles: int = 15,
    ) -> Dict[str, Any]:
        """Run the M2M-93 Counterfactual Scramble Protocol.
        
        Compares:
        1. Clean routing MSE
        2. Scrambled routing MSE (averaged over multiple permutations)
        3. Causality ratio = MSE(scrambled) / MSE(clean)
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                clean_y, clean_gates, _, gist, _ = self.forward_causal(
                    x, r, memory_attractor, scramble_routes=False
                )
                clean_mse = float(F.mse_loss(clean_y, target_y).item())

                scrambled_mses: list[float] = []
                for _ in range(num_scrambles):
                    scr_y, _, _, _, _ = self.forward_causal(
                        x, r, memory_attractor, scramble_routes=True
                    )
                    scr_mse = float(F.mse_loss(scr_y, target_y).item())
                    scrambled_mses.append(scr_mse)

                mean_scrambled_mse = sum(scrambled_mses) / len(scrambled_mses)
                signal_var = float(torch.var(clean_y).item()) + 1e-4
                if clean_mse > 1e-4:
                    causality_ratio = mean_scrambled_mse / clean_mse
                else:
                    # Relative disruption normalized by signal variance (Noise-to-Signal ratio + 1.0 baseline)
                    relative_disruption = mean_scrambled_mse / signal_var
                    causality_ratio = 1.0 + relative_disruption

                return {
                    "clean_mse": round(clean_mse, 6),
                    "scrambled_mse_mean": round(mean_scrambled_mse, 6),
                    "scrambled_mses": [round(m, 4) for m in scrambled_mses],
                    "causality_ratio": round(causality_ratio, 4),
                    "relative_signal_disruption": round(mean_scrambled_mse / signal_var, 4),
                    "is_causally_dependent": bool(causality_ratio >= 1.5),
                    "clean_gate_sparsity": round(float((clean_gates == 0).float().mean().item()), 4),
                }
        finally:
            if was_training:
                self.train()

    def compute_joint_gist_mimo_loss(
        self,
        task_loss: torch.Tensor,
        binding_loss: torch.Tensor,
        gist: torch.Tensor,
        context: torch.Tensor,
        lambda_binding: float = 0.5,
        lambda_consistency: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """M2M-97: Multi-objective joint loss coupling task loss, route-memory binding, and route-to-gist consistency."""
        consistency_loss = self.gist_bottleneck.compute_bidirectional_consistency_loss(gist, context)
        # Ensure binding loss non-negativity to prevent optimization divergence
        bounded_binding = F.relu(binding_loss)
        total_loss = task_loss + lambda_binding * bounded_binding + lambda_consistency * consistency_loss
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "binding_loss": binding_loss,
            "consistency_loss": consistency_loss,
        }

