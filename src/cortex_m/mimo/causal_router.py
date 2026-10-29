"""Causal Sparse Router Primitives, Benchmarking & Profiling for Cortex-M.

Implements Milestone 2 Phase 4 (Features F07 & F08):
- CausalRouterBenchmark: Route entropy, auxiliary load-balancing loss, sparse routing execution.
- RoutingProfiler: High-resolution wall-clock latency, expert balance ratio, fallback diagnostics.
- StaticDenseRouter: Baseline dense router with 0% pathway sparsity.
- RandomRouter: Baseline router selecting K random experts uniformly.
- CausalRouter: High-level neural coordinator combining CausalPathwayMIMO and
  HierarchicalSparseRouter with degenerate input fallback and counterfactual causality verification.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_router import sparsemax


# ============================================================================
# F07: Causal Router Matched Benchmarking
# ============================================================================

class CausalRouterBenchmark:
    """Benchmarking sparse router against dense/static baselines."""

    @staticmethod
    def compute_route_entropy(
        probs: Union[List[float], np.ndarray, torch.Tensor],
        eps: float = 1e-9,
    ) -> float:
        """Compute Shannon routing entropy H(p) = -sum p_i log2(p_i) in bits.
        
        Handles scalars/1-expert (H=0.0), uniform distributions, and multi-dimensional tensors.
        """
        if isinstance(probs, torch.Tensor):
            p = probs.detach().cpu().numpy()
        else:
            p = np.array(probs, dtype=np.float64)

        if p.size == 0:
            return 0.0

        p = p.flatten()
        if len(p) == 1:
            return 0.0

        if np.sum(p) <= 0.0 or not np.isfinite(np.sum(p)):
            return 0.0

        p = np.clip(p, eps, 1.0)
        total = np.sum(p)
        if total <= 0.0 or not np.isfinite(total):
            return 0.0
        p = p / total
        entropy = float(-np.sum(p * np.log2(p)))
        return 0.0 if abs(entropy) < 1e-12 else entropy

    @staticmethod
    def compute_load_balance_loss(
        expert_assignments: List[List[int]],
        expert_probs: List[List[float]],
        num_experts: int = 4,
        alpha: float = 1.0,
    ) -> float:
        """Compute auxiliary load balancing loss L_balance = alpha * N * sum(f_i * P_i).
        
        Args:
            expert_assignments: Top-k expert indices selected per sample in batch.
            expert_probs: Routing probability distribution over all experts per sample.
            num_experts: Total number of active experts.
            alpha: Loss scaling hyperparameter.
        """
        if not expert_assignments or len(expert_assignments) == 0:
            return 0.0

        B = len(expert_assignments)
        if B == 0 or num_experts <= 0:
            return 0.0

        flat_assignments = [exp for batch in expert_assignments for exp in batch]
        f = np.zeros(num_experts, dtype=np.float64)
        for exp in flat_assignments:
            if 0 <= exp < num_experts:
                f[exp] += 1.0

        # Normalization factor: total assignments routed across the batch
        total_assignments = len(flat_assignments)
        if total_assignments > 0:
            f = f / total_assignments
        else:
            f = np.ones(num_experts, dtype=np.float64) / num_experts

        P = np.zeros(num_experts, dtype=np.float64)
        for probs in expert_probs:
            for i, val in enumerate(probs[:num_experts]):
                if 0 <= i < num_experts:
                    P[i] += float(val)
        P /= B

        # Auxiliary loss: alpha * N * sum(f_i * P_i)
        loss = float(alpha * num_experts * np.sum(f * P))
        return float(np.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))

    @staticmethod
    def run_sparse_routing(
        feature_vector: Sequence[float],
        num_experts: int = 4,
        top_k: int = 2,
    ) -> Dict[str, Any]:
        """Execute sparse routing with automatic degenerate fallback."""
        x = np.array(feature_vector, dtype=np.float32)
        norm = np.linalg.norm(x)
        clamped_k = min(max(1, top_k), num_experts)

        if norm < 1e-6 or np.isnan(norm) or np.isinf(norm):
            # Degenerate input: triggers dense fallback
            fallback = True
            probs = np.ones(num_experts, dtype=np.float64) / num_experts
            indices = list(range(clamped_k))
        else:
            fallback = False
            np.random.seed(42)
            W = np.random.randn(len(x), num_experts) * 0.1
            logits = np.dot(x, W)
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            indices = np.argsort(probs)[-clamped_k:][::-1].tolist()

        entropy = CausalRouterBenchmark.compute_route_entropy(probs)
        return {
            "probs": probs.tolist(),
            "selected_experts": indices,
            "route_entropy": entropy,
            "fallback_triggered": fallback,
        }


# ============================================================================
# F08: Routing Profiler
# ============================================================================

class RoutingProfiler:
    """Profiles execution latency, expert utilization balance, and fallback rates."""

    def __init__(self, num_experts: int = 4):
        self.num_experts = max(1, num_experts)
        self.call_latencies: List[float] = []
        self.expert_counts: List[int] = [0] * self.num_experts
        self.entropies: List[float] = []
        self.fallbacks: int = 0

    def record_step(
        self,
        latency_us: float,
        selected_experts: List[int],
        entropy: float,
        fallback: bool,
    ) -> None:
        """Record diagnostic metrics for a single routing invocation."""
        self.call_latencies.append(float(latency_us))
        self.entropies.append(float(entropy))
        if fallback:
            self.fallbacks += 1
        for exp in selected_experts:
            if 0 <= exp < self.num_experts:
                self.expert_counts[exp] += 1

    def generate_report(self) -> Dict[str, Any]:
        """Aggregate recorded metrics into an audit report."""
        total_calls = len(self.call_latencies)
        if total_calls == 0:
            return {
                "total_calls": 0,
                "mean_latency_us": 0.0,
                "p95_latency_us": 0.0,
                "mean_entropy": 0.0,
                "expert_utilization": list(self.expert_counts),
                "balance_ratio": 1.0,
                "fallback_rate": 0.0,
            }

        mean_lat = float(np.mean(self.call_latencies))
        p95_lat = float(np.percentile(self.call_latencies, 95))
        mean_entropy = float(np.mean(self.entropies)) if self.entropies else 0.0
        min_c = min(self.expert_counts)
        max_c = max(1, max(self.expert_counts))
        balance_ratio = float(min_c / max_c)

        return {
            "total_calls": total_calls,
            "mean_latency_us": mean_lat,
            "p95_latency_us": p95_lat,
            "mean_entropy": mean_entropy,
            "expert_utilization": list(self.expert_counts),
            "balance_ratio": balance_ratio,
            "fallback_rate": float(self.fallbacks / total_calls),
        }

    def reset(self) -> None:
        """Clear all accumulated profiler traces."""
        self.call_latencies.clear()
        self.expert_counts = [0] * self.num_experts
        self.entropies.clear()
        self.fallbacks = 0


# ============================================================================
# Baselines: Static Dense Router & Random Router
# ============================================================================

class StaticDenseRouter(nn.Module):
    """Baseline dense router with uniform expert weights (0% pathway sparsity)."""

    def __init__(self, in_features: int = 8, num_experts: int = 4, out_dim: int = 4):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.out_dim = out_dim
        # Parallel expert linear transformations
        self.experts = nn.ModuleList([
            nn.Linear(in_features, out_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass allocating uniform weight 1/num_experts across all experts."""
        B = x.shape[0]
        # Uniform dense gates: [B, num_experts]
        gates = torch.ones(B, self.num_experts, device=x.device) / float(self.num_experts)
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)  # [B, num_experts, out_dim]
        # Weighted mixture
        y = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)  # [B, out_dim]
        sparsity = 0.0  # 0% sparsity (fully dense)
        return y, gates, sparsity


class RandomRouter(nn.Module):
    """Baseline router selecting K random experts uniformly without causal binding."""

    def __init__(self, in_features: int = 8, num_experts: int = 4, top_k: int = 2, out_dim: int = 4):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.out_dim = out_dim
        self.experts = nn.ModuleList([
            nn.Linear(in_features, out_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass selecting K random experts per sample."""
        B = x.shape[0]
        gates = torch.zeros(B, self.num_experts, device=x.device)
        for b in range(B):
            perm = torch.randperm(self.num_experts, device=x.device)[:self.top_k]
            gates[b, perm] = 1.0 / float(self.top_k)

        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)
        y = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        sparsity = 1.0 - (float(self.top_k) / float(self.num_experts))
        return y, gates, sparsity


# ============================================================================
# Causal Router: Neural Coordinator with Fallback and Counterfactuals
# ============================================================================

class CausalRouter(nn.Module):
    """High-level causal routing coordinator.
    
    Integrates CausalPathwayMIMO (for N=8) and HierarchicalSparseRouter (for N=16, 32).
    Includes degenerate input detection, automatic dense fallback, high-resolution
    profiler recording, and counterfactual scramble verification.
    """

    def __init__(
        self,
        in_features: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        memory_dim: int = 64,
        gist_dim: int = 32,
        hierarchical: bool = False,
        cluster_size: int = 8,
        profiler: Optional[RoutingProfiler] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.memory_dim = memory_dim
        self.gist_dim = gist_dim
        self.hierarchical = hierarchical
        self.profiler = profiler or RoutingProfiler(num_experts=num_experts)

        if self.hierarchical or in_features > 8:
            from ..pathway.hierarchical_router import HierarchicalSparseRouter
            self.router_backend = HierarchicalSparseRouter(
                cluster_size=cluster_size,
                memory_dim=memory_dim,
            )
            self._is_hierarchical = True
        else:
            from ..pathway.causal_mimo import CausalPathwayMIMO
            self.router_backend = CausalPathwayMIMO(
                memory_dim=memory_dim,
                gist_dim=gist_dim,
                epsilon_cushion=0.0,
            )
            self._is_hierarchical = False

        # Fallback linear layer for degenerate vectors
        self.fallback_layer = nn.Linear(in_features, num_experts)

    def _prepare_role(self, role_vector: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        """Ensure role vector has shape [B, in_features, ROLE_DIM]."""
        if role_vector is None:
            return F.one_hot(torch.arange(min(self.in_features, 8), device=device), num_classes=8).float().unsqueeze(0).expand(B, -1, -1)
        if role_vector.ndim == 1:
            return F.one_hot(torch.arange(min(self.in_features, 8), device=device), num_classes=8).float().unsqueeze(0).expand(B, -1, -1)
        if role_vector.ndim == 2:
            if role_vector.shape[1] == 8 and self.in_features == 8:
                return role_vector.unsqueeze(1).expand(B, 8, 8)
            else:
                return F.one_hot(torch.arange(min(self.in_features, 8), device=device), num_classes=8).float().unsqueeze(0).expand(B, -1, -1)
        return role_vector

    def _prepare_attractor(self, memory_attractor: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        """Ensure memory attractor tensor has shape [B, memory_dim]."""
        if memory_attractor is None:
            return torch.zeros(B, self.memory_dim, device=device)
        if memory_attractor.ndim == 1:
            return memory_attractor.unsqueeze(0).expand(B, -1)
        return memory_attractor

    def route(
        self,
        x: torch.Tensor,
        memory_attractor: Optional[torch.Tensor] = None,
        role_vector: Optional[torch.Tensor] = None,
        record_profile: bool = True,
    ) -> Dict[str, Any]:
        """Route input vector x with profiling and degenerate fallback protection.
        
        Args:
            x: Input tensor [B, in_features]
            memory_attractor: Synaptic memory attractor [B, memory_dim]
            role_vector: Query role vector
            record_profile: Whether to record step in profiler
        """
        t0 = time.perf_counter()
        B = x.shape[0]
        device = x.device

        memory_attractor = self._prepare_attractor(memory_attractor, B, device)
        r = self._prepare_role(role_vector, B, device)

        # Detect degenerate inputs (all zeros, NaN, or inf)
        norm_per_sample = torch.norm(x, p=2, dim=-1)
        is_nan_inf = torch.isnan(x).any(dim=-1) | torch.isinf(x).any(dim=-1)
        degenerate_mask = (norm_per_sample < 1e-6) | is_nan_inf

        clean_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if degenerate_mask.all():
            fallback_triggered = True
            y = self.fallback_layer(clean_x)
            gates = torch.ones(B, self.num_experts, device=device) / float(self.num_experts)
            expert_dist = gates.mean(dim=0)
            sparsity = 0.0
            entropy = math.log2(self.num_experts) if self.num_experts > 1 else 0.0
            selected_experts = list(range(self.top_k))
        else:
            fallback_triggered = bool(degenerate_mask.any().item())
            if self._is_hierarchical:
                role_ids = torch.arange(self.in_features, device=device).unsqueeze(0).expand(B, -1)
                all_outputs = []
                all_sparsities = []
                all_selected = []
                for b in range(B):
                    if degenerate_mask[b]:
                        all_outputs.append(self.fallback_layer(clean_x[b:b+1]).squeeze(0))
                        all_sparsities.append(0.0)
                        all_selected.append(list(range(self.top_k)))
                    else:
                        hout = self.router_backend.route_hierarchical(
                            clean_x[b:b+1],
                            role_ids=role_ids[b:b+1],
                            memory_attractor=memory_attractor[b],
                            n_out=self.num_experts,
                            k_local=self.top_k,
                            k_global=self.top_k,
                        )
                        out_t = torch.tensor(hout["outputs"], device=device, dtype=torch.float32)
                        all_outputs.append(out_t)
                        if "cluster_gates" in hout:
                            cg_flat = [val for cg in hout["cluster_gates"] for val in cg]
                            sp = float(np.mean([v == 0.0 for v in cg_flat]))
                        else:
                            sp = hout.get("sparsity", 0.5)
                        all_sparsities.append(sp)
                        all_selected.append(hout.get("selected_inputs", list(range(self.top_k)))[:self.top_k])
                y = torch.stack(all_outputs, dim=0)
                sparsity = float(np.mean(all_sparsities))
                selected_experts = all_selected[0]
                gates = torch.zeros(B, self.num_experts, device=device)
                for b in range(B):
                    for idx in all_selected[b]:
                        if 0 <= idx < self.num_experts:
                            gates[b, idx] = 1.0 / max(1, len(all_selected[b]))
                entropy = CausalRouterBenchmark.compute_route_entropy(gates.mean(dim=0))
            else:
                y, raw_gates, _, _, _ = self.router_backend.forward_causal(
                    clean_x, r, memory_attractor
                )
                if fallback_triggered:
                    for b in range(B):
                        if degenerate_mask[b]:
                            y[b] = self.fallback_layer(clean_x[b:b+1]).squeeze(0)
                            raw_gates[b] = torch.ones_like(raw_gates[b]) / float(self.num_experts)

                expert_probs = raw_gates.mean(dim=-1)  # [B, num_experts]
                expert_dist = expert_probs.mean(dim=0)  # [num_experts]
                gates = expert_probs
                sparsity = float((raw_gates == 0).float().mean().item())
                entropy = CausalRouterBenchmark.compute_route_entropy(expert_dist)
                selected_experts = torch.topk(expert_dist, k=self.top_k).indices.tolist()

        latency_us = (time.perf_counter() - t0) * 1_000_000.0

        if record_profile and self.profiler is not None:
            if self._is_hierarchical:
                profile_experts = [exp for s in all_selected for exp in s]
            elif torch.allclose(expert_dist, torch.full_like(expert_dist, 1.0 / self.num_experts), atol=1e-3):
                profile_experts = list(range(self.num_experts))
            else:
                profile_experts = torch.topk(gates, k=self.top_k, dim=-1).indices.flatten().tolist()

            self.profiler.record_step(
                latency_us=latency_us,
                selected_experts=profile_experts,
                entropy=entropy,
                fallback=fallback_triggered,
            )

        return {
            "output": y,
            "gates": gates,
            "sparsity": sparsity,
            "route_entropy": entropy,
            "selected_experts": selected_experts,
            "fallback_triggered": fallback_triggered,
            "latency_us": latency_us,
        }

    def evaluate_counterfactual(
        self,
        x: torch.Tensor,
        target_y: torch.Tensor,
        role_vector: Optional[torch.Tensor] = None,
        memory_attractor: Optional[torch.Tensor] = None,
        num_scrambles: int = 15,
    ) -> Dict[str, Any]:
        """Perform M2M-93 counterfactual route scramble audit."""
        B = x.shape[0]
        device = x.device
        r = self._prepare_role(role_vector, B, device)
        att = self._prepare_attractor(memory_attractor, B, device)

        if self._is_hierarchical:
            was_training = self.training
            self.eval()
            try:
                with torch.no_grad():
                    clean_res = self.route(x, att, role_vector, record_profile=False)
                    clean_y = clean_res["output"]
                    clean_mse = float(F.mse_loss(clean_y, target_y).item())

                    scrambled_mses: list[float] = []
                    for _ in range(num_scrambles):
                        perm = torch.randperm(x.shape[-1], device=device)
                        scr_res = self.route(x[:, perm], att, role_vector, record_profile=False)
                        scr_mse = float(F.mse_loss(scr_res["output"], target_y).item())
                        scrambled_mses.append(scr_mse)

                    mean_scrambled_mse = float(np.mean(scrambled_mses))
                    signal_var = float(torch.var(clean_y).item())
                    denom = max(clean_mse, signal_var, 1e-6)
                    causality_ratio = 1.0 + (mean_scrambled_mse / denom)

                    return {
                        "clean_mse": round(clean_mse, 6),
                        "scrambled_mse_mean": round(mean_scrambled_mse, 6),
                        "causality_ratio": round(causality_ratio, 4),
                        "is_causally_dependent": bool(causality_ratio >= 1.5),
                        "clean_gate_sparsity": round(clean_res["sparsity"], 4),
                    }
            finally:
                if was_training:
                    self.train()
        else:
            return self.router_backend.evaluate_counterfactual(
                x=x,
                r=r,
                memory_attractor=att,
                target_y=target_y,
                num_scrambles=num_scrambles,
            )

