"""Learned Gist Information Bottleneck: Pre-verbal intent representation.

In human cognitive neuroscience, humans form an internal semantic gist of what
they intend to communicate before generating token-level or verbal expressions.
This is the antithesis of autoregressive next-token guessing:
- Information is forced through an information bottleneck (e.g. 64/384 -> 32 dim).
- The compressed vector captures task mode, semantic intent, and relational focus.
- The gist vector directly conditions the downstream MIMO routing context.
- Unlike binary one-hot flags [1,0], a continuous gist generalizes across unseen modes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GistBottleneck(nn.Module):
    """Information bottleneck compressing semantic vectors into a compact pre-verbal gist.

    Parameters
    ----------
    in_dim : int
        Dimensionality of input semantic embedding (e.g. 64 or 384).
    gist_dim : int
        Dimensionality of the compressed gist bottleneck (e.g. 32).
    out_dim : int
        Dimensionality of the conditioned context for the downstream router (e.g. 64).
    """

    def __init__(
        self,
        in_dim: int = 64,
        gist_dim: int = 32,
        out_dim: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.gist_dim = gist_dim
        self.out_dim = out_dim

        torch.manual_seed(seed)
        # Compression encoder: in_dim -> 128 -> gist_dim
        self.compressor = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, gist_dim),
            nn.LayerNorm(gist_dim),
        )

        # Expansion generator: gist_dim -> 128 -> out_dim
        self.expander = nn.Sequential(
            nn.Linear(gist_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

        # M2M-97: Bidirectional consistency decoder (Routing Context -> Latent Gist)
        self.route_to_gist_decoder = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, gist_dim),
            nn.LayerNorm(gist_dim),
        )

        # Mode prototype codebook: registers continuous prototypes for known semantic intents
        self.register_buffer("prototypes", torch.zeros(0, gist_dim))
        self.prototype_names: list[str] = []

    def register_prototype(self, name: str, vector: torch.Tensor) -> None:
        """Register a known mode prototype in the gist space."""
        v = F.normalize(vector.detach().view(-1), p=2, dim=0)
        if v.size(0) != self.gist_dim:
            if v.size(0) == self.in_dim:
                v = self.extract_gist(v)
            else:
                raise ValueError(f"Expected vector of dim {self.gist_dim} or {self.in_dim}, got {v.size(0)}")
        self.prototypes = torch.cat([self.prototypes, v.unsqueeze(0)], dim=0)
        self.prototype_names.append(name)

    def extract_gist(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input x into the pre-verbal gist vector."""
        if x.ndim == 1:
            x_in = x.unsqueeze(0)
            gist = self.compressor(x_in).squeeze(0)
        else:
            gist = self.compressor(x)
        return F.normalize(gist, p=2, dim=-1)

    def expand_gist(self, gist: torch.Tensor) -> torch.Tensor:
        """Expand compressed gist into routing context."""
        if gist.ndim == 1:
            g_in = gist.unsqueeze(0)
            out = self.expander(g_in).squeeze(0)
        else:
            out = self.expander(gist)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (gist_vector, expanded_context)."""
        gist = self.extract_gist(x)
        context = self.expand_gist(gist)
        return gist, context

    def inspect_gist(self, gist: torch.Tensor) -> dict[str, Any]:
        """Classify or compare a gist against registered prototypes."""
        g = F.normalize(gist.detach().view(-1), p=2, dim=0)
        info = {
            "gist_dim": self.gist_dim,
            "gist_norm": float(g.norm().item()),
            "top_activations": [round(float(val), 4) for val in torch.topk(g.abs(), k=min(5, self.gist_dim)).values],
            "prototype_similarities": {},
        }
        if self.prototypes.size(0) > 0:
            sims = torch.matmul(self.prototypes, g)
            for i, name in enumerate(self.prototype_names):
                info["prototype_similarities"][name] = round(float(sims[i].item()), 4)
            best_idx = int(sims.argmax().item())
            info["nearest_prototype"] = self.prototype_names[best_idx]
            info["nearest_similarity"] = round(float(sims[best_idx].item()), 4)
        return info

    def decode_route_to_gist(self, route_context: torch.Tensor) -> torch.Tensor:
        """M2M-97: Reconstruct latent gist vector from router representation."""
        if route_context.ndim == 1:
            r_in = route_context.unsqueeze(0)
            g_hat = self.route_to_gist_decoder(r_in).squeeze(0)
        else:
            g_hat = self.route_to_gist_decoder(route_context)
        return F.normalize(g_hat, p=2, dim=-1)

    def compute_bidirectional_consistency_loss(
        self,
        gist: torch.Tensor,
        route_context: torch.Tensor,
    ) -> torch.Tensor:
        """M2M-97: Compute L_consistency = ||g - g_hat||_2^2 enforcing invertible bidirectional coupling."""
        g_hat = self.decode_route_to_gist(route_context)
        return F.mse_loss(gist, g_hat)

    def evaluate_rate_distortion_tradeoff(
        self,
        x: torch.Tensor,
        beta_ib: float = 1.25,
    ) -> dict[str, float]:
        """M2M-96: Evaluate Information Bottleneck rate-distortion metrics at given beta_ib."""
        gist, expanded = self.forward(x)
        reconstruction_error = float(F.mse_loss(expanded, x if x.shape == expanded.shape else x.repeat(1, expanded.shape[-1] // x.shape[-1])).item())
        compression_ratio = float(self.in_dim / self.gist_dim)
        rate = float(-0.5 * torch.sum(1 + torch.log(gist.var(dim=-1).clamp(min=1e-5)) - gist.mean(dim=-1)**2 - gist.var(dim=-1)).item())
        distortion = reconstruction_error
        lagrangian = rate + beta_ib * distortion
        return {
            "beta_ib": beta_ib,
            "compression_ratio": compression_ratio,
            "rate_bits": round(rate, 4),
            "distortion_mse": round(distortion, 6),
            "lagrangian_loss": round(lagrangian, 4),
        }

