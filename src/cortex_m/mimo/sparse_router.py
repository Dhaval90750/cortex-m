"""Evidence-conditioned sparse routing over an explicit relational library.

This is a transparent, non-neural task-adaptation baseline. It uses support
examples to score candidate routes, then Sparsemax selects a sparse route per
output. It is intentionally kept separate from Cortex-M's dependency-free core
and from the archived neural M2M implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import torch


@dataclass(frozen=True)
class Route:
    operation: str
    left: int
    right: int

    @property
    def label(self) -> str:
        return f"{self.operation}({self.left},{self.right})"


def candidate_routes(n_inputs: int) -> tuple[Route, ...]:
    if n_inputs < 2:
        raise ValueError("At least two input features are required")
    routes: list[Route] = []
    for a, b in combinations(range(n_inputs), 2):
        routes.extend((Route("sum", a, b), Route("product", a, b)))
    for a, b in permutations(range(n_inputs), 2):
        routes.extend((Route("difference", a, b), Route("affine", a, b)))
    return tuple(routes)


def evaluate_routes(x: torch.Tensor, routes: tuple[Route, ...]) -> torch.Tensor:
    """Evaluate each candidate route for rows in x; returns shape [rows,routes]."""
    if x.ndim != 2:
        raise ValueError("x must have shape [rows, input_features]")
    if not routes:
        raise ValueError("At least one candidate route is required")
    left = torch.tensor([r.left for r in routes], dtype=torch.long, device=x.device)
    right = torch.tensor([r.right for r in routes], dtype=torch.long, device=x.device)
    a, b = x.index_select(1, left), x.index_select(1, right)
    operations = [r.operation for r in routes]
    out = torch.empty_like(a)
    for op in ("sum", "product", "difference", "affine"):
        mask = torch.tensor([name == op for name in operations], device=x.device)
        if not bool(mask.any()):
            continue
        if op == "sum":
            out[:, mask] = (a + b)[:, mask]
        elif op == "product":
            out[:, mask] = (a * b)[:, mask]
        elif op == "difference":
            out[:, mask] = (a - b)[:, mask]
        else:
            out[:, mask] = (0.5 * a + 0.8 * b)[:, mask]
    return out


def sparsemax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the probability simplex."""
    if scores.ndim == 0:
        raise ValueError("scores must have at least one dimension")
    dim = dim if dim >= 0 else scores.ndim + dim
    if not 0 <= dim < scores.ndim:
        raise ValueError("dim is out of range")
    z = scores - scores.max(dim=dim, keepdim=True).values
    sorted_z = torch.sort(z, dim=dim, descending=True).values
    size = z.shape[dim]
    shape = [1] * z.ndim
    shape[dim] = size
    ranks = torch.arange(1, size + 1, device=z.device, dtype=z.dtype).view(shape)
    cumulative = sorted_z.cumsum(dim) - 1
    support = (1 + ranks * sorted_z) > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_sum = cumulative.gather(dim, support_size - 1)
    tau = tau_sum / support_size.to(z.dtype)
    return torch.clamp(z - tau, min=0)


@dataclass
class RoutingResult:
    prediction: torch.Tensor
    route_indices: torch.Tensor
    route_weights: torch.Tensor
    normalized_losses: torch.Tensor
    confidence_margin: torch.Tensor


class SparseEvidenceRouter:
    """Infer one explicit input-relation route for every output from examples."""

    def __init__(self, temperature: float = 0.01):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def route(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> RoutingResult:
        if support_x.ndim != 2 or support_y.ndim != 2 or query_x.ndim != 2:
            raise ValueError("support_x, support_y, and query_x must be rank 2")
        if support_x.shape[0] != support_y.shape[0] or support_x.shape[0] < 2:
            raise ValueError("support arrays must have equal row counts and at least two rows")
        if support_x.shape[1] != query_x.shape[1]:
            raise ValueError("support/query feature counts must match")
        routes = candidate_routes(support_x.shape[1])
        support_experts = evaluate_routes(support_x, routes)
        query_experts = evaluate_routes(query_x, routes)

        # [outputs, candidate routes]; normalize to compare outputs at different scales.
        residual = support_experts.unsqueeze(0) - support_y.T.unsqueeze(-1)
        mse = residual.square().mean(dim=1)
        scale = support_y.var(dim=0, unbiased=False).clamp_min(1e-8).unsqueeze(-1)
        normalized = mse / scale
        weights = sparsemax(-normalized / self.temperature, dim=-1)
        indices = normalized.argmin(dim=-1)
        sorted_losses = torch.sort(normalized, dim=-1).values
        margin = sorted_losses[:, 1] - sorted_losses[:, 0]
        predictions = torch.stack(
            [(query_experts * weights[o].unsqueeze(0)).sum(dim=-1)
             for o in range(support_y.shape[1])],
            dim=-1,
        )
        return RoutingResult(predictions, indices, weights, normalized, margin)


def route_labels(n_inputs: int, route_ids: list[int]) -> list[str]:
    routes = candidate_routes(n_inputs)
    return [routes[index].label for index in route_ids]
