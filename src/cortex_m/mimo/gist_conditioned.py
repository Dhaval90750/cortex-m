import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .m2m_sparse_route import M2MSparseRoute, N_IN, N_OUT, ROLE_DIM, HIDDEN, K, TAU_MIN, sparsemax


class GistConditionedM2M(M2MSparseRoute):
    """M2M-SparseRoute with retrieved two-state task gist in router context."""
    def __init__(self):
        super().__init__()
        self.gist_encoder = nn.Sequential(
            nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN)
        )

    def _route_scores(self, z, gist, r, batch_size):
        return self.router(z).squeeze(-1)

    def forward(self, x, r, gist, tau=TAU_MIN):
        if gist.shape != (x.shape[0], 2):
            raise ValueError("gist must have shape [batch, 2]")
        B = x.size(0)
        h = self.node(torch.cat([x.unsqueeze(-1), r], dim=-1))
        key = self.context_key(torch.cat([h, r], dim=-1))
        global_role = r.mean(dim=1)
        query = self.context_query(global_role)
        value = self.context_value(h)
        attention_scores = (key * query[:, None]).sum(dim=-1) / math.sqrt(HIDDEN)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_state = (attention_weights[:, :, None] * value).sum(dim=1)
        # Gist changes the actual router context used to compute edge scores.
        c = self.context_pool(context_state) + self.gist_encoder(gist)
        context_logits = self.context_classifier(c)
        out_roles = self.out_role(torch.eye(N_OUT, ROLE_DIM, device=x.device))
        hi = h[:, None].expand(B, N_OUT, N_IN, HIDDEN)
        ho = out_roles[None, :, None].expand(B, N_OUT, N_IN, HIDDEN)
        hc = c[:, None, None].expand(B, N_OUT, N_IN, HIDDEN)
        pair = hi * ho
        z = torch.cat([hi, ho, pair, hc, pair * hc], dim=-1)
        scores = self._route_scores(z, gist, r, B)
        gates = K * sparsemax(scores / tau, dim=2)
        rin = r[:, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        rout = torch.eye(N_OUT, ROLE_DIM, device=x.device)[None, :, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        weights = self.synapse(torch.cat([rin, rout], dim=-1)).squeeze(-1)
        y = (gates * weights * x[:, None]).sum(dim=2) + self.bias
        return y, gates, weights, context_logits, c, attention_weights


def infer_gist(model, values, role_ids, gist_mode, device="cpu"):
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    roles = torch.as_tensor(role_ids, dtype=torch.long, device=device)
    g = torch.as_tensor(gist_mode, dtype=torch.float32, device=device)
    if x.ndim == 1: x = x.unsqueeze(0)
    if roles.ndim == 1: roles = roles.unsqueeze(0)
    if g.ndim == 1: g = g.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != N_IN:
        raise ValueError("values must have shape [8] or [batch, 8]")
    if roles.shape != x.shape or any(sorted(row.tolist()) != list(range(N_IN)) for row in roles):
        raise ValueError("role_ids must be a per-row permutation of 0..7")
    if g.shape != (x.shape[0], 2) or not torch.all((g == 0) | (g == 1)) or not torch.all(g.sum(dim=1) == 1):
        raise ValueError("gist_mode must be a one-hot alpha/beta vector per example")
    r = F.one_hot(roles, ROLE_DIM).to(dtype=x.dtype)
    with torch.inference_mode():
        y, gates, *_ = model(x, r, g, TAU_MIN)
        selected = gates.topk(K, dim=2).indices
    return {"outputs": y.cpu().tolist(), "selected_physical_positions": selected.cpu().tolist(),
            "gates": gates.cpu().tolist(), "gist_mode": g.cpu().tolist()}


def load_gist_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "GistConditionedM2M":
        raise ValueError("Unsupported gist-conditioned MIMO checkpoint architecture")
    if any(key.startswith("gist_route.") for key in checkpoint["state_dict"]):
        from .gist_direct_route import DirectGistRouteM2M
        model = DirectGistRouteM2M().to(device)
    else:
        model = GistConditionedM2M().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
