"""Exact M2M-SparseRoute v1 model definition (optional PyTorch component)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN = 8
N_OUT = 4
ROLE_DIM = 8
HIDDEN = 32
K = 2
TAU_MIN = 0.08

def sparsemax(z, dim=-1):
    """
    Sparsemax projection onto the probability simplex.

    Unlike softmax, sparsemax can produce exact zeros.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    cssv = torch.cumsum(zs, dim=dim) - 1
    r = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    r = r.reshape(shape)
    cond = zs * r > cssv
    k_z = cond.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cssv.gather(dim, k_z - 1) / k_z
    return torch.clamp(z - tau, min=0)

class M2MSparseRoute(nn.Module):

    def __init__(self):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(1 + ROLE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.context_key = nn.Sequential(nn.Linear(HIDDEN + ROLE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
        self.context_query = nn.Sequential(nn.Linear(ROLE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
        self.context_value = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN))
        self.context_pool = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.context_classifier = nn.Linear(HIDDEN, 2)
        self.out_role = nn.Sequential(nn.Linear(ROLE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.router = nn.Sequential(nn.Linear(5 * HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
        self.synapse = nn.Sequential(nn.Linear(2 * ROLE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
        self.bias = nn.Parameter(torch.zeros(N_OUT))

    def forward(self, x, r, tau=TAU_MIN):
        B = x.size(0)
        h = self.node(torch.cat([x.unsqueeze(-1), r], dim=-1))
        key = self.context_key(torch.cat([h, r], dim=-1))
        global_role = r.mean(dim=1)
        query = self.context_query(global_role)
        value = self.context_value(h)
        attention_scores = (key * query[:, None]).sum(dim=-1) / math.sqrt(HIDDEN)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_state = (attention_weights[:, :, None] * value).sum(dim=1)
        c = self.context_pool(context_state)
        context_logits = self.context_classifier(c)
        out_roles = self.out_role(torch.eye(N_OUT, ROLE_DIM, device=x.device))
        hi = h[:, None].expand(B, N_OUT, N_IN, HIDDEN)
        ho = out_roles[None, :, None].expand(B, N_OUT, N_IN, HIDDEN)
        hc = c[:, None, None].expand(B, N_OUT, N_IN, HIDDEN)
        pair = hi * ho
        z = torch.cat([hi, ho, pair, hc, pair * hc], dim=-1)
        scores = self.router(z).squeeze(-1)
        P = sparsemax(scores / tau, dim=2)
        G = K * P
        rin = r[:, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        rout = torch.eye(N_OUT, ROLE_DIM, device=x.device)[None, :, None].expand(B, N_OUT, N_IN, ROLE_DIM)
        W = self.synapse(torch.cat([rin, rout], dim=-1)).squeeze(-1)
        y = (G * W * x[:, None]).sum(dim=2) + self.bias
        return (y, G, W, context_logits, c, attention_weights)

def load_model(checkpoint_path, device="cpu"):
    """Load a saved M2M-35D seed checkpoint for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "M2MSparseRoute":
        raise ValueError("Unsupported MIMO checkpoint architecture")
    model = M2MSparseRoute().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint

def infer(model, values, role_ids, device="cpu"):
    """Run one or more N=8 examples. role_ids maps physical positions to semantic roles."""
    x = torch.as_tensor(values, dtype=torch.float32, device=device)
    roles = torch.as_tensor(role_ids, dtype=torch.long, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if roles.ndim == 1:
        roles = roles.unsqueeze(0)
    if x.ndim != 2 or tuple(x.shape[1:]) != (N_IN,):
        raise ValueError("values must have shape [8] or [batch, 8]")
    if roles.shape != x.shape or not torch.all((roles >= 0) & (roles < ROLE_DIM)):
        raise ValueError("role_ids must match values and contain integers in [0, 7]")
    if any(sorted(row.tolist()) != list(range(N_IN)) for row in roles):
        raise ValueError("each role_ids row must be a permutation of 0..7")
    r = F.one_hot(roles, ROLE_DIM).to(dtype=x.dtype)
    with torch.inference_mode():
        y, gates, weights, context_logits, context, attention = model(x, r, TAU_MIN)
        top = gates.topk(K, dim=2).indices
    return {
        "outputs": y.cpu().tolist(),
        "selected_physical_positions": top.cpu().tolist(),
        "gates": gates.cpu().tolist(),
        "context_logits": context_logits.cpu().tolist(),
        "attention_weights": attention.cpu().tolist(),
    }
