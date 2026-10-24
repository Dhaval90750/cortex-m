import torch
import torch.nn as nn
from .m2m_sparse_route import HIDDEN, N_IN, N_OUT, ROLE_DIM
from .gist_conditioned import GistConditionedM2M


class DirectGistRouteM2M(GistConditionedM2M):
    """Failed alternative: add a separate gist-conditioned route-score head."""
    def __init__(self):
        super().__init__()
        self.gist_route = nn.Sequential(
            nn.Linear(2 + 2 * ROLE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def _route_scores(self, z, gist, r, batch_size):
        base = self.router(z).squeeze(-1)
        mode_nodes = gist[:, None, None].expand(batch_size, N_OUT, N_IN, 2)
        roles = r[:, None].expand(batch_size, N_OUT, N_IN, ROLE_DIM)
        output_roles = torch.eye(N_OUT, ROLE_DIM, device=r.device)[None, :, None].expand(batch_size, N_OUT, N_IN, ROLE_DIM)
        context = torch.cat([mode_nodes, roles, output_roles], dim=-1)
        return base + self.gist_route(context).squeeze(-1)
