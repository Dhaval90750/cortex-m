"""Multi-Tenant Low-Rank Paging Engine for CORTEX-Personal.

Enforces zero cross-profile leakage by representing individual user / tenant state
as low-rank factor slices ($2.0 KB rank-4 factors per tenant):
    Delta W_tenant = U_tenant * V_tenant^T
Paging allows instant profile switching on consumer laptops and edge devices without
swapping out base substrate weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TenantFactorSlice:
    """Isolated low-rank factor slice representing a tenant profile (~2.0 KB)."""
    tenant_id: str
    dimension: int = 64
    rank: int = 4
    U: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    V: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    creation_time: float = 0.0
    last_access_time: float = 0.0
    active_attractors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        """Calculate exact in-memory byte size of tensor factors."""
        return (self.U.numel() + self.V.numel()) * 4  # 4 bytes per float32


class PersonalTenantPagingEngine:
    """Manages active and dormant tenant factor slices with isolated synaptic routing."""

    def __init__(self, base_dimension: int = 64, rank: int = 4, device: str = "cpu"):
        self.base_dimension = base_dimension
        self.rank = rank
        self.device = device
        self._tenants: Dict[str, TenantFactorSlice] = {}
        self._active_tenant_id: Optional[str] = None

    def create_tenant(
        self,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TenantFactorSlice:
        """Allocate a pristine low-rank factor slice for a new tenant profile."""
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant profile {tenant_id!r} already exists.")

        U = torch.zeros(self.base_dimension, self.rank, device=self.device)
        V = torch.zeros(self.base_dimension, self.rank, device=self.device)
        now = time.time()

        slice_obj = TenantFactorSlice(
            tenant_id=tenant_id,
            dimension=self.base_dimension,
            rank=self.rank,
            U=U,
            V=V,
            creation_time=now,
            last_access_time=now,
            metadata=metadata or {},
        )
        self._tenants[tenant_id] = slice_obj
        if self._active_tenant_id is None:
            self._active_tenant_id = tenant_id
        return slice_obj

    def get_tenant(self, tenant_id: str) -> TenantFactorSlice:
        """Retrieve a tenant factor slice."""
        if tenant_id not in self._tenants:
            raise KeyError(f"Tenant profile {tenant_id!r} not found.")
        tenant = self._tenants[tenant_id]
        tenant.last_access_time = time.time()
        return tenant

    def switch_tenant(self, tenant_id: str) -> TenantFactorSlice:
        """Switch current active profile context."""
        tenant = self.get_tenant(tenant_id)
        self._active_tenant_id = tenant_id
        return tenant

    def get_effective_weights(
        self,
        base_weights: torch.Tensor,
        tenant_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Compose base substrate weights with tenant low-rank factor delta."""
        tid = tenant_id or self._active_tenant_id
        if tid is None or tid not in self._tenants:
            return base_weights

        tenant = self._tenants[tid]
        # Delta W = U * V^T
        delta = torch.mm(tenant.U, tenant.V.t())
        delta = 0.5 * (delta + delta.t())
        return base_weights + delta

    def learn_tenant_association(
        self,
        tenant_id: str,
        cue: torch.Tensor,
        learning_rate: float = 0.1,
    ) -> None:
        """Incrementally update tenant's rank-4 factor slice without affecting other profiles."""
        tenant = self.get_tenant(tenant_id)
        z = F.normalize(cue.view(-1).to(self.device), p=2, dim=0)

        # Update rank-1 component into U and V (cycle through rank columns)
        col_idx = len(tenant.active_attractors) % self.rank
        scale = (learning_rate ** 0.5)
        tenant.U[:, col_idx] = scale * z
        tenant.V[:, col_idx] = scale * z
        tenant.active_attractors.append(f"attractor_{len(tenant.active_attractors)}")
        tenant.last_access_time = time.time()

    def list_tenants(self) -> List[str]:
        """Return list of all registered profile identifiers."""
        return list(self._tenants.keys())

    @property
    def active_tenant_id(self) -> Optional[str]:
        return self._active_tenant_id
