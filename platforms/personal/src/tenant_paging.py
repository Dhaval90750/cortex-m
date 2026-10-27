"""Multi-Tenant Low-Rank Paging Engine for CORTEX-Personal.

Enforces zero cross-profile leakage by representing individual user / tenant state
as low-rank factor slices ($2.0 KB rank-4 factors per tenant):
    Delta W_tenant = U_tenant * V_tenant^T
Paging allows instant profile switching on consumer laptops and edge devices without
swapping out base substrate weights.
"""

from __future__ import annotations

import math
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
        self.privacy_ledgers: Dict[str, Dict[str, Any]] = {}

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

    def learn_tenant_association_differentially_private(
        self,
        tenant_id: str,
        cue: torch.Tensor,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """Learn tenant association with rigorous (epsilon, delta)-Differential Privacy noise injection."""
        if epsilon <= 0.0 or delta <= 0.0:
            raise ValueError("Differential privacy parameters epsilon and delta must be strictly positive.")

        tenant = self.get_tenant(tenant_id)
        z = cue.view(-1).to(self.device).float()

        # 1. L2 Norm clipping: z_clipped = z / max(1.0, ||z||_2 / clip_norm)
        norm = torch.norm(z, p=2)
        if norm > clip_norm:
            z_clipped = z * (clip_norm / norm)
        else:
            z_clipped = z

        # 2. Sensitivity Delta_2 = 2 * clip_norm * sqrt(learning_rate)
        sensitivity = 2.0 * clip_norm * math.sqrt(learning_rate)

        # 3. Gaussian mechanism noise standard deviation:
        # sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

        # 4. Sample and add calibrated noise
        noise = torch.randn_like(z_clipped) * sigma
        z_priv = z_clipped + noise

        # Normalize noisy vector
        z_final = F.normalize(z_priv, p=2, dim=0)

        # 5. Update low-rank factors
        col_idx = len(tenant.active_attractors) % self.rank
        scale = math.sqrt(learning_rate)
        tenant.U[:, col_idx] = scale * z_final
        tenant.V[:, col_idx] = scale * z_final
        tenant.active_attractors.append(f"dp_attractor_{len(tenant.active_attractors)}")
        tenant.last_access_time = time.time()

        # 6. Privacy ledger accounting
        if tenant_id not in self.privacy_ledgers:
            self.privacy_ledgers[tenant_id] = {
                "total_epsilon": 0.0,
                "total_delta": 0.0,
                "update_count": 0,
            }
        ledger = self.privacy_ledgers[tenant_id]
        ledger["total_epsilon"] = round(ledger["total_epsilon"] + epsilon, 4)
        ledger["total_delta"] = round(ledger["total_delta"] + delta, 6)
        ledger["update_count"] += 1

        signal_pwr = float(torch.norm(z_clipped, p=2).item())
        noise_pwr = float(torch.norm(noise, p=2).item())
        snr_db = 20.0 * math.log10(max(1e-6, signal_pwr) / max(1e-6, noise_pwr))

        return {
            "tenant_id": tenant_id,
            "epsilon": epsilon,
            "delta": delta,
            "sigma": round(sigma, 4),
            "sensitivity": round(sensitivity, 4),
            "snr_db": round(snr_db, 2),
            "accumulated_epsilon": ledger["total_epsilon"],
            "accumulated_delta": ledger["total_delta"],
            "update_count": ledger["update_count"],
        }

    def list_tenants(self) -> List[str]:
        """Return list of all registered profile identifiers."""
        return list(self._tenants.keys())

    @property
    def active_tenant_id(self) -> Optional[str]:
        return self._active_tenant_id

    def get_storage_footprint(self) -> Dict[str, Any]:
        """Report granular byte memory consumption across all active tenant profiles."""
        total_bytes = sum(t.size_bytes for t in self._tenants.values())
        return {
            "num_tenants": len(self._tenants),
            "total_bytes": total_bytes,
            "average_bytes_per_tenant": round(total_bytes / max(1, len(self._tenants)), 1),
            "tenants": {tid: t.size_bytes for tid, t in self._tenants.items()},
        }

    def page_out_to_disk(self, tenant_id: str, file_path: str) -> str:
        """Serialize low-rank factor slice (2 KB) to persistent local disk storage."""
        from pathlib import Path
        tenant = self.get_tenant(tenant_id)
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "tenant_id": tenant.tenant_id,
            "dimension": tenant.dimension,
            "rank": tenant.rank,
            "U": tenant.U.detach().cpu(),
            "V": tenant.V.detach().cpu(),
            "creation_time": tenant.creation_time,
            "last_access_time": tenant.last_access_time,
            "active_attractors": tenant.active_attractors,
            "metadata": tenant.metadata,
        }
        torch.save(payload, path)
        return str(path)

    def page_in_from_disk(self, file_path: str) -> TenantFactorSlice:
        """Restore dormant tenant factor slice from disk into active runtime."""
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Tenant factor page not found: {path}")

        payload = torch.load(path, map_location=self.device)
        tenant_id = payload["tenant_id"]

        slice_obj = TenantFactorSlice(
            tenant_id=tenant_id,
            dimension=payload["dimension"],
            rank=payload["rank"],
            U=payload["U"].to(self.device),
            V=payload["V"].to(self.device),
            creation_time=payload["creation_time"],
            last_access_time=time.time(),
            active_attractors=payload.get("active_attractors", []),
            metadata=payload.get("metadata", {}),
        )
        self._tenants[tenant_id] = slice_obj
        if self._active_tenant_id is None:
            self._active_tenant_id = tenant_id
        return slice_obj

