"""Personal Substrate Coordinator for CORTEX-Personal.

Main entry point integrating:
1. Newborn Substrate Principle (starts with W = 0, zero pre-baked facts).
2. Continuous Modern Hopfield associative memory & Hebbian on-the-go learning.
3. Pre-attractor epistemic shield with honest abstention (INSUFFICIENT_EVIDENCE).
4. Anti-Hebbian targeted unlearning (zero collateral damage).
5. Offline nocturnal sleep consolidation.
6. Encrypted personal vault persistence (.cortex_vault).
7. Zero-egress network interlock.
8. Multi-tenant low-rank factor paging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from cortex_m.cortex_v2 import CortexV2
from cortex_m.learning.consolidation import NocturnalSleepConsolidator, SleepCycleReport
from .interlock import ZeroEgressInterlock
from .tenant_paging import PersonalTenantPagingEngine
from .vault import EncryptedPersonalVault


@dataclass
class PersonalQueryResult:
    """Result of a personal memory recall query with calibrated epistemic status."""
    cue: str
    status: str  # 'VERIFIED', 'PARTIAL', 'INSUFFICIENT_EVIDENCE'
    recalled_text: Optional[str]
    energy: float
    confidence: float
    memory_id: Optional[str]
    source_context: Optional[str]
    latency_ms: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class PersonalSubstrateCoordinator:
    """Manages the full lifecycle of an edge personal cognitive assistant."""

    def __init__(
        self,
        dim: int = 64,
        gist_dim: int = 32,
        beta: float = 8.0,
        abstention_threshold: float = 0.65,
        device: str = "cpu",
        seed: int = 42,
        enforce_zero_egress: bool = True,
    ):
        self.dim = dim
        self.gist_dim = gist_dim
        self.beta = beta
        self.abstention_threshold = abstention_threshold
        self.device = device

        # 1. Zero-Egress Network Interlock
        self.interlock = ZeroEgressInterlock(strict_mode=True)
        if enforce_zero_egress:
            self.interlock.enable()

        # 2. Core Neural Engine (Tabula Rasa newborn substrate)
        self.cortex = CortexV2(
            dim=dim,
            gist_dim=gist_dim,
            beta=beta,
            device=device,
            seed=seed,
        )
        # Ensure pristine initial weights (Tabula Rasa)
        self.cortex.memory.synaptic_weights.zero_()
        self.cortex.memory.memory_attractors = torch.empty((0, dim), device=device)
        self.cortex.memory.memory_metadata.clear()

        # 3. Nocturnal Sleep Consolidator
        self.consolidator = NocturnalSleepConsolidator(
            dim=dim,
            pruning_threshold=0.04,
            homeostatic_target_norm=8.0,
            device=device,
        )

        # 4. Low-Rank Tenant Paging Engine
        self.paging_engine = PersonalTenantPagingEngine(
            base_dimension=dim,
            rank=4,
            device=device,
        )

        # 5. Stored textual memories mapping memory_id -> text
        self._memory_text_store: Dict[str, str] = {}

    def teach(
        self,
        text: str,
        source_context: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Imprint a new personal fact or note via Hebbian synaptic potentiation."""
        mem_id = memory_id or f"pers_mem_{len(self._memory_text_store) + 1}"
        encoded_vec = self.cortex.encoder.encode(text).view(-1)
        normalized_pattern = F.normalize(encoded_vec, p=2, dim=0)

        # Imprint into synaptic substrate
        self.cortex.memory.learn_pattern(
            pattern=normalized_pattern,
            memory_id=mem_id,
            source_context=source_context,
            metadata={"raw_text": text, "timestamp": time.time()},
        )
        self._memory_text_store[mem_id] = text
        return mem_id

    def query(self, cue: str) -> PersonalQueryResult:
        """Recall associative memory or honestly abstain if evidence is insufficient."""
        start_time = time.perf_counter()

        # Check if substrate is newborn (empty)
        if self.cortex.memory.num_memories == 0:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return PersonalQueryResult(
                cue=cue,
                status="INSUFFICIENT_EVIDENCE",
                recalled_text=None,
                energy=0.0,
                confidence=0.0,
                memory_id=None,
                source_context=None,
                latency_ms=duration_ms,
                diagnostics={"reason": "tabula_rasa_newborn_empty"},
            )

        encoded_cue = self.cortex.encoder.encode(cue).view(-1)
        normalized_cue = F.normalize(encoded_cue, p=2, dim=0)

        # Pre-attractor projection check (Epistemic Shield)
        attractors = self.cortex.memory.memory_attractors
        overlaps = torch.matmul(attractors, normalized_cue)
        max_overlap, best_idx = overlaps.max(dim=0)
        confidence = float(max_overlap.item())

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        if confidence < self.abstention_threshold:
            return PersonalQueryResult(
                cue=cue,
                status="INSUFFICIENT_EVIDENCE",
                recalled_text=None,
                energy=self.cortex.memory.energy(normalized_cue),
                confidence=confidence,
                memory_id=None,
                source_context=None,
                latency_ms=duration_ms,
                diagnostics={"max_overlap": confidence, "threshold": self.abstention_threshold},
            )

        # Settled attractor retrieval
        recalled_state, trace, diag = self.cortex.memory.recall(normalized_cue)
        target_meta = self.cortex.memory.memory_metadata[best_idx]
        target_id = target_meta.get("memory_id")
        recalled_text = self._memory_text_store.get(target_id, target_meta.get("raw_text"))

        status = "VERIFIED" if confidence >= 0.82 else "PARTIAL"

        return PersonalQueryResult(
            cue=cue,
            status=status,
            recalled_text=recalled_text,
            energy=diag.get("final_energy", 0.0),
            confidence=confidence,
            memory_id=target_id,
            source_context=target_meta.get("source_context"),
            latency_ms=duration_ms,
            diagnostics=diag,
        )

    def unlearn(self, memory_id_or_text: str) -> bool:
        """Targeted right-to-be-forgotten via anti-Hebbian synaptic depression."""
        # Find memory ID
        target_id = None
        for mid, txt in self._memory_text_store.items():
            if mid == memory_id_or_text or txt == memory_id_or_text:
                target_id = mid
                break

        if target_id is None:
            # Check metadata directly
            for meta in self.cortex.memory.memory_metadata:
                if meta.get("memory_id") == memory_id_or_text:
                    target_id = memory_id_or_text
                    break

        if target_id is None:
            return False

        # Execute anti-Hebbian unlearning
        res = self.cortex.memory.forget_pattern(target_id)
        if res.get("status") == "unlearned":
            self._memory_text_store.pop(target_id, None)
            return True
        return False

    def sleep_consolidate(self) -> SleepCycleReport:
        """Execute nocturnal sleep consolidation over personal memories."""
        W = self.cortex.memory.synaptic_weights
        P = self.cortex.memory.memory_attractors
        W_cons, report = self.consolidator.consolidate(W, P)
        self.cortex.memory.synaptic_weights = W_cons
        return report

    def save_vault(self, vault_path: Union[str, Path], passphrase: str) -> None:
        """Encrypt and persist personal synaptic state to .cortex_vault file."""
        vault = EncryptedPersonalVault(vault_path)
        meta = {
            "text_store": self._memory_text_store,
            "memory_metadata": self.cortex.memory.memory_metadata,
            "dim": self.dim,
            "beta": self.beta,
        }
        vault.save(
            passphrase=passphrase,
            synaptic_weights=self.cortex.memory.synaptic_weights,
            attractor_patterns=self.cortex.memory.memory_attractors,
            metadata=meta,
        )

    def load_vault(self, vault_path: Union[str, Path], passphrase: str) -> None:
        """Decrypt and restore personal synaptic state from .cortex_vault file."""
        vault = EncryptedPersonalVault(vault_path)
        weights, attractors, meta = vault.load(passphrase=passphrase, device=self.device)
        self.cortex.memory.synaptic_weights = weights
        self.cortex.memory.memory_attractors = attractors
        self._memory_text_store = meta.get("text_store", {})
        self.cortex.memory.memory_metadata = meta.get("memory_metadata", [])

    def shutdown(self) -> None:
        """Cleanly tear down personal coordinator and disable interlock."""
        self.interlock.disable()
