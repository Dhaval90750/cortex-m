"""Continuous Hopfield Episodic Memory Store.

Implements structured episodic memory storage, provenance-aware filtering,
invalidation tracking, and continuous Modern Hopfield network integration
conforming to CORTEX-M Master Execution Guide §26 and Interface Contracts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from cortex_m.pathway.encoder import NeuralFeatureEncoder
from cortex_m.pathway.synaptic_memory import SynapticMemoryNetwork

VALID_EVIDENCE_TYPES = {
    "Observed",     # Direct empirical measurement or execution output
    "Derived",      # Deductive or algebraic inference from verified premises
    "Predicted",    # Prospective output from model simulation / forward pass
    "Literature",   # External factual reference or prior published work
    "Hypothesis",   # Speculative explanatory candidate under test
}


@dataclass
class EpisodicRecord:
    """Structured episodic memory record with provenance, confidence, and invalidation tracking."""

    record_id: str
    timestamp: float = field(default_factory=time.time)
    source: str = "observation"
    claim: str = ""
    evidence_type: str = "Observed"  # Observed, Derived, Predicted, Literature, Hypothesis
    confidence: float = 1.0          # Normalized scalar in [0.0, 1.0]
    invalidation_conditions: List[str] = field(default_factory=list)
    is_invalidated: bool = False
    invalidation_reason: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate record fields against semantic invariants."""
        errors: List[str] = []
        if not self.record_id or not isinstance(self.record_id, str) or not self.record_id.strip():
            errors.append("record_id must be a non-empty string.")
        if not self.claim or not isinstance(self.claim, str) or not self.claim.strip():
            errors.append("claim must be a non-empty string.")
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= float(self.confidence) <= 1.0):
            errors.append(f"confidence must be a float between 0.0 and 1.0, got: {self.confidence}")
        if self.evidence_type not in VALID_EVIDENCE_TYPES:
            errors.append(
                f"evidence_type '{self.evidence_type}' is invalid; must be one of {sorted(VALID_EVIDENCE_TYPES)}"
            )
        if not isinstance(self.timestamp, (int, float)) or self.timestamp <= 0:
            errors.append(f"timestamp must be a positive float, got: {self.timestamp}")
        if not isinstance(self.invalidation_conditions, list):
            errors.append("invalidation_conditions must be a list of strings.")
        if self.is_invalidated and (not self.invalidation_reason or not self.invalidation_reason.strip()):
            errors.append("invalidation_reason must be provided when is_invalidated is True.")
        return len(errors) == 0, errors

    def invalidate(self, reason: str) -> None:
        """Mark record as invalidated with an explicit causal rationale."""
        if not reason or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Invalidation reason must be a non-empty string.")
        self.is_invalidated = True
        self.invalidation_reason = reason.strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicRecord:
        """Construct record from dictionary with default fallbacks."""
        clean_data = dict(data)
        if "timestamp" not in clean_data or clean_data["timestamp"] is None:
            clean_data["timestamp"] = time.time()
        return cls(**clean_data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> EpisodicRecord:
        return cls.from_dict(json.loads(json_str))


class EpisodicMemoryStore:
    """Provenance-aware associative episodic memory store backed by Continuous Modern Hopfield dynamics.
    
    Features:
    - Imprinting claims into synaptic connection weights (W) via Hebbian updates (Delta W = eta * z z^T).
    - Recurrent attractor relaxation via Lyapunov energy descent for associative completion.
    - Anti-Hebbian unlearning (Delta W = -eta * v v^T) for targeted invalidation without collateral damage.
    - Active record filtering: excludes invalidated records unless explicitly requested.
    - Multi-attribute provenance filtering: by evidence_type, source, and confidence thresholds.
    - Graceful epistemic abstention when query cue does not converge to a trustworthy attractor.
    """

    def __init__(
        self,
        dim: int = 64,
        beta: float = 8.0,
        encoder: Optional[Any] = None,
        hopfield: Optional[Any] = None,
    ):
        self.dim = dim
        self.beta = beta
        self.encoder = encoder or NeuralFeatureEncoder(output_dim=dim)
        self.hopfield = hopfield or SynapticMemoryNetwork(dim=dim, beta=beta)
        self._records: dict[str, EpisodicRecord] = {}
        self._record_vectors: dict[str, torch.Tensor] = {}

    @property
    def records(self) -> dict[str, EpisodicRecord]:
        """Direct dictionary access to stored records."""
        return self._records

    def __len__(self) -> int:
        """Return the total number of records currently stored."""
        return len(self._records)

    def store(self, record: EpisodicRecord) -> str:
        """Store an episodic record and imprint its claim into synaptic memory."""
        is_valid, errors = record.validate()
        if not is_valid:
            raise ValueError(f"Invalid EpisodicRecord: {'; '.join(errors)}")

        # If record already exists, purge its existing Hopfield attractor to prevent duplicates
        if record.record_id in self._records:
            while True:
                res = self.hopfield.forget_pattern(record.record_id)
                if not isinstance(res, dict) or res.get("status") in ("not_found", "empty"):
                    break

        self._records[record.record_id] = record

        # Precompute and cache unit-normalized semantic vector
        vec = self.encoder.encode_text(record.claim)
        self._record_vectors[record.record_id] = vec

        # Learn pattern into Hopfield network if not already invalidated
        if not record.is_invalidated:
            self.hopfield.learn_pattern(
                pattern=vec,
                memory_id=record.record_id,
                metadata={
                    "record_id": record.record_id,
                    "evidence_type": record.evidence_type,
                    "confidence": float(record.confidence),
                    "source": record.source,
                    "claim": record.claim,
                },
                source_context=record.source,
            )
        return record.record_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.0,
        evidence_types: Optional[Union[str, Sequence[str]]] = None,
        sources: Optional[Union[str, Sequence[str]]] = None,
        include_invalidated: bool = False,
        relevance_threshold: float = 0.28,
        use_hopfield: bool = True,
    ) -> List[EpisodicRecord]:
        """Retrieve memories matching query cue with provenance, confidence, and invalidation filtering."""
        if not self._records or top_k <= 0:
            return []

        # Normalize filter criteria
        ev_types = {evidence_types} if isinstance(evidence_types, str) else set(evidence_types or [])
        srcs = {sources} if isinstance(sources, str) else set(sources or [])

        def passes_filters(rec: EpisodicRecord) -> bool:
            if not include_invalidated and rec.is_invalidated:
                return False
            if rec.confidence < min_confidence:
                return False
            if ev_types and rec.evidence_type not in ev_types:
                return False
            if srcs and rec.source not in srcs:
                return False
            return True

        candidates = [rec for rec in self._records.values() if passes_filters(rec)]
        if not candidates:
            return []

        q_vec = self.encoder.encode_text(query)
        cand_vecs = torch.stack(
            [self._record_vectors.get(r.record_id, self.encoder.encode_text(r.claim)) for r in candidates],
            dim=0,
        )
        cand_sims = F.cosine_similarity(q_vec.unsqueeze(0), cand_vecs, dim=-1)

        # Epistemic abstention: query cue must have baseline relevance to at least one candidate memory
        max_cand_sim = float(cand_sims.max().item())
        if max_cand_sim < relevance_threshold:
            return []

        # When Hopfield is enabled and we have active memories
        hopfield_scores_map: dict[str, float] = {}
        if use_hopfield and self.hopfield.num_memories > 0 and not include_invalidated:
            try:
                converged_state, trace, diag = self.hopfield.recall(q_vec)
                h_scores = torch.matmul(self.hopfield.memory_attractors, converged_state)
                for idx in range(self.hopfield.num_memories):
                    meta = self.hopfield.memory_metadata[idx]
                    rid = meta.get("record_id")
                    if rid:
                        hopfield_scores_map[rid] = float(h_scores[idx].item())
            except RuntimeError:
                pass

        scored = []
        for i, rec in enumerate(candidates):
            sim = float(cand_sims[i].item())
            if sim < relevance_threshold:
                continue
            h_sim = hopfield_scores_map.get(rec.record_id, sim)
            # Composite score: combination of direct semantic similarity and Hopfield attractor resonance,
            # scaled by record confidence
            composite = (0.5 * sim + 0.5 * max(h_sim, 0.0)) * rec.confidence
            scored.append((composite, sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[2] for item in scored[:top_k]]

    def invalidate(self, record_id: str, reason: str) -> bool:
        """Invalidate an episodic record and apply anti-Hebbian synaptic depression."""
        if record_id not in self._records:
            return False

        record = self._records[record_id]
        record.invalidate(reason)

        # Anti-Hebbian synaptic unlearning: weakens synaptic weight matrix Delta W = -eta v v^T
        # and removes the attractor basin from active memory_attractors
        self.hopfield.forget_pattern(record_id)
        return True

    def get(self, record_id: str) -> Optional[EpisodicRecord]:
        return self._records.get(record_id)

    def active_records(self) -> List[EpisodicRecord]:
        return [r for r in self._records.values() if not r.is_invalidated]

    def invalidated_records(self) -> List[EpisodicRecord]:
        return [r for r in self._records.values() if r.is_invalidated]

    def clear(self) -> None:
        self._records.clear()
        self._record_vectors.clear()
        self.hopfield = SynapticMemoryNetwork(dim=self.dim, beta=self.beta)

    def inspect_store(self) -> dict[str, Any]:
        """Diagnostic state inspection for monitoring and reporting."""
        return {
            "total_records": len(self._records),
            "active_records": len(self.active_records()),
            "invalidated_records": len(self.invalidated_records()),
            "hopfield_attractors": self.hopfield.num_memories,
            "synaptic_weights_norm": float(self.hopfield.synaptic_weights.norm().item()),
            "synaptic_sparsity": float((self.hopfield.synaptic_weights.abs() < 1e-4).float().mean().item()),
        }
