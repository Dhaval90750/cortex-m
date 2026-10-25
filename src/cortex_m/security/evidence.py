"""Structured Evidence Verification Pipeline for Cortex-M.

Transitions from raw scalar Lyapunov energy / binary hallucination flags
to an epistemically calibrated evidence verification framework:
- VERIFIED: Grounded in stable Lyapunov attractor basin with high semantic resonance.
- PARTIALLY_SUPPORTED: Partial semantic alignment; some claims lack direct attractor grounding.
- CONFLICTING_EVIDENCE: Multiple competing attractors with contradictory claims.
- INSUFFICIENT_EVIDENCE (ABSTAIN): Unstable energy trajectory; out-of-distribution input.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class EvidenceStatus(str, enum.Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class GroundedCitation:
    """Attributed source citation from physical synaptic memory."""
    memory_id: str
    source: str
    raw_text: str
    resonance_score: float
    attractor_distance: float
    timestamp: Optional[str] = None
    version: int = 1


@dataclass
class EvidenceVerificationReport:
    """Comprehensive evidence verification result."""
    status: EvidenceStatus
    confidence_score: float  # Calibrated [0.0, 1.0]
    lyapunov_energy: float
    resonance_score: float
    energy_drop: float
    convergence_steps: int
    grounded_citations: List[GroundedCitation] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    competing_attractors: List[Dict[str, Any]] = field(default_factory=list)
    abstention_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence_score": round(self.confidence_score, 4),
            "lyapunov_energy": round(self.lyapunov_energy, 4),
            "resonance_score": round(self.resonance_score, 4),
            "energy_drop": round(self.energy_drop, 4),
            "convergence_steps": self.convergence_steps,
            "grounded_citations": [
                {
                    "memory_id": c.memory_id,
                    "source": c.source,
                    "raw_text": c.raw_text,
                    "resonance_score": round(c.resonance_score, 4),
                    "attractor_distance": round(c.attractor_distance, 4),
                    "timestamp": c.timestamp,
                    "version": c.version,
                }
                for c in self.grounded_citations
            ],
            "unsupported_claims": self.unsupported_claims,
            "competing_attractors": self.competing_attractors,
            "abstention_rationale": self.abstention_rationale,
            "is_grounded": self.status in (EvidenceStatus.VERIFIED, EvidenceStatus.PARTIALLY_SUPPORTED),
        }


class EvidenceVerifier:
    """Epistemic evidence verifier evaluating attractor dynamics and claim support."""

    def __init__(
        self,
        energy_threshold: float = 0.50,
        resonance_verified_threshold: float = 0.70,
        resonance_partial_threshold: float = 0.45,
        conflict_margin: float = 0.08,
    ):
        self.energy_threshold = energy_threshold
        self.resonance_verified_threshold = resonance_verified_threshold
        self.resonance_partial_threshold = resonance_partial_threshold
        self.conflict_margin = conflict_margin

    def verify(
        self,
        query: str,
        best_metadata: Optional[Dict[str, Any]],
        attractor_similarity: float,
        resonance: Dict[str, Any],
        steps_taken: int,
        energy_drop: float,
        final_energy: float = 0.0,
        alternative_candidates: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> EvidenceVerificationReport:
        """Evaluate evidence status and calibrate confidence."""
        res_score = float(resonance.get("resonance_score", attractor_similarity))
        dominant_source = str(resonance.get("dominant_source", "synaptic_attractor"))

        # Detect competing conflicting attractors
        competing: List[Dict[str, Any]] = []
        is_conflict = False
        if alternative_candidates and len(alternative_candidates) > 1:
            top_sim = attractor_similarity
            for cand in alternative_candidates[1:4]:
                cand_sim = float(cand.get("similarity", 0.0))
                if abs(top_sim - cand_sim) <= self.conflict_margin and cand_sim > self.resonance_partial_threshold:
                    competing.append(cand)
                    # Check if text content conflicts
                    cand_text = str(cand.get("raw_text", ""))
                    best_text = str((best_metadata or {}).get("raw_text", ""))
                    if cand_text and best_text and cand_text.lower() != best_text.lower():
                        is_conflict = True

        # Check for empty or non-existent metadata
        if not best_metadata or attractor_similarity < self.resonance_partial_threshold:
            return EvidenceVerificationReport(
                status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                lyapunov_energy=final_energy,
                resonance_score=res_score,
                energy_drop=energy_drop,
                convergence_steps=steps_taken,
                grounded_citations=[],
                unsupported_claims=[query],
                competing_attractors=competing,
                abstention_rationale=f"Attractor similarity ({attractor_similarity:.3f}) fell below minimum grounding threshold ({self.resonance_partial_threshold:.3f}).",
            )

        raw_text = str(best_metadata.get("raw_text", ""))
        mem_id = str(best_metadata.get("memory_id", "unindexed"))
        source = str(best_metadata.get("source", dominant_source))

        citation = GroundedCitation(
            memory_id=mem_id,
            source=source,
            raw_text=raw_text,
            resonance_score=res_score,
            attractor_distance=1.0 - attractor_similarity,
            timestamp=best_metadata.get("timestamp"),
            version=int(best_metadata.get("version", 1)),
        )

        if is_conflict:
            return EvidenceVerificationReport(
                status=EvidenceStatus.CONFLICTING_EVIDENCE,
                confidence_score=0.40,
                lyapunov_energy=final_energy,
                resonance_score=res_score,
                energy_drop=energy_drop,
                convergence_steps=steps_taken,
                grounded_citations=[citation],
                unsupported_claims=[],
                competing_attractors=competing,
                abstention_rationale="Multiple competing memory attractors with contradictory claims detected within conflict margin.",
            )

        if res_score >= self.resonance_verified_threshold and steps_taken <= 15:
            # High confidence verified evidence
            confidence = min(0.99, 0.70 + 0.30 * (res_score - self.resonance_verified_threshold) / (1.0 - self.resonance_verified_threshold + 1e-6))
            return EvidenceVerificationReport(
                status=EvidenceStatus.VERIFIED,
                confidence_score=confidence,
                lyapunov_energy=final_energy,
                resonance_score=res_score,
                energy_drop=energy_drop,
                convergence_steps=steps_taken,
                grounded_citations=[citation],
                unsupported_claims=[],
                competing_attractors=competing,
                abstention_rationale=None,
            )

        # Partially supported
        confidence = 0.45 + 0.25 * ((res_score - self.resonance_partial_threshold) / (self.resonance_verified_threshold - self.resonance_partial_threshold + 1e-6))
        return EvidenceVerificationReport(
            status=EvidenceStatus.PARTIALLY_SUPPORTED,
            confidence_score=confidence,
            lyapunov_energy=final_energy,
            resonance_score=res_score,
            energy_drop=energy_drop,
            convergence_steps=steps_taken,
            grounded_citations=[citation],
            unsupported_claims=["Partial semantic overlap with stored memory"],
            competing_attractors=competing,
            abstention_rationale="Moderate resonance; fact is partially inferred from nearby attractor state.",
        )
