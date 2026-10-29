"""CORTEX-M Structured Pre-Output Gist Buffer.

Provides the 12-field typed dataclass schema, validation logic, serialization,
and versioned revision history tracking for inspectable pre-verbal gist states.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StructuredGist:
    """12-field inspectable pre-output gist buffer.

    Attributes
    ----------
    objective : str
        The primary operational goal or problem being solved.
    core_claim : str
        The central working hypothesis, assertion, or model conclusion.
    evidence : list[str]
        Empirical citations, sensor observations, or retrieved memory IDs.
    assumptions : list[str]
        Theoretical assumptions or boundary conditions under which the claim holds.
    constraints : list[str]
        Inviolable physical, mathematical, domain, or safety constraints.
    predictions : list[str]
        Deductive, falsifiable forecasts deduced from the core claim.
    uncertainty : float
        Epistemic uncertainty calibrated in the interval [0.0, 1.0].
    next_action : str
        The immediate discrete action, experiment, or reasoning step.
    contradictions : list[str]
        Known counter-evidence, paradoxes, or observations violating the claim.
    open_questions : list[str]
        Unresolved inquiries requiring targeted experimental probing.
    provenance_and_time : dict
        Metadata tracking origin, timestamps (created_at, updated_at), and author/session.
    revision_history : list[dict]
        Versioned audit trail of past claims, triggers, and revision reasons.
    """

    objective: str
    core_claim: str
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    uncertainty: float = 0.0
    next_action: str = "conclude"
    contradictions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    provenance_and_time: Dict[str, Any] = field(default_factory=dict)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure default provenance timestamps if empty."""
        if not self.provenance_and_time:
            now_iso = datetime.now(timezone.utc).isoformat()
            self.provenance_and_time = {
                "created_at": now_iso,
                "updated_at": now_iso,
                "source": "cortex_m_gist_generator",
                "version": "1.0.0",
            }
        elif isinstance(self.provenance_and_time, dict):
            now_iso = datetime.now(timezone.utc).isoformat()
            self.provenance_and_time.setdefault("created_at", now_iso)
            self.provenance_and_time.setdefault("source", "cortex_m_gist_generator")

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the structural, type, and semantic constraints of the gist buffer.

        Returns
        -------
        Tuple[bool, List[str]]
            A tuple of (is_valid, error_messages).
        """
        errors: List[str] = []

        # 1. Objective: non-empty string
        if not isinstance(self.objective, str) or not self.objective.strip():
            errors.append("Field 'objective' must be a non-empty string.")

        # 2. Core claim: non-empty string
        if not isinstance(self.core_claim, str) or not self.core_claim.strip():
            errors.append("Field 'core_claim' must be a non-empty string.")

        # 3. Evidence: list of strings
        if not isinstance(self.evidence, list) or not all(isinstance(x, str) for x in self.evidence):
            errors.append("Field 'evidence' must be a list of strings.")

        # 4. Assumptions: list of strings
        if not isinstance(self.assumptions, list) or not all(isinstance(x, str) for x in self.assumptions):
            errors.append("Field 'assumptions' must be a list of strings.")

        # 5. Constraints: list of strings
        if not isinstance(self.constraints, list) or not all(isinstance(x, str) for x in self.constraints):
            errors.append("Field 'constraints' must be a list of strings.")

        # 6. Predictions: list of strings
        if not isinstance(self.predictions, list) or not all(isinstance(x, str) for x in self.predictions):
            errors.append("Field 'predictions' must be a list of strings.")

        # 7. Uncertainty: float/int in [0.0, 1.0]
        if isinstance(self.uncertainty, bool) or not isinstance(self.uncertainty, (int, float)):
            errors.append(f"Field 'uncertainty' must be a float or int, got {type(self.uncertainty).__name__}.")
        elif not (0.0 <= float(self.uncertainty) <= 1.0):
            errors.append(f"Field 'uncertainty' must be between 0.0 and 1.0 (got {self.uncertainty}).")

        # 8. Next action: non-empty string
        if not isinstance(self.next_action, str) or not self.next_action.strip():
            errors.append("Field 'next_action' must be a non-empty string.")

        # 9. Contradictions: list of strings
        if not isinstance(self.contradictions, list) or not all(isinstance(x, str) for x in self.contradictions):
            errors.append("Field 'contradictions' must be a list of strings.")

        # 10. Open questions: list of strings
        if not isinstance(self.open_questions, list) or not all(isinstance(x, str) for x in self.open_questions):
            errors.append("Field 'open_questions' must be a list of strings.")

        # 11. Provenance and time: dictionary with required keys
        if not isinstance(self.provenance_and_time, dict):
            errors.append("Field 'provenance_and_time' must be a dictionary.")
        else:
            if "created_at" not in self.provenance_and_time:
                errors.append("Field 'provenance_and_time' missing required 'created_at' timestamp.")
            if "source" not in self.provenance_and_time:
                errors.append("Field 'provenance_and_time' missing required 'source' identifier.")

        # 12. Revision history: list of dictionaries
        if not isinstance(self.revision_history, list) or not all(isinstance(x, dict) for x in self.revision_history):
            errors.append("Field 'revision_history' must be a list of dictionaries.")
        else:
            for idx, rev in enumerate(self.revision_history):
                required_keys = {"revision_id", "timestamp", "previous_claim", "new_claim"}
                if not required_keys.issubset(rev.keys()):
                    missing = required_keys - set(rev.keys())
                    errors.append(f"Revision entry {idx} missing required keys: {sorted(missing)}.")

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the structured gist to a standard dictionary."""
        return {
            "objective": self.objective,
            "core_claim": self.core_claim,
            "evidence": list(self.evidence),
            "assumptions": list(self.assumptions),
            "constraints": list(self.constraints),
            "predictions": list(self.predictions),
            "uncertainty": float(self.uncertainty),
            "next_action": self.next_action,
            "contradictions": list(self.contradictions),
            "open_questions": list(self.open_questions),
            "provenance_and_time": copy.deepcopy(self.provenance_and_time),
            "revision_history": copy.deepcopy(self.revision_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructuredGist:
        """Construct a StructuredGist instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        def _safe_list(val: Any) -> Any:
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                return list(val)
            return val

        def _safe_dict(val: Any) -> Any:
            if val is None:
                return {}
            if isinstance(val, dict):
                return dict(val)
            return val

        def _safe_str(val: Any, default: str = "") -> Any:
            if val is None:
                return default
            if isinstance(val, str):
                return val
            return val

        return cls(
            objective=_safe_str(data.get("objective", "")),
            core_claim=_safe_str(data.get("core_claim", "")),
            evidence=_safe_list(data.get("evidence", [])),
            assumptions=_safe_list(data.get("assumptions", [])),
            constraints=_safe_list(data.get("constraints", [])),
            predictions=_safe_list(data.get("predictions", [])),
            uncertainty=data.get("uncertainty", 0.0),
            next_action=_safe_str(data.get("next_action"), default="conclude"),
            contradictions=_safe_list(data.get("contradictions", [])),
            open_questions=_safe_list(data.get("open_questions", [])),
            provenance_and_time=_safe_dict(data.get("provenance_and_time", {})),
            revision_history=_safe_list(data.get("revision_history", [])),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> StructuredGist:
        """Deserialize from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def revise(
        self,
        new_claim: str,
        reason_for_change: str,
        trigger_evidence: Optional[List[str]] = None,
        updated_uncertainty: Optional[float] = None,
        new_contradictions: Optional[List[str]] = None,
    ) -> StructuredGist:
        """Produce an updated StructuredGist with a new claim, preserving full audit history.

        Parameters
        ----------
        new_claim : str
            The new revised claim.
        reason_for_change : str
            Explanation of why the prior claim is updated/revised.
        trigger_evidence : Optional[List[str]]
            New observations or citations prompting this revision.
        updated_uncertainty : Optional[float]
            Updated epistemic uncertainty calibration.
        new_contradictions : Optional[List[str]]
            Any new contradictions identified.

        Returns
        -------
        StructuredGist
            A new instance with updated claim and appended revision history entry.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        rev_id = f"rev_{len(self.revision_history) + 1:03d}"

        rev_entry = {
            "revision_id": rev_id,
            "timestamp": now_iso,
            "previous_claim": self.core_claim,
            "new_claim": new_claim,
            "reason_for_change": reason_for_change,
            "trigger_evidence": list(trigger_evidence or []),
        }

        # Merge evidence
        merged_evidence = list(self.evidence)
        if trigger_evidence:
            for ev in trigger_evidence:
                if ev not in merged_evidence:
                    merged_evidence.append(ev)

        # Merge contradictions
        merged_contradictions = list(self.contradictions)
        if new_contradictions:
            for c in new_contradictions:
                if c not in merged_contradictions:
                    merged_contradictions.append(c)

        # Update provenance
        new_prov = copy.deepcopy(self.provenance_and_time)
        new_prov["updated_at"] = now_iso
        new_prov["revision_count"] = len(self.revision_history) + 1

        new_history = copy.deepcopy(self.revision_history) + [rev_entry]

        revised = StructuredGist(
            objective=self.objective,
            core_claim=new_claim,
            evidence=merged_evidence,
            assumptions=list(self.assumptions),
            constraints=list(self.constraints),
            predictions=list(self.predictions),
            uncertainty=self.uncertainty if updated_uncertainty is None else float(updated_uncertainty),
            next_action=self.next_action,
            contradictions=merged_contradictions,
            open_questions=list(self.open_questions),
            provenance_and_time=new_prov,
            revision_history=new_history,
        )
        is_valid, errors = revised.validate()
        if not is_valid:
            raise ValueError(f"Revised StructuredGist failed validation: {'; '.join(errors)}")
        return revised

    def revise_claim(self, new_claim: str, contradiction_note: Optional[str] = None) -> None:
        """In-place claim revision method tracking audit history."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self.revision_history.append({
            "prior_claim": self.core_claim,
            "timestamp": now_iso,
            "reason": contradiction_note or "Updated evidence",
        })
        if contradiction_note and contradiction_note not in self.contradictions:
            self.contradictions.append(contradiction_note)
        self.core_claim = new_claim

    def format_summary(self) -> str:
        """Render a concise, model-visible summary suitable for downstream prompt conditioning."""
        ev_str = "; ".join(self.evidence) if self.evidence else "None"
        con_str = "; ".join(self.constraints) if self.constraints else "None"
        ass_str = "; ".join(self.assumptions) if self.assumptions else "None"
        contra_str = "; ".join(self.contradictions) if self.contradictions else "None"
        return (
            f"[GIST OBJECTIVE]: {self.objective}\n"
            f"[CORE CLAIM]: {self.core_claim} (Uncertainty: {self.uncertainty:.2f})\n"
            f"[KEY EVIDENCE]: {ev_str}\n"
            f"[CONSTRAINTS]: {con_str}\n"
            f"[ASSUMPTIONS]: {ass_str}\n"
            f"[CONTRADICTIONS]: {contra_str}\n"
            f"[NEXT ACTION]: {self.next_action}"
        )
