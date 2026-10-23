from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MemoryStatus = Literal["active", "conflicted", "superseded"]
Action = Literal["answer", "clarify", "abstain", "fallback"]


@dataclass(frozen=True)
class ConversationTurn:
    turn_index: int
    role: Literal["user", "assistant", "system"]
    text: str


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    key: str
    value: str
    source_turn: int
    evidence_text: str
    explicitness: Literal["explicit", "inferred"] = "explicit"
    confidence: float = 1.0
    correction: bool = False
    status: MemoryStatus = "active"
    conflict_group: str | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class EvidenceRef:
    fact: FactRecord
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    evidence: tuple[EvidenceRef, ...]
    status: Literal["ok", "empty", "unavailable"]
    query: str


@dataclass(frozen=True)
class WorkingGist:
    current_goal: str
    active_topics: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    unresolved_questions: tuple[str, ...]
    approximate_tokens: int


@dataclass(frozen=True)
class ResponsePlan:
    action: Action
    goal: str
    evidence: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    uncertainty: str | None = None
    outline: str = ""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    backend_id: str
    stop_reason: str = "unknown"
