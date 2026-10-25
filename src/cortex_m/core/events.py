"""
CORTEX-M Standardized Event Bus & Observability Schemas.
Implements the unified event lifecycle: task.started, model.requested, tool.completed, etc.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    # Task lifecycle
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_PLAN_UPDATED = "task.plan_updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"

    # Model & gateway
    MODEL_REQUESTED = "model.requested"
    MODEL_RESPONDED = "model.responded"
    MODEL_STREAMING_CHUNK = "model.streaming_chunk"
    MODEL_FAILED = "model.failed"

    # Memory & knowledge
    MEMORY_STORED = "memory.stored"
    MEMORY_RETRIEVED = "memory.retrieved"
    MEMORY_DELETED = "memory.deleted"
    EVIDENCE_VERIFIED = "evidence.verified"

    # Tools & execution
    TOOL_REQUESTED = "tool.requested"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Security & policy
    POLICY_EVALUATED = "policy.evaluated"
    POLICY_DENIED = "policy.denied"
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"


@dataclass
class CortexEvent:
    event_type: EventType
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    actor: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "actor": self.actor,
            "payload": self.payload,
        }


class EventBus:
    """In-process pub/sub event bus with in-memory trace history."""

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, List[Callable[[CortexEvent], None]]] = {}
        self._history: List[CortexEvent] = []
        self._max_history = max_history

    def subscribe(self, event_type: str, handler: Callable[[CortexEvent], None]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event: CortexEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        type_key = event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type)
        # Specific handlers
        for handler in self._subscribers.get(type_key, []):
            try:
                handler(event)
            except Exception:
                pass
        # Wildcard handlers
        for handler in self._subscribers.get("*", []):
            try:
                handler(event)
            except Exception:
                pass

    def get_history(self, task_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        events = self._history
        if task_id:
            events = [e for e in events if e.task_id == task_id]
        return [e.to_dict() for e in events[-limit:]]
