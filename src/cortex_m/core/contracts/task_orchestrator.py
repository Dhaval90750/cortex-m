"""
CORTEX-M Task Orchestrator Contracts.
Implements bounded state machine, explicit milestones, budgets, and recovery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_INPUT = "waiting_for_input"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskPlan:
    milestones: List[str]
    current_step: int = 0
    estimated_steps: int = 5


@dataclass
class TaskRecord:
    task_id: str
    objective: str
    status: TaskStatus
    plan: TaskPlan
    created_at: float
    updated_at: float
    tenant_id: str = "default_user"
    budget_max_steps: int = 25
    budget_timeout_seconds: float = 300.0
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class TaskOrchestratorContract(ABC):
    """Abstract contract for task lifecycle management."""

    @abstractmethod
    def submit_task(self, objective: str, tenant_id: str) -> TaskRecord:
        """Submit a new task into the bounded state machine."""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch current task state, plan, and artifacts."""
        pass

    @abstractmethod
    def step_task(self, task_id: str) -> TaskRecord:
        """Advance the task by one planned execution step."""
        pass

    @abstractmethod
    def cancel_task(self, task_id: str, reason: str) -> bool:
        """Gracefully interrupt and cancel an active task."""
        pass
