"""
CORTEX-M Policy & Trust Plane Contracts.
Implements Layer 4 independent policy engine and authorization verification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class PolicyDecisionType(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class PolicyContext:
    actor_id: str
    tenant_id: str
    session_id: str
    task_id: Optional[str] = None
    environment: str = "local"  # local, test, staging, production
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    decision: PolicyDecisionType
    reason: str
    risk_level: str
    approval_token: Optional[str] = None
    restrictions: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.decision == PolicyDecisionType.ALLOW


class PolicyEngineContract(ABC):
    """Abstract contract for policy enforcement."""

    @abstractmethod
    def evaluate(self, action: str, resource: str, context: PolicyContext) -> PolicyDecision:
        """Validate if the proposed action on the resource is authorized."""
        pass

    @abstractmethod
    def record_approval(self, approval_token: str, approved_by: str) -> bool:
        """Record an explicit human or policy approval token."""
        pass
