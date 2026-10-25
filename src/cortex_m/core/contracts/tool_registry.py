"""
CORTEX-M Universal Tool Plane Contracts.
Defines schemas, risk classifications, and deterministic execution contracts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ToolRiskLevel(str, Enum):
    LOW = "low"            # Read permitted files, arithmetic, calendar
    MODERATE = "moderate"  # Modify file in local project workspace
    HIGH = "high"          # Execute code in sandbox, install packages, network egress
    CRITICAL = "critical"  # Delete datasets, publish, deploy, push to production


@dataclass
class ToolDefinition:
    name: str
    description: str
    version: str = "1.0.0"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    requires_approval: bool = False
    timeout_seconds: float = 5.0
    is_idempotent: bool = True


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    artifacts_produced: List[str] = field(default_factory=list)


class ToolRegistryContract(ABC):
    """Abstract registry for tools across all interfaces."""

    @abstractmethod
    def register_tool(self, definition: ToolDefinition, handler: Callable[..., Any]) -> None:
        """Register a new tool with its definition and execution handler."""
        pass

    @abstractmethod
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Retrieve tool metadata by name."""
        pass

    @abstractmethod
    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        pass

    @abstractmethod
    def execute(self, tool_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """Safely execute a registered tool under policy supervision."""
        pass
