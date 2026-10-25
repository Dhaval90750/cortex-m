"""
CORTEX-M Core Subsystem.
Unified architecture, configuration, contracts, events, and runtime.
"""

from .config import CortexConfig, DeploymentProfile, MemoryConfig, ModelConfig, ToolsConfig
from .contracts import (
    MemoryProviderContract,
    MemoryQuery,
    MemoryRecallResult,
    MemoryRecord,
    ModelCapabilities,
    ModelGatewayContract,
    ModelRequest,
    ModelResult,
    PolicyContext,
    PolicyDecision,
    PolicyDecisionType,
    PolicyEngineContract,
    TaskOrchestratorContract,
    TaskPlan,
    TaskRecord,
    TaskStatus,
    ToolDefinition,
    ToolRegistryContract,
    ToolResult,
    ToolRiskLevel,
)
from .errors import (
    ConfigurationError,
    CortexError,
    EpistemicAbstentionError,
    MemoryRecallError,
    ModelGatewayError,
    PolicyViolationError,
    SandboxTimeoutError,
    ToolExecutionError,
)
from .events import CortexEvent, EventBus, EventType
from .runtime import CortexRuntime, DefaultPolicyEngine, SynapseMemoryProvider

__all__ = [
    "CortexConfig",
    "DeploymentProfile",
    "MemoryConfig",
    "ModelConfig",
    "ToolsConfig",
    "CortexError",
    "ConfigurationError",
    "PolicyViolationError",
    "EpistemicAbstentionError",
    "ToolExecutionError",
    "SandboxTimeoutError",
    "MemoryRecallError",
    "ModelGatewayError",
    "CortexEvent",
    "EventType",
    "EventBus",
    "CortexRuntime",
    "SynapseMemoryProvider",
    "DefaultPolicyEngine",
    # Contracts
    "ModelGatewayContract",
    "ModelRequest",
    "ModelResult",
    "ModelCapabilities",
    "MemoryProviderContract",
    "MemoryRecord",
    "MemoryQuery",
    "MemoryRecallResult",
    "ToolRegistryContract",
    "ToolDefinition",
    "ToolResult",
    "ToolRiskLevel",
    "PolicyEngineContract",
    "PolicyDecision",
    "PolicyDecisionType",
    "PolicyContext",
    "TaskOrchestratorContract",
    "TaskRecord",
    "TaskPlan",
    "TaskStatus",
]
