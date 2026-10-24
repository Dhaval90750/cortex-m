"""Cortex-M Multi-Agent Task Completion & Industrial Coding System."""

from .blackboard import AgentMessage, AgentRole, SynapticBlackboard
from .code_synthesizer import DynamicCodeSynthesizer
from .coding_agent import CodeArtifact, CodeReview, CortexCodingAgent
from .task_completer import MultiAgentTaskCompleter, TaskExecutionResult

# Canonical aliases
CortexCode = CortexCodingAgent
CortexCoder = CortexCodingAgent

__all__ = [
    "AgentRole",
    "AgentMessage",
    "SynapticBlackboard",
    "CodeArtifact",
    "CodeReview",
    "CortexCodingAgent",
    "CortexCode",
    "CortexCoder",
    "DynamicCodeSynthesizer",
    "MultiAgentTaskCompleter",
    "TaskExecutionResult",
]
