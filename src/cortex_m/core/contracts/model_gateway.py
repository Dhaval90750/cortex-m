"""
CORTEX-M Model Gateway Contracts.
Defines interfaces for model backends (Local SLM, Native CortexLM, Hosted APIs).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class ModelCapabilities:
    model_id: str
    context_window: int
    supports_streaming: bool = True
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supports_vision: bool = False
    is_local: bool = True
    parameters_billions: Optional[float] = None


@dataclass
class ModelRequest:
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    stop_sequences: List[str] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    text: str
    model_id: str
    latency_ms: float
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"


class ModelGatewayContract(ABC):
    """Abstract contract for model execution."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResult:
        """Synchronously execute a model completion."""
        pass

    @abstractmethod
    def capabilities(self, model_id: str) -> ModelCapabilities:
        """Return the documented capabilities for the target model."""
        pass

    @abstractmethod
    def list_available_models(self) -> List[ModelCapabilities]:
        """List all available and registered model backends."""
        pass
