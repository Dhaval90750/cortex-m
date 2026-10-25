"""
CORTEX-M Core Framework Exceptions.
Standardized domain exception hierarchy for the unified runtime.
"""

from typing import Any, Dict, Optional


class CortexError(Exception):
    """Base exception for all CORTEX-M runtime errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(CortexError):
    """Raised when runtime or profile configuration is invalid."""
    pass


class PolicyViolationError(CortexError):
    """Raised when an operation violates active security or authorization policy."""
    pass


class EpistemicAbstentionError(CortexError):
    """Raised when confidence is below calibrated threshold and system must abstain."""
    pass


class ToolExecutionError(CortexError):
    """Raised when a registered tool fails during execution."""
    pass


class SandboxTimeoutError(ToolExecutionError):
    """Raised when sandboxed execution exceeds the hard real-time limit."""
    pass


class MemoryRecallError(CortexError):
    """Raised when synaptic or episodic recall encounters an unrecoverable failure."""
    pass


class ModelGatewayError(CortexError):
    """Raised when an external model backend or local inference fails."""
    pass
