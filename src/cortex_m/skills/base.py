"""Base Skill Contract for Cortex-M and CortexCode.

Provides unified interface for modular cognitive and software engineering skills.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillResult:
    """Standardized output of a CORTEX-M skill execution."""
    skill_name: str
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    resonance_score: float = 0.85
    findings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill_name,
            "success": self.success,
            "output": self.output,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "resonance_score": self.resonance_score,
            "findings": self.findings,
            "suggestions": self.suggestions,
        }


class BaseSkill(ABC):
    """Abstract base class for all CORTEX-M modular skills."""

    def __init__(
        self,
        name: str,
        description: str,
        category: str = "general",
        cortex: Optional[Any] = None,
    ):
        self.name = name
        self.description = description
        self.category = category  # "planning", "asking", "suggesting", "grilling", "scheduling", "goal", "alignment", "security", "testing", etc.
        self.cortex = cortex

    def set_cortex(self, cortex: Any) -> None:
        """Inject physical CORTEX-M substrate into skill."""
        self.cortex = cortex

    @abstractmethod
    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute skill logic on target input string (code, query, or schema)."""
        pass

    def __repr__(self) -> str:
        return f"<Skill '{self.name}' [{self.category.upper()}]: {self.description[:40]}...>"
