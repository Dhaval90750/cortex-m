"""
CORTEX-M Memory Provider Contracts.
Abstracts associative memory (CORTEX-Synapse), working memory, and durable storage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    memory_id: str
    raw_text: str
    tenant_id: str = "default_user"
    source: str = "user"
    timestamp: float = 0.0
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    attractor_energy: Optional[float] = None


@dataclass
class MemoryQuery:
    query_text: str
    tenant_id: str = "default_user"
    top_k: int = 5
    threshold: float = 0.50
    require_grounding: bool = True


@dataclass
class MemoryRecallResult:
    recalled_records: List[MemoryRecord]
    confidence_score: float
    is_grounded: bool
    evidence_status: str  # VERIFIED, PARTIALLY_SUPPORTED, CONFLICTING_EVIDENCE, INSUFFICIENT_EVIDENCE
    attractor_displacement: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class MemoryProviderContract(ABC):
    """Abstract contract for memory systems."""

    @abstractmethod
    def store(self, record: MemoryRecord) -> str:
        """Store or imprint a memory record into the substrate. Returns memory_id."""
        pass

    @abstractmethod
    def recall(self, query: MemoryQuery) -> MemoryRecallResult:
        """Perform associative or semantic recall against the memory substrate."""
        pass

    @abstractmethod
    def delete(self, memory_id: str, tenant_id: str) -> bool:
        """Perform unlearning or deletion with zero collateral damage."""
        pass

    @abstractmethod
    def list_memories(self, tenant_id: str, limit: int = 100) -> List[MemoryRecord]:
        """List active memories stored for the specified tenant scope."""
        pass
