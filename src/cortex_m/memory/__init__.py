"""CORTEX-M Episodic and Associative Memory Subsystem.

Provides structured episodic memory storage, provenance-aware filtering,
invalidation tracking, and continuous Modern Hopfield network integration.
"""

from __future__ import annotations

from .episodic_store import EpisodicRecord, EpisodicMemoryStore

# Preserve 100% backward compatibility with legacy tests
try:
    from cortex_m.legacy_v1.memory import (
        CortexMemory,
    )
except ImportError:
    pass

__all__ = [
    "EpisodicRecord",
    "EpisodicMemoryStore",
    "CortexMemory",
]
