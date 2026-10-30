"""CORTEX-M Training Data Ecosystem & Synthetic Data Factory."""

from .synthetic import (
    CMSciSimulator,
    CMGistDriftSimulator,
    CMMemoryEpisodeGenerator,
    generate_discovery_batch,
    DiscoveryEpisode,
)

__all__ = [
    "CMSciSimulator",
    "CMGistDriftSimulator",
    "CMMemoryEpisodeGenerator",
    "generate_discovery_batch",
    "DiscoveryEpisode",
]

