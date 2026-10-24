"""Cortex-M: Unified Continuous Attractor Substrate.

Clean, version-free export for the entire Cortex-M framework.
"""

from .cortex_v2 import CortexV2

# Unified canonical class
CortexM = CortexV2

__all__ = ["CortexM", "CortexV2"]
