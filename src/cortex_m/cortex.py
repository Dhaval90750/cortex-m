"""CORTEX-M: Context-Oriented Routing and Trace Exchange for MIMO.

Canonical, version-free export for the entire CORTEX-M framework.
"""

from .cortex_v2 import CortexV2

# Unified canonical class
CortexM = CortexV2
CORTEX_M = CortexV2

__all__ = ["CORTEX_M", "CortexM", "CortexV2"]
