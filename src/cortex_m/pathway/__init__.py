"""Pathway memory and synaptic routing architecture for Cortex-M.

Realizes connection-based memory (no hard disk), activation trajectory tracing,
learned pre-verbal gist bottleneck, causal MIMO sparse routing,
hierarchical scaling (N=16, 32), biological sleep consolidation,
and working memory dynamics.
"""

from .causal_mimo import CausalPathwayMIMO
from .consolidation import SynapticConsolidator
from .encoder import NeuralFeatureEncoder
from .gist_bottleneck import GistBottleneck
from .gist_generator import GistRealizer
from .hierarchical_router import HierarchicalSparseRouter
from .mimo_pathway import MIMOPathwaySystem, SynapticConditionedMIMO
from .synaptic_memory import SynapticMemoryNetwork
from .tracer import PathwayStep, PathwayTrace, PathwayTracer
from .working_memory import WorkingMemoryBuffer

__all__ = [
    "NeuralFeatureEncoder",
    "SynapticMemoryNetwork",
    "PathwayTracer",
    "PathwayTrace",
    "PathwayStep",
    "GistBottleneck",
    "GistRealizer",
    "CausalPathwayMIMO",
    "HierarchicalSparseRouter",
    "SynapticConsolidator",
    "WorkingMemoryBuffer",
    "SynapticConditionedMIMO",
    "MIMOPathwaySystem",
]
