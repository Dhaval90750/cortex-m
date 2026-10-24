"""CORTEX-M: Context-Oriented Routing and Trace Exchange for MIMO.

Continuous neural architecture for associative memory, causal MIMO sparse routing,
and zero-database cognitive intelligence.
"""

from .analytics.cognitive_analyzer import CognitiveAnalyzer
from .cortex_v2 import CortexV2

# Unified, canonical CORTEX-M class (Zero versioning)
CortexM = CortexV2
CORTEX_M = CortexV2

from .legacy_v1.memory import CortexMemory
from .legacy_v1.store import MemoryStore
from .pathway.causal_mimo import CausalPathwayMIMO
from .pathway.consolidation import SynapticConsolidator
from .pathway.dual_stream import DualStreamProcessor
from .pathway.encoder import NeuralFeatureEncoder
from .pathway.gist_bottleneck import GistBottleneck
from .pathway.gist_generator import GistRealizer
from .pathway.hierarchical_router import HierarchicalSparseRouter
from .pathway.mimo_pathway import MIMOPathwaySystem
from .pathway.synaptic_memory import SynapticMemoryNetwork
from .pathway.tenant_paging import SynapticTenantManager
from .pathway.tracer import PathwayTrace, PathwayTracer
from .pathway.working_memory import WorkingMemoryBuffer
from .model.cortex_lm import CortexLM, CortexLMOutput
from .model.real_data_trainer import RealDataTrainer, EVERYDAY_USER_KNOWLEDGE
from .database.db_trainer import DatabaseKnowledgeTrainer, DatabaseSchemaMetadata
from .agents import (
    AgentMessage,
    AgentRole,
    CodeArtifact,
    CodeReview,
    CortexCode,
    CortexCoder,
    CortexCodingAgent,
    DynamicCodeSynthesizer,
    MultiAgentTaskCompleter,
    SynapticBlackboard,
    TaskExecutionResult,
)
from .model.conversational_cognition import ConversationalCognition, ConversationalIntent
from .sdk.engine import CortexEngine, RecallResult, TenantSession
from .sdk.llm_bridge import CortexLLMBridge
from .security.critic import NeuromodulatorySafetyCritic

__all__ = [
    "CORTEX_M",
    "CortexM",
    "CortexLM",
    "CortexLMOutput",
    "CortexCode",
    "CortexCoder",
    "CortexCodingAgent",
    "MultiAgentTaskCompleter",
    "SynapticBlackboard",
    "AgentRole",
    "AgentMessage",
    "CodeArtifact",
    "CodeReview",
    "TaskExecutionResult",
    "RealDataTrainer",
    "EVERYDAY_USER_KNOWLEDGE",
    "DatabaseKnowledgeTrainer",
    "DatabaseSchemaMetadata",
    "CortexEngine",
    "CortexLLMBridge",
    "CognitiveAnalyzer",
    "RecallResult",
    "TenantSession",
    "CortexV2",
    "NeuralFeatureEncoder",
    "SynapticMemoryNetwork",
    "PathwayTracer",
    "PathwayTrace",
    "GistBottleneck",
    "GistRealizer",
    "CausalPathwayMIMO",
    "HierarchicalSparseRouter",
    "SynapticConsolidator",
    "WorkingMemoryBuffer",
    "DualStreamProcessor",
    "SynapticTenantManager",
    "NeuromodulatorySafetyCritic",
    "MIMOPathwaySystem",
    "CortexMemory",
    "MemoryStore",
]

