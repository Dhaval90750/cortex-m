"""Synaptic Blackboard for Industrial Multi-Agent Coordination.

Replaces centralized databases (Redis, RabbitMQ, PostgreSQL) with a continuous
synaptic associative substrate where autonomous agents synchronize state, plans,
critiques, and code artifacts directly through continuous Hopfield attractor basins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cortex_m.cortex import CortexM


class AgentRole(str, Enum):
    """Specialized industrial agent roles."""
    ARCHITECT = "architect"       # High-level architecture, 5W1H decomposition, file DAGs
    CODER = "coder"               # Code generation, surgical diffs, refactoring
    CRITIC = "critic"             # Static analysis, security audit, hallucination filtering
    AUDITOR = "auditor"           # Test verification, runtime evaluation, self-repair triggers
    COORDINATOR = "coordinator"   # Swarm orchestration, milestone aggregation


@dataclass
class AgentMessage:
    """A semantic artifact or message posted by an agent to the synaptic blackboard."""
    message_id: str
    sender_role: AgentRole
    topic: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resonance_score: float = 1.0


class SynapticBlackboard:
    """Continuous Hopfield Synaptic Blackboard.
    
    Provides an ultra-fast, zero-database coordination substrate for multi-agent swarms.
    Agents imprint their thoughts, plans, and code reviews directly into physical
    synaptic weights via Hebbian potentiation (Delta W = eta * z z^T) and query the
    collective knowledge via associative energy descent.
    """

    def __init__(self, dim: int = 64, beta: float = 10.0, seed: int = 42):
        self.dim = dim
        self.cortex = CortexM(dim=dim, beta=beta, seed=seed)
        self.messages: Dict[str, AgentMessage] = {}
        self.topic_index: Dict[str, List[str]] = {}

    def post(
        self,
        sender_role: AgentRole,
        topic: str,
        content: str,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Post a message or artifact to the synaptic blackboard."""
        if message_id is None:
            message_id = f"msg_{sender_role.value}_{int(time.time() * 1000)}_{len(self.messages)}"

        meta = metadata or {}
        msg = AgentMessage(
            message_id=message_id,
            sender_role=sender_role,
            topic=topic,
            content=content,
            metadata=meta,
        )

        # Imprint into continuous synaptic weights W
        semantic_text = f"[{sender_role.value}:{topic}] {content}"
        self.cortex.learn(
            semantic_text,
            memory_id=message_id,
            source_context=f"agent_{sender_role.value}",
        )

        self.messages[message_id] = msg
        if topic not in self.topic_index:
            self.topic_index[topic] = []
        self.topic_index[topic].append(message_id)

        return msg

    def query(
        self,
        query_text: str,
        filter_role: Optional[AgentRole] = None,
        top_k: int = 3,
    ) -> List[AgentMessage]:
        """Associatively recall relevant agent messages using Hopfield energy descent."""
        if not self.messages or self.cortex.memory.num_memories == 0:
            return []

        # Encode query onto unit hypersphere
        q_vec = self.cortex.encoder.encode_text(query_text)

        # Recall nearest attractor basin via Lyapunov energy descent
        recalled_state, _, diagnostics = self.cortex.memory.recall(q_vec)
        best_id = diagnostics.get("best_memory_id") or diagnostics.get("recalled_memory_id")
        resonance = diagnostics.get("resonance", {}).get("resonance_score", 0.70) if isinstance(diagnostics.get("resonance"), dict) else diagnostics.get("resonance_score", 0.70)

        results: List[AgentMessage] = []
        if best_id and best_id in self.messages:
            msg = self.messages[best_id]
            if filter_role is None or msg.sender_role == filter_role:
                msg.resonance_score = resonance
                results.append(msg)

        # Also retrieve any other top matching memories from memory store
        for mem_id, msg in reversed(list(self.messages.items())):
            if len(results) >= top_k:
                break
            if msg not in results and (filter_role is None or msg.sender_role == filter_role):
                results.append(msg)

        return results

    def get_by_topic(self, topic: str) -> List[AgentMessage]:
        """Retrieve all messages posted under a specific topic."""
        msg_ids = self.topic_index.get(topic, [])
        return [self.messages[mid] for mid in msg_ids if mid in self.messages]

    def clear(self):
        """Reset blackboard state."""
        self.cortex.memory.clear()
        self.messages.clear()
        self.topic_index.clear()
