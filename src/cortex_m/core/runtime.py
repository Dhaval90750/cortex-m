"""
CORTEX-M Core Runtime.
Unified coordinator linking the model plane, memory plane, tool plane, and trust plane.
"""

import time
from typing import Any, Dict, List, Optional

from cortex_m.cortex_v2 import CortexV2
from cortex_m.model.cortex_lm import CortexLM
from cortex_m.security.evidence import EvidenceStatus, EvidenceVerifier

from .config import CortexConfig
from .contracts import (
    MemoryProviderContract,
    MemoryQuery,
    MemoryRecallResult,
    MemoryRecord,
    ModelCapabilities,
    ModelGatewayContract,
    ModelRequest,
    ModelResult,
    PolicyContext,
    PolicyDecision,
    PolicyDecisionType,
    PolicyEngineContract,
    ToolDefinition,
    ToolRegistryContract,
    ToolResult,
    ToolRiskLevel,
)
from .errors import ConfigurationError, PolicyViolationError
from .events import CortexEvent, EventBus, EventType


class SynapseMemoryProvider(MemoryProviderContract):
    """MemoryProvider adapter wrapping CORTEX-Synapse / CortexV2."""

    def __init__(self, cortex: CortexV2):
        self._cortex = cortex

    def store(self, record: MemoryRecord) -> str:
        if record.tenant_id and record.tenant_id != "default_user":
            self._cortex.learn_for_tenant(record.tenant_id, record.raw_text)
        else:
            self._cortex.learn(
                record.raw_text,
                memory_id=record.memory_id,
                source_context="global",
                metadata={"tenant_id": record.tenant_id, **record.metadata},
            )
        return record.memory_id

    def recall(self, query: MemoryQuery) -> MemoryRecallResult:
        res = self._cortex.recall(query.query_text)
        recalled_list = res.get("recalled_memories", [])
        records = [
            MemoryRecord(
                memory_id=f"mem_{i}",
                raw_text=m.get("text", "") if isinstance(m, dict) else str(m),
                tenant_id=query.tenant_id,
                attractor_energy=res.get("diagnostics", {}).get("final_energy"),
            )
            for i, m in enumerate(recalled_list)
        ]
        status_str = res.get("evidence_status", "INSUFFICIENT_EVIDENCE")
        return MemoryRecallResult(
            recalled_records=records,
            confidence_score=res.get("confidence_score", 0.0),
            is_grounded=res.get("is_grounded", False),
            evidence_status=status_str,
            attractor_displacement=res.get("attractor_displacement", 1.0),
            diagnostics=res.get("diagnostics", {}),
        )

    def delete(self, memory_id: str, tenant_id: str) -> bool:
        # Calls unlearning on pattern associated with memory_id
        return self._cortex.forget(memory_id, tenant_id=tenant_id)

    def list_memories(self, tenant_id: str, limit: int = 100) -> List[MemoryRecord]:
        return [
            MemoryRecord(
                memory_id=meta.get("memory_id", f"mem_{i}"),
                raw_text=meta.get("raw_text", ""),
                tenant_id=tenant_id,
            )
            for i, meta in enumerate(self._cortex.memory.memory_metadata[:limit])
        ]


class DefaultPolicyEngine(PolicyEngineContract):
    """Layer 4 security policy engine."""

    def __init__(self, require_approvals: bool = True):
        self._require_approvals = require_approvals
        self._approved_tokens: set = set()

    def evaluate(self, action: str, resource: str, context: PolicyContext) -> PolicyDecision:
        if action.startswith("filesystem.delete") or action.startswith("git.push"):
            token = f"approval_{int(time.time())}_{action}"
            return PolicyDecision(
                decision=PolicyDecisionType.REQUIRE_APPROVAL,
                reason="Consequential external action requires explicit human confirmation.",
                risk_level="critical",
                approval_token=token,
            )
        return PolicyDecision(decision=PolicyDecisionType.ALLOW, reason="Action within permitted policy boundary.", risk_level="low")

    def record_approval(self, approval_token: str, approved_by: str) -> bool:
        self._approved_tokens.add(approval_token)
        return True


class CortexRuntime:
    """The central unified runtime instance for CORTEX-M."""

    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig.load()
        self.events = EventBus()
        self.lm = CortexLM(
            dim=self.config.memory.state_dimension,
            beta=self.config.memory.beta,
        )
        self.cortex = self.lm.cortex
        self.memory_provider = SynapseMemoryProvider(self.cortex)
        self.policy_engine = DefaultPolicyEngine(require_approvals=self.config.require_human_approval_for_patches)

    def ask(self, question: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a grounded query through the unified runtime."""
        active_tenant = tenant_id or self.config.active_tenant_id
        self.events.publish(CortexEvent(
            event_type=EventType.MODEL_REQUESTED,
            payload={"query": question, "tenant_id": active_tenant},
            actor=active_tenant,
        ))

        # 1. Epistemic recall from memory plane
        recall_res = self.memory_provider.recall(MemoryQuery(
            query_text=question,
            tenant_id=active_tenant,
            threshold=self.config.memory.hallucination_threshold,
        ))

        # 2. LM reasoning
        lm_tenant = active_tenant if active_tenant != "default_user" else None
        lm_out = self.lm.chat(question, tenant_id=lm_tenant)

        self.events.publish(CortexEvent(
            event_type=EventType.MODEL_RESPONDED,
            payload={"is_grounded": lm_out.is_grounded, "evidence_status": str(lm_out.evidence_status)},
            actor="cortex_lm",
        ))

        return {
            "text": lm_out.text,
            "is_grounded": lm_out.is_grounded,
            "evidence_status": str(lm_out.evidence_status),
            "evidence_report": getattr(lm_out, "evidence_report", {}),
            "latency_ms": lm_out.latency_ms,
            "recalled_count": len(recall_res.recalled_records),
        }

    def learn(self, fact_text: str, tenant_id: Optional[str] = None) -> None:
        """Imprint knowledge into continuous associative memory."""
        active_tenant = tenant_id or self.config.active_tenant_id
        self.memory_provider.store(MemoryRecord(
            memory_id=f"user_{int(time.time()*1000)}",
            raw_text=fact_text,
            tenant_id=active_tenant,
        ))
        self.events.publish(CortexEvent(
            event_type=EventType.MEMORY_STORED,
            payload={"fact": fact_text, "tenant_id": active_tenant},
            actor=active_tenant,
        ))
