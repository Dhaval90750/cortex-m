"""CortexLM: Native Bio-Inspired Neural Language Model.

The standalone language model powered by Cortex-M:
- Zero external database / zero KV-cache memory wall.
- Real-time Hebbian continual learning during live inference.
- Anti-Hebbian instant targeted unlearning.
- Lyapunov energy-grounded hallucination immunity.
- Causal pathway resonance and verifiable provenance explainability.
- Multi-domain intelligence across systems, math, AI, and dialogue.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cortex_v2 import CortexV2
from ..analytics.cognitive_analyzer import CognitiveAnalyzer
from .energy_generator import EnergyGroundedGenerator
from .knowledge_base import FOUNDATIONAL_KNOWLEDGE_CORPUS
from .tokenizer import NeuralTokenizer


class CortexLMOutput:
    """Output container for CortexLM generation results with metacognition & suggestions."""

    def __init__(
        self,
        text: str,
        hallucination_risk: float,
        is_grounded: bool,
        provenance: Optional[Dict[str, Any]],
        factored_gist: Dict[str, Any],
        latency_ms: float,
        explainability: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        metacognition: Optional[Dict[str, Any]] = None,
        comprehension_level: str = "STANDARD_EXPERT",
    ):
        self.text = text
        self.hallucination_risk = hallucination_risk
        self.is_grounded = is_grounded
        self.provenance = provenance
        self.factored_gist = factored_gist
        self.latency_ms = latency_ms
        self.explainability = explainability
        self.suggestions = suggestions or []
        self.metacognition = metacognition or {}
        self.comprehension_level = comprehension_level

    def __repr__(self) -> str:
        status = "GROUNDED" if self.is_grounded else "ABSTAINED"
        return f"<CortexLMOutput [{status}] risk={self.hallucination_risk:.3f} level={self.comprehension_level} text='{self.text[:60]}...'>"


class CortexLM(nn.Module):
    """Native Cortex-M Language Model with Continuous Attractor Dynamics."""

    def __init__(
        self,
        dim: int = 64,
        gist_dim: int = 32,
        beta: float = 8.0,
        device: str = "cpu",
        seed: int = 42,
        imprint_foundation: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.gist_dim = gist_dim
        self.device = device
        self.seed = seed

        # Core Cortex-M substrate
        self.cortex = CortexV2(
            dim=dim,
            gist_dim=gist_dim,
            beta=beta,
            device=device,
            seed=seed,
        )

        # Cognitive diagnostics & safety
        self.analyzer = CognitiveAnalyzer(
            memory_network=self.cortex.memory,
            safety_critic=self.cortex.safety_critic,
        )

        # Neural Tokenizer & Energy-Grounded Generator
        self.tokenizer = NeuralTokenizer(dim=dim, seed=seed)
        self.generator = EnergyGroundedGenerator(
            tokenizer=self.tokenizer,
            dim=dim,
            gist_dim=128,  # 4-head factored gist
            hidden_dim=128,
            seed=seed,
        ).to(device)

        # Metacognitive Reasoning & Database Knowledge Ingestion
        from .metacognition import MetacognitiveEngine
        from ..database.db_trainer import DatabaseKnowledgeTrainer
        from .real_data_trainer import RealDataTrainer
        self.metacognition = MetacognitiveEngine(self)
        self.db_trainer = DatabaseKnowledgeTrainer(self)
        self.real_data_trainer = RealDataTrainer(self)

        # Imprint foundational multi-domain intelligence
        if imprint_foundation:
            self._imprint_foundation_knowledge()

    def _imprint_foundation_knowledge(self) -> None:
        """Imprint multi-domain knowledge corpus into physical synaptic weights."""
        for item in FOUNDATIONAL_KNOWLEDGE_CORPUS:
            self.cortex.learn(
                text_or_vector=item["text"],
                memory_id=item["memory_id"],
                source_context=item["source"],
                salience=1.0,
            )
        self.imprint_everyday_knowledge()
        self.cortex.working_memory.clear()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        tenant_id: Optional[str] = None,
        use_working_memory: bool = False,
    ) -> CortexLMOutput:
        """Generate energy-grounded text output with full provenance and explainability."""
        start_t = time.perf_counter()

        # Handle empty synaptic memory gracefully
        if self.cortex.memory.num_memories == 0:
            return CortexLMOutput(
                text="I have no memories stored in my synaptic substrate yet.",
                hallucination_risk=1.0,
                is_grounded=False,
                provenance=None,
                factored_gist={},
                latency_ms=round((time.perf_counter() - start_t) * 1000.0, 2),
                explainability="No synaptic memories stored in attractor network.",
                suggestions=["Teach me new facts using .learn()."],
                metacognition={"status": "empty_memory"},
                comprehension_level="STANDARD_EXPERT",
            )

        # 1. Dual-stream intent deconstruction (Ventral 128-D Gist + Dorsal Syntax)
        dual_out = self.cortex.process_dual_stream(prompt)
        factored_gist = dual_out.get("factored_heads", {})
        dorsal_meta = dual_out.get("dorsal_meta", {})
        unified_gist = dual_out["unified_gist"]

        # 2. Continuous Hopfield Recurrent Energy Recall
        if tenant_id:
            recall_res = self.cortex.recall_for_tenant(tenant_id, prompt)
            recalled_state = recall_res["recalled_state"]
            best_meta = recall_res.get("best_metadata", {})
            diag = {
                "best_metadata": best_meta,
                "best_memory_id": recall_res["best_memory_id"],
                "attractor_similarity": recall_res["attractor_similarity"],
                "steps_taken": recall_res.get("steps_taken", 2),
                "energy_drop": recall_res.get("energy_drop", 0.5),
                "resonance": recall_res.get("resonance", {"resonance_score": 0.70}),
                "source_attribution": recall_res.get("source_attribution", {"dominant_source": f"tenant_{tenant_id}"}),
            }
        else:
            recall_data = self.cortex.recall(prompt, use_working_memory=use_working_memory)
            recalled_state = recall_data["recalled_state"]
            best_meta = recall_data["best_metadata"]
            diag = {
                "best_metadata": best_meta,
                "best_memory_id": recall_data["best_memory_id"],
                "attractor_similarity": recall_data["attractor_similarity"],
                "steps_taken": recall_data["steps_taken"],
                "energy_drop": recall_data["energy_drop"],
                "resonance": recall_data["resonance"],
                "source_attribution": recall_data["source_attribution"],
            }

        cue_vec = self.cortex.encoder.encode_text(prompt)
        effective_cue = self.cortex.working_memory.prepare_contextual_cue(cue_vec) if use_working_memory else cue_vec

        # 3. Analyze Hallucination Risk & Epistemic Uncertainty via Lyapunov Basin
        risk_analysis = self.analyzer.analyze_hallucination_risk(
            cue_vector=effective_cue,
            recalled_state=recalled_state,
            convergence_steps=diag["steps_taken"],
            energy_drop=diag["energy_drop"],
        )

        # 4. Generate Output conditioned on Attractor + Gist + Dorsal Literals
        gen_result = self.generator.generate(
            prompt=prompt,
            recalled_attractor=recalled_state,
            factored_gist=unified_gist,
            recalled_metadata=best_meta,
            dorsal_meta=dorsal_meta,
            working_memory_state=self.cortex.working_memory.active_state,
            hallucination_assessment=risk_analysis,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # 5. Generate Provenance & Explainability
        explain_report = self.analyzer.generate_explainability_report(
            recall_diagnostics=diag,
            recalled_text=gen_result["generated_text"],
        )

        # 6. Evaluate 5W1H Metacognition & Proactive Suggestions
        meta_eval = self.metacognition.assess_query(
            prompt=prompt,
            attractor_id=diag.get("best_memory_id", "unindexed"),
            hallucination_risk=risk_analysis["hallucination_risk_score"],
            is_grounded=gen_result["grounded"],
        )

        latency = (time.perf_counter() - start_t) * 1000.0

        return CortexLMOutput(
            text=gen_result["generated_text"],
            hallucination_risk=risk_analysis["hallucination_risk_score"],
            is_grounded=gen_result["grounded"],
            provenance=gen_result["provenance"],
            factored_gist=factored_gist,
            latency_ms=round(latency, 2),
            explainability=explain_report["summary_explanation"],
            suggestions=meta_eval.suggested_followups,
            metacognition={
                "when_mode": meta_eval.when_mode,
                "why_rationale": meta_eval.why_rationale,
                "what_topic": meta_eval.what_topic,
                "where_attractor": meta_eval.where_attractor,
                "how_comprehension_level": meta_eval.how_comprehension_level,
                "active_learning_prompt": meta_eval.active_learning_prompt,
            },
            comprehension_level=meta_eval.how_comprehension_level,
        )

    def chat(self, message: str, tenant_id: Optional[str] = None) -> CortexLMOutput:
        """Conversational multi-turn chat with working memory, active pattern learning, and suggestions."""
        # 1. Active Learning: Learn user stack feedback if provided
        self.metacognition.process_user_feedback(message)

        # 2. Generate grounded response with working memory
        output = self.generate(message, tenant_id=tenant_id, use_working_memory=True)

        # 3. Record interaction into Tier 1 and Tier 2 working memory
        if output.is_grounded:
            enc = self.cortex.encoder.encode_text(output.text)
            self.cortex.working_memory.update(enc, label=f"response_{len(self.cortex.working_memory.turn_history)}")
        return output

    def connect_database_schema(
        self,
        schema_or_ddl: str,
        database_name: str = "app_db",
    ) -> Dict[str, Any]:
        """Ingest relational/document database schema directly into continuous synapses."""
        return self.db_trainer.ingest_schema(schema_or_ddl, database_name=database_name)

    def learn_real_data(
        self,
        filepath: str = "data/generated/dev.jsonl",
        max_samples: int = 50,
        source_label: str = "user_episodic_data",
    ) -> Dict[str, Any]:
        """Ingest real JSONL conversational/episodic records into physical synapses."""
        stats = self.real_data_trainer.ingest_jsonl_dataset(
            filepath=filepath,
            max_samples=max_samples,
            source_label=source_label,
        )
        self.cortex.working_memory.clear()
        return stats

    def clear_working_memory(self) -> None:
        """Clear active conversational working memory buffer."""
        self.cortex.working_memory.clear()

    def imprint_everyday_knowledge(self) -> int:
        """Imprint everyday life, health, finance, and productivity facts for normal users."""
        res = self.real_data_trainer.imprint_everyday_knowledge()
        self.cortex.working_memory.clear()
        return res

    def learn(
        self,
        text: str,
        memory_id: Optional[str] = None,
        source: Optional[str] = None,
        salience: float = 1.0,
    ) -> Dict[str, Any]:
        """Real-time Hebbian lifelong learning directly into physical synaptic matrix W."""
        return self.cortex.learn(
            text_or_vector=text,
            memory_id=memory_id,
            source_context=source,
            salience=salience,
        )

    def forget(self, target: str) -> Dict[str, Any]:
        """Anti-Hebbian targeted unlearning (synaptic depression)."""
        return self.cortex.forget(target)

    def sleep(
        self,
        replay_iterations: int = 3,
        noise_level: float = 0.15,
        decay_factor: float = 0.92,
    ) -> Dict[str, Any]:
        """Run biological sleep consolidation and homeostatic synaptic pruning."""
        return self.cortex.sleep_consolidation(
            replay_iterations=replay_iterations,
            noise_level=noise_level,
            decay_factor=decay_factor,
        )

    def audit(self) -> Dict[str, Any]:
        """Audit synaptic substrate health, capacity, and sparsity."""
        return self.analyzer.inspect_synaptic_health()
