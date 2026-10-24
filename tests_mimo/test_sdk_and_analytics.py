"""Unit test suite for Cortex-M Developer SDK & Cognitive Analytics."""

from __future__ import annotations

import unittest
import torch

from cortex_m import (
    CortexEngine,
    CortexLLMBridge,
    CognitiveAnalyzer,
    RecallResult,
)


class TestCortexMSDKAndAnalytics(unittest.TestCase):
    """Test suite verifying developer SDK, LLM bridge, and analysis capabilities."""

    def setUp(self):
        torch.manual_seed(42)
        self.engine = CortexEngine(dim=64, gist_dim=32, seed=42)

    def test_engine_remember_and_recall_fluent(self):
        """Verify engine fluent API for learning and grounded recall with explainability."""
        self.engine.remember(
            "Cortex-M uses continuous Hopfield networks for associative retrieval.",
            memory_id="mem_hopfield",
            source="system_architecture_doc",
        )

        result = self.engine.recall("What type of network does Cortex-M use?")
        self.assertIsInstance(result, RecallResult)
        self.assertEqual(result.memory_id, "mem_hopfield")
        self.assertGreater(result.attractor_similarity, 0.70)
        self.assertGreater(result.resonance_score, 0.50)

        # Check explainability summary
        explanation = result.explain()
        self.assertIn("mem_hopfield", explanation)
        self.assertIn("system_architecture_doc", explanation)

    def test_engine_multi_tenant_session_context(self):
        """Verify engine tenant session isolation with ephemeral low-rank paging."""
        with self.engine.tenant("tenant_finance") as finance:
            finance.remember("Q4 revenue projection is $12M.")
            rec = finance.recall("What is the Q4 projection?")
            self.assertIsNotNone(rec.text)

        with self.engine.tenant("tenant_engineering") as eng:
            # Engineering must have clean state, no leakage from finance
            rec_eng = eng.recall("What is the Q4 projection?")
            # Engineering tenant cannot see finance tenant's private delta
            self.assertNotEqual(rec_eng.text, "Q4 revenue projection is $12M.")

    def test_engine_synaptic_health_audit(self):
        """Verify synaptic health audit and capacity estimation."""
        self.engine.remember("Test pattern alpha", memory_id="p1")
        self.engine.remember("Test pattern beta", memory_id="p2")

        health = self.engine.audit()
        self.assertEqual(health["matrix_dimensions"], [64, 64])
        self.assertGreaterEqual(health["synaptic_sparsity_pct"], 0.0)
        self.assertIn("substrate_health", health)
        self.assertIn("estimated_saturation_pct", health)

    def test_llm_bridge_prompt_augmentation(self):
        """Verify LLM bridge analyzes query and injects grounded context with low hallucination risk."""
        self.engine.remember(
            "The cluster uses port 8080 with TLS 1.3 encryption.",
            memory_id="port_spec",
            source="security_policy",
        )

        bridge = self.engine.get_llm_bridge()
        augmented = bridge.augment_prompt(
            user_query="Which port and encryption protocol should I connect to?",
            system_prompt="You are a DevOps assistant.",
        )

        self.assertIn("augmented_system_prompt", augmented)
        self.assertTrue(augmented["context_injected"])
        self.assertIn("8080", augmented["augmented_system_prompt"])
        self.assertIn("TLS 1.3", augmented["augmented_system_prompt"])
        self.assertLess(augmented["hallucination_assessment"]["hallucination_risk_score"], 0.40)
        self.assertTrue(augmented["hallucination_assessment"]["is_reliable"])
        self.assertLess(augmented["latency_ms"], 100.0)

    def test_llm_bridge_query_tier_routing(self):
        """Verify pre-verbal gist routes code/math vs factual queries to correct model tiers."""
        bridge = self.engine.get_llm_bridge()

        # Query with code
        code_res = bridge.route_query_to_model("def train_step(batch): loss = criterion(y, t); return loss")
        self.assertEqual(code_res["selected_tier"], "code_math")

        # Query with terse intent
        short_res = bridge.route_query_to_model("Define CPU")
        self.assertEqual(short_res["selected_tier"], "fast_compact")

        # Complex multi-step reasoning
        deep_res = bridge.route_query_to_model(
            "Compare the theoretical energy efficiency trade-offs between continuous Hopfield networks and standard self-attention mechanisms in transformer architectures."
        )
        self.assertEqual(deep_res["selected_tier"], "deep_reasoning")

    def test_llm_bridge_wrap_llm_generator(self):
        """Verify wrap_llm attaches turnkey ground truth, hallucination scoring, and WM tracking."""
        self.engine.remember(
            "The API key must be passed via the X-Cortex-Token header.",
            memory_id="api_auth",
            source="api_docs",
        )
        bridge = self.engine.get_llm_bridge()

        def mock_llm(system_prompt: str, user_prompt: str) -> str:
            # Emulate an LLM consuming the system prompt
            if "X-Cortex-Token" in system_prompt:
                return "Use the X-Cortex-Token header to provide your authentication key."
            return "Pass your API key in the authorization header."

        wrapped = bridge.wrap_llm(mock_llm)
        output = wrapped("How do I authenticate with the API?")

        self.assertIn("X-Cortex-Token", output["response"])
        self.assertIn("cortex_diagnostics", output)
        self.assertTrue(output["cortex_diagnostics"]["hallucination_assessment"]["is_reliable"])
        # Working memory should have recorded the response
        self.assertGreater(self.engine.cortex.working_memory.state_summary()["total_turns_retained"], 0)


if __name__ == "__main__":
    unittest.main()
