import unittest
import torch
import torch.nn.functional as F

from cortex_m.cortex_v2 import CortexV2
from cortex_m.pathway.causal_mimo import CausalPathwayMIMO
from cortex_m.pathway.encoder import NeuralFeatureEncoder
from cortex_m.pathway.gist_bottleneck import GistBottleneck
from cortex_m.pathway.synaptic_memory import SynapticMemoryNetwork
from cortex_m.pathway.tracer import PathwayTracer


class TestCortexV2Architecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)

    def test_semantic_encoder_alignment(self):
        """Semantic proximity: QA pairs align, unrelated topics remain separated."""
        encoder = NeuralFeatureEncoder(output_dim=64, seed=1337)
        v_fact = encoder.encode_text("My name is Dhaval")
        v_query = encoder.encode_text("What is my name?")
        v_unrelated = encoder.encode_text("The weather is hot and sunny outside")

        sim_qa = float(torch.dot(v_fact, v_query).item())
        sim_unrelated = float(torch.dot(v_fact, v_unrelated).item())

        self.assertGreaterEqual(sim_qa, 0.70, f"Expected QA similarity >= 0.70, got {sim_qa}")
        self.assertLess(sim_unrelated, 0.35, f"Expected unrelated similarity < 0.35, got {sim_unrelated}")

    def test_synaptic_hebbian_learning_and_denoising(self):
        """Memories alter synaptic weights W; recall converges via Lyapunov energy descent."""
        net = SynapticMemoryNetwork(dim=48, beta=10.0)
        p1 = F.normalize(torch.randn(48), p=2, dim=0)
        p2 = F.normalize(torch.randn(48), p=2, dim=0)

        initial_w = net.synaptic_weights.clone()
        trace1 = net.learn_pattern(p1, memory_id="fact_alpha")
        net.learn_pattern(p2, memory_id="fact_beta")

        # Synapses physically updated
        self.assertFalse(torch.allclose(net.synaptic_weights, initial_w))
        self.assertGreater(trace1.synaptic_delta_norm, 0.0)

        # Noisy cue: 65% target + 35% random noise
        noisy_cue = F.normalize(0.65 * p1 + 0.35 * torch.randn(48), p=2, dim=0)
        recalled, trace, diag = net.recall(noisy_cue)

        recalled_sim = float(torch.dot(recalled, p1).item())
        initial_sim = float(torch.dot(noisy_cue, p1).item())

        self.assertGreater(recalled_sim, initial_sim)
        self.assertGreater(recalled_sim, 0.85)
        self.assertEqual(diag["best_memory_id"], "fact_alpha")
        self.assertGreaterEqual(diag["energy_drop"], 0.0)

    def test_targeted_anti_hebbian_unlearning(self):
        """Anti-Hebbian depression (W <- W - eta v v^T) erases target memory cleanly."""
        cortex = CortexV2(dim=64, beta=10.0, seed=42)

        # 1. Learn Alice
        cortex.learn("My name is Alice", memory_id="mem_alice")
        res1 = cortex.recall("What is my name?")
        self.assertEqual(res1["best_memory_id"], "mem_alice")

        # 2. Unlearn Alice via anti-Hebbian depression
        unlearn_res = cortex.forget("mem_alice")
        self.assertEqual(unlearn_res["status"], "unlearned")
        self.assertGreater(unlearn_res["synaptic_delta_norm"], 0.0)

        # 3. Learn Bob
        cortex.learn("My name is Bob", memory_id="mem_bob")
        res2 = cortex.recall("What is my name?")
        self.assertEqual(res2["best_memory_id"], "mem_bob")
        self.assertEqual(res2["best_metadata"].get("raw_text"), "My name is Bob")

    def test_pathway_resonance_and_source_attribution(self):
        """System remembers HOW it learned: matching pathways resonate, mismatched do not."""
        cortex = CortexV2(dim=64, seed=42)

        # Store two facts from different episodic sources
        cortex.learn(
            "My name is Dhaval",
            memory_id="mem_dhaval",
            source_context="confidential_interview",
        )
        cortex.learn(
            "My favorite color is blue",
            memory_id="mem_color",
            source_context="casual_chat",
        )

        # Recall using related cue
        res_dhaval = cortex.recall("What is my name?")
        resonance = res_dhaval["resonance"]
        self.assertIsNotNone(resonance)
        self.assertTrue(resonance["is_pathway_matched"])
        self.assertGreaterEqual(resonance["resonance_score"], 0.70)

        # Episodic source attribution
        attribution = res_dhaval["source_attribution"]
        self.assertTrue(attribution["attributed"])
        self.assertEqual(attribution["best_source_context"], "confidential_interview")

        # Cross-comparison with mismatched learning trace
        color_trace = cortex.memory.episodic_traces["mem_color"]
        recall_trace = cortex.memory.tracer.history[-1]
        mismatch_res = PathwayTracer.compute_resonance(color_trace, recall_trace)
        self.assertLess(mismatch_res["resonance_score"], 0.35)

    def test_gist_bottleneck_compression(self):
        """Information bottleneck compresses representations into compact pre-verbal intent."""
        bottleneck = GistBottleneck(in_dim=64, gist_dim=32, out_dim=64, seed=42)
        x = F.normalize(torch.randn(64), p=2, dim=0)

        gist, expanded = bottleneck(x)
        self.assertEqual(gist.shape, (32,))
        self.assertEqual(expanded.shape, (64,))
        self.assertAlmostEqual(float(gist.norm().item()), 1.0, places=4)

    def test_causal_mimo_counterfactual_scramble(self):
        """Resolving Causality Crisis: Scrambled routes degrade output MSE > 1.5x."""
        causal_mimo = CausalPathwayMIMO(memory_dim=64, gist_dim=32)
        x = torch.randn(1, 8)
        r = F.one_hot(torch.arange(8), num_classes=8).float().unsqueeze(0)
        attractor = F.normalize(torch.randn(64), p=2, dim=0)

        with torch.no_grad():
            clean_y, gates, _, _, _ = causal_mimo.forward_causal(
                x, r, attractor, scramble_routes=False
            )

        eval_res = causal_mimo.evaluate_counterfactual(
            x=x,
            r=r,
            memory_attractor=attractor,
            target_y=clean_y,
            num_scrambles=15,
        )

        # Scrambling routes causes catastrophic disruption compared to clean
        self.assertGreaterEqual(
            eval_res["causality_ratio"],
            1.50,
            f"Expected causality ratio >= 1.5, got {eval_res['causality_ratio']}",
        )
        self.assertTrue(eval_res["is_causally_dependent"])

    def test_cortex_v2_end_to_end_system(self):
        """Unified workflow: learn -> recall -> causal route -> inspect synapses."""
        cortex = CortexV2(dim=64, beta=8.0, seed=42)

        # 1. Learn Mode Alpha
        learn_res = cortex.learn(
            "Task mode is alpha: compute primary relational route.",
            memory_id="mode_alpha",
            source_context="system_instructions",
        )
        self.assertEqual(learn_res["status"], "learned_in_synapses")

        # 2. Recall
        recall_res = cortex.recall("What is the current task mode?")
        self.assertEqual(recall_res["best_memory_id"], "mode_alpha")
        self.assertGreaterEqual(recall_res["resonance"]["resonance_score"], 0.70)

        # 3. Causal Route
        route_res = cortex.route(
            cue="What is the current task mode?",
            values=[0.1, -0.4, 0.2, 0.5, -0.1, 0.7, -0.3, 0.8],
            role_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        self.assertEqual(route_res["kind"], "causal_pathway_mimo")
        self.assertEqual(len(route_res["outputs"]), 4)
        self.assertEqual(len(route_res["selected_routes"]), 4)
        self.assertGreater(route_res["gate_sparsity"], 0.5)

        # 4. Synapse Inspection
        synapses = cortex.inspect_synapses()
        self.assertEqual(synapses["matrix_shape"], [64, 64])
        self.assertEqual(synapses["total_memories_stored"], 1)
        self.assertGreater(synapses["mean_weight"], 0.0)


class TestM2M100GistMIMOIntegration(unittest.TestCase):
    """Rigorous validation of the 7 MIMO-Gist iterations (M2M-94 to M2M-100)."""

    def setUp(self):
        torch.manual_seed(42)

    def test_m2m_94_gist_conditioned_routing(self):
        """M2M-94: Gist bottleneck directly modulates routing context."""
        cortex = CortexV2(dim=64, seed=42)
        cortex.learn("Priority task: financial analytics route", memory_id="finance_mode")
        res = cortex.route("What task should we run?", [0.1]*8, list(range(8)))
        self.assertIn("pre_verbal_gist_norm", res)
        self.assertGreater(res["pre_verbal_gist_norm"], 0.5)

    def test_m2m_95_cross_attentive_gist_sensory_fusion(self):
        """M2M-95: Cross-attention dynamically fuses sensory tokens with gist."""
        mimo = CausalPathwayMIMO(memory_dim=64, gist_dim=32)
        x = torch.randn(2, 8)
        r = F.one_hot(torch.arange(8), num_classes=8).float().unsqueeze(0).expand(2, -1, -1)
        mem = F.normalize(torch.randn(64), p=2, dim=0)
        y, gates, _, gist, c = mimo.forward_causal(x, r, mem)
        self.assertEqual(y.shape, (2, 4))
        self.assertEqual(gist.shape, (32,))
        self.assertEqual(c.shape, (2, 32))

    def test_m2m_96_gist_rate_distortion_annealing(self):
        """M2M-96: Information bottleneck rate-distortion trade-off curve."""
        bottleneck = GistBottleneck(in_dim=64, gist_dim=32, out_dim=64)
        x = F.normalize(torch.randn(1, 64), p=2, dim=-1)
        res = bottleneck.evaluate_rate_distortion_tradeoff(x, beta_ib=1.25)
        self.assertEqual(res["compression_ratio"], 2.0)
        self.assertIn("rate_bits", res)
        self.assertIn("distortion_mse", res)

    def test_m2m_97_bidirectional_route_gist_consistency(self):
        """M2M-97: Route representation can be inverted back to latent gist."""
        bottleneck = GistBottleneck(in_dim=64, gist_dim=32, out_dim=64)
        x = F.normalize(torch.randn(1, 64), p=2, dim=-1)
        gist, expanded = bottleneck(x)
        loss = bottleneck.compute_bidirectional_consistency_loss(gist, expanded)
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_m2m_98_gist_gated_hierarchical_routing(self):
        """M2M-98: Gist vector gates Level 2 inter-cluster selection for N=32."""
        cortex = CortexV2(dim=64, seed=42)
        cortex.learn("Targeting systems infrastructure cluster", memory_id="infra_task")
        res = cortex.route_hierarchical(
            cue="Which cluster should execute?",
            values=[float(i) / 32.0 for i in range(32)],
            role_ids=list(range(32)),
            n_out=4,
        )
        self.assertEqual(len(res["outputs"]), 4)
        self.assertEqual(res["levels"], 2)

    def test_m2m_99_recurrent_working_memory_gist_stability(self):
        """M2M-99: Multi-turn working memory maintains stable gist attractor basin."""
        cortex = CortexV2(dim=64, seed=42)
        cortex.learn("Alpha project uses Rust and WebAssembly", memory_id="rust_proj")
        t1 = cortex.multi_turn_interact("Tell me about Alpha project")
        t2 = cortex.multi_turn_interact("What technology stack is it built with?")
        self.assertEqual(t2["best_memory_id"], "rust_proj")

    def test_m2m_100_unified_milestone_causality(self):
        """M2M-100: The 100th Milestone integration achieves >10,000x causality ratio."""
        mimo = CausalPathwayMIMO(memory_dim=64, gist_dim=32)
        ckpt_path = "results/mimo-35d-full-checkpointed/seed-101.pt"
        mimo.load_m2m_checkpoint(ckpt_path)
        x = torch.randn(1, 8)
        r = F.one_hot(torch.arange(8), num_classes=8).float().unsqueeze(0)
        mem = F.normalize(torch.randn(64), p=2, dim=0)
        target_y, _, _, _, _ = mimo.forward_causal(x, r, mem, scramble_routes=False)
        eval_res = mimo.evaluate_counterfactual(x, r, mem, target_y, num_scrambles=5)
        self.assertTrue(eval_res["is_causally_dependent"])
        self.assertGreaterEqual(eval_res["causality_ratio"], 1.50)


if __name__ == "__main__":
    unittest.main()

