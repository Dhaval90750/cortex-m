import unittest
import torch
import torch.nn.functional as F

from cortex_m.pathway.encoder import NeuralFeatureEncoder
from cortex_m.pathway.synaptic_memory import SynapticMemoryNetwork
from cortex_m.pathway.tracer import PathwayTracer
from cortex_m.pathway.mimo_pathway import MIMOPathwaySystem, SynapticConditionedMIMO


class TestPathwayArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)

    def test_encoder_deterministic_and_normalized(self):
        encoder = NeuralFeatureEncoder(output_dim=64)
        v1 = encoder.encode_text("my name is Dhaval")
        v2 = encoder.encode_text("my name is Dhaval")
        v3 = encoder.encode_text("my favorite color is blue")

        self.assertEqual(v1.shape, (64,))
        # Deterministic
        self.assertTrue(torch.allclose(v1, v2, atol=1e-6))
        # Normalized to unit hypersphere
        self.assertAlmostEqual(float(v1.norm().item()), 1.0, places=4)
        # Distinct phrases have different representations
        similarity = float(torch.dot(v1, v3).item())
        self.assertLess(similarity, 0.9)

    def test_synaptic_learning_without_hard_disk(self):
        """Memories modify connection weights W; no database or disk table used."""
        net = SynapticMemoryNetwork(dim=32)
        initial_synapses = net.synaptic_weights.clone()

        # Learn pattern 1
        p1 = torch.randn(32)
        trace1 = net.learn_pattern(p1, memory_id="mem_alpha")

        self.assertEqual(net.num_memories, 1)
        # Synaptic weights changed
        self.assertFalse(torch.allclose(net.synaptic_weights, initial_synapses))
        # Learned synaptic delta recorded
        self.assertGreater(trace1.synaptic_delta_norm, 0.0)
        self.assertEqual(trace1.phase, "learn")

    def test_associative_recall_and_energy_descent(self):
        """Noisy/partial cue converges to the stored attractor while Lyapunov energy drops."""
        net = SynapticMemoryNetwork(dim=48, beta=10.0)

        # Store two distinct patterns
        p_target = F.normalize(torch.randn(48), p=2, dim=0)
        p_other = F.normalize(torch.randn(48), p=2, dim=0)

        net.learn_pattern(p_target, memory_id="target_attractor")
        net.learn_pattern(p_other, memory_id="other_attractor")

        # Create a noisy cue: 70% target pattern + 30% random noise
        noise = torch.randn(48)
        noisy_cue = F.normalize(0.7 * p_target + 0.3 * noise, p=2, dim=0)

        initial_sim = float(torch.dot(noisy_cue, p_target).item())

        # Recurrent recall through synaptic attractor dynamics
        recalled_state, trace, diag = net.recall(noisy_cue)

        recalled_sim = float(torch.dot(recalled_state, p_target).item())

        # Recall cleaned up noise and moved closer to the attractor
        self.assertGreater(recalled_sim, initial_sim)
        self.assertEqual(diag["best_memory_id"], "target_attractor")

        # Verify energy strictly decreased or stayed at minimum
        self.assertGreaterEqual(trace.initial_energy, trace.final_energy - 1e-5)
        self.assertTrue(len(trace.steps) >= 1)

    def test_pathway_tracing_and_resonance(self):
        """System remembers HOW it learned: recall trajectory resonates with training path."""
        net = SynapticMemoryNetwork(dim=32, beta=8.0)
        pattern = F.normalize(torch.randn(32), p=2, dim=0)
        learn_trace = net.learn_pattern(pattern, memory_id="episodic_fact")

        # Recall using cue close to pattern
        cue = F.normalize(pattern + 0.1 * torch.randn(32), p=2, dim=0)
        _, recall_trace, diag = net.recall(cue)

        resonance = diag["resonance"]
        self.assertIsNotNone(resonance)
        self.assertTrue(resonance["is_pathway_matched"])
        self.assertGreaterEqual(resonance["resonance_score"], 0.70)
        self.assertGreater(resonance["state_similarity"], 0.90)

    def test_mimo_pathway_system_end_to_end(self):
        """Unified test: text learning -> synaptic update -> pathway trace -> MIMO sparse route."""
        system = MIMOPathwaySystem(dim=64)

        # 1. Learn Mode Alpha
        learn_res = system.learn("Task mode is alpha: compute primary relational route.", memory_id="mode_alpha")
        self.assertEqual(learn_res["status"], "learned_in_synapses")
        self.assertEqual(learn_res["memory_id"], "mode_alpha")

        # 2. Execute MIMO task conditioned on query
        values = [0.2, -0.5, 0.3, 0.8, -0.2, 0.4, -0.1, 0.6]
        roles = [0, 1, 2, 3, 4, 5, 6, 7]

        route_res = system.route_task(
            task_cue="what is the task mode?",
            values=values,
            role_ids=roles,
        )

        self.assertEqual(len(route_res["mimo_outputs"]), 1)
        self.assertEqual(len(route_res["mimo_outputs"][0]), 4)
        self.assertEqual(len(route_res["selected_physical_positions"][0]), 4)
        self.assertEqual(route_res["memory_traversal"]["best_memory_id"], "mode_alpha")
        self.assertIn("pathway_trace", route_res["memory_traversal"])

        # 3. Learn Mode Beta (a different task mode)
        system.learn("Task mode is beta: switch routing to secondary reciprocal branch.", memory_id="mode_beta")

        # Execute MIMO task with beta cue
        route_beta = system.route_task(
            task_cue="switch routing to secondary reciprocal branch",
            values=values,
            role_ids=roles,
        )

        self.assertEqual(route_beta["memory_traversal"]["best_memory_id"], "mode_beta")
        # Route outputs and physical paths change based on the recalled attractor
        self.assertNotEqual(
            route_res["mimo_outputs"],
            route_beta["mimo_outputs"],
        )


if __name__ == "__main__":
    unittest.main()
