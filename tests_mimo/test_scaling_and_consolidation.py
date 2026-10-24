import unittest
import torch

from cortex_m.cortex_v2 import CortexV2
from cortex_m.pathway.consolidation import SynapticConsolidator
from cortex_m.pathway.hierarchical_router import HierarchicalSparseRouter
from cortex_m.pathway.working_memory import WorkingMemoryBuffer


class TestScalingAndConsolidation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_hierarchical_routing_n16_and_n32(self):
        """Break the N=32 collapse: Hierarchical router routes N=16 and N=32 successfully."""
        router = HierarchicalSparseRouter(cluster_size=8, seed=42)

        # Test N=16
        x16 = torch.randn(1, 16)
        r16 = torch.arange(16).unsqueeze(0)
        att16 = torch.randn(64)
        res16 = router.route_hierarchical(x16, r16, memory_attractor=att16, n_out=4)

        self.assertTrue(res16["hierarchical"])
        self.assertEqual(res16["num_clusters"], 2)
        self.assertEqual(len(res16["outputs"]), 4)
        self.assertEqual(len(res16["selected_inputs"]), 4)

        # Test N=32 (The old M2M collapse point)
        x32 = torch.randn(1, 32)
        r32 = torch.arange(32).unsqueeze(0)
        att32 = torch.randn(64)
        res32 = router.route_hierarchical(x32, r32, memory_attractor=att32, n_out=4)

        self.assertTrue(res32["hierarchical"])
        self.assertEqual(res32["num_clusters"], 4)
        self.assertEqual(len(res32["outputs"]), 4)
        self.assertEqual(len(res32["selected_inputs"]), 4)
        for idx in res32["selected_inputs"]:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 32)

    def test_synaptic_sleep_consolidation(self):
        """Sleep consolidation prunes noise, stabilizes norms, and maintains recall."""
        cortex = CortexV2(dim=64, seed=42)

        memories = [
            ("The compiler is written in Rust", "mem_rust"),
            ("The memory network uses Hopfield dynamics", "mem_hopfield"),
            ("The database is completely eliminated", "mem_nodb"),
            ("The server runs on port 9000", "mem_port"),
        ]

        for text, mid in memories:
            cortex.learn(text, memory_id=mid)

        initial_sparsity = cortex.inspect_synapses()["sparsity"]

        # Run biological sleep consolidation
        sleep_report = cortex.sleep_consolidation(
            replay_iterations=2,
            noise_level=0.10,
            decay_factor=0.90,
        )

        self.assertEqual(sleep_report["status"], "consolidated")
        self.assertEqual(sleep_report["replayed_memories"], 4)
        self.assertGreater(sleep_report["pruned_synapses"], 0)

        # After sleep, recall of core memory still succeeds cleanly
        recall_res = cortex.recall("Which language is the compiler written in?")
        self.assertEqual(recall_res["best_memory_id"], "mem_rust")
        self.assertGreaterEqual(recall_res["attractor_similarity"], 0.80)

    def test_capacity_stress_scaling(self):
        """Stress-test Hopfield network capacity across N=10, 50, 100 patterns."""
        results = SynapticConsolidator.evaluate_capacity(
            dim=64,
            num_patterns_list=(10, 50, 100),
            noise_level=0.20,
            seed=42,
        )

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertGreaterEqual(
                r["recall_accuracy"],
                0.85,
                f"Failed capacity target at N={r['num_stored_patterns']}: {r['recall_accuracy']}",
            )

    def test_working_memory_multi_turn_threading(self):
        """Working memory maintains conversational context across turns."""
        cortex = CortexV2(dim=64, seed=42)

        # Turn 1: Learn about a project
        cortex.learn(
            "I am building a high performance compiler in Rust",
            memory_id="proj_compiler",
            source_context="standup_meeting",
        )

        # Turn 2: Query using pronoun 'it' with working memory active
        t2_res = cortex.multi_turn_interact("What language is it written in?")
        self.assertEqual(t2_res["best_memory_id"], "proj_compiler")
        self.assertIn("Rust", t2_res["verbal_response"])
        self.assertTrue(cortex.working_memory.state_summary()["has_active_context"])

    def test_end_to_end_hierarchical_routing(self):
        """End-to-end CortexV2 hierarchical routing call for N=32 inputs."""
        cortex = CortexV2(dim=64, seed=42)
        cortex.learn("Task mode is compute clustered routes", memory_id="cluster_mode")

        values32 = [float(i * 0.1) for i in range(32)]
        roles32 = list(range(32))

        route_res = cortex.route_hierarchical(
            cue="What is the current task mode?",
            values=values32,
            role_ids=roles32,
            n_out=4,
        )

        self.assertEqual(len(route_res["outputs"]), 4)
        self.assertEqual(route_res["num_clusters"], 4)
        self.assertEqual(route_res["recalled_memory_id"], "cluster_mode")


if __name__ == "__main__":
    unittest.main()
