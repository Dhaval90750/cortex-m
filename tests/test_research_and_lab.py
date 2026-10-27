"""Unit tests for Cortex Research Engine, Lab Harness, and SOTA Synaptic Memory Upgrades."""

import unittest
import torch

from cortex_m.cortex_v2 import CortexV2
from cortex_m.research.research_engine import CortexResearchEngine, ResearchReport
from cortex_m.lab.lab_engine import CortexLabHarness, BenchmarkSuiteReport, AblationResult


class TestCortexResearchAndLab(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cortex = CortexV2(dim=64, beta=8.0)
        self.research_engine = CortexResearchEngine()
        self.lab_harness = CortexLabHarness(target_cortex=self.cortex)

    def test_synaptic_memory_quantization(self):
        # Learn into cortex
        self.cortex.learn("Theoretical physics of Hopfield networks", source_context="physics")
        self.cortex.learn("Autonomous satellite attitude determination", source_context="space")

        # High-level CortexV2 quantization
        q_res = self.cortex.quantize_weights(num_bits=8)
        self.assertIn("quantized_weight_bytes", q_res)
        self.assertIn("compression_ratio", q_res)
        self.assertGreater(q_res["compression_ratio"], 3.0)
        self.assertIn("frobenius_reconstruction_error", q_res)

        # Low-level synaptic memory direct quantization
        raw_q = self.cortex.memory.quantize_synapses(num_bits=8)
        self.assertEqual(raw_q["bitwidth"], 8)
        self.assertEqual(raw_q["compression_ratio"], 4.0)
        self.assertLess(raw_q["frobenius_reconstruction_error"], 0.2)

    def test_multi_head_recall(self):
        self.cortex.learn("Quantum mechanical annealing", source_context="quantum")
        cue = "Quantum annealing"

        # CortexV2 API
        res = self.cortex.multi_head_recall(cue, num_heads=4)
        self.assertEqual(res["num_heads"], 4)
        self.assertEqual(len(res["head_attributions"]), 4)
        self.assertIn("recalled_vector_norm", res)

        # Direct memory API
        v = torch.randn(64)
        mem_res = self.cortex.memory.multi_head_recall(v, num_heads=4)
        self.assertIn("recalled_vector", mem_res)
        self.assertEqual(mem_res["recalled_vector"].shape, (64,))

    def test_annealed_lyapunov_recall(self):
        self.cortex.learn("Lyapunov stability theorem for dynamical systems", source_context="math")
        cue = "Lyapunov stability theorem"

        res = self.cortex.annealed_lyapunov_recall(cue, beta_start=2.0, beta_end=16.0, steps=5)
        self.assertIn("final_energy", res)
        self.assertIn("lyapunov_certified", res)
        self.assertIn("trajectory", res)
        self.assertEqual(len(res["trajectory"]), 5)

        # Direct memory API
        v = torch.randn(64)
        mem_res = self.cortex.memory.annealed_lyapunov_recall(v, beta_start=2.0, beta_end=16.0, steps=5)
        self.assertIn("recalled_vector", mem_res)
        self.assertEqual(mem_res["recalled_vector"].shape, (64,))

    def test_exponential_capacity(self):
        cap_info = self.cortex.exponential_capacity()
        self.assertIn("theoretical_exponential_capacity", cap_info)
        self.assertGreater(cap_info["theoretical_exponential_capacity"], 1e6)
        self.assertEqual(cap_info["dimension"], 64)

    def test_research_engine_synthesis(self):
        topic = "Modern Hopfield Networks Energy Convergence"
        report = self.research_engine.conduct_deep_research(topic)
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.topic, topic)
        self.assertGreaterEqual(len(report.sub_questions), 2)
        self.assertGreater(len(report.synthesis_markdown), 50)
        self.assertGreaterEqual(len(report.citations), 1)
        self.assertIn(report.epistemic_status, ["VERIFIED", "PARTIAL", "CONFLICTING"])
        self.assertGreater(len(report.provenance_hash), 8)

    def test_lab_harness_benchmarks(self):
        bench = self.lab_harness.run_associative_benchmark(
            num_patterns=10,
            dimension=64,
            noise_level=0.10,
            iterations=2,
        )
        self.assertEqual(bench["num_patterns"], 10)
        self.assertIn("mean_recovery_cosine_sim", bench)
        self.assertGreater(bench["mean_recovery_cosine_sim"], 0.8)
        self.assertGreater(bench["p50_latency_us"], 0.0)
        self.assertGreaterEqual(bench["p95_latency_us"], bench["p50_latency_us"])

    def test_lab_harness_ablation_and_full_suite(self):
        ablations = self.lab_harness.run_ablation_matrix()
        self.assertEqual(len(ablations), 4)
        for a in ablations:
            self.assertIsInstance(a, AblationResult)
            self.assertLess(a.ablated_metric, a.baseline_metric)

        report = self.lab_harness.execute_full_suite()
        self.assertIsInstance(report, BenchmarkSuiteReport)
        self.assertTrue(report.passed_all_gates)
        self.assertGreater(report.mean_accuracy_pct, 90.0)
        self.assertGreater(report.memory_footprint_kb, 0.0)


if __name__ == "__main__":
    unittest.main()
