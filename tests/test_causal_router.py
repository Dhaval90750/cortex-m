"""Unit test suite for CORTEX-M Causal Router, Benchmarking & Profiling (Feature F07 & F08).

Tests:
- CausalRouterBenchmark: Shannon entropy, load-balancing loss, sparse routing execution.
- RoutingProfiler: Latency tracking, expert utilization, balance ratio, fallbacks.
- StaticDenseRouter: 0% pathway sparsity, uniform gating.
- RandomRouter: K-expert random gating, expected sparsity.
- CausalRouter: High-level neural routing, fallback protection, counterfactual causality ratio.
"""

from __future__ import annotations

import math
import unittest
import numpy as np
import torch

from cortex_m.mimo.causal_router import (
    CausalRouter,
    CausalRouterBenchmark,
    RandomRouter,
    RoutingProfiler,
    StaticDenseRouter,
)


class TestCausalRouterBenchmark(unittest.TestCase):
    """Test CausalRouterBenchmark static methods."""

    def test_01_route_entropy_single_expert_zero(self):
        probs = [1.0]
        entropy = CausalRouterBenchmark.compute_route_entropy(probs)
        self.assertAlmostEqual(entropy, 0.0, places=5)

    def test_02_route_entropy_uniform_4_experts(self):
        probs = [0.25, 0.25, 0.25, 0.25]
        entropy = CausalRouterBenchmark.compute_route_entropy(probs)
        self.assertAlmostEqual(entropy, 2.0, places=4)

    def test_03_route_entropy_uniform_8_experts(self):
        probs = [0.125] * 8
        entropy = CausalRouterBenchmark.compute_route_entropy(probs)
        self.assertAlmostEqual(entropy, 3.0, places=4)

    def test_04_route_entropy_tensor_and_numpy(self):
        t_probs = torch.tensor([0.5, 0.5])
        np_probs = np.array([0.5, 0.5])
        h_t = CausalRouterBenchmark.compute_route_entropy(t_probs)
        h_np = CausalRouterBenchmark.compute_route_entropy(np_probs)
        self.assertAlmostEqual(h_t, 1.0, places=4)
        self.assertAlmostEqual(h_np, 1.0, places=4)

    def test_05_route_entropy_empty_and_zero_sum(self):
        self.assertEqual(CausalRouterBenchmark.compute_route_entropy([]), 0.0)
        self.assertEqual(CausalRouterBenchmark.compute_route_entropy([0.0, 0.0]), 0.0)

    def test_06_load_balance_loss_empty_batch(self):
        loss = CausalRouterBenchmark.compute_load_balance_loss([], [], num_experts=4)
        self.assertEqual(loss, 0.0)

    def test_07_load_balance_loss_balanced_vs_skewed(self):
        balanced_assignments = [[0, 1], [2, 3]]
        balanced_probs = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
        balanced_loss = CausalRouterBenchmark.compute_load_balance_loss(
            balanced_assignments, balanced_probs, num_experts=4
        )

        skewed_assignments = [[0, 0], [0, 0]]
        skewed_probs = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        skewed_loss = CausalRouterBenchmark.compute_load_balance_loss(
            skewed_assignments, skewed_probs, num_experts=4
        )

        self.assertGreater(skewed_loss, balanced_loss)

    def test_08_load_balance_loss_alpha_scaling(self):
        assignments = [[0, 1], [0, 1]]
        probs = [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
        loss1 = CausalRouterBenchmark.compute_load_balance_loss(assignments, probs, num_experts=4, alpha=1.0)
        loss2 = CausalRouterBenchmark.compute_load_balance_loss(assignments, probs, num_experts=4, alpha=2.5)
        self.assertAlmostEqual(loss2, loss1 * 2.5, places=4)

    def test_09_sparse_routing_normal_input(self):
        x = [1.2, -0.4, 0.8, 2.1]
        res = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=2)
        self.assertFalse(res["fallback_triggered"])
        self.assertEqual(len(res["selected_experts"]), 2)
        self.assertAlmostEqual(sum(res["probs"]), 1.0, places=4)
        self.assertGreater(res["route_entropy"], 0.0)

    def test_10_sparse_routing_degenerate_fallback(self):
        zero_x = [0.0, 0.0, 0.0, 0.0]
        res = CausalRouterBenchmark.run_sparse_routing(zero_x, num_experts=4, top_k=2)
        self.assertTrue(res["fallback_triggered"])
        self.assertEqual(len(res["selected_experts"]), 2)
        self.assertAlmostEqual(res["probs"][0], 0.25, places=4)

    def test_11_sparse_routing_clamping_k(self):
        x = [1.0, 2.0]
        res_large_k = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=10)
        self.assertEqual(len(res_large_k["selected_experts"]), 4)

        res_small_k = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=0)
        self.assertEqual(len(res_small_k["selected_experts"]), 1)


class TestRoutingProfiler(unittest.TestCase):
    """Test RoutingProfiler operational metrics."""

    def setUp(self):
        self.profiler = RoutingProfiler(num_experts=4)

    def test_01_zero_recorded_steps(self):
        report = self.profiler.generate_report()
        self.assertEqual(report["total_calls"], 0)
        self.assertEqual(report["mean_latency_us"], 0.0)
        self.assertEqual(report["p95_latency_us"], 0.0)
        self.assertEqual(report["balance_ratio"], 1.0)
        self.assertEqual(report["fallback_rate"], 0.0)

    def test_02_record_step_and_aggregates(self):
        self.profiler.record_step(latency_us=100.0, selected_experts=[0, 1], entropy=1.5, fallback=False)
        self.profiler.record_step(latency_us=200.0, selected_experts=[2, 3], entropy=1.7, fallback=False)
        report = self.profiler.generate_report()
        self.assertEqual(report["total_calls"], 2)
        self.assertAlmostEqual(report["mean_latency_us"], 150.0)
        self.assertAlmostEqual(report["mean_entropy"], 1.6)
        self.assertEqual(report["expert_utilization"], [1, 1, 1, 1])
        self.assertEqual(report["balance_ratio"], 1.0)
        self.assertEqual(report["fallback_rate"], 0.0)

    def test_03_fallback_rate_metric(self):
        self.profiler.record_step(50.0, [0], 1.0, fallback=True)
        self.profiler.record_step(50.0, [1], 1.0, fallback=False)
        self.profiler.record_step(50.0, [2], 1.0, fallback=False)
        self.profiler.record_step(50.0, [3], 1.0, fallback=True)
        report = self.profiler.generate_report()
        self.assertAlmostEqual(report["fallback_rate"], 0.5)

    def test_04_monopolized_expert_balance_zero(self):
        for _ in range(5):
            self.profiler.record_step(20.0, [0], 0.0, fallback=False)
        report = self.profiler.generate_report()
        self.assertEqual(report["balance_ratio"], 0.0)
        self.assertEqual(report["expert_utilization"][0], 5)
        self.assertEqual(report["expert_utilization"][1], 0)

    def test_05_p95_latency_distribution(self):
        for lat in range(1, 101):
            self.profiler.record_step(float(lat), [0, 1], 1.0, False)
        report = self.profiler.generate_report()
        self.assertGreaterEqual(report["p95_latency_us"], 94.0)

    def test_06_out_of_bounds_expert_ignored(self):
        self.profiler.record_step(10.0, [-1, 99], 1.0, False)
        report = self.profiler.generate_report()
        self.assertEqual(sum(report["expert_utilization"]), 0)

    def test_07_reset_clears_metrics(self):
        self.profiler.record_step(100.0, [0, 1], 1.5, True)
        self.profiler.reset()
        report = self.profiler.generate_report()
        self.assertEqual(report["total_calls"], 0)
        self.assertEqual(report["expert_utilization"], [0, 0, 0, 0])


class TestBaselineRouters(unittest.TestCase):
    """Test StaticDenseRouter and RandomRouter baseline behaviors."""

    def test_01_static_dense_router(self):
        router = StaticDenseRouter(in_features=8, num_experts=4, out_dim=4)
        x = torch.randn(3, 8)
        y, gates, sparsity = router(x)
        self.assertEqual(y.shape, (3, 4))
        self.assertEqual(gates.shape, (3, 4))
        self.assertEqual(sparsity, 0.0)
        # Verify uniform gating
        self.assertTrue(torch.allclose(gates, torch.full((3, 4), 0.25)))

    def test_02_random_router(self):
        router = RandomRouter(in_features=8, num_experts=4, top_k=2, out_dim=4)
        x = torch.randn(4, 8)
        y, gates, sparsity = router(x)
        self.assertEqual(y.shape, (4, 4))
        self.assertEqual(gates.shape, (4, 4))
        self.assertEqual(sparsity, 0.5)  # 1 - 2/4 = 0.5
        # Each row should have exactly 2 non-zero entries of value 0.5
        for b in range(4):
            non_zeros = (gates[b] > 0).sum().item()
            self.assertEqual(non_zeros, 2)
            self.assertAlmostEqual(gates[b].sum().item(), 1.0, places=4)


class TestCausalRouter(unittest.TestCase):
    """Test CausalRouter neural coordinator with MIMO and Hierarchical backends."""

    def test_01_mimo_normal_route(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        x = torch.randn(2, 8)
        res = router.route(x)
        self.assertEqual(res["output"].shape, (2, 4))
        self.assertEqual(res["gates"].shape, (2, 4))
        self.assertFalse(res["fallback_triggered"])
        self.assertEqual(len(res["selected_experts"]), 2)
        self.assertGreater(res["route_entropy"], 0.0)
        self.assertGreater(res["latency_us"], 0.0)

    def test_02_mimo_all_zero_fallback(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        x = torch.zeros(2, 8)
        res = router.route(x)
        self.assertTrue(res["fallback_triggered"])
        self.assertEqual(res["sparsity"], 0.0)
        self.assertAlmostEqual(res["route_entropy"], 2.0, places=4)

    def test_03_mimo_partial_fallback(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        x = torch.randn(2, 8)
        x[1] = 0.0  # second sample is degenerate
        res = router.route(x)
        self.assertTrue(res["fallback_triggered"])

    def test_04_mimo_counterfactual_evaluation(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        x = torch.randn(4, 8)
        # Use initial clean output as target
        clean_out = router.route(x, record_profile=False)["output"]
        res = router.evaluate_counterfactual(x, target_y=clean_out, num_scrambles=10)
        self.assertIn("clean_mse", res)
        self.assertIn("scrambled_mse_mean", res)
        self.assertIn("causality_ratio", res)
        self.assertIn("is_causally_dependent", res)
        self.assertGreaterEqual(res["causality_ratio"], 1.5)
        self.assertTrue(res["is_causally_dependent"])

    def test_05_hierarchical_route_and_counterfactual(self):
        router = CausalRouter(in_features=16, num_experts=4, hierarchical=True, cluster_size=8)
        x = torch.randn(2, 16)
        res = router.route(x)
        self.assertEqual(res["output"].shape, (2, 4))
        self.assertFalse(res["fallback_triggered"])
        self.assertGreater(res["sparsity"], 0.50)

        clean_y = res["output"]
        cf_res = router.evaluate_counterfactual(x, target_y=clean_y, num_scrambles=5)
        self.assertIn("causality_ratio", cf_res)
        self.assertIn("clean_mse", cf_res)


if __name__ == "__main__":
    unittest.main()
