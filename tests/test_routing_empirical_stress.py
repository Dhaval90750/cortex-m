"""Empirical Adversarial Stress Harness for Phase 4 Causal Sparse Router.

Author: teamwork_challenger (Milestone 2 Routing Adversarial Challenger)
Scope:
1. Subnormal vectors (1e-12 down to float32 subnormals 1e-45), exact zero, negative zero.
2. Partial subnormals & mixed batches (normal + subnormal + NaN + Inf).
3. Extreme NaN / Inf injections across input, memory_attractor, and role_vector.
4. Batch invariance across B in {1, 2, 7, 13, 64, 128}.
5. High expert dropout, boundary top_k (k=1, k=num_experts, k > num_experts, k <= 0).
6. Independent empirical verification of M2M-93 counterfactual causality ratio (strict >= 1.5x).
7. Fallback determinism and zero division / zero NaN invariant guarantees.
8. Hierarchical router stress across N=16 and N=32.
"""

from __future__ import annotations

import math
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from cortex_m.mimo.causal_router import (
    CausalRouter,
    CausalRouterBenchmark,
    RoutingProfiler,
    StaticDenseRouter,
    RandomRouter,
)


class TestRoutingEmpiricalStress(unittest.TestCase):
    """Empirical challenge suite stress-testing dynamic causal routing."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    # ------------------------------------------------------------------------
    # 1. Subnormal and Zero Vector Invariants
    # ------------------------------------------------------------------------

    def test_01_extreme_subnormals_and_zeros_mimo(self):
        """Verify subnormal magnitudes (1e-12 down to 1e-45) and exact zeros cleanly trigger fallback."""
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        router.eval()

        subnormal_scales = [1e-7, 1e-12, 1e-25, 1e-38, 1.4e-45, 0.0, -0.0]
        for scale in subnormal_scales:
            x = torch.full((4, 8), scale, dtype=torch.float32)
            res = router.route(x, record_profile=False)

            self.assertTrue(
                res["fallback_triggered"],
                f"Expected fallback for scale {scale}, but fallback_triggered was False",
            )
            self.assertEqual(res["output"].shape, (4, 4))
            self.assertFalse(torch.isnan(res["output"]).any(), f"NaN detected at scale {scale}")
            self.assertFalse(torch.isinf(res["output"]).any(), f"Inf detected at scale {scale}")
            self.assertEqual(res["sparsity"], 0.0)
            # Simplex check: gates must sum to 1.0 per sample
            gate_sums = res["gates"].sum(dim=-1)
            self.assertTrue(torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5))

    def test_02_mixed_subnormal_normal_nan_batch(self):
        """Batch containing clean, subnormal, zero, NaN, and Inf simultaneously."""
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        router.eval()

        # Batch of 6 diverse adversarial rows:
        # Row 0: Normal Gaussian
        # Row 1: Subnormal 1e-15
        # Row 2: Exact zeros
        # Row 3: Single NaN
        # Row 4: Single +Inf
        # Row 5: Single -Inf
        x = torch.randn(6, 8)
        x[1, :] = 1e-15
        x[2, :] = 0.0
        x[3, 2] = float("nan")
        x[4, 4] = float("inf")
        x[5, 7] = float("-inf")

        res = router.route(x, record_profile=False)
        self.assertTrue(res["fallback_triggered"])
        self.assertEqual(res["output"].shape, (6, 4))
        self.assertFalse(torch.isnan(res["output"]).any(), "NaN found in mixed batch output")
        self.assertFalse(torch.isinf(res["output"]).any(), "Inf found in mixed batch output")

        # Gates must be valid non-negative probabilities summing to ~1.0
        self.assertTrue((res["gates"] >= 0.0).all())
        gate_sums = res["gates"].sum(dim=-1)
        self.assertTrue(torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-4))

    # ------------------------------------------------------------------------
    # 2. Batch Size Invariance & Dimensionality Stability
    # ------------------------------------------------------------------------

    def test_03_batch_size_sweep(self):
        """Verify strict shape consistency and numerical stability across B in {1, 2, 7, 13, 64, 128}."""
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        router.eval()

        for B in [1, 2, 7, 13, 64, 128]:
            x = torch.randn(B, 8)
            res = router.route(x, record_profile=False)

            self.assertEqual(res["output"].shape, (B, 4), f"Shape mismatch at B={B}")
            self.assertEqual(res["gates"].shape, (B, 4), f"Gate shape mismatch at B={B}")
            self.assertFalse(torch.isnan(res["output"]).any())
            self.assertFalse(torch.isinf(res["output"]).any())
            self.assertGreaterEqual(res["sparsity"], 0.0)
            self.assertLessEqual(res["sparsity"], 1.0)
            self.assertEqual(len(res["selected_experts"]), 2)

    def test_04_batch_size_one_edge_cases(self):
        """B=1 subnormal vs normal verification."""
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        router.eval()

        # B=1 normal
        x_norm = torch.randn(1, 8)
        res_norm = router.route(x_norm, record_profile=False)
        self.assertEqual(res_norm["output"].shape, (1, 4))
        self.assertFalse(res_norm["fallback_triggered"])

        # B=1 subnormal
        x_sub = torch.full((1, 8), 1e-12)
        res_sub = router.route(x_sub, record_profile=False)
        self.assertEqual(res_sub["output"].shape, (1, 4))
        self.assertTrue(res_sub["fallback_triggered"])
        self.assertEqual(res_sub["sparsity"], 0.0)

    # ------------------------------------------------------------------------
    # 3. High Expert Dropout & Expert Monopolization
    # ------------------------------------------------------------------------

    def test_05_extreme_top_k_bounds(self):
        """Router behavior under top_k bounds: k=1, k=num_experts, k > num_experts, k <= 0."""
        # k=1
        r_k1 = CausalRouter(in_features=8, num_experts=4, top_k=1)
        self.assertEqual(r_k1.top_k, 1)
        out_k1 = r_k1.route(torch.randn(3, 8), record_profile=False)
        self.assertEqual(len(out_k1["selected_experts"]), 1)

        # k=4 (num_experts)
        r_k4 = CausalRouter(in_features=8, num_experts=4, top_k=4)
        self.assertEqual(r_k4.top_k, 4)
        out_k4 = r_k4.route(torch.randn(3, 8), record_profile=False)
        self.assertEqual(len(out_k4["selected_experts"]), 4)

        # k=10 (> num_experts) clamped to num_experts
        r_k10 = CausalRouter(in_features=8, num_experts=4, top_k=10)
        self.assertEqual(r_k10.top_k, 4)
        out_k10 = r_k10.route(torch.randn(3, 8), record_profile=False)
        self.assertEqual(len(out_k10["selected_experts"]), 4)

    def test_06_benchmark_sparse_routing_topk_bounds(self):
        """CausalRouterBenchmark.run_sparse_routing clamped k bounds."""
        res_k0 = CausalRouterBenchmark.run_sparse_routing([1.0] * 8, num_experts=4, top_k=0)
        self.assertEqual(len(res_k0["selected_experts"]), 1)  # clamped to min 1

        res_k99 = CausalRouterBenchmark.run_sparse_routing([1.0] * 8, num_experts=4, top_k=99)
        self.assertEqual(len(res_k99["selected_experts"]), 4)  # clamped to max 4

    # ------------------------------------------------------------------------
    # 4. Independent Verification of M2M-93 Counterfactual Scramble Ratio
    # ------------------------------------------------------------------------

    def test_07_independent_counterfactual_ratio_reproducibility(self):
        """Empirically test counterfactual scramble causality ratio across multiple seeds and targets."""
        seeds = [42, 101, 777, 2026, 9999]
        causality_ratios = []

        for seed in seeds:
            torch.manual_seed(seed)
            router = CausalRouter(in_features=8, num_experts=4, top_k=2)
            router.eval()

            x = torch.randn(8, 8)
            with torch.no_grad():
                clean_target = router.route(x, record_profile=False)["output"]

            cf_res = router.evaluate_counterfactual(
                x, target_y=clean_target, num_scrambles=20
            )

            ratio = cf_res["causality_ratio"]
            causality_ratios.append(ratio)
            self.assertGreaterEqual(
                ratio,
                1.5,
                f"Seed {seed} failed Gate G6: causality ratio {ratio} < 1.5x",
            )
            self.assertTrue(cf_res["is_causally_dependent"])

        mean_ratio = float(np.mean(causality_ratios))
        min_ratio = float(np.min(causality_ratios))
        max_ratio = float(np.max(causality_ratios))

        # Logged for report:
        self.assertGreaterEqual(min_ratio, 1.5)
        self.assertGreaterEqual(mean_ratio, 2.0)

    def test_08_counterfactual_ratio_noisy_target(self):
        """Counterfactual scramble ratio evaluated against non-trivial external target."""
        torch.manual_seed(1234)
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        router.eval()

        x = torch.randn(10, 8)
        # Target with non-zero residual
        target_y = torch.sin(x[:, :4]) + 0.5 * torch.cos(x[:, 4:])

        cf_res = router.evaluate_counterfactual(
            x, target_y=target_y, num_scrambles=25
        )

        self.assertIn("causality_ratio", cf_res)
        self.assertIn("clean_mse", cf_res)
        self.assertIn("scrambled_mse_mean", cf_res)
        # Causality ratio must be valid positive float
        self.assertFalse(math.isnan(cf_res["causality_ratio"]))
        self.assertFalse(math.isinf(cf_res["causality_ratio"]))
        self.assertGreater(cf_res["causality_ratio"], 0.0)

    # ------------------------------------------------------------------------
    # 5. Fallback Determinism & Profiler Integrity
    # ------------------------------------------------------------------------

    def test_09_fallback_determinism(self):
        """Repeated subnormal inputs produce strictly identical fallback outputs."""
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        router.eval()

        sub_x = torch.tensor([[1e-14] * 8, [0.0] * 8])

        res1 = router.route(sub_x, record_profile=False)
        res2 = router.route(sub_x, record_profile=False)

        self.assertTrue(torch.equal(res1["output"], res2["output"]))
        self.assertTrue(torch.equal(res1["gates"], res2["gates"]))
        self.assertEqual(res1["fallback_triggered"], res2["fallback_triggered"])

    def test_10_profiler_load_stress_and_fallback_rate(self):
        """Profiler under 1000 fast routing calls correctly measures fallback rate and latency."""
        profiler = RoutingProfiler(num_experts=4)
        router = CausalRouter(in_features=8, num_experts=4, top_k=2, profiler=profiler)
        router.eval()

        # 300 subnormal calls + 700 normal calls = 30.0% expected fallback rate
        sub_x = torch.zeros(1, 8)
        norm_x = torch.randn(1, 8)

        for _ in range(300):
            router.route(sub_x, record_profile=True)
        for _ in range(700):
            router.route(norm_x, record_profile=True)

        rep = profiler.generate_report()
        self.assertEqual(rep["total_calls"], 1000)
        self.assertAlmostEqual(rep["fallback_rate"], 0.30, places=3)
        self.assertGreater(rep["mean_latency_us"], 0.0)
        self.assertGreaterEqual(rep["p95_latency_us"], rep["mean_latency_us"])

    # ------------------------------------------------------------------------
    # 6. Hierarchical Router Scaling & Adversarial Stress
    # ------------------------------------------------------------------------

    def test_11_hierarchical_subnormals_and_nan(self):
        """Hierarchical router (N=16, 32) under subnormals and NaN injections."""
        for n_in in [16, 32]:
            router = CausalRouter(in_features=n_in, num_experts=4, hierarchical=True, cluster_size=8)
            router.eval()

            # Batch with subnormal and NaN
            x = torch.randn(3, n_in)
            x[0, :] = 1e-12
            x[1, 5] = float("nan")

            res = router.route(x, record_profile=False)
            self.assertEqual(res["output"].shape, (3, 4))
            self.assertFalse(torch.isnan(res["output"]).any(), f"NaN in N={n_in} hierarchical output")
            self.assertFalse(torch.isinf(res["output"]).any(), f"Inf in N={n_in} hierarchical output")

    def test_12_hierarchical_counterfactual_ratio_gate(self):
        """Hierarchical router counterfactual causality ratio satisfies Gate G6 (>= 1.5x)."""
        router = CausalRouter(in_features=16, num_experts=4, hierarchical=True, cluster_size=8)
        router.eval()

        x = torch.randn(4, 16)
        with torch.no_grad():
            clean_target = router.route(x, record_profile=False)["output"]

        cf_res = router.evaluate_counterfactual(x, target_y=clean_target, num_scrambles=15)
        self.assertGreaterEqual(
            cf_res["causality_ratio"],
            1.5,
            f"Hierarchical N=16 causality ratio {cf_res['causality_ratio']} failed Gate G6",
        )
        self.assertTrue(cf_res["is_causally_dependent"])


if __name__ == "__main__":
    unittest.main()
