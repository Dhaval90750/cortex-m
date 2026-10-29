"""Adversarial stress and boundary tests for CORTEX-M Causal Sparse Router.

Tests:
1. NaN / Inf injection robustness (graceful dense fallback, zero NaNs in output).
2. Expert collapse detection (load balance loss penalty, balance ratio 0.0).
3. Simplex invariants of sparsemax under extreme logit regimes.
4. Subnormal and near-zero norm vectors triggering fallback without ZeroDivisionError.
5. Counterfactual ratio >= 1.5x causality verification gate.
6. Epsilon perturbation stability against chaotic route flipping.
7. Combinatorial scaling invariants for Hierarchical CausalRouter (N=16, 32).
"""

from __future__ import annotations

import math
import unittest
import numpy as np
import torch

from cortex_m.mimo.causal_router import (
    CausalRouter,
    CausalRouterBenchmark,
    RoutingProfiler,
    sparsemax,
)


class TestRoutingAdversarial(unittest.TestCase):
    """Adversarial tests for causal routing robustness and invariants."""

    def test_01_nan_and_inf_inputs_trigger_fallback(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        # Construct adversarial batch with NaN and Inf
        x_adv = torch.randn(4, 8)
        x_adv[0, 2] = float("nan")
        x_adv[1, 5] = float("inf")
        x_adv[2, 0] = float("-inf")
        x_adv[3, :] = float("nan")

        res = router.route(x_adv)
        self.assertTrue(res["fallback_triggered"])
        # Outputs must be finite (no NaNs or Infs)
        self.assertFalse(torch.isnan(res["output"]).any())
        self.assertFalse(torch.isinf(res["output"]).any())

    def test_02_expert_collapse_load_balancing_penalty(self):
        # Normal balanced distribution
        balanced_assignments = [[0, 1], [2, 3], [0, 1], [2, 3]]
        balanced_probs = [[0.25] * 4 for _ in range(4)]
        loss_balanced = CausalRouterBenchmark.compute_load_balance_loss(
            balanced_assignments, balanced_probs, num_experts=4
        )

        # Adversarially collapsed distribution (all tokens to expert 0)
        collapsed_assignments = [[0, 0], [0, 0], [0, 0], [0, 0]]
        collapsed_probs = [[1.0, 0.0, 0.0, 0.0] for _ in range(4)]
        loss_collapsed = CausalRouterBenchmark.compute_load_balance_loss(
            collapsed_assignments, collapsed_probs, num_experts=4
        )

        # Collapse should be penalized with significantly higher loss
        self.assertGreater(loss_collapsed, loss_balanced * 2.0)

        # Profiler detects complete balance collapse
        profiler = RoutingProfiler(num_experts=4)
        for _ in range(20):
            profiler.record_step(50.0, [0, 0], 0.0, False)
        rep = profiler.generate_report()
        self.assertEqual(rep["balance_ratio"], 0.0)

    def test_03_sparsemax_extreme_simplex_invariants(self):
        # 1. Very large positive logits
        z_large = torch.tensor([[1000.0, 999.0, 998.0, 500.0]])
        s_large = sparsemax(z_large, dim=-1)
        self.assertTrue((s_large >= 0.0).all())
        self.assertAlmostEqual(s_large.sum().item(), 1.0, places=4)

        # 2. Very large negative logits
        z_neg = torch.tensor([[-1000.0, -1005.0, -1010.0, -2000.0]])
        s_neg = sparsemax(z_neg, dim=-1)
        self.assertTrue((s_neg >= 0.0).all())
        self.assertAlmostEqual(s_neg.sum().item(), 1.0, places=4)

        # 3. All equal logits
        z_eq = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        s_eq = sparsemax(z_eq, dim=-1)
        self.assertTrue(torch.allclose(s_eq, torch.full((1, 4), 0.25)))

        # 4. Single dominant spike
        z_spike = torch.tensor([[1e6, -1e6, -1e6, -1e6]])
        s_spike = sparsemax(z_spike, dim=-1)
        self.assertAlmostEqual(s_spike[0, 0].item(), 1.0, places=4)
        self.assertEqual(s_spike[0, 1].item(), 0.0)

    def test_04_subnormal_and_near_zero_norm_vectors(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        # Norms 1e-8 and 1e-15 are well below the 1e-6 degenerate threshold
        x_subnormal = torch.tensor([[1e-8] * 8, [1e-15] * 8], dtype=torch.float32)
        res = router.route(x_subnormal)
        self.assertTrue(res["fallback_triggered"])
        self.assertFalse(torch.isnan(res["output"]).any())
        self.assertEqual(res["sparsity"], 0.0)

    def test_05_counterfactual_ratio_gate_exceeds_threshold(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        x = torch.randn(8, 8)
        clean_target = router.route(x, record_profile=False)["output"]

        cf_audit = router.evaluate_counterfactual(x, target_y=clean_target, num_scrambles=15)
        # Mandatory Acceptance Gate G6: Causality ratio >= 1.5x
        self.assertGreaterEqual(
            cf_audit["causality_ratio"],
            1.5,
            f"Causality ratio {cf_audit['causality_ratio']} failed Gate G6 (>= 1.5x)",
        )
        self.assertTrue(cf_audit["is_causally_dependent"])

    def test_06_perturbation_stability_against_chaotic_flipping(self):
        router = CausalRouter(in_features=8, num_experts=4, top_k=2)
        x = torch.randn(2, 8)
        res_clean = router.route(x, record_profile=False)

        # Tiny epsilon perturbation
        epsilon = 1e-5
        x_perturbed = x + torch.randn_like(x) * epsilon
        res_perturbed = router.route(x_perturbed, record_profile=False)

        # Expert selection should be identical or near-identical under micro-noise
        self.assertEqual(res_clean["selected_experts"], res_perturbed["selected_experts"])
        # Output delta should be bounded proportionally to epsilon
        diff = torch.norm(res_clean["output"] - res_perturbed["output"]).item()
        self.assertLess(diff, 0.1)

    def test_07_hierarchical_combinatorial_scaling(self):
        for n_in in [16, 32]:
            router = CausalRouter(in_features=n_in, num_experts=4, hierarchical=True, cluster_size=8)
            x = torch.randn(2, n_in)
            res = router.route(x)
            # Must satisfy output shape
            self.assertEqual(res["output"].shape, (2, 4))
            # Must satisfy sparsity >= 50%
            self.assertGreaterEqual(
                res["sparsity"],
                0.50,
                f"Hierarchical N={n_in} sparsity {res['sparsity']} must be >= 50%",
            )
            self.assertFalse(torch.isnan(res["output"]).any())


if __name__ == "__main__":
    unittest.main()
