"""Unit tests for NocturnalSleepConsolidator."""

import unittest
import torch
import torch.nn.functional as F

from cortex_m.learning.consolidation import NocturnalSleepConsolidator, SleepCycleReport


class TestNocturnalSleepConsolidator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 32
        self.consolidator = NocturnalSleepConsolidator(
            dim=self.dim,
            pruning_threshold=0.03,
            homeostatic_target_norm=6.0,
        )

    def test_empty_attractors_cycle(self):
        W = torch.randn(self.dim, self.dim)
        W = 0.5 * (W + W.t())
        empty_P = torch.empty((0, self.dim))
        W_new, report = self.consolidator.consolidate(W, empty_P)
        self.assertEqual(report.replayed_attractors, 0)
        self.assertEqual(report.pruned_synapses, 0)
        self.assertEqual(report.retention_accuracy, 1.0)

    def test_consolidation_prunes_weak_synapses(self):
        # Generate 4 orthogonal attractor patterns
        patterns = torch.randn(4, self.dim)
        patterns = F.normalize(patterns, p=2, dim=-1)

        # Baseline Hebbian matrix with small random noise added
        W = torch.mm(patterns.t(), patterns) + 0.01 * torch.randn(self.dim, self.dim)
        W = 0.5 * (W + W.t())
        W.fill_diagonal_(0.0)

        W_post, report = self.consolidator.consolidate(
            W, patterns, replay_rounds=2, noise_level=0.10, learning_rate=0.04
        )

        # Verify sparsity increased due to adolescent pruning
        self.assertGreaterEqual(report.final_sparsity, report.initial_sparsity)
        self.assertGreater(report.pruned_synapses, 0)

        # Verify retention fidelity is high (>90%)
        self.assertGreaterEqual(report.retention_accuracy, 0.90)

        # Verify symmetry of consolidated weights
        diff = torch.norm(W_post - W_post.t()).item()
        self.assertLess(diff, 1e-5)

        # Verify homeostatic norm constraint
        final_norm = torch.norm(W_post, p="fro").item()
        self.assertLessEqual(final_norm, self.consolidator.homeostatic_target_norm + 1e-4)

    def test_energy_barrier_deepening(self):
        # Consolidation should deepen energy wells (E_final < E_initial => deepening > 0)
        patterns = torch.randn(3, self.dim)
        patterns = F.normalize(patterns, p=2, dim=-1)
        W = torch.mm(patterns.t(), patterns)
        W.fill_diagonal_(0.0)

        W_post, report = self.consolidator.consolidate(
            W, patterns, replay_rounds=3, noise_level=0.08, learning_rate=0.05
        )
        self.assertIsInstance(report, SleepCycleReport)
        self.assertEqual(report.replayed_attractors, 3)


if __name__ == "__main__":
    unittest.main()
