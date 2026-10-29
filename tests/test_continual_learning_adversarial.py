"""Adversarial and stress test suite for CORTEX-M Continual Learning Subsystem.

Tests:
1. Zero-leakage invariant: Q^T P_perp == 0.0 and P_perp * prototypes.T == 0.0.
2. Collinear and duplicate prototypes handling (rank-deficient QR stability).
3. Subspace saturation / rank exhaustion: M >= d yields P_perp == 0_matrix.
4. High-dimensional sparse prototype projection (d=128).
5. Zero-division immunity in retention and BWT when scores drop to 0.0.
6. Resilient Fisher information matrix under adversarial zero-gradient inputs.
7. Projected update orthogonality under high-energy adversarial weight updates.
"""

from __future__ import annotations

import math
import unittest
import numpy as np
import torch
import torch.nn as nn

from cortex_m.learning.continual import (
    ContinualLearningSequence,
    EWCRegularizer,
)


class TestContinualLearningAdversarial(unittest.TestCase):
    """Adversarial stress tests for continual learning primitives."""

    def test_01_zero_leakage_invariant(self):
        rng = np.random.RandomState(42)
        dim = 64
        num_protos = 8
        protos = rng.randn(num_protos, dim).astype(np.float32)

        P_perp = ContinualLearningSequence.orthogonal_projector(protos)

        # Invariant 1: P_perp must be symmetric idempotent: P_perp^2 = P_perp
        P_perp_sq = np.dot(P_perp, P_perp)
        self.assertTrue(np.allclose(P_perp, P_perp_sq, atol=1e-5))

        # Invariant 2: P_perp * prototypes.T == 0 (zero leakage)
        leakage = np.dot(P_perp, protos.T)
        max_leakage = float(np.max(np.abs(leakage)))
        self.assertLessEqual(max_leakage, 1e-5)

    def test_02_collinear_and_duplicate_prototypes(self):
        dim = 32
        v = np.random.RandomState(101).randn(dim).astype(np.float32)
        v = v / np.linalg.norm(v)

        # Construct collinear prototypes: v, 2v, -3.5v, 0.1v
        collinear_protos = np.vstack([v, 2.0 * v, -3.5 * v, 0.1 * v])
        P_perp = ContinualLearningSequence.orthogonal_projector(collinear_protos)

        self.assertEqual(P_perp.shape, (dim, dim))
        self.assertFalse(np.isnan(P_perp).any())
        self.assertFalse(np.isinf(P_perp).any())

        # Residual on base vector must be 0
        residual = np.dot(P_perp, v)
        self.assertAlmostEqual(float(np.linalg.norm(residual)), 0.0, places=5)

    def test_03_rank_exhaustion_subspace_saturation(self):
        # When number of orthogonal prototypes equals dimension, null space is empty
        dim = 16
        # Generate full-rank orthonormal basis
        basis, _ = np.linalg.qr(np.random.RandomState(202).randn(dim, dim))
        P_perp = ContinualLearningSequence.orthogonal_projector(basis.T)

        # P_perp should be the zero matrix
        self.assertTrue(np.allclose(P_perp, np.zeros((dim, dim)), atol=1e-5))

        # Any gradient update projected through saturated P_perp is nullified
        grad = np.ones(dim, dtype=np.float32)
        projected = np.dot(P_perp, grad)
        self.assertAlmostEqual(float(np.linalg.norm(projected)), 0.0, places=5)

    def test_04_high_dimensional_sparse_prototypes(self):
        dim = 128
        # Sparse rank-1 prototype with single non-zero component
        sparse_proto = np.zeros((1, dim), dtype=np.float32)
        sparse_proto[0, 42] = 1.0

        P_perp = ContinualLearningSequence.orthogonal_projector(sparse_proto)
        self.assertEqual(P_perp.shape, (dim, dim))

        # P_perp component [42, 42] must be 0.0, other diagonals must be 1.0
        self.assertAlmostEqual(P_perp[42, 42], 0.0, places=5)
        self.assertAlmostEqual(P_perp[0, 0], 1.0, places=5)
        self.assertAlmostEqual(P_perp[100, 100], 1.0, places=5)

    def test_05_zero_division_immunity_scores_drop_to_zero(self):
        # Adversarial history where initial score is near zero or zero
        history_zero = {
            "Task_A": [0.0, 0.0],
            "Task_B": [0.9, 0.0],
        }
        res = ContinualLearningSequence.evaluate_retention(history_zero)
        self.assertFalse(math.isnan(res["mean_retention"]))
        self.assertFalse(math.isnan(res["mean_forgetting_index"]))
        self.assertEqual(res["per_task_retention"]["Task_B"], 0.0)
        self.assertEqual(res["per_task_forgetting"]["Task_B"], 0.9)

    def test_06_ewc_zero_gradient_resilience(self):
        model = nn.Linear(8, 2)
        ewc = EWCRegularizer(lambda_reg=200.0)

        # Batch where output has zero loss / constant
        zero_dataset = [
            (torch.zeros(8), torch.tensor(0, dtype=torch.long)),
            (torch.zeros(8), torch.tensor(0, dtype=torch.long)),
        ]
        ewc.consolidate_task("ZeroTask", model, zero_dataset)
        penalty = ewc.penalty(model)
        self.assertFalse(torch.isnan(penalty))
        self.assertEqual(penalty.item(), 0.0)

    def test_07_adversarial_high_energy_gradient_nullification(self):
        dim = 32
        rng = np.random.RandomState(303)
        prototypes = rng.randn(3, dim).astype(np.float32)
        P_perp = ContinualLearningSequence.orthogonal_projector(prototypes)

        # Huge adversarial perturbation update (magnitude 1000.0)
        adversarial_delta = rng.randn(dim).astype(np.float32) * 1000.0
        grafted_delta = np.dot(P_perp, adversarial_delta)

        # Residual interference on all prototypes must strictly remain zero
        interference = np.dot(prototypes, grafted_delta)
        max_interf = float(np.max(np.abs(interference)))
        self.assertLessEqual(max_interf, 1e-3)


if __name__ == "__main__":
    unittest.main()
