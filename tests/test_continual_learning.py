"""Unit tests for CORTEX-M Continual Learning and Retention Benchmark (Features F09 & F10).

Tests:
1. EWCRegularizer: Fisher information computation, task consolidation, quadratic parameter penalty.
2. ContinualLearningSequence: Orthogonal null-space projector invariants, retention evaluation, BWT, FWT.
3. SequentialTaskSuite: Scientific task domains A -> B -> C -> D, prototypes, continuous features.
4. CM050RetentionBenchmark: Compare methods, evaluate model, single seed execution.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex_m.learning.continual import (
    CM050RetentionBenchmark,
    ContinualLearningSequence,
    EWCRegularizer,
    SequentialTaskSuite,
    SimpleClassifier,
)


class TestEWCRegularizer(unittest.TestCase):
    """Test EWC Regularizer Fisher computation and quadratic penalty."""

    def setUp(self):
        self.model = nn.Linear(8, 2)
        self.ewc = EWCRegularizer(lambda_reg=100.0)
        self.dataset = [
            (torch.randn(8), torch.tensor(0, dtype=torch.long)),
            (torch.randn(8), torch.tensor(1, dtype=torch.long)),
        ]

    def test_01_empty_penalty_zero(self):
        penalty = self.ewc.penalty(self.model)
        self.assertAlmostEqual(penalty.item(), 0.0, places=5)

    def test_02_compute_fisher_non_negative(self):
        fisher = self.ewc.compute_fisher(self.model, self.dataset)
        for name, tensor in fisher.items():
            self.assertTrue((tensor >= 0.0).all())
            param = dict(self.model.named_parameters())[name]
            self.assertEqual(tensor.shape, param.shape)

    def test_03_consolidate_task(self):
        self.ewc.consolidate_task("Task_A", self.model, self.dataset)
        self.assertIn("Task_A", self.ewc.consolidated_tasks)
        self.assertIn("Task_A", self.ewc.anchors)
        self.assertIn("Task_A", self.ewc.fishers)

    def test_04_penalty_increases_with_parameter_drift(self):
        self.ewc.consolidate_task("Task_A", self.model, self.dataset)
        initial_penalty = self.ewc.penalty(self.model).item()
        self.assertAlmostEqual(initial_penalty, 0.0, places=5)

        # Perturb model parameters away from anchor
        with torch.no_grad():
            self.model.weight.add_(torch.ones_like(self.model.weight) * 0.5)

        drifted_penalty = self.ewc.penalty(self.model).item()
        self.assertGreater(drifted_penalty, 0.0)


class TestContinualLearningSequence(unittest.TestCase):
    """Test ContinualLearningSequence null-space projector, retention, BWT, FWT."""

    def test_01_orthogonal_projector_exact_nullification(self):
        prototypes = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        P_perp = ContinualLearningSequence.orthogonal_projector(prototypes)
        residual = np.dot(P_perp, prototypes.T)
        self.assertAlmostEqual(float(np.linalg.norm(residual)), 0.0, places=5)

    def test_02_orthogonal_projector_preserves_orthogonal_space(self):
        prototypes = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        P_perp = ContinualLearningSequence.orthogonal_projector(prototypes)
        y = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        projected = np.dot(P_perp, y)
        self.assertAlmostEqual(float(np.linalg.norm(projected - y)), 0.0, places=5)

    def test_03_orthogonal_projector_empty_prototypes(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        P_perp = ContinualLearningSequence.orthogonal_projector(empty)
        self.assertTrue(np.allclose(P_perp, np.eye(4)))

    def test_04_evaluate_retention_all_retained(self):
        history = {
            "Task_A": [0.90, 0.90, 0.90],
            "Task_B": [0.85, 0.85],
        }
        res = ContinualLearningSequence.evaluate_retention(history)
        self.assertEqual(res["mean_retention"], 1.0)
        self.assertEqual(res["mean_forgetting_index"], 0.0)

    def test_05_evaluate_retention_partial_forgetting(self):
        history = {
            "Task_A": [0.90, 0.80, 0.72],
        }
        res = ContinualLearningSequence.evaluate_retention(history)
        self.assertAlmostEqual(res["per_task_retention"]["Task_A"], 0.80, places=2)
        self.assertAlmostEqual(res["per_task_forgetting"]["Task_A"], 0.18, places=2)

    def test_06_evaluate_bwt_calculation(self):
        # 3 tasks: initial scores [0.8, 0.85, 0.9], final scores [0.8, 0.85, 0.9] -> BWT = 0.0
        matrix_r = np.array([
            [0.80, 0.00, 0.00],
            [0.80, 0.85, 0.00],
            [0.80, 0.85, 0.90],
        ])
        bwt = ContinualLearningSequence.evaluate_bwt(matrix_r)
        self.assertAlmostEqual(bwt, 0.0, places=4)

        # Forgetting on Task 0: drops from 0.80 to 0.60 -> diff = -0.20
        matrix_r_forget = np.array([
            [0.80, 0.00],
            [0.60, 0.85],
        ])
        bwt_forget = ContinualLearningSequence.evaluate_bwt(matrix_r_forget)
        self.assertAlmostEqual(bwt_forget, -0.20, places=4)

    def test_07_evaluate_fwt_calculation(self):
        matrix_r = np.array([
            [0.80, 0.35],
            [0.80, 0.85],
        ])
        fwt = ContinualLearningSequence.evaluate_fwt(matrix_r, baseline_scores=[0.25, 0.25])
        # Task 1 zero-shot score after Task 0 is 0.35; baseline is 0.25 -> FWT = +0.10
        self.assertAlmostEqual(fwt, 0.10, places=4)


class TestSequentialTaskSuite(unittest.TestCase):
    """Test SequentialTaskSuite domain generation and properties."""

    def test_01_suite_task_count_and_dimensions(self):
        suite = SequentialTaskSuite(dim=32, samples_per_task=20, seed=42)
        self.assertEqual(len(suite.TASK_NAMES), 4)
        for task_name in suite.TASK_NAMES:
            self.assertEqual(len(suite.train_data[task_name]), 20)
            self.assertEqual(len(suite.test_data[task_name]), 30)
            self.assertEqual(suite.prototypes[task_name].shape, (32,))


class TestCM050RetentionBenchmark(unittest.TestCase):
    """Test CM050RetentionBenchmark methods."""

    def test_01_compare_methods_structure(self):
        tasks = ["Task_A", "Task_B"]
        results = {
            "naive_finetune": {"Task_A": [0.90, 0.30]},
            "ewc_plus_orthogonal_grafting": {"Task_A": [0.90, 0.88]},
        }
        report = CM050RetentionBenchmark.compare_methods(tasks, results)
        self.assertIn("comparisons", report)
        self.assertEqual(report["champion"], "ewc_plus_orthogonal_grafting")
        self.assertLess(report["comparisons"]["naive_finetune"]["mean_retention"], 0.50)
        self.assertGreaterEqual(report["comparisons"]["ewc_plus_orthogonal_grafting"]["mean_retention"], 0.90)

    def test_02_run_single_seed_execution(self):
        res = CM050RetentionBenchmark.run_single_seed(seed=42, dim=32)
        self.assertEqual(len(res), 5)
        self.assertIn("naive_finetune", res)
        self.assertIn("ewc_plus_orthogonal_grafting", res)
        # Verify champion retention >= 90%
        self.assertGreaterEqual(res["ewc_plus_orthogonal_grafting"]["mean_retention"], 0.90)
        # Verify catastrophic forgetting in naive fine-tune
        self.assertLess(res["naive_finetune"]["mean_retention"], 0.50)
        # Verify zero residual interference
        self.assertLessEqual(res["ewc_plus_orthogonal_grafting"]["residual_interference"], 1e-5)


if __name__ == "__main__":
    unittest.main()
