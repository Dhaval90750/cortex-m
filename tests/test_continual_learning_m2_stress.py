"""Empirical Challenger Adversarial Stress Suite for Continual Learning (Phase 5).

Milestone 2 Continual Learning Stress Challenger
Target: src/cortex_m/learning/continual.py

Evaluates:
1. Null-Space Projector (P_perp) under high task counts (T=8, T=16).
2. Rank exhaustion stress: M -> d, M = d, M > d, M >> d.
3. Collinear, duplicate, scaled, zero, and mixed prototype vectors.
4. High-energy adversarial gradient injections (1,000x - 10,000x scale) with zero leakage.
5. Champion ewc_plus_orthogonal_grafting retention >= 90.0% under adversarial cross-talk.
"""

from __future__ import annotations

import math
import random
import unittest
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex_m.learning.continual import (
    ContinualLearningSequence,
    EWCRegularizer,
    SimpleClassifier,
)


class TestContinualLearningM2Stress(unittest.TestCase):
    """Deep adversarial stress tests for CORTEX-M Continual Learning."""

    def setUp(self):
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    # ------------------------------------------------------------------------
    # 1. High Task Counts (T=8, T=16) Stress
    # ------------------------------------------------------------------------
    def test_01_high_task_counts_t8_and_t16(self):
        """Stress P_perp under high task counts T=8 and T=16."""
        dim = 64
        rng = np.random.RandomState(1001)

        for T in (8, 16):
            # Generate T independent prototypes
            raw_protos = rng.randn(T, dim).astype(np.float32)
            # Normalize to unit sphere
            protos = raw_protos / np.linalg.norm(raw_protos, axis=1, keepdims=True)

            P_perp = ContinualLearningSequence.orthogonal_projector(protos)

            # Invariant 1: Dimensions
            self.assertEqual(P_perp.shape, (dim, dim))

            # Invariant 2: Symmetry ||P_perp - P_perp^T||_inf < 1e-6
            symm_err = float(np.max(np.abs(P_perp - P_perp.T)))
            self.assertLess(symm_err, 1e-5, f"T={T}: Projector symmetry violation {symm_err}")

            # Invariant 3: Idempotency ||P_perp^2 - P_perp||_inf < 1e-5
            P_sq = np.dot(P_perp, P_perp)
            idemp_err = float(np.max(np.abs(P_sq - P_perp)))
            self.assertLess(idemp_err, 1e-5, f"T={T}: Projector idempotency violation {idemp_err}")

            # Invariant 4: Zero leakage on all prototypes: ||protos * P_perp||_inf < 1e-5
            leakage = np.dot(protos, P_perp)
            max_leakage = float(np.max(np.abs(leakage)))
            self.assertLess(max_leakage, 1e-5, f"T={T}: Prototype leakage detected {max_leakage}")

            # Invariant 5: Trace property Tr(P_perp) = dim - T
            trace_val = float(np.trace(P_perp))
            expected_trace = dim - T
            self.assertAlmostEqual(
                trace_val,
                expected_trace,
                places=3,
                msg=f"T={T}: Expected trace {expected_trace}, got {trace_val}",
            )

            # Invariant 6: Eigenvalues must be exactly 0 (T times) and 1 (dim - T times)
            evals = np.linalg.eigvalsh(P_perp)
            near_zeros = np.sum(np.abs(evals) < 1e-4)
            near_ones = np.sum(np.abs(evals - 1.0) < 1e-4)
            self.assertEqual(near_zeros, T, f"T={T}: Expected {T} zero eigenvalues, got {near_zeros}")
            self.assertEqual(
                near_ones,
                dim - T,
                f"T={T}: Expected {dim - T} unit eigenvalues, got {near_ones}",
            )

    # ------------------------------------------------------------------------
    # 2. Rank Exhaustion Stress (M -> d, M = d, M > d, M >> d)
    # ------------------------------------------------------------------------
    def test_02_rank_exhaustion_stress(self):
        """Stress null-space projector as prototype count approaches and exceeds dim."""
        dim = 32
        rng = np.random.RandomState(2002)

        # 2a. M = d - 1 (Subspace almost full, 1 degree of freedom remaining)
        basis_full, _ = np.linalg.qr(rng.randn(dim, dim))
        protos_d_minus_1 = basis_full[:, : dim - 1].T.astype(np.float32)
        P_d_minus_1 = ContinualLearningSequence.orthogonal_projector(protos_d_minus_1)

        self.assertAlmostEqual(float(np.trace(P_d_minus_1)), 1.0, places=3)
        # Residual on last basis vector should be exact identity (magnitude 1.0)
        last_basis_vec = basis_full[:, dim - 1].astype(np.float32)
        proj_last = np.dot(P_d_minus_1, last_basis_vec)
        self.assertAlmostEqual(float(np.linalg.norm(proj_last)), 1.0, places=4)

        # 2b. M = d (Full rank saturation: null space is empty {0})
        protos_d = basis_full.T.astype(np.float32)
        P_d = ContinualLearningSequence.orthogonal_projector(protos_d)
        max_entry = float(np.max(np.abs(P_d)))
        self.assertLess(max_entry, 1e-5, f"M=d: Expected zero matrix, got max entry {max_entry}")
        self.assertAlmostEqual(float(np.trace(P_d)), 0.0, places=4)

        # 2c. M = d + 16 (Overdetermined: M > d)
        protos_over = rng.randn(dim + 16, dim).astype(np.float32)
        P_over = ContinualLearningSequence.orthogonal_projector(protos_over)
        self.assertEqual(P_over.shape, (dim, dim))
        self.assertFalse(np.isnan(P_over).any())
        self.assertFalse(np.isinf(P_over).any())
        self.assertLess(float(np.max(np.abs(P_over))), 1e-5)

        # 2d. M = 4 * d (Extreme saturation: M >> d)
        protos_extreme = rng.randn(4 * dim, dim).astype(np.float32)
        P_extreme = ContinualLearningSequence.orthogonal_projector(protos_extreme)
        self.assertEqual(P_extreme.shape, (dim, dim))
        self.assertLess(float(np.max(np.abs(P_extreme))), 1e-5)

        # Verify any arbitrary gradient is completely quenched under full saturation
        test_grad = rng.randn(dim).astype(np.float32) * 50.0
        quenched = np.dot(P_extreme, test_grad)
        self.assertLess(float(np.linalg.norm(quenched)), 1e-5)

    # ------------------------------------------------------------------------
    # 3. Collinear, Identical, Scaled, and Zero Prototypes
    # ------------------------------------------------------------------------
    def test_03_collinear_and_degenerate_prototypes(self):
        """Stress projector with collinear, duplicate, scaled, and zero prototypes."""
        dim = 48
        rng = np.random.RandomState(3003)
        v1 = rng.randn(dim).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = rng.randn(dim).astype(np.float32)
        v2 /= np.linalg.norm(v2)

        # 3a. Identical duplicates (10 identical copies of v1)
        identical_protos = np.tile(v1, (10, 1))
        P_identical = ContinualLearningSequence.orthogonal_projector(identical_protos)
        self.assertAlmostEqual(float(np.trace(P_identical)), dim - 1, places=3)
        leakage_ident = float(np.max(np.abs(np.dot(identical_protos, P_identical))))
        self.assertLess(leakage_ident, 1e-5)

        # 3b. Scaled collinear vectors: [v1, -3*v1, 0.005*v1, 1000*v1]
        scaled_protos = np.vstack([v1, -3.0 * v1, 0.005 * v1, 1000.0 * v1])
        P_scaled = ContinualLearningSequence.orthogonal_projector(scaled_protos)
        self.assertAlmostEqual(float(np.trace(P_scaled)), dim - 1, places=3)
        res_v1 = np.dot(P_scaled, v1)
        self.assertLess(float(np.linalg.norm(res_v1)), 1e-5)

        # 3c. Linear combinations: [v1, v2, 2*v1 + 3*v2, -v1 + 0.5*v2]
        linear_combo_protos = np.vstack([v1, v2, 2.0 * v1 + 3.0 * v2, -1.0 * v1 + 0.5 * v2])
        P_combo = ContinualLearningSequence.orthogonal_projector(linear_combo_protos)
        # Numerical rank should be exactly 2
        self.assertAlmostEqual(float(np.trace(P_combo)), dim - 2, places=3)
        leakage_combo = float(np.max(np.abs(np.dot(linear_combo_protos, P_combo))))
        self.assertLess(leakage_combo, 1e-5)

        # 3d. Pure all-zero prototype: np.zeros((1, dim))
        zero_proto = np.zeros((1, dim), dtype=np.float32)
        P_zero = ContinualLearningSequence.orthogonal_projector(zero_proto)
        # Should return identity matrix without NaN or ZeroDivisionError
        self.assertTrue(np.allclose(P_zero, np.eye(dim), atol=1e-6))

        # 3e. Mixed zero and non-zero: [v1, 0, v2, 0]
        mixed_protos = np.vstack([v1, np.zeros((1, dim), dtype=np.float32), v2, np.zeros((1, dim), dtype=np.float32)])
        P_mixed = ContinualLearningSequence.orthogonal_projector(mixed_protos)
        self.assertFalse(np.isnan(P_mixed).any())
        self.assertAlmostEqual(float(np.trace(P_mixed)), dim - 2, places=3)
        res_mixed_v1 = float(np.linalg.norm(np.dot(P_mixed, v1)))
        res_mixed_v2 = float(np.linalg.norm(np.dot(P_mixed, v2)))
        self.assertLess(res_mixed_v1, 1e-5)
        self.assertLess(res_mixed_v2, 1e-5)

    # ------------------------------------------------------------------------
    # 4. High-Energy Adversarial Gradient Injection (1000x - 10000x scale)
    # ------------------------------------------------------------------------
    def test_04_high_energy_adversarial_gradient_injection(self):
        """Inject 1000x-10000x energy gradient perturbations and verify zero leakage."""
        dim = 64
        num_classes = 4
        rng = np.random.RandomState(4004)

        # Initialize model
        model = SimpleClassifier(in_features=dim, num_classes=num_classes)

        # 4 consolidated task prototypes
        Q_raw = rng.randn(4, dim).astype(np.float32)
        Q = Q_raw / np.linalg.norm(Q_raw, axis=1, keepdims=True)
        P_perp = ContinualLearningSequence.orthogonal_projector(Q)

        # Evaluate model outputs on consolidated prototypes prior to attack
        Q_tensor = torch.from_numpy(Q).float()
        with torch.no_grad():
            initial_outputs = model(Q_tensor).clone()

        # Adversarial attack 1: Gradient constructed ENTIRELY inside span(Q)
        # grad_adv = 1000 * alpha * Q
        alpha = rng.randn(num_classes, 4).astype(np.float32) * 1000.0
        grad_in_span = np.dot(alpha, Q)  # shape [num_classes, dim]

        # Apply null-space filter to gradient
        grad_proj_span = np.dot(grad_in_span, P_perp)
        # Because grad_in_span is entirely in span(Q), grad_proj_span MUST be 0
        norm_proj_span = float(np.linalg.norm(grad_proj_span))
        self.assertLess(
            norm_proj_span,
            1e-3,
            f"Adversarial in-span gradient leaked through P_perp with norm {norm_proj_span}",
        )

        # Adversarial attack 2: Massive 10,000x general perturbation gradient
        grad_huge = rng.randn(num_classes, dim).astype(np.float32) * 10000.0
        grad_proj_huge = np.dot(grad_huge, P_perp)

        # Mathematical zero leakage check: grad_proj_huge * Q.T == 0
        # In float32 arithmetic, 10,000x gradient * machine epsilon (~1.19e-7) * dimension yields numerical floor ~ 0.005
        leakage_matrix = np.dot(grad_proj_huge, Q.T)
        max_leakage_huge = float(np.max(np.abs(leakage_matrix)))
        self.assertLess(
            max_leakage_huge,
            0.01,
            f"Adversarial 10000x gradient leaked into protected prototypes: {max_leakage_huge}",
        )

        # Update model weight with projected adversarial gradient
        lr = 0.05
        with torch.no_grad():
            model.fc.weight.data.sub_(torch.from_numpy(grad_proj_huge) * lr)
            post_attack_outputs = model(Q_tensor)

        # Verify output logits on protected prototypes suffered negligible perturbation under 10,000x gradient
        output_diff = torch.max(torch.abs(post_attack_outputs - initial_outputs)).item()
        self.assertLess(
            output_diff,
            5e-4,
            f"Adversarial weight update shifted protected prototype outputs by {output_diff}",
        )

    # ------------------------------------------------------------------------
    # 5. Champion EWC + Orthogonal Grafting Under Adversarial Cross-Talk
    # ------------------------------------------------------------------------
    def test_05_champion_retention_under_adversarial_crosstalk(self):
        """Stress Champion retention under deliberate adversarial subspace cross-talk."""
        dim = 64
        num_classes = 4
        rng = np.random.RandomState(5005)

        # Construct Task A prototype
        proto_A = rng.randn(dim).astype(np.float32)
        proto_A /= np.linalg.norm(proto_A)

        # Construct Task B, C, D prototypes with deliberate high cross-talk:
        # proto_k = 0.85 * proto_A + 0.15 * orthogonal_component
        cross_talk_protos = {"Task_A": proto_A}
        for task_name in ("Task_B", "Task_C", "Task_D"):
            ortho = rng.randn(dim).astype(np.float32)
            ortho -= np.dot(ortho, proto_A) * proto_A
            ortho /= np.linalg.norm(ortho)
            # High adversarial cross-talk: 85% overlap with Task A!
            proto_k = 0.85 * proto_A + 0.15 * ortho
            proto_k /= np.linalg.norm(proto_k)
            cross_talk_protos[task_name] = proto_k

        # Generate synthetic data for each task
        def make_task_data(proto: np.ndarray, n_samples: int = 50) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            data = []
            for _ in range(n_samples):
                c = int(rng.randint(0, num_classes))
                vec = np.roll(proto, (c + 1) * 4) + 0.5 * proto
                vec = vec / np.linalg.norm(vec) + rng.randn(dim).astype(np.float32) * 0.10
                data.append((torch.from_numpy(vec).float(), torch.tensor(c, dtype=torch.long)))
            return data

        train_sets = {t: make_task_data(p, 60) for t, p in cross_talk_protos.items()}
        test_sets = {t: make_task_data(p, 30) for t, p in cross_talk_protos.items()}

        def evaluate(mod: nn.Module, ds: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
            mod.eval()
            corr = 0
            with torch.no_grad():
                for x, y in ds:
                    pred = int(torch.argmax(mod(x.unsqueeze(0)), dim=-1).item())
                    if pred == int(y.item()):
                        corr += 1
            return float(corr / len(ds))

        # --- Baseline: Naive Fine-Tuning under Adversarial Cross-Talk ---
        naive_model = SimpleClassifier(in_features=dim, num_classes=num_classes)
        # Train on Task A
        for _ in range(8):
            for x, y in train_sets["Task_A"]:
                loss = F.cross_entropy(naive_model(x.unsqueeze(0)), y.unsqueeze(0))
                naive_model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p in naive_model.parameters():
                        if p.grad is not None:
                            p.sub_(p.grad * 0.05)

        task_a_init_naive = evaluate(naive_model, test_sets["Task_A"])

        # Sequentially train on B, C, D (adversarial interferers)
        for t_name in ("Task_B", "Task_C", "Task_D"):
            for _ in range(8):
                for x, y in train_sets[t_name]:
                    loss = F.cross_entropy(naive_model(x.unsqueeze(0)), y.unsqueeze(0))
                    naive_model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for p in naive_model.parameters():
                            if p.grad is not None:
                                p.sub_(p.grad * 0.05)

        task_a_final_naive = evaluate(naive_model, test_sets["Task_A"])
        naive_retention = task_a_final_naive / max(1e-6, task_a_init_naive)

        # --- Champion: EWC + Orthogonal Grafting under Adversarial Cross-Talk ---
        champ_model = SimpleClassifier(in_features=dim, num_classes=num_classes)
        ewc = EWCRegularizer(lambda_reg=400.0)
        accum_protos = []

        # Train on Task A
        for _ in range(8):
            for x, y in train_sets["Task_A"]:
                loss = F.cross_entropy(champ_model(x.unsqueeze(0)), y.unsqueeze(0))
                champ_model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p in champ_model.parameters():
                        if p.grad is not None:
                            p.sub_(p.grad * 0.05)

        task_a_init_champ = evaluate(champ_model, test_sets["Task_A"])
        ewc.consolidate_task("Task_A", champ_model, train_sets["Task_A"])
        accum_protos.append(cross_talk_protos["Task_A"].reshape(1, -1))

        # Sequentially train on B, C, D with null-space projection & EWC
        for t_name in ("Task_B", "Task_C", "Task_D"):
            P_perp = ContinualLearningSequence.orthogonal_projector(np.vstack(accum_protos))

            for _ in range(8):
                for x, y in train_sets[t_name]:
                    loss = F.cross_entropy(champ_model(x.unsqueeze(0)), y.unsqueeze(0)) + ewc.penalty(champ_model)
                    champ_model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        for name, param in champ_model.named_parameters():
                            if "weight" in name and param.grad is not None:
                                grad_np = param.grad.data.cpu().numpy()
                                proj_np = np.dot(grad_np, P_perp)
                                param.grad.data.copy_(torch.from_numpy(proj_np))
                            if param.grad is not None:
                                param.sub_(param.grad * 0.05)

            ewc.consolidate_task(t_name, champ_model, train_sets[t_name])
            accum_protos.append(cross_talk_protos[t_name].reshape(1, -1))

        task_a_final_champ = evaluate(champ_model, test_sets["Task_A"])
        champ_retention = task_a_final_champ / max(1e-6, task_a_init_champ)

        # Empirical Assertions:
        # 1. Catastrophic forgetting check in naive baseline or retention differential
        retention_gap = champ_retention - naive_retention
        self.assertGreaterEqual(
            retention_gap,
            0.0,
            f"Champion retention ({champ_retention*100:.1f}%) should meet or exceed naive ({naive_retention*100:.1f}%)",
        )

        # 2. Champion retention must maintain >= 90.0% even under severe 85% cross-talk
        self.assertGreaterEqual(
            champ_retention,
            0.90,
            f"Champion retention collapsed under cross-talk: {champ_retention*100:.1f}% < 90.0%",
        )


if __name__ == "__main__":
    unittest.main()
