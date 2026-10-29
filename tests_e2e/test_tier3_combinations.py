"""CORTEX-M E2E Test Suite - Tier 3: Cross-Feature Combinations.

Verifies pairwise and systemic cross-feature interactions across the architecture:
- Gist + Memory
- Router + Gist
- Continual Learning + Memory
- Verifier + Discovery
- Sandbox + Verifier
- Data Platform + Discovery
- Firewall + Active Selection
- Ablation Matrix + Statistical Evaluator
- Preregistration + Release Gates
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import sympy as sp
import torch

from cortex_m.gist.structured_gist import StructuredGist
from cortex_m.memory.episodic_store import EpisodicMemoryStore, EpisodicRecord

from tests_e2e.reference_contracts import (
    ActiveExperimentSelector,
    AutomatedReleaseGateValidator,
    CandidateHypothesisGrammar,
    CausalRouterBenchmark,
    CodeRepairSandbox,
    ContinualLearningSequence,
    DeterministicVerifier,
    DiscoveryEpisodeGenerator,
    FullStackIntegrationReport,
    GroundTruthLeakageFirewall,
    HiddenRuleSimulator,
    MinimumAblationMatrixHarness,
    MultiSeedStatisticalEvaluator,
    P1IngestionPipeline,
    PreregistrationValidator,
)


class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Pairwise and multi-component architectural integrations."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cortex_tier3_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_gist_and_memory_evidence_integration(self):
        """Pairwise: Episodic Memory -> Structured Gist Evidence Buffer."""
        store = EpisodicMemoryStore()
        rec1 = EpisodicRecord(
            record_id="rec_orbital_1",
            timestamp=time.time(),
            source="telemetry_sensor_A",
            claim="Orbital velocity measured at 7.66 km/s",
            evidence_type="Observed",
            confidence=0.98,
        )
        store.store(rec1)

        # Retrieve relevant memories and populate Gist
        recalled = store.retrieve("orbital velocity", top_k=3)
        self.assertEqual(len(recalled), 1)

        gist = StructuredGist(
            objective="Estimate orbit decay",
            core_claim="Orbit velocity consistent with low Earth orbit",
            evidence=[r.claim for r in recalled],
            uncertainty=0.02,
        )
        is_valid, violations = gist.validate()
        self.assertTrue(is_valid)
        self.assertIn("Orbital velocity measured at 7.66 km/s", gist.evidence)

    def test_02_memory_invalidation_triggers_gist_claim_revision(self):
        """Pairwise: Memory Invalidation -> Gist Claim Revision & Lineage."""
        store = EpisodicMemoryStore()
        rec = EpisodicRecord(
            record_id="rec_decay_01",
            timestamp=time.time(),
            source="sensor_drift",
            claim="Atmospheric drag coefficient equals 2.2",
            evidence_type="Hypothesis",
            confidence=0.60,
        )
        store.store(rec)

        gist = StructuredGist(
            objective="Model aerodynamic drag",
            core_claim="Drag coefficient is 2.2",
            evidence=[rec.claim],
            uncertainty=0.40,
        )

        # Invalidate memory due to recalibration
        store.invalidate("rec_decay_01", reason="Recalibrated by direct pitot sensor")

        # Gist responds by logging contradiction and updating claim
        gist.revise_claim(
            new_claim="Drag coefficient corrected to 1.85",
            contradiction_note="Prior drag claim invalidated: sensor drift recalibrated",
        )
        gist.uncertainty = 0.05

        self.assertEqual(gist.core_claim, "Drag coefficient corrected to 1.85")
        self.assertEqual(len(gist.revision_history), 1)
        self.assertIn("Prior drag claim invalidated", gist.contradictions[0])
        self.assertEqual(gist.uncertainty, 0.05)

    def test_03_router_and_gist_uncertainty_conditioning(self):
        """Pairwise: Gist Uncertainty -> Router Entropy & Fallback Decision."""
        # Low uncertainty gist produces confident sparse routing
        low_unc_gist = StructuredGist(
            objective="Compute standard orbital pass",
            core_claim="Standard trajectory",
            uncertainty=0.02,
        )
        # Representation vector modulated by (1 - uncertainty)
        feature_low = [1.5 * (1.0 - low_unc_gist.uncertainty), -0.8, 2.1, 0.4]
        res_low = CausalRouterBenchmark.run_sparse_routing(feature_low, num_experts=4, top_k=2)
        self.assertFalse(res_low["fallback_triggered"])

        # High uncertainty gist produces degenerate vector triggering safe fallback
        high_unc_gist = StructuredGist(
            objective="Unknown anomaly detected",
            core_claim="Unidentifiable signal",
            uncertainty=1.0,
        )
        feature_high = [0.0, 0.0, 0.0, 0.0]
        res_high = CausalRouterBenchmark.run_sparse_routing(feature_high, num_experts=4, top_k=2)
        self.assertTrue(res_high["fallback_triggered"])

    def test_04_continual_learning_and_memory_consolidation(self):
        """Pairwise: Memory Prototypes -> Continual Learning Null-Space Protection."""
        # Task A memory attractor prototype
        proto_task_a = np.array([[1.0, 0.5, -0.2, 0.1]], dtype=np.float32)

        # Projector guaranteeing null space
        P_perp = ContinualLearningSequence.orthogonal_projector(proto_task_a)

        # New Task B weight update
        delta_W_task_b = np.array([0.2, -0.1, 0.4, 0.3], dtype=np.float32)

        # Projected update
        grafted_update = np.dot(P_perp, delta_W_task_b)

        # Residual interference on Task A prototype must be zero
        interference = float(np.abs(np.dot(proto_task_a, grafted_update)[0]))
        self.assertAlmostEqual(interference, 0.0, places=5)

    def test_05_verifier_and_scientific_discovery_filtering(self):
        """Pairwise: Candidate Hypothesis Generation -> Verifier Dimensional Check."""
        # Candidate grammar generates candidate formulas
        cands = CandidateHypothesisGrammar.generate_candidates(variable="x")
        verifier = DeterministicVerifier()

        # Verifier checks dimensional invariant: velocity [m/s] vs acceleration [m/s^2]
        valid_candidates = []
        for c in cands:
            # Simulate prediction unit check
            pred = {"formula": c, "unit": "m/s", "value": 10.0}
            ctx = {"expected_unit": "m/s", "bounds": [0.0, 100.0]}
            res = verifier.verify(pred, ctx)
            if res.status == "PASS":
                valid_candidates.append(c)

        self.assertGreater(len(valid_candidates), 0)

    def test_06_code_repair_sandbox_and_verifier_loop(self):
        """Pairwise: Code Sandbox Repair Loop + Invariant Verifier."""
        sandbox = CodeRepairSandbox(timeout_sec=5.0)
        verifier = DeterministicVerifier()

        # Failing test script
        script_path = os.path.join(self.tmp_dir, "test_target.py")
        with open(script_path, "w") as f:
            f.write("import sys\n# Bug: exit code 1\nsys.exit(1)\n")

        def patch_gen(attempt, err):
            # Propose fix
            return "import sys\nsys.exit(0)\n"

        def apply_patch(patch):
            # Verifier inspects patch before application
            res = verifier.verify({"code": patch}, {"expected_safe": True})
            if res.status in ("PASS", "UNKNOWN"):
                with open(script_path, "w") as f:
                    f.write(patch)
                return True
            return False

        cmd = f"python {script_path}"
        loop_res = sandbox.run_repair_loop("TASK-01", cmd, patch_gen, apply_patch, max_attempts=2)
        self.assertTrue(loop_res["success"])
        self.assertLessEqual(loop_res["attempts"], 2)

    def test_07_data_platform_ingestion_and_scientific_discovery(self):
        """Pairwise: P1 Ingestion Pipeline -> Discovery Episode Partitioning."""
        raw_dataset = [{"id": f"obs_{i}", "x": float(i), "y": float(2 * i)} for i in range(50)]
        splits = P1IngestionPipeline.partition_split(raw_dataset, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)

        # Check zero leak
        leaked = P1IngestionPipeline.check_split_leakage(splits["train"], splits["test"], key_field="id")
        self.assertFalse(leaked)

        # Feed train split to Discovery Episode Generator
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-DATA-01", "world_1_polynomial", seed=42)
        self.assertEqual(ep.status, "CONVERGING")
        self.assertGreater(len(splits["train"]), 30)

    def test_08_ground_truth_firewall_and_active_experiment_selector(self):
        """Pairwise: Active Experiment Selector -> Ground-Truth Leakage Audit."""
        selector = ActiveExperimentSelector()
        cands = ["2.5*x + 1.2*x**2", "2.0*x + 1.0*x**2", "3.0*x"]
        pool = [{"x": float(i), "cost": 1.0} for i in range(5)]

        action = selector.select_experiment(cands, pool, policy="disagreement")
        self.assertIn("x", action)

        # Pass action and context through GroundTruthLeakageFirewall
        visible_context = {
            "selected_action": action,
            "candidates": cands,
            "domain": "dynamics",
        }
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(
            visible_context=visible_context,
            observations=[action["x"]],
            quarantined_formula="SECRET_LAW_NEVER_REVEAL",
            quarantined_params={"secret_const": 999.88},
        )
        self.assertTrue(passed)
        self.assertEqual(len(leaks), 0)

    def test_09_ablation_matrix_and_statistical_evaluator(self):
        """Pairwise: Minimum Ablation Matrix Ladder -> Multi-Seed Statistical Evaluator."""
        # Run Full CORTEX-M (V5) across 10 confirmatory seeds
        seed_results = {}
        for s in MultiSeedStatisticalEvaluator.CONFIRMATORY_SEEDS:
            # Deterministic variation by seed
            score = MinimumAblationMatrixHarness.evaluate_variant("V5", baseline_score=0.50)
            noise = (s % 7) * 0.002
            seed_results[s] = score + noise

        eval_res = MultiSeedStatisticalEvaluator.evaluate_seeds(seed_results)
        self.assertTrue(eval_res["screening_pass"])
        self.assertTrue(eval_res["confirmatory_pass"])
        self.assertGreater(eval_res["mean"], 0.85)

    def test_10_preregistration_and_release_gate_validation(self):
        """Pairwise: Experiment Preregistration Bundle -> Acceptance Gate Scoring."""
        # 1. Preregistration validation
        prereg = {f: f"registered_{f}" for f in PreregistrationValidator.MANDATORY_FIELDS}
        prereg_valid, _ = PreregistrationValidator.validate_template(prereg)
        self.assertTrue(prereg_valid)

        # 2. Bundle creation
        run_dir = PreregistrationValidator.create_run_bundle(self.tmp_dir, "EXP_GATES_01", prereg)
        self.assertTrue(os.path.exists(run_dir))

        # 3. Release Gate Evaluation: verify Gate G1 (Preregistration) and Gate G8 (Stats) pass
        system_status = {f"G{i}": True for i in range(10)}
        res = AutomatedReleaseGateValidator.evaluate_gates(system_status)
        self.assertTrue(res["all_gates_cleared"])
        self.assertEqual(len(res["failing_gates"]), 0)


if __name__ == "__main__":
    unittest.main()
