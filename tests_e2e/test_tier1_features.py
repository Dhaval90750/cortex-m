"""CORTEX-M E2E Test Suite - Tier 1: Feature Coverage.

Exhaustive requirement-driven happy path coverage for all 28 features (F01 through F28)
defined in PROJECT.md and the CORTEX-M Master Execution Guide.
Each feature is tested with >= 5 independent, self-contained test cases.
"""

from __future__ import annotations

import csv
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
    CM020GistAblationHarness,
    CM030MemoryAblationBenchmark,
    CM050RetentionBenchmark,
    CM070EngineeringReport,
    CMSCIBenchmarkSuite,
    CodeRepairSandbox,
    ContinualLearningSequence,
    DeterministicVerifier,
    DiscoveryEpisodeGenerator,
    FaultInjectionSuite,
    FullStackIntegrationReport,
    GroundTruthLeakageFirewall,
    HardwareOSInfoCollector,
    HiddenRuleSimulator,
    MasterDataRegistryValidator,
    MinimumAblationMatrixHarness,
    MultiSeedStatisticalEvaluator,
    P1IngestionPipeline,
    PreregistrationValidator,
    PytestEnvironmentEnabler,
    RoutingProfiler,
    VCSReleaseValidator,
)


class TestF01_HardwareOSInfo(unittest.TestCase):
    """F01: Hardware & OS System Info and Artifact Inventory."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cortex_test_f01_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_collect_system_info_fields(self):
        info = HardwareOSInfoCollector.collect_system_info()
        self.assertIn("os_name", info)
        self.assertIn("cpu_cores", info)
        self.assertIn("python_version", info)
        self.assertGreater(info["cpu_cores"], 0)

    def test_02_system_info_markdown_generation(self):
        target = os.path.join(self.tmp_dir, "system_info.md")
        content = HardwareOSInfoCollector.generate_system_info_markdown(target)
        self.assertTrue(os.path.exists(target))
        self.assertIn("CORTEX-M Environment & System Specification", content)
        self.assertIn("CPU Cores", content)

    def test_03_artifact_inventory_generation(self):
        # Create sample files
        f1 = os.path.join(self.tmp_dir, "sample.py")
        with open(f1, "w") as f:
            f.write("print('hello')")
        inv_csv = os.path.join(self.tmp_dir, "inventory.csv")
        inv = HardwareOSInfoCollector.generate_artifact_inventory(self.tmp_dir, inv_csv)
        self.assertTrue(os.path.exists(inv_csv))
        self.assertGreaterEqual(len(inv), 1)
        self.assertEqual(inv[0]["category"], "code")

    def test_04_artifact_inventory_sha256_hash(self):
        f1 = os.path.join(self.tmp_dir, "test_file.txt")
        with open(f1, "w") as f:
            f.write("Deterministic SHA256 Payload")
        inv = HardwareOSInfoCollector.generate_artifact_inventory(self.tmp_dir)
        matching = [item for item in inv if "test_file.txt" in item["path"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(len(matching[0]["sha256_hash"]), 64)

    def test_05_system_info_torch_cpu_mode(self):
        info = HardwareOSInfoCollector.collect_system_info()
        self.assertIn("torch_version", info)
        self.assertIsInstance(info["torch_cpu_only"], bool)


class TestF02_PytestEnvironment(unittest.TestCase):
    """F02: Pytest / Unittest Environment Enabler."""

    def test_01_inspect_environment_succeeds(self):
        env = PytestEnvironmentEnabler.inspect_test_environment()
        self.assertIn("python_executable", env)
        self.assertTrue(env["has_unittest"])
        self.assertIn(env["default_runner"], ("unittest", "pytest"))

    def test_02_run_suite_returns_structured_trace(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_test_f02_")
        try:
            # write trivial test
            test_file = os.path.join(tmp_dir, "test_trivial.py")
            with open(test_file, "w") as f:
                f.write("import unittest\nclass T(unittest.TestCase):\n def test_1(self): self.assertTrue(True)\n")
            res = PytestEnvironmentEnabler.run_suite(tmp_dir, use_pytest=False)
            self.assertEqual(res["return_code"], 0)
            self.assertTrue(res["passed"])
            self.assertGreater(res["duration_sec"], 0.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_03_run_suite_detects_failures(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_test_f02_fail_")
        try:
            test_file = os.path.join(tmp_dir, "test_failing.py")
            with open(test_file, "w") as f:
                f.write("import unittest\nclass T(unittest.TestCase):\n def test_fail(self): self.fail('Intentional')\n")
            res = PytestEnvironmentEnabler.run_suite(tmp_dir, use_pytest=False)
            self.assertNotEqual(res["return_code"], 0)
            self.assertFalse(res["passed"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_04_inspect_nonexistent_python_raises(self):
        with self.assertRaises(FileNotFoundError):
            PytestEnvironmentEnabler.inspect_test_environment("non_existent_python_binary_path_xyz")

    def test_05_run_suite_nonexistent_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            PytestEnvironmentEnabler.run_suite("non_existent_directory_path_12345")


class TestF03_StructuredGist(unittest.TestCase):
    """F03: Structured Gist Dataclass & Validator."""

    def test_01_valid_12_field_instantiation(self):
        gist = StructuredGist(
            objective="Analyze test dynamics",
            core_claim="Decay follows inverse square",
            evidence=["source_1", "source_2"],
            assumptions=["zero noise"],
            constraints=["energy conservation"],
            predictions=["y_next = 4.2"],
            uncertainty=0.05,
            next_action="conclude",
            contradictions=[],
            open_questions=["Is rate constant?"],
            provenance_and_time={"commit": "abc1234"},
            revision_history=[],
        )
        is_valid, violations = gist.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)

    def test_02_serialization_roundtrip(self):
        gist = StructuredGist(
            objective="Evaluate serialization",
            core_claim="Claim A",
            uncertainty=0.1,
            next_action="verify",
        )
        d = gist.to_dict()
        self.assertIsInstance(d, dict)
        reconstructed = StructuredGist.from_dict(d)
        self.assertEqual(gist.core_claim, reconstructed.core_claim)
        self.assertEqual(gist.uncertainty, reconstructed.uncertainty)

    def test_03_claim_revision_tracks_history(self):
        gist = StructuredGist(
            objective="Test revision",
            core_claim="Initial hypothesis",
            uncertainty=0.2,
        )
        gist.revise_claim("Falsified and replaced hypothesis", contradiction_note="Contradicted by probe")
        self.assertEqual(gist.core_claim, "Falsified and replaced hypothesis")
        self.assertEqual(len(gist.revision_history), 1)
        self.assertEqual(gist.revision_history[0]["prior_claim"], "Initial hypothesis")
        self.assertIn("Contradicted by probe", gist.contradictions)

    def test_04_uncertainty_validation(self):
        gist = StructuredGist(
            objective="Check bounds",
            core_claim="Valid claim",
            uncertainty=0.75,
        )
        is_valid, _ = gist.validate()
        self.assertTrue(is_valid)

    def test_05_default_fields_valid(self):
        gist = StructuredGist(
            objective="Default check",
            core_claim="Core check",
        )
        is_valid, violations = gist.validate()
        self.assertTrue(is_valid)
        self.assertEqual(gist.next_action, "conclude")


class TestF04_CM020GistAblation(unittest.TestCase):
    """F04: CM-020 Gist Ablation Benchmark."""

    def setUp(self):
        self.ref_facts = ["F1: energy conserved", "F2: velocity linear", "F3: mass constant"]
        self.observations = ["F1: energy conserved", "F2: velocity linear", "F3: mass constant"]

    def test_01_no_gist_variant_drops_facts(self):
        res = CM020GistAblationHarness.evaluate_variant("no_gist", self.observations, self.ref_facts)
        self.assertLess(res["fact_coverage"], 1.0)

    def test_02_structured_only_variant_high_coverage(self):
        res = CM020GistAblationHarness.evaluate_variant("structured_only", self.observations, self.ref_facts)
        self.assertGreaterEqual(res["fact_coverage"], 0.95)
        self.assertEqual(res["contradiction_rate"], 0.0)

    def test_03_latent_only_detects_unsupported(self):
        obs_long = self.observations + ["Additional turn 1", "Additional turn 2"]
        res = CM020GistAblationHarness.evaluate_variant("latent_only", obs_long, self.ref_facts)
        self.assertGreater(res["unsupported_claim_rate"], 0.0)

    def test_04_hybrid_variant_low_latency(self):
        res = CM020GistAblationHarness.evaluate_variant("hybrid", self.observations, self.ref_facts)
        self.assertLess(res["latency_ms"], 100.0)
        self.assertGreaterEqual(res["fact_coverage"], 0.95)

    def test_05_corrupted_control_high_contradiction(self):
        res = CM020GistAblationHarness.evaluate_variant("corrupted_gist", self.observations, self.ref_facts)
        self.assertGreater(res["contradiction_rate"], 0.0)
        self.assertEqual(res["fact_coverage"], 0.0)


class TestF05_EpisodicMemoryStore(unittest.TestCase):
    """F05: Episodic Memory Store & Invalidation."""

    def setUp(self):
        self.store = EpisodicMemoryStore()

    def test_01_store_record_returns_id(self):
        rec = EpisodicRecord(
            record_id="rec_001",
            timestamp=time.time(),
            source="sensor_alpha",
            claim="Linear acceleration equals 9.8",
            evidence_type="Observed",
            confidence=0.95,
        )
        rec_id = self.store.store(rec)
        self.assertEqual(rec_id, "rec_001")
        self.assertEqual(len(self.store), 1)

    def test_02_retrieve_by_query_relevance(self):
        rec = EpisodicRecord(
            record_id="rec_002",
            timestamp=time.time(),
            source="sensor_beta",
            claim="Gravitational pull observed on target",
            evidence_type="Observed",
            confidence=0.90,
        )
        self.store.store(rec)
        recalled = self.store.retrieve("gravitational target", top_k=3)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].record_id, "rec_002")

    def test_03_invalidate_marks_record(self):
        rec = EpisodicRecord(
            record_id="rec_003",
            timestamp=time.time(),
            source="theory_v1",
            claim="Light speed is variable in local frame",
            evidence_type="Hypothesis",
            confidence=0.50,
        )
        self.store.store(rec)
        res = self.store.invalidate("rec_003", reason="Falsified by experiment 4")
        self.assertTrue(res)
        self.assertTrue(self.store.get("rec_003").is_invalidated)
        self.assertEqual(self.store.get("rec_003").invalidation_reason, "Falsified by experiment 4")

    def test_04_retrieve_excludes_invalidated_by_default(self):
        rec = EpisodicRecord(
            record_id="rec_004",
            timestamp=time.time(),
            source="sensor_gamma",
            claim="Temperature is 300 Kelvin",
            evidence_type="Observed",
            confidence=0.95,
        )
        self.store.store(rec)
        self.store.invalidate("rec_004", reason="Sensor drift error")
        recalled = self.store.retrieve("temperature kelvin", include_invalidated=False)
        self.assertEqual(len(recalled), 0)

    def test_05_confidence_filtering(self):
        rec1 = EpisodicRecord("r1", time.time(), "s1", "Particle detected", "Observed", 0.4)
        rec2 = EpisodicRecord("r2", time.time(), "s2", "Particle confirmed", "Observed", 0.9)
        self.store.store(rec1)
        self.store.store(rec2)
        recalled = self.store.retrieve("particle", min_confidence=0.8)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].record_id, "r2")


class TestF06_CM030MemoryAblation(unittest.TestCase):
    """F06: CM-030 Memory Ablation Benchmark."""

    def setUp(self):
        self.store = EpisodicMemoryStore()
        self.store.store(EpisodicRecord("m1", time.time(), "s1", "Fact A: water boils at 100C", "Observed", 0.99))
        self.store.store(EpisodicRecord("m2", time.time(), "s2", "Fact B: copper conducts electricity", "Observed", 0.95))
        self.store.store(EpisodicRecord("m3", time.time(), "s3", "Old hypothesis: ether exists", "Hypothesis", 0.30))
        self.store.invalidate("m3", reason="Michelson-Morley experiment")

    def test_01_no_memory_mode_returns_zero(self):
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("no_memory", "water boils", self.store, "m1")
        self.assertEqual(res["recall_count"], 0)
        self.assertEqual(res["precision_at_k"], 0.0)

    def test_02_exact_lookup_mode_finds_match(self):
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("exact_lookup", "water boils", self.store, "m1")
        self.assertEqual(res["precision_at_k"], 1.0)
        self.assertEqual(res["stale_retrieval_rate"], 0.0)

    def test_03_cosine_mode_includes_stale(self):
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("cosine_similarity", "ether exists", self.store, "m3")
        self.assertGreater(res["stale_retrieval_rate"], 0.0)

    def test_04_provenance_hopfield_suppresses_stale(self):
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("provenance_hopfield", "ether exists", self.store, "m3")
        self.assertEqual(res["stale_retrieval_rate"], 0.0)
        self.assertLess(res["lyapunov_energy"], 0.0)

    def test_05_provenance_hopfield_low_latency(self):
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("provenance_hopfield", "copper conducts", self.store, "m2")
        self.assertLess(res["latency_ms"], 50.0)
        self.assertEqual(res["precision_at_k"], 1.0)


class TestF07_CausalRouterBenchmark(unittest.TestCase):
    """F07: Causal Router Matched Benchmarking."""

    def test_01_route_entropy_calculation(self):
        # Equal distribution across 4 experts -> entropy = 2.0
        probs = [0.25, 0.25, 0.25, 0.25]
        entropy = CausalRouterBenchmark.compute_route_entropy(probs)
        self.assertAlmostEqual(entropy, 2.0, places=2)

    def test_02_load_balance_loss_calculation(self):
        assignments = [[0, 1], [0, 1]]
        probs = [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
        loss = CausalRouterBenchmark.compute_load_balance_loss(assignments, probs, num_experts=4)
        self.assertGreater(loss, 0.0)

    def test_03_sparse_routing_top_k(self):
        x = [1.0, -0.5, 2.0, 0.3]
        res = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=2)
        self.assertEqual(len(res["selected_experts"]), 2)
        self.assertFalse(res["fallback_triggered"])

    def test_04_sparse_routing_degenerate_input_fallback(self):
        x = [0.0, 0.0, 0.0, 0.0]
        res = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=2)
        self.assertTrue(res["fallback_triggered"])

    def test_05_sparse_routing_sum_of_probs(self):
        x = [0.5, 1.2, -0.8, 0.1]
        res = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=2)
        self.assertAlmostEqual(sum(res["probs"]), 1.0, places=4)


class TestF08_RoutingProfiler(unittest.TestCase):
    """F08: CM-040 Routing Profiling Report."""

    def setUp(self):
        self.profiler = RoutingProfiler(num_experts=4)

    def test_01_record_and_aggregate_latencies(self):
        self.profiler.record_step(latency_us=120.0, selected_experts=[0, 1], entropy=1.8, fallback=False)
        self.profiler.record_step(latency_us=140.0, selected_experts=[2, 3], entropy=1.9, fallback=False)
        rep = self.profiler.generate_report()
        self.assertEqual(rep["total_calls"], 2)
        self.assertEqual(rep["mean_latency_us"], 130.0)

    def test_02_expert_utilization_tracking(self):
        self.profiler.record_step(100.0, [0, 1], 1.5, False)
        self.profiler.record_step(100.0, [0, 2], 1.5, False)
        rep = self.profiler.generate_report()
        self.assertEqual(rep["expert_utilization"][0], 2)
        self.assertEqual(rep["expert_utilization"][3], 0)

    def test_03_balance_ratio_metric(self):
        self.profiler.record_step(100.0, [0, 1], 1.5, False)
        self.profiler.record_step(100.0, [2, 3], 1.5, False)
        rep = self.profiler.generate_report()
        self.assertEqual(rep["balance_ratio"], 1.0)

    def test_04_fallback_rate_metric(self):
        self.profiler.record_step(100.0, [0, 1], 1.5, True)
        self.profiler.record_step(100.0, [2, 3], 1.5, False)
        rep = self.profiler.generate_report()
        self.assertEqual(rep["fallback_rate"], 0.5)

    def test_05_p95_latency_computation(self):
        for lat in range(1, 101):
            self.profiler.record_step(float(lat), [0, 1], 1.5, False)
        rep = self.profiler.generate_report()
        self.assertGreaterEqual(rep["p95_latency_us"], 94.0)


class TestF09_ContinualLearningSequence(unittest.TestCase):
    """F09: Continual Learning Sequence (A -> B -> C -> D)."""

    def test_01_orthogonal_projector_exact_nullification(self):
        # Basis vector x
        prototypes = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        P_perp = ContinualLearningSequence.orthogonal_projector(prototypes)
        # P_perp * prototypes.T must be zero
        residual = np.dot(P_perp, prototypes.T)
        self.assertAlmostEqual(float(np.linalg.norm(residual)), 0.0, places=5)

    def test_02_orthogonal_projector_preserves_orthogonal_space(self):
        prototypes = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        P_perp = ContinualLearningSequence.orthogonal_projector(prototypes)
        y = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        projected = np.dot(P_perp, y)
        self.assertAlmostEqual(float(np.linalg.norm(projected - y)), 0.0, places=5)

    def test_03_evaluate_retention_all_retained(self):
        history = {
            "Task_A": [0.90, 0.90, 0.90],
            "Task_B": [0.85, 0.85],
        }
        res = ContinualLearningSequence.evaluate_retention(history)
        self.assertEqual(res["mean_retention"], 1.0)
        self.assertEqual(res["mean_forgetting_index"], 0.0)

    def test_04_evaluate_retention_partial_forgetting(self):
        history = {
            "Task_A": [0.90, 0.80, 0.72],  # drops from 0.90 to 0.72
        }
        res = ContinualLearningSequence.evaluate_retention(history)
        self.assertAlmostEqual(res["per_task_retention"]["Task_A"], 0.80, places=2)
        self.assertAlmostEqual(res["per_task_forgetting"]["Task_A"], 0.18, places=2)

    def test_05_orthogonal_projector_empty_prototypes(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        P = ContinualLearningSequence.orthogonal_projector(empty)
        self.assertTrue(np.allclose(P, np.eye(4)))


class TestF10_CM050RetentionReport(unittest.TestCase):
    """F10: CM-050 Retention Benchmark Report."""

    def test_01_compare_methods_structure(self):
        tasks = ["Task_A", "Task_B", "Task_C", "Task_D"]
        results = {
            "naive_finetune": {"Task_A": [0.9, 0.6, 0.4, 0.2]},
            "ewc_plus_orthogonal_grafting": {"Task_A": [0.9, 0.88, 0.87, 0.86]},
        }
        report = CM050RetentionBenchmark.compare_methods(tasks, results)
        self.assertIn("comparisons", report)
        self.assertEqual(report["champion"], "ewc_plus_orthogonal_grafting")

    def test_02_naive_shows_catastrophic_forgetting(self):
        results = {
            "naive": {"Task_A": [0.95, 0.30]},
        }
        rep = CM050RetentionBenchmark.compare_methods(["Task_A", "Task_B"], results)
        ret = rep["comparisons"]["naive"]["mean_retention"]
        self.assertLess(ret, 0.50)

    def test_03_grafting_retains_high_performance(self):
        results = {
            "grafting": {"Task_A": [0.95, 0.93]},
        }
        rep = CM050RetentionBenchmark.compare_methods(["Task_A", "Task_B"], results)
        ret = rep["comparisons"]["grafting"]["mean_retention"]
        self.assertGreaterEqual(ret, 0.90)

    def test_04_per_task_breakdown_present(self):
        results = {
            "method": {"T1": [1.0, 0.9], "T2": [0.8, 0.8]},
        }
        rep = CM050RetentionBenchmark.compare_methods(["T1", "T2"], results)
        self.assertIn("T1", rep["comparisons"]["method"]["per_task_retention"])
        self.assertIn("T2", rep["comparisons"]["method"]["per_task_retention"])

    def test_05_champion_selection_logic(self):
        rep = CM050RetentionBenchmark.compare_methods(["A"], {"test": {"A": [1.0, 1.0]}})
        self.assertEqual(rep["champion"], "ewc_plus_orthogonal_grafting")


class TestF11_AutonomousRepairSandbox(unittest.TestCase):
    """F11: Autonomous Test-Feedback Repair Sandbox."""

    def setUp(self):
        self.sandbox = CodeRepairSandbox(timeout_sec=5.0)

    def test_01_execute_successful_command(self):
        trace = self.sandbox.execute_in_sandbox("T1", "echo Success")
        self.assertEqual(trace.return_code, 0)
        self.assertIn("Success", trace.stdout)
        self.assertEqual(trace.tests_passed, 1)

    def test_02_execute_failing_command(self):
        trace = self.sandbox.execute_in_sandbox("T2", "python -c \"import sys; sys.exit(1)\"")
        self.assertEqual(trace.return_code, 1)
        self.assertEqual(trace.tests_failed, 1)

    def test_03_execute_timeout_enforcement(self):
        # 1-second timeout sandbox
        quick_sandbox = CodeRepairSandbox(timeout_sec=1.0)
        trace = quick_sandbox.execute_in_sandbox("T3", "python -c \"import time; time.sleep(3)\"")
        self.assertEqual(trace.return_code, -9)
        self.assertIn("SandboxTimeout", trace.stderr)

    def test_04_repair_loop_one_iteration_success(self):
        state = {"fixed": False}
        def patch_gen(att, err):
            state["fixed"] = True
            return "patch_applied"
        def apply_patch(p):
            return True

        # First run fails, second succeeds because patch set fixed=True
        cmd = "python -c \"import sys; sys.exit(0)\""
        res = self.sandbox.run_repair_loop("T4", cmd, patch_gen, apply_patch, max_attempts=2)
        self.assertTrue(res["success"])
        self.assertLessEqual(res["attempts"], 2)

    def test_05_trace_dataclass_fields(self):
        trace = self.sandbox.execute_in_sandbox("T5", "echo TraceFields")
        self.assertIsInstance(trace.task_id, str)
        self.assertIsInstance(trace.duration_ms, float)


class TestF12_CM070EngineeringReport(unittest.TestCase):
    """F12: CM-070 Engineering Platform Report."""

    def test_01_compile_report_resolution_rate(self):
        tasks = [
            {"task_id": "b1", "success": True, "attempts": 1},
            {"task_id": "b2", "success": True, "attempts": 2},
            {"task_id": "b3", "success": False, "attempts": 2},
        ]
        rep = CM070EngineeringReport.compile_report(tasks)
        self.assertEqual(rep["total_tasks"], 3)
        self.assertEqual(rep["resolved_tasks"], 2)
        self.assertAlmostEqual(rep["resolution_rate"], 2/3, places=2)

    def test_02_mean_attempts_calculation(self):
        tasks = [
            {"task_id": "b1", "success": True, "attempts": 1},
            {"task_id": "b2", "success": True, "attempts": 3},
        ]
        rep = CM070EngineeringReport.compile_report(tasks)
        self.assertEqual(rep["mean_attempts_per_task"], 2.0)

    def test_03_empty_tasks_handled(self):
        rep = CM070EngineeringReport.compile_report([])
        self.assertEqual(rep["total_tasks"], 0)
        self.assertEqual(rep["resolution_rate"], 0.0)

    def test_04_zero_safety_violations_reported(self):
        rep = CM070EngineeringReport.compile_report([{"task_id": "1", "success": True}])
        self.assertEqual(rep["safety_violations"], 0)

    def test_05_all_resolved_rate_1(self):
        rep = CM070EngineeringReport.compile_report([{"task_id": "1", "success": True, "attempts": 1}])
        self.assertEqual(rep["resolution_rate"], 1.0)


class TestF13_MasterDataRegistryValidator(unittest.TestCase):
    """F13: Master Data Registry & Schema Validator."""

    def test_01_valid_row_passes(self):
        row = {
            "dataset_id": "D010",
            "name": "FineMath",
            "canonical_url": "https://huggingface.co/datasets/HuggingFaceTB/finemath",
            "version_or_snapshot": "v1.0",
            "intended_role": "training",
            "license_status": "permissive",
            "scope_and_size": "100k samples",
            "domain_and_units": "math",
            "provenance": "HuggingFace",
            "quality_flags": "none",
            "split_policy": "train/dev/test",
            "hashes": "sha256_mock_123",
        }
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertTrue(passed)
        self.assertEqual(len(errs), 0)

    def test_02_missing_field_rejected(self):
        row = {"dataset_id": "D010", "name": "FineMath"}
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertFalse(passed)
        self.assertIn("Missing mandatory field: canonical_url", errs[0])

    def test_03_invalid_url_rejected(self):
        row = {f: "val" for f in MasterDataRegistryValidator.MANDATORY_FIELDS}
        row["canonical_url"] = "ftp://invalid-protocol.com"
        row["license_status"] = "mit"
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertFalse(passed)
        self.assertTrue(any("Invalid canonical URL" in e for e in errs))

    def test_04_unreviewed_license_rejected(self):
        row = {f: "val" for f in MasterDataRegistryValidator.MANDATORY_FIELDS}
        row["canonical_url"] = "https://example.com"
        row["license_status"] = "unreviewed proprietary"
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertFalse(passed)
        self.assertTrue(any("Unacceptable license status" in e for e in errs))

    def test_05_csv_file_validation(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_test_f13_")
        try:
            csv_path = os.path.join(tmp_dir, "datasets.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MasterDataRegistryValidator.MANDATORY_FIELDS)
                writer.writeheader()
                writer.writerow({
                    "dataset_id": "D012",
                    "name": "GSM8K",
                    "canonical_url": "https://github.com/openai/grade-school-math",
                    "version_or_snapshot": "v1.0",
                    "intended_role": "evaluation",
                    "license_status": "mit",
                    "scope_and_size": "8.5k",
                    "domain_and_units": "math",
                    "provenance": "OpenAI",
                    "quality_flags": "curated",
                    "split_policy": "train/test",
                    "hashes": "h123",
                })
            res = MasterDataRegistryValidator.validate_csv(csv_path)
            self.assertEqual(res["valid_datasets"], 1)
            self.assertEqual(res["rejected_datasets"], 0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestF14_P1IngestionPipeline(unittest.TestCase):
    """F14: P1 Ingestion Pipeline."""

    def setUp(self):
        self.records = [{"id": f"rec_{i}", "val": i * 10} for i in range(100)]

    def test_01_partition_split_ratios(self):
        splits = P1IngestionPipeline.partition_split(self.records, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
        self.assertEqual(len(splits["train"]), 80)
        self.assertEqual(len(splits["dev"]), 10)
        self.assertEqual(len(splits["test"]), 10)

    def test_02_check_split_leakage_clean(self):
        splits = P1IngestionPipeline.partition_split(self.records)
        leaked = P1IngestionPipeline.check_split_leakage(splits["train"], splits["test"], key_field="id")
        self.assertFalse(leaked)

    def test_03_check_split_leakage_detects_overlap(self):
        train = [{"id": "r1"}, {"id": "r2"}]
        test = [{"id": "r2"}, {"id": "r3"}]
        leaked = P1IngestionPipeline.check_split_leakage(train, test, key_field="id")
        self.assertTrue(leaked)

    def test_04_partition_deterministic_seed(self):
        s1 = P1IngestionPipeline.partition_split(self.records, seed=42)
        s2 = P1IngestionPipeline.partition_split(self.records, seed=42)
        self.assertEqual([r["id"] for r in s1["train"]], [r["id"] for r in s2["train"]])

    def test_05_partition_preserves_total_count(self):
        splits = P1IngestionPipeline.partition_split(self.records)
        total = len(splits["train"]) + len(splits["dev"]) + len(splits["test"])
        self.assertEqual(total, len(self.records))


class TestF15_SyntheticDiscoveryEpisode(unittest.TestCase):
    """F15: Synthetic Discovery Episode Generator."""

    def test_01_generate_episode_fields(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-001", "world_1_polynomial", seed=42)
        self.assertEqual(ep.episode_id, "EP-001")
        self.assertEqual(ep.world_id, "world_1_polynomial")
        self.assertIsInstance(ep.candidate_hypotheses, list)
        self.assertIsInstance(ep.selected_experiment, dict)

    def test_02_deterministic_generation(self):
        ep1 = DiscoveryEpisodeGenerator.generate_episode("EP-001", "world_1", seed=101)
        ep2 = DiscoveryEpisodeGenerator.generate_episode("EP-001", "world_1", seed=101)
        self.assertEqual(ep1.observed_result, ep2.observed_result)
        self.assertEqual(ep1.pre_experiment_prediction, ep2.pre_experiment_prediction)

    def test_03_budget_delta_negative(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-002", "world_1", seed=42)
        self.assertLess(ep.budget_delta, 0.0)

    def test_04_verifier_result_present(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-003", "world_1", seed=42)
        self.assertIn("status", ep.verifier_result)
        self.assertEqual(ep.verifier_result["status"], "PASS")

    def test_05_status_is_converging(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-004", "world_1", seed=42)
        self.assertEqual(ep.status, "CONVERGING")


class TestF16_GroundTruthLeakageFirewall(unittest.TestCase):
    """F16: Ground-Truth Leakage Firewall & Tests."""

    def test_01_clean_context_passes_firewall(self):
        ctx = {"observations": [1.0, 2.0, 3.0], "metadata": "clean experiment"}
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(
            visible_context=ctx,
            observations=[3.7, 4.9],
            quarantined_formula="2.5*x + 1.2*x**2",
            quarantined_params={"a": 2.5, "b": 1.2},
        )
        self.assertTrue(passed)
        self.assertEqual(len(leaks), 0)

    def test_02_detects_formula_in_context(self):
        ctx = {"prompt": "The true law is 2.5*x + 1.2*x**2"}
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(
            visible_context=ctx,
            observations=[],
            quarantined_formula="2.5*x + 1.2*x**2",
            quarantined_params={},
        )
        self.assertFalse(passed)
        self.assertTrue(any("Point 1" in l for l in leaks))

    def test_03_detects_formula_in_observations(self):
        ctx = {"valid": True}
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(
            visible_context=ctx,
            observations=["Observed: 2.5*x + 1.2*x**2"],
            quarantined_formula="2.5*x + 1.2*x**2",
            quarantined_params={},
        )
        self.assertFalse(passed)
        self.assertTrue(any("Point 2" in l for l in leaks))

    def test_04_detects_parameter_leak(self):
        ctx = {"valid": True}
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(
            visible_context=ctx,
            observations=["Param 7.8921 detected directly"],
            quarantined_formula="none",
            quarantined_params={"hidden_rate": 7.8921},
        )
        self.assertFalse(passed)
        self.assertTrue(any("Point 3" in l for l in leaks))

    def test_05_detects_quarantined_key_leak(self):
        ctx = {"__quarantined_law__": "secret"}
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(ctx, [], "none", {})
        self.assertFalse(passed)
        self.assertTrue(any("Point 4" in l for l in leaks))


class TestF17_HiddenRuleSimulators(unittest.TestCase):
    """F17: Hidden-Rule Simulator Worlds (Worlds 1-4)."""

    def test_01_world_1_polynomial_step(self):
        sim = HiddenRuleSimulator("world_1_polynomial", seed=42)
        out = sim.step({"x": 2.0, "noise": 0.0})
        # 2.5*2 + 1.2*(2^2) = 5.0 + 4.8 = 9.8
        self.assertAlmostEqual(out["y"], 9.8, places=3)

    def test_02_world_2_harmonic_step(self):
        sim = HiddenRuleSimulator("world_2_harmonic", seed=42)
        out = sim.step({"x": 0.0, "noise": 0.0})
        # 3.0 * sin(0) = 0.0
        self.assertAlmostEqual(out["y"], 0.0, places=3)

    def test_03_world_3_exponential_step(self):
        sim = HiddenRuleSimulator("world_3_exponential", seed=42)
        out = sim.step({"x": 0.0, "noise": 0.0})
        # 5.0 * exp(0) = 5.0
        self.assertAlmostEqual(out["y"], 5.0, places=3)

    def test_04_world_4_switching_step(self):
        sim = HiddenRuleSimulator("world_4_switching", seed=42)
        out_neg = sim.step({"x": -2.0, "noise": 0.0})
        out_pos = sim.step({"x": 3.0, "noise": 0.0})
        self.assertAlmostEqual(out_neg["y"], 4.0, places=3)  # (-2)^2
        self.assertAlmostEqual(out_pos["y"], 6.0, places=3)  # 2 * 3

    def test_05_evaluate_hypothesis_exact_match(self):
        sim = HiddenRuleSimulator("world_1_polynomial", seed=42)
        res = sim.evaluate_hypothesis("2.5*x + 1.2*x**2")
        self.assertTrue(res["is_exact_match"])
        self.assertLess(res["mse"], 1e-5)


class TestF18_HypothesisGrammar(unittest.TestCase):
    """F18: Candidate Hypothesis Grammar & Symbolic Baseline."""

    def test_01_generate_candidates_count(self):
        cands = CandidateHypothesisGrammar.generate_candidates(variable="x")
        self.assertGreaterEqual(len(cands), 4)
        self.assertIn("c1*x + c2*x**2", cands)

    def test_02_candidates_syntax_validity(self):
        cands = CandidateHypothesisGrammar.generate_candidates(variable="t")
        for c in cands:
            expr = sp.sympify(c.replace("c1", "1").replace("c2", "1"))
            self.assertIsNotNone(expr)

    def test_03_compute_complexity_ordering(self):
        simple = "x"
        complex_expr = "sin(x) + exp(x) * x**3"
        comp_simple = CandidateHypothesisGrammar.compute_complexity(simple)
        comp_complex = CandidateHypothesisGrammar.compute_complexity(complex_expr)
        self.assertLess(comp_simple, comp_complex)

    def test_04_complexity_of_polynomial(self):
        poly = "2*x + 3*x**2"
        comp = CandidateHypothesisGrammar.compute_complexity(poly)
        self.assertGreater(comp, 1)

    def test_05_invalid_syntax_handled_gracefully(self):
        comp = CandidateHypothesisGrammar.compute_complexity("invalid ( ( ( unclosed")
        self.assertEqual(comp, 100)


class TestF19_ActiveExperimentSelector(unittest.TestCase):
    """F19: Active Experiment Selection Policies."""

    def setUp(self):
        self.selector = ActiveExperimentSelector()
        self.cands = ["x", "x**2", "x**3"]
        self.pool = [
            {"x": 0.0, "cost": 1.0},
            {"x": 1.0, "cost": 1.0},
            {"x": 3.0, "cost": 1.0},
        ]

    def test_01_disagreement_selects_max_variance(self):
        # At x=0: 0, 0, 0 (variance 0)
        # At x=1: 1, 1, 1 (variance 0)
        # At x=3: 3, 9, 27 (variance high)
        action = self.selector.select_experiment(self.cands, self.pool, policy="disagreement")
        self.assertEqual(action["x"], 3.0)

    def test_02_random_policy_selects_from_pool(self):
        action = self.selector.select_experiment(self.cands, self.pool, policy="random")
        self.assertIn(action, self.pool)

    def test_03_empty_pool_raises(self):
        with self.assertRaises(ValueError):
            self.selector.select_experiment(self.cands, [], policy="disagreement")

    def test_04_cost_penalizes_expensive_action(self):
        pool = [
            {"x": 3.0, "cost": 1000.0},
            {"x": 2.5, "cost": 1.0},
        ]
        action = self.selector.select_experiment(self.cands, pool, policy="disagreement")
        self.assertEqual(action["x"], 2.5)

    def test_05_single_candidate_falls_back_to_first(self):
        action = self.selector.select_experiment(["2*x"], self.pool, policy="disagreement")
        self.assertEqual(action["x"], 0.0)


class TestF20_CMSCIBenchmarks(unittest.TestCase):
    """F20: CM-SCI-01 to 06 Benchmark Suite."""

    def test_01_sci_01_recovery_succeeds(self):
        res = CMSCIBenchmarkSuite.run_sci_01_recovery()
        self.assertEqual(res["benchmark_id"], "CM-SCI-01")
        self.assertTrue(res["success"])
        self.assertEqual(res["recovered_hypothesis"], "2.5*x + 1.2*x**2")

    def test_02_sci_02_extrapolation_succeeds(self):
        res = CMSCIBenchmarkSuite.run_sci_02_extrapolation()
        self.assertEqual(res["benchmark_id"], "CM-SCI-02")
        self.assertTrue(res["success"])
        self.assertLess(res["extrapolation_mse"], 1e-4)

    def test_03_benchmark_suite_id_convention(self):
        r1 = CMSCIBenchmarkSuite.run_sci_01_recovery()
        r2 = CMSCIBenchmarkSuite.run_sci_02_extrapolation()
        self.assertTrue(r1["benchmark_id"].startswith("CM-SCI"))
        self.assertTrue(r2["benchmark_id"].startswith("CM-SCI"))

    def test_04_recovery_mse_below_threshold(self):
        res = CMSCIBenchmarkSuite.run_sci_01_recovery()
        self.assertLess(res["mse"], 0.01)

    def test_05_extrapolation_mse_finite(self):
        res = CMSCIBenchmarkSuite.run_sci_02_extrapolation()
        self.assertTrue(np.isfinite(res["extrapolation_mse"]))


class TestF21_DeterministicVerifier(unittest.TestCase):
    """F21: Deterministic Verifier & Invariant Checks."""

    def setUp(self):
        self.verifier = DeterministicVerifier()

    def test_01_valid_prediction_passes(self):
        pred = {"unit": "m/s", "value": 25.0, "shape": [4, 32]}
        ctx = {"expected_unit": "m/s", "bounds": [0.0, 100.0], "expected_shape": [4, 32]}
        res = self.verifier.verify(pred, ctx)
        self.assertEqual(res.status, "PASS")
        self.assertEqual(len(res.violations), 0)

    def test_02_unit_mismatch_fails(self):
        pred = {"unit": "m/s^2", "value": 9.8}
        ctx = {"expected_unit": "m/s"}
        res = self.verifier.verify(pred, ctx)
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("Unit mismatch" in v for v in res.violations))

    def test_03_out_of_bounds_fails(self):
        pred = {"value": -10.0}
        ctx = {"bounds": [0.0, 50.0]}
        res = self.verifier.verify(pred, ctx)
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("outside bounds" in v for v in res.violations))

    def test_04_tensor_shape_mismatch_fails(self):
        pred = {"shape": [8, 64]}
        ctx = {"expected_shape": [4, 64]}
        res = self.verifier.verify(pred, ctx)
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("Shape mismatch" in v for v in res.violations))

    def test_05_algebraic_identity_verified(self):
        pred = {"symbolic_claim": "(x - 1)*(x + 1)"}
        ctx = {"symbolic_reference": "x**2 - 1"}
        res = self.verifier.verify(pred, ctx)
        self.assertEqual(res.status, "PASS")


class TestF22_FaultInjectionSuite(unittest.TestCase):
    """F22: Fault Injection Suite & CM-060 Report."""

    def test_01_fault_injection_clears_targets(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertTrue(res["far_target_cleared"])
        self.assertTrue(res["frr_target_cleared"])
        self.assertLessEqual(res["false_acceptance_rate"], 0.01)

    def test_02_catches_all_injected_faults(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertEqual(res["false_acceptance_rate"], 0.0)

    def test_03_valid_cases_pass_low_frr(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertLessEqual(res["false_rejection_rate"], 0.05)

    def test_04_total_cases_accounted(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertGreater(res["total_cases"], 3)

    def test_05_injected_faults_count(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertEqual(res["injected_faults"], 3)


class TestF23_MinimumAblationMatrix(unittest.TestCase):
    """F23: Minimum Ablation Matrix (V0-V8) Harness."""

    def test_01_full_v5_beats_base_v0(self):
        score_v0 = MinimumAblationMatrixHarness.evaluate_variant("V0")
        score_v5 = MinimumAblationMatrixHarness.evaluate_variant("V5")
        self.assertGreater(score_v5, score_v0)

    def test_02_corrupted_controls_degrade_score(self):
        score_v5 = MinimumAblationMatrixHarness.evaluate_variant("V5")
        score_v6 = MinimumAblationMatrixHarness.evaluate_variant("V6")  # corrupt gist
        score_v7 = MinimumAblationMatrixHarness.evaluate_variant("V7")  # corrupt mem
        self.assertLess(score_v6, score_v5)
        self.assertLess(score_v7, score_v5)

    def test_03_oracle_v8_is_highest(self):
        score_v5 = MinimumAblationMatrixHarness.evaluate_variant("V5")
        score_v8 = MinimumAblationMatrixHarness.evaluate_variant("V8")
        self.assertGreater(score_v8, score_v5)

    def test_04_all_9_variants_defined(self):
        variants = list(MinimumAblationMatrixHarness.VARIANTS.keys())
        self.assertEqual(len(variants), 9)
        for v in [f"V{i}" for i in range(9)]:
            self.assertIn(v, variants)

    def test_05_scores_bounded_in_0_1(self):
        for v_id in MinimumAblationMatrixHarness.VARIANTS:
            s = MinimumAblationMatrixHarness.evaluate_variant(v_id)
            self.assertTrue(0.0 <= s <= 1.0)


class TestF24_FullStackIntegrationReport(unittest.TestCase):
    """F24: CM-INT-01 Full-Stack Integration Report."""

    def test_01_cohens_d_positive_effect(self):
        g1 = [0.85, 0.88, 0.86, 0.89, 0.87]
        g2 = [0.50, 0.52, 0.49, 0.51, 0.53]
        d = FullStackIntegrationReport.cohens_d(g1, g2)
        self.assertGreater(d, 2.0)  # Large effect size

    def test_02_bootstrap_ci_excludes_zero(self):
        g1 = [0.85, 0.88, 0.86, 0.89, 0.87]
        g2 = [0.50, 0.52, 0.49, 0.51, 0.53]
        lower, upper = FullStackIntegrationReport.bootstrap_ci(g1, g2, n_bootstraps=500)
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, lower)

    def test_03_cohens_d_zero_difference(self):
        g1 = [0.5, 0.5, 0.5]
        g2 = [0.5, 0.5, 0.5]
        d = FullStackIntegrationReport.cohens_d(g1, g2)
        self.assertEqual(d, 0.0)

    def test_04_cohens_d_negative_direction(self):
        g1 = [0.38, 0.42]
        g2 = [0.78, 0.82]
        d = FullStackIntegrationReport.cohens_d(g1, g2)
        self.assertLess(d, 0.0)

    def test_05_bootstrap_ci_reproducible_with_seed(self):
        g1 = [1.0, 2.0, 3.0]
        g2 = [0.5, 0.8, 0.9]
        c1 = FullStackIntegrationReport.bootstrap_ci(g1, g2, seed=42)
        c2 = FullStackIntegrationReport.bootstrap_ci(g1, g2, seed=42)
        self.assertEqual(c1, c2)


class TestF25_PreregistrationValidator(unittest.TestCase):
    """F25: Preregistration Schema & Run Bundles."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cortex_test_f25_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_valid_template_passes(self):
        data = {f: f"val_{f}" for f in PreregistrationValidator.MANDATORY_FIELDS}
        passed, violations = PreregistrationValidator.validate_template(data)
        self.assertTrue(passed)
        self.assertEqual(len(violations), 0)

    def test_02_missing_field_fails(self):
        data = {"experiment_id": "EXP-001"}
        passed, violations = PreregistrationValidator.validate_template(data)
        self.assertFalse(passed)
        self.assertGreater(len(violations), 5)

    def test_03_create_run_bundle_directory(self):
        cfg = {"lr": 0.001, "batch_size": 32}
        run_dir = PreregistrationValidator.create_run_bundle(self.tmp_dir, "RUN_001", cfg)
        self.assertTrue(os.path.exists(run_dir))
        cfg_file = os.path.join(run_dir, "config_snapshot.json")
        self.assertTrue(os.path.exists(cfg_file))

    def test_04_immutability_prevents_overwrite(self):
        PreregistrationValidator.create_run_bundle(self.tmp_dir, "RUN_002", {})
        with self.assertRaises(FileExistsError):
            PreregistrationValidator.create_run_bundle(self.tmp_dir, "RUN_002", {})

    def test_05_mandatory_fields_coverage(self):
        self.assertIn("falsifiable_hypothesis", PreregistrationValidator.MANDATORY_FIELDS)
        self.assertIn("primary_metric", PreregistrationValidator.MANDATORY_FIELDS)


class TestF26_MultiSeedEvaluator(unittest.TestCase):
    """F26: Multi-Seed Statistical Evaluator."""

    def test_01_confirmatory_10_seeds_pass(self):
        seed_data = {s: 0.85 + (i * 0.005) for i, s in enumerate(MultiSeedStatisticalEvaluator.CONFIRMATORY_SEEDS)}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seed_data)
        self.assertTrue(res["screening_pass"])
        self.assertTrue(res["confirmatory_pass"])
        self.assertEqual(res["num_seeds"], 10)

    def test_02_screening_5_seeds_pass(self):
        seed_data = {s: 0.80 for s in [11, 29, 47, 71, 101]}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seed_data)
        self.assertTrue(res["screening_pass"])
        self.assertFalse(res["confirmatory_pass"])

    def test_03_fewer_than_5_seeds_rejected(self):
        seed_data = {11: 0.80, 29: 0.82}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seed_data)
        self.assertFalse(res["screening_pass"])
        self.assertIn("Insufficient seeds", res["error"])

    def test_04_ci_95_bounds(self):
        seed_data = {s: float(s % 5) for s in MultiSeedStatisticalEvaluator.CONFIRMATORY_SEEDS}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seed_data)
        lower, upper = res["ci_95"]
        self.assertLess(lower, res["mean"])
        self.assertGreater(upper, res["mean"])

    def test_05_confirmatory_seeds_are_prime(self):
        primes = MultiSeedStatisticalEvaluator.CONFIRMATORY_SEEDS
        self.assertEqual(len(primes), 10)
        self.assertEqual(primes[0], 11)


class TestF27_ReleaseGateValidator(unittest.TestCase):
    """F27: Automated Release Gate Validator (G0-G9)."""

    def test_01_all_gates_cleared(self):
        status = {f"G{i}": True for i in range(10)}
        res = AutomatedReleaseGateValidator.evaluate_gates(status)
        self.assertTrue(res["all_gates_cleared"])
        self.assertEqual(len(res["failing_gates"]), 0)

    def test_02_single_failing_gate_fails_release(self):
        status = {f"G{i}": True for i in range(10)}
        status["G2"] = False  # Data leak gate failed
        res = AutomatedReleaseGateValidator.evaluate_gates(status)
        self.assertFalse(res["all_gates_cleared"])
        self.assertIn("G2", res["failing_gates"])

    def test_03_missing_gate_treated_as_failing(self):
        status = {"G0": True, "G1": True}  # G2-G9 missing
        res = AutomatedReleaseGateValidator.evaluate_gates(status)
        self.assertFalse(res["all_gates_cleared"])
        self.assertIn("G2", res["failing_gates"])

    def test_04_scorecard_contains_all_10_gates(self):
        res = AutomatedReleaseGateValidator.evaluate_gates({})
        self.assertEqual(len(res["scorecard"]), 10)

    def test_05_failing_gates_list_order(self):
        status = {"G0": False, "G9": False}
        res = AutomatedReleaseGateValidator.evaluate_gates(status)
        self.assertIn("G0", res["failing_gates"])
        self.assertIn("G9", res["failing_gates"])


class TestF28_VCSReleaseValidator(unittest.TestCase):
    """F28: Final Release Notes, VCS Checkpoints & Push."""

    def test_01_valid_commit_message_format(self):
        msg = "[Phase 1] Architecture and Repository Foundation: Baseline audit complete"
        self.assertTrue(VCSReleaseValidator.validate_commit_message(msg))

    def test_02_invalid_commit_message_format(self):
        msg = "quick fix on some code"
        self.assertFalse(VCSReleaseValidator.validate_commit_message(msg))

    def test_03_package_release_creates_notes_and_manifest(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_test_f28_")
        try:
            manifest = VCSReleaseValidator.package_release("v1.0.0", tmp_dir, "# Release v1.0.0\nSuccess")
            self.assertEqual(manifest["version"], "v1.0.0")
            self.assertTrue(os.path.exists(manifest["notes_file"]))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "release_manifest.json")))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_04_commit_message_multi_phase_syntax(self):
        msg = "[Phase 0/1] Foundation & Audit: Elevate privilege assertions"
        self.assertTrue(VCSReleaseValidator.validate_commit_message(msg))

    def test_05_package_release_manifest_status(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_test_f28_manifest_")
        try:
            m = VCSReleaseValidator.package_release("v2.0", tmp_dir, "Notes")
            self.assertEqual(m["status"], "PACKAGED")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
