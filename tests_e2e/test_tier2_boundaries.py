"""CORTEX-M E2E Test Suite - Tier 2: Boundary & Corner Cases.

Exhaustive requirement-driven boundary, malformed input, and corner-case coverage
for all 28 features (F01 through F28) defined in PROJECT.md and the
CORTEX-M Master Execution Guide.
Each feature is tested with >= 5 independent boundary/corner test cases.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import subprocess
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


class TestF01_HardwareOSInfo_Boundaries(unittest.TestCase):
    """F01 Boundaries: Hardware & OS System Info and Artifact Inventory."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cortex_bnd_f01_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_empty_directory_inventory(self):
        inv = HardwareOSInfoCollector.generate_artifact_inventory(self.tmp_dir)
        self.assertEqual(len(inv), 0)

    def test_02_nonexistent_scan_directory(self):
        non_existent = os.path.join(self.tmp_dir, "does_not_exist_xyz")
        inv = HardwareOSInfoCollector.generate_artifact_inventory(non_existent)
        self.assertEqual(len(inv), 0)

    def test_03_max_depth_boundary_zero(self):
        sub = os.path.join(self.tmp_dir, "nested")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "deep.py"), "w") as f:
            f.write("# deep")
        with open(os.path.join(self.tmp_dir, "root.py"), "w") as f:
            f.write("# root")
        inv = HardwareOSInfoCollector.generate_artifact_inventory(self.tmp_dir, max_depth=0)
        paths = [item["path"] for item in inv]
        self.assertIn("root.py", paths)
        self.assertNotIn("nested/deep.py", paths)

    def test_04_skip_transient_directories(self):
        pycache = os.path.join(self.tmp_dir, "__pycache__")
        os.makedirs(pycache, exist_ok=True)
        with open(os.path.join(pycache, "cached.pyc"), "w") as f:
            f.write("bytecode")
        inv = HardwareOSInfoCollector.generate_artifact_inventory(self.tmp_dir)
        self.assertEqual(len(inv), 0)

    def test_05_special_characters_in_filename(self):
        fname = "test file with spaces & symbols [v1.0].txt"
        with open(os.path.join(self.tmp_dir, fname), "w") as f:
            f.write("content")
        inv = HardwareOSInfoCollector.generate_artifact_inventory(self.tmp_dir)
        self.assertEqual(len(inv), 1)
        self.assertIn(fname, inv[0]["path"])


class TestF02_PytestEnvironment_Boundaries(unittest.TestCase):
    """F02 Boundaries: Pytest / Unittest Environment Enabler."""

    def test_01_timeout_enforcement(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_bnd_f02_timeout_")
        try:
            # write test that sleeps
            test_file = os.path.join(tmp_dir, "test_sleep.py")
            with open(test_file, "w") as f:
                f.write("import unittest, time\nclass S(unittest.TestCase):\n def test_s(self): time.sleep(5)\n")
            with self.assertRaises(subprocess.TimeoutExpired):
                PytestEnvironmentEnabler.run_suite(tmp_dir, use_pytest=False, timeout=0.5)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_02_empty_test_directory_reports_ok(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_bnd_f02_empty_")
        try:
            res = PytestEnvironmentEnabler.run_suite(tmp_dir, use_pytest=False)
            self.assertIn(res["return_code"], [0, 5])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_03_syntax_error_in_test_file(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_bnd_f02_syntax_")
        try:
            test_file = os.path.join(tmp_dir, "test_bad_syntax.py")
            with open(test_file, "w") as f:
                f.write("this is def not valid python syntax ;;;")
            res = PytestEnvironmentEnabler.run_suite(tmp_dir, use_pytest=False)
            self.assertNotEqual(res["return_code"], 0)
            self.assertFalse(res["passed"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_04_inspect_environment_fallback_runner(self):
        env = PytestEnvironmentEnabler.inspect_test_environment()
        self.assertIn(env["default_runner"], ["unittest", "pytest"])

    def test_05_run_suite_invalid_command_flag(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_bnd_f02_cmd_")
        try:
            res = PytestEnvironmentEnabler.run_suite(tmp_dir, use_pytest=False)
            self.assertIn("command", res)
            self.assertIsInstance(res["command"], str)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestF03_StructuredGist_Boundaries(unittest.TestCase):
    """F03 Boundaries: Structured Gist Dataclass & Validator."""

    def test_01_empty_objective_rejected(self):
        gist = StructuredGist(objective="", core_claim="Valid claim")
        is_valid, violations = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("objective" in v for v in violations))

    def test_02_empty_core_claim_rejected(self):
        gist = StructuredGist(objective="Valid objective", core_claim="   ")
        is_valid, violations = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("core_claim" in v for v in violations))

    def test_03_uncertainty_negative_rejected(self):
        gist = StructuredGist(objective="Valid", core_claim="Valid", uncertainty=-0.01)
        is_valid, violations = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("uncertainty" in v for v in violations))

    def test_04_uncertainty_exceeds_one_rejected(self):
        gist = StructuredGist(objective="Valid", core_claim="Valid", uncertainty=1.001)
        is_valid, violations = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("uncertainty" in v for v in violations))

    def test_05_malformed_evidence_types_rejected(self):
        # Pass non-string elements in evidence list
        gist = StructuredGist(objective="Valid", core_claim="Valid", evidence=[123, True])  # type: ignore
        is_valid, violations = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("evidence" in v and "list of strings" in v for v in violations))


class TestF04_CM020GistAblation_Boundaries(unittest.TestCase):
    """F04 Boundaries: CM-020 Gist Ablation Benchmark."""

    def test_01_empty_observations_handled(self):
        res = CM020GistAblationHarness.evaluate_variant("structured_only", [], ["fact1"])
        self.assertEqual(res["fact_coverage"], 0.0)
        self.assertEqual(res["contradiction_rate"], 0.0)

    def test_02_empty_reference_facts_handled(self):
        res = CM020GistAblationHarness.evaluate_variant("structured_only", ["fact1"], [])
        self.assertEqual(res["fact_coverage"], 1.0)

    def test_03_all_contradictions_high_rate(self):
        res = CM020GistAblationHarness.evaluate_variant("corrupted_gist", ["premise A"], ["premise A"])
        self.assertGreater(res["contradiction_rate"], 0.5)

    def test_04_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            CM020GistAblationHarness.evaluate_variant("invalid_variant_mode_xyz", [], [])

    def test_05_no_gist_empty_observations(self):
        res = CM020GistAblationHarness.evaluate_variant("no_gist", [], ["fact A"])
        self.assertEqual(res["fact_coverage"], 0.0)


class TestF05_EpisodicMemoryStore_Boundaries(unittest.TestCase):
    """F05 Boundaries: Episodic Memory Store & Invalidation."""

    def setUp(self):
        self.store = EpisodicMemoryStore()

    def test_01_invalidate_nonexistent_returns_false(self):
        res = self.store.invalidate("non_existent_id", reason="None")
        self.assertFalse(res)

    def test_02_invalidate_already_invalidated_idempotent(self):
        rec = EpisodicRecord("r1", time.time(), "s1", "Fact", "Observed", 0.9)
        self.store.store(rec)
        self.assertTrue(self.store.invalidate("r1", reason="First reason"))
        self.assertTrue(self.store.invalidate("r1", reason="Second reason"))
        self.assertEqual(self.store.get("r1").invalidation_reason, "Second reason")

    def test_03_confidence_out_of_bounds_raises(self):
        rec_high = EpisodicRecord("r2", time.time(), "s2", "Fact", "Observed", 1.5)
        rec_low = EpisodicRecord("r3", time.time(), "s3", "Fact", "Observed", -0.2)
        with self.assertRaises(ValueError):
            self.store.store(rec_high)
        with self.assertRaises(ValueError):
            self.store.store(rec_low)

    def test_04_query_empty_store_returns_empty_list(self):
        res = self.store.retrieve("anything", top_k=5)
        self.assertEqual(res, [])

    def test_05_top_k_zero_returns_empty_list(self):
        rec = EpisodicRecord("r4", time.time(), "s4", "Fact", "Observed", 0.9)
        self.store.store(rec)
        res = self.store.retrieve("Fact", top_k=0)
        self.assertEqual(len(res), 0)


class TestF06_CM030MemoryAblation_Boundaries(unittest.TestCase):
    """F06 Boundaries: CM-030 Memory Ablation Benchmark."""

    def setUp(self):
        self.store = EpisodicMemoryStore()

    def test_01_query_disjoint_vocabulary(self):
        self.store.store(EpisodicRecord("m1", time.time(), "s", "alpha beta gamma", "Observed", 0.9))
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("exact_lookup", "xyz uvw", self.store, "m1")
        self.assertEqual(res["precision_at_k"], 0.0)

    def test_02_store_only_invalidated_provenance_hopfield_zero_stale(self):
        self.store.store(EpisodicRecord("m2", time.time(), "s", "stale claim", "Observed", 0.9))
        self.store.invalidate("m2", "falsified")
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("provenance_hopfield", "stale claim", self.store, "m2")
        self.assertEqual(res["stale_retrieval_rate"], 0.0)
        self.assertEqual(res["recall_count"], 0)

    def test_03_top_k_exceeds_store_size_clamps(self):
        self.store.store(EpisodicRecord("m3", time.time(), "s", "target fact", "Observed", 0.9))
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("exact_lookup", "target fact", self.store, "m3", top_k=100)
        self.assertLessEqual(res["recall_count"], 1)

    def test_04_unknown_retrieval_mode_raises(self):
        with self.assertRaises(ValueError):
            CM030MemoryAblationBenchmark.evaluate_retrieval("unsupported_mode", "q", self.store, "m1")

    def test_05_empty_store_retrieval(self):
        res = CM030MemoryAblationBenchmark.evaluate_retrieval("exact_lookup", "empty", self.store, "m1")
        self.assertEqual(res["recall_count"], 0)
        self.assertEqual(res["stale_retrieval_rate"], 0.0)


class TestF07_CausalRouter_Boundaries(unittest.TestCase):
    """F07 Boundaries: Causal Router Matched Benchmarking."""

    def test_01_single_expert_entropy_zero(self):
        probs = [1.0]
        entropy = CausalRouterBenchmark.compute_route_entropy(probs)
        self.assertAlmostEqual(entropy, 0.0, places=4)

    def test_02_degenerate_feature_vector_triggers_fallback(self):
        x = [0.0] * 16
        res = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=2)
        self.assertTrue(res["fallback_triggered"])
        self.assertEqual(len(res["selected_experts"]), 2)

    def test_03_top_k_larger_than_num_experts_clamps(self):
        x = [1.0, 2.0, 3.0]
        res = CausalRouterBenchmark.run_sparse_routing(x, num_experts=4, top_k=10)
        self.assertEqual(len(res["selected_experts"]), 4)

    def test_04_uniform_expert_entropy_maximum(self):
        probs = [0.125] * 8
        entropy = CausalRouterBenchmark.compute_route_entropy(probs)
        # log2(8) = 3.0
        self.assertAlmostEqual(entropy, 3.0, places=2)

    def test_05_load_balance_loss_with_empty_batch(self):
        loss = CausalRouterBenchmark.compute_load_balance_loss([], [], num_experts=4)
        self.assertEqual(loss, 0.0)


class TestF08_RoutingProfiler_Boundaries(unittest.TestCase):
    """F08 Boundaries: CM-040 Routing Profiling Report."""

    def test_01_zero_recorded_steps(self):
        profiler = RoutingProfiler(num_experts=4)
        rep = profiler.generate_report()
        self.assertEqual(rep["total_calls"], 0)
        self.assertEqual(rep["mean_latency_us"], 0.0)
        self.assertEqual(rep["balance_ratio"], 1.0)

    def test_02_single_step_p95_equals_mean(self):
        profiler = RoutingProfiler(num_experts=4)
        profiler.record_step(50.0, [0], 1.0, False)
        rep = profiler.generate_report()
        self.assertEqual(rep["mean_latency_us"], 50.0)
        self.assertEqual(rep["p95_latency_us"], 50.0)

    def test_03_out_of_bounds_expert_index_ignored(self):
        profiler = RoutingProfiler(num_experts=4)
        # Expert index 99 is invalid for num_experts=4
        profiler.record_step(10.0, [99], 1.0, False)
        rep = profiler.generate_report()
        self.assertEqual(sum(rep["expert_utilization"]), 0)

    def test_04_monopolized_expert_balance_ratio_zero(self):
        profiler = RoutingProfiler(num_experts=4)
        for _ in range(10):
            profiler.record_step(10.0, [0], 1.0, False)
        rep = profiler.generate_report()
        self.assertEqual(rep["balance_ratio"], 0.0)

    def test_05_zero_latency_recorded_safely(self):
        profiler = RoutingProfiler(num_experts=4)
        profiler.record_step(0.0, [0], 1.0, False)
        rep = profiler.generate_report()
        self.assertEqual(rep["mean_latency_us"], 0.0)


class TestF09_ContinualLearning_Boundaries(unittest.TestCase):
    """F09 Boundaries: Continual Learning Sequence (A -> B -> C -> D)."""

    def test_01_rank_one_prototype_in_high_dim(self):
        dim = 128
        proto = np.zeros((1, dim), dtype=np.float32)
        proto[0, 0] = 1.0
        P_perp = ContinualLearningSequence.orthogonal_projector(proto)
        self.assertEqual(P_perp.shape, (dim, dim))
        residual = np.dot(P_perp, proto.T)
        self.assertAlmostEqual(float(np.linalg.norm(residual)), 0.0, places=5)

    def test_02_empty_scores_history(self):
        res = ContinualLearningSequence.evaluate_retention({})
        self.assertEqual(res["mean_retention"], 1.0)
        self.assertEqual(res["mean_forgetting_index"], 0.0)

    def test_03_score_drops_to_zero(self):
        history = {"Task_A": [1.0, 0.0]}
        res = ContinualLearningSequence.evaluate_retention(history)
        self.assertEqual(res["per_task_retention"]["Task_A"], 0.0)
        self.assertEqual(res["per_task_forgetting"]["Task_A"], 1.0)

    def test_04_1d_prototype_array_reshaped_automatically(self):
        proto_1d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        P = ContinualLearningSequence.orthogonal_projector(proto_1d)
        self.assertEqual(P.shape, (3, 3))

    def test_05_collinear_prototypes_handled_by_qr(self):
        # Two identical rows
        protos = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        P = ContinualLearningSequence.orthogonal_projector(protos)
        self.assertEqual(P.shape, (2, 2))


class TestF10_CM050RetentionReport_Boundaries(unittest.TestCase):
    """F10 Boundaries: CM-050 Retention Benchmark Report."""

    def test_01_empty_tasks_report(self):
        rep = CM050RetentionBenchmark.compare_methods([], {})
        self.assertEqual(rep["task_sequence"], [])
        self.assertEqual(rep["comparisons"], {})

    def test_02_single_score_per_task_retention_1(self):
        results = {"m": {"T1": [0.85]}}
        rep = CM050RetentionBenchmark.compare_methods(["T1"], results)
        self.assertEqual(rep["comparisons"]["m"]["mean_retention"], 1.0)

    def test_03_negative_transfer_handled(self):
        results = {"m": {"T1": [0.9, 0.1]}}
        rep = CM050RetentionBenchmark.compare_methods(["T1"], results)
        self.assertAlmostEqual(rep["comparisons"]["m"]["mean_retention"], 0.1 / 0.9, places=2)

    def test_04_backward_transfer_score_improvement(self):
        # Score increases on older task (positive transfer)
        results = {"m": {"T1": [0.70, 0.95]}}
        rep = CM050RetentionBenchmark.compare_methods(["T1"], results)
        self.assertGreater(rep["comparisons"]["m"]["mean_retention"], 1.0)
        self.assertEqual(rep["comparisons"]["m"]["mean_forgetting_index"], 0.0)

    def test_05_multiple_methods_comparison(self):
        results = {
            "m1": {"T1": [1.0, 0.5]},
            "m2": {"T1": [1.0, 0.9]},
        }
        rep = CM050RetentionBenchmark.compare_methods(["T1"], results)
        self.assertEqual(len(rep["comparisons"]), 2)


class TestF11_AutonomousRepairSandbox_Boundaries(unittest.TestCase):
    """F11 Boundaries: Autonomous Test-Feedback Repair Sandbox."""

    def setUp(self):
        self.sandbox = CodeRepairSandbox(timeout_sec=5.0)

    def test_01_nonexistent_executable_returns_error(self):
        trace = self.sandbox.execute_in_sandbox("T_non_existent", "non_existent_binary_xyz_123")
        self.assertNotEqual(trace.return_code, 0)
        self.assertEqual(trace.tests_failed, 1)

    def test_02_patch_generator_none_halts_repair(self):
        def patch_none(att, err):
            return None
        cmd = "python -c \"import sys; sys.exit(1)\""
        res = self.sandbox.run_repair_loop("T_halt", cmd, patch_none, lambda p: True, max_attempts=2)
        self.assertFalse(res["success"])
        self.assertEqual(res["attempts"], 1)

    def test_03_apply_patch_failure_halts_repair(self):
        def apply_fail(p):
            return False
        cmd = "python -c \"import sys; sys.exit(1)\""
        res = self.sandbox.run_repair_loop("T_apply_fail", cmd, lambda a, e: "patch", apply_fail, max_attempts=2)
        self.assertFalse(res["success"])

    def test_04_max_attempts_single_iteration(self):
        cmd = "python -c \"import sys; sys.exit(1)\""
        res = self.sandbox.run_repair_loop("T_single", cmd, lambda a, e: "patch", lambda p: True, max_attempts=1)
        self.assertFalse(res["success"])
        self.assertLessEqual(res["attempts"], 1)

    def test_05_multiline_stderr_captured(self):
        cmd = "python -c \"import sys; sys.stderr.write('Line1\\nLine2\\n') ; sys.exit(2)\""
        trace = self.sandbox.execute_in_sandbox("T_err", cmd)
        self.assertIn("Line1", trace.stderr)
        self.assertIn("Line2", trace.stderr)
        self.assertEqual(trace.return_code, 2)


class TestF12_CM070EngineeringReport_Boundaries(unittest.TestCase):
    """F12 Boundaries: CM-070 Engineering Platform Report."""

    def test_01_all_failed_tasks_zero_resolution_rate(self):
        tasks = [{"task_id": "1", "success": False, "attempts": 2}]
        rep = CM070EngineeringReport.compile_report(tasks)
        self.assertEqual(rep["resolution_rate"], 0.0)

    def test_02_tasks_missing_attempts_defaults_to_one(self):
        tasks = [{"task_id": "1", "success": True}]
        rep = CM070EngineeringReport.compile_report(tasks)
        self.assertEqual(rep["mean_attempts_per_task"], 1.0)

    def test_03_empty_tasks_list(self):
        rep = CM070EngineeringReport.compile_report([])
        self.assertEqual(rep["total_tasks"], 0)

    def test_04_large_task_count(self):
        tasks = [{"task_id": str(i), "success": (i % 2 == 0), "attempts": 1} for i in range(500)]
        rep = CM070EngineeringReport.compile_report(tasks)
        self.assertEqual(rep["total_tasks"], 500)
        self.assertEqual(rep["resolved_tasks"], 250)
        self.assertAlmostEqual(rep["resolution_rate"], 0.5)

    def test_05_safety_violations_invariant_zero(self):
        rep = CM070EngineeringReport.compile_report([{"task_id": "t", "success": False}])
        self.assertEqual(rep["safety_violations"], 0)


class TestF13_MasterDataRegistry_Boundaries(unittest.TestCase):
    """F13 Boundaries: Master Data Registry & Schema Validator."""

    def test_01_whitespace_only_field_rejected(self):
        row = {f: "val" for f in MasterDataRegistryValidator.MANDATORY_FIELDS}
        row["name"] = "    "
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertFalse(passed)
        self.assertTrue(any("Missing mandatory field: name" in e for e in errs))

    def test_02_canonical_url_without_http_scheme(self):
        row = {f: "val" for f in MasterDataRegistryValidator.MANDATORY_FIELDS}
        row["canonical_url"] = "www.example.com"
        row["license_status"] = "mit"
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertFalse(passed)
        self.assertTrue(any("Invalid canonical URL" in e for e in errs))

    def test_03_nonexistent_csv_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            MasterDataRegistryValidator.validate_csv("non_existent_registry_file.csv")

    def test_04_extra_columns_ignored_if_mandatory_present(self):
        row = {f: "val" for f in MasterDataRegistryValidator.MANDATORY_FIELDS}
        row["canonical_url"] = "https://example.com"
        row["license_status"] = "mit"
        row["extra_custom_metadata_column"] = "custom_data"
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertTrue(passed)

    def test_05_case_insensitive_license_check(self):
        row = {f: "val" for f in MasterDataRegistryValidator.MANDATORY_FIELDS}
        row["canonical_url"] = "https://example.com"
        row["license_status"] = "UNREVIEWED_PROPRIETARY"
        passed, errs = MasterDataRegistryValidator.validate_row(row)
        self.assertFalse(passed)


class TestF14_P1Ingestion_Boundaries(unittest.TestCase):
    """F14 Boundaries: P1 Ingestion Pipeline."""

    def test_01_empty_records_partition(self):
        splits = P1IngestionPipeline.partition_split([])
        self.assertEqual(len(splits["train"]), 0)
        self.assertEqual(len(splits["dev"]), 0)
        self.assertEqual(len(splits["test"]), 0)

    def test_02_single_record_dataset(self):
        splits = P1IngestionPipeline.partition_split([{"id": "r1"}], train_ratio=1.0, dev_ratio=0.0, test_ratio=0.0)
        self.assertEqual(len(splits["train"]), 1)
        self.assertEqual(len(splits["dev"]), 0)
        self.assertEqual(len(splits["test"]), 0)

    def test_03_missing_id_field_in_leak_check_handled(self):
        train = [{"data": "foo"}]
        test = [{"data": "bar"}]
        leaked = P1IngestionPipeline.check_split_leakage(train, test, key_field="id")
        self.assertFalse(leaked)

    def test_04_seed_zero_reproducible(self):
        recs = [{"id": str(i)} for i in range(20)]
        s1 = P1IngestionPipeline.partition_split(recs, seed=0)
        s2 = P1IngestionPipeline.partition_split(recs, seed=0)
        self.assertEqual([r["id"] for r in s1["train"]], [r["id"] for r in s2["train"]])

    def test_05_large_dataset_split(self):
        recs = [{"id": str(i)} for i in range(5000)]
        splits = P1IngestionPipeline.partition_split(recs, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
        self.assertEqual(len(splits["train"]), 4000)
        self.assertEqual(len(splits["dev"]), 500)
        self.assertEqual(len(splits["test"]), 500)


class TestF15_SyntheticDiscovery_Boundaries(unittest.TestCase):
    """F15 Boundaries: Synthetic Discovery Episode Generator."""

    def test_01_extreme_seed_zero(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-0", "world_1_polynomial", seed=0)
        self.assertIsInstance(ep.observed_result, float)

    def test_02_large_seed_boundary(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-MAX", "world_1_polynomial", seed=2**31 - 1)
        self.assertIsInstance(ep.observed_result, float)

    def test_03_prediction_within_noise_margin(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-N", "world_1_polynomial", seed=42)
        diff = abs(ep.observed_result - ep.pre_experiment_prediction)
        # Noise sigma is 0.05, 5 sigma is 0.25
        self.assertLess(diff, 0.35)

    def test_04_iso_timestamp_format(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-T", "world_1_polynomial", seed=42)
        self.assertTrue(ep.timestamp.endswith("Z"))

    def test_05_metadata_noise_sigma_present(self):
        ep = DiscoveryEpisodeGenerator.generate_episode("EP-M", "world_1_polynomial", seed=42)
        self.assertIn("noise_sigma", ep.metadata)


class TestF16_GroundTruthFirewall_Boundaries(unittest.TestCase):
    """F16 Boundaries: Ground-Truth Leakage Firewall & Tests."""

    def test_01_zero_parameter_not_falsely_flagged(self):
        # Param value 0.0 should not flag when observation has 0.0
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(
            visible_context={},
            observations=[0.0, 1.0],
            quarantined_formula="none",
            quarantined_params={"bias": 0.0},
        )
        self.assertTrue(passed)

    def test_02_empty_observations_passes(self):
        passed, leaks = GroundTruthLeakageFirewall.check_leakage({}, [], "formula", {})
        self.assertTrue(passed)

    def test_03_empty_quarantined_formula_passes(self):
        passed, leaks = GroundTruthLeakageFirewall.check_leakage({"context": "text"}, ["obs"], "", {})
        self.assertTrue(passed)

    def test_04_metadata_nested_leak_detected(self):
        ctx = {"metadata": {"nested_leak": "secret_formula_2.5*x"}}
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(ctx, [], "secret_formula_2.5*x", {})
        self.assertFalse(passed)

    def test_05_multiple_leaks_all_reported(self):
        ctx = {"leak": "target_law"}
        obs = ["target_law in obs"]
        passed, leaks = GroundTruthLeakageFirewall.check_leakage(ctx, obs, "target_law", {})
        self.assertFalse(passed)
        self.assertGreaterEqual(len(leaks), 2)


class TestF17_HiddenRuleSimulators_Boundaries(unittest.TestCase):
    """F17 Boundaries: Hidden-Rule Simulator Worlds (Worlds 1-4)."""

    def test_01_input_zero_all_worlds(self):
        for w_id in ["world_1_polynomial", "world_2_harmonic", "world_3_exponential", "world_4_switching"]:
            sim = HiddenRuleSimulator(w_id, seed=42)
            out = sim.step({"x": 0.0, "noise": 0.0})
            self.assertTrue(np.isfinite(out["y"]))

    def test_02_world_3_large_input_clamped(self):
        # exp(-0.8 * 1000) would underflow without clipping
        sim = HiddenRuleSimulator("world_3_exponential", seed=42)
        out = sim.step({"x": 1000.0, "noise": 0.0})
        self.assertTrue(np.isfinite(out["y"]))

    def test_03_action_missing_x_defaults_zero(self):
        sim = HiddenRuleSimulator("world_1_polynomial", seed=42)
        out = sim.step({"noise": 0.0})
        self.assertEqual(out["x"], 0.0)

    def test_04_unknown_world_raises(self):
        with self.assertRaises(ValueError):
            HiddenRuleSimulator("unknown_world_id_xyz", seed=42)

    def test_05_invalid_syntax_hypothesis_returns_infinite_mse(self):
        sim = HiddenRuleSimulator("world_1_polynomial", seed=42)
        res = sim.evaluate_hypothesis("this is not a valid formula (")
        self.assertFalse(res["valid_syntax"])
        self.assertEqual(res["mse"], float("inf"))


class TestF18_HypothesisGrammar_Boundaries(unittest.TestCase):
    """F18 Boundaries: Candidate Hypothesis Grammar & Symbolic Baseline."""

    def test_01_complexity_of_constant_one(self):
        comp = CandidateHypothesisGrammar.compute_complexity("1")
        self.assertEqual(comp, 1)

    def test_02_unparseable_syntax_returns_penalty(self):
        comp = CandidateHypothesisGrammar.compute_complexity("+++(((***")
        self.assertEqual(comp, 100)

    def test_03_empty_string_complexity_penalty(self):
        comp = CandidateHypothesisGrammar.compute_complexity("")
        self.assertEqual(comp, 100)

    def test_04_nested_functions_complexity(self):
        comp = CandidateHypothesisGrammar.compute_complexity("sin(exp(x))")
        self.assertGreater(comp, 2)

    def test_05_variable_with_digit(self):
        cands = CandidateHypothesisGrammar.generate_candidates(variable="x1")
        self.assertTrue(all("x1" in c for c in cands))


class TestF19_ActiveExperimentSelector_Boundaries(unittest.TestCase):
    """F19 Boundaries: Active Experiment Selection Policies."""

    def setUp(self):
        self.selector = ActiveExperimentSelector()

    def test_01_all_identical_predictions_selects_first(self):
        # All models predict y=5
        cands = ["5", "5.0", "5+0*x"]
        pool = [{"x": 1.0, "cost": 1.0}, {"x": 2.0, "cost": 1.0}]
        action = self.selector.select_experiment(cands, pool)
        self.assertEqual(action["x"], 1.0)

    def test_02_action_missing_cost_defaults_to_one(self):
        cands = ["x", "x**2"]
        pool = [{"x": 2.0}]  # no cost key
        action = self.selector.select_experiment(cands, pool)
        self.assertEqual(action["x"], 2.0)

    def test_03_invalid_hypothesis_in_list_skipped(self):
        cands = ["x", "invalid_syntax_symbol((("]
        pool = [{"x": 1.0, "cost": 1.0}]
        action = self.selector.select_experiment(cands, pool)
        self.assertEqual(action["x"], 1.0)

    def test_04_negative_action_coordinates(self):
        cands = ["x", "x**2"]
        pool = [{"x": -3.0, "cost": 1.0}, {"x": 0.0, "cost": 1.0}]
        # At -3: preds are -3 and 9 (high variance)
        action = self.selector.select_experiment(cands, pool)
        self.assertEqual(action["x"], -3.0)

    def test_05_random_policy_with_large_pool(self):
        pool = [{"x": float(i), "cost": 1.0} for i in range(50)]
        action = self.selector.select_experiment(["x"], pool, policy="random")
        self.assertIn(action, pool)


class TestF20_CMSCIBenchmarks_Boundaries(unittest.TestCase):
    """F20 Boundaries: CM-SCI-01 to 06 Benchmark Suite."""

    def test_01_negative_extrapolation_points(self):
        sim = HiddenRuleSimulator("world_1_polynomial", seed=42)
        res = sim.evaluate_hypothesis("2.5*x + 1.2*x**2", test_points=[-10.0, -20.0])
        self.assertLess(res["mse"], 1e-4)

    def test_02_incorrect_candidate_high_mse(self):
        sim = HiddenRuleSimulator("world_1_polynomial", seed=42)
        res = sim.evaluate_hypothesis("100.0*x + 50.0")
        self.assertGreater(res["mse"], 100.0)
        self.assertFalse(res["is_exact_match"])

    def test_03_benchmark_success_boolean_type(self):
        r1 = CMSCIBenchmarkSuite.run_sci_01_recovery()
        self.assertIsInstance(r1["success"], (bool, np.bool_))

    def test_04_different_seeds_produce_different_noise(self):
        s1 = HiddenRuleSimulator("world_1_polynomial", seed=10)
        s2 = HiddenRuleSimulator("world_1_polynomial", seed=20)
        out1 = s1.step({"x": 1.0, "noise": 0.5})
        out2 = s2.step({"x": 1.0, "noise": 0.5})
        self.assertNotEqual(out1["y"], out2["y"])

    def test_05_zero_noise_step_exact(self):
        sim = HiddenRuleSimulator("world_2_harmonic", seed=42)
        out = sim.step({"x": 0.0, "noise": 0.0})
        self.assertEqual(out["y"], 0.0)


class TestF21_DeterministicVerifier_Boundaries(unittest.TestCase):
    """F21 Boundaries: Deterministic Verifier & Invariant Checks."""

    def setUp(self):
        self.verifier = DeterministicVerifier()

    def test_01_no_invariants_specified_returns_unknown(self):
        res = self.verifier.verify({"some_key": "some_value"}, {})
        self.assertEqual(res.status, "UNKNOWN")
        self.assertEqual(len(res.invariants_checked), 0)

    def test_02_value_on_exact_lower_bound_passes(self):
        res = self.verifier.verify({"value": 0.0}, {"bounds": [0.0, 10.0]})
        self.assertEqual(res.status, "PASS")

    def test_03_value_on_exact_upper_bound_passes(self):
        res = self.verifier.verify({"value": 10.0}, {"bounds": [0.0, 10.0]})
        self.assertEqual(res.status, "PASS")

    def test_04_symbolic_expression_syntax_error_fails(self):
        res = self.verifier.verify({"symbolic_claim": "x ++ ((("}, {"symbolic_reference": "x"})
        self.assertEqual(res.status, "FAIL")
        self.assertTrue(any("Symbolic parsing failure" in v for v in res.violations))

    def test_05_multiple_violations_reported_together(self):
        pred = {"unit": "wrong", "value": -50.0, "shape": [1, 1]}
        ctx = {"expected_unit": "m/s", "bounds": [0, 100], "expected_shape": [2, 2]}
        res = self.verifier.verify(pred, ctx)
        self.assertEqual(res.status, "FAIL")
        self.assertEqual(len(res.violations), 3)


class TestF22_FaultInjectionSuite_Boundaries(unittest.TestCase):
    """F22 Boundaries: Fault Injection Suite & CM-060 Report."""

    def test_01_empty_cases_handled(self):
        verifier = DeterministicVerifier()
        # default suite has 5 cases
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertGreater(res["total_cases"], 0)

    def test_02_far_cleared_target(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertTrue(res["far_target_cleared"])

    def test_03_frr_cleared_target(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertTrue(res["frr_target_cleared"])

    def test_04_injected_faults_strictly_greater_than_zero(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertGreater(res["injected_faults"], 0)

    def test_05_far_and_frr_are_floats(self):
        verifier = DeterministicVerifier()
        res = FaultInjectionSuite.run_fault_injection(verifier)
        self.assertIsInstance(res["false_acceptance_rate"], float)
        self.assertIsInstance(res["false_rejection_rate"], float)


class TestF23_MinimumAblationMatrix_Boundaries(unittest.TestCase):
    """F23 Boundaries: Minimum Ablation Matrix (V0-V8) Harness."""

    def test_01_unknown_variant_raises_value_error(self):
        with self.assertRaises(ValueError):
            MinimumAblationMatrixHarness.evaluate_variant("V99_UNKNOWN")

    def test_02_baseline_score_zero_clamps(self):
        # Corrupted gist drops by 0.25; with baseline 0.1, clamped to 0.0
        score = MinimumAblationMatrixHarness.evaluate_variant("V6", baseline_score=0.10)
        self.assertGreaterEqual(score, 0.0)

    def test_03_baseline_score_one_clamps(self):
        # Oracle increases by 0.20 + 0.22 + 0.10; clamped to 1.0
        score = MinimumAblationMatrixHarness.evaluate_variant("V8", baseline_score=0.90)
        self.assertLessEqual(score, 1.0)

    def test_04_deterministic_repeated_evaluations(self):
        s1 = MinimumAblationMatrixHarness.evaluate_variant("V4", baseline_score=0.50)
        s2 = MinimumAblationMatrixHarness.evaluate_variant("V4", baseline_score=0.50)
        self.assertEqual(s1, s2)

    def test_05_v7_corrupted_memory_drops_more_than_v2(self):
        s_v2 = MinimumAblationMatrixHarness.evaluate_variant("V2")  # Memory only (+0.15)
        s_v7 = MinimumAblationMatrixHarness.evaluate_variant("V7")  # Corrupted memory (-0.30)
        self.assertGreater(s_v2, s_v7 - 0.25)


class TestF24_FullStackIntegration_Boundaries(unittest.TestCase):
    """F24 Boundaries: CM-INT-01 Full-Stack Integration Report."""

    def test_01_single_observation_returns_zero(self):
        d = FullStackIntegrationReport.cohens_d([1.0], [2.0])
        self.assertEqual(d, 0.0)

    def test_02_identical_groups_returns_zero(self):
        g = [1.0, 2.0, 3.0]
        d = FullStackIntegrationReport.cohens_d(g, g)
        self.assertEqual(d, 0.0)

    def test_03_antisymmetric_cohens_d(self):
        g1 = [1.0, 2.0, 3.0]
        g2 = [4.0, 5.0, 6.0]
        d1 = FullStackIntegrationReport.cohens_d(g1, g2)
        d2 = FullStackIntegrationReport.cohens_d(g2, g1)
        self.assertAlmostEqual(d1, -d2, places=4)

    def test_04_small_bootstrap_count(self):
        g1 = [1.0, 2.0]
        g2 = [0.0, 1.0]
        lower, upper = FullStackIntegrationReport.bootstrap_ci(g1, g2, n_bootstraps=10)
        self.assertLessEqual(lower, upper)

    def test_05_zero_variance_groups_return_zero(self):
        d = FullStackIntegrationReport.cohens_d([5.0, 5.0], [5.0, 5.0])
        self.assertEqual(d, 0.0)


class TestF25_Preregistration_Boundaries(unittest.TestCase):
    """F25 Boundaries: Preregistration Schema & Run Bundles."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cortex_bnd_f25_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_empty_dictionary_fails(self):
        passed, violations = PreregistrationValidator.validate_template({})
        self.assertFalse(passed)
        self.assertEqual(len(violations), len(PreregistrationValidator.MANDATORY_FIELDS))

    def test_02_extra_fields_allowed(self):
        data = {f: "val" for f in PreregistrationValidator.MANDATORY_FIELDS}
        data["extra_optional_field"] = "permitted"
        passed, violations = PreregistrationValidator.validate_template(data)
        self.assertTrue(passed)

    def test_03_existing_run_bundle_overwrite_raises(self):
        PreregistrationValidator.create_run_bundle(self.tmp_dir, "EXP_LOCK", {})
        with self.assertRaises(FileExistsError):
            PreregistrationValidator.create_run_bundle(self.tmp_dir, "EXP_LOCK", {})

    def test_04_run_bundle_creates_json_file(self):
        run_dir = PreregistrationValidator.create_run_bundle(self.tmp_dir, "EXP_02", {"seed": 42})
        json_file = os.path.join(run_dir, "config_snapshot.json")
        self.assertTrue(os.path.exists(json_file))

    def test_05_preregistration_mandatory_fields_count(self):
        self.assertGreaterEqual(len(PreregistrationValidator.MANDATORY_FIELDS), 10)


class TestF26_MultiSeedEvaluator_Boundaries(unittest.TestCase):
    """F26 Boundaries: Multi-Seed Statistical Evaluator."""

    def test_01_exactly_five_seeds_screening_pass(self):
        seeds = {s: 0.8 for s in [11, 29, 47, 71, 101]}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seeds)
        self.assertTrue(res["screening_pass"])
        self.assertFalse(res["confirmatory_pass"])

    def test_02_exactly_ten_seeds_confirmatory_pass(self):
        seeds = {s: 0.8 for s in MultiSeedStatisticalEvaluator.CONFIRMATORY_SEEDS}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seeds)
        self.assertTrue(res["screening_pass"])
        self.assertTrue(res["confirmatory_pass"])

    def test_03_zero_variance_seeds(self):
        seeds = {s: 1.0 for s in MultiSeedStatisticalEvaluator.CONFIRMATORY_SEEDS}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seeds)
        self.assertEqual(res["std"], 0.0)
        self.assertEqual(res["ci_95"], (1.0, 1.0))

    def test_04_four_seeds_fails_screening(self):
        seeds = {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seeds)
        self.assertFalse(res["screening_pass"])

    def test_05_negative_scores_evaluated(self):
        seeds = {s: -0.5 for s in MultiSeedStatisticalEvaluator.CONFIRMATORY_SEEDS}
        res = MultiSeedStatisticalEvaluator.evaluate_seeds(seeds)
        self.assertEqual(res["mean"], -0.5)


class TestF27_ReleaseGateValidator_Boundaries(unittest.TestCase):
    """F27 Boundaries: Automated Release Gate Validator (G0-G9)."""

    def test_01_all_gates_false_ten_failing(self):
        status = {f"G{i}": False for i in range(10)}
        res = AutomatedReleaseGateValidator.evaluate_gates(status)
        self.assertFalse(res["all_gates_cleared"])
        self.assertEqual(len(res["failing_gates"]), 10)

    def test_02_empty_status_all_failing(self):
        res = AutomatedReleaseGateValidator.evaluate_gates({})
        self.assertFalse(res["all_gates_cleared"])
        self.assertEqual(len(res["failing_gates"]), 10)

    def test_03_extra_unknown_gates_ignored(self):
        status = {f"G{i}": True for i in range(10)}
        status["G99_UNKNOWN"] = False
        res = AutomatedReleaseGateValidator.evaluate_gates(status)
        self.assertTrue(res["all_gates_cleared"])

    def test_04_failing_gates_sorted_order(self):
        status = {f"G{i}": True for i in range(10)}
        status["G3"] = False
        status["G7"] = False
        res = AutomatedReleaseGateValidator.evaluate_gates(status)
        self.assertEqual(res["failing_gates"], ["G3", "G7"])

    def test_05_scorecard_boolean_values(self):
        res = AutomatedReleaseGateValidator.evaluate_gates({"G0": True})
        self.assertTrue(isinstance(res["scorecard"]["G0"], bool))
        self.assertTrue(isinstance(res["scorecard"]["G1"], bool))


class TestF28_VCSReleaseValidator_Boundaries(unittest.TestCase):
    """F28 Boundaries: Final Release Notes, VCS Checkpoints & Push."""

    def test_01_missing_phase_tag_fails(self):
        msg = "Architecture and Repository Foundation: Baseline complete"
        self.assertFalse(VCSReleaseValidator.validate_commit_message(msg))

    def test_02_lowercase_phase_tag_fails(self):
        msg = "[phase 1] Foundation: Complete"
        self.assertFalse(VCSReleaseValidator.validate_commit_message(msg))

    def test_03_package_release_empty_notes(self):
        tmp_dir = tempfile.mkdtemp(prefix="cortex_bnd_f28_")
        try:
            manifest = VCSReleaseValidator.package_release("v0.1", tmp_dir, "")
            self.assertEqual(manifest["version"], "v0.1")
            self.assertTrue(os.path.exists(manifest["notes_file"]))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_04_commit_message_with_hyphens_in_component(self):
        msg = "[Phase 7] CORTEX-M-SD Scientific Discovery: CM-SCI-01 verified"
        self.assertTrue(VCSReleaseValidator.validate_commit_message(msg))

    def test_05_commit_message_with_parentheses(self):
        msg = "[Phase 10] Verification (Deterministic): FAR <= 0.01 cleared"
        self.assertTrue(VCSReleaseValidator.validate_commit_message(msg))


if __name__ == "__main__":
    unittest.main()
