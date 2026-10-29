# CORTEX-M E2E Test Infrastructure Specification

**Document:** `TEST_INFRA.md`  
**Governing Standard:** CORTEX-M Master Execution Guide (§32-34) & PRD/SRS Baseline  
**Version:** 1.0.0  
**Scope:** Phases 1 through 13 (Features F01 through F28)  
**Execution Environment:** `e:\Research\.venv-mimo\Scripts\python.exe` (Python 3.12, PyTorch CPU, NumPy, SymPy)

---

## 1. Executive Summary & Architecture

The CORTEX-M End-to-End (E2E) Test Infrastructure provides an independent, opaque-box, requirement-driven verification harness for the complete CORTEX-M cognitive and scientific research substrate. It verifies all 28 features (F01 to F28) defined in `PROJECT.md` across the 13 research phases.

### 4-Tier Verification Architecture

```text
tests_e2e/
├── reference_contracts.py      # Authoritative contracts, oracles & dynamic dispatchers
├── test_tier1_features.py      # Tier 1: Feature Coverage (>=5 happy path tests per feature, 140 tests)
├── test_tier2_boundaries.py    # Tier 2: Boundaries & Corners (>=5 boundary/edge tests per feature, 140 tests)
├── test_tier3_combinations.py  # Tier 3: Cross-Feature Combinations (10 pairwise integration tests)
├── test_tier4_scenarios.py     # Tier 4: Real-World Application Scenarios (5 operational workflows)
└── run_all_e2e.py              # Standalone test runner with formatted reporting
```

### Core Testing Principles

1. **Opaque-Box Requirement-Driven Testing:** Tests interface solely with public contracts, request/response payloads, and observable system outputs. No internal private variables or undocumented mechanics are accessed.
2. **Progressive Testability & Dynamic Dispatch:** Tests execute against live `cortex_m` implementations when present, while utilizing mathematical reference contracts (`reference_contracts.py`) conforming to `PROJECT.md` and `CORTEX_M_Master_Execution_Guide.md` to ensure progressive testability before and during milestone delivery.
3. **Strict Self-Containment & Zero Side-Effects:** Every test sets up isolated temporary directories (`tempfile.mkdtemp`), initializes deterministic random seeds, and cleans up all temporary state upon teardown.
4. **Zero Web Browser Usage:** All validations operate locally and deterministically via Python, subprocess isolation, and mathematical solvers without any browser automation.

---

## 2. Test Suite Inventory & Tier Breakdown

| Tier | Focus | Test File | Test Count | Description |
|---|---|---|---|---|
| **Tier 1** | **Feature Coverage** | `test_tier1_features.py` | **140** | Representative happy paths across all 28 features (5 tests per feature). |
| **Tier 2** | **Boundary & Corner Cases** | `test_tier2_boundaries.py` | **140** | Malformed inputs, empty cases, extreme sizes, timeouts, and error handling (5 tests per feature). |
| **Tier 3** | **Cross-Feature Combinations** | `test_tier3_combinations.py` | **10** | Pairwise and systemic interactions (Gist+Memory, Router+Gist, Continual+Memory, etc.). |
| **Tier 4** | **Real-World Scenarios** | `test_tier4_scenarios.py` | **5** | Closed-loop rule discovery, repository repair, multi-task retention, leakage defense, release gate audit. |
| **Total** | **Full E2E Suite** | **All 4 Tiers** | **295** | Complete requirement-driven verification covering Phases 1–13. |

---

## 3. Feature Traceability Matrix (F01 – F28)

Every feature in `PROJECT.md` is explicitly covered by dedicated test suites in Tier 1 and Tier 2:

| Feature ID | Feature Name | Milestone | Tier 1 Test Class | Tier 2 Test Class | Verified Contracts |
|---|---|---|---|---|---|
| **F01** | Hardware & OS System Info | M1 | `TestF01_HardwareOSInfo` | `TestF01_HardwareOSInfo_Boundaries` | `HardwareOSInfoCollector` |
| **F02** | Pytest Environment Enabler | M1 | `TestF02_PytestEnvironment` | `TestF02_PytestEnvironment_Boundaries` | `PytestEnvironmentEnabler` |
| **F03** | Structured Gist Dataclass & Validator | M1 | `TestF03_StructuredGist` | `TestF03_StructuredGist_Boundaries` | `StructuredGist` (12 fields) |
| **F04** | CM-020 Gist Ablation Benchmark | M1 | `TestF04_CM020GistAblation` | `TestF04_CM020GistAblation_Boundaries` | `CM020GistAblationHarness` |
| **F05** | Episodic Memory Store & Invalidation | M1 | `TestF05_EpisodicMemoryStore` | `TestF05_EpisodicMemoryStore_Boundaries` | `EpisodicMemoryStore`, `EpisodicRecord` |
| **F06** | CM-030 Memory Ablation Benchmark | M1 | `TestF06_CM030MemoryAblation` | `TestF06_CM030MemoryAblation_Boundaries` | `CM030MemoryAblationBenchmark` |
| **F07** | Causal Router Matched Benchmarking | M2 | `TestF07_CausalRouterBenchmark` | `TestF07_CausalRouter_Boundaries` | `CausalRouterBenchmark` |
| **F08** | CM-040 Routing Profiling Report | M2 | `TestF08_RoutingProfiler` | `TestF08_RoutingProfiler_Boundaries` | `RoutingProfiler` |
| **F09** | Sequential Continual Learning Sequence | M2 | `TestF09_ContinualLearningSequence` | `TestF09_ContinualLearning_Boundaries` | `ContinualLearningSequence` ($P_\perp$) |
| **F10** | CM-050 Retention Benchmark Report | M2 | `TestF10_CM050RetentionReport` | `TestF10_CM050RetentionReport_Boundaries` | `CM050RetentionBenchmark` |
| **F11** | Autonomous Test-Feedback Repair Sandbox | M3 | `TestF11_AutonomousRepairSandbox` | `TestF11_AutonomousRepairSandbox_Boundaries` | `CodeRepairSandbox` (5.0s timeout) |
| **F12** | CM-070 Engineering Platform Report | M3 | `TestF12_CM070EngineeringReport` | `TestF12_CM070EngineeringReport_Boundaries` | `CM070EngineeringReport` |
| **F13** | Master Data Registry & Schema Validator | M4 | `TestF13_MasterDataRegistryValidator` | `TestF13_MasterDataRegistry_Boundaries` | `MasterDataRegistryValidator` (12 fields) |
| **F14** | P1 Ingestion Pipeline | M4 | `TestF14_P1IngestionPipeline` | `TestF14_P1Ingestion_Boundaries` | `P1IngestionPipeline` (zero leak) |
| **F15** | Synthetic Discovery Episode Generator | M4 | `TestF15_SyntheticDiscoveryEpisode` | `TestF15_SyntheticDiscovery_Boundaries` | `DiscoveryEpisodeGenerator` (13 fields) |
| **F16** | Ground-Truth Leakage Firewall & Tests | M4 | `TestF16_GroundTruthLeakageFirewall` | `TestF16_GroundTruthFirewall_Boundaries` | `GroundTruthLeakageFirewall` (5-point) |
| **F17** | Hidden-Rule Simulator Worlds (Worlds 1-4) | M5 | `TestF17_HiddenRuleSimulators` | `TestF17_HiddenRuleSimulators_Boundaries` | `HiddenRuleSimulator` |
| **F18** | Candidate Hypothesis Grammar & Symbolic Baseline | M5 | `TestF18_HypothesisGrammar` | `TestF18_HypothesisGrammar_Boundaries` | `CandidateHypothesisGrammar` |
| **F19** | Active Experiment Selection Policies | M5 | `TestF19_ActiveExperimentSelector` | `TestF19_ActiveExperimentSelector_Boundaries` | `ActiveExperimentSelector` |
| **F20** | CM-SCI-01 to 06 Benchmark Suite | M5 | `TestF20_CMSCIBenchmarks` | `TestF20_CMSCIBenchmarks_Boundaries` | `CMSCIBenchmarkSuite` |
| **F21** | Deterministic Verifier & Invariant Checks | M5 | `TestF21_DeterministicVerifier` | `TestF21_DeterministicVerifier_Boundaries` | `DeterministicVerifier` |
| **F22** | Fault Injection Suite & CM-060 Report | M5 | `TestF22_FaultInjectionSuite` | `TestF22_FaultInjectionSuite_Boundaries` | `FaultInjectionSuite` (FAR $\le 1\%$, FRR $\le 5\%$) |
| **F23** | Minimum Ablation Matrix (V0-V8) Harness | M6 | `TestF23_MinimumAblationMatrix` | `TestF23_MinimumAblationMatrix_Boundaries` | `MinimumAblationMatrixHarness` |
| **F24** | CM-INT-01 Full-Stack Integration Report | M6 | `TestF24_FullStackIntegrationReport` | `TestF24_FullStackIntegration_Boundaries` | `FullStackIntegrationReport` (Cohen's $d$, CI) |
| **F25** | Preregistration Schema & Run Bundles | M6 | `TestF25_PreregistrationValidator` | `TestF25_Preregistration_Boundaries` | `PreregistrationValidator` (28 fields) |
| **F26** | Multi-Seed Statistical Evaluator | M6 | `TestF26_MultiSeedEvaluator` | `TestF26_MultiSeedEvaluator_Boundaries` | `MultiSeedStatisticalEvaluator` (10 seeds) |
| **F27** | Automated Release Gate Validator (G0-G9) | M6 | `TestF27_ReleaseGateValidator` | `TestF27_ReleaseGateValidator_Boundaries` | `AutomatedReleaseGateValidator` |
| **F28** | Final Release Notes, VCS Checkpoints & Push | M6 | `TestF28_VCSReleaseValidator` | `TestF28_VCSReleaseValidator_Boundaries` | `VCSReleaseValidator` |

---

## 4. Execution Guide & Commands

### Standalone Runner (Recommended)

Execute the complete 295-test suite with formatted summary:

```powershell
& "e:\Research\.venv-mimo\Scripts\python.exe" tests_e2e/run_all_e2e.py
```

### Python Unittest Discovery

Run via Python's built-in `unittest` runner:

```powershell
& "e:\Research\.venv-mimo\Scripts\python.exe" -m unittest discover -s tests_e2e -v
```

### Running Individual Tiers

```powershell
# Tier 1 (Feature Coverage)
& "e:\Research\.venv-mimo\Scripts\python.exe" -m unittest discover -s tests_e2e -p "test_tier1_features.py"

# Tier 2 (Boundaries & Edge Cases)
& "e:\Research\.venv-mimo\Scripts\python.exe" -m unittest discover -s tests_e2e -p "test_tier2_boundaries.py"

# Tier 3 (Cross-Feature Combinations)
& "e:\Research\.venv-mimo\Scripts\python.exe" -m unittest discover -s tests_e2e -p "test_tier3_combinations.py"

# Tier 4 (Real-World Application Scenarios)
& "e:\Research\.venv-mimo\Scripts\python.exe" -m unittest discover -s tests_e2e -p "test_tier4_scenarios.py"
```

---

## 5. Verification Results Summary

- **Total Test Cases:** 295
- **Pass Rate:** 100% (295 / 295 passed)
- **Failures / Errors:** 0
- **Regression on Existing Suites:** Zero regressions (65/65 passed in `tests/`, 80/80 passed in `tests_mimo/`)
- **Total Execution Duration:** ~6.6 seconds on workstation CPU
