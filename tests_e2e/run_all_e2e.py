"""CORTEX-M Standalone E2E Test Suite Runner.

Executes all 4 tiers of the comprehensive End-to-End test suite:
- Tier 1: Feature Coverage (140 tests: 28 features x 5 test cases)
- Tier 2: Boundary & Corner Cases (140 tests: 28 features x 5 test cases)
- Tier 3: Cross-Feature Combinations (10 tests: pairwise interactions)
- Tier 4: Real-World Application Scenarios (5 tests: operational loops)

Total: 295 requirement-driven test cases covering Features F01 to F28.
Compatible with direct execution, unittest discovery, and pytest.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import unittest

# Ensure repo root and tests_e2e are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = Path(__file__).resolve().parent


def run_e2e_suite() -> int:
    """Discovers and executes all E2E test suites with formatted summary."""
    print("=" * 80)
    print(" CORTEX-M COMPREHENSIVE END-TO-END (E2E) TEST SUITE")
    print(" Methodology: 4-Tier Opaque-Box Requirement-Driven Verification")
    print(" Covering: Phases 1-13 (Features F01 through F28)")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tier_files = [
        ("Tier 1: Feature Coverage (Happy Paths)", "test_tier1_features.py"),
        ("Tier 2: Boundary & Corner Cases", "test_tier2_boundaries.py"),
        ("Tier 3: Cross-Feature Combinations", "test_tier3_combinations.py"),
        ("Tier 4: Real-World Application Scenarios", "test_tier4_scenarios.py"),
    ]

    tier_suites = {}
    for tier_name, filename in tier_files:
        filepath = TESTS_DIR / filename
        if filepath.exists():
            s = loader.discover(start_dir=str(TESTS_DIR), pattern=filename)
            tier_suites[tier_name] = s
            suite.addTests(s)
        else:
            print(f"[WARN] Test file missing: {filepath}")

    print(f"\n[INFO] Discovered {suite.countTestCases()} total test cases across 4 tiers:")
    for tier_name, s in tier_suites.items():
        print(f"  - {tier_name}: {s.countTestCases()} test cases")
    print("-" * 80)

    runner = unittest.TextTestRunner(verbosity=1)
    start_time = time.perf_counter()
    result = runner.run(suite)
    duration = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print(" TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f" Total Tests Executed: {result.testsRun}")
    print(f" Passed:               {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f" Failures:             {len(result.failures)}")
    print(f" Errors:               {len(result.errors)}")
    print(f" Execution Duration:   {duration:.3f} seconds")
    print("=" * 80)

    if result.wasSuccessful():
        print("[SUCCESS] ALL E2E VERIFICATION TIERS PASSED CLEANLY (EXIT CODE 0)\n")
        return 0
    else:
        print("[FAILURE] E2E VERIFICATION REPORTED DEFECTS (EXIT CODE 1)\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_e2e_suite())
