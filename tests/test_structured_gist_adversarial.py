"""Adversarial and Empirical Stress Test Suite for CORTEX-M Structured Gist Subsystem.

Probes edge cases, boundary conditions, malformed structures, mutation attacks,
timestamp monotonicity, and serialization/unicode fuzzing against StructuredGist.
"""

import copy
from datetime import datetime, timezone
import gc
import json
import math
import threading
import unittest
from typing import Any, Dict

from cortex_m.gist.structured_gist import StructuredGist


class TestStructuredGistAdversarial(unittest.TestCase):
    """Adversarial stress harness for StructuredGist."""

    def setUp(self):
        self.base_valid_data: Dict[str, Any] = {
            "objective": "Determine energy conservation in non-conservative fluid damping.",
            "core_claim": "Viscous dissipation rate matches Rayleigh dissipation function R = 1/2 * b * v^2.",
            "evidence": ["PIV velocity field trace V-892", "Thermocouple delta-T = 0.042 K"],
            "assumptions": ["Navier-Stokes continuum approximation holds", "Constant heat capacity C_v"],
            "constraints": ["Entropy production dS/dt >= 0", "Viscosity mu in [1e-4, 1e2] Pa*s"],
            "predictions": ["Total internal energy rise exactly equals mechanical work dissipated."],
            "uncertainty": 0.05,
            "next_action": "Measure temperature rise in adiabatic calorimeter with precision thermistor.",
            "contradictions": [],
            "open_questions": ["Is shear-thinning noticeable at acoustic boundary layers?"],
            "provenance_and_time": {
                "created_at": "2026-10-29T00:00:00+00:00",
                "updated_at": "2026-10-29T00:00:00+00:00",
                "source": "cortex_sd_adversarial_suite",
                "version": "1.0.0",
            },
            "revision_history": [],
        }
        self.gist = StructuredGist.from_dict(self.base_valid_data)

    def tearDown(self):
        # Physical RAM constraint hygiene
        gc.collect()

    # =========================================================================
    # 1. Stress Test: Extreme Boundary Values for Uncertainty
    # =========================================================================

    def test_uncertainty_exact_boundaries(self):
        """Valid boundary values 0.0 and 1.0, and int 0 and 1."""
        for val in [0.0, 1.0, 0, 1, 0.5]:
            self.gist.uncertainty = val
            is_valid, errors = self.gist.validate()
            self.assertTrue(is_valid, f"Expected {val} to be valid, got errors: {errors}")
            self.assertEqual(len(errors), 0)

    def test_uncertainty_epsilon_out_of_bounds(self):
        """Values just outside [0.0, 1.0] must be rejected."""
        invalid_floats = [
            -1e-9,
            -1e-15,
            -0.0000000001,
            -1.0,
            -100.0,
            1.0000001,
            1.0 + 1e-9,
            1.0000000000000002,
            2.0,
            1e9,
        ]
        for val in invalid_floats:
            self.gist.uncertainty = val
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid, f"Expected value {val} to fail validation.")
            self.assertTrue(
                any("between 0.0 and 1.0" in e for e in errors),
                f"Error message missing range clarification for {val}: {errors}",
            )

    def test_uncertainty_nan_and_inf(self):
        """NaN, +Inf, -Inf must fail validation."""
        for special_val in [float("nan"), float("inf"), float("-inf"), math.nan, math.inf, -math.inf]:
            self.gist.uncertainty = special_val
            is_valid, errors = self.gist.validate()
            self.assertFalse(
                is_valid,
                f"Expected special float {special_val} to fail validation, but it passed.",
            )
            self.assertTrue(
                any("between 0.0 and 1.0" in e for e in errors),
                f"Expected range error for {special_val}, got {errors}",
            )

    def test_uncertainty_adversarial_types(self):
        """Non-numeric, complex, and boolean types must be rejected."""
        adversarial_types = [
            None,
            "0.5",
            "-0.1",
            "NaN",
            "1.0",
            True,   # bool is subclass of int in Python
            False,  # bool is subclass of int in Python
            complex(0.5, 0.0),
            complex(0.0, 1.0),
            [0.5],
            {"uncertainty": 0.5},
            (0.5,),
            object(),
        ]
        for bad_val in adversarial_types:
            self.gist.uncertainty = bad_val  # type: ignore
            is_valid, errors = self.gist.validate()
            self.assertFalse(
                is_valid,
                f"Expected {type(bad_val).__name__} ({bad_val}) to fail validation.",
            )
            self.assertTrue(
                any("must be a float or int" in e for e in errors),
                f"Expected type error for {bad_val}, got {errors}",
            )

    # =========================================================================
    # 2. Stress Test: Malformed Structures & Missing Mandatory Keys
    # =========================================================================

    def test_objective_adversarial_values(self):
        """Objective must be non-empty string."""
        for bad_obj in ["", "   ", "\t\t\n", None, 123, [], {}, True]:
            self.gist.objective = bad_obj  # type: ignore
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid, f"Objective {bad_obj!r} should fail.")
            self.assertTrue(any("Field 'objective' must be a non-empty string" in e for e in errors))

    def test_core_claim_adversarial_values(self):
        """Core claim must be non-empty string."""
        for bad_claim in ["", "   ", "\r\n", None, 999.9, ["claim"]]:
            self.gist.core_claim = bad_claim  # type: ignore
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid, f"Core claim {bad_claim!r} should fail.")
            self.assertTrue(any("Field 'core_claim' must be a non-empty string" in e for e in errors))

    def test_next_action_adversarial_values(self):
        """Next action must be non-empty string."""
        for bad_action in ["", "  ", None, {}, 0]:
            self.gist.next_action = bad_action  # type: ignore
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid, f"Next action {bad_action!r} should fail.")
            self.assertTrue(any("Field 'next_action' must be a non-empty string" in e for e in errors))

    def test_list_fields_corrupted_elements(self):
        """List fields (evidence, assumptions, constraints, predictions, contradictions, open_questions)
        must reject non-string elements (ints, dicts, None, nested lists)."""
        list_fields = [
            "evidence",
            "assumptions",
            "constraints",
            "predictions",
            "contradictions",
            "open_questions",
        ]
        corrupted_lists = [
            [123],
            ["valid string", None],
            ["valid string", 456.7],
            [{"nested": "dict"}],
            [["nested_list"]],
            [True],
            "not a list at all",
            None,
            12345,
            {"key": "not a list"},
        ]

        for field_name in list_fields:
            for bad_list in corrupted_lists:
                gist = StructuredGist.from_dict(self.base_valid_data)
                setattr(gist, field_name, bad_list)
                is_valid, errors = gist.validate()
                self.assertFalse(
                    is_valid,
                    f"Field {field_name} with value {bad_list!r} should fail validation.",
                )
                self.assertTrue(
                    any(f"Field '{field_name}' must be a list of strings." in e for e in errors),
                    f"Missing expected error message for {field_name} = {bad_list!r}: {errors}",
                )

    def test_provenance_and_time_malformed(self):
        """Check provenance_and_time when type is wrong or missing required keys."""
        # Non-dict
        for bad_prov in [None, "2026-10-29", [1, 2, 3], 12345, True]:
            self.gist.provenance_and_time = bad_prov  # type: ignore
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid)
            self.assertTrue(any("Field 'provenance_and_time' must be a dictionary." in e for e in errors))

        # Dict missing 'created_at'
        self.gist.provenance_and_time = {"source": "unit_test"}
        is_valid, errors = self.gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("missing required 'created_at' timestamp." in e for e in errors))

        # Dict missing 'source'
        self.gist.provenance_and_time = {"created_at": "2026-10-29T00:00:00Z"}
        is_valid, errors = self.gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("missing required 'source' identifier." in e for e in errors))

        # Dict missing both
        self.gist.provenance_and_time = {"custom_field": "test"}
        is_valid, errors = self.gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("missing required 'created_at'" in e for e in errors))
        self.assertTrue(any("missing required 'source'" in e for e in errors))

    def test_revision_history_malformed_entries(self):
        """Check revision_history when non-list or containing invalid entries."""
        # Non-list
        for bad_hist in ["history", None, 123, {}]:
            self.gist.revision_history = bad_hist  # type: ignore
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid)
            self.assertTrue(any("Field 'revision_history' must be a list of dictionaries." in e for e in errors))

        # List containing non-dicts
        for bad_elem in ["not_a_dict", 123, None, ["nested"]]:
            self.gist.revision_history = [bad_elem]  # type: ignore
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid)
            self.assertTrue(any("Field 'revision_history' must be a list of dictionaries." in e for e in errors))

        # List containing dicts missing required keys: revision_id, timestamp, previous_claim, new_claim
        incomplete_entries = [
            {},
            {"revision_id": "rev_001"},
            {"revision_id": "rev_001", "timestamp": "2026-10-29T00:00:00Z"},
            {"revision_id": "rev_001", "timestamp": "2026-10-29T00:00:00Z", "previous_claim": "old"},
        ]
        for entry in incomplete_entries:
            self.gist.revision_history = [entry]
            is_valid, errors = self.gist.validate()
            self.assertFalse(is_valid, f"Entry {entry} should fail validation.")
            self.assertTrue(any("missing required keys" in e for e in errors))

    # =========================================================================
    # 3. Stress Test: Revision Tracking Integrity & Mutation Attack
    # =========================================================================

    def test_revision_immutability_and_chaining(self):
        """Verify that calling revise() produces an independent copy and does not mutate the original."""
        orig_claim = self.gist.core_claim
        orig_uncertainty = self.gist.uncertainty
        orig_history_len = len(self.gist.revision_history)
        orig_evidence_len = len(self.gist.evidence)

        # First revision
        rev1 = self.gist.revise(
            new_claim="Revision claim 1: Non-Newtonian shear thickening observed.",
            reason_for_change="High-shear rate experiment EXP-02 results.",
            trigger_evidence=["EXP-02 rheometer curve"],
            updated_uncertainty=0.12,
            new_contradictions=["Newtonian linearity fails at gamma_dot > 100 s^-1"],
        )

        # Verify orig remains completely untouched
        self.assertEqual(self.gist.core_claim, orig_claim)
        self.assertEqual(self.gist.uncertainty, orig_uncertainty)
        self.assertEqual(len(self.gist.revision_history), orig_history_len)
        self.assertEqual(len(self.gist.evidence), orig_evidence_len)

        # Verify rev1 properties
        self.assertEqual(len(rev1.revision_history), 1)
        self.assertEqual(rev1.revision_history[0]["revision_id"], "rev_001")
        self.assertEqual(rev1.revision_history[0]["previous_claim"], orig_claim)
        self.assertEqual(rev1.revision_history[0]["new_claim"], rev1.core_claim)
        self.assertEqual(rev1.uncertainty, 0.12)
        self.assertIn("EXP-02 rheometer curve", rev1.evidence)

        # Timestamp verification
        t1_str = rev1.revision_history[0]["timestamp"]
        t1 = datetime.fromisoformat(t1_str)
        self.assertIsNotNone(t1)

        # Second revision chained on rev1
        rev2 = rev1.revise(
            new_claim="Revision claim 2: Power-law fluid model Ostwald-de Waele tau = K * gamma_dot^n.",
            reason_for_change="Fitted power law index n = 1.34.",
            trigger_evidence=["Curve fit R^2 = 0.994"],
            updated_uncertainty=0.08,
        )

        # Verify rev1 remains untouched by rev2
        self.assertEqual(len(rev1.revision_history), 1)
        self.assertEqual(rev1.core_claim, "Revision claim 1: Non-Newtonian shear thickening observed.")

        # Verify rev2 history chain
        self.assertEqual(len(rev2.revision_history), 2)
        self.assertEqual(rev2.revision_history[0]["revision_id"], "rev_001")
        self.assertEqual(rev2.revision_history[1]["revision_id"], "rev_002")
        self.assertEqual(rev2.revision_history[1]["previous_claim"], rev1.core_claim)
        self.assertEqual(rev2.revision_history[1]["new_claim"], rev2.core_claim)

        # Monotonicity of timestamps
        t2_str = rev2.revision_history[1]["timestamp"]
        t2 = datetime.fromisoformat(t2_str)
        self.assertGreaterEqual(t2, t1)

        # Validations of all instances
        self.assertTrue(self.gist.validate()[0])
        self.assertTrue(rev1.validate()[0])
        self.assertTrue(rev2.validate()[0])

    def test_direct_mutation_isolation(self):
        """Verify that modifying the dict returned by to_dict() does not mutate the gist internals."""
        gist_dict = self.gist.to_dict()
        gist_dict["evidence"].append("HACKED_EVIDENCE")
        gist_dict["provenance_and_time"]["source"] = "MALICIOUS_SOURCE"
        gist_dict["revision_history"].append({"fake": "entry"})

        self.assertNotIn("HACKED_EVIDENCE", self.gist.evidence)
        self.assertEqual(self.gist.provenance_and_time["source"], "cortex_sd_adversarial_suite")
        self.assertEqual(len(self.gist.revision_history), 0)

    def test_unpicklable_object_in_revise_fails_gracefully(self):
        """Non-serializable / unpicklable threading object in provenance raises expected error on revise."""
        lock = threading.Lock()
        gist = StructuredGist.from_dict(self.base_valid_data)
        gist.provenance_and_time["lock"] = lock

        with self.assertRaises(TypeError):
            gist.revise(new_claim="New Claim", reason_for_change="Reason")

    # =========================================================================
    # 4. Stress Test: Serialization Round-Trip Fuzzing
    # =========================================================================

    def test_json_escaping_and_special_characters(self):
        """JSON serialization must handle quotes, escapes, backslashes, tabs, and newlines."""
        adversarial_strings = [
            r'String with "double quotes" and \'single quotes\'',
            r"Path: C:\Users\Admin\AppData\Local\Temp\test.log",
            "Multi-line:\nLine 1\r\nLine 2\rLine 3\tTabbed\b\f",
            r'Escaped JSON: {"key": "value", "nested": [1, 2, "\"quote\""]}',
            'XML / HTML payload: <script>alert("XSS & entropy");</script>',
            r"LaTeX / Regex: \int_{0}^{\infty} \frac{\partial^2 \psi}{\partial x^2} \, dx \approx \sum_{i=1}^N \lambda_i",
        ]

        for s in adversarial_strings:
            gist = StructuredGist.from_dict(self.base_valid_data)
            gist.objective = s
            gist.core_claim = s
            gist.evidence = [s, s + " suffix"]
            gist.next_action = s

            # Round-trip through JSON
            json_blob = gist.to_json()
            restored = StructuredGist.from_json(json_blob)

            self.assertEqual(restored.objective, s)
            self.assertEqual(restored.core_claim, s)
            self.assertEqual(restored.evidence, [s, s + " suffix"])
            self.assertEqual(restored.next_action, s)

            is_valid, errors = restored.validate()
            self.assertTrue(is_valid, f"Restored gist failed validation: {errors}")

    def test_unicode_and_multilingual_fuzzing(self):
        """JSON serialization must accurately round-trip complex Unicode, math, CJK, RTL, and emojis."""
        unicode_samples = [
            # Mathematical symbols & Greek
            "∇ · B = 0 ∧ ∇ × E = -∂B/∂t, α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, π, ρ, σ, τ, υ, φ, χ, ψ, ω",
            # Emojis & pictorial symbols
            "🔬 Quantum Simulation ⚛️ -> Entropy ΔS ≥ 0 🧪 🚀 💡 🧠 ⚡ 🌌",
            # CJK scripts (Chinese, Japanese, Korean)
            "量子力学における散逸系の解析: 波動関数の崩壊および熱力学的平衡の検証 (양자역학)",
            # RTL scripts (Arabic, Hebrew)
            "دراسة الديناميكا الحرارية في الأنظمة الكمومية המכניקה הקוונטית",
            # Accented & Cyrillic characters
            "Mössbauer-Effekt: Über die Resonanzfluoreszenz von Gammastrahlen in Kernen (Резонансная флуоресценция)",
            # Byte-order marks and zero-width spaces
            "\ufeffZero-width\u200bspace\u200cand\u200djoiner\u200ftest",
        ]

        for sample in unicode_samples:
            gist = StructuredGist.from_dict(self.base_valid_data)
            gist.core_claim = sample
            gist.open_questions = [sample, "Secondary question: " + sample]

            json_str = gist.to_json()
            restored = StructuredGist.from_json(json_str)

            self.assertEqual(restored.core_claim, sample)
            self.assertEqual(restored.open_questions, [sample, "Secondary question: " + sample])
            is_valid, errors = restored.validate()
            self.assertTrue(is_valid, f"Unicode validation failed: {errors}")

    def test_empty_collection_roundtrip(self):
        """Validate gist behavior when optional lists are empty."""
        minimal_data = {
            "objective": "Minimal valid baseline test.",
            "core_claim": "Claim is supported by axiomatic derivation.",
            "evidence": [],
            "assumptions": [],
            "constraints": [],
            "predictions": [],
            "uncertainty": 0.0,
            "next_action": "Halt and report.",
            "contradictions": [],
            "open_questions": [],
            "provenance_and_time": {
                "created_at": "2026-10-29T00:00:00Z",
                "source": "minimal_test",
            },
            "revision_history": [],
        }
        gist = StructuredGist.from_dict(minimal_data)
        is_valid, errors = gist.validate()
        self.assertTrue(is_valid, f"Minimal gist failed: {errors}")

        json_blob = gist.to_json()
        restored = StructuredGist.from_json(json_blob)
        self.assertEqual(restored.evidence, [])
        self.assertEqual(restored.contradictions, [])
        self.assertEqual(restored.uncertainty, 0.0)
        self.assertTrue(restored.validate()[0])

    def test_format_summary_with_adversarial_fields(self):
        """Format summary should render cleanly without throwing exceptions even with empty lists."""
        gist = StructuredGist(
            objective="Test format summary stability",
            core_claim="Everything is stable",
            next_action="Continue monitoring",
        )
        summary = gist.format_summary()
        self.assertIn("[GIST OBJECTIVE]: Test format summary stability", summary)
        self.assertIn("[KEY EVIDENCE]: None", summary)
        self.assertIn("[CONSTRAINTS]: None", summary)
        self.assertIn("[ASSUMPTIONS]: None", summary)
        self.assertIn("[CONTRADICTIONS]: None", summary)


if __name__ == "__main__":
    unittest.main()
