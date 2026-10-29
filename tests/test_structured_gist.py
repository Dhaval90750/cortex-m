"""Unit test suite for CORTEX-M Structured Gist Buffer.

Tests schema validation, error boundaries, type integrity, serialization,
revision history tracking, and summary rendering.
"""

import json
import unittest
from cortex_m.gist import StructuredGist


class TestStructuredGist(unittest.TestCase):
    def setUp(self):
        self.valid_data = {
            "objective": "Characterize damped harmonic oscillator under variable viscosity.",
            "core_claim": "Damping coefficient gamma is proportional to fluid viscosity eta: gamma = 6 * pi * eta * r.",
            "evidence": [
                "Observation series EXP-01: decay envelope matches exp(-gamma*t)",
                "Sensor log: oscillation frequency shifted from 5.0 Hz to 4.82 Hz",
            ],
            "assumptions": ["Laminar flow regime (Reynolds number Re < 1.0)", "Incompressible fluid"],
            "constraints": ["Radius r > 0", "Viscosity eta >= 0.001 Pa*s", "Maximum displacement x_max <= 0.2 m"],
            "predictions": ["Doubling viscosity from 0.01 to 0.02 will halve relaxation time tau."],
            "uncertainty": 0.15,
            "next_action": "Execute calibration run with high-viscosity glycerin solution at 293 K.",
            "contradictions": [],
            "open_questions": ["Does boundary layer separation occur at maximum amplitude?"],
            "provenance_and_time": {
                "created_at": "2026-10-29T00:00:00Z",
                "updated_at": "2026-10-29T00:00:00Z",
                "source": "cortex_sd_hypothesis_engine",
                "version": "1.0.0",
            },
            "revision_history": [],
        }
        self.gist = StructuredGist.from_dict(self.valid_data)

    def test_valid_gist_validation(self):
        is_valid, errors = self.gist.validate()
        self.assertTrue(is_valid, f"Validation unexpectedly failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_missing_objective_fails(self):
        gist = StructuredGist.from_dict(self.valid_data)
        gist.objective = "   "
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertIn("Field 'objective' must be a non-empty string.", errors)

    def test_missing_core_claim_fails(self):
        gist = StructuredGist.from_dict(self.valid_data)
        gist.core_claim = ""
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertIn("Field 'core_claim' must be a non-empty string.", errors)

    def test_missing_next_action_fails(self):
        gist = StructuredGist.from_dict(self.valid_data)
        gist.next_action = ""
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertIn("Field 'next_action' must be a non-empty string.", errors)

    def test_invalid_uncertainty_range(self):
        # Case 1: Negative uncertainty
        gist = StructuredGist.from_dict(self.valid_data)
        gist.uncertainty = -0.05
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Field 'uncertainty' must be between 0.0 and 1.0" in e for e in errors))

        # Case 2: Greater than 1.0
        gist.uncertainty = 1.05
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Field 'uncertainty' must be between 0.0 and 1.0" in e for e in errors))

        # Case 3: Non-numeric
        gist.uncertainty = "low"  # type: ignore
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Field 'uncertainty' must be a float or int" in e for e in errors))

        # Case 4: Boolean (which is subclass of int in Python)
        gist.uncertainty = True  # type: ignore
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Field 'uncertainty' must be a float or int" in e for e in errors))

        # Case 5: Exact boundary values (0.0 and 1.0) must pass
        gist.uncertainty = 0.0
        self.assertTrue(gist.validate()[0])
        gist.uncertainty = 1.0
        self.assertTrue(gist.validate()[0])

    def test_list_type_violations(self):
        gist = StructuredGist.from_dict(self.valid_data)
        gist.evidence = "Not a list"  # type: ignore
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertIn("Field 'evidence' must be a list of strings.", errors)

        gist.evidence = ["Valid observation", 12345]  # type: ignore
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertIn("Field 'evidence' must be a list of strings.", errors)

    def test_constraints_and_assumptions_type_checks(self):
        gist = StructuredGist.from_dict(self.valid_data)
        gist.constraints = [None]  # type: ignore
        self.assertFalse(gist.validate()[0])

        gist = StructuredGist.from_dict(self.valid_data)
        gist.assumptions = [10.5]  # type: ignore
        self.assertFalse(gist.validate()[0])

    def test_provenance_missing_keys(self):
        gist = StructuredGist.from_dict(self.valid_data)
        gist.provenance_and_time = {"source": "manual"}  # missing created_at
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("missing required 'created_at'" in e for e in errors))

        gist.provenance_and_time = {"created_at": "2026-10-29T00:00:00Z"}  # missing source
        is_valid, errors = gist.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("missing required 'source'" in e for e in errors))

    def test_serialization_dict_roundtrip(self):
        d = self.gist.to_dict()
        reconstructed = StructuredGist.from_dict(d)
        self.assertEqual(reconstructed.objective, self.gist.objective)
        self.assertEqual(reconstructed.core_claim, self.gist.core_claim)
        self.assertEqual(reconstructed.evidence, self.gist.evidence)
        self.assertEqual(reconstructed.constraints, self.gist.constraints)
        self.assertEqual(reconstructed.uncertainty, self.gist.uncertainty)
        self.assertEqual(reconstructed.provenance_and_time, self.gist.provenance_and_time)
        self.assertTrue(reconstructed.validate()[0])

    def test_serialization_json_roundtrip(self):
        json_str = self.gist.to_json()
        self.assertIsInstance(json_str, str)
        reconstructed = StructuredGist.from_json(json_str)
        self.assertEqual(reconstructed.objective, self.gist.objective)
        self.assertEqual(reconstructed.core_claim, self.gist.core_claim)
        self.assertEqual(reconstructed.predictions, self.gist.predictions)
        self.assertTrue(reconstructed.validate()[0])

    def test_revision_history_tracking(self):
        initial_history_len = len(self.gist.revision_history)
        self.assertEqual(initial_history_len, 0)

        # Trigger revision with new counter-evidence
        new_claim = "Damping includes non-linear turbulent drag term: F_drag = -6*pi*eta*r*v - c_d*v^2."
        reason = "Observed quadratic deviation at velocity v > 0.4 m/s in high-amplitude regime."
        new_evidence = ["High-speed video analysis: boundary layer separation confirmed at x=0.15m"]

        revised_gist = self.gist.revise(
            new_claim=new_claim,
            reason_for_change=reason,
            trigger_evidence=new_evidence,
            updated_uncertainty=0.25,
            new_contradictions=["Linear Stokes formula fails at Re > 10"],
        )

        # 1. Prior gist remains unchanged (immutability)
        self.assertEqual(len(self.gist.revision_history), 0)
        self.assertEqual(self.gist.core_claim, self.valid_data["core_claim"])

        # 2. Revised gist has updated fields and logged revision entry
        self.assertEqual(len(revised_gist.revision_history), 1)
        self.assertEqual(revised_gist.core_claim, new_claim)
        self.assertEqual(revised_gist.uncertainty, 0.25)
        self.assertIn(new_evidence[0], revised_gist.evidence)
        self.assertIn("Linear Stokes formula fails at Re > 10", revised_gist.contradictions)

        # 3. Check revision entry content
        entry = revised_gist.revision_history[0]
        self.assertEqual(entry["revision_id"], "rev_001")
        self.assertEqual(entry["previous_claim"], self.valid_data["core_claim"])
        self.assertEqual(entry["new_claim"], new_claim)
        self.assertEqual(entry["reason_for_change"], reason)
        self.assertEqual(entry["trigger_evidence"], new_evidence)

        # 4. Check validation of revised gist
        is_valid, errors = revised_gist.validate()
        self.assertTrue(is_valid, f"Revised gist failed validation: {errors}")

    def test_format_summary_rendering(self):
        summary = self.gist.format_summary()
        self.assertIn("[GIST OBJECTIVE]: Characterize damped harmonic oscillator", summary)
        self.assertIn("[CORE CLAIM]: Damping coefficient gamma", summary)
        self.assertIn("[KEY EVIDENCE]: Observation series EXP-01", summary)
        self.assertIn("[CONSTRAINTS]: Radius r > 0", summary)
        self.assertIn("[NEXT ACTION]: Execute calibration run", summary)

    def test_from_dict_non_dict_raises_type_error(self):
        """Passing non-dict to from_dict must raise TypeError."""
        for invalid_input in ["not a dict", 12345, [1, 2, 3], None]:
            with self.assertRaises(TypeError):
                StructuredGist.from_dict(invalid_input)  # type: ignore

    def test_revise_invalid_claim_raises_value_error(self):
        """Revising to an invalid state must fail validation and raise ValueError."""
        with self.assertRaises(ValueError):
            self.gist.revise(new_claim="", reason_for_change="Empty claim is invalid")
        with self.assertRaises(ValueError):
            self.gist.revise(new_claim="Valid claim", reason_for_change="Valid reason", updated_uncertainty=-0.5)


if __name__ == "__main__":
    unittest.main()
