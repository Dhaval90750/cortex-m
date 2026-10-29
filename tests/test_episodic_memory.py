"""Comprehensive Unit Test Suite for CORTEX-M Episodic & Associative Memory Subsystem.

Conforms to CORTEX-M Master Execution Guide §26 (Phase 6, Milestone 1).
Tests schema validation, Continuous Modern Hopfield integration, Lyapunov energy descent,
provenance filtering, invalidation tracking, anti-Hebbian unlearning, zero collateral degradation,
and epistemic abstention.
"""

from __future__ import annotations

import unittest
import torch

from cortex_m.memory import EpisodicRecord, EpisodicMemoryStore
from cortex_m.pathway.encoder import NeuralFeatureEncoder


class TestEpisodicMemorySubsystem(unittest.TestCase):
    """Test suite for EpisodicRecord and EpisodicMemoryStore."""

    def setUp(self):
        torch.manual_seed(42)
        self.encoder = NeuralFeatureEncoder(output_dim=64, seed=42)
        self.store = EpisodicMemoryStore(dim=64, beta=8.0, encoder=self.encoder)

    # =========================================================================
    # 1. Schema & Validation Tests
    # =========================================================================

    def test_record_validation_success(self):
        record = EpisodicRecord(
            record_id="rec_001",
            claim="Superconducting transition temperature of YBCO is 93 Kelvin at atmospheric pressure.",
            evidence_type="Observed",
            confidence=0.98,
            source="condensed_matter_lab",
            invalidation_conditions=["Pressure exceeds 50 GPa", "Sample degraded by moisture"],
        )
        is_valid, errors = record.validate()
        self.assertTrue(is_valid, f"Validation failed unexpectedly: {errors}")
        self.assertEqual(len(errors), 0)

    def test_record_validation_invalid_confidence(self):
        # Confidence > 1.0
        rec_high = EpisodicRecord(
            record_id="rec_high",
            claim="Energy is conserved.",
            confidence=1.5,
        )
        is_valid, errors = rec_high.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("confidence" in e for e in errors))

        # Confidence < 0.0
        rec_low = EpisodicRecord(
            record_id="rec_low",
            claim="Energy is conserved.",
            confidence=-0.1,
        )
        is_valid, errors = rec_low.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("confidence" in e for e in errors))

    def test_record_validation_invalid_evidence_type(self):
        record = EpisodicRecord(
            record_id="rec_ev",
            claim="Quantum entanglement is non-local.",
            evidence_type="Rumor",  # Invalid type
        )
        is_valid, errors = record.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("evidence_type" in e for e in errors))

    def test_record_validation_missing_fields(self):
        rec_no_id = EpisodicRecord(record_id="", claim="Valid claim.")
        is_valid, errors = rec_no_id.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("record_id" in e for e in errors))

        rec_no_claim = EpisodicRecord(record_id="valid_id", claim="   ")
        is_valid, errors = rec_no_claim.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("claim" in e for e in errors))

    def test_record_validation_invalidation_without_reason(self):
        record = EpisodicRecord(
            record_id="rec_inv",
            claim="Phlogiston is released during combustion.",
            is_invalidated=True,
            invalidation_reason=None,
        )
        is_valid, errors = record.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("invalidation_reason" in e for e in errors))

    def test_record_serialization_round_trip(self):
        record = EpisodicRecord(
            record_id="rec_serial",
            timestamp=1700000000.0,
            claim="DNA transcription proceeds from 5-prime to 3-prime direction.",
            evidence_type="Observed",
            confidence=0.99,
            source="molecular_biology_core",
            invalidation_conditions=["Reverse transcriptase mechanism observed"],
            tags=["genetics", "biochemistry"],
            context={"cell_type": "eukaryotic"},
        )
        # Dict round-trip
        data = record.to_dict()
        restored = EpisodicRecord.from_dict(data)
        self.assertEqual(record.record_id, restored.record_id)
        self.assertEqual(record.claim, restored.claim)
        self.assertEqual(record.evidence_type, restored.evidence_type)
        self.assertEqual(record.confidence, restored.confidence)
        self.assertEqual(record.tags, restored.tags)
        self.assertEqual(record.context, restored.context)

        # JSON round-trip
        json_str = record.to_json()
        restored_json = EpisodicRecord.from_json(json_str)
        self.assertEqual(record.record_id, restored_json.record_id)
        self.assertEqual(record.claim, restored_json.claim)

    # =========================================================================
    # 2. Store & Lookup Tests
    # =========================================================================

    def test_store_and_exact_get(self):
        rec = EpisodicRecord(
            record_id="rec_get",
            claim="Speed of light in vacuum is exactly 299792458 meters per second.",
            evidence_type="Observed",
            confidence=1.0,
        )
        rid = self.store.store(rec)
        self.assertEqual(rid, "rec_get")
        retrieved = self.store.get("rec_get")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.claim, rec.claim)

    def test_store_invalid_record_raises_error(self):
        bad_rec = EpisodicRecord(record_id="", claim="")
        with self.assertRaises(ValueError):
            self.store.store(bad_rec)

    def test_active_and_invalidated_record_lists(self):
        r1 = EpisodicRecord(record_id="r1", claim="Fact 1", evidence_type="Observed")
        r2 = EpisodicRecord(record_id="r2", claim="Fact 2", evidence_type="Observed")
        self.store.store(r1)
        self.store.store(r2)
        self.assertEqual(len(self.store.active_records()), 2)
        self.assertEqual(len(self.store.invalidated_records()), 0)

        self.store.invalidate("r1", "Disproven by replication study")
        self.assertEqual(len(self.store.active_records()), 1)
        self.assertEqual(len(self.store.invalidated_records()), 1)
        self.assertEqual(self.store.invalidated_records()[0].record_id, "r1")

    # =========================================================================
    # 3. Associative Retrieval & Continuous Hopfield Dynamics
    # =========================================================================

    def test_associative_recall_with_paraphrase(self):
        rec = EpisodicRecord(
            record_id="fluid_01",
            claim="Fluid viscosity directly determines the damping rate of the mechanical oscillator.",
            evidence_type="Observed",
            confidence=0.95,
        )
        self.store.store(rec)

        # Paraphrased query
        query = "How does fluid viscosity affect oscillator damping rate?"
        results = self.store.retrieve(query, top_k=1, relevance_threshold=0.20)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record_id, "fluid_01")

    def test_lyapunov_energy_descent_during_recall(self):
        # Store multiple attractors
        claims = [
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
            "Gravitational wave emissions detected from binary black hole coalescence.",
            "CRISPR Cas9 endonuclease introduces targeted double strand DNA breaks.",
        ]
        for idx, clm in enumerate(claims):
            self.store.store(
                EpisodicRecord(record_id=f"rec_dyn_{idx}", claim=clm, evidence_type="Observed")
            )

        # Recall using cue and trace energy descent directly via underlying network
        q_vec = self.encoder.encode_text("Black hole merger gravitational radiation")
        converged_state, trace, diag = self.store.hopfield.recall(q_vec)

        # Verify Lyapunov energy strictly non-increasing across iterations
        energies = [step.energy for step in trace.steps]
        self.assertGreater(len(energies), 0)
        for i in range(1, len(energies)):
            # Lyapunov property: E(x_{t+1}) <= E(x_t) + numerical eps
            self.assertLessEqual(energies[i], energies[i - 1] + 1e-4)

    # =========================================================================
    # 4. Provenance & Confidence Filtering
    # =========================================================================

    def test_provenance_evidence_type_filtering(self):
        rec_obs = EpisodicRecord(
            record_id="cand_obs",
            claim="Muon anomalous magnetic dipole moment exceeds standard model prediction by 4.2 sigma.",
            evidence_type="Observed",
            confidence=0.95,
        )
        rec_hyp = EpisodicRecord(
            record_id="cand_hyp",
            claim="Supersymmetric sleptons explain muon magnetic dipole moment anomaly.",
            evidence_type="Hypothesis",
            confidence=0.60,
        )
        self.store.store(rec_obs)
        self.store.store(rec_hyp)

        query = "muon magnetic dipole moment anomaly"
        # Filter for Observed evidence only
        results_obs = self.store.retrieve(query, evidence_types=["Observed"], top_k=5, relevance_threshold=0.20)
        r_ids_obs = [r.record_id for r in results_obs]
        self.assertIn("cand_obs", r_ids_obs)
        self.assertNotIn("cand_hyp", r_ids_obs)

        # Filter for Hypothesis only
        results_hyp = self.store.retrieve(query, evidence_types=["Hypothesis"], top_k=5, relevance_threshold=0.20)
        r_ids_hyp = [r.record_id for r in results_hyp]
        self.assertIn("cand_hyp", r_ids_hyp)
        self.assertNotIn("cand_obs", r_ids_hyp)

    def test_confidence_threshold_filtering(self):
        rec_high = EpisodicRecord(
            record_id="conf_high",
            claim="Electron charge is negative 1.602 times 10 to the minus 19 Coulombs.",
            evidence_type="Observed",
            confidence=0.99,
        )
        rec_low = EpisodicRecord(
            record_id="conf_low",
            claim="Fractional charges observed in ambient graphite flakes.",
            evidence_type="Observed",
            confidence=0.35,
        )
        self.store.store(rec_high)
        self.store.store(rec_low)

        query = "electron charge negative Coulombs"
        # Require min_confidence=0.80
        results = self.store.retrieve(query, min_confidence=0.80, top_k=5, relevance_threshold=0.20)
        r_ids = [r.record_id for r in results]
        self.assertIn("conf_high", r_ids)
        self.assertNotIn("conf_low", r_ids)

    def test_source_filtering(self):
        rec_cern = EpisodicRecord(
            record_id="src_cern",
            claim="Higgs boson mass measured at 125.1 GeV.",
            evidence_type="Observed",
            source="CERN_ATLAS",
        )
        rec_fermi = EpisodicRecord(
            record_id="src_fermi",
            claim="Top quark mass measured at 172.7 GeV.",
            evidence_type="Observed",
            source="Fermilab_CDF",
        )
        self.store.store(rec_cern)
        self.store.store(rec_fermi)

        query = "Higgs boson mass measured"
        results_cern = self.store.retrieve(query, sources=["CERN_ATLAS"], top_k=5, relevance_threshold=0.20)
        self.assertTrue(all(r.source == "CERN_ATLAS" for r in results_cern))
        self.assertIn("src_cern", [r.record_id for r in results_cern])
        self.assertNotIn("src_fermi", [r.record_id for r in results_cern])

    # =========================================================================
    # 5. Invalidation & Active Filtering
    # =========================================================================

    def test_invalidation_marks_record_and_sets_reason(self):
        rec = EpisodicRecord(
            record_id="rec_flawed",
            claim="Faster than light neutrinos observed in OPERA experiment.",
            evidence_type="Observed",
            confidence=0.85,
        )
        self.store.store(rec)
        ok = self.store.invalidate("rec_flawed", "Loose optical fiber timing cable identified")
        self.assertTrue(ok)

        stored = self.store.get("rec_flawed")
        self.assertTrue(stored.is_invalidated)
        self.assertEqual(stored.invalidation_reason, "Loose optical fiber timing cable identified")

    def test_invalidated_record_excluded_from_default_retrieval(self):
        rec = EpisodicRecord(
            record_id="neutrino_flawed",
            claim="Faster than light neutrinos observed traveling from CERN to Gran Sasso.",
            evidence_type="Observed",
            confidence=0.90,
        )
        self.store.store(rec)
        self.store.invalidate("neutrino_flawed", "Defective optical connector artifact")

        query = "superluminal neutrino transmission from CERN"
        results = self.store.retrieve(query, top_k=5)
        self.assertEqual(len(results), 0)

    def test_invalidated_record_retrievable_when_flag_enabled(self):
        rec = EpisodicRecord(
            record_id="cold_fusion",
            claim="Electrochemical excess heat generation indicates room temperature fusion.",
            evidence_type="Observed",
            confidence=0.70,
        )
        self.store.store(rec)
        self.store.invalidate("cold_fusion", "Calorimetric calibration error")

        query = "electrochemical excess heat generation room temperature"
        # Default excludes it
        self.assertEqual(len(self.store.retrieve(query)), 0)

        # Flag retrieves it for audit/historical analysis
        results = self.store.retrieve(query, include_invalidated=True, top_k=5, relevance_threshold=0.20)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record_id, "cold_fusion")
        self.assertTrue(results[0].is_invalidated)

    # =========================================================================
    # 6. Anti-Hebbian Unlearning & Zero Collateral Degradation (CRITICAL)
    # =========================================================================

    def test_anti_hebbian_unlearning_zero_collateral_degradation(self):
        """Verify that targeted anti-Hebbian unlearning depresses the invalidated pattern
        while leaving all unrelated patterns 100% intact (0.0% collateral degradation).
        """
        facts = [
            ("fact_01", "Planetary orbits form ellipses with the Sun at one focus.", "planetary orbital ellipses around the Sun"),
            ("fact_02", "Mitochondria produce cellular adenosine triphosphate via oxidative phosphorylation.", "mitochondria cellular ATP production"),
            ("fact_03", "Transistors amplify electronic signals using semiconductor charge carriers.", "transistors amplify signals with semiconductors"),
            ("fact_04", "Plate tectonics drives continental drift through mantle convection currents.", "plate tectonics and continental drift convection"),
            ("fact_05", "Enzymes lower reaction activation energy without undergoing permanent consumption.", "enzymes lower chemical reaction activation energy"),
            ("fact_06", "General relativity describes gravitation as spacetime curvature induced by mass-energy.", "general relativity gravitation as spacetime curvature"),
            ("fact_07", "Hemoglobin binds molecular oxygen cooperatively using iron porphyrin complexes.", "hemoglobin binding molecular oxygen cooperatively"),
            ("fact_08", "Maxwell equations unify electricity and magnetism into electromagnetic waves.", "Maxwell equations electromagnetic wave unification"),
            ("fact_09", "Specific heat capacity of liquid water is approximately 4184 Joules per Kelvin.", "specific heat capacity of water in Joules per Kelvin"),
            ("fact_10", "STALE: Cold dark matter consists exclusively of sterile neutrinos of 3 keV mass.", "cold dark matter sterile neutrinos hypothesis"),
        ]

        # 1. Imprint all 10 facts
        for fid, clm, _ in facts:
            self.store.store(
                EpisodicRecord(
                    record_id=fid,
                    claim=clm,
                    evidence_type="Observed",
                    confidence=0.95,
                    source="canonical_corpus",
                )
            )

        # 2. Verify all 10 facts are retrievable at Top-1
        initial_recall = 0
        for fid, _, probe_query in facts:
            res = self.store.retrieve(probe_query, top_k=1, relevance_threshold=0.15)
            if res and res[0].record_id == fid:
                initial_recall += 1
        self.assertEqual(initial_recall, 10, f"Expected 10/10 initial recall, got: {initial_recall}")

        # 3. Apply targeted invalidation and anti-Hebbian depression on target fact_10
        target_fid = "fact_10"
        unlearn_ok = self.store.invalidate(
            target_fid,
            "Ruled out by Lyman-alpha forest clustering and NuSTAR X-ray non-detection",
        )
        self.assertTrue(unlearn_ok)

        # 4. Verify target fact is completely unlearned from active retrieval
        target_probe = "cold dark matter sterile neutrinos hypothesis"
        target_res = self.store.retrieve(target_probe, top_k=1, relevance_threshold=0.15)
        if target_res:
            self.assertNotEqual(
                target_res[0].record_id,
                target_fid,
                "Target fact should NOT be retrieved after anti-Hebbian unlearning",
            )

        # 5. Verify all 9 unrelated facts remain 100% intact (Collateral Degradation = 0.0%)
        retained_count = 0
        for fid, _, probe_query in facts[:9]:
            res = self.store.retrieve(probe_query, top_k=1, relevance_threshold=0.15)
            if res and res[0].record_id == fid:
                retained_count += 1

        self.assertEqual(
            retained_count,
            9,
            f"Collateral degradation detected! Expected 9/9 intact facts, got: {retained_count}",
        )
        collateral_degradation = (9 - retained_count) / 9.0
        self.assertEqual(collateral_degradation, 0.0)

    # =========================================================================
    # 7. Edge Cases & Epistemic Abstention
    # =========================================================================

    def test_empty_store_retrieval_returns_empty_list(self):
        res = self.store.retrieve("Any query in empty store")
        self.assertEqual(res, [])

    def test_out_of_distribution_query_abstains(self):
        # Store facts strictly in physics and molecular biology
        self.store.store(
            EpisodicRecord(
                record_id="phys_01",
                claim="Photon momentum equals Planck constant divided by wavelength.",
                evidence_type="Observed",
                confidence=0.99,
            )
        )
        self.store.store(
            EpisodicRecord(
                record_id="bio_01",
                claim="Ribosomes synthesize polypeptides through messenger RNA translation.",
                evidence_type="Observed",
                confidence=0.99,
            )
        )

        # Out-of-distribution query completely irrelevant to physics/biology
        ood_query = "What is the secret recipe for baking artisanal chocolate croissants?"
        res = self.store.retrieve(ood_query, top_k=5, relevance_threshold=0.20)
        # Should cleanly abstain (empty list)
        self.assertEqual(
            len(res),
            0,
            f"Expected epistemic abstention on OOD query, but got: {[r.claim for r in res]}",
        )

    def test_clear_store_resets_hopfield_and_records(self):
        self.store.store(
            EpisodicRecord(record_id="temp_rec", claim="Temporary statement.", evidence_type="Observed")
        )
        self.assertEqual(len(self.store._records), 1)
        self.assertEqual(self.store.hopfield.num_memories, 1)

        self.store.clear()
        self.assertEqual(len(self.store._records), 0)
        self.assertEqual(self.store.hopfield.num_memories, 0)

    def test_inspect_store_diagnostics(self):
        self.store.store(
            EpisodicRecord(record_id="diag_1", claim="Quantum entanglement invariant.", evidence_type="Observed")
        )
        info = self.store.inspect_store()
        self.assertIn("total_records", info)
        self.assertIn("active_records", info)
        self.assertIn("invalidated_records", info)
        self.assertIn("hopfield_attractors", info)
        self.assertIn("synaptic_weights_norm", info)
        self.assertIn("synaptic_sparsity", info)
        self.assertEqual(info["total_records"], 1)
        self.assertEqual(info["active_records"], 1)
        self.assertEqual(info["invalidated_records"], 0)
        self.assertEqual(info["hopfield_attractors"], 1)

    def test_duplicate_store_replaces_attractor_without_leak(self):
        """Storing an updated record with same record_id replaces attractor rather than duplicating."""
        rec1 = EpisodicRecord(record_id="dup_01", claim="Initial version of the fact.", evidence_type="Observed")
        self.store.store(rec1)
        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.store.hopfield.num_memories, 1)

        # Store updated version with same record_id
        rec2 = EpisodicRecord(record_id="dup_01", claim="Updated and refined version of the fact.", evidence_type="Observed")
        self.store.store(rec2)
        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.store.hopfield.num_memories, 1)
        self.assertEqual(self.store.records["dup_01"].claim, "Updated and refined version of the fact.")

        # Invalidate and ensure attractor count cleanly drops to 0
        self.store.invalidate("dup_01", "Superseded")
        self.assertEqual(self.store.hopfield.num_memories, 0)
        self.assertEqual(len(self.store.active_records()), 0)
        self.assertEqual(len(self.store.invalidated_records()), 1)


if __name__ == "__main__":
    unittest.main()
