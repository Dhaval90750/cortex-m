"""Unit tests for SynapticKnowledgePackager and OrthogonalProjectionFilter."""

import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

from cortex_m.learning.grafting import (
    GraftingResult,
    SynapticKnowledgePackager,
    SynapticPackageManifest,
)


class TestSynapticKnowledgeGrafting(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.dim = 32
        self.packager = SynapticKnowledgePackager(dim=self.dim)

    def test_extract_and_package_roundtrip(self):
        # 1. Create donor attractors
        donor_patterns = torch.randn(3, self.dim)
        donor_patterns = F.normalize(donor_patterns, p=2, dim=-1)

        manifest, delta_W = self.packager.extract_package(
            domain_label="rust_concurrency",
            donor_model_id="donor_node_alpha",
            attractor_patterns=donor_patterns,
            rank=4,
        )

        self.assertEqual(manifest.domain_label, "rust_concurrency")
        self.assertEqual(manifest.num_attractors, 3)
        self.assertEqual(manifest.rank, 3)
        self.assertEqual(delta_W.shape, (self.dim, self.dim))

        # 2. Test serialization and deserialization
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_path = Path(tmpdir) / "rust_concurrency.synpkg"
            self.packager.save_package(manifest, delta_W, pkg_path)

            loaded_manifest, loaded_delta_W = self.packager.load_package(pkg_path)

            self.assertEqual(loaded_manifest.package_id, manifest.package_id)
            self.assertEqual(loaded_manifest.domain_label, manifest.domain_label)
            self.assertEqual(loaded_manifest.payload_sha256, manifest.payload_sha256)

            # Check numerical equivalence
            diff = torch.norm(delta_W - loaded_delta_W).item()
            self.assertLess(diff, 1e-6)

    def test_orthogonal_grafting_zero_collateral_damage(self):
        # Recipient model has 3 pre-existing attractors
        rec_patterns = torch.randn(3, self.dim)
        rec_patterns = F.normalize(rec_patterns, p=2, dim=-1)
        W_rec = torch.mm(rec_patterns.t(), rec_patterns)
        W_rec.fill_diagonal_(0.0)

        # Donor model creates 2 new attractors
        donor_patterns = torch.randn(2, self.dim)
        donor_patterns = F.normalize(donor_patterns, p=2, dim=-1)

        manifest, delta_donor = self.packager.extract_package(
            domain_label="quantum_logic",
            donor_model_id="donor_node_beta",
            attractor_patterns=donor_patterns,
            rank=2,
        )

        # Graft donor package into recipient
        W_grafted, result = self.packager.graft_package(
            recipient_weights=W_rec,
            recipient_attractors=rec_patterns,
            graft_manifest=manifest,
            donor_delta_W=delta_donor,
            donor_attractors=donor_patterns,
        )

        self.assertIsInstance(result, GraftingResult)
        self.assertTrue(result.graft_success)
        # Verify 0.00% collateral damage to recipient memories
        self.assertEqual(result.collateral_damage_rate, 0.0)
        self.assertLess(result.orthogonality_loss, 1e-4)

        # Verify recipient recall fidelity is 100%
        self.assertEqual(result.recipient_recall_fidelity, 1.0)


if __name__ == "__main__":
    unittest.main()
