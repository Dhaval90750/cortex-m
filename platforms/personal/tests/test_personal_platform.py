"""Unit tests for CORTEX-Personal platform."""

import socket
import tempfile
import unittest
from pathlib import Path
import torch

from platforms.personal.src.interlock import ZeroEgressInterlock, ZeroEgressViolation
from platforms.personal.src.substrate import PersonalSubstrateCoordinator
from platforms.personal.src.tenant_paging import PersonalTenantPagingEngine
from platforms.personal.src.vault import EncryptedPersonalVault


class TestCortexPersonalPlatform(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.coordinator = PersonalSubstrateCoordinator(
            dim=32,
            gist_dim=16,
            beta=6.0,
            abstention_threshold=0.60,
            enforce_zero_egress=False,  # Keep off during general test setup
        )

    def tearDown(self):
        self.coordinator.shutdown()

    def test_newborn_tabula_rasa_abstention(self):
        # Fresh instance has zero facts and must honestly abstain
        res = self.coordinator.query("What is Elena's confidential source code?")
        self.assertEqual(res.status, "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(res.recalled_text)
        self.assertEqual(res.confidence, 0.0)

    def test_on_the_go_hebbian_learning_and_verified_recall(self):
        # 1. Teach a personal confidential note
        fact = "The underground facility location is Sector 7."
        mid = self.coordinator.teach(fact, source_context="field_report_2026")
        self.assertTrue(mid.startswith("pers_mem_"))

        # 2. Query with semantically matching cue
        res = self.coordinator.query("Where is the facility location?")
        self.assertIn(res.status, ["VERIFIED", "PARTIAL"])
        self.assertEqual(res.recalled_text, fact)
        self.assertEqual(res.source_context, "field_report_2026")

        # 3. Query with completely unrelated cue (should abstain)
        unrelated = self.coordinator.query("What is the recipe for chocolate soufflé?")
        self.assertEqual(unrelated.status, "INSUFFICIENT_EVIDENCE")

    def test_anti_hebbian_targeted_unlearning(self):
        fact1 = "Project Blue is stored on Server 101."
        fact2 = "Project Red is stored on Server 202."
        m1 = self.coordinator.teach(fact1)
        m2 = self.coordinator.teach(fact2)

        # Verify fact1 is present
        self.assertIsNotNone(self.coordinator.query("Where is Project Blue stored?").recalled_text)

        # Unlearn fact1
        success = self.coordinator.unlearn(m1)
        self.assertTrue(success)

        # Fact 1 must now be forgotten
        res1 = self.coordinator.query("Where is Project Blue stored?")
        self.assertNotEqual(res1.recalled_text, fact1)

        # Fact 2 must remain perfectly intact (0.00% collateral damage)
        res2 = self.coordinator.query("Where is Project Red stored?")
        self.assertEqual(res2.recalled_text, fact2)

    def test_encrypted_vault_save_and_load_roundtrip(self):
        self.coordinator.teach("Confidential passkey is 9874-XK.")
        passphrase = "SuperSecretPersonalKey!2026"

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_file = Path(tmpdir) / "profile.cortex_vault"

            # 1. Save encrypted vault
            self.coordinator.save_vault(vault_file, passphrase)
            self.assertTrue(vault_file.exists())
            self.assertGreater(vault_file.stat().st_size, 100)

            # 2. Load into fresh coordinator
            fresh_coordinator = PersonalSubstrateCoordinator(
                dim=32,
                gist_dim=16,
                beta=6.0,
                enforce_zero_egress=False,
            )
            fresh_coordinator.load_vault(vault_file, passphrase)

            # Verify memory restored accurately
            res = fresh_coordinator.query("What is the confidential passkey?")
            self.assertIn("9874-XK", res.recalled_text)
            fresh_coordinator.shutdown()

            # 3. Test wrong passphrase raises PermissionError
            bad_coordinator = PersonalSubstrateCoordinator(
                dim=32,
                gist_dim=16,
                beta=6.0,
                enforce_zero_egress=False,
            )
            with self.assertRaises(PermissionError):
                bad_coordinator.load_vault(vault_file, "WrongPassphrase123")
            bad_coordinator.shutdown()

    def test_zero_egress_interlock_blocks_external_wan(self):
        interlock = ZeroEgressInterlock(strict_mode=True)
        interlock.enable()
        self.assertTrue(interlock.is_active())

        # Attempting to connect to remote public IP should raise ZeroEgressViolation
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with self.assertRaises(ZeroEgressViolation):
            sock.connect(("8.8.8.8", 53))
        sock.close()

        interlock.disable()
        self.assertFalse(interlock.is_active())

    def test_personal_tenant_paging_isolation(self):
        engine = PersonalTenantPagingEngine(base_dimension=32, rank=4)
        t1 = engine.create_tenant("profile_journalism")
        t2 = engine.create_tenant("profile_medical")

        self.assertIn("profile_journalism", engine.list_tenants())
        self.assertIn("profile_medical", engine.list_tenants())
        self.assertLessEqual(t1.size_bytes, 2048)

        cue = torch.randn(32)
        engine.learn_tenant_association("profile_journalism", cue)

        # Verify t1 has association, t2 is completely unaffected
        self.assertEqual(len(t1.active_attractors), 1)
        self.assertEqual(len(t2.active_attractors), 0)

    def test_tenant_disk_paging_and_footprint(self):
        engine = PersonalTenantPagingEngine(base_dimension=32, rank=4)
        t = engine.create_tenant("profile_finance", metadata={"tier": "vip"})
        cue = torch.randn(32)
        engine.learn_tenant_association("profile_finance", cue)

        # Storage footprint
        footprint = engine.get_storage_footprint()
        self.assertEqual(footprint["num_tenants"], 1)
        self.assertGreater(footprint["total_bytes"], 0)

        # Page out to disk
        with tempfile.TemporaryDirectory() as tmpdir:
            page_path = Path(tmpdir) / "profile_finance.slice"
            saved_path = engine.page_out_to_disk("profile_finance", str(page_path))
            self.assertTrue(Path(saved_path).exists())

            # Page into fresh engine
            fresh_engine = PersonalTenantPagingEngine(base_dimension=32, rank=4)
            restored = fresh_engine.page_in_from_disk(str(page_path))
            self.assertEqual(restored.tenant_id, "profile_finance")
            self.assertEqual(restored.metadata["tier"], "vip")
            self.assertEqual(len(restored.active_attractors), 1)

    def test_differentially_private_tenant_learning(self):
        engine = PersonalTenantPagingEngine(base_dimension=32, rank=4)
        engine.create_tenant("profile_private")
        cue = torch.randn(32)

        res = engine.learn_tenant_association_differentially_private(
            tenant_id="profile_private",
            cue=cue,
            epsilon=0.5,
            delta=1e-5,
            clip_norm=1.0,
        )
        self.assertEqual(res["tenant_id"], "profile_private")
        self.assertEqual(res["epsilon"], 0.5)
        self.assertGreater(res["sigma"], 0.0)
        self.assertIn("snr_db", res)
        self.assertEqual(res["accumulated_epsilon"], 0.5)
        self.assertEqual(res["update_count"], 1)

        # Second update accumulates budget
        res2 = engine.learn_tenant_association_differentially_private(
            tenant_id="profile_private",
            cue=cue,
            epsilon=0.5,
            delta=1e-5,
        )
        self.assertEqual(res2["accumulated_epsilon"], 1.0)
        self.assertEqual(res2["update_count"], 2)


if __name__ == "__main__":
    unittest.main()

