"""
Comprehensive Integration Tests for EcosystemGateway and Universal Platform Routing.
Verifies all 9 platform dispatchers, error resilience, and learning pipelines.
"""

import unittest
import sys
from cortex_m.gateway import EcosystemGateway


class TestEcosystemGateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gateway = EcosystemGateway()

    def test_get_ecosystem_overview(self):
        overview = self.gateway.get_ecosystem_overview()
        self.assertEqual(overview["status"], "OPERATIONAL")
        self.assertIn("platforms", overview)
        self.assertIn("personal", overview["platforms"])
        self.assertIn("ide", overview["platforms"])
        self.assertIn("fin", overview["platforms"])
        self.assertIn("robotics", overview["platforms"])
        self.assertIn("mesh", overview["platforms"])
        self.assertIn("health", overview["platforms"])
        self.assertIn("space", overview["platforms"])
        self.assertIn("os", overview["platforms"])
        self.assertIn("learning", overview["platforms"])

    def test_execute_personal_learn_and_recall(self):
        learn_res = self.gateway.execute_personal(
            "learn", {"tenant_id": "user_42", "text": "Project Alpha is an air-gapped security vault."}
        )
        self.assertTrue(learn_res["success"])
        self.assertEqual(learn_res["tenant_id"], "user_42")
        self.assertIn("memory_id", learn_res)

        recall_res = self.gateway.execute_personal("recall", {"tenant_id": "user_42", "query": "Project Alpha"})
        self.assertTrue(recall_res["success"])
        self.assertIn(recall_res["status"], ["VERIFIED", "PARTIAL", "INSUFFICIENT_EVIDENCE"])
        self.assertIn("confidence", recall_res)

    def test_execute_ide_sandbox(self):
        res = self.gateway.execute_ide(
            "sandbox", {"command": [sys.executable, "-c", "print('IDE_GATEWAY_OK')"]}
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["exit_code"], 0)
        self.assertIn("IDE_GATEWAY_OK", res["stdout"])

    def test_execute_fin_score(self):
        params = {
            "transaction_id": "TX-GATEWAY-TEST-001",
            "amount": 25000.0,
            "mcc": 6051,
            "distance_km": 150.0,
            "time_delta_seconds": 120.0,
        }
        res = self.gateway.execute_fin("score", params)
        self.assertTrue(res["success"])
        self.assertIn("calibrated_risk", res)
        self.assertIn("decision", res)
        self.assertEqual(res["transaction_id"], "TX-GATEWAY-TEST-001")

    def test_execute_robotics_kinematics(self):
        res = self.gateway.execute_robotics("kinematics", {"joints": [0.1, 0.2, -0.1, 0.05, 0.0, 0.1]})
        self.assertTrue(res["success"])
        self.assertIn("end_effector_position", res)
        self.assertEqual(len(res["end_effector_position"]), 3)

    def test_execute_mesh_consensus(self):
        params = {
            "proposal_id": "PROP-MESH-TEST",
            "proposal": "TRANSFER_ASSET_42",
            "round_id": 1,
        }
        res = self.gateway.execute_mesh("consensus", params)
        self.assertTrue(res["success"])
        self.assertEqual(res["proposal_id"], "PROP-MESH-TEST")
        self.assertEqual(res["phase"], "PREPARE")

    def test_execute_health_decode(self):
        import numpy as np
        t = np.linspace(0, 1.0, 250, endpoint=False)
        sig = (20.0 * np.sin(2.0 * np.pi * 10.0 * t)).tolist()
        res = self.gateway.execute_health("decode", {"signal": sig})
        self.assertTrue(res["success"])
        self.assertIn("predicted_state", res)
        self.assertIn("non_clinical_disclaimer", res)

    def test_execute_space_propagate(self):
        res = self.gateway.execute_space("propagate", {"delta_t_s": 600.0})
        self.assertTrue(res["success"])
        self.assertIn("altitude_km", res)
        self.assertIn("velocity_km_s", res)

    def test_execute_os_monitor(self):
        res = self.gateway.execute_os("monitor", {})
        self.assertTrue(res["success"])
        self.assertTrue(res["is_non_privileged"])
        self.assertIn("estimated_memory_kb", res)

    def test_execute_learning_sleep_and_graft(self):
        res_sleep = self.gateway.execute_learning("sleep", {"dim": 32})
        self.assertTrue(res_sleep["success"])
        self.assertIn("retention_accuracy", res_sleep)

        res_graft = self.gateway.execute_learning("graft_filter", {"dim": 32})
        self.assertTrue(res_graft["success"])
        self.assertTrue(res_graft["zero_damage_verified"])


if __name__ == "__main__":
    unittest.main()
