"""
Comprehensive Integration Tests for EcosystemGateway and Universal Platform Routing.
Verifies all 9 platform dispatchers, error resilience, and learning pipelines.
"""

import unittest
import sys
from pathlib import Path
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

    def test_cli_entrypoints_exist(self):
        import cortex_m.cli as cli
        for ep in ["cortex_unified_main", "cortex_studio_main", "cortex_code_main", "cortex_substrate_main", "cortex_lm_main"]:
            self.assertTrue(hasattr(cli, ep), f"Missing CLI entry point {ep}")

    def test_all_gateway_action_aliases(self):
        # 1. Personal
        self.assertTrue(self.gateway.execute_personal("vault_status", {})["success"])
        # 2. IDE
        self.assertTrue(self.gateway.execute_ide("parse_ast", {"code": "x = 42"})["success"])
        diff_res = self.gateway.execute_ide("diff", {"original_code": "x=1", "modified_code": "x=2"})
        self.assertTrue(diff_res["success"])
        self.assertEqual(diff_res["line_changes"], 2)
        # 3. Fin
        replay_res = self.gateway.execute_fin("replay", {"count": 10})
        self.assertTrue(replay_res["success"])
        self.assertEqual(replay_res["total_transactions"], 10)
        # 4. Robotics
        self.assertTrue(self.gateway.execute_robotics("safety_check", {})["success"])
        self.assertTrue(self.gateway.execute_robotics("plan_path", {})["success"])
        # 5. Mesh
        self.assertTrue(self.gateway.execute_mesh("consensus_round", {"proposal_id": "P_TEST"})["success"])
        self.assertTrue(self.gateway.execute_mesh("route_task", {"task_id": "T_TEST"})["success"])
        # 6. Health
        self.assertTrue(self.gateway.execute_health("filter_signal", {})["success"])
        self.assertTrue(self.gateway.execute_health("wavelet_bandpower", {})["success"])
        # 7. Space
        self.assertTrue(self.gateway.execute_space("propagate_orbit", {})["success"])
        self.assertTrue(self.gateway.execute_space("supervise_telemetry", {})["success"])
        self.assertTrue(self.gateway.execute_space("predict_passes", {"duration_s": 3600.0})["success"])

        # 8. All newly upgraded features
        # Fin explain & velocity
        exp = self.gateway.execute_fin("explain", {"amount": 5000.0, "distance_km": 800.0, "mcc": 7995})
        self.assertTrue(exp["success"])
        self.assertIn("adverse_action_codes", exp)
        vel = self.gateway.execute_fin("velocity_anomalies", {})
        self.assertTrue(vel["success"])
        self.assertGreaterEqual(vel["num_anomalies"], 1)

        # Health snr, electrode_check, welch_psd, decode_stream
        snr_res = self.gateway.execute_health("snr", {})
        self.assertTrue(snr_res["success"])
        self.assertIn("snr_db", snr_res)
        leads_res = self.gateway.execute_health("electrode_check", {})
        self.assertTrue(leads_res["success"])
        self.assertTrue(leads_res["all_leads_contact_good"])
        psd_res = self.gateway.execute_health("welch_psd", {})
        self.assertTrue(psd_res["success"])
        self.assertIn("peak_frequency_hz", psd_res)
        stream_res = self.gateway.execute_health("decode_stream", {})
        self.assertTrue(stream_res["success"])

        # Mesh view_change & vector_clock
        vc_res = self.gateway.execute_mesh("view_change", {"new_view": 2})
        self.assertTrue(vc_res["success"])
        clock_res = self.gateway.execute_mesh("vector_clock", {"node_id": "test_node"})
        self.assertTrue(clock_res["success"])

        # OS cluster & predict_cluster
        cluster_res = self.gateway.execute_os("cluster", {})
        self.assertTrue(cluster_res["success"])
        self.assertIn("newman_girvan_modularity", cluster_res)
        pred_cl = self.gateway.execute_os("predict_cluster", {"text": "orbit telemetry"})
        self.assertTrue(pred_cl["success"])

        # IDE atomic_patch & cyclomatic complexity
        audit_res = self.gateway.execute_ide("audit", {"code": "def branch(x):\n    if x > 0:\n        return 1\n    return 0"})
        self.assertTrue(audit_res["success"])
        self.assertIn("functions_meta", audit_res)
        self.assertGreaterEqual(audit_res["functions_meta"]["branch"]["cyclomatic_complexity"], 2)
        atomic_res = self.gateway.execute_ide("atomic_patch", {"bundle": {"scratch/test_patch.py": "def f(): return 10\n"}})
        self.assertTrue(atomic_res["success"])

        # Personal paging & footprint
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            slice_path = Path(tmpdir) / "slice.pt"
            po_res = self.gateway.execute_personal("page_out", {"tenant_id": "tenant_test", "path": str(slice_path)})
            self.assertTrue(po_res["success"])
            self.assertTrue(slice_path.exists())
            pi_res = self.gateway.execute_personal("page_in", {"path": str(slice_path)})
            self.assertTrue(pi_res["success"])
        fp_res = self.gateway.execute_personal("footprint", {})
        self.assertTrue(fp_res["success"])

    def test_studio_server_static_and_platform_api(self):
        import urllib.request
        import json
        import threading
        import time
        from cortex_m.server.app import ThreadingHTTPServer, CortexStudioHandler

        # Bind to ephemeral port
        server = ThreadingHTTPServer(('127.0.0.1', 0), CortexStudioHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.3)

        try:
            # 1. GET /
            req = urllib.request.urlopen(f'http://127.0.0.1:{port}/')
            html = req.read().decode('utf-8')
            self.assertEqual(req.status, 200)
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("CORTEX-M Studio", html)

            # 2. GET /studio.css
            req_css = urllib.request.urlopen(f'http://127.0.0.1:{port}/studio.css')
            self.assertEqual(req_css.status, 200)
            self.assertIn("text/css", req_css.headers.get("Content-Type"))

            # 3. GET /studio.js
            req_js = urllib.request.urlopen(f'http://127.0.0.1:{port}/studio.js')
            self.assertEqual(req_js.status, 200)
            self.assertIn("javascript", req_js.headers.get("Content-Type"))

            # 4. POST /api/platforms/fin/score
            body = json.dumps({"amount": 100.0, "mcc": 5411, "distance_km": 2.0, "time_delta_seconds": 60.0}).encode('utf-8')
            req_post = urllib.request.Request(f'http://127.0.0.1:{port}/api/platforms/fin/score', data=body, headers={'Content-Type': 'application/json'})
            res_post = urllib.request.urlopen(req_post)
            self.assertEqual(res_post.status, 200)
            data = json.loads(res_post.read().decode('utf-8'))
            self.assertTrue(data["success"])
            self.assertIn("result", data)
            self.assertIn("decision", data["result"])

            # 5. POST /api/memory/learn with tenant and explicit memory_id (Flaw A verification)
            learn_tenant_body = json.dumps({
                "text": "Isolated tenant secret credential",
                "tenant_id": "tenant_suite_42",
                "is_tenant": True,
                "memory_id": "mem_tenant_custom_99",
                "salience": 0.8
            }).encode('utf-8')
            req_learn = urllib.request.Request(f'http://127.0.0.1:{port}/api/memory/learn', data=learn_tenant_body, headers={'Content-Type': 'application/json'})
            res_learn = urllib.request.urlopen(req_learn)
            self.assertEqual(res_learn.status, 200)
            learn_data = json.loads(res_learn.read().decode('utf-8'))
            self.assertTrue(learn_data["success"])
            self.assertEqual(learn_data["result"]["tenant_id"], "tenant_suite_42")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
