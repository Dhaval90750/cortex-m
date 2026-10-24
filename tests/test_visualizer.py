from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from scripts import visualizer


class VisualizerSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="cortex-m-http-")
        self.db_path = Path(self.temp.name) / "memory.sqlite3"
        self.db_patch = patch.object(visualizer, "DB_PATH", self.db_path)
        self.db_patch.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), visualizer.Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.db_patch.stop()
        self.temp.cleanup()

    def request(self, path: str, *, payload: object | None = None,
                headers: dict[str, str] | None = None) -> tuple[int, dict]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base + path, data=body, headers=headers or {})
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_same_origin_real_api_stores_and_recalls_fact(self) -> None:
        headers = {"Content-Type": "application/json", "Origin": self.base}
        status, stored = self.request("/api/task", payload={
            "mode": "statement", "text": "my name is dhaval"
        }, headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(stored["accepted_count"], 1)
        status, answer = self.request("/api/task", payload={
            "mode": "question", "text": "what's my name?"
        }, headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(answer["response"], "Your recorded name is dhaval.")

    def test_single_prompt_auto_routes_statement_then_question(self) -> None:
        headers = {"Content-Type": "application/json", "Origin": self.base}
        status, stored = self.request("/api/task", payload={
            "mode": "auto", "text": "my name is dhaval"
        }, headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(stored["kind"], "statement")
        self.assertEqual(stored["accepted_count"], 1)
        status, answer = self.request("/api/task", payload={
            "mode": "auto", "text": "what is my name"
        }, headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(answer["kind"], "question")
        self.assertEqual(answer["response"], "Your recorded name is dhaval.")

    def test_optional_mimo_checkpoint_is_reachable_from_the_same_composer(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Optional CPU PyTorch environment required for MIMO UI route")
        checkpoint = (Path(__file__).resolve().parents[1] / "results" /
                      "mimo-35d-full-checkpointed" / "checkpoints" / "seed-101.pt")
        with patch.object(visualizer, "MIMO_CHECKPOINT", checkpoint):
            status, result = self.request("/api/task", payload={
                "mode": "auto", "text": json.dumps({"task": "mimo_n8",
                    "gist_query": "what is my name", "values": [0.1, -0.4, 0.2, 0.5, -0.1, 0.7, -0.3, 0.8],
                    "role_ids": [0, 1, 2, 3, 4, 5, 6, 7]})
            }, headers={"Content-Type": "application/json", "Origin": self.base})
        self.assertEqual(status, 200)
        self.assertEqual(result["kind"], "integrated_mimo")
        self.assertEqual(len(result["solver_output"]["outputs"][0]), 4)
        self.assertFalse(result["gist_used_for_computation"])

    def test_retrieved_gist_mimo_runs_through_the_same_composer(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Optional CPU PyTorch environment required for gist-conditioned MIMO UI route")
        checkpoint = (Path(__file__).resolve().parents[1] / "results" /
                      "mimo-gist-conditioned" / "checkpoints" / "seed-101.pt")
        with patch.object(visualizer, "MIMO_GIST_CHECKPOINT", checkpoint):
            headers = {"Content-Type": "application/json", "Origin": self.base}
            status, stored = self.request("/api/task", payload={"mode": "auto", "text": "My mode is alpha."}, headers=headers)
            self.assertEqual(status, 200)
            self.assertEqual(stored["accepted_count"], 1)
            status, result = self.request("/api/task", payload={
                "mode": "auto", "text": json.dumps({"task": "mimo_gist_n8", "gist_query": "what is my mode?",
                    "values": [0.1, -0.4, 0.2, 0.5, -0.1, 0.7, -0.3, 0.8],
                    "role_ids": [0, 1, 2, 3, 4, 5, 6, 7]})
            }, headers=headers)
        self.assertEqual(status, 200)
        self.assertEqual(result["kind"], "integrated_gist_mimo")
        self.assertTrue(result["gist_used_for_computation"])
        self.assertEqual(result["stages"][-2]["data"]["conditioning_vector"], [1.0, 0.0])

    def test_single_prompt_is_honest_about_unsupported_code_generation(self) -> None:
        status, result = self.request("/api/task", payload={
            "mode": "auto", "text": "Write a Python function that computes factorial."
        }, headers={"Content-Type": "application/json", "Origin": self.base})
        self.assertEqual(status, 200)
        self.assertEqual(result["kind"], "unsupported")
        self.assertIn("does not provide general chat", result["response"])
        self.assertIn("generate or execute code", result["response"])

    def test_visualizer_contains_one_composer_and_no_mode_switch(self) -> None:
        page = visualizer.HTML.read_text(encoding="utf-8")
        self.assertEqual(page.count('id="task"'), 1)
        self.assertNotIn('class="mode', page)

    def test_cross_origin_write_is_rejected_without_mutating_memory(self) -> None:
        status, _ = self.request("/api/task", payload={
            "mode": "statement", "text": "my name is attacker"
        }, headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        status, state = self.request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(state["facts"], [])

    def test_host_header_outside_loopback_is_rejected(self) -> None:
        status, _ = self.request("/api/state", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_non_json_post_is_rejected(self) -> None:
        request = Request(self.base + "/api/task", data=b'{"mode":"statement"}',
                          headers={"Content-Type": "text/plain"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=3)
        self.assertEqual(caught.exception.code, 415)

    def test_oversized_post_is_rejected(self) -> None:
        body = b" " * 20_001
        request = Request(self.base + "/api/task", data=body,
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=3)
        self.assertEqual(caught.exception.code, 413)


if __name__ == "__main__":
    unittest.main()
