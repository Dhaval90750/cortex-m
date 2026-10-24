import json
from pathlib import Path
import unittest

from cortex_m.legacy_v1.integrated import CortexMIMOSystem
from cortex_m.legacy_v1.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results" / "mimo-35d-full-checkpointed" / "checkpoints" / "seed-101.pt"
GIST_CHECKPOINT = ROOT / "results" / "mimo-gist-conditioned" / "checkpoints" / "seed-101.pt"


class MimoInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("Optional PyTorch environment required") from exc

    def test_checkpoint_inference_and_single_prompt_gist_orchestration(self):
        self.assertTrue(CHECKPOINT.exists(), f"missing trained checkpoint: {CHECKPOINT}")
        store = MemoryStore(":memory:")
        system = CortexMIMOSystem(keys=["name"], values=[], store=store,
                                  checkpoint_path=CHECKPOINT)
        try:
            system.observe("My name is Dhaval")
            recalled = system.run("what is my name?")
            self.assertEqual(recalled["response"], "Your recorded name is Dhaval.")
            prompt = json.dumps({"task": "mimo_n8", "gist_query": "what is my name",
                                 "values": [0.1, -0.4, 0.2, 0.5, -0.1, 0.7, -0.3, 0.8],
                                 "role_ids": [0, 1, 2, 3, 4, 5, 6, 7]})
            result = system.run(prompt)
            self.assertEqual(result["kind"], "integrated_mimo")
            self.assertEqual(result["gist"]["response"], "Your recorded name is Dhaval.")
            self.assertFalse(result["gist_used_for_computation"])
            self.assertEqual(len(result["solver_output"]["outputs"]), 1)
            self.assertEqual(len(result["solver_output"]["outputs"][0]), 4)
            self.assertEqual(len(result["solver_output"]["selected_physical_positions"][0]), 4)
            self.assertIn("M2M-SparseRoute outputs", result["response"])
        finally:
            system.close()

    def test_mimo_rejects_non_permutation_roles(self):
        store = MemoryStore(":memory:")
        system = CortexMIMOSystem(store=store, checkpoint_path=CHECKPOINT)
        try:
            result = system.run(json.dumps({"task": "mimo_n8", "values": [0] * 8,
                                           "role_ids": [0] * 8}))
            self.assertEqual(result["kind"], "error")
            self.assertIn("permutation", result["response"])
        finally:
            system.close()

    def test_retrieved_fact_conditions_routes_and_correction_changes_output(self):
        self.assertTrue(GIST_CHECKPOINT.exists(), f"missing gist checkpoint: {GIST_CHECKPOINT}")
        store = MemoryStore(":memory:")
        system = CortexMIMOSystem(keys=["mode"], values=["alpha", "beta"], store=store,
                                  gist_checkpoint_path=GIST_CHECKPOINT)
        request = {"task": "mimo_gist_n8", "gist_query": "what is my mode?",
                   "values": [0.1, -0.4, 0.2, 0.5, -0.1, 0.7, -0.3, 0.8],
                   "role_ids": [0, 1, 2, 3, 4, 5, 6, 7]}
        try:
            system.observe("My mode is alpha.")
            alpha = system.run(json.dumps(request))
            self.assertEqual(alpha["kind"], "integrated_gist_mimo")
            self.assertTrue(alpha["gist_used_for_computation"])
            self.assertEqual(alpha["stages"][-2]["data"]["conditioning_vector"], [1.0, 0.0])
            system.observe("Correction: set mode to beta, replacing alpha.")
            beta = system.run(json.dumps(request))
            self.assertEqual(beta["kind"], "integrated_gist_mimo")
            self.assertEqual(beta["stages"][-2]["data"]["conditioning_vector"], [0.0, 1.0])
            self.assertNotEqual(alpha["solver_output"]["selected_physical_positions"],
                                beta["solver_output"]["selected_physical_positions"])
            self.assertNotEqual(alpha["solver_output"]["outputs"], beta["solver_output"]["outputs"])
        finally:
            system.close()

    def test_missing_or_conflicted_gist_abstains_before_solver(self):
        store = MemoryStore(":memory:")
        system = CortexMIMOSystem(keys=["mode"], values=["alpha", "beta"], store=store,
                                  gist_checkpoint_path=GIST_CHECKPOINT)
        request = json.dumps({"task": "mimo_gist_n8", "values": [0] * 8,
                              "role_ids": list(range(8))})
        try:
            missing = system.run(request)
            self.assertEqual(missing["kind"], "abstain")
            system.observe("My mode is alpha.")
            system.observe("My mode is beta.")
            conflicted = system.run(request)
            self.assertEqual(conflicted["kind"], "abstain")
            self.assertFalse(conflicted["gist_used_for_computation"])
        finally:
            system.close()

    def test_variable_support_examples_form_dynamic_task_gist(self):
        from cortex_m.mimo.sparse_router import candidate_routes, evaluate_routes
        routes = candidate_routes(5)
        desired = ["sum(0,1)", "product(1,3)", "difference(4,2)", "affine(0,4)"]
        ids = [next(i for i, route in enumerate(routes) if route.label == label)
               for label in desired]
        gen = __import__("torch").Generator().manual_seed(441)
        support_x = __import__("torch").randn((32, 5), generator=gen)
        query_x = __import__("torch").randn((12, 5), generator=gen)
        support_y = evaluate_routes(support_x, routes)[:, ids]
        expected = evaluate_routes(query_x, routes)[:, ids]
        store = MemoryStore(":memory:")
        system = CortexMIMOSystem(store=store)
        try:
            result = system.run(json.dumps({"task": "support_gist_route",
                "support_x": support_x.tolist(), "support_y": support_y.tolist(),
                "query_x": query_x.tolist()}))
            self.assertEqual(result["kind"], "support_gist_route")
            self.assertTrue(result["gist_used_for_computation"])
            self.assertEqual(result["solver_output"]["route_labels"], desired)
            self.assertEqual(result["solver_output"]["support_example_count"], 32)
            self.assertTrue(__import__("torch").allclose(
                __import__("torch").tensor(result["solver_output"]["prediction"]), expected, atol=1e-5))
        finally:
            system.close()


if __name__ == "__main__":
    unittest.main()
