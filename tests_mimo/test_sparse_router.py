import unittest

import torch

from cortex_m.mimo.sparse_router import SparseEvidenceRouter, candidate_routes, evaluate_routes, sparsemax


class SparseRouterTests(unittest.TestCase):
    def test_candidate_count_and_labels(self):
        routes = candidate_routes(8)
        self.assertEqual(len(routes), 168)
        self.assertEqual(len({route.label for route in routes}), 168)

    def test_sparsemax_is_simplex_and_sparse(self):
        weights = sparsemax(torch.tensor([[5.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2)))
        self.assertEqual(int((weights[0] > 0).sum()), 1)
        self.assertEqual(int((weights[1] > 0).sum()), 3)

    def test_recovers_known_relations_from_support(self):
        routes = candidate_routes(5)
        route_ids = [next(i for i, r in enumerate(routes) if r.label == label)
                     for label in ("sum(0,1)", "product(1,3)", "difference(4,2)", "affine(0,4)")]
        gen = torch.Generator().manual_seed(7441)
        sx = torch.randn((64, 5), generator=gen)
        qx = torch.randn((80, 5), generator=gen)
        support_experts = evaluate_routes(sx, routes)
        query_experts = evaluate_routes(qx, routes)
        sy = support_experts[:, route_ids]
        expected = query_experts[:, route_ids]
        result = SparseEvidenceRouter(temperature=0.01).route(sx, sy, qx)
        self.assertTrue(torch.equal(result.route_indices, torch.tensor(route_ids)))
        self.assertTrue(torch.allclose(result.prediction, expected, atol=1e-5, rtol=1e-5))

    def test_routes_are_equivariant_to_consistent_input_permutation(self):
        routes = candidate_routes(5)
        labels = [next(i for i, r in enumerate(routes) if r.label == label)
                  for label in ("sum(0,1)", "product(1,3)", "difference(4,2)", "affine(0,4)")]
        gen = torch.Generator().manual_seed(901)
        sx0 = torch.randn((48, 5), generator=gen)
        qx0 = torch.randn((32, 5), generator=gen)
        sy = evaluate_routes(sx0, routes)[:, labels]
        qy = evaluate_routes(qx0, routes)[:, labels]
        permutation = torch.tensor([3, 0, 4, 1, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(5)
        sx, qx = sx0[:, permutation], qx0[:, permutation]
        permuted_labels = []
        for index in labels:
            old = routes[index]
            a, b = int(inverse[old.left]), int(inverse[old.right])
            permuted_labels.append(next(i for i, r in enumerate(routes)
                                        if r.operation == old.operation
                                        and ((r.left, r.right) == (a, b)
                                             or (old.operation in ("sum", "product")
                                                 and (r.left, r.right) == (b, a)))))
        result = SparseEvidenceRouter().route(sx, sy, qx)
        self.assertTrue(torch.equal(result.route_indices, torch.tensor(permuted_labels)))
        self.assertTrue(torch.allclose(result.prediction, qy, atol=1e-5, rtol=1e-5))

    def test_bad_shapes_fail_explicitly(self):
        router = SparseEvidenceRouter()
        with self.assertRaises(ValueError):
            router.route(torch.ones(2, 3), torch.ones(2), torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
