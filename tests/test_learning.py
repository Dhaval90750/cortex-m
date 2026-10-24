from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cortex_m.learning.dataset import load_numeric_csv, split_indices
from cortex_m.learning.experiment import run_linear_regression
from cortex_m.learning.regression import LinearRegressionModel, fit_linear_regression, regression_metrics


class DatasetTests(unittest.TestCase):
    def test_csv_load_and_feature_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            path.write_text("x,z,y\n1,10,3\n2,20,5\n3,30,7\n", encoding="utf-8")
            data = load_numeric_csv(path, "y", ["x"])
            self.assertEqual(data.feature_names, ("x",))
            self.assertEqual(data.features, ((1.0,), (2.0,), (3.0,)))
            self.assertEqual(data.target, (3.0, 5.0, 7.0))

    def test_csv_rejects_missing_and_non_numeric_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            path.write_text("x,y\n1,2\n,3\n4,5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing value"):
                load_numeric_csv(path, "y")
            path.write_text("x,y\n1,2\nabc,3\n4,5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-numeric"):
                load_numeric_csv(path, "y")
            with self.assertRaisesRegex(ValueError, "Target column cannot"):
                load_numeric_csv(path, "y", ["x", "y"])

    def test_split_is_repeatable_and_disjoint(self):
        train_a, test_a = split_indices(25, 0.2, 17)
        train_b, test_b = split_indices(25, 0.2, 17)
        self.assertEqual((train_a, test_a), (train_b, test_b))
        self.assertFalse(set(train_a) & set(test_a))
        self.assertEqual(set(train_a) | set(test_a), set(range(25)))


class RegressionTests(unittest.TestCase):
    def test_recovers_exact_simple_linear_relationship(self):
        x = [[float(i)] for i in range(10)]
        y = [5.0 + 2.0 * row[0] for row in x]
        model = fit_linear_regression(x, y, ["x"])
        self.assertAlmostEqual(model.predict_one([12.0]), 29.0, places=9)
        self.assertAlmostEqual(model.intercept, 14.0, places=9)
        self.assertAlmostEqual(model.coefficients_standardized[0], 5.7445626465, places=8)
        self.assertAlmostEqual(model.raw_coefficients[0], 2.0, places=9)
        self.assertAlmostEqual(model.raw_intercept, 5.0, places=9)

    def test_recovers_multivariate_linear_relationship(self):
        x = [[float(i), float((i * i) % 11)] for i in range(30)]
        y = [7.0 + 1.5 * row[0] - 3.25 * row[1] for row in x]
        model = fit_linear_regression(x, y, ["a", "b"])
        for row, expected in zip(x, y):
            self.assertAlmostEqual(model.predict_one(row), expected, places=8)

    def test_rejects_rank_deficient_or_too_small_training_set(self):
        with self.assertRaisesRegex(ValueError, "linearly dependent"):
            fit_linear_regression([[float(i), float(i * 2)] for i in range(6)],
                                  [float(i) for i in range(6)], ["x", "two_x"])
        with self.assertRaisesRegex(ValueError, "at least 3 training rows"):
            fit_linear_regression([[1, 2], [2, 3]], [1, 2], ["a", "b"])

    def test_model_artifact_round_trip(self):
        x = [[float(i)] for i in range(8)]
        y = [4.0 - 0.75 * row[0] for row in x]
        model = fit_linear_regression(x, y, ["x"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            model.save(path)
            loaded = LinearRegressionModel.load(path)
            self.assertEqual(loaded.feature_names, model.feature_names)
            self.assertAlmostEqual(loaded.predict_one([10]), model.predict_one([10]), places=12)

    def test_metrics(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], (2 / 3) ** 0.5)
        self.assertEqual(metrics["rows"], 3)

    def test_experiment_saves_model_report_and_held_out_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "training.csv"
            csv_path.write_text("x,y\n" + "".join(f"{i},{4 + 3*i}\n" for i in range(30)), encoding="utf-8")
            data = load_numeric_csv(csv_path, "y")
            result = run_linear_regression(data, Path(temp) / "results", 0.25, 7)
            folder = Path(result["run_dir"])
            self.assertTrue((folder / "model.json").exists())
            self.assertTrue((folder / "report.json").exists())
            self.assertTrue((folder / "report.md").exists())
            report = result["report"]
            self.assertEqual(report["metrics"]["test"]["rows"], 8)
            self.assertGreater(report["metrics"]["mean_baseline_test"]["mae"], report["metrics"]["test"]["mae"])
            self.assertLess(report["metrics"]["test"]["mae"], 1e-8)
            self.assertEqual(len(report["predictions"]), 8)
            self.assertFalse(set(report["split"]["train_indices"]) & set(report["split"]["test_indices"]))


if __name__ == "__main__":
    unittest.main()
