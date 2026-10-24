from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LinearRegressionModel:
    """Ordinary least squares with train-fitted feature standardization."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients_standardized: tuple[float, ...]
    intercept: float

    @property
    def raw_coefficients(self) -> tuple[float, ...]:
        return tuple(coef / scale for coef, scale in
                     zip(self.coefficients_standardized, self.scales))

    @property
    def raw_intercept(self) -> float:
        return self.intercept - sum(coef * mean for coef, mean in
                                    zip(self.raw_coefficients, self.means))

    def predict_one(self, values: Sequence[float]) -> float:
        if len(values) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} feature values; got {len(values)}")
        normalized = [(float(v) - mean) / scale
                      for v, mean, scale in zip(values, self.means, self.scales)]
        return self.intercept + sum(coef * value for coef, value in
                                    zip(self.coefficients_standardized, normalized))

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_one(row) for row in rows]

    def to_dict(self) -> dict:
        return {"algorithm": "ordinary_least_squares",
                "feature_names": list(self.feature_names),
                "means": list(self.means), "scales": list(self.scales),
                "coefficients_standardized": list(self.coefficients_standardized),
                "intercept": self.intercept,
                "raw_coefficients": list(self.raw_coefficients),
                "raw_intercept": self.raw_intercept}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearRegressionModel":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("algorithm") != "ordinary_least_squares":
            raise ValueError("Unsupported model artifact algorithm")
        return cls(tuple(obj["feature_names"]), tuple(obj["means"]),
                   tuple(obj["scales"]), tuple(obj["coefficients_standardized"]),
                   float(obj["intercept"]))


def fit_linear_regression(features: Sequence[Sequence[float]],
                          target: Sequence[float],
                          feature_names: Sequence[str]) -> LinearRegressionModel:
    """Fit OLS by solving standardized normal equations with pivoting.

    This first implementation intentionally supports dense numeric data only.
    Householder QR is used to avoid the worse conditioning of normal equations.
    Linearly dependent features are rejected instead of silently regularized.
    """
    if len(features) != len(target):
        raise ValueError("Feature rows and target values must have the same length")
    if not features:
        raise ValueError("Training data is empty")
    width = len(feature_names)
    if width == 0 or any(len(row) != width for row in features):
        raise ValueError("Every row must contain one numeric value per feature")
    if len(features) < width + 1:
        raise ValueError(f"Need at least {width + 1} training rows for {width} features and an intercept")
    if any(not math.isfinite(float(value)) for row in features for value in row):
        raise ValueError("Features must be finite numbers")
    if any(not math.isfinite(float(value)) for value in target):
        raise ValueError("Target values must be finite numbers")

    means = [sum(float(row[j]) for row in features) / len(features)
             for j in range(width)]
    scales = []
    for j, mean in enumerate(means):
        variance = sum((float(row[j]) - mean) ** 2 for row in features) / len(features)
        scale = math.sqrt(variance)
        if scale <= 1e-14:
            raise ValueError(f"Feature {feature_names[j]!r} is constant in the training split")
        scales.append(scale)
    design = [[1.0] + [(float(row[j]) - means[j]) / scales[j]
                       for j in range(width)] for row in features]
    solution = _least_squares_qr(design, [float(y) for y in target])
    return LinearRegressionModel(tuple(feature_names), tuple(means), tuple(scales),
                                 tuple(solution[1:]), solution[0])


def _least_squares_qr(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Householder QR least-squares solve for a tall full-rank design matrix."""
    rows, columns = len(matrix), len(matrix[0])
    if rows < columns:
        raise ValueError("Training split needs at least as many rows as model parameters")
    a = [list(row) for row in matrix]
    b = list(vector)
    scale = max((abs(value) for row in a for value in row), default=1.0)
    tolerance = max(1e-14, scale * max(rows, columns) * 1e-13)
    for k in range(columns):
        norm = math.sqrt(sum(a[i][k] * a[i][k] for i in range(k, rows)))
        if norm <= tolerance:
            raise ValueError("Training features are linearly dependent; remove redundant columns")
        alpha = -math.copysign(norm, a[k][k])
        reflector = [a[i][k] for i in range(k, rows)]
        reflector[0] -= alpha
        reflector_norm = sum(value * value for value in reflector)
        if reflector_norm <= tolerance * tolerance:
            raise ValueError("Training features are linearly dependent; remove redundant columns")
        factor = 2.0 / reflector_norm
        for j in range(k, columns):
            projection = factor * sum(reflector[i-k] * a[i][j]
                                      for i in range(k, rows))
            for i in range(k, rows):
                a[i][j] -= projection * reflector[i-k]
        projection = factor * sum(reflector[i-k] * b[i] for i in range(k, rows))
        for i in range(k, rows):
            b[i] -= projection * reflector[i-k]

    solution = [0.0] * columns
    for i in range(columns - 1, -1, -1):
        diagonal = a[i][i]
        if abs(diagonal) <= tolerance:
            raise ValueError("Training features are linearly dependent; remove redundant columns")
        remainder = sum(a[i][j] * solution[j] for j in range(i + 1, columns))
        solution[i] = (b[i] - remainder) / diagonal
    return solution


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metrics need equal, non-empty actual and predicted values")
    errors = [float(p) - float(y) for y, p in zip(actual, predicted)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mean = sum(actual) / len(actual)
    total = sum((float(y) - mean) ** 2 for y in actual)
    residual = sum(e * e for e in errors)
    r2 = 1.0 - residual / total if total > 0 else (1.0 if residual == 0 else 0.0)
    return {"mae": mae, "rmse": rmse, "r2": r2, "rows": len(actual)}
