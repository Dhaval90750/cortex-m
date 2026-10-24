from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NumericDataset:
    path: str
    feature_names: tuple[str, ...]
    target_name: str
    features: tuple[tuple[float, ...], ...]
    target: tuple[float, ...]

    @property
    def row_count(self) -> int:
        return len(self.target)


def load_numeric_csv(path: str | Path, target: str,
                     features: list[str] | tuple[str, ...] | None = None) -> NumericDataset:
    """Load a CSV with numeric features and target; reject ambiguous/missing data."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must have a header row")
        headers = [name.strip() for name in reader.fieldnames]
        reader.fieldnames = headers
        if len(set(headers)) != len(headers):
            raise ValueError("CSV column names must be unique")
        if target not in headers:
            raise ValueError(f"Target column {target!r} is not in the CSV")
        names = tuple(features) if features is not None else tuple(h for h in headers if h != target)
        if not names:
            raise ValueError("Select at least one feature column")
        if target in names:
            raise ValueError("Target column cannot also be used as a feature")
        if len(set(names)) != len(names):
            raise ValueError("Feature column names must be unique")
        missing = [name for name in (*names, target) if name not in headers]
        if missing:
            raise ValueError(f"Columns not found in CSV: {', '.join(missing)}")
        xs: list[tuple[float, ...]] = []
        ys: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            if not row or all(value is None or not value.strip() for value in row.values()):
                continue
            values = []
            for column in names:
                values.append(_number(row.get(column), column, row_number))
            ys.append(_number(row.get(target), target, row_number))
            xs.append(tuple(values))
    if len(xs) < 3:
        raise ValueError("At least three complete data rows are required for a train/test experiment")
    return NumericDataset(str(source.resolve()), names, target, tuple(xs), tuple(ys))


def _number(raw: str | None, column: str, row_number: int) -> float:
    if raw is None or not raw.strip():
        raise ValueError(f"Missing value in column {column!r} at CSV row {row_number}")
    try:
        number = float(raw)
    except ValueError as exc:
        raise ValueError(f"Non-numeric value {raw!r} in column {column!r} at CSV row {row_number}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value in column {column!r} at CSV row {row_number}")
    return number


def split_indices(row_count: int, test_size: float = 0.2,
                  seed: int = 20261023) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if row_count < 3:
        raise ValueError("Need at least three rows to split train and test data")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1")
    test_count = min(row_count - 1, max(1, round(row_count * test_size)))
    indexes = list(range(row_count))
    random.Random(seed).shuffle(indexes)
    return tuple(indexes[test_count:]), tuple(indexes[:test_count])


def load_feature_csv(path: str | Path, feature_names: tuple[str, ...] | list[str]) -> tuple[list[tuple[float, ...]], list[dict[str, str]]]:
    """Load inference rows using exactly the feature columns stored in a model."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [name.strip() for name in (reader.fieldnames or [])]
        reader.fieldnames = headers
        missing = [name for name in feature_names if name not in headers]
        if missing:
            raise ValueError(f"Feature columns not found in CSV: {', '.join(missing)}")
        rows: list[tuple[float, ...]] = []
        originals: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if not row or all(value is None or not value.strip() for value in row.values()):
                continue
            rows.append(tuple(_number(row.get(name), name, row_number) for name in feature_names))
            originals.append({str(k): str(v or "") for k, v in row.items()})
    if not rows:
        raise ValueError("Inference CSV has no data rows")
    return rows, originals
