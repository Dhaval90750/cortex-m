from __future__ import annotations

import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

from .dataset import NumericDataset, split_indices
from .regression import fit_linear_regression, regression_metrics


def run_linear_regression(dataset: NumericDataset, output_root: str | Path,
                          test_size: float = 0.2, seed: int = 20261023) -> dict:
    train_idx, test_idx = split_indices(dataset.row_count, test_size, seed)
    width = len(dataset.feature_names)
    if len(train_idx) < width + 1:
        raise ValueError(f"Training split has {len(train_idx)} rows; at least {width + 1} are needed")
    train_x = [dataset.features[i] for i in train_idx]
    train_y = [dataset.target[i] for i in train_idx]
    test_x = [dataset.features[i] for i in test_idx]
    test_y = [dataset.target[i] for i in test_idx]
    model = fit_linear_regression(train_x, train_y, dataset.feature_names)
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)
    train_mean = sum(train_y) / len(train_y)
    baseline_predictions = [train_mean] * len(test_y)
    now = datetime.now(timezone.utc)
    run_id = "linear-regression-" + now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model.save(run_dir / "model.json")
    report = {
        "run_id": run_id,
        "created_utc": now.isoformat(),
        "algorithm": "ordinary_least_squares",
        "dataset": {"path": dataset.path, "sha256": hashlib.sha256(Path(dataset.path).read_bytes()).hexdigest(),
                    "rows": dataset.row_count,
                    "features": list(dataset.feature_names), "target": dataset.target_name},
        "runtime": {"python": platform.python_version(), "dependencies": "Python standard library only"},
        "split": {"train_rows": len(train_idx), "test_rows": len(test_idx),
                  "test_size": test_size, "seed": seed,
                  "train_indices": list(train_idx), "test_indices": list(test_idx)},
        "metrics": {"train": regression_metrics(train_y, train_predictions),
                    "test": regression_metrics(test_y, test_predictions),
                    "mean_baseline_test": regression_metrics(test_y, baseline_predictions)},
        "model": model.to_dict(),
        "predictions": [{"row_index": index, "actual": dataset.target[index],
                         "predicted": prediction,
                         "error": prediction - dataset.target[index]}
                        for index, prediction in zip(test_idx, test_predictions)],
        "interpretation": "Test metrics are computed on rows excluded from model fitting. This result is valid only for the supplied dataset and split; it is not a general performance guarantee.",
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    return {"run_dir": str(run_dir.resolve()), "report": report}


def render_report(report: dict) -> str:
    metrics = report["metrics"]
    lines = [f"# Linear regression experiment: {report['run_id']}", "",
             f"Created: {report['created_utc']}", "",
             "## Dataset", "",
             f"- File: `{report['dataset']['path']}`",
             f"- SHA-256: `{report['dataset']['sha256']}`",
             f"- Rows: {report['dataset']['rows']}",
             f"- Features: {', '.join(report['dataset']['features'])}",
             f"- Target: {report['dataset']['target']}", "",
             "## Split", "",
             f"- Train/test rows: {report['split']['train_rows']} / {report['split']['test_rows']}",
             f"- Test fraction: {report['split']['test_size']}",
             f"- Random seed: {report['split']['seed']}", "",
             "## Metrics", "",
             "| Set | MAE | RMSE | R² | Rows |", "|---|---:|---:|---:|---:|"]
    for name in ("train", "test", "mean_baseline_test"):
        m = metrics[name]
        label = "mean baseline (test)" if name == "mean_baseline_test" else name
        lines.append(f"| {label} | {m['mae']:.6g} | {m['rmse']:.6g} | {m['r2']:.6g} | {m['rows']} |")
    lines += ["", "## Learned model", "",
              f"Intercept: `{report['model']['intercept']:.8g}`", "",
              "| Feature | Train mean | Train scale | Standardized coefficient |",
              "|---|---:|---:|---:|"]
    for feature, mean, scale, coef in zip(
            report["model"]["feature_names"], report["model"]["means"],
            report["model"]["scales"], report["model"]["coefficients_standardized"]):
        lines.append(f"| {feature} | {mean:.8g} | {scale:.8g} | {coef:.8g} |")
    lines += ["", "## Notes", "", report["interpretation"],
              "", "Raw-unit equation: `prediction = " + f"{report['model']['raw_intercept']:.8g}" + " + " + " + ".join(
                  f"({coef:.8g} × {name})" for name, coef in
                  zip(report["model"]["feature_names"], report["model"]["raw_coefficients"])) + "`",
              "", "Model parameters are saved in `model.json`; per-test-row predictions and errors are saved in `report.json`.", ""]
    return "\n".join(lines)
