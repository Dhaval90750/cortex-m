from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cortex_m.learning.dataset import load_feature_csv, load_numeric_csv
from cortex_m.learning.experiment import run_linear_regression
from cortex_m.learning.regression import LinearRegressionModel

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run offline classical Cortex-M models.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train an ordinary least-squares regression model from a numeric CSV.")
    train.add_argument("csv", type=Path)
    train.add_argument("--target", required=True)
    train.add_argument("--features", help="Comma-separated feature columns; defaults to every non-target column.")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=20261023)
    train.add_argument("--output", type=Path, default=ROOT / "results" / "learning")
    predict = sub.add_parser("predict", help="Predict numeric targets with a saved Cortex-M regression model.")
    predict.add_argument("model", type=Path)
    predict.add_argument("csv", type=Path)
    predict.add_argument("--output", type=Path, help="Prediction CSV path; defaults beside the model.")
    args = parser.parse_args()

    if args.command == "train":
        feature_names = tuple(x.strip() for x in args.features.split(",") if x.strip()) if args.features else None
        data = load_numeric_csv(args.csv, args.target, feature_names)
        result = run_linear_regression(data, args.output, args.test_size, args.seed)
        report, folder = result["report"], Path(result["run_dir"])
        print(f"Trained ordinary least squares on {report['split']['train_rows']} rows; evaluated on {report['split']['test_rows']} held-out rows.")
        print(f"Test MAE: {report['metrics']['test']['mae']:.6g}")
        print(f"Test RMSE: {report['metrics']['test']['rmse']:.6g}")
        print(f"Test R²: {report['metrics']['test']['r2']:.6g}")
        print(f"Model: {folder / 'model.json'}")
        print(f"Report: {folder / 'report.md'}")
    else:
        model = LinearRegressionModel.load(args.model)
        rows, originals = load_feature_csv(args.csv, model.feature_names)
        predictions = model.predict(rows)
        output = args.output or args.model.parent / f"{args.csv.stem}_predictions.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*originals[0].keys(), "prediction"])
            writer.writeheader()
            for row, prediction in zip(originals, predictions):
                writer.writerow({**row, "prediction": f"{prediction:.12g}"})
        print(f"Predicted {len(predictions)} row(s). Output: {output.resolve()}")


if __name__ == "__main__":
    main()
