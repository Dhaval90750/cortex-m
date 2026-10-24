# Cortex-M learning and task system

## Goal

Extend Cortex-M with offline, inspectable classical learning tools. Cortex-M's
existing fact memory remains a separate feature: the memory store tracks facts,
source evidence, corrections, and conflicts; learning modules fit parameters
from tabular examples and predict target values. No generative model, hosted
AI API, or network dependency is required.

## Shared system contract

Every task module should follow the same experiment lifecycle:

1. Load data and validate columns, types, missing values, and row counts.
2. Record the source path, SHA-256, task, feature names, target, and settings.
3. Split data reproducibly before fitting any learned preprocessing.
4. Fit only on training rows; evaluate only on held-out rows.
5. Save the model in a documented portable format, alongside metrics,
   per-row predictions/errors, configuration, runtime, and interpretation.
6. Support loading a saved artifact for predictions on new examples.
7. Reject malformed or unsupported inputs with actionable messages.

Training artifacts go under `results/learning/<run-id>/`. Raw user datasets are
not copied into the result folder; their path and content hash are recorded.

## Capability phases

| Phase | Capability | Expected input | Primary evaluation | Status |
|---|---|---|---|---|
| 0 | Shared numeric CSV loading, deterministic splitting, experiment artifacts | CSV; declared target | Validation tests, repeatable split, no preprocessing leakage | Complete |
| 1 | Ordinary least-squares linear regression | Numeric features and numeric target | MAE, RMSE, R² on held-out data | Complete as v1; full 44-test suite passes; saved train-and-predict demonstration |
| 2 | Classification (logistic regression and naive Bayes first) | Features and category labels | Accuracy, precision/recall, confusion matrix; stratified split | Planned |
| 3 | Additional regression (ridge; then nonlinear trees if justified) | Numeric features and target | Held-out regression metrics and overfit comparison | Planned |
| 4 | Clustering (k-means) | Numeric features, labels optional | Inertia, stability, human-readable cluster summaries | Planned |
| 5 | Forecasting (naive, moving average, trend; seasonality later) | Ordered timestamps and values | Time-ordered holdout; MAE/RMSE vs naive baseline | Planned |
| 6 | Anomaly detection (z-score and robust IQR baseline) | Numeric baseline observations | Synthetic injected anomalies; false-positive rate | Planned |
| 7 | Similarity lookup | Feature rows and historical cases | Neighbor ranking and distance checks | Planned |
| 8 | Recommendation | Labeled ratings or item-feature matrix | Leave-one-out ranking metrics and cold-start reporting | Planned |
| 9 | Rules/workflows and feedback/retraining | Explicit policy plus reviewed outcomes | Replay tests, auditability, versioned comparison | Existing rule memory is separate; learning integration planned |

This is a practical implementation roadmap, not a claim that every algorithm
fits every dataset. Task choice depends on the target, labels, sample size, and
data collection process. Forecasting must preserve time order; classification
must avoid leaking repeated entities across train and test; recommendations
need observed preference or interaction data.

## Current first capability

`cortex_m.learning.dataset` loads numeric CSV data and makes a seeded split.
`cortex_m.learning.regression` fits ordinary least squares with scaling learned
from the training rows. `cortex_m.learning.experiment` saves the model, source
hash, runtime, held-out predictions, and Markdown/JSON reports. `scripts/train.py`
provides the train and predict commands. The estimator uses train-only
standardization and Householder QR. Current implementation rejects
missing/non-numeric values, constant training features, and linearly dependent
columns; encoding, imputation, and automatic feature selection are not yet
implemented. The synthetic demonstration achieved test MAE 0, RMSE 0, and
R² 1.0 on five held-out rows because its target was generated exactly by a
linear equation. This is a plumbing check, not evidence of real-world quality.

The example `data/examples/linear_regression_demo.csv` is synthetic and must
not be presented as real-world performance evidence.
