> INVALIDATED exploratory run: demo target typo at row 6. Retained for audit; do not use as valid model performance.

# Linear regression experiment: linear-regression-20261023T180248751967Z

Created: 2026-10-23T18:02:48.751967+00:00

## Dataset

- File: `E:\Research\Cortex-M\data\examples\linear_regression_demo.csv`
- SHA-256: `67417421d09d946b7baa0234586d9053494f4eb953aed5218826a3af0a95fa35`
- Rows: 20
- Features: ad_spend, website_visits
- Target: revenue

## Split

- Train/test rows: 15 / 5
- Test fraction: 0.25
- Random seed: 42

## Metrics

| Set | MAE | RMSE | R² | Rows |
|---|---:|---:|---:|---:|
| train | 1.42109e-15 | 2.59454e-15 | 1 | 15 |
| test | 0.2 | 0.447214 | 0.998589 | 5 |

## Learned model

Intercept: `27.633333`

| Feature | Train mean | Train scale | Standardized coefficient |
|---|---:|---:|---:|
| ad_spend | 10.266667 | 5.7904135 | 11.580827 |
| website_visits | 8.2 | 5.52811 | 2.764055 |

## Notes

Test metrics are computed on rows excluded from model fitting. This result is valid only for the supplied dataset and split; it is not a general performance guarantee.

Model parameters are saved in `model.json`; per-test-row predictions and errors are saved in `report.json`.
