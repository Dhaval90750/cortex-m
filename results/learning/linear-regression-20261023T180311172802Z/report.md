# Linear regression experiment: linear-regression-20261023T180311172802Z

Created: 2026-10-23T18:03:11.172802+00:00

## Dataset

- File: `E:\Research\Cortex-M\data\examples\linear_regression_demo.csv`
- SHA-256: `d0690b6ebcb2c44d5ebff95660200df3de1c590a584b23142b97cc071f842796`
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
| test | 1.42109e-15 | 3.17764e-15 | 1 | 5 |

## Learned model

Intercept: `27.633333`

| Feature | Train mean | Train scale | Standardized coefficient |
|---|---:|---:|---:|
| ad_spend | 10.266667 | 5.7904135 | 11.580827 |
| website_visits | 8.2 | 5.52811 | 2.764055 |

## Notes

Test metrics are computed on rows excluded from model fitting. This result is valid only for the supplied dataset and split; it is not a general performance guarantee.

Model parameters are saved in `model.json`; per-test-row predictions and errors are saved in `report.json`.
