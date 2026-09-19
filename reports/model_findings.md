# Model findings on validation (2022-2023; train 2010-2021; test sealed)

Reproduce: `uv run python -m risklens.run_validation` (full tables in `model_validation.md`, forecasts in `results/validation_forecasts.csv`). 501 one-day-ahead walk-forward forecasts, expanding window, refit every 21 days. Implementation checks: the GARCH variance recursion matches the `arch` library's own forecast; the mean-model walk-forward is proven free of look-ahead (`tests/test_models.py`).

## Risk task (SPY volatility)

| Model | QLIKE (lower better) | 95% coverage | Kupiec p | Cond. coverage p |
|---|---|---|---|---|
| Rolling 21-day (baseline) | -7.9479 | 93.6% | 0.17 | 0.27 |
| EWMA λ=0.94 | -7.9738 | 94.2% | 0.43 | 0.12 |
| GARCH(1,1) Student-t | -7.9462 | 95.2% | 0.83 | 0.29 |
| GJR-GARCH(1,1) Student-t | -7.9439 | 94.2% | 0.43 | 0.61 |

- No model beats the 21-day rolling baseline on QLIKE with statistical significance (Diebold-Mariano p: EWMA 0.08, GARCH 0.93). D4 acceptance criterion (2) has two parts: the coverage part holds (95.2%, inside the 90–97% band, not rejected by Kupiec or Christoffersen); the QLIKE part does not hold on validation.
- All four models produce well-calibrated 95% intervals. The QLIKE differences are tiny relative to their noise (about 500 daily observations, squared-return proxy).
- Asymmetry (GJR) helps in 2023 (-8.624 vs -8.594) but not in 2022. Not a robust gain.
- GARCH annualized forecast volatility ranged 8.4%–42.2% over the window.

## Mean task (SPY next-day return)

| Model | RMSE | Directional accuracy | DM vs naive |
|---|---|---|---|
| Naive zero (baseline) | 0.01229 | n/a | n/a |
| Always up (direction baseline) | n/a | 49.7% | n/a |
| ARIMA(2,0,0) | 0.01247 | 50.3% | naive better, p=0.021 |
| ARIMAX(2,0,0) + vix_change | 0.01246 | 48.3% | naive better, p=0.012 |
| ARIMAX(2,0,0) + vix_change + dgs10_change | 0.01247 | 49.9% | naive better, p=0.016 |

- ARMA order by BIC on train: (2,0,0). Exogenous set by BIC on train: none (vix_change +0.4 BIC worse than no regressor).
- Every ARIMA/ARIMAX model is significantly worse than the naive zero forecast on RMSE, in both 2022 and 2023. The EDA correlation of vix_change with the next-day return (r=+0.11) did not survive walk-forward out-of-sample testing.
- This is the outcome D2 and D4 anticipated (near-efficient markets). Under D4 §6 step 10 no mean model qualifies as a validated return forecaster, so signals stay EXPERIMENTAL.

## Consequences for Day 4

- Report the negative result on return predictability plainly; the risk layer (calibrated intervals, volatility forecasts) is the validated product.
- Forecast scale is not degenerate: ARIMAX expected return ranges ±0.5%, and |score| exceeds 0.05 on 69% of days (max 0.31), so the θ grid produces trades.
- Candidate strategies to calibrate on validation and pre-register before opening the test:
  1. D4 rule with ARIMAX(vix_change) expected return (experimental).
  2. Volatility-filter strategy: hold SPY when forecast volatility is below v_max, otherwise cash. Uses only the validated risk model.
