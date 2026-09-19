# Implementation notes — design vs. what was built

Deliverables 1–5 in `docs/entregas/` are left exactly as submitted. This file records how the implementation follows the instructor feedback (25 Jul) and where it refines the design, so both stay traceable.

## Instructor feedback compliance

| Feedback point | Where it is implemented |
|---|---|
| BUY/SELL/HOLD thresholds fixed on validation only | `src/risklens/calibrate.py`: grid on 2022–2023, coarse grid (12 rule combinations, 7 filter levels), sensitivity saved in `reports/results/calibration_grid_*.csv` |
| Execution convention: signal at *t*, execution at *t+1* | `src/risklens/strategy.py` (`held_positions`). Both readings are reported, see below |
| Per-asset buy-and-hold as primary baseline, SPY as reference | `run_test.py`: buy-and-hold of the same asset (SPY is the asset and the market reference) |
| Freeze test date and data vintage | Vintage `data/raw/VINTAGE.json` (2026-09-19, file hashes), tag `vintage-2026-09-19`; test window 2024-01-02 to 2026-09-18 |
| Winning model selected before opening the test set | `config/preregistered.json`, tag `preregistered-v1`; `run_test.py` refuses to run if the tag is missing or frozen files changed |
| Auditability fields | `gold_signals_daily`: `model_version`, `train_end_date` (asserted `< date`), `horizon_days`, `rule_version`, `run_timestamp` |
| MVP with risk as the core; signals experimental unless proven | Outcome of the pre-registered test: risk criterion met, signals not validated, so they stay EXPERIMENTAL |

## Refinements to the design

| Design (D2–D5) | Implementation | Reason |
|---|---|---|
| EGARCH as optional asymmetric model | GJR-GARCH(1,1) Student-t (tested, not selected) | Same leverage effect, closed-form recursion; EDA Q7 supported asymmetry but it did not improve out-of-sample |
| ARIMAX (VIX, yields) as flagship mean model | ARIMA(2,0,0) | BIC on the training sample rejected the exogenous variables; simpler model wins (D4 §6.10). ARIMAX variants are evaluated and reported in `reports/model_validation.md` |
| Score rule only | Score rule (A) plus a volatility filter (B: long when forecast volatility is below a training-percentile ceiling) | D4 §8 fallback: risk-based signal that depends only on the validated risk model. Both were calibrated on validation and pre-registered together |
| SELL undefined | SELL goes flat; HOLD keeps the previous position; strategies start flat | Long-only, no shorting |
| `date` in `gold_signals_daily` | `date` = forecast target day; new `origin_date` = day whose information was used (join key to `gold_market_daily`) | D3 defined both readings. Invariant `train_end_date < date` holds as written |
| 95% interval only | Added `pi_lower_50/80`, `pi_upper_50/80`, `vol_percentile`, `vol_filter_state`, `split` (validation / test / live) | The dashboard fan chart and risk gauge need them |

## Execution convention

D4 §7 gives a formula (position(*t*) × return(*t+1*)) and prose ("cannot be traded at the very close that produced it"). Those are two different simulations, so both are computed:

- **Primary (stricter):** the decision at *t* executes at the close of *t+1* and earns the return of *t+2*.
- **Shift-1 convention:** the position is shifted one day, so it earns the return of *t+1* (fills at the closing price that produced the signal).

The conclusions are the same under either, see `reports/test_results.md`. The primary convention was pre-registered before the test was opened. The report labels were clarified afterwards; that rerun reproduced every number.

## Headline results (sealed test, 2024-01-02 to 2026-09-18)

- Volatility: GARCH(1,1)-t beats the 21-day rolling baseline on QLIKE (Diebold-Mariano p=0.010); 95% coverage 95.2%, 50/80/95% joint bands calibrated.
- Returns: ARIMA(2,0,0) does not beat the naive forecast (RMSE 0.00975 vs 0.00980).
- Strategies: neither beats buy-and-hold SPY (Sharpe 1.27) under the pre-registered rule, under either convention.

## Extension assets: AAPL, MSFT, JPM

The design's nice-to-have tier was implemented with the same protocol, per asset, so the instructor's point about per-asset baselines now applies in full.

- **Procedure (`src/risklens/extension.py`):** ARMA order by BIC on each asset's training data with no exogenous regressors (the SPY study showed they do not help), GARCH(1,1) with Student-t errors, thresholds and volatility ceilings fixed on validation only, exactly as for SPY.
- **Pre-registration:** configuration in `config/preregistered_extension.json`, committed and tagged `preregistered-v2` before any of the three test windows were used. The gated runner (`run_test --extension`) refuses to run if the tag is missing or any frozen file changed; the runner and the shared modules are themselves frozen at that tag.
- **Runner generalization:** `run_test.py` was refactored to serve any asset; rerunning SPY through it reproduced the previous metrics, equity curves and signals exactly.
- **Baselines:** primary is buy-and-hold of the same asset; buy-and-hold SPY is reported as the market reference.
- **Selection fallback (declared before the test):** if no threshold combination reaches the minimum number of trades on validation, the most active one is registered and flagged as not passing the constraints. This happened for AAPL's score rule.

| Asset | ARMA order | GARCH-t vs rolling (QLIKE, DM p) | 95% coverage (Kupiec p) | Sharpe: buy-and-hold / A / B |
|---|---|---|---|---|
| AAPL | (0,0,0) | -7.21 vs -7.06 (0.008) | 94.1% (0.31) | 0.90 / 0.95 / 0.07 |
| MSFT | (1,0,0) | -7.23 vs -7.08 (0.003) | 94.6% (0.61) | 0.54 / 0.35 / 0.18 |
| JPM | (2,0,2) | -7.39 vs -7.28 (0.002) | 94.9% (0.87) | 1.33 / 1.01 / 0.70 |

The risk result replicates on all three assets. AAPL's score rule formally meets the acceptance rule but made one trade (constant mean model, so it never signals SELL and behaves as buy-and-hold); it is treated as not validated, and the dashboard says so. Reports: `reports/test_results_<TICKER>.md`, `reports/extension_validation.md`, `reports/extension_calibration.md`.

## Reproduce

```
uv sync
uv run python -m risklens.clean && uv run python -m risklens.build_gold
uv run python -m risklens.eda
uv run python -m risklens.data_quality && uv run python -m risklens.diagnostics
uv run python -m risklens.run_validation && uv run python -m risklens.calibrate
uv run python -m risklens.run_test      # requires tag preregistered-v1 and unchanged frozen files
uv run python -m risklens.extension && uv run python -m risklens.run_test --extension
uv run pytest
```
