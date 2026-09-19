# Risk-Aware Stock Forecasting and Trading Decision Support System

**Máster en Data Science — Trabajo de Fin de Máster**

A risk-first financial decision support system: its core is rigorous **volatility forecasting and risk explanation** (GARCH, calibrated prediction intervals), complemented by an **experimental return-forecasting and trading-signal layer** (ARIMAX + rule-based BUY/SELL/HOLD) that is promoted to "recommendation" status only if it proves consistent out-of-sample value in cost-adjusted, walk-forward backtesting.

> ⚠️ **Disclaimer:** This is an academic project. Its outputs are probabilistic decision-support prototypes, **not financial advice**. Backtested performance does not guarantee future results.

## Project structure

```
.
|-- docs/
|   |-- entregas/                     # Deliverables 1-5 (design documents, unchanged)
|   |-- IMPLEMENTATION_NOTES.md       # Feedback compliance and design-vs-implementation notes
|   |-- PROJECT_CHECKLIST.md          # Work plan and status
|   `-- assets/05_mockup_frontal.png  # Deliverable 5 mockup
|-- src/risklens/                     # Pipeline, models, backtest, dashboard data and charts
|-- app/                              # Streamlit dashboard (streamlit_app.py + calibration page)
|-- tests/                            # 31 automated tests
|-- config/preregistered.json         # Frozen models, thresholds and protocol (tag preregistered-v1)
|-- data/
|   |-- raw/                          # Immutable source snapshots + VINTAGE.json (tag vintage-2026-09-19)
|   |-- processed/                    # Cleaned, standardized tables (CSV)
|   `-- gold/                         # gold_market_daily and gold_signals_daily (Parquet)
|-- reports/                          # EDA, validation, calibration and sealed-test results, figures
|-- pyproject.toml, uv.lock           # Locked environment (uv, Python 3.12)
`-- requirements.txt                  # For Streamlit Community Cloud
```

Deliverables 2–4 were revised to incorporate instructor feedback (25 Jul); each carries a revision note describing the changes for traceability.

## Deliverables

| # | Document | Status |
|---|---|---|
| 1 | [Product ideas](docs/entregas/01_ideas_producto.md) | ✅ Delivered |
| 2 | [Selected idea & data requirements](docs/entregas/02_datos_necesarios.md) | ✅ Delivered · revised per feedback |
| 3 | [Data model & gold layer](docs/entregas/03_modelo_datos.md) | ✅ Delivered · revised per feedback |
| 4 | [Analysis design & modeling strategy](docs/entregas/04_analisis_modelado.md) | ✅ Delivered · revised per feedback |
| 5 | [Frontend design & UX](docs/entregas/05_diseno_frontal.md) | ✅ Delivered |

## Frontend (Deliverable 5) — RiskLens

Single-screen Streamlit dashboard, dark-themed for data-dense monitoring, with a risk-first hierarchy and a numbered workflow: **1 · Configure** (asset, run, audit trail of model/rule versions, data health) → **2 · Risk assessment — core** (risk gauge with historical percentile as the lead number, Bank-of-England-style fan chart with 50/80/95% prediction bands, forecast-vs-realized volatility, risk-regime timeline 2010–today, calibration health) → **3 · Signal & explanation — experimental** (HOLD/BUY/SELL with its threshold margin, grounded generated explanation, backtest equity curves vs buy-and-hold after costs). Exception states (stale data vintage, high-uncertainty HOLD, unvalidated signals) are designed into the UI.

![RiskLens mockup](docs/assets/05_mockup_frontal.png)

## MVP scope (risk-first)

| Tier | Content |
|---|---|
| **Core (must have)** | Validated risk system for SPY: GARCH volatility forecasts, low/medium/high risk levels, calibrated 95% prediction intervals, risk explanation layer |
| **Conditional** | ARIMAX return forecasts + BUY/SELL/HOLD signals — shown as recommendations only if they beat per-asset buy-and-hold out-of-sample after costs; otherwise labeled *experimental information* |
| **Nice to have** | Extension to AAPL, MSFT, JPM; full interactive calibration-report page |

## Data architecture (Deliverable 3)

Three-layer flat-file design — **CSV** for `raw/` and `processed/` (transparent, auditable), **Parquet** for `gold/` (typed schema, consumed by code). No database: ~35k rows total makes one unjustifiable.

The gold layer is a two-dataset **data contract**:

| Gold dataset | Granularity | Role |
|---|---|---|
| `gold_market_daily.parquet` | One row per (date, ticker) · ~16,000 rows | **Model input** — adjusted prices, log returns (target), lags, rolling volatility, VIX & Treasury regressors |
| `gold_signals_daily.parquet` | One row per (date, ticker) | **Model output** — forecasts, risk level, signal, plus full audit trail: `model_version`, `train_end_date` (asserted `< date`: machine-checked no-look-ahead proof), `horizon_days`, `rule_version` |

Every decision row is reproducible: model and rule versions map to repository tags, so any historical signal can be re-generated exactly.

## Modeling strategy (Deliverable 4)

Two coupled forecasting tasks + a transparent decision rule, always measured against demanding baselines:

| Task | Baseline | Candidates | Primary metrics |
|---|---|---|---|
| Next-day return (mean) | Naive zero-return (random walk) | ARIMA → ARIMAX (VIX, Treasury yield regressors) | RMSE/MAE vs. naive, directional accuracy |
| Next-day volatility (risk) | 21-day rolling volatility | GARCH(1,1) → EGARCH (optional) | QLIKE, 95% prediction-interval coverage |
| Trading strategy | **Buy-and-hold of the same asset** (primary) · buy-and-hold SPY (market reference) | Rule: `score = expected_return / volatility` → BUY/SELL/HOLD, thresholds calibrated on validation only | Sharpe, max drawdown, hit ratio — after transaction costs |

**Validation guarantees:**
- Temporal split: train 2010–2021 · validation 2022–2023 · test 2024 → frozen end date, with walk-forward re-fitting
- **Sealed test protocol:** test period and raw-data vintage frozen before modeling; winning models and thresholds pre-registered (repository commit) before the test is opened; test evaluated exactly once
- **t+1 execution convention:** a signal computed after the close of day *t* is executed at *t+1* — positions shifted one day, eliminating execution look-ahead
- Strict leakage controls: lag-only features, train-only statistics, forward-fill-only alignment

## Data sources (all open & free)

- **Yahoo Finance** (`yfinance`) — daily OHLCV for SPY, AAPL, MSFT, JPM; VIX
- **FRED** — 10Y Treasury yield (DGS10), yield spread (T10Y2Y), CPI (CPIAUCSL)
- **CBOE** — official VIX history (backup source)

Raw downloads are committed to the repository as an immutable, dated vintage — the project never depends on live source availability (`yfinance` uses unofficial access that may change).

*Attribution and terms:* prices and VIX come from Yahoo Finance (via `yfinance`), yields and CPI from FRED (Federal Reserve Bank of St. Louis). The raw files are included only to make this academic project reproducible; check each source's terms of use before any other reuse.

## Results (sealed test, 2024-01-02 to 2026-09-18, pre-registered and run once)

| Question | Result |
|---|---|
| Are the volatility forecasts calibrated? | Yes. GARCH(1,1)-t 95% intervals cover 95.2% (Kupiec p = 0.85); the joint 50/80/95% fan-chart bands cover 49.5% / 78.7% / 95.3% |
| Does GARCH beat rolling volatility? | Yes on QLIKE (-8.50 vs -8.36, Diebold-Mariano p = 0.010); the rolling baseline's 95% intervals are rejected (coverage 93.0%, Kupiec p = 0.02) |
| Do ARIMA/ARIMAX forecast returns better than zero? | No (RMSE 0.00975 vs 0.00980, p = 0.54) |
| Do the signal strategies beat buy-and-hold SPY after 5 bps costs? | No: Sharpe 0.73 (score rule) and 0.17 (volatility filter) vs 1.27; signals stay labeled EXPERIMENTAL |

Details: `reports/test_results.md`, `reports/model_findings.md`, and `docs/IMPLEMENTATION_NOTES.md` (how the implementation follows the instructor feedback and where it refines the design).

## Tech stack

Python 3.12 · uv · pandas · statsmodels (ARIMA) · arch (GARCH) · Plotly · Streamlit · ruff · pytest · CSV/Parquet layered data storage

## Run it

```
uv sync
uv run python -m risklens.clean && uv run python -m risklens.build_gold   # from the frozen raw vintage
uv run python -m risklens.eda
uv run python -m risklens.run_validation && uv run python -m risklens.calibrate
uv run python -m risklens.run_test      # needs tag preregistered-v1 and unchanged frozen files
uv run streamlit run app/streamlit_app.py
uv run pytest
```

`risklens.ingest` re-downloads data and refuses to overwrite the frozen vintage without `--force`.
