# Risk-Aware Stock Forecasting and Trading Decision Support System

**Máster en Data Science — Trabajo de Fin de Máster**

A risk-first financial decision support system: its core is rigorous **volatility forecasting and risk explanation** (GARCH, calibrated prediction intervals), complemented by an **experimental return-forecasting and trading-signal layer** (ARIMAX + rule-based BUY/SELL/HOLD) that is promoted to "recommendation" status only if it proves consistent out-of-sample value in cost-adjusted, walk-forward backtesting.

> ⚠️ **Disclaimer:** This is an academic project. Its outputs are probabilistic decision-support prototypes, **not financial advice**. Backtested performance does not guarantee future results.

## Project structure

```
.
|-- docs/
|   |-- entregas/
|   |   |-- 01_ideas_producto.md      # Deliverable 1 — Product ideas explored
|   |   |-- 02_datos_necesarios.md    # Deliverable 2 — Selected idea & data requirements (rev.)
|   |   |-- 03_modelo_datos.md        # Deliverable 3 — Data model & gold layer design (rev.)
|   |   |-- 04_analisis_modelado.md   # Deliverable 4 — Analysis design & modeling strategy (rev.)
|   |   `-- 05_diseno_frontal.md      # Deliverable 5 — Frontend design & UX
|   `-- assets/
|       `-- 05_mockup_frontal.png     # Main mockup of the RiskLens dashboard
|-- data/
|   |-- raw/                          # Immutable source snapshots (CSV) — the frozen data vintage
|   |-- processed/                    # Cleaned, standardized tables (CSV)
|   `-- gold/                         # Model-ready datasets (Parquet) — the data contract
`-- README.md
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

## Planned tech stack

Python · pandas · statsmodels (ARIMAX) · arch (GARCH) · Plotly · Streamlit (dashboard) · CSV/Parquet layered data storage
