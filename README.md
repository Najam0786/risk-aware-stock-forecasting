# Risk-Aware Stock Forecasting and Trading Decision Support System

**Máster en Data Science — Trabajo de Fin de Máster**

A data-driven, risk-aware financial decision support system that forecasts stock returns (ARIMAX) and volatility (GARCH), and translates them into actionable Buy / Sell / Hold trading signals validated through walk-forward backtesting.

> ⚠️ **Disclaimer:** This is an academic project. Its outputs are probabilistic decision-support prototypes, **not financial advice**. Backtested performance does not guarantee future results.

## Project structure

```
.
|-- docs/
|   `-- entregas/
|       |-- 01_ideas_producto.md      # Deliverable 1 — Product ideas explored
|       |-- 02_datos_necesarios.md    # Deliverable 2 — Selected idea & data requirements
|       |-- 03_modelo_datos.md        # Deliverable 3 — Data model & gold layer design
|       `-- 04_analisis_modelado.md   # Deliverable 4 — Analysis design & modeling strategy
|-- data/
|   |-- raw/                          # Immutable source snapshots (CSV, as downloaded)
|   |-- processed/                    # Cleaned, standardized tables (CSV)
|   `-- gold/                         # Model-ready datasets (Parquet) — the data contract
`-- README.md
```

## Deliverables

| # | Document | Status |
|---|---|---|
| 1 | [Product ideas](docs/entregas/01_ideas_producto.md) | ✅ Delivered |
| 2 | [Selected idea & data requirements](docs/entregas/02_datos_necesarios.md) | ✅ Delivered |
| 3 | [Data model & gold layer](docs/entregas/03_modelo_datos.md) | ✅ Delivered |
| 4 | [Analysis design & modeling strategy](docs/entregas/04_analisis_modelado.md) | ✅ Delivered |

## Data architecture (Deliverable 3)

Three-layer flat-file design — **CSV** for `raw/` and `processed/` (transparent, auditable), **Parquet** for `gold/` (typed schema, consumed by code). No database: ~35k rows total makes one unjustifiable.

The gold layer is a two-dataset **data contract**:

| Gold dataset | Granularity | Role |
|---|---|---|
| `gold_market_daily.parquet` | One row per (date, ticker) · ~16,000 rows | **Model input** — adjusted prices, log returns (target), lags, rolling volatility, VIX & Treasury regressors |
| `gold_signals_daily.parquet` | One row per (date, ticker) | **Model output** — expected return, forecast volatility, risk level, BUY/SELL/HOLD signal → consumed by backtest & dashboard |

## Modeling strategy (Deliverable 4)

Two coupled forecasting tasks + a transparent decision rule, always measured against demanding baselines:

| Task | Baseline | Candidates | Primary metrics |
|---|---|---|---|
| Next-day return (mean) | Naive zero-return (random walk) | ARIMA → ARIMAX (VIX, Treasury yield regressors) | RMSE/MAE vs. naive, directional accuracy |
| Next-day volatility (risk) | 21-day rolling volatility | GARCH(1,1) → EGARCH (optional) | QLIKE, 95% prediction-interval coverage |
| Trading strategy | Buy-and-hold SPY | Rule: `score = expected_return / volatility` → BUY/SELL/HOLD | Sharpe, max drawdown, hit ratio (with transaction costs) |

**Validation:** temporal split (train 2010–2021 · validation 2022–2023 · test 2024+) with walk-forward re-fitting and strict leakage controls (lag-only features, train-only statistics, forward-fill-only alignment).

## Data sources (all open & free)

- **Yahoo Finance** (`yfinance`) — daily OHLCV for SPY, AAPL, MSFT, JPM; VIX
- **FRED** — 10Y Treasury yield (DGS10), yield spread (T10Y2Y), CPI (CPIAUCSL)
- **CBOE** — official VIX history (backup source)

## Planned tech stack

Python · pandas · statsmodels (ARIMAX) · arch (GARCH) · Streamlit (dashboard) · CSV/Parquet layered data storage
