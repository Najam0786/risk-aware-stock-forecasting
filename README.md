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
|       `-- 02_datos_necesarios.md    # Deliverable 2 — Selected idea & data requirements
|-- data/                             # (from Deliverable 3 onwards)
|   |-- raw/                          # Immutable source snapshots
|   |-- processed/                    # Cleaned / intermediate data
|   `-- gold/                         # Final model-ready datasets
`-- README.md
```

## Deliverables

| # | Document | Status |
|---|---|---|
| 1 | [Product ideas](docs/entregas/01_ideas_producto.md) | ✅ Delivered |
| 2 | [Selected idea & data requirements](docs/entregas/02_datos_necesarios.md) | ✅ Delivered |
| 3 | Data model & gold layer | 🔜 |
| 4 | Analysis design & modeling strategy | 🔜 |

## Data sources (all open & free)

- **Yahoo Finance** (`yfinance`) — daily OHLCV for SPY, AAPL, MSFT, JPM; VIX
- **FRED** — 10Y Treasury yield (DGS10), yield spread (T10Y2Y), CPI (CPIAUCSL)
- **CBOE** — official VIX history (backup source)

## Planned tech stack

Python · pandas · statsmodels (ARIMAX) · arch (GARCH) · Streamlit (dashboard) · CSV/Parquet layered data storage
