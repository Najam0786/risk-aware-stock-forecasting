# Deliverable 3 — Data Model and Gold Layer Design

**Project:** Risk-Aware Stock Forecasting and Trading Decision Support System
**Máster en Data Science — Trabajo de Fin de Máster**

---

## 1. Project and Data Summary

**Problem.** Retail investors make buy/sell/hold decisions without tooling that quantifies expected return *and* risk together, leading to emotion-driven decisions in a low signal-to-noise domain.

**Solution.** A decision support system that models daily **log returns** with ARIMAX (expected return) and GARCH (conditional volatility), and converts both into a rule-based Buy / Sell / Hold signal (`Signal = Expected Return / Volatility`), validated through walk-forward backtesting and presented in a Streamlit dashboard.

**Data sources and what each contributes** (defined in Deliverable 2):

| Source | Information contributed |
|---|---|
| Yahoo Finance (`yfinance`) | Daily OHLCV + Adjusted Close for SPY, AAPL, MSFT, JPM — the core price history from which log returns are computed; also the VIX index (^VIX) |
| FRED | Macroeconomic exogenous regressors: 10Y Treasury yield (DGS10), 10Y–2Y yield spread (T10Y2Y), and optionally monthly CPI (CPIAUCSL) |
| CBOE | Backup/verification source for official VIX history |

Coverage: ~15 years of daily data (2010–present), ≈ 4,000 trading days per asset.

## 2. Storage Technology and Format

**Decision: flat files — CSV for the raw and processed layers, Parquet for the gold layer. No database.**

| Layer | Format | Justification |
|---|---|---|
| Raw | **CSV** | Matches the native export format of the sources (FRED serves CSV; `yfinance` DataFrames persist naturally to CSV). Human-readable, diff-able in Git, and keeps the raw snapshots transparent and auditable. |
| Processed | **CSV** | Intermediate cleaning results remain inspectable during development; easy to open and verify manually while iterating on cleaning rules. |
| Gold | **Parquet** | The gold layer is a *data contract* consumed by code (models, backtest, dashboard), not by humans. Parquet preserves column data types exactly (dates stay `datetime64`, returns stay `float64`), eliminating the type-parsing ambiguity of CSV, and provides columnar compression. Every downstream phase reads identical, guaranteed schemas. |

**Why not a relational database (SQLite/PostgreSQL)?** The total volume is ~35,000 rows (a few MB), fully rewritten on each pipeline run, with no concurrent writers, no transactional needs, and a single consumer (the modeling pipeline). A database would add setup, connection management, and migration overhead without any benefit at this scale. The layered flat-file design achieves the same guarantees (clear contract, typed schema) with technology proportional to the project. If the project ever needed multi-user access or incremental daily appends in production, SQLite would be the natural upgrade path — documented here as a conscious trade-off, not an omission.

## 3. Data Layer Structure

The repository follows the standard three-layer ("medallion-style") structure already declared in the project README:

```
data/
|-- raw/                      # Immutable source snapshots, exactly as downloaded
|   |-- prices_SPY.csv
|   |-- prices_AAPL.csv
|   |-- prices_MSFT.csv
|   |-- prices_JPM.csv
|   |-- vix.csv
|   |-- fred_dgs10.csv
|   |-- fred_t10y2y.csv
|   `-- fred_cpi.csv          # (optional series)
|-- processed/                # Cleaned, standardized, per-source tables
|   |-- prices_clean.csv      # All tickers stacked, standardized columns, valid trading days
|   `-- macro_daily.csv       # VIX + FRED series aligned to the trading calendar
`-- gold/                     # Final model-ready datasets (the data contract)
    |-- gold_market_daily.parquet
    `-- gold_signals_daily.parquet   # produced later by the modeling phase
```

**Layer policies:**

- **Raw** — written once per download, never edited. Each file keeps the source's original column names. A download date is recorded so results are reproducible even if sources change later. Raw files are committed to the repository (volume is small), making the entire project reproducible without live source access.
- **Processed** — output of deterministic cleaning scripts: standardized column names (`snake_case`), enforced dtypes, deduplicated, date-indexed, restricted to valid trading days.
- **Gold** — final joined and feature-engineered datasets. Only gold files may be read by the EDA, models, backtest, and dashboard. No downstream code ever reads raw or processed directly — this is the enforcement rule that makes the gold layer a genuine contract.

## 4. Gold Layer Definition (Data Contract)

The gold layer consists of **two datasets**: one *input* dataset feeding the models (built during data preparation) and one *output* dataset produced by the models and consumed by the backtest and dashboard. Defining both now fixes the interfaces of the whole system.

### 4.1 `gold_market_daily.parquet` — model input

| Property | Definition |
|---|---|
| Functional description | Complete, model-ready market dataset: prices, returns, engineered features, and macro regressors, aligned on the official trading calendar |
| Granularity | **One row per (trading date, ticker)** |
| Expected records | ≈ 4,000 trading days × 4 tickers ≈ **16,000 rows** |
| Primary key | Composite key (`date`, `ticker`) — unique, no duplicates permitted |
| Target variable | `log_return` (modeled by ARIMAX; its conditional variance modeled by GARCH) |
| Consumed by | EDA (Deliverable 4 analysis), ARIMAX/GARCH model training, feature analysis |

**Schema:**

| Field | Type | Description |
|---|---|---|
| `date` | date | Trading date (YYYY-MM-DD), part of PK |
| `ticker` | string | Asset symbol (SPY, AAPL, MSFT, JPM), part of PK |
| `adj_close` | float64 | Adjusted closing price (splits/dividends corrected) |
| `volume` | int64 | Daily traded volume |
| `log_return` | float64 | **Target.** `ln(adj_close_t / adj_close_t-1)` |
| `ret_lag_1` … `ret_lag_5` | float64 | Lagged log returns (momentum features) |
| `roll_vol_21` | float64 | 21-day rolling std of log returns (realized volatility) |
| `roll_mean_21` | float64 | 21-day rolling mean of log returns |
| `vix_close` | float64 | VIX level at close |
| `vix_change` | float64 | Daily change in VIX (Δ, per "changes not levels" principle) |
| `dgs10` | float64 | 10Y Treasury yield (%) |
| `dgs10_change` | float64 | Daily change in 10Y yield (Δ) |
| `t10y2y` | float64 | 10Y–2Y yield spread (%) |
| `day_of_week` | int8 | 0–4 (Mon–Fri), calendar/seasonality feature |
| `cpi_yoy` | float64 | *(desirable)* Year-over-year CPI inflation, as-of aligned |

### 4.2 `gold_signals_daily.parquet` — model output

| Property | Definition |
|---|---|
| Functional description | Model forecasts and trading decisions per asset and day; the interface between the modeling phase and the backtest/dashboard |
| Granularity | **One row per (forecast date, ticker)** |
| Expected records | One row per test/backtest day per ticker (≈ 1,000–4,000 rows depending on evaluation window) |
| Primary key | Composite key (`date`, `ticker`) |
| Key output columns | `signal` (the trading decision), `expected_return`, `forecast_volatility` |
| Consumed by | Backtesting engine, Streamlit dashboard, final evaluation report |

**Schema:**

| Field | Type | Description |
|---|---|---|
| `date` | date | Forecast target date, part of PK |
| `ticker` | string | Asset symbol, part of PK |
| `expected_return` | float64 | ARIMAX one-step-ahead forecast of log return |
| `forecast_volatility` | float64 | GARCH one-step-ahead conditional volatility |
| `risk_adjusted_score` | float64 | `expected_return / forecast_volatility` |
| `risk_level` | category | low / medium / high (volatility bucket) |
| `signal` | category | **BUY / SELL / HOLD** (rule-based decision) |
| `pi_lower`, `pi_upper` | float64 | Prediction interval bounds for the return |
| `run_timestamp` | datetime | When the forecast was generated (reproducibility) |

## 5. Relationships Between Data

The project combines **four price series and three macro series** into a single gold table. The relationships are:

```
prices (per ticker)   N ────── 1   macro_daily (per date)
        (date, ticker)    join on date    (date)

gold_market_daily (date, ticker)  1 ────── 1  gold_signals_daily (date, ticker)
                        (for dates in the evaluation window)
```

- **Prices ↔ macro: N:1 on `date`.** Each trading date has 4 price rows (one per ticker) but exactly one macro row (VIX, DGS10, T10Y2Y are market-wide, not per-ticker). The join broadcasts macro values across tickers.
- **The trading calendar of the price data is the spine.** All other series are aligned *to* it: macro rows on dates when the stock market is closed are dropped; missing macro values on trading days are forward-filled (see §8).
- **`gold_market_daily` ↔ `gold_signals_daily`: 1:1 on (`date`, `ticker`)** within the evaluation window — every forecast row traces back to exactly one input row, guaranteeing full traceability from decision to data.
- **Expected join problems:** bond-market holidays differ from stock-market holidays (FRED gaps on trading days); FRED encodes missing values as `"."` strings; CPI is monthly and published with a lag, requiring as-of (not exact-date) alignment; `yfinance` returns timezone-aware timestamps while FRED dates are naive — all dates will be normalized to plain dates before joining.

A more complex relational model (separate dimension/fact tables) is unnecessary: there is a single fact grain (daily per ticker) and the "dimensions" (ticker metadata) are four constant values.

## 6. Initial Data Dictionary

Core fields relevant to the analysis, model, and dashboard:

| Field | Description | Type | Source | Mandatory | Observations |
|---|---|---|---|---|---|
| `date` | Trading date | date | Yahoo (calendar spine) | Yes | Normalized to YYYY-MM-DD, timezone stripped |
| `ticker` | Asset symbol | string | Constant (SPY/AAPL/MSFT/JPM) | Yes | Categorical, 4 values |
| `adj_close` | Adjusted close price (USD) | float64 | Yahoo Finance | Yes | Must be > 0; source of all return computations |
| `volume` | Shares traded | int64 | Yahoo Finance | Yes | ≥ 0; occasional zero-volume days flagged |
| `log_return` | Daily log return | float64 | Derived | Yes | Target variable; first row per ticker is NaN by construction and dropped |
| `roll_vol_21` | 21-day realized volatility | float64 | Derived | Yes | First 20 rows per ticker NaN by construction |
| `vix_close` | VIX index close | float64 | Yahoo (^VIX) / CBOE | Yes | Range check: ~9–90 historically |
| `dgs10` | 10Y Treasury yield | float64 | FRED (DGS10) | Yes | FRED missing values arrive as "." → parsed to NaN → forward-filled |
| `t10y2y` | 10Y–2Y yield spread | float64 | FRED (T10Y2Y) | No (desirable) | Negative values are valid (curve inversion) |
| `cpi_yoy` | CPI inflation, year-over-year | float64 | FRED (CPIAUCSL) | No (desirable) | Monthly → as-of aligned to daily; publication lag respected |
| `signal` | Trading decision | category | Model output | Yes (gold_signals) | BUY / SELL / HOLD only |
| `expected_return` | Forecasted next-day log return | float64 | Model output | Yes (gold_signals) | Paired with `pi_lower`/`pi_upper` |
| `forecast_volatility` | Forecasted conditional volatility | float64 | Model output | Yes (gold_signals) | Must be > 0 |

## 7. Expected Data Quality Problems (Specific to This Project)

1. **FRED missing-value encoding.** FRED CSV downloads represent missing observations as the literal string `"."`, which silently turns the whole column into text if not handled. Must be parsed to NaN explicitly.
2. **Calendar misalignment between bond and stock markets.** FRED daily series have gaps on bond-market holidays that are stock trading days (e.g., Columbus Day, Veterans Day) — so a naive inner join would *drop valid trading days*, and a naive outer join would create rows on days with no prices.
3. **Monthly CPI vs daily prices.** CPI has one value per month, published with ~2 weeks' lag. Joining the value to all days of its own month would use information not yet public on those days (leakage). Requires as-of alignment respecting publication timing.
4. **Splits and dividends.** Raw close prices jump artificially on split/dividend dates (e.g., AAPL 4:1 split, Aug 2020). Solved by using **Adjusted Close** exclusively for return computation; raw close kept only for reference.
5. **Extreme-but-valid outliers.** March 2020 (COVID crash) contains daily moves of ±10%+ and VIX above 80. These are *not* errors and must **not** be removed or winsorized — capturing exactly these regimes is the purpose of the GARCH model. The cleaning rules must distinguish data errors from genuine extremes.
6. **`yfinance` structural quirks.** Multi-ticker downloads return MultiIndex columns; single-ticker downloads do not. Timestamps can be timezone-aware. Both must be normalized to a stable flat schema at the raw→processed step.
7. **Duplicates on re-download.** Re-running the ingestion can append overlapping date ranges. Deduplication on (`date`, `ticker`) with a "keep latest download" rule is required, and the PK uniqueness is asserted before writing gold.
8. **Construction NaNs from feature engineering.** Lags and 21-day rolling windows create NaNs in the first rows of each ticker's history (and after the join, at the series start). These are expected and handled by trimming the warm-up window, not by imputation.
9. **Zero/abnormal volume days.** Rare half-day sessions (e.g., day after Thanksgiving) show unusually low volume. Kept, but flagged — relevant when interpreting volatility features.
10. **Series start differences.** T10Y2Y and VIX have full coverage for 2010+, but if the window were ever extended earlier, availability differs by series; the common-coverage window is computed and documented at build time.

## 8. Planned Cleaning and Transformation Decisions

Initial hypotheses (may be refined later, with changes documented for traceability):

- **Null handling:**
  - Macro series (VIX, DGS10, T10Y2Y) on trading days: **forward-fill only** (carry last known value). Backward-fill is prohibited — it would leak future information into the past.
  - CPI: as-of join — each trading day receives the most recent CPI value *published on or before* that day.
  - Price nulls: not filled. A missing price row is investigated; if a source error, re-downloaded; if the market was closed, the date should not exist in the spine.
- **Duplicates:** drop on (`date`, `ticker`), keeping the most recent download; assert PK uniqueness before every gold write.
- **Normalization:** all column names to `snake_case`; all dates to timezone-naive `YYYY-MM-DD`; all prices in USD (single currency, no conversion needed); yields kept in percentage points, returns in log units (documented to avoid unit confusion).
- **Derived variables:** `log_return`, lags 1–5, `roll_mean_21`, `roll_vol_21`, `vix_change`, `dgs10_change`, `day_of_week` — computed in the processed→gold step, per ticker, strictly from past data (no look-ahead in any rolling window).
- **Aggregations:** none required — the analysis grain equals the source grain (daily). CPI is the only frequency conversion (monthly → daily via as-of).
- **Discarded data:** intraday fields beyond OHLCV (not needed), pre-2010 history (out of declared scope), raw unadjusted close for modeling (kept in raw only), and any macro series columns other than the declared ones.
- **Record validity criteria:** a gold row is valid iff `adj_close > 0`, `log_return` is finite, (`date`, `ticker`) is unique, `date` is a NYSE trading day, and all mandatory fields are non-null after the warm-up trim. Rows failing validation are counted and reported by the pipeline, not silently dropped.

## 9. Risks of the Data Model

- **Clearest part:** the price pipeline (raw → clean returns). Yahoo adjusted prices for four ultra-liquid assets are as reliable as free market data gets, and log-return computation is deterministic and easily testable.
- **Most uncertain part:** the macro alignment logic — the forward-fill rules, holiday handling, and especially the as-of CPI join. These involve subtle leakage risks that require careful implementation and testing.
- **Most problematic source:** FRED is institutionally rock-solid as a *source*, but its calendar quirks make it the most error-prone at *join time*; `yfinance` is the most fragile at *download time* (unofficial API). Both risks are mitigated by committing immutable raw snapshots, so a source failure never breaks reproducibility.
- **If the gold layer cannot be built as defined:** the fallback is graceful degradation, not redesign — (1) drop `cpi_yoy` (declared desirable, not essential); (2) drop `t10y2y`, keeping VIX + DGS10 as the exogenous set; (3) in the extreme case, a SPY-only gold table with VIX as the single regressor still supports the full ARIMAX + GARCH + signal + backtest pipeline. The schema and contract remain identical throughout — only columns/tickers shrink.
- **Simplification alternative:** the two-dataset gold design could collapse to `gold_market_daily` only, with model outputs written as a plain results CSV. This loses the clean model/consumer interface but preserves every deliverable of the MVP.

---

*This document corresponds to Deliverable 3. Deliverables 1 and 2 are maintained unchanged in this same directory for traceability.*
