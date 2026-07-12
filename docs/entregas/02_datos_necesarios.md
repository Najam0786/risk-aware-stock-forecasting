# Deliverable 2 — Selected Project Idea and Data Requirements Analysis

**Project:** Risk-Aware Stock Forecasting and Trading Decision Support System
**Máster en Data Science — Trabajo de Fin de Máster**

---

## 1. Selected Idea

The project selected for the rest of the course is the **Risk-Aware Stock Forecasting and Trading Decision Support System**, evolved from the ideas explored in Deliverable 1.

**Paragraph 1 — Problem it solves.** Retail investors and non-professional traders face a structural disadvantage in financial markets: they must make buy/sell/hold decisions under uncertainty, without the quantitative tooling that institutional investors use to measure expected return *and* risk together. Most freely available tools show raw price charts or point predictions, which are misleading in a domain with a very low signal-to-noise ratio. The result is decision-making driven by emotion, headlines, or overconfident forecasts that ignore volatility. The problem is relevant because poor risk awareness — not poor return prediction — is the main cause of large retail losses (buying in calm markets and panic-selling in volatile ones). A system that quantifies both the expected movement and the uncertainty around it, and translates them into a disciplined, systematic recommendation, brings real value: it replaces gut feeling with a transparent, risk-adjusted decision process.

**Paragraph 2 — Proposed solution.** The solution is a data-driven decision support system built on time-series modeling of **log returns** (price changes) rather than raw prices, which acknowledges the low signal-to-noise nature of markets and ensures statistically sound (stationary) inputs. The approach combines two complementary models: a mean model (ARIMAX on log returns, using exogenous market and macroeconomic variables such as the VIX and Treasury yields) to forecast the expected return, and a volatility model (GARCH-family) to forecast the conditional risk. A rule-based **strategy layer** then converts these two outputs into an actionable trading signal — BUY, SELL, or HOLD — using a risk-adjusted score (expected return divided by forecasted volatility). The entire pipeline is validated with time-series-correct methodology: temporal splits, walk-forward validation, and a backtesting framework that simulates how the strategy would have performed historically against a passive benchmark.

**Paragraph 3 — MVP of the final project.** The MVP presented at the end of the course will be a working, end-to-end pipeline plus an interactive presentation layer. Concretely, the following will be seen functioning: (1) an automated data pipeline that ingests historical market and macroeconomic data into a clean, documented gold dataset; (2) fitted ARIMAX and GARCH models producing next-period return forecasts, volatility forecasts, and prediction intervals for the selected assets; (3) a decision engine outputting a Buy/Sell/Hold signal with a risk level (low/medium/high) and a risk-adjusted confidence indicator; (4) a backtesting report comparing the strategy against a buy-and-hold baseline using cumulative returns, Sharpe ratio, maximum drawdown, and hit ratio; and (5) an interactive dashboard (Streamlit) where the user can select an asset and see the forecast, the risk assessment, the current signal, and the historical backtest performance.

## 2. Required Data

### Scope decision

To keep the project realistic within the course timeline while still demonstrating generality, the system will cover **SPY (the S&P 500 ETF) as the primary asset**, plus **three liquid large-cap stocks (AAPL, MSFT, JPM)** as a generalization check. SPY is ideal as the core asset: it is extremely liquid, has decades of clean history, and represents the overall market, which makes results interpretable.

### Variables / fields needed

| Group | Fields | Purpose |
|---|---|---|
| Price data (per asset) | Date, Open, High, Low, Close, Adjusted Close, Volume | Core input; log returns are computed from Adjusted Close |
| Market volatility | VIX index (daily close) | Exogenous regressor; market "fear gauge", key risk regime indicator |
| Interest rates | 10-Year Treasury yield (DGS10), 10Y–2Y yield spread (T10Y2Y) | Macro exogenous regressors; rate changes and curve inversion signal regime shifts |
| Inflation (desirable) | CPI (monthly, CPIAUCSL) | Optional macro context; monthly frequency requires alignment to daily data |
| Derived features (computed, not sourced) | Log returns, lagged returns, rolling means, rolling volatility, weekday/seasonality flags, yield-change deltas | Feature engineering layer; built from the fields above |

Key principle inherited from the project design: the models consume **changes and surprises (Δ)**, not raw levels, so first differences of macro variables will be computed during preparation.

### Granularity

**One row per asset per trading day.** Daily frequency is the right balance: intraday data would add enormous volume and noise without improving a decision-support use case, while weekly/monthly data would leave too few observations for GARCH volatility modeling, which needs to capture day-to-day volatility clustering.

### Historical depth

**Approximately 15 years of daily data (January 2010 – present).** This depth is deliberately chosen to include multiple market regimes: the post-2008 recovery, the low-volatility 2017 period, the COVID-19 crash of March 2020, the 2022 inflation-driven bear market, and recent years. Volatility models are only meaningful if trained across calm and turbulent regimes. Data before 2010 is desirable but not essential (market microstructure changed materially after 2008).

### Approximate volume

~252 trading days/year × ~16 years ≈ **4,000 rows per asset**, × 4 assets ≈ **16,000 price rows**, plus ~4,000 rows for each daily macro series (VIX, DGS10, T10Y2Y). Total on the order of **30,000–35,000 rows** — small in storage terms (a few MB), but statistically sufficient for ARIMAX/GARCH estimation and walk-forward backtesting.

### Essential vs. desirable data

| Category | Data | Justification |
|---|---|---|
| **Essential** | Daily OHLCV + Adjusted Close for SPY (and the 3 stocks) | Without price history there is no project; adjusted prices are required to handle splits/dividends correctly |
| **Essential** | VIX daily series | Core exogenous risk regressor and central to the risk-aware framing |
| **Essential** | 10Y Treasury yield (DGS10) | Main macro regressor; long, clean, documented history |
| **Desirable** | Yield spread (T10Y2Y), CPI | Enrich the macro feature set; project works without them |
| **Desirable (advanced)** | News sentiment / earnings events | Listed as a future extension; explicitly out of scope for the MVP due to cost, complexity, and data-quality risk |

## 3. Planned Data Sources

| Source | Data provided | Access | Format | History | Link |
|---|---|---|---|---|---|
| **Yahoo Finance** (via the `yfinance` Python library) | Daily OHLCV + Adjusted Close for SPY, AAPL, MSFT, JPM; also VIX (^VIX) | Free, no API key, no registration | API → pandas DataFrame, persisted as CSV/Parquet | Decades of daily data (SPY since 1993) | https://finance.yahoo.com / https://pypi.org/project/yfinance/ |
| **FRED** (Federal Reserve Bank of St. Louis) | DGS10 (10Y Treasury yield), T10Y2Y (yield spread), CPIAUCSL (CPI) | Free, public; optional free API key via `fredapi` or direct CSV download | CSV / API | DGS10 since 1962; T10Y2Y since 1976; CPI since 1947 | https://fred.stlouisfed.org/series/DGS10 · https://fred.stlouisfed.org/series/T10Y2Y · https://fred.stlouisfed.org/series/CPIAUCSL |
| **CBOE** (backup for VIX) | Official VIX historical data | Free, public download | CSV | Since 1990 | https://www.cboe.com/tradable_products/vix/vix_historical_data/ |

**Openness:** all sources are free and publicly accessible with no payments, credentials, or restrictive licensing for academic use. FRED is an official government-backed source with excellent documentation.

**Historical availability:** confirmed for all series, far exceeding the 15-year target.

**Stability assessment:** FRED is institutionally maintained and extremely stable. Yahoo Finance is stable as a data provider but `yfinance` is an *unofficial* library that scrapes/wraps Yahoo's endpoints, which have changed in the past — this is the main source risk (see below).

**Detected risks:**

- **`yfinance` breakage:** Yahoo has historically changed its endpoints, temporarily breaking the library. Mitigation: download once, persist immutable raw snapshots in the repository (`data/raw/`), so the project never depends on live availability; fallback sources identified in Section 5.
- **Adjusted vs. unadjusted prices:** using unadjusted closes would corrupt return calculations around splits/dividends. Mitigation: always use Adjusted Close; documented in the data dictionary (Deliverable 3).
- **Calendar misalignment across sources:** FRED daily series contain values on some days markets are closed (and NaNs on others); CPI is monthly. Mitigation: join everything onto the official trading-day calendar of the price data, with explicit forward-fill rules for macro variables.
- **Rate limits:** `yfinance` applies informal throttling. Not a real risk at our volume (a handful of tickers, one-off downloads).
- **Survivorship bias:** mitigated by design — SPY (an index ETF) and three companies selected for liquidity, not past performance; the project draws no cross-sectional "which stock wins" conclusions.

## 4. Privacy and Data Protection Considerations

- **Personally identifiable information:** none. All data are aggregated, public market and macroeconomic series (prices, indices, yields). No individuals, accounts, transactions, or behavioral data are involved.
- **Anonymization / filtering:** not applicable — there is nothing personal to anonymize.
- **Safe academic use:** yes. All sources are public, free, and widely used in academic research; Yahoo Finance and FRED data are standard in the financial econometrics literature.
- **Ethical and legal risks:** the relevant ethical consideration is not privacy but **responsible framing of outputs**. The system will be explicitly presented as an academic decision-*support* prototype, not financial advice. Documentation and the dashboard will carry a clear disclaimer: forecasts are probabilistic, backtested performance does not guarantee future results, and the tool must not be used to make real investment decisions. This aligns with the project's "Limitations" analysis (low signal-to-noise ratio, black-swan events not captured).
- **Deliberately avoided data:** individual-level trading data, social-media user data, and paid alternative-data sources were considered and discarded — partly for privacy/licensing reasons and partly to keep the project fully reproducible with open data.

## 5. Initial Project Viability

**Is it feasible to obtain the necessary data?** Yes, with high confidence. All essential series are free, public, documented, and downloadable in minutes. A quick verification download of SPY and DGS10 has already been prototyped conceptually with `yfinance`/FRED and presents no barriers.

**Does the available information have sufficient quality, granularity, and depth?** Yes. Daily granularity over ~15 years yields ~4,000 observations per asset — ample for ARIMAX/GARCH estimation and walk-forward validation across multiple market regimes (COVID crash, 2022 bear market). FRED data are institutional-grade; Yahoo adjusted prices are the de-facto academic standard for this kind of project.

**Can the idea be developed realistically during the course?** Yes. The data volume is small, the modeling stack (statsmodels/`arch` in Python) is mature and well documented, and the MVP is deliberately scoped: one primary asset (SPY) with three additional tickers as a generalization check, a rule-based strategy layer, and a Streamlit dashboard. Advanced extensions (sentiment, ML models) are explicitly out of MVP scope.

**Riskiest part of the project right now?** Not data availability, but **signal extraction**: it is entirely possible that the ARIMAX mean model barely beats a naive baseline in return forecasting, because markets are close to efficient. This risk is managed by design: (1) the volatility model (GARCH) is expected to add clear value regardless, since volatility *is* predictable; (2) the project's success criterion is a rigorous, honest, risk-aware decision framework — a well-calibrated system that says "HOLD, uncertainty is high" is a valid and defensible result; (3) the backtest compares against buy-and-hold, making any conclusion (including "the strategy does not beat the market") a legitimate, evidence-based finding.

**Alternative if the main data source fails?** If `yfinance` breaks: (1) **Stooq** (free CSV downloads of daily OHLCV, stable, no key), (2) **Alpha Vantage** (free API key tier, daily adjusted series), (3) official **CBOE** files for VIX, and (4) static Kaggle historical dumps as a last resort for reproducibility. Because raw snapshots are versioned in the repository from day one, a source failure affects only future updates, never the reproducibility of the delivered project.

---

*This document corresponds to Deliverable 2 of the Data Science / AI Master's Final Project. Deliverable 1 (`01_ideas_producto.md`) is maintained in this same directory for traceability.*
