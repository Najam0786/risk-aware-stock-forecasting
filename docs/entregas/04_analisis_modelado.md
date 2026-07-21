# Deliverable 4 — Analysis Design and Modeling Strategy

**Project:** Risk-Aware Stock Forecasting and Trading Decision Support System
**Máster en Data Science — Trabajo de Fin de Máster**

---

## 1. Problem Being Solved

**What happens currently and why it is a problem.** A retail investor deciding whether to buy, sell, or hold an asset today has no quantitative view of two things at once: the *expected* short-term movement and the *uncertainty* around it. Free tools show charts and point predictions; the risk dimension is invisible. The practical consequence is systematically bad timing — entering positions in deceptively calm markets and exiting in panic during volatile ones — because decisions are driven by emotion instead of a consistent, risk-aware rule.

**Who will use the result and for what decision.** The end user is a non-professional investor (represented in the MVP by the dashboard user). The decision supported is concrete and daily: *given today's information, should I buy, sell, or hold asset X — and how much confidence does that recommendation deserve?*

**What concrete result the project must produce to be useful.** For each asset and trading day, the system must output: a next-day expected log return with a prediction interval, a next-day volatility forecast mapped to a risk level (low/medium/high), a BUY/SELL/HOLD signal derived from the risk-adjusted score, and backtested evidence of how that rule would have performed historically against simply holding the market. The project is useful if the recommendations are *calibrated and honest* — including transparently showing when uncertainty is too high to act (HOLD).

## 2. Planned Analysis (EDA) and Expected Usefulness

The analysis is organized around questions that directly validate modeling assumptions or feed the MVP — not a generic gallery of plots.

**Questions to answer with the data:**

| # | Question | Analysis | Why it matters |
|---|---|---|---|
| Q1 | Are raw prices non-stationary, and are log returns stationary? | Price vs. log-return series plots; ADF/KPSS tests per ticker | Validates the core design decision to model returns, not prices (prerequisite for ARIMA-family models) |
| Q2 | Do returns show autocorrelation that a mean model can exploit? | ACF/PACF of log returns | Directly informs ARIMA(p,q) order selection; if autocorrelation is negligible, anticipates the "hard to beat the naive baseline" risk |
| Q3 | Is volatility clustering present? | ACF of squared/absolute returns; rolling 21-day volatility plots | Justifies GARCH; this is the pattern the risk model exists to capture |
| Q4 | Are return distributions heavy-tailed and asymmetric? | Histograms + QQ-plots vs. normal; skewness/kurtosis; largest daily moves table | Motivates Student-t innovations in GARCH and honest prediction intervals |
| Q5 | Do exogenous variables carry signal? | Cross-correlation of VIX changes / yield changes vs. next-day returns and volatility; scatter by regime | Justifies (or challenges) the "X" in ARIMAX before adding complexity |
| Q6 | Do market regimes differ visibly? | Volatility and return comparison across sub-periods (2010–19 calm, 2020 crash, 2022 bear, 2023+ recovery) | Feeds walk-forward design; regime contrast becomes a key dashboard visual |
| Q7 | Is there leverage asymmetry (bad news → more volatility)? | Correlation between negative returns and subsequent volatility | Decides whether EGARCH is worth testing |
| Q8 | Are there calendar effects? | Mean return/volatility by weekday | Cheap check on the `day_of_week` feature; dropped if empty |

**Hypotheses to verify:** H1 — log returns are approximately stationary with near-zero autocorrelation (markets are near-efficient); H2 — volatility is strongly autocorrelated and therefore predictable; H3 — VIX changes correlate with next-day volatility more than with next-day returns; H4 — extreme moves cluster in identifiable regimes.

**What flows into the MVP:** the price/return/rolling-volatility charts, the regime comparison, and the distribution visuals become dashboard components; ACF diagnostics and stationarity tests become the technical appendix justifying model choices in the final defense.

**How the analysis helps:** every EDA output either (a) validates a modeling assumption before spending effort, (b) selects model orders/variants, or (c) becomes a user-facing explanation of *why* the system reasons in terms of risk.

## 3. Model Types to Be Compared

**Task type.** Two coupled forecasting tasks on daily data, followed by a deterministic rule layer:
1. **Mean task** — one-step-ahead regression of next-day log return (continuous target).
2. **Risk task** — one-step-ahead forecast of conditional volatility (continuous, positive target).
3. **Decision layer** — not a learned model: a transparent rule mapping (expected return, volatility) → BUY/SELL/HOLD. Justified as rule-based deliberately: interpretability is a product requirement, and a learned policy would multiply data needs and overfitting risk.

**Model comparison table:**

| Alternative | Type | Why it is proposed | Main limitation |
|---|---|---|---|
| **Baseline (mean):** naive zero-return forecast (random walk) | Naive | The efficient-market benchmark; trivially simple yet notoriously hard to beat in finance — exactly the "demanding baseline" the methodology requires | Provides no signal at all; any useful model must beat it out-of-sample |
| **Baseline (risk):** 21-day rolling historical volatility | Naive statistical | Simple, widely used practitioner benchmark for volatility | Reacts slowly to regime changes; no forward-looking component |
| **Baseline (strategy):** buy-and-hold SPY | Passive rule | The real-world alternative any trading system must justify itself against | Full market drawdown exposure |
| **Candidate M1:** ARIMA(p,0,q) on log returns | Interpretable statistical | Captures any exploitable autocorrelation; coefficients are directly interpretable; fast to fit in walk-forward | Linear; likely only marginal edge over naive given near-efficiency |
| **Candidate M2:** ARIMAX (ARIMA + exogenous VIX/yield changes) | Interpretable statistical + external drivers | Tests the project's core hypothesis that market/macro context improves return forecasts; coefficient signs tell an economic story | Risk of adding noise; exogenous values must be strictly lagged to avoid leakage |
| **Candidate R1:** GARCH(1,1), Student-t innovations | Volatility model | The canonical volatility-clustering model; persistence parameters (α, β) interpretable; expected to clearly beat rolling vol | Symmetric response to good/bad news |
| **Candidate R2 (optional):** EGARCH | Asymmetric volatility model | Captures leverage effect if EDA (Q7) shows asymmetry | More parameters; only kept if it improves validation metrics |

Scope discipline: ML models (XGBoost, LSTM) remain explicitly out of MVP scope, as declared since Deliverable 1 — the comparison above is small, coherent, and answerable within the course timeline.

## 4. Input Data for the Analysis and Models

**Source:** `gold_market_daily.parquet` — the gold-layer contract defined in Deliverable 3.

| Input | Description | Granularity / type | Use |
|---|---|---|---|
| `gold_market_daily` | Model-ready market dataset (prices, returns, features, macro regressors) | One row per (trading date, ticker); ~16,000 rows, 4 tickers, 2010–present | Sole input for EDA and model training |
| `log_return` | Daily log return of adjusted close | float | **Target** of the mean task; input to volatility estimation |
| `ret_lag_1`…`ret_lag_5` | Lagged returns | float | ARIMA/ARIMAX autoregressive information |
| `vix_change`, `dgs10_change` | Daily changes in VIX and 10Y yield | float | Exogenous regressors (ARIMAX), lagged one day |
| `t10y2y` | Yield-curve spread | float | Secondary exogenous regressor (regime context) |
| `roll_vol_21` | 21-day realized volatility | float | Risk baseline and GARCH evaluation reference |
| `day_of_week` | Weekday indicator | int | Tested in EDA (Q8); kept only if signal exists |

**Primary key and reference date:** (`date`, `ticker`); every feature on row *t* describes information available at the close of day *t*, and the model predicts day *t+1*.

**Variables excluded and why:**
- `open`, `high`, `low`, raw `close` — redundant with adjusted close for this design; intraday range modeling is out of scope.
- `volume` — kept in gold for context/EDA but not fed to ARIMAX in the MVP (weak documented link to next-day returns; reduces model surface).
- `cpi_yoy` — desirable-tier variable; monthly frequency adds as-of complexity with low expected daily signal. Revisited only if ARIMAX shows macro variables matter.
- Same-day (t+1) values of any exogenous variable — **prohibited as leakage**: when predicting day t+1, only information up to day t's close may be used.

**Availability at prediction time.** All inputs are known at market close of day *t*: prices/VIX close simultaneously, and FRED daily series are aligned with forward-fill (Deliverable 3, §8), so no field in the feature row depends on future information. This time-availability audit is re-verified in code before training.

## 5. Output Data and Form of Consumption

**Destination:** `gold_signals_daily.parquet` — the output contract already defined in Deliverable 3, §4.2.

| Output field | Description | Type | Consumed by |
|---|---|---|---|
| `date`, `ticker` | Forecast target day and asset (PK) | date / string | Traceability joins back to `gold_market_daily` |
| `expected_return` | ARIMAX one-step-ahead forecast of log return | float | Decision rule; dashboard forecast panel |
| `pi_lower`, `pi_upper` | 95% prediction interval for the return | float | Honest uncertainty display; calibration evaluation |
| `forecast_volatility` | GARCH one-step-ahead conditional volatility | float | Decision rule; risk level mapping |
| `risk_level` | low / medium / high volatility bucket | category | Dashboard risk indicator |
| `risk_adjusted_score` | `expected_return / forecast_volatility` | float | The single decision variable |
| `signal` | BUY / SELL / HOLD | category | Backtest engine; dashboard recommendation |
| `run_timestamp` | Generation time | datetime | Reproducibility control |

**Granularity of output:** one row per (forecast date, ticker) over the evaluation window — identical grain to the input, guaranteeing 1:1 traceability from every recommendation to the exact data that produced it.

**Format and consumption path:** Parquet file → (1) backtesting engine simulates the signals and produces performance metrics; (2) Streamlit dashboard shows, per asset: current signal, expected return with interval, risk level, and historical backtest curve vs. buy-and-hold.

**How the user acts on it:** the user reads a recommendation plus its confidence context and decides whether to follow it; a HOLD with "high uncertainty" is explicitly designed to be as informative as a BUY. Every signal display includes the prediction interval and risk level — never a bare point forecast — plus the standing academic disclaimer (not financial advice).

## 6. Strategy for Designing and Selecting the Model

Process, in order:

1. **Modeling dataset preparation** — load `gold_market_daily`, trim feature warm-up NaNs, freeze the temporal split (below) before any model sees the data.
2. **Target definition** — next-day log return (mean task); next-day conditional volatility, evaluated against realized proxies (risk task).
3. **Baselines first** — naive zero-return, rolling-vol, and buy-and-hold are implemented and scored *before* any candidate, fixing the bar.
4. **Candidates, few and justified** — M1 → M2 for the mean; R1 (→ R2 only if EDA Q7 supports it) for risk. Orders (p,q) selected on train via AIC/BIC, confirmed on validation — never on test.
5. **Preprocessing** — minimal by design: gold data is already clean; exogenous regressors standardized on training statistics only; no imputation beyond the gold-layer rules; residual diagnostics (Ljung-Box, ARCH-LM) after each fit.
6. **Comparison criteria** — predictive quality vs. baseline, stability across walk-forward windows, interpretability of coefficients, computational cost of re-fitting, and usefulness of outputs for the decision layer (e.g., interval calibration matters more than raw RMSE for the product).
7. **Final decision rule** — a candidate is selected iff: (a) it beats its baseline on the primary metric on validation, (b) the advantage persists in walk-forward test evaluation (not one lucky window), (c) residual diagnostics are acceptable, and (d) added complexity is justified — when in doubt, the simpler model wins. A model that maximizes a metric but produces unstable or uninterpretable behavior is rejected: the deliverable is a *decision-support product*, not a leaderboard score.

## 7. Validation and Evaluation Strategy

| Element | Planned decision | Justification |
|---|---|---|
| Data split | **Temporal**: Train 2010–2021 · Validation 2022–2023 · Test 2024–present | Random splits are invalid for time series; validation contains a bear market (2022) — a demanding, realistic tuning period; test is fully unseen recent data |
| Walk-forward | Rolling re-fit on the test period (expanding window, refit every ~21 trading days) | Reproduces real deployment: the model only ever predicts with past data; measures stability, not one lucky split |
| Leakage prevention | Features strictly from day ≤ t for predicting t+1; forward-fill only; scalers fit on train only; per-ticker series never mixed across split boundaries | Enforced in code with automated time-availability checks; leakage is the project's most explicitly monitored risk (see §8) |
| Mean-task metrics | **RMSE / MAE vs. naive baseline** (primary), directional accuracy (secondary) | Point-error metrics on returns are the standard; MAPE is *rejected* — returns cross zero, making MAPE undefined/explosive |
| Risk-task metrics | RMSE and QLIKE against realized volatility proxy (squared returns / rolling realized vol); **95% prediction-interval coverage** | QLIKE is the standard volatility loss; coverage ≈ 95% is the calibration test — the product's honesty metric |
| Strategy metrics | Cumulative return, **Sharpe ratio**, maximum drawdown, hit ratio — vs. buy-and-hold, including a transaction-cost sensitivity check (e.g., 5 bps/trade) | Measures whether signals add value after realism adjustments, not just in a frictionless world |
| Error analysis | Per-regime evaluation (calm vs. crisis sub-periods), per-ticker breakdown, worst-day inspection, interval-failure analysis | Reveals *when* the system fails — essential for a risk-management product and for the defense |
| Baseline comparison | Every reported metric always shown side-by-side with its baseline | The improvement, not the absolute number, is the result |

**Minimum acceptable result and contingency.** The system is acceptable if **at least one** holds on test: (1) ARIMAX beats naive RMSE with stable walk-forward advantage; (2) GARCH beats rolling-vol on QLIKE **and** achieves interval coverage in the 90–97% band; (3) the signal strategy achieves a Sharpe ratio ≥ buy-and-hold with lower maximum drawdown. Expectation, stated honestly: (2) is highly likely (volatility is predictable), (1) is genuinely uncertain (markets are near-efficient), (3) is exploratory. If **no** criterion is met, the deliverable pivots to a *calibrated risk-assessment system* (volatility + intervals + HOLD-biased signals) with a rigorous negative result on return predictability — methodologically valid, fully documented, and defensible.

## 8. Risks and Alternatives

- **Is the target available and representative?** Yes — `log_return` is computed deterministically from adjusted prices for every trading day; no label scarcity, no class imbalance (continuous target). The volatility "target" is latent (never directly observed), which is why evaluation uses standard realized proxies — a known, documented limitation of all volatility modeling.
- **Data leakage risk.** The project's #1 methodological risk, with three concrete traps identified: same-day exogenous values (VIX at close of t+1 used to predict t+1), scalers or model orders selected using test data, and backward-filled macro gaps. Mitigations are already embedded: lag-only features, train-only fitting of all statistics, forward-fill-only policy (Deliverable 3), and an automated pre-training audit asserting every feature's availability date.
- **Volume, history, quality sufficiency.** ~4,000 daily observations per ticker across multiple regimes is ample for ARIMA/GARCH-family estimation (parametric, few parameters) and supports walk-forward evaluation with wide windows. Quality risks were mapped in Deliverable 3 and are handled upstream of modeling.
- **Temporal instability / regime shifts.** Model parameters fitted in calm periods may fail in turbulent ones. Mitigated by walk-forward re-fitting, per-regime error reporting, and the risk layer itself (rising GARCH volatility automatically pushes signals toward HOLD in turmoil).
- **Greatest uncertainty in the strategy.** Whether the mean model extracts *any* stable signal beyond noise — acknowledged since Deliverable 1 ("low signal-to-noise ratio"). The architecture is deliberately robust to this: the risk half of the system carries value independently.
- **If no model beats the baseline or validation fails.** Documented fallback chain: (1) reduce to SPY-only modeling (cleaner series, fewer comparisons); (2) simplify exogenous set to VIX only; (3) re-scope the product to calibrated volatility forecasting + uncertainty-aware HOLD/alert signals, presenting the return-predictability analysis as an honest empirical finding. In all cases the MVP (pipeline → models → signals → backtest → dashboard) remains demonstrable end-to-end.

---

*This document corresponds to Deliverable 4. Deliverables 1–3 are maintained unchanged in this same directory for traceability. The input contract (§4) and output contract (§5) reference the gold layer defined in Deliverable 3 without modification.*
