# Deliverable 1 — Product Ideas

**Máster en Data Science — Trabajo de Fin de Máster**

This document records the product ideas explored at the start of the project, maintaining the traceability required by the course. The work evolved through two documented iterations, from a pure forecasting system toward a complete decision-support product.

---

## Idea A (initial version) — Risk-Aware Stock Forecasting System

**Concept.** A data-driven system providing probabilistic forecasts of stock price movements and their associated risk (volatility) to support informed investment decisions.

**Key design principles:**
- Model **log returns** (price changes) rather than raw prices, acknowledging the low signal-to-noise ratio of financial markets.
- Combine a **mean model** (ARIMAX on log returns with exogenous market/macro drivers) with a **volatility model** (GARCH 1,1; optionally EGARCH for asymmetric shocks).
- Use **changes and surprises (Δ), not raw values**, as model inputs: lagged returns, rolling statistics, VIX, interest-rate changes, yield spread.

**Output:** forecasted price range (via return reconstruction) and a risk level (low / medium / high).

**Evaluation:** time-based train/validation/test split, walk-forward validation, RMSE/MAE/MAPE, predicted-vs-realized volatility comparison, and prediction-interval coverage (calibration).

**Positioning:** not a "price predictor" but a decision-support system — explaining decisions, not just predictions (ARIMAX coefficients for momentum/macro influence, GARCH parameters for volatility persistence).

**Limitation identified:** the system stopped at *prediction*. It quantified expected movement and risk, but left the final trading decision entirely to the user, limiting its practical value.

---

## Idea B (revised version — SELECTED) — Risk-Aware Stock Forecasting and Trading Decision Support System

**Concept.** An evolution of Idea A that closes the gap between prediction and action. It keeps the full forecasting core (ARIMAX + GARCH) and adds two new layers:

1. **Trading Strategy & Decision Engine.** A rule-based framework converts model outputs into actionable signals:
   - **BUY** — expected return sufficiently positive and volatility within acceptable limits
   - **SELL** — expected return sufficiently negative
   - **HOLD** — weak signal or high uncertainty
   
   The core decision variable is a **risk-adjusted signal**: `Signal = Expected Return / Forecasted Volatility`, ensuring both return potential and risk enter every decision. Optional extension: risk-based position sizing.

2. **Backtesting & Strategy Evaluation.** A walk-forward simulation applies the generated signals to historical data and tracks portfolio performance using cumulative returns, Sharpe ratio, maximum drawdown, and hit ratio — validating that the strategy is not only theoretically sound but practically effective.

**Why this idea was selected:**
- It transforms a forecasting exercise into a **complete, business-oriented product**: forecast → risk assessment → decision → validated performance.
- It demonstrates a wider range of Data Science capabilities: time-series modeling, risk modeling, strategy design, backtesting, and rigorous evaluation.
- It remains fully feasible with open, free data sources (market prices, VIX, Treasury yields) and mature Python tooling.
- Its honest treatment of market unpredictability (probabilistic outputs, explicit limitations, black-swan caveats) makes it defensible academically.

**Known limitations (accepted by design):** financial markets are inherently hard to predict; models provide probabilistic insights, not guarantees; extreme events may not be captured; signals depend on model assumptions and threshold choices.

**Future extensions (out of MVP scope):** strategy parameter optimization, event-based analysis (earnings/news), NLP sentiment modeling, interactive Streamlit dashboard as productized front-end, and ML/multivariate models (VAR, XGBoost, LSTM).

---

## Final selection

**Idea B — Risk-Aware Stock Forecasting and Trading Decision Support System** is the idea selected for the remainder of the course. Deliverable 2 (`02_datos_necesarios.md`) develops its data requirements, sources, privacy considerations, and initial viability analysis.
