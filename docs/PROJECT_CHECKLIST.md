# Project Checklist — RiskLens

Deadline: TBD (plan assumes 7 working days). Tick `[x]` as items are finished.

## Decisions locked

- Scope: SPY first; AAPL/MSFT/JPM if time allows (code is per-ticker from the start).
- Test window: 2024-01 to the last trading day of the frozen data vintage.
- SELL = flat (long-only). Costs: 5 bps per position change.
- Raw data stays committed (documented design); README carries a source-attribution note.
- Cut unless time remains: EGARCH, live re-download button, LLM wording pass, interactive calibration page.
- Tooling: `uv` (Python 3.12, lockfile), `ruff` (lint + format), `pytest`. No MLflow/DVC/Docker/pre-commit for a one-week scope.
- Virtualenv lives outside OneDrive: `UV_PROJECT_ENVIRONMENT=C:\Users\nazmu\.venvs\risklens`.

## Already done

- [x] Deliverables 1–5 written, revised per feedback, pushed to GitHub
- [x] README, LICENSE, `.gitignore`
- [x] RiskLens mockup (numbers illustrative; regenerate from real output)

## Day 1 — environment, data, gold layer

- [x] `pyproject.toml`, `uv sync` (Python 3.12), `uv.lock` created (not yet committed)
- [x] `.gitignore` additions (`.env`, `.streamlit/secrets.toml`, `.ipynb_checkpoints/`)
- [x] `src/risklens/ingest.py` — 8 raw files in `data/raw/`, `VINTAGE.json` manifest with hashes (vintage 2026-09-19, last price date 2026-09-18)
- [x] `src/risklens/clean.py` — `data/processed/` (snake_case, tz-naive, dedupe, FRED `"."`/blank to NaN, trading-day spine, forward-fill only)
- [x] `src/risklens/build_gold.py` — `gold_market_daily.parquet` (16,728 rows, 4 tickers, 2010-02-03 to 2026-09-18)
- [x] `tests/test_pipeline.py` — 6 tests pass: PK unique, `adj_close > 0`, finite returns, no NaN after warm-up, no look-ahead, forward-fill/lag, CPI as-of
- [x] FRED lag: DGS10 was missing the latest day (2026-09-18) at download, so FRED series are lagged 1 trading day (`FRED_LAG_DAYS`)
- [ ] Commit Day 1 work; freeze vintage + test window; tag commit `vintage-2026-09-19`
- [ ] README note on data sources/attribution

## Day 2 — EDA and visualization

- [x] `src/risklens/eda.py` (train + validation only, test period excluded): Q1 stationarity (ADF/KPSS)
- [x] Q2 ACF/PACF of returns
- [x] Q3 volatility clustering (ACF of squared returns, rolling vol)
- [x] Q4 heavy tails (QQ-plot, skew/kurtosis, largest moves)
- [x] Q5 exogenous variables vs next-day returns/vol
- [x] Q6 regime comparison (2010–19, 2020, 2021, 2022, 2023)
- [x] Q7 leverage effect (supports GJR/EGARCH as R2)
- [x] Q8 weekday effects (none; drop `day_of_week`)
- [x] Figures in `reports/figures/` (7); H1–H4 verdicts in `reports/eda_findings.md`
- [ ] Notebook version of the EDA for the appendix (optional, only if time remains)
- [ ] Short data-quality report (rows, holidays, COVID extremes kept)

## Day 3 — baselines and models (train + validation only)

- [x] Baselines: naive zero return, 21-day rolling vol, EWMA (buy-and-hold comes with the Day 4 backtest)
- [x] GARCH(1,1) Student-t and GJR-GARCH (asymmetric variant, replaces EGARCH), walk-forward (expanding, refit every 21 days)
- [x] Risk metrics: QLIKE, MAE, 95% coverage (Kupiec, Christoffersen), Diebold-Mariano vs rolling vol
- [x] ARIMA then ARIMAX (VIX/yield changes, lagged); order by BIC on train (2,0,0)
- [x] Mean metrics vs naive: RMSE, MAE, directional accuracy, Diebold-Mariano
- [x] Tests: GARCH recursion matches `arch`, mean walk-forward has no look-ahead, Kupiec/Christoffersen/DM (12 tests pass)
- [x] Results in `reports/model_findings.md`: GARCH gives calibrated intervals but does not significantly beat rolling vol; no mean model beats naive
- [ ] Residual diagnostics (Ljung-Box, ARCH-LM) on the chosen models
- [ ] Risk buckets low/medium/high from training percentiles only

## Day 4 — rule, pre-registration, sealed test

- [x] `strategy.py`: score rule (A) + volatility filter (B), grid on validation (12 + 7 combos) with min-trades and drawdown constraints
- [x] Backtest: primary execution at t+1 close (decision at t earns the return of t+2), 5 bps costs, bootstrap CIs; same-close reported as optimistic sensitivity
- [x] Pre-registered: config + frozen code committed and tagged `preregistered-v1` before the test window was opened
- [x] Sealed test opened once through a gated runner (fails if frozen files differ from the tag); `gold_signals_daily.parquet` written (1,183 rows: validation, test, 1 live), invariants asserted
- [x] Acceptance (D4 §7) documented in `reports/test_results.md`: criterion 2 met (GARCH beats rolling vol, calibrated); criteria 1 and 3 not met, so signals stay EXPERIMENTAL
- [ ] Fix `date` semantics in D3 (date = forecast target date, `origin_date` = information date) and add the new columns to the D3 data dictionary

## Day 5 — Streamlit app

- [x] App reads only precomputed outputs (gold parquet files, test metrics JSON, config); no fitting at runtime (`app/streamlit_app.py`, `src/risklens/dashboard_data.py`, `charts.py`)
- [x] Zone 1: asset, date picker, Run forecast, audit trail with look-ahead check, data health, protocol cards
- [x] Zone 2: risk gauge (percentile), forecast vs realized volatility, fan chart 50/80/95, regime timeline with historical analogue, coverage KPI
- [x] Zone 3: signal card + EXPERIMENTAL badge, score vs threshold, grounded template explanation, equity curves and strategy table vs buy-and-hold
- [x] Stale-data banner (shows when the last close is 2+ days old), disclaimer footer, calibration report page
- [x] Charts follow the dataviz palette (validated on the app surface); tests build every chart and render both pages (31 tests pass)
- [x] Deployed to Streamlit Community Cloud: https://risklensspy.streamlit.app/ (fresh-clone install from `requirements.txt` verified)
- [ ] Confirm the app opens in a private window without signing in (Share setting must be public)
- [ ] Optional: extension tickers AAPL/MSFT/JPM (would need per-ticker calibration and a new pre-registration tag)

## Day 6 — presentation and polish

- [x] Slides (12, English, 15 minutes): problem, approach, data, protocol, EDA, validation, risk results, returns and signals, dashboard, trust, conclusions (published as a private artifact with speaker notes and timings)
- [x] Real screenshot and final metrics in slides
- [x] README updated (results, structure, how to run, source attribution); implementation notes added
- [ ] Rehearse once against the timings in the speaker notes; export the deck to PDF as a backup
- [ ] One-page results summary (optional)

## Day 7 — buffer and rehearsal

- [ ] Full run-through with the live app
- [ ] 2–3 min fallback demo video
- [ ] Fresh clone → `uv sync` → run reproduces results
- [ ] Prepared answers: weak return model, test window choice, leakage controls

## Risks

- Return model likely ≈ naive: report as a finding; risk system is the core.
- Time slip on Day 3: ARIMA only, SPY only.
- Live demo failure: fallback video + screenshots.
