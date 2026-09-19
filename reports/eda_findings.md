# EDA findings (train + validation, 2010-02 to 2023-12; test period untouched)

Reproduce: `uv run python -m risklens.eda` (tables in `eda_summary.md`, figures in `figures/`).

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 Returns stationary, near-zero autocorrelation | Stationary yes; autocorrelation small but not zero | ADF p≈0 and KPSS p≥0.10 on returns; prices ADF p 0.85–0.99, KPSS p=0.01. SPY lag-1 ACF -0.105, only -0.03 without 2020. Expect at most a marginal edge over the naive forecast |
| H2 Volatility is autocorrelated and predictable | Confirmed | Squared and absolute return ACF significant to lag 30 (fig 03); `roll_vol_21` vs next-day abs return r=0.455 |
| H3 VIX changes explain next-day volatility more than returns | Not supported | `vix_change` vs next-day return r=+0.111, vs next-day abs return r=+0.046. Positive sign: VIX spikes tend to precede rebounds. Realized volatility is the strong volatility predictor |
| H4 Extremes cluster in regimes | Confirmed | 47% of the top-1% |return| days fall in 2020, 19% in 2011, 14% in 2022 |

## Decisions that follow

- **Heavy tails (Q4):** SPY excess kurtosis 11.6, skew -0.72, Jarque-Bera p≈0. Use Student-t innovations and t-based intervals.
- **Leverage effect (Q7):** corr(r_t, |r_t+1|) = -0.12; mean |next return| is 0.83% after down days vs 0.64% after up days (p≈0). Include GJR/EGARCH as the R2 candidate.
- **Weekday (Q8):** Kruskal-Wallis p=0.97 (returns), 0.60 (abs returns). Drop `day_of_week` from the models.
- **Exogenous (Q5):** `dgs10_change` and `t10y2y` show no link to next-day returns (r≈0.00) and only weak links to next-day abs return (r≈-0.08). ARIMAX keeps `vix_change` as the main regressor; yields are optional.
- **Regimes (Q6):** annualized SPY volatility 14.7% (2010–19), 33.6% (2020), 13.0% (2021), 24.3% (2022), 13.1% (2023). Report errors per regime.
- **Caution:** several correlations are dominated by March 2020. Check stability by refitting without 2020 before trusting any mean-model coefficient.
