from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "gold_market_daily.parquet"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "eda_summary.md"
EDA_END = "2023-12-31"
PRIMARY = "SPY"
TRADING_DAYS = 252
REGIMES = {
    "2010-2019 calm": ("2010-01-01", "2019-12-31"),
    "2020 COVID": ("2020-01-01", "2020-12-31"),
    "2021 recovery": ("2021-01-01", "2021-12-31"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
    "2023 recovery": ("2023-01-01", "2023-12-31"),
}


def load_eda_data() -> pd.DataFrame:
    gold = pd.read_parquet(GOLD_PATH)
    return gold[gold["date"] <= EDA_END].copy()


def stationarity(gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in gold.groupby("ticker"):
        g = g.sort_values("date")
        for name, series in (
            ("log_price", np.log(g["adj_close"])),
            ("log_return", g["log_return"]),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_p = kpss(series, regression="c", nlags="auto")[1]
            rows.append(
                {
                    "ticker": ticker,
                    "series": name,
                    "adf_p": adfuller(series, autolag="AIC", result_object=False)[1],
                    "kpss_p": kpss_p,
                }
            )
    return pd.DataFrame(rows)


def ljung_box_table(gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in gold.groupby("ticker"):
        r = g.sort_values("date")["log_return"]
        rows.append(
            {
                "ticker": ticker,
                "returns_p_lag10": acorr_ljungbox(r, lags=[10])["lb_pvalue"].iloc[0],
                "squared_p_lag10": acorr_ljungbox(r**2, lags=[10])["lb_pvalue"].iloc[0],
                "abs_p_lag10": acorr_ljungbox(r.abs(), lags=[10])["lb_pvalue"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def distribution_table(gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in gold.groupby("ticker"):
        r = g["log_return"]
        rows.append(
            {
                "ticker": ticker,
                "n": len(r),
                "mean_bp": r.mean() * 1e4,
                "std_pct": r.std() * 100,
                "skew": stats.skew(r),
                "excess_kurtosis": stats.kurtosis(r),
                "jarque_bera_p": stats.jarque_bera(r).pvalue,
                "min_pct": r.min() * 100,
                "max_pct": r.max() * 100,
            }
        )
    return pd.DataFrame(rows)


def largest_moves(spy: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    top = spy.reindex(spy["log_return"].abs().sort_values(ascending=False).index).head(n)
    return top[["date", "log_return", "vix_close"]].assign(
        log_return=lambda d: d["log_return"] * 100
    )


def exog_correlations(spy: pd.DataFrame) -> pd.DataFrame:
    spy = spy.sort_values("date")
    targets = {
        "next_day_return": spy["log_return"].shift(-1),
        "next_day_abs_return": spy["log_return"].abs().shift(-1),
    }
    rows = []
    for feature in ("vix_change", "dgs10_change", "t10y2y", "roll_vol_21"):
        for target_name, target in targets.items():
            pair = pd.concat([spy[feature], target], axis=1).dropna()
            r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append({"feature": feature, "target": target_name, "pearson_r": r, "p": p})
    return pd.DataFrame(rows)


def regime_table(spy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (start, end) in REGIMES.items():
        r = spy[(spy["date"] >= start) & (spy["date"] <= end)]["log_return"]
        rows.append(
            {
                "regime": name,
                "days": len(r),
                "ann_return_pct": r.mean() * TRADING_DAYS * 100,
                "ann_vol_pct": r.std() * np.sqrt(TRADING_DAYS) * 100,
                "worst_day_pct": r.min() * 100,
                "best_day_pct": r.max() * 100,
            }
        )
    return pd.DataFrame(rows)


def leverage_table(spy: pd.DataFrame) -> dict[str, float]:
    spy = spy.sort_values("date")
    r, nxt = spy["log_return"], spy["log_return"].shift(-1)
    pair = pd.concat([r, nxt], axis=1).dropna()
    after_down = pair[pair.iloc[:, 0] < 0].iloc[:, 1].abs()
    after_up = pair[pair.iloc[:, 0] > 0].iloc[:, 1].abs()
    return {
        "corr_r_t_vs_abs_r_t1": stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1].abs())[0],
        "corr_r_t_vs_sq_r_t1": stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1] ** 2)[0],
        "mean_abs_next_after_down_pct": after_down.mean() * 100,
        "mean_abs_next_after_up_pct": after_up.mean() * 100,
        "mannwhitney_p": stats.mannwhitneyu(after_down, after_up).pvalue,
    }


def weekday_table(spy: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    g = spy.assign(abs_ret=spy["log_return"].abs())
    table = g.groupby("day_of_week").agg(
        mean_bp=("log_return", lambda s: s.mean() * 1e4),
        std_pct=("log_return", lambda s: s.std() * 100),
        n=("log_return", "size"),
    )
    table.index = labels
    groups = [d["log_return"] for _, d in g.groupby("day_of_week")]
    abs_groups = [d["abs_ret"] for _, d in g.groupby("day_of_week")]
    return table, stats.kruskal(*groups).pvalue, stats.kruskal(*abs_groups).pvalue


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def make_figures(spy: pd.DataFrame, regimes: pd.DataFrame) -> None:
    spy = spy.sort_values("date").set_index("date")
    r = spy["log_return"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(spy["adj_close"], color="#1f77b4")
    axes[0].set_title("SPY adjusted close (non-stationary)")
    axes[1].plot(r * 100, color="#444", lw=0.6)
    axes[1].set_title("SPY daily log return, % (stationary, volatility clusters)")
    save(fig, "01_price_vs_return.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_acf(r, lags=30, ax=axes[0], title="ACF of returns")
    plot_pacf(r, lags=30, ax=axes[1], title="PACF of returns", method="ywm")
    save(fig, "02_acf_pacf_returns.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_acf(r**2, lags=30, ax=axes[0], title="ACF of squared returns")
    plot_acf(r.abs(), lags=30, ax=axes[1], title="ACF of absolute returns")
    save(fig, "03_acf_squared_returns.png")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(spy["roll_vol_21"] * np.sqrt(TRADING_DAYS) * 100, color="#d98e04")
    ax.set_title("SPY 21-day realized volatility (annualized, %)")
    save(fig, "04_rolling_volatility.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(r * 100, bins=120, density=True, color="#1f77b4", alpha=0.8)
    grid = np.linspace(r.min() * 100, r.max() * 100, 400)
    axes[0].plot(grid, stats.norm.pdf(grid, r.mean() * 100, r.std() * 100), color="red", lw=1)
    axes[0].set_title("Return distribution vs normal (red)")
    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot vs normal")
    save(fig, "05_distribution_qq.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(regimes["regime"], regimes["ann_vol_pct"], color="#d98e04")
    axes[0].set_title("Annualized volatility by regime, %")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(regimes["regime"], regimes["worst_day_pct"], color="#c0392b")
    axes[1].set_title("Worst day by regime, %")
    axes[1].tick_params(axis="x", rotation=25)
    save(fig, "06_regimes.png")

    nxt = r.shift(-1).abs() * 100
    pair = pd.concat([r * 100, nxt], axis=1).dropna()
    pair.columns = ["r_t", "abs_r_t1"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(pair["r_t"], pair["abs_r_t1"], s=4, alpha=0.3)
    ax.set_xlabel("return on day t, %")
    ax.set_ylabel("|return| on day t+1, %")
    ax.set_title("Leverage effect: bad days precede larger moves")
    save(fig, "07_leverage.png")


def frame_block(df: pd.DataFrame, digits: int = 4) -> str:
    numeric = df.select_dtypes("number").columns
    rounded = df.assign(**{c: df[c].round(digits) for c in numeric})
    return "```\n" + rounded.to_string(index=False) + "\n```\n"


def build_report() -> str:
    gold = load_eda_data()
    spy = gold[gold["ticker"] == PRIMARY].sort_values("date")
    stat = stationarity(gold)
    lb = ljung_box_table(gold)
    dist = distribution_table(gold)
    moves = largest_moves(spy)
    exog = exog_correlations(spy)
    regimes = regime_table(spy)
    lev = leverage_table(spy)
    weekday, wd_ret_p, wd_abs_p = weekday_table(spy)
    make_figures(spy, regimes)

    top_1pct = spy.loc[spy["log_return"].abs() >= spy["log_return"].abs().quantile(0.99)]
    share_by_year = top_1pct["date"].dt.year.value_counts(normalize=True).head(3)

    lines = [
        "# EDA summary (train + validation only)",
        "",
        f"Data: `gold_market_daily`, {spy['date'].min():%Y-%m-%d} to {spy['date'].max():%Y-%m-%d}. "
        f"Sealed test period (2024+) excluded. Primary asset: {PRIMARY}.",
        "",
        "## Q1 Stationarity (ADF: H0 unit root; KPSS: H0 stationary)",
        frame_block(stat),
        "## Q2/Q3 Ljung-Box p-values, lag 10 (low p = autocorrelation)",
        frame_block(lb),
        "## Q4 Distribution",
        frame_block(dist),
        f"Ten largest |moves| for {PRIMARY} (return in %):",
        frame_block(moves),
        "## Q5 Exogenous variables (SPY, feature on day t vs target on day t+1)",
        frame_block(exog),
        "## Q6 Regimes",
        frame_block(regimes),
        "## Q7 Leverage effect (SPY)",
        "```\n" + "\n".join(f"{k}: {v:.5f}" for k, v in lev.items()) + "\n```\n",
        "## Q8 Weekday effects (SPY)",
        "```\n" + weekday.round(4).to_string() + "\n```\n",
        f"Kruskal-Wallis p (returns): {wd_ret_p:.4f} | (absolute returns): {wd_abs_p:.4f}",
        "",
        "## Extremes clustering (top 1% |return| days, share by year)",
        "```\n" + share_by_year.round(3).to_string() + "\n```\n",
    ]
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
