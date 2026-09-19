from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from risklens import evaluation as ev
from risklens import strategy as st
from risklens.calibrate import CONFIG_PATH, TRAIN_END
from risklens.live import live_mean_forecast
from risklens.mean_models import MeanSpec, walk_forward_mean
from risklens.run_validation import RESULTS_DIR, VALIDATION_END, mean_report, risk_report
from risklens.signals import INTERVAL_LEVELS, block_train_end, build_signals, t_multiplier
from risklens.volatility import (
    ewma_variance,
    rolling_variance,
    train_conditional_sigma,
    walk_forward_garch,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "data" / "gold"
REPORT_PATH = ROOT / "reports" / "test_results.md"
TAG = "preregistered-v1"
FROZEN_FILES = [
    "src/risklens/volatility.py",
    "src/risklens/mean_models.py",
    "src/risklens/strategy.py",
    "src/risklens/evaluation.py",
    "src/risklens/calibrate.py",
    "config/preregistered.json",
]
Z95 = float(stats.norm.ppf(0.975))


def assert_preregistered() -> str:
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    if git("rev-parse", "--verify", f"refs/tags/{TAG}").returncode != 0:
        raise SystemExit(f"tag {TAG} missing: pre-register before opening the test window")
    if git("diff", "--quiet", TAG, "--", *FROZEN_FILES).returncode != 0:
        raise SystemExit("frozen modeling files changed since the pre-registration tag")
    return git("rev-parse", "HEAD").stdout.strip()


def load_returns() -> pd.Series:
    gold = pd.read_parquet(GOLD_DIR / "gold_market_daily.parquet")
    spy = gold[gold["ticker"] == "SPY"].sort_values("date")
    return spy.set_index("date")["log_return"].iloc[1:]


def fenced(df: pd.DataFrame, digits: int = 4) -> str:
    return "```\n" + df.round(digits).to_string() + "\n```\n"


def main() -> None:
    head = assert_preregistered()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    returns = load_returns()
    val_last = int(returns.index.searchsorted(pd.Timestamp(VALIDATION_END), side="right") - 1)
    train_end_pos = int(returns.index.searchsorted(pd.Timestamp(TRAIN_END), side="right") - 1)
    origins_all = np.arange(val_last, len(returns))
    n_eval = len(returns) - 1 - val_last
    order = tuple(cfg["mean_model"]["order"])

    mean_eval = walk_forward_mean(
        returns, MeanSpec(order, ()), None, origins_all[:n_eval], val_last
    ).to_numpy()
    block_first = int(origins_all[(len(origins_all) - 1) // 21 * 21])
    mean_live = live_mean_forecast(returns, order, block_first)
    mean_fc = np.append(mean_eval, mean_live)
    garch = walk_forward_garch(returns, origins_all, asymmetric=False)
    idx_all = returns.index[origins_all]
    target_all = returns.index[np.minimum(origins_all + 1, len(returns) - 1)].to_series()
    target_all.iloc[-1] = returns.index[-1] + pd.offsets.BDay(1)
    test_fc = pd.DataFrame(
        {
            "mean": mean_fc,
            "var": garch["var_forecast"].to_numpy(),
            "nu": garch["nu"].to_numpy(),
            "target_date": target_all.to_numpy(),
            "train_end_date": block_train_end(idx_all),
            "split": ["test"] * n_eval + ["live"],
        },
        index=idx_all,
    )
    ev_fc = test_fc.iloc[:n_eval]
    actual = returns.iloc[origins_all[:n_eval] + 1].to_numpy()
    target_dates = pd.DatetimeIndex(ev_fc["target_date"])
    sigma = np.sqrt(ev_fc["var"].to_numpy())
    mean = ev_fc["mean"].to_numpy()
    nu = ev_fc["nu"].to_numpy()

    # risk task
    baselines = {
        "rolling_21": rolling_variance(returns).iloc[origins_all[:n_eval]],
        "ewma_094": ewma_variance(returns).iloc[origins_all[:n_eval]],
    }
    risk_rows, losses = {}, {}
    for name, var in baselines.items():
        risk_rows[name], losses[name] = risk_report(
            actual, var, np.full(n_eval, Z95), np.zeros(n_eval)
        )
    garch_var = pd.Series(ev_fc["var"].to_numpy(), index=ev_fc.index)
    mult95 = t_multiplier(nu, 0.95)
    risk_rows["garch_t"], losses["garch_t"] = risk_report(
        actual, garch_var, mult95, garch["mu"].to_numpy()[:n_eval]
    )
    risk = pd.DataFrame(risk_rows).T
    for name in ("ewma_094", "garch_t"):
        stat, p = ev.diebold_mariano(losses["rolling_21"], losses[name])
        risk.loc[name, "dm_vs_rolling_stat"], risk.loc[name, "dm_vs_rolling_p"] = stat, p
    years = target_dates.year
    qlike_year = pd.DataFrame(
        {n: pd.Series(loss).groupby(years).mean() for n, loss in losses.items()}
    ).T
    joint = pd.DataFrame(
        {
            f"{int(level * 100)}%": ev.interval_report(
                actual,
                mean - t_multiplier(nu, level) * sigma,
                mean + t_multiplier(nu, level) * sigma,
                nominal=level,
            )
            for level in INTERVAL_LEVELS
        }
    ).T

    # mean task
    mean_row = mean_report(actual, mean, actual**2)
    mean_table = pd.DataFrame(
        {
            "naive_zero": {"rmse": ev.rmse(actual, np.zeros(n_eval))},
            "arima_200": mean_row,
            "always_up_directional": {"directional_acc": float(np.mean(actual > 0))},
        }
    ).T

    # strategies
    v_max_a = cfg["strategy_A_rule"]["v_max"]
    v_max_a = np.inf if v_max_a is None else v_max_a
    theta = cfg["strategy_A_rule"]["theta_buy"]
    signals_a = st.rule_signals(mean / sigma, sigma, theta, v_max_a)
    decided = {
        "A_score_rule": st.positions_from_signals(signals_a),
        "B_vol_filter": st.vol_filter_positions(sigma, cfg["strategy_B_vol_filter"]["v_max"]),
    }

    def simulate(dec: np.ndarray | None, delay: int, cost: float):
        held = np.ones(n_eval) if dec is None else st.held_positions(dec, delay)
        r = st.strategy_returns(actual, held, cost)
        return r, held, st.performance(r, held)

    bh_r, bh_held, bh_perf = simulate(None, 0, st.COST_BPS)
    perf_rows = {"buy_and_hold_SPY": bh_perf}
    series = {"buy_and_hold_SPY": (bh_r, bh_held)}
    boot_rows, verdicts = {}, {}
    for name, dec in decided.items():
        r, held, perf = simulate(dec, st.PRIMARY_DELAY, st.COST_BPS)
        perf_rows[f"{name} (primary: next-close execution)"] = perf
        series[name] = (r, held)
        _, _, same = simulate(dec, 0, st.COST_BPS)
        perf_rows[f"{name} (shift-1 convention: trade at signal close)"] = same
        boot_rows[name] = st.bootstrap_sharpe(r, bh_r)
        verdicts[name] = bool(
            perf["sharpe"] >= bh_perf["sharpe"] and perf["max_drawdown"] >= bh_perf["max_drawdown"]
        )
    perf_table = pd.DataFrame(perf_rows).T
    boot_table = pd.DataFrame(boot_rows).T

    def sharpe_at(dec: np.ndarray | None, cost: float) -> float:
        held = np.ones(n_eval) if dec is None else st.held_positions(dec, st.PRIMARY_DELAY)
        return st.performance(st.strategy_returns(actual, held, cost), held)["sharpe"]

    cost_table = pd.DataFrame(
        {
            f"{c:g} bps": {"buy_and_hold_SPY": sharpe_at(None, c)}
            | {name: sharpe_at(dec, c) for name, dec in decided.items()}
            for c in (0.0, 5.0, 10.0, 20.0)
        }
    )
    year_rows = []
    for name, (r, held) in series.items():
        for y in sorted(set(years)):
            mask = years == y
            p = st.performance(r[mask], held[mask])
            year_rows.append(
                {"strategy": name, "year": y, "days": int(mask.sum())}
                | {k: p[k] for k in ("cumulative_return", "sharpe", "max_drawdown", "exposure")}
            )
    by_year = pd.DataFrame(year_rows)

    equity = pd.DataFrame({"date": target_dates})
    for name, (r, _) in series.items():
        equity[f"equity_{name}"] = np.cumprod(1 + r)
    equity.to_csv(RESULTS_DIR / "test_equity_curves.csv", index=False)
    test_fc.to_csv(RESULTS_DIR / "test_forecasts.csv", index_label="origin_date")

    # gold_signals_daily: validation + test + one live forecast
    val = pd.read_csv(
        RESULTS_DIR / "validation_forecasts.csv",
        parse_dates=["origin_date"],
        index_col="origin_date",
    )
    val_pos = returns.index.get_indexer(val.index)
    val_fc = pd.DataFrame(
        {
            "mean": val["mean_arima"].to_numpy(),
            "var": val["var_garch_t"].to_numpy(),
            "nu": val["nu_garch_t"].to_numpy(),
            "target_date": returns.index[val_pos + 1].to_numpy(),
            "train_end_date": block_train_end(val.index),
            "split": "validation",
        },
        index=val.index,
    )
    train_sigma = train_conditional_sigma(returns.iloc[: train_end_pos + 1]).to_numpy()
    signals = build_signals(pd.concat([val_fc, test_fc]), cfg, train_sigma, "SPY")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(GOLD_DIR / "gold_signals_daily.parquet", index=False)

    opened = datetime.now(UTC).isoformat(timespec="seconds")
    report = "\n".join(
        [
            f"# Sealed test results ({target_dates.min():%Y-%m-%d} to "
            f"{target_dates.max():%Y-%m-%d})",
            "",
            f"Pre-registration tag `{TAG}`; run at commit `{head[:10]}` on {opened}. "
            f"{n_eval} one-day-ahead forecasts, expanding window, refit every 21 days, "
            "configuration frozen in `config/preregistered.json`.",
            "",
            "## Risk task: volatility (QLIKE lower is better; coverage target 95%)",
            fenced(risk),
            "QLIKE by year:",
            fenced(qlike_year),
            "Joint interval calibration (ARIMA mean + GARCH-t sigma):",
            fenced(joint),
            "## Mean task: next-day return",
            fenced(mean_table, 5),
            "## Strategies vs buy-and-hold SPY (5 bps costs)",
            fenced(perf_table),
            "Bootstrap (stationary, 2000 draws) of the Sharpe difference vs buy-and-hold:",
            fenced(boot_table),
            "Sharpe by transaction cost (primary execution):",
            fenced(cost_table),
            "By calendar year (primary execution):",
            fenced(by_year),
            "## Pre-registered acceptance (Sharpe >= buy-and-hold and smaller drawdown)",
            "```\n" + json.dumps(verdicts, indent=2) + "\n```\n",
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
