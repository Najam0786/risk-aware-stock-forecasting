from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from risklens import evaluation as ev
from risklens import strategy as st
from risklens.calibrate import (
    A_VMAX_LABELS,
    B_VMAX_PERCENTILES,
    MIN_TRADES,
    RISK_LEVEL_PERCENTILES,
    RULE_VERSION,
    THETAS,
    evaluate,
    select,
)
from risklens.mean_models import MeanSpec, select_order, walk_forward_mean
from risklens.run_validation import RESULTS_DIR, TRAIN_END, VALIDATION_END, mean_report, risk_report
from risklens.volatility import (
    ewma_variance,
    rolling_variance,
    t_interval_multiplier,
    train_conditional_sigma,
    walk_forward_garch,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "gold_market_daily.parquet"
CONFIG_PATH = ROOT / "config" / "preregistered_extension.json"
VALIDATION_REPORT = ROOT / "reports" / "extension_validation.md"
CALIBRATION_REPORT = ROOT / "reports" / "extension_calibration.md"
TICKERS = ("AAPL", "MSFT", "JPM")
Z95 = float(stats.norm.ppf(0.975))


def load_returns(ticker: str, until: str | None = None) -> pd.Series:
    gold = pd.read_parquet(GOLD_PATH)
    rows = gold[gold["ticker"] == ticker].sort_values("date")
    if until is not None:
        rows = rows[rows["date"] <= until]
    return rows.set_index("date")["log_return"].iloc[1:]


def block(df: pd.DataFrame, digits: int = 4) -> str:
    return "```\n" + df.round(digits).to_string() + "\n```\n"


def select_rule(grid: pd.DataFrame, bh: dict[str, float]) -> pd.Series:
    """Same selection as SPY; when no combination reaches MIN_TRADES the rule carries no timing
    information, so register the most active one (ties: best Sharpe) and flag it infeasible."""
    if (grid["trades"] >= MIN_TRADES).any():
        return select(grid, bh)
    best = grid.sort_values(["trades", "sharpe"], ascending=False).iloc[0].copy()
    best["feasible"] = False
    return best


def validate_ticker(ticker: str) -> dict:
    returns = load_returns(ticker, VALIDATION_END)
    train_end_pos = int(returns.index.searchsorted(pd.Timestamp(TRAIN_END), side="right") - 1)
    origins = np.arange(train_end_pos, len(returns) - 1)
    idx = returns.index[origins]
    actual = returns.iloc[origins + 1].to_numpy()

    order_table = select_order(returns.iloc[: train_end_pos + 1])
    order = (int(order_table.iloc[0]["p"]), 0, int(order_table.iloc[0]["q"]))
    mean_fc = walk_forward_mean(returns, MeanSpec(order, ()), None, origins, train_end_pos)
    garch = walk_forward_garch(returns, origins, asymmetric=False)

    variances = {
        "rolling_21": rolling_variance(returns).iloc[origins],
        "ewma_094": ewma_variance(returns).iloc[origins],
        "garch_t": pd.Series(garch["var_forecast"].to_numpy(), index=idx),
    }
    multipliers = {
        "rolling_21": np.full(len(origins), Z95),
        "ewma_094": np.full(len(origins), Z95),
        "garch_t": np.array([t_interval_multiplier(n) for n in garch["nu"]]),
    }
    means = {
        "rolling_21": np.zeros(len(origins)),
        "ewma_094": np.zeros(len(origins)),
        "garch_t": garch["mu"].to_numpy(),
    }
    rows, losses = {}, {}
    for name, var in variances.items():
        rows[name], losses[name] = risk_report(actual, var, multipliers[name], means[name])
    risk = pd.DataFrame(rows).T
    for name in ("ewma_094", "garch_t"):
        stat, p = ev.diebold_mariano(losses["rolling_21"], losses[name])
        risk.loc[name, "dm_vs_rolling_stat"], risk.loc[name, "dm_vs_rolling_p"] = stat, p

    mean = pd.DataFrame(
        {
            "naive_zero": {"rmse": ev.rmse(actual, np.zeros(len(actual)))},
            "arima": mean_report(actual, mean_fc.to_numpy(), actual**2),
            "always_up_directional": {"directional_acc": float(np.mean(actual > 0))},
        }
    ).T

    val = pd.DataFrame(
        {
            "actual_next_return": actual,
            "var_garch_t": variances["garch_t"].to_numpy(),
            "nu_garch_t": garch["nu"].to_numpy(),
            "mean_arima": mean_fc.to_numpy(),
            "var_rolling_21": variances["rolling_21"].to_numpy(),
            "var_ewma_094": variances["ewma_094"].to_numpy(),
        },
        index=idx,
    )
    val.to_csv(RESULTS_DIR / f"validation_forecasts_{ticker}.csv", index_label="origin_date")
    return {
        "val": val,
        "returns": returns,
        "train_end_pos": train_end_pos,
        "order": order,
        "order_table": order_table,
        "risk": risk,
        "mean": mean,
    }


def calibrate_ticker(ticker: str, result: dict, vintage: str, last_date: str) -> tuple[dict, dict]:
    returns, train_end_pos, order = result["returns"], result["train_end_pos"], result["order"]
    train_sigma = train_conditional_sigma(returns.iloc[: train_end_pos + 1]).to_numpy()
    val = result["val"]
    actual = val["actual_next_return"].to_numpy()
    sigma = np.sqrt(val["var_garch_t"].to_numpy())
    score = val["mean_arima"].to_numpy() / sigma

    bh_held = np.ones(len(actual))
    bh = st.performance(st.strategy_returns(actual, bh_held, st.COST_BPS), bh_held)

    a_rows = []
    for label, pct in A_VMAX_LABELS.items():
        v_max = np.inf if pct is None else float(np.percentile(train_sigma, pct))
        for theta in THETAS:
            signals = st.rule_signals(score, sigma, theta, v_max)
            decided = st.positions_from_signals(signals)
            a_rows.append(
                {"theta": theta, "v_max_label": label, "v_max": v_max}
                | evaluate(actual, decided)
                | {"buy_signals": int((signals == "BUY").sum())}
            )
    grid_a = pd.DataFrame(a_rows)
    best_a = select_rule(grid_a, bh)

    b_rows = []
    for pct in B_VMAX_PERCENTILES:
        v_max = float(np.percentile(train_sigma, pct))
        decided = st.vol_filter_positions(sigma, v_max)
        b_rows.append({"v_max_label": f"P{pct}", "v_max": v_max} | evaluate(actual, decided))
    grid_b = pd.DataFrame(b_rows)
    best_b = select_rule(grid_b, bh)
    grid_a.to_csv(RESULTS_DIR / f"calibration_grid_A_{ticker}.csv", index=False)
    grid_b.to_csv(RESULTS_DIR / f"calibration_grid_B_{ticker}.csv", index=False)

    lo, hi = (float(np.percentile(train_sigma, p)) for p in RISK_LEVEL_PERCENTILES)
    config = {
        "asset": ticker,
        "splits": {
            "train_end": TRAIN_END,
            "validation_end": VALIDATION_END,
            "test_start": "2024-01-01",
            "test_end": last_date,
        },
        "data_vintage": vintage,
        "model_version": f"arima{order[0]}0{order[2]}_garch11t_v1",
        "mean_model": {"type": "ARIMA", "order": list(order), "exog": [], "trend": "constant"},
        "volatility_model": {"type": "GARCH(1,1)", "innovations": "student_t", "mean": "constant"},
        "walk_forward": {"window": "expanding", "refit_every_days": 21},
        "execution": {"primary_delay": st.PRIMARY_DELAY},
        "costs_bps": st.COST_BPS,
        "risk_levels": {
            "low_below_pct": RISK_LEVEL_PERCENTILES[0],
            "high_above_pct": RISK_LEVEL_PERCENTILES[1],
            "low_sigma": lo,
            "high_sigma": hi,
            "reference": "GARCH conditional sigma on the training sample",
        },
        "rule_version": f"{RULE_VERSION}_{ticker}",
        "strategy_A_rule": {
            "score": "expected_return / forecast_volatility",
            "theta_buy": float(best_a["theta"]),
            "theta_sell": float(best_a["theta"]),
            "v_max_label": best_a["v_max_label"],
            "v_max": float(best_a["v_max"]) if np.isfinite(best_a["v_max"]) else None,
            "signal_to_position": "BUY long, SELL flat, HOLD unchanged, start flat",
            "passes_validation_constraints": bool(best_a["feasible"]),
            "validation": {k: float(best_a[k]) for k in ("sharpe", "max_drawdown", "trades")},
        },
        "strategy_B_vol_filter": {
            "rule": "long when forecast sigma < v_max, otherwise cash",
            "v_max_label": best_b["v_max_label"],
            "v_max": float(best_b["v_max"]),
            "passes_validation_constraints": bool(best_b["feasible"]),
            "validation": {k: float(best_b[k]) for k in ("sharpe", "max_drawdown", "trades")},
        },
        "validation_buy_and_hold": bh,
    }
    return config, {"grid_a": grid_a, "grid_b": grid_b, "bh": bh, "a": best_a, "b": best_b}


def main() -> None:
    vintage = json.loads((ROOT / "data" / "raw" / "VINTAGE.json").read_text("utf-8"))
    gold = pd.read_parquet(GOLD_PATH)
    last_date = f"{gold['date'].max():%Y-%m-%d}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs, val_sections, cal_sections = {}, [], []
    for ticker in TICKERS:
        result = validate_ticker(ticker)
        config, cal = calibrate_ticker(ticker, result, vintage["vintage_date"], last_date)
        configs[ticker] = config
        val_sections += [
            f"## {ticker}",
            f"ARMA order by BIC on training data: {result['order']}",
            "Volatility (QLIKE lower is better; coverage target 95%):",
            block(result["risk"]),
            "Next-day return:",
            block(result["mean"], 5),
            "ARMA order selection (BIC):",
            block(result["order_table"]),
        ]
        cal_sections += [
            f"## {ticker}",
            f"Buy-and-hold on validation: Sharpe {cal['bh']['sharpe']:.3f}, "
            f"max drawdown {cal['bh']['max_drawdown']:.3f}, cumulative {cal['bh']['cumulative_return']:.3f}",
            f"Feasible: max drawdown no worse than buy-and-hold and at least {MIN_TRADES} trades.",
            "Strategy A (score rule):",
            block(cal["grid_a"]),
            f"Selected: theta={cal['a']['theta']}, v_max={cal['a']['v_max_label']}, "
            f"feasible={bool(cal['a']['feasible'])}",
            "Strategy B (volatility filter):",
            block(cal["grid_b"]),
            f"Selected: v_max={cal['b']['v_max_label']}, feasible={bool(cal['b']['feasible'])}",
            "",
        ]
        print(
            ticker,
            result["order"],
            "A:",
            config["strategy_A_rule"]["theta_buy"],
            config["strategy_A_rule"]["v_max_label"],
            "B:",
            config["strategy_B_vol_filter"]["v_max_label"],
        )

    payload = {
        "version": "prereg_v2",
        "registered_on": date.today().isoformat(),
        "protocol": "same procedure as prereg_v1 applied per asset: ARMA order by BIC on training "
        "data (no exogenous regressors), GARCH(1,1)-t, thresholds fixed on validation only, "
        "primary baseline is buy-and-hold of the same asset with SPY as market reference",
        "acceptance_criterion_on_test": "strategy Sharpe >= buy-and-hold of the same asset and "
        "max drawdown less severe than that buy-and-hold, after 5 bps costs, primary execution",
        "selection_fallback": "if no threshold combination reaches the minimum number of trades "
        "on validation, the most active combination is registered and flagged as not passing the "
        "validation constraints",
        "test_protocol": "each asset evaluated exactly once with this configuration",
        "tickers": configs,
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    header = "# {} (train 2010-2021, validation 2022-2023; test sealed)\n"
    VALIDATION_REPORT.write_text(
        header.format("Extension assets: model validation") + "\n".join(val_sections),
        encoding="utf-8",
    )
    CALIBRATION_REPORT.write_text(
        header.format("Extension assets: threshold calibration") + "\n".join(cal_sections),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
