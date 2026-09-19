from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from risklens import strategy as st
from risklens.run_validation import RESULTS_DIR, TRAIN_END, VALIDATION_END, load_spy
from risklens.volatility import train_conditional_sigma

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "preregistered.json"
REPORT_PATH = ROOT / "reports" / "calibration.md"
THETAS = (0.05, 0.10, 0.15, 0.20)
A_VMAX_LABELS = {"none": None, "P75": 75, "P90": 90}
B_VMAX_PERCENTILES = (50, 60, 70, 75, 80, 90, 95)
RISK_LEVEL_PERCENTILES = (40, 85)
MIN_TRADES = 10
MODEL_VERSION = "arima200_garch11t_v1"
RULE_VERSION = "rules_v1"


def evaluate(actual: np.ndarray, decided: np.ndarray) -> dict[str, float]:
    held = st.held_positions(decided, st.PRIMARY_DELAY)
    perf = st.performance(st.strategy_returns(actual, held, st.COST_BPS), held)
    same_close = st.held_positions(decided, 0)
    perf["sharpe_same_close"] = st.performance(
        st.strategy_returns(actual, same_close, st.COST_BPS), same_close
    )["sharpe"]
    return perf


def select(grid: pd.DataFrame, bh: dict[str, float]) -> pd.Series:
    grid = grid.assign(
        feasible=(grid["max_drawdown"] >= bh["max_drawdown"]) & (grid["trades"] >= MIN_TRADES)
    )
    pool = grid[grid["feasible"]] if grid["feasible"].any() else grid[grid["trades"] >= MIN_TRADES]
    return pool.sort_values("sharpe", ascending=False).iloc[0]


def block(df: pd.DataFrame, digits: int = 4) -> str:
    return "```\n" + df.round(digits).to_string() + "\n```\n"


def main() -> None:
    frame, _, train_end_pos = load_spy()
    returns = frame["log_return"]
    train_sigma = train_conditional_sigma(returns.iloc[: train_end_pos + 1]).to_numpy()
    val = pd.read_csv(
        RESULTS_DIR / "validation_forecasts.csv",
        parse_dates=["origin_date"],
        index_col="origin_date",
    )
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
    best_a = select(grid_a, bh)

    b_rows = []
    for pct in B_VMAX_PERCENTILES:
        v_max = float(np.percentile(train_sigma, pct))
        decided = st.vol_filter_positions(sigma, v_max)
        b_rows.append({"v_max_label": f"P{pct}", "v_max": v_max} | evaluate(actual, decided))
    grid_b = pd.DataFrame(b_rows)
    best_b = select(grid_b, bh)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_a.to_csv(RESULTS_DIR / "calibration_grid_A.csv", index=False)
    grid_b.to_csv(RESULTS_DIR / "calibration_grid_B.csv", index=False)

    last_date = str(
        pd.read_parquet(ROOT / "data" / "gold" / "gold_market_daily.parquet")["date"].max()
    )[:10]
    lo, hi = (float(np.percentile(train_sigma, p)) for p in RISK_LEVEL_PERCENTILES)
    config = {
        "version": "prereg_v1",
        "registered_on": date.today().isoformat(),
        "data_vintage": json.loads((ROOT / "data" / "raw" / "VINTAGE.json").read_text())[
            "vintage_date"
        ],
        "splits": {
            "train_end": TRAIN_END,
            "validation_end": VALIDATION_END,
            "test_start": "2024-01-01",
            "test_end": last_date,
        },
        "asset": "SPY",
        "model_version": MODEL_VERSION,
        "mean_model": {"type": "ARIMA", "order": [2, 0, 0], "exog": [], "trend": "constant"},
        "volatility_model": {"type": "GARCH(1,1)", "innovations": "student_t", "mean": "constant"},
        "walk_forward": {"window": "expanding", "refit_every_days": 21},
        "execution": {
            "primary_delay": st.PRIMARY_DELAY,
            "meaning": "decision at origin t executes at the close of t+1 and earns the return "
            "of t+2; delay 0 (trade at the signal close) is reported only as a sensitivity",
        },
        "costs_bps": st.COST_BPS,
        "risk_levels": {
            "low_below_pct": RISK_LEVEL_PERCENTILES[0],
            "high_above_pct": RISK_LEVEL_PERCENTILES[1],
            "low_sigma": lo,
            "high_sigma": hi,
            "reference": "GARCH conditional sigma on the training sample",
        },
        "rule_version": RULE_VERSION,
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
        "acceptance_criterion_on_test": "strategy Sharpe >= own buy-and-hold Sharpe and "
        "max drawdown less severe than buy-and-hold, after 5 bps costs, primary execution",
        "test_protocol": "evaluated exactly once with this configuration",
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = "\n".join(
        [
            "# Validation calibration (2022-2023, primary execution t+1, 5 bps costs)",
            "",
            f"Buy-and-hold on validation: Sharpe {bh['sharpe']:.3f}, "
            f"max drawdown {bh['max_drawdown']:.3f}, cumulative {bh['cumulative_return']:.3f}",
            f"Feasible: max drawdown no worse than buy-and-hold and at least {MIN_TRADES} trades.",
            "",
            "## Strategy A: score rule (theta_buy = theta_sell = theta)",
            block(grid_a),
            f"Selected: theta={best_a['theta']}, v_max={best_a['v_max_label']}, "
            f"feasible={bool(best_a['feasible'])}",
            "",
            "## Strategy B: volatility filter",
            block(grid_b),
            f"Selected: v_max={best_b['v_max_label']}, feasible={bool(best_b['feasible'])}",
            "",
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
