from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from risklens import evaluation as ev
from risklens.mean_models import (
    MeanSpec,
    lagged_exog,
    select_exog,
    select_order,
    walk_forward_mean,
)
from risklens.volatility import (
    ewma_variance,
    rolling_variance,
    t_interval_multiplier,
    walk_forward_garch,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "gold_market_daily.parquet"
RESULTS_DIR = ROOT / "reports" / "results"
REPORT_PATH = ROOT / "reports" / "model_validation.md"
TRAIN_END = "2021-12-31"
VALIDATION_END = "2023-12-31"
EXOG_CANDIDATES = {
    ("vix_change",): ("vix_change",),
    ("vix_change", "dgs10_change"): ("vix_change", "dgs10_change"),
}
Z95_NORMAL = float(stats.norm.ppf(0.975))


def load_spy() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    gold = pd.read_parquet(GOLD_PATH)
    spy = gold[(gold["ticker"] == "SPY") & (gold["date"] <= VALIDATION_END)].sort_values("date")
    if spy["date"].max() > pd.Timestamp(VALIDATION_END):
        raise ValueError("modeling data must not include the sealed test period")
    frame = spy.set_index("date")[["log_return", "vix_change", "dgs10_change"]]
    exog = lagged_exog(frame, ("vix_change", "dgs10_change"))
    frame, exog = frame.iloc[1:], exog.iloc[1:]
    train_end_pos = int(frame.index.searchsorted(pd.Timestamp(TRAIN_END), side="right") - 1)
    return frame, exog, train_end_pos


def risk_report(
    actual: np.ndarray, var: pd.Series, mult: np.ndarray, mean: np.ndarray
) -> tuple[dict, np.ndarray]:
    sigma = np.sqrt(var.to_numpy())
    loss = ev.qlike_loss(var.to_numpy(), actual**2)
    interval = ev.interval_report(actual, mean - mult * sigma, mean + mult * sigma)
    return {
        "qlike": loss.mean(),
        "mae_vol": ev.mae(np.abs(actual), sigma),
        "rmse_var": ev.rmse(actual**2, var.to_numpy()),
        **interval,
    }, loss


def mean_report(actual: np.ndarray, forecast: np.ndarray, naive_loss: np.ndarray) -> dict:
    loss = (actual - forecast) ** 2
    dm_stat, dm_p = ev.diebold_mariano(naive_loss, loss)
    return {
        "rmse": ev.rmse(actual, forecast),
        "mae": ev.mae(actual, forecast),
        "directional_acc": ev.directional_accuracy(actual, forecast),
        "dm_vs_naive_stat": dm_stat,
        "dm_vs_naive_p": dm_p,
    }


def block(df: pd.DataFrame, digits: int = 4) -> str:
    return "```\n" + df.round(digits).to_string() + "\n```\n"


def main() -> None:
    frame, exog_all, train_end_pos = load_spy()
    returns = frame["log_return"]
    origins = np.arange(train_end_pos, len(frame) - 1)
    idx = returns.index[origins]
    actual = returns.iloc[origins + 1].to_numpy()
    years = returns.index[origins + 1].year

    var_forecasts = {
        "rolling_21": rolling_variance(returns).iloc[origins],
        "ewma_094": ewma_variance(returns).iloc[origins],
    }
    multipliers = {
        "rolling_21": np.full(len(origins), Z95_NORMAL),
        "ewma_094": np.full(len(origins), Z95_NORMAL),
    }
    means = {"rolling_21": np.zeros(len(origins)), "ewma_094": np.zeros(len(origins))}
    nus: dict[str, np.ndarray] = {}
    for name, asym in (("garch_t", False), ("gjr_garch_t", True)):
        wf = walk_forward_garch(returns, origins, asym)
        var_forecasts[name] = wf["var_forecast"]
        multipliers[name] = np.array([t_interval_multiplier(n) for n in wf["nu"]])
        means[name] = wf["mu"].to_numpy()
        nus[name] = wf["nu"].to_numpy()

    risk_rows, risk_losses = {}, {}
    for name, var in var_forecasts.items():
        risk_rows[name], risk_losses[name] = risk_report(
            actual, var, multipliers[name], means[name]
        )
    risk = pd.DataFrame(risk_rows).T
    for name in risk.index.drop("rolling_21"):
        stat, p = ev.diebold_mariano(risk_losses["rolling_21"], risk_losses[name])
        risk.loc[name, "dm_vs_rolling_stat"], risk.loc[name, "dm_vs_rolling_p"] = stat, p
    by_year = pd.DataFrame(
        {name: pd.Series(loss).groupby(years).mean() for name, loss in risk_losses.items()}
    ).T

    train = returns.iloc[: train_end_pos + 1]
    order_table = select_order(train)
    best = order_table.iloc[0]
    order = (int(best["p"]), 0, int(best["q"]))
    exog_frames = {cols: exog_all[list(cols)] for cols in EXOG_CANDIDATES}
    exog_table = select_exog(train, exog_frames, order, train_end_pos)

    naive_loss = actual**2
    mean_rows = {"naive_zero": mean_report(actual, np.zeros(len(actual)), naive_loss)}
    mean_forecasts = {"naive_zero": pd.Series(0.0, index=idx)}
    mean_rows["always_up"] = {
        "directional_acc": float(np.mean(actual > 0)),
        "rmse": np.nan,
        "mae": np.nan,
    }
    arima = walk_forward_mean(returns, MeanSpec(order, ()), None, origins, train_end_pos)
    mean_forecasts["arima"] = arima
    mean_rows["arima"] = mean_report(actual, arima.to_numpy(), naive_loss)
    for cols, frame_x in exog_frames.items():
        name = "arimax_" + "+".join(cols)
        fc = walk_forward_mean(returns, MeanSpec(order, cols), frame_x, origins, train_end_pos)
        mean_forecasts[name] = fc
        mean_rows[name] = mean_report(actual, fc.to_numpy(), naive_loss)
    mean = pd.DataFrame(mean_rows).T
    mean_by_year = pd.DataFrame(
        {
            name: pd.Series((actual - fc.to_numpy()) ** 2).groupby(years).mean() ** 0.5
            for name, fc in mean_forecasts.items()
        }
    ).T

    out = pd.DataFrame({"actual_next_return": actual}, index=idx)
    for name, var in var_forecasts.items():
        out[f"var_{name}"] = var.to_numpy()
    out["nu_garch_t"] = nus["garch_t"]
    for name, fc in mean_forecasts.items():
        out[f"mean_{name}"] = fc.to_numpy()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "validation_forecasts.csv", index_label="origin_date")

    best_vol = risk.loc[["garch_t", "gjr_garch_t"], "qlike"].idxmin()
    config = {
        "train_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "arma_order": list(order),
        "exog_selected_by_bic_on_train": exog_table.iloc[0]["exog"],
        "best_vol_model_on_validation": best_vol,
    }
    (RESULTS_DIR / "selected_config.json").write_text(json.dumps(config, indent=2))

    report = "\n".join(
        [
            "# Model validation (train 2010-2021, validation 2022-2023; test sealed)",
            "",
            f"Walk-forward, expanding window, refit every 21 days. "
            f"{len(origins)} one-day-ahead forecasts.",
            "",
            "## Risk task: volatility forecasts (QLIKE lower is better; coverage target 95%)",
            block(risk),
            "QLIKE by year:",
            block(by_year),
            "## Mean task: ARMA order selection on train (BIC)",
            block(order_table),
            "## Exogenous set selection on train (BIC)",
            block(exog_table),
            "## Mean task: next-day return forecasts on validation",
            block(mean, 5),
            "RMSE by year:",
            block(mean_by_year, 6),
            "```\n" + json.dumps(config, indent=2) + "\n```\n",
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
