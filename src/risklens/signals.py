from __future__ import annotations

from bisect import bisect_left, insort
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import stats

from risklens import strategy as st

INTERVAL_LEVELS = (0.50, 0.80, 0.95)
REFIT_EVERY = 21


def t_multiplier(nu: np.ndarray, level: float) -> np.ndarray:
    return stats.t.ppf(0.5 + level / 2, nu) * np.sqrt((nu - 2) / nu)


def block_train_end(origin_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    first_of_block = (np.arange(len(origin_dates)) // REFIT_EVERY) * REFIT_EVERY
    return origin_dates[first_of_block]


def expanding_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    history = sorted(reference.tolist())
    out = np.empty(len(values))
    for i, v in enumerate(values):
        out[i] = 100 * bisect_left(history, v) / len(history)
        insort(history, v)
    return out


def build_signals(
    forecasts: pd.DataFrame, config: dict, train_sigma: np.ndarray, ticker: str
) -> pd.DataFrame:
    """One row per forecast target date; `forecasts` is indexed by origin date."""
    sigma = np.sqrt(forecasts["var"].to_numpy())
    mean = forecasts["mean"].to_numpy()
    nu = forecasts["nu"].to_numpy()
    rule = config["strategy_A_rule"]
    v_max_a = np.inf if rule["v_max"] is None else rule["v_max"]
    v_max_b = config["strategy_B_vol_filter"]["v_max"]
    levels = config["risk_levels"]

    out = pd.DataFrame(
        {
            "date": forecasts["target_date"].to_numpy(),
            "ticker": ticker,
            "origin_date": forecasts.index.to_numpy(),
            "expected_return": mean,
            "forecast_volatility": sigma,
            "risk_adjusted_score": mean / sigma,
        }
    )
    out["risk_level"] = np.select(
        [sigma < levels["low_sigma"], sigma > levels["high_sigma"]], ["low", "high"], "medium"
    )
    out["signal"] = st.rule_signals(
        out["risk_adjusted_score"].to_numpy(), sigma, rule["theta_buy"], v_max_a
    )
    out["vol_filter_state"] = np.where(sigma < v_max_b, "RISK_ON", "RISK_OFF")
    out["vol_percentile"] = expanding_percentile(train_sigma, sigma)
    for level in INTERVAL_LEVELS:
        half = t_multiplier(nu, level) * sigma
        suffix = "" if level == 0.95 else f"_{int(level * 100)}"
        out[f"pi_lower{suffix}"] = mean - half
        out[f"pi_upper{suffix}"] = mean + half
    out["model_version"] = config["model_version"]
    out["train_end_date"] = forecasts["train_end_date"].to_numpy()
    out["horizon_days"] = np.int8(1)
    out["rule_version"] = config["rule_version"]
    out["run_timestamp"] = datetime.now(UTC).replace(tzinfo=None)
    out["split"] = forecasts["split"].to_numpy()
    validate_signals(out)
    return out


def validate_signals(signals: pd.DataFrame) -> None:
    if signals.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate (date, ticker) keys in signals")
    if not (pd.to_datetime(signals["train_end_date"]) < pd.to_datetime(signals["date"])).all():
        raise ValueError("train_end_date must be strictly before the forecast date")
    if not (pd.to_datetime(signals["origin_date"]) < pd.to_datetime(signals["date"])).all():
        raise ValueError("origin_date must precede the forecast date")
    if not (signals["forecast_volatility"] > 0).all():
        raise ValueError("forecast_volatility must be positive")
    if not signals["signal"].isin(["BUY", "SELL", "HOLD"]).all():
        raise ValueError("signal must be BUY, SELL or HOLD")
