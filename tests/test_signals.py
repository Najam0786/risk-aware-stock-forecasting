from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.signals import (
    block_train_end,
    build_signals,
    expanding_percentile,
    validate_signals,
)

GOLD_SIGNALS = "data/gold/gold_signals_daily.parquet"


def test_expanding_percentile_uses_only_past_values() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    out = expanding_percentile(reference, np.array([2.5, 10.0]))
    assert out[0] == pytest.approx(200 / 3)
    assert out[1] == pytest.approx(100.0)


def test_block_train_end_resets_every_21_origins() -> None:
    dates = pd.bdate_range("2024-01-01", periods=50)
    ends = block_train_end(dates)
    assert (ends[:21] == dates[0]).all()
    assert (ends[21:42] == dates[21]).all()
    assert ends[49] == dates[42]


def synthetic_forecasts(n: int = 30) -> pd.DataFrame:
    origins = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mean": rng.normal(0, 0.001, n),
            "var": np.full(n, 1e-4),
            "nu": np.full(n, 6.0),
            "target_date": origins + pd.offsets.BDay(1),
            "train_end_date": block_train_end(origins),
            "split": "test",
        },
        index=origins,
    )


def config() -> dict:
    return {
        "model_version": "m",
        "rule_version": "r",
        "strategy_A_rule": {"theta_buy": 0.05, "v_max": None},
        "strategy_B_vol_filter": {"v_max": 0.02},
        "risk_levels": {"low_sigma": 0.005, "high_sigma": 0.015},
    }


def test_build_signals_produces_valid_nested_intervals() -> None:
    signals = build_signals(synthetic_forecasts(), config(), np.full(100, 0.01), "SPY")
    assert (signals["pi_lower"] < signals["pi_lower_80"]).all()
    assert (signals["pi_lower_80"] < signals["pi_lower_50"]).all()
    assert (signals["pi_upper_50"] < signals["pi_upper_80"]).all()
    assert (signals["pi_upper_80"] < signals["pi_upper"]).all()
    assert set(signals["signal"]) <= {"BUY", "SELL", "HOLD"}
    assert (signals["risk_level"] == "medium").all()


def test_validate_signals_rejects_look_ahead_training() -> None:
    signals = build_signals(synthetic_forecasts(), config(), np.full(100, 0.01), "SPY")
    bad = signals.assign(train_end_date=signals["date"])
    with pytest.raises(ValueError):
        validate_signals(bad)


def test_gold_signals_file_contract() -> None:
    try:
        signals = pd.read_parquet(GOLD_SIGNALS)
    except FileNotFoundError:
        pytest.skip("gold_signals_daily not built")
    validate_signals(signals)
    assert set(signals["split"]) == {"validation", "test", "live"}
    assert (signals["pi_lower_50"] > signals["pi_lower"]).all()
