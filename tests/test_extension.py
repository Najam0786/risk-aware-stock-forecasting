from __future__ import annotations

import pandas as pd

from risklens import run_test
from risklens.extension import select_rule


def test_select_rule_falls_back_to_most_active_when_nothing_trades_enough() -> None:
    grid = pd.DataFrame(
        {
            "theta": [0.05, 0.10],
            "max_drawdown": [-0.2, -0.1],
            "trades": [3, 1],
            "sharpe": [0.1, 0.9],
        }
    )
    best = select_rule(grid, {"max_drawdown": -0.3})
    assert best["theta"] == 0.05
    assert not best["feasible"]


def test_select_rule_uses_standard_selection_when_trades_are_sufficient() -> None:
    grid = pd.DataFrame(
        {
            "theta": [0.05, 0.10],
            "max_drawdown": [-0.2, -0.4],
            "trades": [30, 40],
            "sharpe": [0.5, 0.9],
        }
    )
    best = select_rule(grid, {"max_drawdown": -0.3})
    assert best["theta"] == 0.05
    assert best["feasible"]


def test_write_signals_replaces_only_the_given_asset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_test, "GOLD_DIR", tmp_path)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    spy = pd.DataFrame({"date": dates, "ticker": "SPY", "signal": "HOLD"})
    aapl = pd.DataFrame({"date": dates, "ticker": "AAPL", "signal": "BUY"})
    run_test.write_signals(spy, "SPY")
    run_test.write_signals(aapl, "AAPL")
    run_test.write_signals(aapl.assign(signal="SELL"), "AAPL")
    result = pd.read_parquet(tmp_path / "gold_signals_daily.parquet")
    assert set(result["ticker"]) == {"SPY", "AAPL"}
    assert (result[result["ticker"] == "SPY"]["signal"] == "HOLD").all()
    assert (result[result["ticker"] == "AAPL"]["signal"] == "SELL").all()
    assert len(result) == 4
