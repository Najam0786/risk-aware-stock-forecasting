from __future__ import annotations

import numpy as np
import pytest

from risklens import strategy as st


def test_signals_and_position_state_machine() -> None:
    score = np.array([0.2, 0.0, -0.2, 0.0, 0.3])
    sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.05])
    signals = st.rule_signals(score, sigma, theta=0.1, v_max=0.02)
    assert signals.tolist() == ["BUY", "HOLD", "SELL", "HOLD", "HOLD"]
    assert st.positions_from_signals(signals).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_delay_one_uses_previous_decision() -> None:
    decided = np.array([1.0, 0.0, 1.0, 1.0])
    assert st.held_positions(decided, 0).tolist() == [1.0, 0.0, 1.0, 1.0]
    assert st.held_positions(decided, 1).tolist() == [0.0, 1.0, 0.0, 1.0]


def test_costs_charged_on_position_changes_only() -> None:
    log_ret = np.zeros(4)
    held = np.array([1.0, 1.0, 0.0, 0.0])
    result = st.strategy_returns(log_ret, held, cost_bps=10)
    assert result.tolist() == pytest.approx([-0.001, 0.0, -0.001, 0.0])


def test_hindsight_signal_loses_its_edge_under_delay() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 2000)
    knows_next_return = (r > 0).astype(float)
    same_close = st.strategy_returns(r, st.held_positions(knows_next_return, 0), 0)
    delayed = st.strategy_returns(r, st.held_positions(knows_next_return, 1), 0)
    assert st.performance(same_close, np.ones(2000))["sharpe"] > 5.0
    assert st.performance(delayed, np.ones(2000))["sharpe"] < 1.0


def test_performance_metrics_on_known_path() -> None:
    returns = np.array([0.10, -0.20, 0.10])
    perf = st.performance(returns, np.ones(3))
    assert perf["max_drawdown"] == pytest.approx(-0.2)
    assert perf["cumulative_return"] == pytest.approx(1.1 * 0.8 * 1.1 - 1)
    assert perf["trades"] == 1


def test_bootstrap_detects_clear_outperformance() -> None:
    rng = np.random.default_rng(2)
    bench = rng.normal(0.0002, 0.01, 750)
    strat = bench + 0.002
    res = st.bootstrap_sharpe(strat, bench, n_boot=300, seed=1)
    assert res["diff_ci_low"] > 0
    assert res["prob_strategy_not_better"] < 0.05
