from __future__ import annotations

import numpy as np

TRADING_DAYS = 252
COST_BPS = 5.0
PRIMARY_DELAY = 1


def rule_signals(score: np.ndarray, sigma: np.ndarray, theta: float, v_max: float) -> np.ndarray:
    signal = np.full(len(score), "HOLD", dtype=object)
    signal[(score > theta) & (sigma < v_max)] = "BUY"
    signal[score < -theta] = "SELL"
    return signal


def positions_from_signals(signals: np.ndarray) -> np.ndarray:
    """BUY goes long, SELL goes flat, HOLD keeps the previous position; starts flat."""
    out = np.zeros(len(signals))
    state = 0.0
    for i, s in enumerate(signals):
        if s == "BUY":
            state = 1.0
        elif s == "SELL":
            state = 0.0
        out[i] = state
    return out


def vol_filter_positions(sigma: np.ndarray, v_max: float) -> np.ndarray:
    return (sigma < v_max).astype(float)


def held_positions(decided: np.ndarray, delay: int) -> np.ndarray:
    """Position earned on target day i, given decisions made at origin i.

    delay=0 trades at the close that produced the signal (optimistic); delay=1 executes at
    the close of the following day, so the decision at origin i earns the return of target i+1.
    """
    held = np.zeros_like(decided)
    if delay == 0:
        return decided.copy()
    held[delay:] = decided[:-delay]
    return held


def strategy_returns(log_returns: np.ndarray, held: np.ndarray, cost_bps: float) -> np.ndarray:
    turnover = np.abs(np.diff(held, prepend=0.0))
    return held * np.expm1(log_returns) - cost_bps / 1e4 * turnover


def performance(returns: np.ndarray, held: np.ndarray) -> dict[str, float]:
    equity = np.concatenate([[1.0], np.cumprod(1 + returns)])
    drawdown = equity / np.maximum.accumulate(equity) - 1
    std = returns.std(ddof=1)
    in_market = held > 0
    return {
        "cumulative_return": float(equity[-1] - 1),
        "annual_return": float(equity[-1] ** (TRADING_DAYS / len(returns)) - 1),
        "sharpe": float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "hit_ratio": float((returns[in_market] > 0).mean()) if in_market.any() else float("nan"),
        "exposure": float(in_market.mean()),
        "trades": int((np.abs(np.diff(held, prepend=0.0)) > 0).sum()),
    }


def _sharpe(r: np.ndarray) -> float:
    std = r.std(ddof=1)
    return float(r.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")


def stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    restart = rng.random(n) < 1 / mean_block
    starts = rng.integers(0, n, n)
    idx = np.empty(n, dtype=int)
    idx[0] = starts[0]
    for i in range(1, n):
        idx[i] = starts[i] if restart[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe(
    strategy: np.ndarray,
    benchmark: np.ndarray,
    n_boot: int = 2000,
    mean_block: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs, s_strat, s_bench = [], [], []
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(len(strategy), mean_block, rng)
        a, b = _sharpe(strategy[idx]), _sharpe(benchmark[idx])
        s_strat.append(a)
        s_bench.append(b)
        diffs.append(a - b)
    diffs_arr = np.array(diffs)
    return {
        "sharpe_diff": _sharpe(strategy) - _sharpe(benchmark),
        "diff_ci_low": float(np.nanpercentile(diffs_arr, 2.5)),
        "diff_ci_high": float(np.nanpercentile(diffs_arr, 97.5)),
        "prob_strategy_not_better": float(np.nanmean(diffs_arr <= 0)),
        "strategy_sharpe_ci_low": float(np.nanpercentile(s_strat, 2.5)),
        "strategy_sharpe_ci_high": float(np.nanpercentile(s_strat, 97.5)),
        "benchmark_sharpe_ci_low": float(np.nanpercentile(s_bench, 2.5)),
        "benchmark_sharpe_ci_high": float(np.nanpercentile(s_bench, 97.5)),
    }
