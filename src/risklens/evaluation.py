from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import xlog1py, xlogy


def qlike_loss(var_forecast: np.ndarray, realized_sq: np.ndarray) -> np.ndarray:
    return np.log(var_forecast) + realized_sq / var_forecast


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def directional_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
    if not np.any(forecast != 0):
        return float("nan")
    return float(np.mean(np.sign(actual) == np.sign(forecast)))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Positive statistic: model A has the larger loss (B is better)."""
    d = loss_a - loss_b
    n = len(d)
    centered = d - d.mean()
    lags = int(n ** (1 / 3))
    long_run = float(centered @ centered) / n
    for k in range(1, lags + 1):
        weight = 1 - k / (lags + 1)
        long_run += 2 * weight * float(centered[k:] @ centered[:-k]) / n
    if long_run <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(long_run / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def _bernoulli_loglik(successes: float, trials: float, p: float) -> float:
    return float(xlogy(successes, p) + xlog1py(trials - successes, -p))


def kupiec_uc(violations: np.ndarray, expected_rate: float) -> tuple[float, float]:
    n, x = len(violations), int(violations.sum())
    observed = x / n
    lr = -2 * (_bernoulli_loglik(x, n, expected_rate) - _bernoulli_loglik(x, n, observed))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_ind(violations: np.ndarray) -> tuple[float, float]:
    v = violations.astype(int)
    prev, curr = v[:-1], v[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    restricted = _bernoulli_loglik(n01 + n11, n00 + n01 + n10 + n11, pi)
    unrestricted = _bernoulli_loglik(n01, n00 + n01, pi01) + _bernoulli_loglik(n11, n10 + n11, pi11)
    lr = -2 * (restricted - unrestricted)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def interval_report(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float = 0.95
) -> dict[str, float]:
    violations = (actual < lower) | (actual > upper)
    lr_uc, p_uc = kupiec_uc(violations, 1 - nominal)
    lr_ind, p_ind = christoffersen_ind(violations)
    p_cc = float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2))
    return {
        "coverage": float(1 - violations.mean()),
        "kupiec_p": p_uc,
        "christoffersen_ind_p": p_ind,
        "cond_coverage_p": p_cc,
    }
