from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

PCT = 100.0
REFIT_EVERY = 21
EWMA_LAMBDA = 0.94


@dataclass(frozen=True)
class GarchParams:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float
    last_variance: float


def fit_garch(returns: pd.Series, asymmetric: bool) -> GarchParams:
    model = arch_model(
        returns * PCT,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1 if asymmetric else 0,
        q=1,
        dist="t",
        rescale=False,
    )
    res = model.fit(disp="off", options={"maxiter": 500})
    p = res.params
    return GarchParams(
        mu=float(p["mu"]),
        omega=float(p["omega"]),
        alpha=float(p["alpha[1]"]),
        gamma=float(p["gamma[1]"]) if asymmetric else 0.0,
        beta=float(p["beta[1]"]),
        nu=float(p["nu"]),
        last_variance=float(res.conditional_volatility.iloc[-1] ** 2),
    )


def one_step_variances(params: GarchParams, block_returns_pct: np.ndarray) -> np.ndarray:
    """Variance forecast for the day after each origin in the block (percent^2)."""
    h = params.last_variance
    out = np.empty(len(block_returns_pct))
    for i, r in enumerate(block_returns_pct):
        eps = r - params.mu
        h = params.omega + (params.alpha + params.gamma * (eps < 0)) * eps**2 + params.beta * h
        out[i] = h
    return out


def t_interval_multiplier(nu: float, level: float = 0.95) -> float:
    return float(stats.t.ppf(0.5 + level / 2, nu) * np.sqrt((nu - 2) / nu))


def walk_forward_garch(
    returns: pd.Series,
    origin_positions: np.ndarray,
    asymmetric: bool,
    refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    rows = []
    for start in range(0, len(origin_positions), refit_every):
        block = origin_positions[start : start + refit_every]
        first = int(block[0])
        params = fit_garch(returns.iloc[: first + 1], asymmetric)
        block_pct = returns.iloc[block].to_numpy() * PCT
        # first origin's variance already sits in `last_variance` as day `first`'s h_t
        variances = one_step_variances(params, block_pct)
        rows.append(
            pd.DataFrame(
                {
                    "var_forecast": variances / PCT**2,
                    "mu": params.mu / PCT,
                    "nu": params.nu,
                },
                index=returns.index[block],
            )
        )
    return pd.concat(rows)


def train_conditional_sigma(train_returns: pd.Series) -> pd.Series:
    model = arch_model(
        train_returns * PCT, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False
    )
    return model.fit(disp="off", options={"maxiter": 500}).conditional_volatility / PCT


def rolling_variance(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window).var(ddof=1)


def ewma_variance(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    return (returns**2).ewm(alpha=1 - lam, adjust=False, min_periods=21).mean()
