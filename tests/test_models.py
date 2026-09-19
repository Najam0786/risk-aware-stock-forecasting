from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from risklens import evaluation as ev
from risklens.mean_models import MeanSpec, walk_forward_mean
from risklens.volatility import fit_garch, one_step_variances


def simulated_returns(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    h, eps = 1e-4, 0.0
    out = np.empty(n)
    for i in range(n):
        h = 2e-6 + 0.08 * eps**2 + 0.9 * h
        eps = np.sqrt(h) * rng.standard_t(6) * np.sqrt(4 / 6)
        out[i] = eps
    return pd.Series(out, index=pd.bdate_range("2015-01-01", periods=n))


@pytest.mark.parametrize("asymmetric", [False, True])
def test_variance_recursion_matches_arch_forecast(asymmetric: bool) -> None:
    returns = simulated_returns(900)
    train = returns.iloc[:700]
    params = fit_garch(train, asymmetric)
    ours = one_step_variances(params, train.iloc[-1:].to_numpy() * 100)[0]
    model = arch_model(
        train * 100, mean="Constant", vol="GARCH", p=1, o=1 if asymmetric else 0, q=1, dist="t"
    )
    reference = model.fit(disp="off").forecast(horizon=1).variance.iloc[-1, 0]
    assert ours == pytest.approx(reference, rel=1e-3)


def test_mean_walk_forward_has_no_lookahead() -> None:
    returns = simulated_returns(420)
    origins = np.arange(330, 380)
    spec = MeanSpec((1, 0, 1), ())
    base = walk_forward_mean(returns, spec, None, origins, train_end_pos=330)
    altered = returns.copy()
    altered.iloc[351:] *= 3
    changed = walk_forward_mean(altered, spec, None, origins, train_end_pos=330)
    upto = returns.index[350]
    pd.testing.assert_series_equal(base.loc[:upto], changed.loc[:upto])


def test_kupiec_accepts_exact_rate_and_rejects_bad_rate() -> None:
    exact = np.zeros(200, dtype=bool)
    exact[::20] = True
    assert ev.kupiec_uc(exact, 0.05)[1] == pytest.approx(1.0)
    too_many = np.zeros(200, dtype=bool)
    too_many[:40] = True
    assert ev.kupiec_uc(too_many, 0.05)[1] < 0.001


def test_christoffersen_flags_clustered_violations() -> None:
    clustered = np.zeros(300, dtype=bool)
    clustered[100:115] = True
    spread = np.zeros(300, dtype=bool)
    spread[::20] = True
    assert ev.christoffersen_ind(clustered)[1] < 0.01
    assert ev.christoffersen_ind(spread)[1] > 0.05


def test_diebold_mariano_detects_better_model() -> None:
    rng = np.random.default_rng(3)
    worse = rng.normal(1.0, 0.3, 500)
    better = rng.normal(0.8, 0.3, 500)
    stat, p = ev.diebold_mariano(worse, better)
    assert stat > 0
    assert p < 0.01
