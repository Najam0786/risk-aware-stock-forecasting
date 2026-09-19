from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def live_mean_forecast(returns: pd.Series, order: tuple[int, int, int], block_first: int) -> float:
    """One-step forecast beyond the last observation, using the parameters of the current block.

    Numpy input avoids statsmodels' need for a regular date index when forecasting out of sample.
    """
    endog = returns.to_numpy() * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(endog[: block_first + 1], order=order, trend="c").fit(disp=False, maxiter=200)
        applied = res.apply(endog, refit=False)
    return float(applied.forecast(1)[0]) / 100
