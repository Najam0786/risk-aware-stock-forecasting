from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

REFIT_EVERY = 21


@dataclass(frozen=True)
class MeanSpec:
    order: tuple[int, int, int]
    exog: tuple[str, ...]


def lagged_exog(features: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Row j holds the feature values known at the close of day j-1, for predicting r_j."""
    return features[list(columns)].shift(1)


def standardize(exog: pd.DataFrame, train_end_pos: int) -> pd.DataFrame:
    train = exog.iloc[: train_end_pos + 1]
    return (exog - train.mean()) / train.std(ddof=0)


def _fit(endog: pd.Series, order: tuple[int, int, int], exog: pd.DataFrame | None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog * 100,
            exog=exog,
            order=order,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        return model.fit(disp=False, maxiter=200)


def select_order(train: pd.Series, max_p: int = 2, max_q: int = 2) -> pd.DataFrame:
    rows = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        res = _fit(train, (p, 0, q), None)
        rows.append({"p": p, "q": q, "aic": res.aic, "bic": res.bic})
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)


def select_exog(
    train: pd.Series,
    exog_all: dict[tuple[str, ...], pd.DataFrame],
    order: tuple[int, int, int],
    train_end_pos: int,
) -> pd.DataFrame:
    base = _fit(train, order, None)
    rows = [{"exog": "none", "aic": base.aic, "bic": base.bic}]
    for columns, exog in exog_all.items():
        z = standardize(exog, train_end_pos).iloc[: train_end_pos + 1]
        res = _fit(train, order, z)
        rows.append({"exog": "+".join(columns), "aic": res.aic, "bic": res.bic})
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)


def walk_forward_mean(
    returns: pd.Series,
    spec: MeanSpec,
    exog: pd.DataFrame | None,
    origin_positions: np.ndarray,
    train_end_pos: int,
    refit_every: int = REFIT_EVERY,
) -> pd.Series:
    """Forecast of r_{origin+1} made at each origin, refitting on data up to the block start."""
    z = standardize(exog, train_end_pos) if exog is not None else None
    forecasts = pd.Series(index=returns.index[origin_positions], dtype="float64")
    for start in range(0, len(origin_positions), refit_every):
        block = origin_positions[start : start + refit_every]
        first = int(block[0])
        stop = int(block[-1]) + 2
        z_fit = z.iloc[: first + 1] if z is not None else None
        res = _fit(returns.iloc[: first + 1], spec.order, z_fit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = res.apply(
                returns.iloc[:stop] * 100,
                exog=z.iloc[:stop] if z is not None else None,
                refit=False,
            )
            pred = applied.get_prediction(start=first + 1, end=stop - 1, dynamic=False)
        forecasts.iloc[start : start + len(block)] = pred.predicted_mean.to_numpy() / 100
    return forecasts
