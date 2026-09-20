from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from risklens.calibrate import TRAIN_END
from risklens.live import live_mean_forecast
from risklens.mean_models import MeanSpec, walk_forward_mean
from risklens.run_validation import VALIDATION_END
from risklens.signals import REFIT_EVERY, block_train_end, build_signals
from risklens.volatility import train_conditional_sigma, walk_forward_garch

ROOT = Path(__file__).resolve().parents[2]
SIGNALS_LIVE_FILE = "signals_live.parquet"
POST_FREEZE = "post_freeze"
TICKERS = ("SPY", "AAPL", "MSFT", "JPM")


def load_config(ticker: str, root: Path = ROOT) -> dict:
    if ticker == "SPY":
        return json.loads((root / "config" / "preregistered.json").read_text("utf-8"))
    payload = json.loads((root / "config" / "preregistered_extension.json").read_text("utf-8"))
    return payload["tickers"][ticker]


def score_ticker(
    ticker: str, market: pd.DataFrame, frozen: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """Signal rows from the last frozen origin onward, by the pre-registered procedure.

    `market` holds frozen plus post-freeze rows; `frozen` is gold_signals_daily. Nothing here
    is tuned: the walk-forward blocks stay aligned to the frozen schedule (every 21 origins from
    the end of validation), and thresholds come from the frozen config.
    """
    rows = market[market["ticker"] == ticker].sort_values("date")
    returns = rows.set_index("date")["log_return"].iloc[1:]
    mine = frozen[frozen["ticker"] == ticker]
    frozen_origin = pd.Timestamp(mine.loc[mine["split"] == "live", "origin_date"].iloc[0])
    frozen_pos = int(returns.index.get_loc(frozen_origin))

    val_last = int(returns.index.searchsorted(pd.Timestamp(VALIDATION_END), side="right") - 1)
    train_end_pos = int(returns.index.searchsorted(pd.Timestamp(TRAIN_END), side="right") - 1)
    block_start = val_last + (frozen_pos - val_last) // REFIT_EVERY * REFIT_EVERY
    origins = np.arange(block_start, len(returns))
    n_realized = len(origins) - 1
    order = tuple(cfg["mean_model"]["order"])

    mean_realized = (
        walk_forward_mean(
            returns, MeanSpec(order, ()), None, origins[:n_realized], val_last
        ).to_numpy()
        if n_realized
        else np.empty(0)
    )
    block_first = int(origins[(len(origins) - 1) // REFIT_EVERY * REFIT_EVERY])
    mean = np.append(mean_realized, live_mean_forecast(returns, order, block_first))
    garch = walk_forward_garch(returns, origins, asymmetric=False)
    origin_dates = returns.index[origins]
    targets = returns.index[np.minimum(origins + 1, len(returns) - 1)].to_series()
    targets.iloc[-1] = returns.index[-1] + pd.offsets.BDay(1)
    forecasts = pd.DataFrame(
        {
            "mean": mean,
            "var": garch["var_forecast"].to_numpy(),
            "nu": garch["nu"].to_numpy(),
            "target_date": targets.to_numpy(),
            "train_end_date": block_train_end(origin_dates),
            "split": [POST_FREEZE] * n_realized + ["live"],
        },
        index=origin_dates,
    )

    train_sigma = train_conditional_sigma(returns.iloc[: train_end_pos + 1]).to_numpy()
    seen = mine[mine["split"].isin(["validation", "test"])].sort_values("date")
    history = np.concatenate([train_sigma, seen["forecast_volatility"].to_numpy()])
    return build_signals(forecasts[origins >= frozen_pos], cfg, history, ticker)


def score_all(market: pd.DataFrame, frozen: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    parts = [score_ticker(t, market, frozen, load_config(t, root)) for t in TICKERS]
    return pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"], ignore_index=True)


def write_signals_live(signals: pd.DataFrame, live_dir: Path) -> None:
    path = live_dir / SIGNALS_LIVE_FILE
    tmp = path.with_suffix(".parquet.tmp")
    signals.to_parquet(tmp, index=False)
    tmp.replace(path)
