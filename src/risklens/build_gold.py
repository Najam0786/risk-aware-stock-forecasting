from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
GOLD_DIR = ROOT / "data" / "gold"
LAGS = range(1, 6)
VOL_WINDOW = 21
MANDATORY = [
    "adj_close",
    "volume",
    "log_return",
    *[f"ret_lag_{k}" for k in LAGS],
    "roll_vol_21",
    "roll_mean_21",
    "vix_close",
    "vix_change",
    "dgs10",
    "dgs10_change",
    "t10y2y",
    "day_of_week",
]


def add_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["log_return"] = np.log(g["adj_close"]).diff()
    for k in LAGS:
        g[f"ret_lag_{k}"] = g["log_return"].shift(k)
    g["roll_vol_21"] = g["log_return"].rolling(VOL_WINDOW).std()
    g["roll_mean_21"] = g["log_return"].rolling(VOL_WINDOW).mean()
    return g


def build_gold_market(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    macro = macro.copy()
    macro["vix_change"] = macro["vix_close"].diff()
    macro["dgs10_change"] = macro["dgs10"].diff()
    featured = pd.concat(
        [add_ticker_features(g) for _, g in prices.groupby("ticker", sort=True)],
        ignore_index=True,
    )
    gold = featured.merge(macro, left_on="date", right_index=True, how="left")
    gold["day_of_week"] = gold["date"].dt.dayofweek.astype("int8")
    gold = gold.dropna(subset=MANDATORY).reset_index(drop=True)
    columns = [
        "date",
        "ticker",
        "adj_close",
        "volume",
        "log_return",
        *[f"ret_lag_{k}" for k in LAGS],
        "roll_vol_21",
        "roll_mean_21",
        "vix_close",
        "vix_change",
        "dgs10",
        "dgs10_change",
        "t10y2y",
        "day_of_week",
        "cpi_yoy",
    ]
    return gold[columns]


def validate_gold_market(gold: pd.DataFrame) -> None:
    if gold.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate (date, ticker) keys")
    if not (gold["adj_close"] > 0).all():
        raise ValueError("adj_close must be > 0")
    if not np.isfinite(gold["log_return"]).all():
        raise ValueError("log_return must be finite")
    if gold[MANDATORY].isna().any().any():
        raise ValueError("mandatory columns contain nulls")
    if not gold["date"].dt.dayofweek.between(0, 4).all():
        raise ValueError("weekend date in gold")


def main() -> None:
    prices = pd.read_csv(PROCESSED_DIR / "prices_clean.csv", parse_dates=["date"])
    macro = pd.read_csv(PROCESSED_DIR / "macro_daily.csv", parse_dates=["date"], index_col="date")
    gold = build_gold_market(prices, macro)
    validate_gold_market(gold)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(GOLD_DIR / "gold_market_daily.parquet", index=False)
    first, last = gold["date"].min(), gold["date"].max()
    print(f"gold_market_daily: {len(gold)} rows | {first:%Y-%m-%d} -> {last:%Y-%m-%d}")
    print(gold.groupby("ticker").size().to_string())


if __name__ == "__main__":
    main()
