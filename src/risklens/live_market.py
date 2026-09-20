from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from risklens import build_gold, clean
from risklens.live_ingest import MACRO_FILE, VIX_FILE, read_live_prices, read_live_series

ROOT = Path(__file__).resolve().parents[2]
LIVE_MARKET_FILE = "gold_market_live.parquet"
FROZEN_TOLERANCE = 1e-9


def extended_prices(processed: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Continue each ticker's adjusted-price level with the live returns."""
    base = processed[["date", "ticker", "adj_close", "volume"]]
    parts = [base]
    for ticker, rows in live.sort_values("date").groupby("ticker"):
        last_level = float(base[base["ticker"] == ticker].sort_values("date")["adj_close"].iloc[-1])
        levels = last_level * np.exp(rows["log_return"].cumsum())
        parts.append(
            pd.DataFrame(
                {
                    "date": rows["date"].to_numpy(),
                    "ticker": ticker,
                    "adj_close": levels.to_numpy(),
                    "volume": rows["volume"].to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _extend(vintage: pd.Series, live: pd.Series, name: str) -> pd.Series:
    vintage = vintage.dropna()
    return pd.concat([vintage, live.dropna()[live.dropna().index > vintage.index.max()]]).rename(
        name
    )


def extended_macro(raw_dir: Path, live_dir: Path, spine: pd.DatetimeIndex) -> pd.DataFrame:
    vix = _extend(
        clean.read_vix(raw_dir / "vix.csv"),
        read_live_series(live_dir / VIX_FILE, "vix_close"),
        "vix_close",
    )
    live_macro = live_dir / MACRO_FILE
    fred = {}
    for name in ("DGS10", "T10Y2Y"):
        vintage = clean.read_fred(raw_dir / f"fred_{name.lower()}.csv")
        fred[name] = _extend(vintage, read_live_series(live_macro, name.lower()), name)
    return clean.build_macro(
        spine, vix, fred["DGS10"], fred["T10Y2Y"], clean.read_fred(raw_dir / "fred_cpi.csv")
    )


def assert_matches_frozen(rebuilt: pd.DataFrame, frozen: pd.DataFrame) -> None:
    keys = ["ticker", "date"]
    a = rebuilt.sort_values(keys).reset_index(drop=True)
    b = frozen.sort_values(keys).reset_index(drop=True)
    if a.shape != b.shape or not a[keys].equals(b[keys]):
        raise ValueError("extended market layer does not reproduce the frozen calendar")
    numeric = [c for c in b.columns if c not in keys]
    if not np.allclose(
        a[numeric].to_numpy(dtype="float64"),
        b[numeric].to_numpy(dtype="float64"),
        rtol=FROZEN_TOLERANCE,
        atol=FROZEN_TOLERANCE,
        equal_nan=True,
    ):
        raise ValueError("extended market layer changed frozen history")


def build_live_market(
    raw_dir: Path, processed_dir: Path, gold_dir: Path, live_dir: Path
) -> pd.DataFrame:
    """Post-freeze rows in the gold_market_daily schema, features built by the frozen code."""
    gold = pd.read_parquet(gold_dir / "gold_market_daily.parquet")
    processed = pd.read_csv(processed_dir / "prices_clean.csv", parse_dates=["date"])
    prices = extended_prices(processed, read_live_prices(live_dir))
    spine = pd.DatetimeIndex(sorted(prices["date"].unique()))
    rebuilt = build_gold.build_gold_market(prices, extended_macro(raw_dir, live_dir, spine))
    build_gold.validate_gold_market(rebuilt)
    frozen_last = gold["date"].max()
    assert_matches_frozen(rebuilt[rebuilt["date"] <= frozen_last], gold)
    new = rebuilt[rebuilt["date"] > frozen_last].reset_index(drop=True)
    return new.astype(gold.dtypes.to_dict())


def write_live_market(market: pd.DataFrame, live_dir: Path) -> None:
    path = live_dir / LIVE_MARKET_FILE
    tmp = path.with_suffix(".parquet.tmp")
    market.to_parquet(tmp, index=False)
    tmp.replace(path)
