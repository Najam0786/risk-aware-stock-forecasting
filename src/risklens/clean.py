from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TICKERS = ["SPY", "AAPL", "MSFT", "JPM"]
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
FRED_LAG_DAYS = 1
CPI_PUBLICATION_LAG = pd.DateOffset(months=1, days=15)


def read_prices(path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["ticker"] = ticker
    df["volume"] = df["volume"].astype("int64")
    return df[PRICE_COLUMNS].drop_duplicates(["date", "ticker"], keep="last")


def clean_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = [read_prices(raw_dir / f"prices_{t}.csv", t) for t in TICKERS]
    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    if (prices["adj_close"] <= 0).any() or prices["adj_close"].isna().any():
        raise ValueError("adj_close must be positive and non-null in raw prices")
    date_sets = {t: set(g["date"]) for t, g in prices.groupby("ticker")}
    if len({frozenset(s) for s in date_sets.values()}) != 1:
        raise ValueError("tickers do not share one trading calendar")
    return prices.reset_index(drop=True)


def read_fred(path: Path) -> pd.Series:
    df = pd.read_csv(path, na_values=[".", ""], parse_dates=["observation_date"])
    return df.set_index("observation_date").iloc[:, 0].astype("float64").rename(df.columns[1])


def read_vix(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["Close"].astype("float64").rename("vix_close")


def cpi_yoy_asof(cpi: pd.Series, spine: pd.DatetimeIndex) -> pd.Series:
    yoy = cpi.dropna().pct_change(12).dropna().mul(100)
    available = pd.DataFrame(
        {"available_from": yoy.index + CPI_PUBLICATION_LAG, "cpi_yoy": yoy.to_numpy()}
    )
    left = pd.DataFrame({"date": spine})
    merged = pd.merge_asof(left, available, left_on="date", right_on="available_from")
    return merged.set_index("date")["cpi_yoy"]


def build_macro(
    spine: pd.DatetimeIndex,
    vix: pd.Series,
    dgs10: pd.Series,
    t10y2y: pd.Series,
    cpi: pd.Series,
) -> pd.DataFrame:
    macro = pd.DataFrame(index=spine)
    macro.index.name = "date"
    macro["vix_close"] = vix.reindex(spine).ffill()
    for series in (dgs10, t10y2y):
        macro[series.name.lower()] = series.reindex(spine).ffill().shift(FRED_LAG_DAYS)
    macro["cpi_yoy"] = cpi_yoy_asof(cpi, spine)
    return macro


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices = clean_prices()
    prices.to_csv(PROCESSED_DIR / "prices_clean.csv", index=False)
    spine = pd.DatetimeIndex(sorted(prices["date"].unique()))
    macro = build_macro(
        spine,
        read_vix(RAW_DIR / "vix.csv"),
        read_fred(RAW_DIR / "fred_dgs10.csv"),
        read_fred(RAW_DIR / "fred_t10y2y.csv"),
        read_fred(RAW_DIR / "fred_cpi.csv"),
    )
    macro.to_csv(PROCESSED_DIR / "macro_daily.csv")
    print(f"prices_clean: {len(prices)} rows | macro_daily: {len(macro)} rows")
    print(macro.isna().sum().to_string())


if __name__ == "__main__":
    main()
