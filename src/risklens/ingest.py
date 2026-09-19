from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
START = "2010-01-01"
PRICE_FILES = {
    "SPY": "prices_SPY.csv",
    "AAPL": "prices_AAPL.csv",
    "MSFT": "prices_MSFT.csv",
    "JPM": "prices_JPM.csv",
    "^VIX": "vix.csv",
}
FRED_FILES = {
    "DGS10": "fred_dgs10.csv",
    "T10Y2Y": "fred_t10y2y.csv",
    "CPIAUCSL": "fred_cpi.csv",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MANIFEST = "VINTAGE.json"


def download_prices(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=True)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no rows for {ticker}")
    df.index = df.index.tz_localize(None).rename("Date")
    return df


def download_fred(series_id: str) -> str:
    resp = requests.get(FRED_URL, params={"id": series_id, "cosd": START}, timeout=60)
    resp.raise_for_status()
    return resp.text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(force: bool = False) -> dict:
    manifest_path = RAW_DIR / MANIFEST
    if manifest_path.exists() and not force:
        raise SystemExit(f"{manifest_path} exists: vintage is immutable. Use --force to overwrite.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    for ticker, name in PRICE_FILES.items():
        df = download_prices(ticker)
        df.to_csv(RAW_DIR / name)
        files[name] = {
            "source": "yahoo",
            "symbol": ticker,
            "rows": len(df),
            "first_date": df.index.min().strftime("%Y-%m-%d"),
            "last_date": df.index.max().strftime("%Y-%m-%d"),
        }
    for series_id, name in FRED_FILES.items():
        text = download_fred(series_id)
        (RAW_DIR / name).write_text(text, encoding="utf-8", newline="\n")
        df = pd.read_csv(RAW_DIR / name)
        files[name] = {
            "source": "fred",
            "symbol": series_id,
            "rows": len(df),
            "first_date": str(df.iloc[0, 0]),
            "last_date": str(df.iloc[-1, 0]),
        }
    for name in files:
        files[name]["sha256"] = sha256(RAW_DIR / name)

    manifest = {
        "vintage_date": date.today().isoformat(),
        "start": START,
        "libraries": {"yfinance": version("yfinance"), "pandas": version("pandas")},
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the raw data vintage")
    parser.add_argument("--force", action="store_true", help="overwrite an existing vintage")
    manifest = ingest(force=parser.parse_args().force)
    for name, meta in manifest["files"].items():
        print(f"{name:20s} {meta['rows']:6d} rows  {meta['first_date']} -> {meta['last_date']}")
    print(f"vintage_date = {manifest['vintage_date']}")


if __name__ == "__main__":
    main()
