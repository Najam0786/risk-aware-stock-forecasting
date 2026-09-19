from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
GOLD = ROOT / "data" / "gold" / "gold_market_daily.parquet"
REPORT_PATH = ROOT / "reports" / "data_quality.md"
TICKERS = ["SPY", "AAPL", "MSFT", "JPM"]


def raw_summary() -> pd.DataFrame:
    manifest = json.loads((RAW / "VINTAGE.json").read_text("utf-8"))
    rows = []
    for name, meta in manifest["files"].items():
        if meta["source"] == "yahoo":
            df = pd.read_csv(RAW / name, parse_dates=["Date"])
            dates, values = df["Date"], df["Adj Close"]
        else:
            df = pd.read_csv(RAW / name, na_values=[".", ""], parse_dates=["observation_date"])
            dates, values = df["observation_date"], df.iloc[:, 1]
        rows.append(
            {
                "file": name,
                "rows": len(df),
                "first": f"{dates.min():%Y-%m-%d}",
                "last": f"{dates.max():%Y-%m-%d}",
                "duplicate_dates": int(dates.duplicated().sum()),
                "missing_values": int(values.isna().sum()),
                "sha256_matches_manifest": hashlib.sha256((RAW / name).read_bytes()).hexdigest()
                == meta["sha256"],
            }
        )
    return pd.DataFrame(rows)


def calendar_checks() -> dict[str, object]:
    spy = pd.read_csv(RAW / "prices_SPY.csv", parse_dates=["Date"])["Date"]
    vix = pd.read_csv(RAW / "vix.csv", parse_dates=["Date"])["Date"]
    dgs10 = pd.read_csv(RAW / "fred_dgs10.csv", parse_dates=["observation_date"])
    same_calendar = all(
        set(pd.read_csv(RAW / f"prices_{t}.csv", parse_dates=["Date"])["Date"]) == set(spy)
        for t in TICKERS
    )
    return {
        "all four tickers share one trading calendar": same_calendar,
        "VIX dates that are not trading days (dropped by the spine)": [
            f"{d:%Y-%m-%d}" for d in sorted(set(vix) - set(spy))
        ],
        "DGS10 blank days (bond-market holidays, forward-filled)": int(
            dgs10.iloc[:, 1].isna().sum()
        ),
        "latest DGS10 date vs latest price date": (
            f"{dgs10['observation_date'].max():%Y-%m-%d} vs {spy.max():%Y-%m-%d}"
        ),
    }


def gold_validity() -> tuple[dict[str, object], pd.DataFrame]:
    gold = pd.read_parquet(GOLD)
    mandatory = gold.drop(columns=["cpi_yoy"])
    checks = {
        "gold rows": len(gold),
        "duplicate (date, ticker) keys": int(gold.duplicated(["date", "ticker"]).sum()),
        "adj_close <= 0": int((gold["adj_close"] <= 0).sum()),
        "non-finite log returns": int((~np.isfinite(gold["log_return"])).sum()),
        "nulls in mandatory columns": int(mandatory.isna().sum().sum()),
        "weekend dates": int((gold["date"].dt.dayofweek > 4).sum()),
        "zero-volume days": int((gold["volume"] == 0).sum()),
        "cpi_yoy nulls (desirable field, first year has no year-over-year)": int(
            gold["cpi_yoy"].isna().sum()
        ),
    }
    extremes = (
        gold.assign(abs_ret=gold["log_return"].abs())
        .sort_values("abs_ret", ascending=False)
        .groupby("ticker")
        .head(3)[["ticker", "date", "log_return", "vix_close"]]
        .sort_values(["ticker", "date"])
    )
    extremes["date"] = extremes["date"].dt.strftime("%Y-%m-%d")
    extremes["log_return"] = (extremes["log_return"] * 100).round(2)
    return checks, extremes.reset_index(drop=True)


def main() -> None:
    raw = raw_summary()
    calendar = calendar_checks()
    checks, extremes = gold_validity()
    lines = [
        "# Data quality report",
        "",
        "Rows failing a validity rule are counted here, never dropped silently.",
        "",
        "## Raw files (frozen vintage)",
        "```",
        raw.to_string(index=False),
        "```",
        "",
        "## Calendar and alignment",
        *[f"- {k}: {v}" for k, v in calendar.items()],
        "",
        "## Gold layer validity",
        *[f"- {k}: {v}" for k, v in checks.items()],
        "",
        "## Extreme moves kept on purpose (real events, not errors)",
        "```",
        extremes.to_string(index=False),
        "```",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
