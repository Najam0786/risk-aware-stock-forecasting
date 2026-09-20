from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from risklens import clean

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
LIVE_DIR = ROOT / "data" / "live"
TICKERS = clean.TICKERS
MACRO_SERIES = ("DGS10", "T10Y2Y")
NEW_YORK = ZoneInfo("America/New_York")
SESSION_FINAL_AT = dtime(17, 0)
LOOKBACK_DAYS = 45
MIN_OVERLAP = 5
RETURN_TOLERANCE = 1e-3
VIX_TOLERANCE = 0.5
FRED_TOLERANCE = 1e-6
MAX_ABS_RETURN = 0.30
MAX_VIX = 200.0
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
PRICE_FIELDS = ["date", "close", "adj_close", "volume"]
LIVE_PRICE_COLUMNS = ["date", "ticker", "close", "adj_close", "volume", "log_return", "source"]
PRICES_FILE = "prices_live.csv"
VIX_FILE = "vix_live.csv"
MACRO_FILE = "macro_live.csv"
STATUS_FILE = "data_status.json"


class SourceError(RuntimeError):
    pass


class ValidationError(ValueError):
    pass


PriceFetcher = Callable[[str, pd.Timestamp], pd.DataFrame]
SeriesFetcher = Callable[[pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class Sources:
    prices: list[tuple[str, PriceFetcher]]
    vix: list[tuple[str, SeriesFetcher]]
    macro: dict[str, SeriesFetcher]


def with_retries(fn: Callable, attempts: int = 3, base_delay: float = 2.0, sleep=time.sleep):
    def wrapped(*args):
        for attempt in range(attempts):
            try:
                return fn(*args)
            except SourceError:
                raise
            except Exception as exc:
                if attempt == attempts - 1:
                    raise SourceError(f"{type(exc).__name__}: {exc}") from exc
                sleep(base_delay * 2**attempt)

    return wrapped


def _naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def yahoo_prices(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=f"{start:%Y-%m-%d}", auto_adjust=False)
    if df.empty:
        raise SourceError(f"Yahoo returned no rows for {ticker}")
    return pd.DataFrame(
        {
            "date": _naive_dates(df.index),
            "close": df["Close"].to_numpy(),
            "adj_close": df["Adj Close"].to_numpy(),
            "volume": df["Volume"].to_numpy(),
        }
    )


def yahoo_series(symbol: str, start: pd.Timestamp) -> pd.Series:
    df = yf.Ticker(symbol).history(start=f"{start:%Y-%m-%d}", auto_adjust=False)
    if df.empty:
        raise SourceError(f"Yahoo returned no rows for {symbol}")
    return pd.Series(df["Close"].to_numpy(), index=_naive_dates(df.index))


def tiingo_prices(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise SourceError("TIINGO_API_KEY is not set")
    resp = requests.get(
        TIINGO_URL.format(ticker=ticker),
        params={"startDate": f"{start:%Y-%m-%d}"},
        headers={"Authorization": f"Token {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise SourceError(f"Tiingo returned no rows for {ticker}")
    dates = pd.to_datetime([r["date"] for r in rows], utc=True).tz_localize(None).normalize()
    return pd.DataFrame(
        {
            "date": dates,
            "close": [r["close"] for r in rows],
            "adj_close": [r["adjClose"] for r in rows],
            "volume": [r["volume"] for r in rows],
        }
    )


def fred_series(series_id: str, start: pd.Timestamp) -> pd.Series:
    resp = requests.get(FRED_URL, params={"id": series_id, "cosd": f"{start:%Y-%m-%d}"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=[".", ""], parse_dates=["observation_date"])
    series = df.set_index("observation_date").iloc[:, 0].astype("float64").dropna()
    if series.empty:
        raise SourceError(f"FRED returned no observations for {series_id}")
    return series


def default_sources(sleep=time.sleep) -> Sources:
    def retry(fn: Callable) -> Callable:
        return with_retries(fn, sleep=sleep)

    return Sources(
        prices=[("yahoo", retry(yahoo_prices)), ("tiingo", retry(tiingo_prices))],
        vix=[
            ("yahoo", retry(lambda start: yahoo_series("^VIX", start))),
            ("fred", retry(lambda start: fred_series("VIXCLS", start))),
        ],
        macro={sid: retry(lambda start, sid=sid: fred_series(sid, start)) for sid in MACRO_SERIES},
    )


def last_complete_date(now: datetime) -> pd.Timestamp:
    local = now.astimezone(NEW_YORK)
    day = local.date() if local.time() >= SESSION_FINAL_AT else local.date() - timedelta(days=1)
    return pd.Timestamp(day)


def expected_last_session(now: datetime) -> pd.Timestamp:
    day = last_complete_date(now)
    return pd.Timestamp(np.busday_offset(day.date(), 0, roll="backward"))


def sessions_behind(last_close: pd.Timestamp, now: datetime) -> int:
    expected = expected_last_session(now)
    if last_close >= expected:
        return 0
    return int(np.busday_count((last_close + pd.Timedelta(days=1)).date(), expected.date()) + 1)


def check_consistency(
    fetched: pd.Series, reference: pd.Series, tolerance: float, what: str
) -> None:
    overlap = fetched.index.intersection(reference.index)
    if len(overlap) < MIN_OVERLAP:
        raise ValidationError(f"{what}: only {len(overlap)} overlapping days with stored data")
    worst = float((fetched.loc[overlap] - reference.loc[overlap]).abs().max())
    if worst > tolerance:
        raise ValidationError(f"{what}: overlap differs from stored data by {worst:.6f}")


def validated_prices(raw: pd.DataFrame, reference: pd.Series, now: datetime) -> pd.DataFrame:
    """New completed sessions after `reference`, checked against the stored return history."""
    if not set(PRICE_FIELDS) <= set(raw.columns):
        raise ValidationError(f"missing columns {sorted(set(PRICE_FIELDS) - set(raw.columns))}")
    frame = raw[PRICE_FIELDS].copy()
    frame["date"] = _naive_dates(pd.DatetimeIndex(frame["date"]))
    frame = frame[(frame["date"].dt.dayofweek < 5) & (frame["date"] <= last_complete_date(now))]
    frame = frame.sort_values("date")
    if frame.empty:
        raise ValidationError("no completed sessions")
    if frame["date"].duplicated().any():
        raise ValidationError("duplicate dates")
    values = frame[["close", "adj_close", "volume"]].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValidationError("non-finite values")
    if (frame[["close", "adj_close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValidationError("non-positive price or negative volume")
    frame["log_return"] = np.log(frame["adj_close"]).diff()
    fetched = frame.set_index("date")["log_return"].dropna()
    check_consistency(fetched, reference, RETURN_TOLERANCE, "returns")
    new = frame[frame["date"] > reference.index.max()]
    if (new["log_return"].abs() > MAX_ABS_RETURN).any():
        raise ValidationError(f"daily move above {MAX_ABS_RETURN:.0%}")
    return new.reset_index(drop=True)


def validated_series(
    raw: pd.Series, reference: pd.Series, now: datetime, tolerance: float, upper: float | None
) -> pd.Series:
    series = raw.dropna().astype("float64")
    series.index = _naive_dates(pd.DatetimeIndex(series.index))
    series = series[series.index <= last_complete_date(now)].sort_index()
    if series.empty:
        raise ValidationError("no completed observations")
    if series.index.duplicated().any():
        raise ValidationError("duplicate dates")
    if (series <= 0).any() or (upper is not None and (series > upper).any()):
        raise ValidationError("value out of range")
    check_consistency(series, reference, tolerance, "series")
    return series[series.index > reference.index.max()]


def first_valid(
    chain: list[tuple[str, Callable]], label: str, validate: Callable, *args: object
) -> tuple[str | None, object, list[str]]:
    errors: list[str] = []
    for name, fetcher in chain:
        try:
            return name, validate(fetcher(*args)), errors
        except (SourceError, ValidationError) as exc:
            errors.append(f"{label} via {name}: {exc}")
    return None, None, errors


def common_prefix(frames: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Dates, in order, that every ticker has, stopping at the first date any ticker lacks."""
    date_sets = [set(f["date"]) for f in frames.values()]
    accepted: list[pd.Timestamp] = []
    for day in sorted(set.union(*date_sets)):
        if not all(day in s for s in date_sets):
            break
        accepted.append(day)
    return accepted


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_live_prices(live_dir: Path) -> pd.DataFrame:
    path = live_dir / PRICES_FILE
    if not path.exists():
        return pd.DataFrame(columns=LIVE_PRICE_COLUMNS).astype({"date": "datetime64[ns]"})
    return pd.read_csv(path, parse_dates=["date"])


def read_live_series(path: Path, column: str) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="date"))
    df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=[column])
    return df.set_index("date")[column]


def read_status(live_dir: Path = LIVE_DIR) -> dict | None:
    path = live_dir / STATUS_FILE
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def price_reference(raw_dir: Path, live: pd.DataFrame, ticker: str) -> pd.Series:
    vintage = clean.read_prices(raw_dir / f"prices_{ticker}.csv", ticker).set_index("date")
    returns = np.log(vintage["adj_close"]).diff().dropna()
    stored = live[live["ticker"] == ticker].set_index("date")["log_return"]
    return pd.concat([returns, stored]).sort_index()


def series_reference(vintage: pd.Series, live: pd.Series) -> pd.Series:
    return pd.concat([vintage, live]).sort_index()


def _series_frame(new: pd.Series, column: str, source: str) -> pd.DataFrame:
    return pd.DataFrame({"date": new.index, column: new.to_numpy(), "source": source})


def refresh(
    now: datetime | None = None,
    sources: Sources | None = None,
    raw_dir: Path = RAW_DIR,
    live_dir: Path = LIVE_DIR,
) -> dict:
    """Fetch, validate and store new sessions; on any failure keep the last good data."""
    now = now or datetime.now(UTC)
    sources = sources or default_sources()
    live_dir.mkdir(parents=True, exist_ok=True)
    previous = read_status(live_dir) or {}
    errors: list[str] = []

    live_prices = read_live_prices(live_dir)
    references = {t: price_reference(raw_dir, live_prices, t) for t in TICKERS}
    price_sources: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        reference = references[ticker]
        start = reference.index.max() - pd.Timedelta(days=LOOKBACK_DAYS)
        name, frame, errs = first_valid(
            sources.prices,
            ticker,
            lambda raw, reference=reference: validated_prices(raw, reference, now),
            ticker,
            start,
        )
        errors += errs
        if frame is not None:
            frames[ticker], price_sources[ticker] = frame, name
    prices_ok = len(frames) == len(TICKERS)
    accepted = common_prefix(frames) if prices_ok else []
    fetched_dates = set().union(*(set(f["date"]) for f in frames.values()))
    calendar_ok = not (prices_ok and len(fetched_dates) > len(accepted))
    if not calendar_ok:
        errors.append("prices: tickers disagree on the trading calendar, later dates held back")
    if accepted:
        new_prices = pd.concat(
            [
                frames[t][frames[t]["date"].isin(accepted)].assign(
                    ticker=t, source=price_sources[t]
                )
                for t in TICKERS
            ]
        )[LIVE_PRICE_COLUMNS]
        write_atomic(
            pd.concat([live_prices, new_prices]).sort_values(["date", "ticker"]),
            live_dir / PRICES_FILE,
        )
    last_close = references["SPY"].index.max()
    if accepted:
        last_close = accepted[-1]

    vix_reference = series_reference(
        clean.read_vix(raw_dir / "vix.csv"), read_live_series(live_dir / VIX_FILE, "vix_close")
    )
    vix_start = vix_reference.index.max() - pd.Timedelta(days=LOOKBACK_DAYS)
    vix_name, vix_new, errs = first_valid(
        sources.vix,
        "VIX",
        lambda raw: validated_series(raw, vix_reference, now, VIX_TOLERANCE, MAX_VIX),
        vix_start,
    )
    errors += errs
    if vix_new is not None and len(vix_new):
        stored = (
            pd.read_csv(live_dir / VIX_FILE, parse_dates=["date"])
            if (live_dir / VIX_FILE).exists()
            else pd.DataFrame(columns=["date", "vix_close", "source"])
        )
        write_atomic(
            pd.concat([stored, _series_frame(vix_new, "vix_close", vix_name)]),
            live_dir / VIX_FILE,
        )

    macro_new: dict[str, pd.Series] = {}
    macro_ok = True
    stored_macro = (
        pd.read_csv(live_dir / MACRO_FILE, parse_dates=["date"], index_col="date")
        if (live_dir / MACRO_FILE).exists()
        else pd.DataFrame(columns=[s.lower() for s in MACRO_SERIES])
    )
    for series_id, fetcher in sources.macro.items():
        vintage = clean.read_fred(raw_dir / f"fred_{series_id.lower()}.csv").dropna()
        reference = series_reference(
            vintage, stored_macro.get(series_id.lower(), pd.Series(dtype=float)).dropna()
        )
        start = reference.index.max() - pd.Timedelta(days=LOOKBACK_DAYS)
        _, new, errs = first_valid(
            [("fred", fetcher)],
            series_id,
            lambda raw, reference=reference: validated_series(
                raw, reference, now, FRED_TOLERANCE, None
            ),
            start,
        )
        errors += errs
        if new is None:
            macro_ok = False
        elif len(new):
            macro_new[series_id.lower()] = new
    if macro_new:
        merged = stored_macro.combine_first(pd.DataFrame(macro_new)).sort_index()
        write_atomic(merged.rename_axis("date").reset_index(), live_dir / MACRO_FILE)

    behind = sessions_behind(last_close, now)
    failed = not (prices_ok and calendar_ok and vix_new is not None and macro_ok)
    if failed:
        state = "fallback"
    elif behind == 0:
        state = "current"
    else:
        state = "delayed"
    status = {
        "checked_at": now.astimezone(UTC).isoformat(timespec="seconds"),
        "last_successful_fetch_at": (
            now.astimezone(UTC).isoformat(timespec="seconds")
            if not failed
            else previous.get("last_successful_fetch_at")
        ),
        "state": state,
        "last_close": f"{last_close:%Y-%m-%d}",
        "expected_last_session": f"{expected_last_session(now):%Y-%m-%d}",
        "sessions_behind": behind,
        "new_price_rows": len(accepted),
        "backup_used": bool(errors) and not failed,
        "components": {
            "prices": {"ok": prices_ok, "sources": price_sources},
            "vix": {"ok": vix_new is not None, "source": vix_name},
            "macro": {"ok": macro_ok},
        },
        "errors": errors,
    }
    status_path = live_dir / STATUS_FILE
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, status_path)
    return status


def main() -> None:
    status = refresh()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
