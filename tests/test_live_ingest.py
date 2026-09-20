from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklens import live_ingest as li
from risklens.live_ingest import SourceError, Sources

BDAYS = pd.bdate_range("2026-06-01", "2026-09-25")
VINTAGE_LAST = pd.Timestamp("2026-09-11")
NOW = datetime(2026, 9, 15, 23, 0, tzinfo=UTC)
FIRST_NEW = [pd.Timestamp("2026-09-14"), pd.Timestamp("2026-09-15")]


def _walk(seed: int, level: float, vol: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level * np.exp(np.cumsum(rng.normal(0, vol, len(BDAYS))))


PRICES = {t: _walk(i, 100 + 50 * i, 0.01) for i, t in enumerate(li.TICKERS)}
VIX = pd.Series(_walk(10, 15, 0.03), index=BDAYS)
FRED = {
    "DGS10": pd.Series(np.round(4 + np.cumsum(np.full(len(BDAYS), 0.001)), 2), index=BDAYS),
    "T10Y2Y": pd.Series(np.round(0.5 + np.cumsum(np.full(len(BDAYS), 0.002)), 2), index=BDAYS),
}


def _price_frame(ticker: str, adj_factor: float = 1.0) -> pd.DataFrame:
    p = PRICES[ticker]
    return pd.DataFrame(
        {"date": BDAYS, "close": p, "adj_close": p * adj_factor, "volume": 1_000_000}
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    keep = BDAYS <= VINTAGE_LAST
    for t in li.TICKERS:
        f = _price_frame(t, 0.98).loc[keep]
        pd.DataFrame(
            {
                "Date": f["date"],
                "Open": f["close"],
                "High": f["close"],
                "Low": f["close"],
                "Close": f["close"],
                "Adj Close": f["adj_close"],
                "Volume": f["volume"],
            }
        ).to_csv(raw / f"prices_{t}.csv", index=False)
    pd.DataFrame({"Date": BDAYS[keep], "Close": VIX[keep].to_numpy()}).to_csv(
        raw / "vix.csv", index=False
    )
    for sid, s in FRED.items():
        pd.DataFrame({"observation_date": BDAYS[keep], sid: s[keep].to_numpy()}).to_csv(
            raw / f"fred_{sid.lower()}.csv", index=False
        )
    return raw


@pytest.fixture
def live_dir(tmp_path: Path) -> Path:
    return tmp_path / "live"


def prices_fetcher(
    adj_factor: float = 1.0,
    fail: bool = False,
    last_date: str | None = None,
    drop: dict[str, list[str]] | None = None,
    spike: dict[str, float] | None = None,
):
    def fetch(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
        if fail:
            raise SourceError("source down")
        f = _price_frame(ticker, adj_factor)
        f = f[f["date"] >= start]
        if last_date:
            f = f[f["date"] <= pd.Timestamp(last_date)]
        if drop and ticker in drop:
            f = f[~f["date"].isin(pd.to_datetime(drop[ticker]))]
        if spike and ticker in spike:
            f = f.copy()
            f.loc[f["date"] > VINTAGE_LAST, ["close", "adj_close"]] *= spike[ticker]
        return f.reset_index(drop=True)

    return fetch


def series_fetcher(series: pd.Series, fail: bool = False, scale: float = 1.0):
    def fetch(start: pd.Timestamp) -> pd.Series:
        if fail:
            raise SourceError("source down")
        return series[series.index >= start] * scale

    return fetch


def make_sources(
    price_chain: list | None = None,
    vix_chain: list | None = None,
    macro_fail: bool = False,
    macro: dict[str, pd.Series] | None = None,
) -> Sources:
    return Sources(
        prices=price_chain if price_chain is not None else [("yahoo", prices_fetcher())],
        vix=vix_chain if vix_chain is not None else [("yahoo", series_fetcher(VIX))],
        macro={sid: series_fetcher(s, fail=macro_fail) for sid, s in (macro or FRED).items()},
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_sessions_are_appended_with_returns_from_the_fetched_history(raw_dir, live_dir):
    status = li.refresh(NOW, make_sources(), raw_dir, live_dir)
    stored = pd.read_csv(live_dir / li.PRICES_FILE, parse_dates=["date"])
    assert status["state"] == "current"
    assert status["last_close"] == "2026-09-15"
    assert status["new_price_rows"] == 2
    assert sorted(stored["date"].unique()) == FIRST_NEW
    assert set(stored["ticker"]) == set(li.TICKERS)
    assert not status["backup_used"]
    assert status["errors"] == []
    spy = stored[stored["ticker"] == "SPY"].set_index("date")
    day = BDAYS.get_loc(FIRST_NEW[0])
    expected = np.log(PRICES["SPY"][day] / PRICES["SPY"][day - 1])
    assert spy.loc[FIRST_NEW[0], "log_return"] == pytest.approx(expected)
    assert (stored["source"] == "yahoo").all()


def test_restated_adjusted_prices_do_not_change_returns(raw_dir, live_dir):
    sources = make_sources([("yahoo", prices_fetcher(adj_factor=0.9))])
    status = li.refresh(NOW, sources, raw_dir, live_dir)
    assert status["state"] == "current"
    assert status["new_price_rows"] == 2


def test_unfinished_session_is_not_ingested(raw_dir, live_dir):
    before_close = datetime(2026, 9, 15, 19, 0, tzinfo=UTC)
    status = li.refresh(before_close, make_sources(), raw_dir, live_dir)
    stored = pd.read_csv(live_dir / li.PRICES_FILE, parse_dates=["date"])
    assert stored["date"].max() == pd.Timestamp("2026-09-14")
    assert status["last_close"] == "2026-09-14"
    assert status["state"] == "current"


def test_second_run_adds_nothing(raw_dir, live_dir):
    li.refresh(NOW, make_sources(), raw_dir, live_dir)
    first = digest(live_dir / li.PRICES_FILE)
    status = li.refresh(NOW, make_sources(), raw_dir, live_dir)
    assert digest(live_dir / li.PRICES_FILE) == first
    assert status["new_price_rows"] == 0
    assert status["state"] == "current"


def test_incremental_run_extends_stored_data(raw_dir, live_dir):
    early = datetime(2026, 9, 14, 23, 0, tzinfo=UTC)
    li.refresh(early, make_sources(), raw_dir, live_dir)
    status = li.refresh(NOW, make_sources(), raw_dir, live_dir)
    stored = pd.read_csv(live_dir / li.PRICES_FILE, parse_dates=["date"])
    assert status["new_price_rows"] == 1
    assert sorted(stored["date"].unique()) == FIRST_NEW
    assert not stored.duplicated(["date", "ticker"]).any()


def test_backup_price_source_keeps_the_data_current_and_is_flagged(raw_dir, live_dir):
    sources = make_sources(
        [("yahoo", prices_fetcher(fail=True)), ("tiingo", prices_fetcher(adj_factor=0.97))]
    )
    status = li.refresh(NOW, sources, raw_dir, live_dir)
    stored = pd.read_csv(live_dir / li.PRICES_FILE)
    assert (stored["source"] == "tiingo").all()
    assert status["components"]["prices"]["sources"] == dict.fromkeys(li.TICKERS, "tiingo")
    assert status["state"] == "current"
    assert status["backup_used"]
    assert any("yahoo" in e for e in status["errors"])


def test_all_sources_failing_keeps_the_last_good_data(raw_dir, live_dir):
    li.refresh(datetime(2026, 9, 14, 23, 0, tzinfo=UTC), make_sources(), raw_dir, live_dir)
    files = [live_dir / li.PRICES_FILE, live_dir / li.VIX_FILE, live_dir / li.MACRO_FILE]
    before = [digest(f) for f in files]
    good_at = li.read_status(live_dir)["last_successful_fetch_at"]
    down = make_sources(
        [("yahoo", prices_fetcher(fail=True)), ("tiingo", prices_fetcher(fail=True))],
        [("yahoo", series_fetcher(VIX, fail=True)), ("fred", series_fetcher(VIX, fail=True))],
        macro_fail=True,
    )
    status = li.refresh(NOW, down, raw_dir, live_dir)
    assert [digest(f) for f in files] == before
    assert status["state"] == "fallback"
    assert status["last_close"] == "2026-09-14"
    assert status["sessions_behind"] == 1
    assert status["last_successful_fetch_at"] == good_at
    assert not status["components"]["prices"]["ok"]
    assert not status["components"]["macro"]["ok"]
    assert len(status["errors"]) >= 4


def test_first_run_with_everything_down_still_writes_a_status(raw_dir, live_dir):
    down = make_sources(
        [("yahoo", prices_fetcher(fail=True))],
        [("yahoo", series_fetcher(VIX, fail=True))],
        macro_fail=True,
    )
    status = li.refresh(NOW, down, raw_dir, live_dir)
    assert status["state"] == "fallback"
    assert status["last_close"] == "2026-09-11"
    assert status["last_successful_fetch_at"] is None
    assert not (live_dir / li.PRICES_FILE).exists()
    assert json.loads((live_dir / li.STATUS_FILE).read_text("utf-8")) == status


def test_source_disagreeing_with_stored_history_is_rejected(raw_dir, live_dir):
    rng = np.random.default_rng(99)
    skewed = _price_frame("SPY")
    skewed["adj_close"] = skewed["adj_close"] * np.exp(rng.normal(0, 0.02, len(skewed)))

    def bad(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
        frame = prices_fetcher()(ticker, start)
        return skewed[skewed["date"] >= start] if ticker == "SPY" else frame

    status = li.refresh(NOW, make_sources([("yahoo", bad)]), raw_dir, live_dir)
    assert status["state"] == "fallback"
    assert not (live_dir / li.PRICES_FILE).exists()
    assert any("differs from stored data" in e for e in status["errors"])


def test_absurd_daily_move_is_rejected(raw_dir, live_dir):
    sources = make_sources([("yahoo", prices_fetcher(spike={"AAPL": 2.0}))])
    status = li.refresh(NOW, sources, raw_dir, live_dir)
    assert status["state"] == "fallback"
    assert not (live_dir / li.PRICES_FILE).exists()
    assert any("daily move" in e for e in status["errors"])


def test_dates_are_held_back_where_a_ticker_lacks_them(raw_dir, live_dir):
    sources = make_sources([("yahoo", prices_fetcher(drop={"MSFT": ["2026-09-15"]}))])
    status = li.refresh(NOW, sources, raw_dir, live_dir)
    stored = pd.read_csv(live_dir / li.PRICES_FILE, parse_dates=["date"])
    assert stored["date"].max() == pd.Timestamp("2026-09-14")
    assert status["last_close"] == "2026-09-14"
    assert any("trading calendar" in e for e in status["errors"])


def test_a_gap_in_one_ticker_stops_all_tickers_at_the_gap(raw_dir, live_dir):
    sources = make_sources([("yahoo", prices_fetcher(drop={"JPM": ["2026-09-14"]}))])
    li.refresh(NOW, sources, raw_dir, live_dir)
    assert not (live_dir / li.PRICES_FILE).exists()


def test_vix_falls_back_to_fred_and_records_the_source(raw_dir, live_dir):
    sources = make_sources(
        vix_chain=[
            ("yahoo", series_fetcher(VIX, fail=True)),
            ("fred", series_fetcher(VIX, scale=1.0001)),
        ]
    )
    status = li.refresh(NOW, sources, raw_dir, live_dir)
    stored = pd.read_csv(live_dir / li.VIX_FILE, parse_dates=["date"])
    assert stored["source"].unique().tolist() == ["fred"]
    assert stored["date"].max() == pd.Timestamp("2026-09-15")
    assert status["components"]["vix"] == {"ok": True, "source": "fred"}


def test_macro_series_are_stored_side_by_side(raw_dir, live_dir):
    li.refresh(NOW, make_sources(), raw_dir, live_dir)
    macro = pd.read_csv(live_dir / li.MACRO_FILE, parse_dates=["date"])
    assert list(macro.columns) == ["date", "dgs10", "t10y2y"]
    assert macro["date"].max() == pd.Timestamp("2026-09-15")


def test_late_fred_observation_fills_the_gap_in_the_stored_row(raw_dir, live_dir):
    lagged = {"DGS10": FRED["DGS10"][:"2026-09-14"], "T10Y2Y": FRED["T10Y2Y"]}
    li.refresh(NOW, make_sources(macro=lagged), raw_dir, live_dir)
    first = pd.read_csv(live_dir / li.MACRO_FILE, parse_dates=["date"]).set_index("date")
    assert np.isnan(first.loc["2026-09-15", "dgs10"])
    li.refresh(NOW, make_sources(), raw_dir, live_dir)
    macro = pd.read_csv(live_dir / li.MACRO_FILE, parse_dates=["date"])
    assert not macro["date"].duplicated().any()
    filled = macro.set_index("date").loc["2026-09-15"]
    assert filled["dgs10"] == pytest.approx(FRED["DGS10"]["2026-09-15"])
    assert filled["t10y2y"] == pytest.approx(FRED["T10Y2Y"]["2026-09-15"])


def test_source_that_lags_reports_delayed_not_fallback(raw_dir, live_dir):
    sources = make_sources([("yahoo", prices_fetcher(last_date="2026-09-14"))])
    status = li.refresh(NOW, sources, raw_dir, live_dir)
    assert status["last_close"] == "2026-09-14"
    assert status["sessions_behind"] == 1
    assert status["state"] == "delayed"


def test_vintage_files_are_never_written(raw_dir, live_dir):
    before = {p.name: digest(p) for p in raw_dir.iterdir()}
    li.refresh(NOW, make_sources(), raw_dir, live_dir)
    assert {p.name: digest(p) for p in raw_dir.iterdir()} == before


def test_no_temp_files_are_left_behind(raw_dir, live_dir):
    li.refresh(NOW, make_sources(), raw_dir, live_dir)
    assert not list(live_dir.glob("*.tmp"))


def test_retries_recover_from_transient_errors_then_give_up():
    calls: list[int] = []
    delays: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("blip")
        return "ok"

    assert li.with_retries(flaky, attempts=3, base_delay=2, sleep=delays.append)() == "ok"
    assert delays == [2, 4]

    def broken() -> str:
        raise ConnectionError("down")

    with pytest.raises(SourceError, match="ConnectionError"):
        li.with_retries(broken, attempts=2, sleep=lambda s: None)()


def test_source_errors_are_not_retried():
    calls: list[int] = []

    def no_key() -> str:
        calls.append(1)
        raise SourceError("no key")

    with pytest.raises(SourceError):
        li.with_retries(no_key, attempts=3, sleep=lambda s: None)()
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 9, 20, 15, 0, tzinfo=UTC), "2026-09-18"),
        (datetime(2026, 9, 21, 12, 0, tzinfo=UTC), "2026-09-18"),
        (datetime(2026, 9, 21, 22, 30, tzinfo=UTC), "2026-09-21"),
        (datetime(2026, 12, 15, 22, 30, tzinfo=UTC), "2026-12-15"),
        (datetime(2026, 12, 15, 21, 30, tzinfo=UTC), "2026-12-14"),
    ],
)
def test_expected_last_session_follows_new_york_close(now, expected):
    assert f"{li.expected_last_session(now):%Y-%m-%d}" == expected


class FakeResponse:
    def __init__(self, payload: object = None, text: str = "") -> None:
        self._payload, self.text = payload, text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_tiingo_adapter_parses_adjusted_prices(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "secret")
    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return FakeResponse(
            [
                {"date": "2026-09-17T00:00:00.000Z", "close": 10.0, "adjClose": 9.9, "volume": 5},
                {"date": "2026-09-18T00:00:00.000Z", "close": 11.0, "adjClose": 10.9, "volume": 6},
            ]
        )

    monkeypatch.setattr(li.requests, "get", fake_get)
    df = li.tiingo_prices("SPY", pd.Timestamp("2026-09-01"))
    assert df["date"].tolist() == [pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-18")]
    assert df["adj_close"].tolist() == [9.9, 10.9]
    assert seen["headers"] == {"Authorization": "Token secret"}
    assert "secret" not in seen["url"]


def test_tiingo_adapter_needs_a_key(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(SourceError, match="TIINGO_API_KEY"):
        li.tiingo_prices("SPY", pd.Timestamp("2026-09-01"))


def test_fred_adapter_drops_missing_observations(monkeypatch):
    text = "observation_date,VIXCLS\n2026-09-16,15.5\n2026-09-17,.\n2026-09-18,14.8\n"
    monkeypatch.setattr(li.requests, "get", lambda url, params, timeout: FakeResponse(text=text))
    series = li.fred_series("VIXCLS", pd.Timestamp("2026-09-01"))
    assert series.index.tolist() == [pd.Timestamp("2026-09-16"), pd.Timestamp("2026-09-18")]


def test_yahoo_adapter_strips_time_zones(monkeypatch):
    index = pd.DatetimeIndex(["2026-09-17", "2026-09-18"]).tz_localize("America/New_York")
    history = pd.DataFrame(
        {"Close": [10.0, 11.0], "Adj Close": [9.9, 11.0], "Volume": [5, 6]}, index=index
    )

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, start: str, auto_adjust: bool) -> pd.DataFrame:
            return history

    monkeypatch.setattr(li.yf, "Ticker", FakeTicker)
    df = li.yahoo_prices("SPY", pd.Timestamp("2026-09-01"))
    assert df["date"].dt.tz is None
    assert df["date"].tolist() == [pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-18")]


def test_yahoo_empty_response_is_a_source_error(monkeypatch):
    class Empty:
        def __init__(self, symbol: str) -> None:
            pass

        def history(self, start: str, auto_adjust: bool) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(li.yf, "Ticker", Empty)
    with pytest.raises(SourceError):
        li.yahoo_prices("SPY", pd.Timestamp("2026-09-01"))
