from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from bisect import bisect_left
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklens import charts as ch
from risklens import clean
from risklens import dashboard_data as dd
from risklens import live_ingest as li
from risklens import live_market as lm
from risklens import live_monitor as mon
from risklens import live_pipeline as pipe
from risklens import live_scoring as ls
from risklens.calibrate import TRAIN_END
from risklens.signals import validate_signals
from risklens.volatility import train_conditional_sigma

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "data" / "gold" / "gold_signals_daily.parquet",
    ROOT / "reports" / "results" / "test_metrics.json",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED), reason="pipeline outputs not built"
)
NEW_DAYS = pd.bdate_range("2026-09-21", "2026-09-23")
NOW = datetime(2026, 9, 23, 23, 0, tzinfo=UTC)
FROZEN_LIVE_TARGET = pd.Timestamp("2026-09-21")
NUMERIC_FIELDS = [
    "expected_return",
    "forecast_volatility",
    "risk_adjusted_score",
    "vol_percentile",
    "pi_lower",
    "pi_upper",
    "pi_lower_80",
    "pi_upper_80",
    "pi_lower_50",
    "pi_upper_50",
]


def synthetic_sources(raw: Path) -> li.Sources:
    """Real vintage history plus three invented sessions; DGS10 lags one day like FRED."""
    rng = np.random.default_rng(7)
    prices: dict[str, pd.DataFrame] = {}
    for ticker in li.TICKERS:
        hist = clean.read_prices(raw / f"prices_{ticker}.csv", ticker)
        hist = hist[["date", "close", "adj_close", "volume"]]
        steps = np.exp(np.cumsum(rng.normal(0, 0.01, len(NEW_DAYS))))
        last = hist.iloc[-1]
        new = pd.DataFrame(
            {
                "date": NEW_DAYS,
                "close": last["close"] * steps,
                "adj_close": last["adj_close"] * steps,
                "volume": 1_000_000,
            }
        )
        prices[ticker] = pd.concat([hist, new], ignore_index=True)

    def extend(hist: pd.Series, values: np.ndarray, days: pd.DatetimeIndex) -> pd.Series:
        return pd.concat([hist.dropna(), pd.Series(values, index=days)])

    vix = extend(clean.read_vix(raw / "vix.csv"), 15 + rng.normal(size=3), NEW_DAYS)
    fred = {}
    for name in li.MACRO_SERIES:
        hist = clean.read_fred(raw / f"fred_{name.lower()}.csv").dropna()
        days = NEW_DAYS[:-1] if name == "DGS10" else NEW_DAYS
        fred[name] = extend(hist, hist.iloc[-1] + 0.01 * np.arange(len(days)), days)

    def yahoo(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
        frame = prices[ticker]
        return frame[frame["date"] >= start].reset_index(drop=True)

    return li.Sources(
        prices=[("yahoo", yahoo)],
        vix=[("yahoo", lambda start: vix[vix.index >= start])],
        macro={n: (lambda start, n=n: fred[n][fred[n].index >= start]) for n in fred},
    )


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> pipe.Paths:
    root = tmp_path_factory.mktemp("project")
    for rel in ("data/raw", "data/processed", "data/gold", "config", "reports/results"):
        shutil.copytree(ROOT / rel, root / rel)
    paths = pipe.Paths(root)
    paths_status = pipe.run(NOW, synthetic_sources(paths.raw), paths)
    (root / "status.json").write_text(json.dumps(paths_status), encoding="utf-8")
    return paths


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_reports_current_data_and_writes_the_live_layer(project):
    status = json.loads((project.root / "status.json").read_text("utf-8"))
    assert status["state"] == "current"
    assert status["last_close"] == "2026-09-23"
    assert status["new_price_rows"] == 3
    for name in (
        li.PRICES_FILE,
        li.VIX_FILE,
        li.MACRO_FILE,
        li.STATUS_FILE,
        lm.LIVE_MARKET_FILE,
        ls.SIGNALS_LIVE_FILE,
        mon.MONITOR_FILE,
    ):
        assert (project.live / name).exists(), name
    assert (project.reports / pipe.REPORT_NAME).exists()


def test_live_text_files_use_unix_line_endings(project):
    files = [
        project.live / n
        for n in (li.PRICES_FILE, li.VIX_FILE, li.MACRO_FILE, li.STATUS_FILE, mon.MONITOR_FILE)
    ] + [project.reports / pipe.REPORT_NAME]
    assert all(b"\r" not in f.read_bytes() for f in files)


def test_frozen_outputs_are_untouched(project):
    for rel in (
        "data/gold/gold_market_daily.parquet",
        "data/gold/gold_signals_daily.parquet",
        "data/raw/VINTAGE.json",
        "reports/results/test_metrics.json",
    ):
        assert digest(project.root / rel) == digest(ROOT / rel), rel


def test_live_market_extends_each_ticker_by_the_new_sessions(project):
    market = pd.read_parquet(project.live / lm.LIVE_MARKET_FILE)
    gold = pd.read_parquet(project.gold / "gold_market_daily.parquet")
    assert len(market) == 4 * len(NEW_DAYS)
    assert market["date"].min() > gold["date"].max()
    assert sorted(market["date"].unique()) == list(NEW_DAYS)
    assert list(market.columns) == list(gold.columns)
    assert market.notna().all().all()


def test_live_price_level_continues_the_frozen_series(project):
    market = pd.read_parquet(project.live / lm.LIVE_MARKET_FILE)
    gold = pd.read_parquet(project.gold / "gold_market_daily.parquet")
    spy = market[market["ticker"] == "SPY"].sort_values("date")
    last_frozen = gold[gold["ticker"] == "SPY"].sort_values("date")["adj_close"].iloc[-1]
    first = spy.iloc[0]
    assert first["adj_close"] == pytest.approx(last_frozen * np.exp(first["log_return"]))


def test_signal_rows_follow_the_frozen_schedule(project):
    signals = pd.read_parquet(project.live / ls.SIGNALS_LIVE_FILE)
    validate_signals(signals)
    assert signals["vol_percentile"].between(0, 100).all()
    for ticker, rows in signals.groupby("ticker"):
        rows = rows.sort_values("date")
        assert rows["split"].tolist() == ["post_freeze"] * 3 + ["live"], ticker
        assert rows["origin_date"].iloc[0] == pd.Timestamp("2026-09-18")
        assert rows["date"].iloc[0] == FROZEN_LIVE_TARGET
        assert rows["date"].iloc[-1] == pd.Timestamp("2026-09-24")
        assert rows["origin_date"].iloc[-1] == pd.Timestamp("2026-09-23")


def test_forecast_for_the_last_frozen_origin_is_unchanged_by_later_data(project):
    signals = pd.read_parquet(project.live / ls.SIGNALS_LIVE_FILE)
    frozen = pd.read_parquet(project.gold / "gold_signals_daily.parquet")
    for ticker in li.TICKERS:
        new = signals[(signals["ticker"] == ticker) & (signals["date"] == FROZEN_LIVE_TARGET)]
        old = frozen[(frozen["ticker"] == ticker) & (frozen["split"] == "live")]
        for field in NUMERIC_FIELDS:
            assert float(new[field].iloc[0]) == pytest.approx(float(old[field].iloc[0]), abs=1e-12)
        assert new["signal"].iloc[0] == old["signal"].iloc[0]
        assert new["train_end_date"].iloc[0] == old["train_end_date"].iloc[0]


def test_percentile_history_includes_earlier_forecasts(project):
    signals = pd.read_parquet(project.live / ls.SIGNALS_LIVE_FILE)
    frozen = pd.read_parquet(project.gold / "gold_signals_daily.parquet")
    gold = pd.read_parquet(project.gold / "gold_market_daily.parquet")
    returns = gold[gold["ticker"] == "SPY"].sort_values("date").set_index("date")["log_return"]
    returns = returns.iloc[1:]
    train_end = int(returns.index.searchsorted(pd.Timestamp(TRAIN_END), side="right") - 1)
    train_sigma = train_conditional_sigma(returns.iloc[: train_end + 1]).to_numpy()
    seen = frozen[(frozen["ticker"] == "SPY") & frozen["split"].isin(["validation", "test"])]
    live = signals[signals["ticker"] == "SPY"].sort_values("date")
    history = sorted(
        [*train_sigma, *seen["forecast_volatility"], live["forecast_volatility"].iloc[0]]
    )
    second = live.iloc[1]
    expected = 100 * bisect_left(history, second["forecast_volatility"]) / len(history)
    assert second["vol_percentile"] == pytest.approx(expected)


def test_monitor_counts_realized_forecasts_and_waits_for_enough_data(project):
    monitor = json.loads((project.live / mon.MONITOR_FILE).read_text("utf-8"))
    for ticker in li.TICKERS:
        m = monitor["tickers"][ticker]
        assert m["n_forecasts"] == 3
        assert m["status"] == "insufficient_data"
        assert 0 <= m["coverage_95"] <= 1
    report = (project.reports / pipe.REPORT_NAME).read_text("utf-8")
    assert "| SPY | 3 |" in report
    assert "Latest signals" in report


def test_second_run_without_new_data_does_not_rescore(project):
    before = digest(project.live / ls.SIGNALS_LIVE_FILE)
    status = pipe.run(NOW, synthetic_sources(project.raw), project)
    assert status["new_price_rows"] == 0
    assert digest(project.live / ls.SIGNALS_LIVE_FILE) == before


def test_dashboard_merges_the_live_layer(project):
    data = dd.load_dashboard_data(project.root, "SPY")
    assert data.market.index[-1] == pd.Timestamp("2026-09-23")
    assert data.signals.index[-1] == pd.Timestamp("2026-09-24")
    assert data.signals.loc[FROZEN_LIVE_TARGET, "split"] == "post_freeze"
    assert data.signals["split"].iloc[-1] == "live"
    assert not data.signals.index.duplicated().any()
    assert data.status["state"] == "current"
    assert dd.data_chip(data) == "Live data · close 2026-09-23"
    assert dd.live_banner(data, pd.Timestamp("2026-09-24")) is None


def test_context_for_realized_and_pending_live_rows(project):
    data = dd.load_dashboard_data(project.root, "AAPL")
    realized = dd.forecast_context(data, pd.Timestamp("2026-09-22"))
    pending = dd.forecast_context(data, data.signals.index[-1])
    assert realized["split"] == "post_freeze"
    assert realized["realized_price"] is not None
    assert pending["realized_price"] is None
    assert realized["no_lookahead_ok"] and pending["no_lookahead_ok"]
    regimes = dd.regime_table(data.market, data.config)
    assert all(len(f.data) > 0 for f in (ch.fan_chart(data, pending), ch.vol_chart(data, pending)))
    assert ch.regime_strip(regimes, pending).data


def test_banner_reports_each_data_state(project):
    data = dd.load_dashboard_data(project.root, "SPY")
    today = pd.Timestamp("2026-09-24")

    def banner(**changes):
        status = data.status | changes
        return dd.live_banner(dataclasses.replace(data, status=status), today)

    delayed = banner(state="delayed")
    assert delayed["level"] == "info"
    fallback = banner(state="fallback", errors=["SPY via yahoo: source down"])
    assert fallback["level"] == "warn"
    assert "source down" in fallback["text"]
    behind = banner(last_close="2026-09-25")
    assert "computed only through" in behind["text"]
    dead = dd.live_banner(data, pd.Timestamp("2026-10-05"))
    assert "has not run since" in dead["text"]


def test_banner_without_a_status_uses_the_frozen_vintage_rule():
    data = dd.load_dashboard_data(ROOT, "SPY")
    data = dataclasses.replace(data, status=None)
    assert dd.live_banner(data, pd.Timestamp("2026-09-20")) is None
    assert "days old" in dd.live_banner(data, pd.Timestamp("2026-09-30"))["text"]


def test_monitor_rows_for_the_app(project):
    data = dd.load_dashboard_data(project.root, "SPY")
    rows = dict(dd.monitor_rows(data))
    assert rows["Realized forecasts"] == "3"
    assert rows["Status"] == "insufficient data"
    assert dd.monitor_rows(dataclasses.replace(data, monitor=None)) is None


def test_fingerprint_changes_when_the_live_layer_is_rewritten(project, tmp_path):
    assert dd.data_fingerprint(tmp_path) == ()
    first = dd.data_fingerprint(project.root)
    target = project.live / mon.MONITOR_FILE
    original = target.read_bytes()
    try:
        target.write_bytes(original + b" ")
        assert dd.data_fingerprint(project.root) != first
    finally:
        target.write_bytes(original)


def test_app_renders_with_the_live_layer(project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    original = dd.load_dashboard_data
    monkeypatch.setattr(
        dd, "load_dashboard_data", lambda root, ticker: original(project.root, ticker)
    )
    monkeypatch.setattr(dd, "data_fingerprint", lambda root=None: ("synthetic-live",))
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Live data · close 2026-09-23" in text
    assert "Live monitor" in text
    assert "Forecast for Thu 24 Sep 2026" in text


def test_market_layer_refuses_a_history_that_differs_from_the_frozen_gold(project):
    gold = pd.read_parquet(project.gold / "gold_market_daily.parquet")
    tampered = gold.copy()
    tampered.loc[tampered.index[-5], "log_return"] += 0.01
    with pytest.raises(ValueError, match="changed frozen history"):
        lm.assert_matches_frozen(tampered, gold)
    with pytest.raises(ValueError, match="calendar"):
        lm.assert_matches_frozen(gold.iloc[:-1], gold)


def test_monitor_flags_poor_calibration_only_with_enough_forecasts():
    n = 30
    idx = pd.bdate_range("2026-10-01", periods=n)
    rows = pd.DataFrame(
        {
            "forecast_volatility": np.full(n, 0.01),
            "pi_lower": -0.02,
            "pi_upper": 0.02,
            "pi_lower_80": -0.013,
            "pi_upper_80": 0.013,
            "pi_lower_50": -0.007,
            "pi_upper_50": 0.007,
        },
        index=idx,
    )
    calm = np.full(n, 0.005)
    ok = mon.ticker_metrics(rows, calm, np.full(n, 0.02**2))
    assert ok["status"] == "ok"
    assert ok["coverage_95"] == 1.0
    wild = np.full(n, 0.05)
    bad = mon.ticker_metrics(rows, wild, np.full(n, 0.02**2))
    assert bad["status"] == "watch"
    assert bad["coverage_95"] == 0.0
    short = mon.ticker_metrics(rows.iloc[:5], wild[:5], np.full(5, 0.02**2))
    assert short["status"] == "insufficient_data"
    assert mon.ticker_metrics(rows.iloc[:0], wild[:0], wild[:0])["status"] == "insufficient_data"
