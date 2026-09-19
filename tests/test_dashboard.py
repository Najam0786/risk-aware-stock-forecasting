from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from risklens import charts as ch
from risklens import dashboard_data as dd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "data" / "gold" / "gold_signals_daily.parquet",
    ROOT / "reports" / "results" / "test_metrics.json",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED), reason="pipeline outputs not built"
)


@pytest.fixture(scope="module")
def data() -> dd.DashboardData:
    return dd.load_dashboard_data(ROOT)


@pytest.mark.parametrize("split", ["live", "test", "validation"])
def test_context_and_explanation_for_each_split(data: dd.DashboardData, split: str) -> None:
    date = data.signals[data.signals["split"] == split].index[-1]
    ctx = dd.forecast_context(data, date)
    assert ctx["no_lookahead_ok"]
    assert ctx["bands"][0.95][0] < ctx["bands"][0.80][0] < ctx["bands"][0.50][0]
    assert ctx["bands"][0.50][1] < ctx["bands"][0.80][1] < ctx["bands"][0.95][1]
    assert (ctx["realized_price"] is None) == (split == "live")
    text = dd.build_explanation(ctx)
    assert f"{ctx['forecast_vol_pct']:.1f}%" in text
    assert ctx["risk_level"] in text


def test_analogue_never_looks_forward(data: dd.DashboardData) -> None:
    date = data.signals[data.signals["split"] == "validation"].index[10]
    ctx = dd.forecast_context(data, date)
    assert ctx["analog_date"] < ctx["origin_date"]


def test_all_charts_build(data: dd.DashboardData) -> None:
    ctx = dd.forecast_context(data, data.signals.index[-1])
    regimes = dd.regime_table(data.market, data.config)
    assert set(regimes["regime"].unique()) <= {0, 1, 2}
    figures = [
        ch.risk_gauge(ctx["vol_percentile"], ctx["risk_level"], 40, 85),
        ch.fan_chart(data, ctx),
        ch.vol_chart(data, ctx),
        ch.regime_strip(regimes, ctx),
        ch.score_bar(ctx["score"], ctx["theta"]),
        ch.equity_chart(data.equity),
        ch.sparkline(data.signals["forecast_volatility"].tail(30)),
    ]
    assert all(len(f.data) > 0 for f in figures)


def test_staleness_banner_logic() -> None:
    vintage = {
        "vintage_date": "2026-09-19",
        "files": {"prices_SPY.csv": {"last_date": "2026-09-18"}},
    }
    assert dd.staleness(vintage, pd.Timestamp("2026-09-19")) is None
    assert dd.staleness(vintage, pd.Timestamp("2026-09-22"))["days"] == 4


def test_calibration_report_page_renders() -> None:
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Calibration_report.py")).run()
    assert not page.exception
    assert len(page.tabs) == 4


def test_main_page_renders_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not app.exception
    assert any("RiskLens" in m.value for m in app.markdown)
