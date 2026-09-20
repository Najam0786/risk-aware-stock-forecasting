from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risklens import charts as ch  # noqa: E402
from risklens import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="RiskLens", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 1rem; max-width: 1850px;}
header[data-testid="stHeader"] {background: transparent;}
.stAppDeployButton {display: none;}
.zone {letter-spacing: .14em; font-size: .74rem; color: #8fa3c4; font-weight: 700; margin: .1rem 0 .5rem;}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .66rem;
        font-weight: 700; letter-spacing: .07em; margin-left: .5rem; vertical-align: middle;}
.badge-core {background: rgba(57,135,229,.18); color: #9cc6f7; border: 1px solid rgba(57,135,229,.5);}
.badge-exp {background: rgba(144,133,233,.18); color: #c0b8f5; border: 1px solid rgba(144,133,233,.55);}
.title {font-size: 1.7rem; font-weight: 800; letter-spacing: -.01em; color: #fff;}
.subtitle {font-size: .7rem; letter-spacing: .16em; color: #8fa3c4; margin-left: .6rem;}
.chip {display: inline-block; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08);
       border-radius: 999px; padding: 4px 12px; font-size: .8rem; color: #d3dcef; margin-right: .4rem;}
.tape {color: #d3dcef; font-size: .9rem;}
.tape b {color: #fff;}
.banner {border-left: 4px solid #fab219; background: rgba(250,178,25,.10); border-radius: 8px;
         padding: .55rem 1rem; font-size: .88rem; color: #f3e6c4; margin: .5rem 0 .7rem;}
.banner b {color: #fff;}
.kpi-label {font-size: .7rem; letter-spacing: .1em; color: #8fa3c4; text-transform: uppercase;}
.kpi-value {font-size: 2.1rem; font-weight: 800; line-height: 1.15; color: #fff;}
.kpi-sub {font-size: .82rem; color: #b8c4dc;}
.muted {color: #8fa3c4; font-size: .8rem;}
.item {display: flex; justify-content: space-between; gap: .6rem; font-size: .84rem; padding: 3px 0;
       border-bottom: 1px solid rgba(255,255,255,.05); color: #d3dcef;}
.item:last-child {border-bottom: none;}
.item span:first-child {color: #8fa3c4;}
.item span:last-child {text-align: right; font-weight: 600; color: #fff;}
.card-title {font-weight: 700; color: #fff; margin-bottom: .35rem; font-size: .95rem;}
.ok {display: block; margin-top: .4rem; padding: 5px 10px; border-radius: 8px; font-size: .78rem;
     background: rgba(12,163,12,.14); border: 1px solid rgba(12,163,12,.45); color: #d9f5dc;}
.dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;}
.signal {font-size: 2.3rem; font-weight: 800; letter-spacing: .04em; color: #fff; text-align: center;
         border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04);
         border-radius: 12px; padding: .5rem 1.1rem; display: inline-block;}
.explain {font-size: .92rem; line-height: 1.55; color: #e8edf7;}
.chips span {display: inline-block; font-size: .72rem; color: #b8c4dc; border: 1px solid rgba(255,255,255,.14);
             border-radius: 999px; padding: 3px 10px; margin: 6px 6px 0 0;}
.note {border-left: 4px solid #9085e9; background: rgba(144,133,233,.10); border-radius: 8px;
       padding: .5rem .9rem; font-size: .84rem; color: #e1ddfa; margin-top: .5rem;}
table.bt {width: 100%; border-collapse: collapse; font-size: .82rem; color: #d3dcef;}
table.bt th {color: #8fa3c4; font-weight: 600; text-align: right; padding: 4px 6px; font-size: .72rem;}
table.bt td {text-align: right; padding: 4px 6px; border-top: 1px solid rgba(255,255,255,.06);}
table.bt th:first-child, table.bt td:first-child {text-align: left;}
.legend {font-size: .8rem; color: #b8c4dc;}
.footer {margin-top: .8rem; padding-top: .6rem; border-top: 1px solid rgba(255,255,255,.08);
         font-size: .78rem; color: #8fa3c4;}
</style>
"""

SPLIT_TEXT = {
    "live": "Live forecast for the next session (not yet realized)",
    "test": "Sealed test period (out of sample)",
    "validation": "Validation period (rule thresholds were calibrated on this window)",
}


@st.cache_resource(show_spinner=False)
def load_data(ticker: str) -> dd.DashboardData:
    return dd.load_dashboard_data(ROOT, ticker)


@st.cache_resource(show_spinner=False)
def load_regimes(_data: dd.DashboardData) -> pd.DataFrame:
    return dd.regime_table(_data.market, _data.config)


def snap_to_forecast_date(index: pd.DatetimeIndex, picked: date) -> pd.Timestamp:
    earlier = index[index <= pd.Timestamp(picked)]
    return earlier[-1] if len(earlier) else index[0]


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def item(label: str, value: str) -> str:
    return f'<div class="item"><span>{label}</span><span>{value}</span></div>'


def plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def choose_asset() -> None:
    label = st.session_state["asset"]
    st.session_state["ticker"] = next(t for t, name in dd.ASSET_LABELS.items() if name == label)
    st.session_state.pop("run_date", None)
    st.session_state.pop("date_pick", None)


ticker = st.session_state.get("ticker", "SPY")
try:
    data = load_data(ticker)
except FileNotFoundError as error:
    st.error(
        f"No forecast available for {ticker}: a required data file is missing "
        f"({error.filename}). Rebuild the outputs with the commands in the README."
    )
    st.stop()
regimes = load_regimes(data)
signals = data.signals
st.markdown(CSS, unsafe_allow_html=True)

if "run_date" not in st.session_state:
    st.session_state["run_date"] = signals.index[-1]
ctx = dd.forecast_context(data, st.session_state["run_date"])
last_market = data.market.iloc[-1]

head_a, head_b, head_c = st.columns([2.2, 3.4, 2.4])
head_a.markdown(
    '<span class="title">RiskLens</span><span class="subtitle">RISK-AWARE MARKET MONITOR</span>',
    unsafe_allow_html=True,
)
head_b.markdown(
    f'<span class="tape">{ticker} <b>{last_market["adj_close"]:,.2f}</b> &nbsp;|&nbsp; '
    f"VIX <b>{last_market['vix_close']:.2f}</b> &nbsp;|&nbsp; "
    f"10Y <b>{last_market['dgs10']:.2f}%</b> "
    f'<span class="muted">(as of {data.market.index[-1]:%d %b %Y})</span></span>',
    unsafe_allow_html=True,
)
head_c.markdown(
    f'<span class="chip">Forecast for {ctx["target_date"]:%a %d %b %Y}</span>'
    f'<span class="chip">Vintage {data.vintage["vintage_date"]}</span>',
    unsafe_allow_html=True,
)

stale = dd.staleness(data.vintage, pd.Timestamp.today())
if stale:
    st.markdown(
        f'<div class="banner"><b>! Market data is {stale["days"]} days old '
        f"(vintage {stale['vintage']}).</b> The forecast uses the last available close "
        f"({stale['last_close']}), a newer close may exist. Re-run the ingestion after the next "
        "market close to refresh.</div>",
        unsafe_allow_html=True,
    )

col_a, col_b, col_c = st.columns([1.05, 2.75, 1.7], gap="medium")

with col_a:
    st.markdown('<div class="zone">1 · CONFIGURE</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.selectbox(
            "Asset",
            list(dd.ASSET_LABELS.values()),
            index=list(dd.ASSET_LABELS).index(ticker),
            key="asset",
            on_change=choose_asset,
        )
        st.caption("Each asset has its own pre-registered model and thresholds.")
        st.text_input("Forecast horizon", "1 trading day (fixed in MVP)", disabled=True)
        picked = st.date_input(
            "Forecast for",
            value=signals.index[-1].date(),
            min_value=signals.index[0].date(),
            max_value=signals.index[-1].date(),
            key="date_pick",
        )

        def run_forecast() -> None:
            st.session_state["run_date"] = snap_to_forecast_date(
                signals.index, st.session_state["date_pick"]
            )

        st.button("Run forecast", type="primary", width="stretch", on_click=run_forecast)
        st.caption(SPLIT_TEXT[ctx["split"]])
    with st.container(border=True):
        check = (
            '<span class="ok">✓ train_end_date &lt; forecast date: verified</span>'
            if ctx["no_lookahead_ok"]
            else '<span class="ok" style="border-color:#d03b3b">! look-ahead check failed</span>'
        )
        rule = data.config["strategy_A_rule"]
        st.markdown(
            '<div class="card-title">Model and rules: audit trail</div>'
            + item("Model version", ctx["model_version"])
            + item("Decision rules", f"{ctx['rule_version']} (θ = {rule['theta_buy']:.2f})")
            + item("Trained through", f"{ctx['train_end_date']:%Y-%m-%d}")
            + check,
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        rows = "".join(
            item(f"{dot(ch.GOOD if h['status'] == 'ok' else ch.WARN)}{h['label']}", h["detail"])
            for h in dd.data_health(data.vintage, ticker)
        )
        st.markdown('<div class="card-title">Data health</div>' + rows, unsafe_allow_html=True)
    with st.container(border=True):
        window = data.metrics["window"]
        st.markdown(
            '<div class="card-title">Evaluation protocol</div>'
            + item("Sealed test", f"{window[0]} to {window[1]}")
            + item("Walk-forward refit", "every 21 days")
            + item("Configuration", "pre-registered, tested once")
            + item("Transaction costs", "5 bps per trade"),
            unsafe_allow_html=True,
        )
    st.page_link("pages/1_Calibration_report.py", label="View calibration report")

with col_b:
    st.markdown(
        '<div class="zone">2 · RISK ASSESSMENT<span class="badge badge-core">CORE</span></div>',
        unsafe_allow_html=True,
    )
    k1, k2, k3 = st.columns(3)
    levels = data.config["risk_levels"]
    with k1, st.container(border=True, height=270):
        color = ch.LEVEL_COLOR[ctx["risk_level"]]
        st.markdown(
            '<div class="kpi-label">Risk level</div>'
            f'<div class="kpi-value">{dot(color)}{ctx["risk_level"]}</div>'
            f'<div class="kpi-sub">{dd.ordinal(ctx["vol_percentile"])} percentile of history</div>',
            unsafe_allow_html=True,
        )
        plot(
            ch.risk_gauge(
                ctx["vol_percentile"],
                ctx["risk_level"],
                levels["low_below_pct"],
                levels["high_above_pct"],
            ),
            "gauge",
        )
    with k2, st.container(border=True, height=270):
        delta = ctx["forecast_vol_pct"] - ctx["realized_vol_pct"]
        sign = "above" if delta > 0 else "below"
        st.markdown(
            '<div class="kpi-label">Forecast volatility (annualized)</div>'
            f'<div class="kpi-value">{ctx["forecast_vol_pct"]:.1f}%</div>'
            f'<div class="kpi-sub">{abs(delta):.1f} pp {sign} 21-day realized '
            f"{ctx['realized_vol_pct']:.1f}%</div>",
            unsafe_allow_html=True,
        )
        recent = signals.loc[: ctx["target_date"]].tail(60)
        plot(ch.sparkline(recent["forecast_volatility"] * dd.ANNUALIZE * 100), "spark")
        st.caption("Last 60 forecasts")
    with k3, st.container(border=True, height=270):
        calibrated = ctx["coverage_kupiec_p"] > 0.05
        state = (
            f"{dot(ch.GOOD)}calibrated (Kupiec p = {ctx['coverage_kupiec_p']:.2f})"
            if calibrated
            else f"{dot(ch.CRIT)}not calibrated (Kupiec p = {ctx['coverage_kupiec_p']:.2f})"
        )
        st.markdown(
            '<div class="kpi-label">95% interval coverage (sealed test)</div>'
            f'<div class="kpi-value">{ctx["coverage_pct"]:.1f}%</div>'
            f'<div class="kpi-sub">target 95% · {state}</div>'
            f'<div class="muted" style="margin-top:.6rem">{data.metrics["n_forecasts"]} out-of-sample '
            "one-day forecasts. Coverage is the honesty metric: it says how far the ranges can be "
            "trusted.</div>",
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title">{ticker} price with 50 / 80 / 95% prediction bands '
            f'<span class="muted">· forecast for {ctx["target_date"]:%d %b %Y}</span></div>',
            unsafe_allow_html=True,
        )
        plot(ch.fan_chart(data, ctx), "fan")
        st.caption(
            "Bands follow the fan-chart convention: darker means more likely. They are estimates, "
            "not certainties. The panel on the right zooms into the one-day-ahead range."
        )
    v1, v2 = st.columns(2)
    with v1, st.container(border=True):
        st.markdown(
            '<div class="card-title">Volatility: forecast vs realized</div>', unsafe_allow_html=True
        )
        plot(ch.vol_chart(data, ctx), "vol")
        gap = ctx["forecast_vol_pct"] - ctx["realized_vol_pct"]
        st.caption(
            f"Forecast is {abs(gap):.1f} pp {'above' if gap > 0 else 'below'} the 21-day realized volatility."
        )
    with v2, st.container(border=True):
        st.markdown(
            '<div class="card-title">Risk regimes, 2010 to today</div>', unsafe_allow_html=True
        )
        plot(ch.regime_strip(regimes, ctx), "regimes")
        st.markdown(
            f'<div class="legend">{dot(ch.GOOD)}Low &nbsp; {dot(ch.WARN)}Medium &nbsp; '
            f'{dot(ch.CRIT)}High &nbsp;<span class="muted">(21-day realized volatility vs training '
            f"percentiles). White line: selected date.</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Closest historical analogue: {ctx['analog_date']:%b %Y} "
            f"(realized volatility {ctx['analog_vol_pct']:.1f}%)."
        )

with col_c:
    st.markdown(
        '<div class="zone">3 · SIGNAL &amp; EXPLANATION<span class="badge badge-exp">EXPERIMENTAL</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title">Signal for {ctx["target_date"]:%d %b %Y} · {ticker}</div>',
            unsafe_allow_html=True,
        )
        s1, s2 = st.columns([1, 1.5])
        s1.markdown(f'<div class="signal">{ctx["signal"]}</div>', unsafe_allow_html=True)
        s2.markdown(
            '<div class="kpi-label">Expected return (1 day)</div>'
            f'<div class="kpi-sub"><b style="color:#fff;font-size:1.2rem">'
            f"{ctx['expected_return_pct']:+.2f}%</b><br>95%: {ctx['ret_lo_pct']:+.2f}% to "
            f"{ctx['ret_hi_pct']:+.2f}%</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="kpi-label" style="margin-top:.5rem">Risk-adjusted score vs threshold '
            f'θ = {ctx["theta"]:.2f}: <b style="color:#fff">{ctx["score"]:+.3f}</b></div>',
            unsafe_allow_html=True,
        )
        plot(ch.score_bar(ctx["score"], ctx["theta"]), "score")
        state_text = "risk-on" if ctx["vol_filter_state"] == "RISK_ON" else "risk-off"
        st.caption(
            f"Volatility filter (strategy B): {state_text}. Holding is an informed decision, "
            "not a missing answer."
        )
    with st.container(border=True):
        st.markdown(
            '<div class="card-title">Explanation, generated from model output only</div>'
            f'<div class="explain">{dd.build_explanation(ctx)}</div>'
            '<div class="chips"><span>Source: gold_signals_daily · '
            f"{ctx['model_version']}</span><span>No values invented or added</span></div>",
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        perf = data.metrics["performance"]
        st.markdown(
            f'<div class="card-title">Strategies vs buy-and-hold {ticker}: sealed test, after costs</div>',
            unsafe_allow_html=True,
        )
        plot(ch.equity_chart(data.equity, ticker), "equity")
        body = "".join(
            f"<tr><td>{label}</td><td>{perf[key]['sharpe']:.2f}</td>"
            f"<td>{perf[key]['max_drawdown'] * 100:.1f}%</td>"
            f"<td>{perf[key]['cumulative_return'] * 100:+.1f}%</td><td>{perf[key]['trades']:.0f}</td></tr>"
            for key, label in dd.strategy_keys(ticker).items()
        )
        st.markdown(
            '<table class="bt"><tr><th>Strategy</th><th>Sharpe</th><th>Max DD</th><th>Cumulative</th>'
            f"<th>Trades</th></tr>{body}</table>",
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="note">{dd.strategy_note(data)}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Academic prototype, not financial advice. Forecasts are probabilistic estimates; '
    "backtested performance does not guarantee future results. Every signal is traceable to its model, "
    "rules and training window.</div>",
    unsafe_allow_html=True,
)
