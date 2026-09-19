from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from risklens.dashboard_data import ANNUALIZE, DashboardData

SURFACE = "#111c30"
INK = "#e8edf7"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "rgba(255,255,255,0.07)"
BASELINE = "rgba(255,255,255,0.22)"
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
VIOLET = "#9085e9"
GOOD = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
LEVEL_COLOR = {"LOW": GOOD, "MEDIUM": WARN, "HIGH": CRIT}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(height: int, **extra) -> dict:
    layout = {
        "height": height,
        "margin": {"l": 44, "r": 16, "t": 28, "b": 30},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "color": INK_2, "size": 12},
        "hoverlabel": {"bgcolor": SURFACE, "font": {"color": INK, "family": FONT}},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "font": {"size": 12, "color": INK_2},
        },
    }
    layout.update(extra)
    return layout


def style_axes(fig: go.Figure, **y_extra) -> None:
    fig.update_xaxes(showgrid=False, linecolor=BASELINE, tickfont={"color": MUTED, "size": 11})
    fig.update_yaxes(
        gridcolor=GRID, zeroline=False, tickfont={"color": MUTED, "size": 11}, **y_extra
    )


def risk_gauge(percentile: float, level: str, low_pct: float, high_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=percentile,
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, low_pct, high_pct, 100],
                    "ticktext": ["0", f"P{low_pct:g}", f"P{high_pct:g}", "100"],
                    "tickfont": {"size": 11, "color": MUTED},
                },
                "bar": {"color": "rgba(0,0,0,0)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, low_pct], "color": rgba(GOOD, 0.75)},
                    {"range": [low_pct, high_pct], "color": rgba(WARN, 0.75)},
                    {"range": [high_pct, 100], "color": rgba(CRIT, 0.80)},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 5},
                    "thickness": 0.95,
                    "value": percentile,
                },
            },
        )
    )
    fig.update_layout(base_layout(150, margin={"l": 34, "r": 34, "t": 8, "b": 6}))
    return fig


def sparkline(values: pd.Series, color: str = BLUE) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=values.index,
            y=values,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate="%{x|%d %b}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[values.index[-1]],
            y=[values.iloc[-1]],
            mode="markers",
            marker={"size": 9, "color": color, "line": {"color": SURFACE, "width": 2}},
            hoverinfo="skip",
        )
    )
    fig.update_layout(base_layout(56, margin={"l": 0, "r": 8, "t": 4, "b": 4}, showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def fan_chart(data: DashboardData, ctx: dict, history_days: int = 60) -> go.Figure:
    hist = data.market.loc[: ctx["origin_date"], "adj_close"].iloc[-history_days:]
    x0, x1, p0 = ctx["origin_date"], ctx["target_date"], ctx["price0"]
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.76, 0.24], horizontal_spacing=0.02
    )
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist,
            mode="lines",
            name="Price",
            line={"color": BLUE, "width": 2},
            hovertemplate="%{x|%d %b %Y}<br>$%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    alphas = {0.95: 0.20, 0.80: 0.34, 0.50: 0.55}
    for level in (0.95, 0.80, 0.50):
        lo, hi = ctx["bands"][level]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0],
                y=[p0, hi, lo, p0],
                mode="lines",
                fill="toself",
                line={"width": 0},
                fillcolor=rgba(BLUE, alphas[level]),
                name=f"{int(level * 100)}% band",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1, 1, 0],
                y=[lo, lo, hi, hi],
                mode="lines",
                fill="toself",
                line={"width": 0},
                fillcolor=rgba(BLUE, alphas[level]),
                showlegend=False,
                hovertemplate=f"{int(level * 100)}% range: ${lo:,.2f} to ${hi:,.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[0.06, 0.94],
            y=[ctx["median_price"]] * 2,
            mode="lines",
            line={"color": "#ffffff", "width": 2},
            name="Expected",
            hovertemplate=f"Expected ${ctx['median_price']:,.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    if ctx["realized_price"] is not None:
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[ctx["realized_price"]],
                mode="markers",
                name="Realized",
                marker={"size": 11, "color": ORANGE, "line": {"color": SURFACE, "width": 2}},
                hovertemplate=f"Realized ${ctx['realized_price']:,.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    lo95, hi95 = ctx["bands"][0.95]
    fig.add_annotation(
        x=0.5,
        y=hi95,
        xref="x2",
        yref="y",
        text=f"95% range ${lo95:,.0f} to ${hi95:,.0f}",
        showarrow=False,
        yshift=14,
        font={"color": INK, "size": 12},
    )
    low = min(hist.min(), lo95)
    high = max(hist.max(), hi95)
    pad = (high - low) * 0.08
    fig.update_layout(base_layout(330, margin={"l": 52, "r": 12, "t": 34, "b": 28}))
    fig.update_yaxes(
        range=[low - pad, high + pad * 2], tickprefix="$", gridcolor=GRID, zeroline=False
    )
    fig.update_xaxes(showgrid=False, linecolor=BASELINE, tickfont={"color": MUTED, "size": 11})
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=2)
    fig.update_yaxes(tickfont={"color": MUTED, "size": 11})
    return fig


def vol_chart(data: DashboardData, ctx: dict, days: int = 250) -> go.Figure:
    sig = data.signals.loc[: ctx["target_date"]].tail(days)
    forecast = sig["forecast_volatility"] * ANNUALIZE * 100
    realized = data.market.loc[sig["origin_date"], "roll_vol_21"].to_numpy() * ANNUALIZE * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sig.index,
            y=realized,
            name="21-day realized",
            mode="lines",
            line={"color": MUTED, "width": 2},
            hovertemplate="%{x|%d %b %Y}: %{y:.1f}%<extra>21-day realized</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sig.index,
            y=forecast,
            name="GARCH forecast",
            mode="lines",
            line={"color": BLUE, "width": 2},
            hovertemplate="%{x|%d %b %Y}: %{y:.1f}%<extra>GARCH forecast</extra>",
        )
    )
    fig.update_layout(base_layout(230, hovermode="x unified"))
    style_axes(fig, ticksuffix="%")
    return fig


def regime_strip(regimes: pd.DataFrame, ctx: dict) -> go.Figure:
    scale = [
        [0, GOOD],
        [1 / 3, GOOD],
        [1 / 3, WARN],
        [2 / 3, WARN],
        [2 / 3, CRIT],
        [1, CRIT],
    ]
    fig = go.Figure(
        go.Heatmap(
            x=regimes.index,
            y=[""],
            z=[regimes["regime"].to_numpy()],
            customdata=[regimes["realized_vol_pct"].to_numpy()],
            colorscale=scale,
            zmin=0,
            zmax=2,
            showscale=False,
            hovertemplate="%{x|%d %b %Y}<br>21-day realized vol %{customdata:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(x=ctx["origin_date"], line={"color": "#ffffff", "width": 2})
    events = (
        ("COVID crash", "2020-03-16", "right"),
        ("2022 bear", "2022-06-15", "left"),
    )
    for label, when, anchor in events:
        fig.add_annotation(
            x=when,
            y=1.0,
            yref="paper",
            yshift=12,
            text=label,
            showarrow=False,
            xanchor=anchor,
            font={"color": INK_2, "size": 11},
        )
    fig.update_layout(base_layout(96, margin={"l": 8, "r": 8, "t": 30, "b": 26}, showlegend=False))
    fig.update_xaxes(showgrid=False, tickfont={"color": MUTED, "size": 11})
    fig.update_yaxes(visible=False)
    return fig


def score_bar(score: float, theta: float) -> go.Figure:
    limit = max(0.4, abs(score) * 1.3, theta * 2.5)
    fig = go.Figure(
        go.Bar(
            x=[score],
            y=[""],
            orientation="h",
            marker={"color": VIOLET},
            width=0.42,
            hovertemplate="score %{x:.3f}<extra></extra>",
        )
    )
    thresholds = (
        (theta, f"BUY above {theta:.2f}", "left"),
        (-theta, f"SELL below -{theta:.2f}", "right"),
    )
    for value, label, anchor in thresholds:
        fig.add_vline(x=value, line={"color": WARN, "width": 2, "dash": "dot"})
        fig.add_annotation(
            x=value,
            y=1.25,
            yref="paper",
            text=label,
            showarrow=False,
            xanchor=anchor,
            font={"color": INK_2, "size": 11},
        )
    fig.update_layout(base_layout(96, margin={"l": 8, "r": 8, "t": 26, "b": 24}, showlegend=False))
    fig.update_xaxes(
        range=[-limit, limit],
        gridcolor=GRID,
        zeroline=True,
        zerolinecolor=BASELINE,
        tickfont={"color": MUTED, "size": 11},
    )
    fig.update_yaxes(visible=False)
    return fig


def equity_chart(equity: pd.DataFrame, ticker: str = "SPY") -> go.Figure:
    series = [
        (f"equity_buy_and_hold_{ticker}", f"Buy-and-hold {ticker}", BLUE, None),
        ("equity_market_reference_buy_and_hold_SPY", "Buy-and-hold SPY (market)", MUTED, "dot"),
        ("equity_A_score_rule", "A. Score rule", ORANGE, None),
        ("equity_B_vol_filter", "B. Volatility filter", AQUA, None),
    ]
    fig = go.Figure()
    for column, name, color, dash in series:
        if column not in equity:
            continue
        line = {"color": color, "width": 2}
        if dash:
            line["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=equity["date"],
                y=equity[column],
                name=name,
                mode="lines",
                line=line,
                hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra>" + name + "</extra>",
            )
        )
        fig.add_annotation(
            x=equity["date"].iloc[-1],
            y=equity[column].iloc[-1],
            text=f"{equity[column].iloc[-1]:.2f}",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font={"color": INK_2, "size": 11},
        )
    fig.update_layout(
        base_layout(250, margin={"l": 40, "r": 40, "t": 34, "b": 28}, hovermode="x unified")
    )
    style_axes(fig, tickprefix="$")
    return fig
