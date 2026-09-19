from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANNUALIZE = float(np.sqrt(252))
BAND_SUFFIX = {0.50: "_50", 0.80: "_80", 0.95: ""}
ASSET_LABELS = {
    "SPY": "SPY - S&P 500 ETF",
    "AAPL": "AAPL - Apple",
    "MSFT": "MSFT - Microsoft",
    "JPM": "JPM - JPMorgan Chase",
}
A_KEY = "A_score_rule (primary: next-close execution)"
B_KEY = "B_vol_filter (primary: next-close execution)"


def strategy_keys(ticker: str) -> dict[str, str]:
    keys = {f"buy_and_hold_{ticker}": f"Buy-and-hold {ticker}"}
    if ticker != "SPY":
        keys["market_reference_buy_and_hold_SPY"] = "Buy-and-hold SPY (market)"
    keys[A_KEY] = "A. Score rule"
    keys[B_KEY] = "B. Volatility filter"
    return keys


@dataclass(frozen=True)
class DashboardData:
    ticker: str
    signals: pd.DataFrame
    market: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict
    config: dict
    vintage: dict


def load_dashboard_data(root: Path = ROOT, ticker: str = "SPY") -> DashboardData:
    suffix = "" if ticker == "SPY" else f"_{ticker}"
    signals = pd.read_parquet(root / "data" / "gold" / "gold_signals_daily.parquet")
    signals = signals[signals["ticker"] == ticker].sort_values("date").set_index("date", drop=False)
    signals.index.name = None
    market = pd.read_parquet(root / "data" / "gold" / "gold_market_daily.parquet")
    market = market[market["ticker"] == ticker].sort_values("date").set_index("date")
    results = root / "reports" / "results"
    equity = pd.read_csv(results / f"test_equity_curves{suffix}.csv", parse_dates=["date"])
    metrics = json.loads((results / f"test_metrics{suffix}.json").read_text("utf-8"))
    if ticker == "SPY":
        config = json.loads((root / "config" / "preregistered.json").read_text("utf-8"))
    else:
        extension = json.loads(
            (root / "config" / "preregistered_extension.json").read_text("utf-8")
        )
        config = extension["tickers"][ticker]
    vintage = json.loads((root / "data" / "raw" / "VINTAGE.json").read_text("utf-8"))
    return DashboardData(ticker, signals, market, equity, metrics, config, vintage)


def strategy_note(data: DashboardData) -> str:
    """Plain-language status of the signal layer, derived from the pre-registered test result."""
    perf, ticker = data.metrics["performance"], data.ticker
    bh = perf[f"buy_and_hold_{ticker}"]
    a, b = perf[A_KEY], perf[B_KEY]
    verdicts = data.metrics["verdicts"]
    if not any(verdicts.values()):
        return (
            "Signals stay EXPERIMENTAL. The pre-registered acceptance rule was not met on the "
            f"sealed test: Sharpe {a['sharpe']:.2f} (A) and {b['sharpe']:.2f} (B) vs "
            f"{bh['sharpe']:.2f} for buy-and-hold {ticker}. Promotion requires a stable "
            "advantage across walk-forward windows."
        )
    return (
        "Signals stay EXPERIMENTAL. The score rule formally meets the acceptance rule "
        f"(Sharpe {a['sharpe']:.2f} vs {bh['sharpe']:.2f} for buy-and-hold {ticker}), but it made "
        f"{a['trades']:.0f} trade(s): it behaves as buy-and-hold, so this is not evidence of "
        f"timing skill. The volatility filter (Sharpe {b['sharpe']:.2f}) did not meet the rule."
    )


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def closest_analogue(
    market: pd.DataFrame, origin_date: pd.Timestamp, forecast_vol_pct: float, min_gap: int = 63
) -> tuple[pd.Timestamp, float]:
    """Past date (at least `min_gap` trading days back) with the closest 21-day realized volatility."""
    history = market.loc[:origin_date]
    history = history.iloc[:-min_gap] if len(history) > min_gap else history
    realized = history["roll_vol_21"] * ANNUALIZE * 100
    date = (realized - forecast_vol_pct).abs().idxmin()
    return date, float(realized.loc[date])


def regime_table(market: pd.DataFrame, config: dict) -> pd.DataFrame:
    realized = market["roll_vol_21"] * ANNUALIZE * 100
    train = realized.loc[: config["splits"]["train_end"]]
    levels = config["risk_levels"]
    low, high = np.percentile(train, [levels["low_below_pct"], levels["high_above_pct"]])
    raw = pd.Series(np.select([realized < low, realized > high], [0, 2], 1), index=market.index)
    smoothed = raw.rolling(63, min_periods=1).median().round().astype(int)
    return pd.DataFrame({"regime": smoothed, "realized_vol_pct": realized}, index=market.index)


def forecast_context(data: DashboardData, target_date: pd.Timestamp) -> dict:
    row = data.signals.loc[target_date]
    origin = pd.Timestamp(row["origin_date"])
    price0 = float(data.market.loc[origin, "adj_close"])
    forecast_vol = float(row["forecast_volatility"]) * ANNUALIZE * 100
    realized_vol = float(data.market.loc[origin, "roll_vol_21"]) * ANNUALIZE * 100
    bands = {
        level: (
            price0 * float(np.exp(row[f"pi_lower{suffix}"])),
            price0 * float(np.exp(row[f"pi_upper{suffix}"])),
        )
        for level, suffix in BAND_SUFFIX.items()
    }
    analog_date, analog_vol = closest_analogue(data.market, origin, forecast_vol)
    rule = data.config["strategy_A_rule"]
    coverage = data.metrics["joint_intervals"]["95%"]
    realized_price = (
        float(data.market.loc[target_date, "adj_close"])
        if target_date in data.market.index
        else None
    )
    return {
        "target_date": pd.Timestamp(target_date),
        "origin_date": origin,
        "split": str(row["split"]),
        "price0": price0,
        "median_price": price0 * float(np.exp(row["expected_return"])),
        "bands": bands,
        "realized_price": realized_price,
        "expected_return_pct": float(row["expected_return"]) * 100,
        "ret_lo_pct": float(row["pi_lower"]) * 100,
        "ret_hi_pct": float(row["pi_upper"]) * 100,
        "forecast_vol_pct": forecast_vol,
        "realized_vol_pct": realized_vol,
        "vol_percentile": float(row["vol_percentile"]),
        "risk_level": str(row["risk_level"]).upper(),
        "score": float(row["risk_adjusted_score"]),
        "theta": float(rule["theta_buy"]),
        "signal": str(row["signal"]),
        "vol_filter_state": str(row["vol_filter_state"]),
        "model_version": str(row["model_version"]),
        "rule_version": str(row["rule_version"]),
        "train_end_date": pd.Timestamp(row["train_end_date"]),
        "no_lookahead_ok": bool(pd.Timestamp(row["train_end_date"]) < pd.Timestamp(target_date)),
        "coverage_pct": float(coverage["coverage"]) * 100,
        "coverage_kupiec_p": float(coverage["kupiec_p"]),
        "analog_date": analog_date,
        "analog_vol_pct": analog_vol,
    }


def build_explanation(ctx: dict) -> str:
    """Grounded restatement: every number comes from the displayed forecast row."""
    relation = "above" if ctx["forecast_vol_pct"] > ctx["realized_vol_pct"] else "below"
    score, theta = ctx["score"], ctx["theta"]
    if ctx["signal"] == "BUY":
        rule_text = (
            f"the score {score:.2f} is above the BUY threshold {theta:.2f}, so the rule signals BUY"
        )
    elif ctx["signal"] == "SELL":
        rule_text = f"the score {score:.2f} is below the SELL threshold -{theta:.2f}, so the rule signals SELL"
    else:
        rule_text = (
            f"the score {score:.2f} is inside the ±{theta:.2f} band, so the rule keeps the position "
            "on HOLD"
        )
    return (
        f"Forecast volatility for {ctx['target_date']:%d %b %Y} is {ctx['forecast_vol_pct']:.1f}% "
        f"annualized, {relation} the 21-day realized {ctx['realized_vol_pct']:.1f}% and in the "
        f"{ordinal(ctx['vol_percentile'])} percentile of history: the risk regime is "
        f"{ctx['risk_level']}. The expected return ({ctx['expected_return_pct']:+.2f}%) relative to "
        f"that risk: {rule_text}. On the sealed test the 95% ranges covered "
        f"{ctx['coverage_pct']:.1f}% of outcomes (target 95%)."
    )


def data_health(vintage: dict, ticker: str = "SPY") -> list[dict]:
    files = vintage["files"]
    prices = files[f"prices_{ticker}.csv"]
    rows = [
        {"label": "Prices (Yahoo)", "detail": f"{prices['rows']:,} rows", "status": "ok"},
        {"label": "VIX (^VIX)", "detail": f"to {files['vix.csv']['last_date']}", "status": "ok"},
        {
            "label": "FRED DGS10 / T10Y2Y",
            "detail": "lagged 1 trading day",
            "status": "note",
        },
    ]
    return rows


def staleness(vintage: dict, today: pd.Timestamp) -> dict | None:
    last = pd.Timestamp(files_last_date(vintage))
    gap = (today.normalize() - last).days
    if gap < 2:
        return None
    return {
        "vintage": vintage["vintage_date"],
        "last_close": f"{last:%Y-%m-%d}",
        "days": gap,
    }


def files_last_date(vintage: dict) -> str:
    return vintage["files"]["prices_SPY.csv"]["last_date"]
