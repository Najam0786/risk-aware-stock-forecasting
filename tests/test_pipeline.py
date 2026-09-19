from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklens.build_gold import GOLD_DIR, build_gold_market, validate_gold_market
from risklens.clean import build_macro, cpi_yoy_asof, read_fred


def synthetic_prices(n: int = 60, tickers: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                "date": dates,
                "ticker": t,
                "adj_close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                "volume": 1000,
            }
        )
        for t in tickers
    ]
    return pd.concat(frames, ignore_index=True)


def synthetic_macro(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "vix_close": 15 + rng.normal(0, 1, len(dates)),
            "dgs10": 3 + rng.normal(0, 0.05, len(dates)),
            "t10y2y": 0.5 + rng.normal(0, 0.05, len(dates)),
            "cpi_yoy": 2.0,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_read_fred_parses_dot_and_blank(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("observation_date,DGS10\n2020-01-01,.\n2020-01-02,\n2020-01-03,1.5\n")
    series = read_fred(path)
    assert series.dtype == "float64"
    assert series.isna().tolist() == [True, True, False]


def test_macro_forward_fill_only_and_lagged() -> None:
    spine = pd.bdate_range("2020-01-06", periods=5)
    dgs10 = pd.Series(
        [1.0, np.nan, 3.0], index=pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"])
    )
    dgs10.name = "DGS10"
    t10y2y = dgs10.rename("T10Y2Y")
    vix = pd.Series(20.0, index=spine, name="vix_close")
    cpi = pd.Series(
        np.arange(100.0, 115.0), index=pd.date_range("2018-01-01", periods=15, freq="MS")
    )
    macro = build_macro(spine, vix, dgs10, t10y2y, cpi)
    assert np.isnan(macro["dgs10"].iloc[0])
    assert macro["dgs10"].iloc[1] == 1.0
    assert macro["dgs10"].iloc[2] == 1.0
    assert macro["dgs10"].iloc[3] == 3.0
    assert macro["dgs10"].iloc[4] == 3.0


def test_cpi_asof_respects_publication_lag() -> None:
    months = pd.date_range("2019-01-01", periods=15, freq="MS")
    cpi = pd.Series(np.linspace(100, 115, 15), index=months)
    spine = pd.DatetimeIndex(["2020-02-10", "2020-02-20", "2020-04-20"])
    result = cpi_yoy_asof(cpi, spine)
    jan_yoy = (cpi.loc["2020-01-01"] / cpi.loc["2019-01-01"] - 1) * 100
    assert np.isnan(result.loc["2020-02-10"]) or result.loc["2020-02-10"] != jan_yoy
    assert result.loc["2020-02-20"] == pytest.approx(jan_yoy)


def test_features_do_not_look_ahead() -> None:
    prices = synthetic_prices()
    macro = synthetic_macro(pd.DatetimeIndex(prices["date"].unique()))
    base = build_gold_market(prices, macro)
    changed = prices.copy()
    last_date = changed["date"].max()
    changed.loc[changed["date"] == last_date, "adj_close"] *= 1.5
    altered = build_gold_market(changed, macro)
    before = base[base["date"] < last_date].reset_index(drop=True)
    after = altered[altered["date"] < last_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_gold_validation_catches_duplicates() -> None:
    prices = synthetic_prices()
    gold = build_gold_market(prices, synthetic_macro(pd.DatetimeIndex(prices["date"].unique())))
    validate_gold_market(gold)
    with pytest.raises(ValueError):
        validate_gold_market(pd.concat([gold, gold.iloc[[0]]], ignore_index=True))


@pytest.mark.skipif(
    not (GOLD_DIR / "gold_market_daily.parquet").exists(), reason="gold layer not built"
)
def test_gold_file_contract() -> None:
    gold = pd.read_parquet(GOLD_DIR / "gold_market_daily.parquet")
    validate_gold_market(gold)
    assert set(gold["ticker"]) == {"SPY", "AAPL", "MSFT", "JPM"}
    assert str(gold["day_of_week"].dtype) == "int8"
    assert gold["log_return"].abs().max() < 0.5
    spy = gold[gold["ticker"] == "SPY"].sort_values("date")
    assert spy["date"].is_monotonic_increasing
    recomputed = np.log(spy["adj_close"]).diff().dropna()
    np.testing.assert_allclose(spy["log_return"].iloc[1:], recomputed, atol=1e-12)
