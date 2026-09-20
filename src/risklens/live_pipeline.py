from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from risklens import live_ingest as li
from risklens import live_market as lm
from risklens import live_monitor as mon
from risklens import live_scoring as ls

ROOT = Path(__file__).resolve().parents[2]
REPORT_NAME = "live_monitoring.md"


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def gold(self) -> Path:
        return self.root / "data" / "gold"

    @property
    def live(self) -> Path:
        return self.root / "data" / "live"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def outputs_current(paths: Paths, market: pd.DataFrame) -> bool:
    signals_path = paths.live / ls.SIGNALS_LIVE_FILE
    if not signals_path.exists() or not (paths.live / lm.LIVE_MARKET_FILE).exists():
        return False
    signals = pd.read_parquet(signals_path)
    last_origin = signals.loc[signals["split"] == "live", "origin_date"].max()
    return bool(last_origin == market["date"].max())


def run(
    now: datetime | None = None,
    sources: li.Sources | None = None,
    paths: Paths | None = None,
    force: bool = False,
) -> dict:
    """Refresh data, then rebuild the live market, signals and monitoring from it."""
    paths = paths or Paths()
    status = li.refresh(now, sources, paths.raw, paths.live)
    if li.read_live_prices(paths.live).empty:
        return status

    market = lm.build_live_market(paths.raw, paths.processed, paths.gold, paths.live)
    if not force and outputs_current(paths, market):
        return status

    frozen = pd.read_parquet(paths.gold / "gold_market_daily.parquet")
    full_market = pd.concat([frozen, market], ignore_index=True)
    frozen_signals = pd.read_parquet(paths.gold / "gold_signals_daily.parquet")
    signals = ls.score_all(full_market, frozen_signals, paths.root)
    monitor = mon.build_monitor(signals, full_market, now)

    lm.write_live_market(market, paths.live)
    mon.write_monitor(monitor, paths.live)
    (paths.reports / REPORT_NAME).write_text(
        mon.render_report(monitor, status, signals), encoding="utf-8", newline="\n"
    )
    ls.write_signals_live(signals, paths.live)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh live data, signals and monitoring")
    parser.add_argument("--force", action="store_true", help="rescore even if outputs are current")
    status = run(force=parser.parse_args().force)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
