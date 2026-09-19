from __future__ import annotations

import hashlib
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_raw_files_match_the_vintage_manifest() -> None:
    manifest = json.loads((RAW / "VINTAGE.json").read_text(encoding="utf-8"))
    for name, meta in manifest["files"].items():
        digest = hashlib.sha256((RAW / name).read_bytes()).hexdigest()
        assert digest == meta["sha256"], f"{name} differs from the frozen vintage"
