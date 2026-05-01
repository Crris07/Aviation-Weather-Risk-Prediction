"""
Quick sanity test: pull 1 month of KDEN and print column summary.
Run BEFORE the full ingest so we don't waste an hour on bad params.

    python -m src.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .ingest import (
    NOAA_URL, build_station_id, parse_isd, add_units, load_config, fetch_year
)

def main() -> int:
    cfg = load_config()
    meta = cfg["stations"]["train"]["KDEN"]
    sid = build_station_id(meta["usaf"], meta["wban"])
    print(f"Smoke test: KDEN = {sid}, fetching 2023 ...")

    raw = fetch_year(sid, 2023, cfg["noaa_data_types"])
    if raw is None or raw.empty:
        print("❌ no data returned")
        return 1

    print(f"\nRaw columns ({len(raw.columns)}): {list(raw.columns)[:15]}...")
    print(f"Raw rows: {len(raw):,}")

    parsed = parse_isd(raw)
    parsed["timestamp"] = pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
    parsed = add_units(parsed)

    print("\nParsed summary:")
    print(parsed.describe(include="all").T[["count", "mean", "min", "max"]])
    print(f"\nNon-null counts:\n{parsed.notna().sum()}")
    print(f"\nFirst 3 rows:\n{parsed.head(3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
