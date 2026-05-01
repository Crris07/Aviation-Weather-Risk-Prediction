"""Retry KBOS 2025 (and inspect current file). Non-destructive."""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

from .ingest import (
    ROOT, load_config, fetch_year, parse_isd, add_units, build_station_id,
)

def main() -> int:
    cfg = load_config()
    meta = cfg["stations"]["test"]["KBOS"]
    sid = build_station_id(meta["usaf"], meta["wban"])
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    parquet = raw_dir / "KBOS_2024_2025.parquet"

    # Look at what's in there now
    if parquet.exists():
        cur = pd.read_parquet(parquet)
        per_year = cur.groupby(cur["timestamp"].dt.year).size()
        print(f"Current KBOS rows per year:\n{per_year}\n")
    else:
        cur = pd.DataFrame()

    # Retry 2025
    print("Retrying 2025 ...")
    for attempt in range(1, 4):
        raw = fetch_year(sid, 2025, cfg["noaa_data_types"], timeout=120)
        if raw is not None and len(raw) > 200:
            break
        print(f"  attempt {attempt} empty/failed, waiting...")
        time.sleep(5)
    else:
        print("❌ could not fetch 2025")
        return 1

    parsed = parse_isd(raw)
    parsed["timestamp"] = pd.to_datetime(raw["DATE"], errors="coerce", utc=True)
    parsed["airport"]  = "KBOS"
    parsed["usaf"]     = meta["usaf"]
    parsed["wban"]     = meta["wban"]
    parsed = add_units(parsed)

    # Drop rows with no usable weather signal at all
    core_cols = ["temp_c", "dew_c", "vis_m", "wind_mps"]
    parsed = parsed.dropna(subset=core_cols, how="all")

    print(f"Fetched 2025: {len(parsed):,} rows")

    # Merge with existing 2024 data (dropping any overlap) and re-save
    if not cur.empty:
        cur_24 = cur[cur["timestamp"].dt.year == 2024]
        combined = pd.concat([cur_24, parsed], ignore_index=True)
    else:
        combined = parsed

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_parquet(parquet, index=False)
    per_year = combined.groupby(combined["timestamp"].dt.year).size()
    print(f"\nNew KBOS rows per year:\n{per_year}")
    print(f"Total: {len(combined):,} rows → {parquet.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
