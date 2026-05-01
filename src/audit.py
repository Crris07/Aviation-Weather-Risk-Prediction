"""Quick audit of ingested parquet files: row counts, year coverage, missingness."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .ingest import ROOT, load_config


def main() -> int:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]

    for role in ("train", "test"):
        print(f"\n=== {role.upper()} ===")
        start = cfg["windows"][role]["start"]
        end = cfg["windows"][role]["end"]
        for code, meta in cfg["stations"][role].items():
            path = raw_dir / f"{code}_{start}_{end}.parquet"
            if not path.exists():
                print(f"{code}: FILE MISSING")
                continue
            df = pd.read_parquet(path)
            ts = df["timestamp"]
            by_year = df.groupby(ts.dt.year).size().to_dict()
            miss_pct = (df[["temp_c","dew_c","vis_m","wind_mps","slp_hpa","ceiling_m","precip_mm"]]
                        .isna().mean() * 100).round(1).to_dict()
            print(f"\n{code} ({meta['geography']}) — {len(df):,} rows")
            print(f"  years: {by_year}")
            print(f"  missing %: {miss_pct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
