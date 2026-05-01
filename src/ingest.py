"""
NOAA ISD global-hourly ingest for the Aviation Weather Risk project.

Pulls hourly observations per airport, year by year, parses the NOAA
packed column formats (TMP, DEW, VIS, WND, SLP, CIG, AA1), and caches
cleaned parquet files in data/raw/.

Usage:
    python -m src.ingest --role train
    python -m src.ingest --role test
    python -m src.ingest --role all --force     # re-download even if cached

NOAA ISD reference (column packed formats):
    https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
"""
from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests
import yaml
from tqdm import tqdm

# Paths / config                                                       
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



# NOAA parsing helpers                                                 
# Missing-value sentinels in ISD packed fields. A value equals the
# sentinel -> missing. Compare as strings before division.
_MISSING_TMP  = "+9999"
_MISSING_DEW  = "+9999"
_MISSING_VIS  = "999999"
_MISSING_WND_SPEED = "9999"
_MISSING_WND_DIR   = "999"
_MISSING_SLP  = "99999"
_MISSING_CIG  = "99999"
_MISSING_AA1_DEPTH = "9999"


def _split(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return DataFrame of split parts of a comma-packed ISD column.
    Missing column -> empty DataFrame of correct length."""
    if col not in df.columns:
        return pd.DataFrame(index=df.index)
    return df[col].astype(str).str.split(",", expand=True)


def _to_float(series: pd.Series, missing: str, scale: float = 1.0) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.where(s != missing, other=pd.NA)
    return pd.to_numeric(s, errors="coerce") / scale


def parse_isd(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the packed NOAA ISD columns into tidy numeric columns."""
    out = pd.DataFrame(index=df.index)

    # TMP: "temp,quality" ; temp is tenths of °C 
    tmp = _split(df, "TMP")
    if not tmp.empty:
        out["temp_c"] = _to_float(tmp[0], _MISSING_TMP, scale=10.0)

    # DEW: "dew,quality" ; tenths of °C 
    dew = _split(df, "DEW")
    if not dew.empty:
        out["dew_c"] = _to_float(dew[0], _MISSING_DEW, scale=10.0)

    # VIS: "distance,var,var_quality,quality" ; metres 
    vis = _split(df, "VIS")
    if not vis.empty:
        out["vis_m"] = _to_float(vis[0], _MISSING_VIS, scale=1.0)

    # WND: "dir,dir_q,type,speed,speed_q" 
    # speed is in tenths of m/s
    wnd = _split(df, "WND")
    if not wnd.empty:
        out["wind_dir_deg"] = _to_float(wnd[0], _MISSING_WND_DIR, scale=1.0)
        if wnd.shape[1] >= 4:
            out["wind_mps"] = _to_float(wnd[3], _MISSING_WND_SPEED, scale=10.0)

    # SLP: "pressure,quality" ; tenths of hPa 
    slp = _split(df, "SLP")
    if not slp.empty:
        out["slp_hpa"] = _to_float(slp[0], _MISSING_SLP, scale=10.0)

    # CIG: "height_m,quality,determination,cavok"
    cig = _split(df, "CIG")
    if not cig.empty:
        out["ceiling_m"] = _to_float(cig[0], _MISSING_CIG, scale=1.0)

    # AA1: "period_hr,depth_mm_tenths,condition,quality" 
    aa1 = _split(df, "AA1")
    if not aa1.empty and aa1.shape[1] >= 2:
        out["precip_mm"] = _to_float(aa1[1], _MISSING_AA1_DEPTH, scale=10.0)
    else:
        out["precip_mm"] = pd.NA

    return out


# Derived units                                                         
def add_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw NOAA units to model units AND drop the originals.

    The label thresholds are defined in km / knots (matching aviation
    conventions), so the model should only ever see km / knots. Keeping
    `vis_m` and `wind_mps` alongside `vis_km` and `wind_knots` made the
    same physical quantity appear twice in the feature matrix (correlation
    0.9999 / 1.0000) and silently doubled the model's effective splits on
    visibility and wind. Bug-found 2026-04-26.
    """
    df = df.copy()
    if "vis_m" in df:
        df["vis_km"] = df["vis_m"] / 1000.0
    if "wind_mps" in df:
        df["wind_knots"] = df["wind_mps"] * 1.9438444924
    return df.drop(columns=[c for c in ("vis_m", "wind_mps") if c in df.columns])


def strip_legacy_unit_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove `vis_m` / `wind_mps` if they're still on disk in raw parquets.

    Idempotent. Safe to call on a frame that's already clean. Used at load
    time (in label.load_and_label) so historical raw parquets that were
    written before the add_units fix don't poison downstream features.
    """
    return df.drop(columns=[c for c in ("vis_m", "wind_mps") if c in df.columns])


# NOAA API fetch                                                        
NOAA_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def fetch_year(
    station_id: str,
    year: int,
    data_types: Iterable[str],
    max_retries: int = 3,
    timeout: int = 60,
) -> Optional[pd.DataFrame]:
    """Fetch one (station, year) from NOAA ISD. Returns None if empty."""
    params = {
        "dataset": "global-hourly",
        "stations": station_id,
        "startDate": f"{year}-01-01T00:00:00",
        "endDate":   f"{year}-12-31T23:59:59",
        "dataTypes": ",".join(data_types),
        "format": "csv",
        # NOTE: NOT passing units=metric — we parse raw packed fields ourselves
        "includeAttributes": "false",
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(NOAA_URL, params=params, timeout=timeout)
            res.raise_for_status()
            if len(res.text.strip()) < 200:
                return None
            return pd.read_csv(StringIO(res.text), low_memory=False)
        except Exception as e:     # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            time.sleep(wait)
    print(f"    ❌ giving up on {station_id} {year}: {last_err}", file=sys.stderr)
    return None



# Per-station orchestration                                   
def build_station_id(usaf: str, wban: str) -> str:
    return f"{usaf}{wban}"


def ingest_station(
    code: str,
    meta: dict,
    start_year: int,
    end_year: int,
    data_types: Iterable[str],
    raw_dir: Path,
    force: bool = False,
) -> Optional[Path]:
    """Ingest one station across [start_year, end_year]. Returns output path or None."""
    out_path = raw_dir / f"{code}_{start_year}_{end_year}.parquet"
    if out_path.exists() and not force:
        print(f"  ⏩ cached: {out_path.name}")
        return out_path

    station_id = build_station_id(meta["usaf"], meta["wban"])
    pieces: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        df = fetch_year(station_id, year, data_types)
        if df is None:
            print(f"  ⚠️  {code} {year}: empty response")
            continue
        parsed = parse_isd(df)
        parsed["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
        parsed["airport"]   = code
        parsed["usaf"]      = meta["usaf"]
        parsed["wban"]      = meta["wban"]
        pieces.append(parsed)
        time.sleep(0.6)   # be polite to NOAA

    if not pieces:
        print(f"  ❌ no data at all for {code}")
        return None

    full = pd.concat(pieces, ignore_index=True)
    full = add_units(full)

    # drop rows with no usable weather signal at all
    # (post add_units, so we use the model-unit columns)
    core_cols = ["temp_c", "dew_c", "vis_km", "wind_knots"]
    full = full.dropna(subset=core_cols, how="all")

    full = full.sort_values("timestamp").reset_index(drop=True)
    full.to_parquet(out_path, index=False)
    print(f"  ✅ {code}: {len(full):,} rows → {out_path.name}")
    return out_path

# Main                                                                  
def ingest_role(role: str, cfg: dict, force: bool = False) -> None:
    stations = cfg["stations"][role]
    window   = cfg["windows"][role]
    dtypes   = cfg["noaa_data_types"]
    raw_dir  = ROOT / cfg["paths"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Ingesting {role.upper()} ({window['start']}–{window['end']}) ===")
    for code, meta in stations.items():
        print(f"\n▶ {code} ({meta['geography']}) — {meta['name']}")
        ingest_station(
            code, meta,
            start_year=window["start"],
            end_year=window["end"],
            data_types=dtypes,
            raw_dir=raw_dir,
            force=force,
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest NOAA ISD hourly data")
    parser.add_argument("--role", choices=["train", "test", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    cfg = load_config()
    roles = ["train", "test"] if args.role == "all" else [args.role]
    for r in roles:
        ingest_role(r, cfg, force=args.force)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
