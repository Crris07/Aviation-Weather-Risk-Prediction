"""
Label construction for Aviation Weather Risk project.

Defines:
    y_risk   — 1 if ANY timestep in (t+1h, t+2h] has
                  visibility < 3 km OR wind > 25 knots
    y_vis_future   — min visibility (km) in (t+1h, t+2h]
    y_wind_future  — max wind (knots) in (t+1h, t+2h]

Requires a clean *hourly* grid. We resample the raw NOAA data to a regular
hourly grid per airport before labelling (NOAA obs are ~every 53min
with irregular offsets).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ingest import ROOT, load_config, strip_legacy_unit_cols


def to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Resample one airport's raw frame to a regular hourly grid.

    Uses nearest obs within +/- 30 min of each hour. Fields that are
    missing stay NaN. Preserves airport/usaf/wban columns.
    """
    df = df.sort_values("timestamp").copy()

    # De-duplicate timestamps (NOAA sometimes has multiple report types per minute).
    # Collapse by averaging numeric fields within the same timestamp.
    numeric_cols = [c for c in df.columns
                    if df[c].dtype.kind in "fc" or df[c].dtype.kind in "iu"]
    meta_cols = [c for c in ("airport", "usaf", "wban") if c in df.columns]
    agg_map = {c: "mean" for c in numeric_cols}
    agg_map.update({c: "first" for c in meta_cols})
    df = df.groupby("timestamp", as_index=True).agg(agg_map).sort_index()

    # build hourly index covering the whole span
    start = df.index.min().floor("h")
    end = df.index.max().ceil("h")
    grid = pd.date_range(start=start, end=end, freq="h", tz="UTC")

    # reindex using nearest obs within 30min tolerance
    hourly = (df[numeric_cols]
              .reindex(grid, method="nearest", tolerance=pd.Timedelta("30min")))
    hourly.index.name = "timestamp"

    # meta columns (airport, usaf, wban) — just broadcast
    if "airport" in df.columns:
        hourly["airport"] = df["airport"].iloc[0]
    if "usaf" in df.columns:
        hourly["usaf"] = df["usaf"].iloc[0]
    if "wban" in df.columns:
        hourly["wban"] = df["wban"].iloc[0]

    return hourly.reset_index()


def build_labels(
    df_hourly: pd.DataFrame,
    vis_km_threshold: float = 3.0,
    wind_knots_threshold: float = 25.0,
    horizon_hours: int = 2,
) -> pd.DataFrame:
    """Add y_risk, y_vis_future, y_wind_future columns.

    Expects hourly regular grid. Horizon=2 -> look at t+1h and t+2h.
    """
    out = df_hourly.copy().sort_values(["airport", "timestamp"]).reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    for _, g in out.groupby("airport", sort=False):
        g = g.copy()
        vis = g["vis_km"]
        wind = g["wind_knots"]

        # future values at t+1, t+2, ...
        fut_vis = [vis.shift(-k) for k in range(1, horizon_hours + 1)]
        fut_wnd = [wind.shift(-k) for k in range(1, horizon_hours + 1)]

        fut_vis_df = pd.concat(fut_vis, axis=1)
        fut_wnd_df = pd.concat(fut_wnd, axis=1)

        # regression targets
        g["y_vis_future"]  = fut_vis_df.min(axis=1, skipna=True)
        g["y_wind_future"] = fut_wnd_df.max(axis=1, skipna=True)

        # classification target
        unsafe_vis  = (fut_vis_df < vis_km_threshold).any(axis=1)
        unsafe_wnd  = (fut_wnd_df > wind_knots_threshold).any(axis=1)
        y_risk = (unsafe_vis | unsafe_wnd).astype("Int8")

        # If ALL future values are missing, y_risk is undefined -> NA
        all_missing = fut_vis_df.isna().all(axis=1) & fut_wnd_df.isna().all(axis=1)
        y_risk[all_missing] = pd.NA
        g["y_risk"] = y_risk

        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


def load_and_label(role: str, cfg: dict | None = None) -> pd.DataFrame:
    """Convenience: load all parquet files for a role, stack, resample, label."""
    if cfg is None:
        cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    start = cfg["windows"][role]["start"]
    end = cfg["windows"][role]["end"]

    frames = []
    for code in cfg["stations"][role]:
        path = raw_dir / f"{code}_{start}_{end}.parquet"
        frames.append(pd.read_parquet(path))
    raw = pd.concat(frames, ignore_index=True)
    raw = strip_legacy_unit_cols(raw)

    hourly_parts = []
    for code, g in raw.groupby("airport", sort=False):
        hourly_parts.append(to_hourly_grid(g))
    hourly = pd.concat(hourly_parts, ignore_index=True)

    labelled = build_labels(
        hourly,
        vis_km_threshold=cfg["thresholds"]["visibility_km_lt"],
        wind_knots_threshold=cfg["thresholds"]["wind_knots_gt"],
        horizon_hours=cfg["thresholds"]["horizon_hours"],
    )
    return labelled
