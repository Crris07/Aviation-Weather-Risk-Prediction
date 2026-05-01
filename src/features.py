"""
Feature engineering for Aviation Weather Risk project.

INVARIANTS (read before editing — breaking any of these silently poisons the model):

1.  Every time-dependent transform (lag, rolling, diff) is grouped by `airport`.
    Cross-station leakage is the #1 way to make a time-series model look great
    on dev and die in production. We never do ungrouped `.shift()` or `.rolling()`.

2.  Rolling windows are CAUSAL. We implement causality as
        grouped.shift(1).rolling(window).agg(...)
    i.e. the rolling window does NOT include the current timestep. Pandas'
    default is trailing but INCLUSIVE of current row, which is a subtle leak
    when `current` encodes what we're trying to predict.

3.  Cyclical quantities (hour-of-day, day-of-year, wind direction) are encoded
    as (sin, cos) pairs so 23h and 0h are adjacent, and 359° and 1° are adjacent.

4.  Labels (`y_risk`, `y_vis_future`, `y_wind_future`) are built in
    `src/label.py` using forward shifts on the already-hourly grid. We do NOT
    reinvent them here. Targets and features MUST be aligned on the same hourly
    index before splitting.

5.  Feature matrix rows where ANY lag-12 / roll-12 feature is NaN are the
    warm-up tail of each airport — they must be dropped, not imputed, or we
    teach the model to trust zero-padded history.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ingest import load_config, ROOT



# Configuration                                                         #

LAG_HOURS: tuple[int, ...] = (1, 2, 3, 6, 12)
ROLL_WINDOWS: tuple[int, ...] = (3, 6, 12)

# Fields we build lags / rollings on. Targets are NOT in here.
DYNAMIC_FIELDS: tuple[str, ...] = (
    "temp_c", "dew_c", "vis_km", "wind_knots",
    "slp_hpa", "ceiling_m", "precip_mm",
)

# NOAA visibility is reported up to 16,093 m (10 statute miles) and everything
# above that is a sentinel meaning "clear, >10 miles". We clip to 16 km so the
# upper tail stops being a noise generator.
VIS_CLIP_KM = 16.0



# Helpers                                                               #

def _clip_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions we locked in during EDA.

    - visibility clipped to 16 km (NOAA cap)
    - precipitation NaN -> 0 mm (NOAA omits precip when none observed)
    - ceiling missing -> flag column + impute with very high value
    """
    df = df.copy()

    # Visibility cap
    df["vis_km"] = df["vis_km"].clip(upper=VIS_CLIP_KM)

    # Precip: NOAA omits the AA1 block when there's no precip in the hour
    df["precip_mm"] = df["precip_mm"].fillna(0.0)

    # Ceiling: encode "unknown" explicitly, then impute with a large value
    # (effectively "unlimited") so the model can use ceiling_unknown as a flag
    df["ceiling_unknown"] = df["ceiling_m"].isna().astype("int8")
    df["ceiling_m"] = df["ceiling_m"].fillna(22000.0)  # higher than any real cloud

    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encodings for hour-of-day and day-of-year.

    Linear hour is *wrong*: the model would think hour 23 and hour 0 are 23
    units apart when they're 1. Same for Dec 31 / Jan 1.
    """
    ts = df["timestamp"]
    hour = ts.dt.hour.astype("float64")
    doy = ts.dt.dayofyear.astype("float64")
    month = ts.dt.month.astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = month  # kept for grouping / analysis, not as a model feature
    return df


def _add_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Wind direction is circular — encode as (sin, cos).

    Also: when wind speed is ~0, direction is meaningless. We zero both
    components in that case so the model doesn't learn spurious patterns
    from reported-but-meaningless direction on calm hours.
    """
    rad = np.deg2rad(df["wind_dir_deg"].astype("float64"))
    sin = np.sin(rad)
    cos = np.cos(rad)

    calm = (df["wind_knots"].fillna(0.0) < 1.0) | df["wind_dir_deg"].isna()
    sin = sin.where(~calm, 0.0)
    cos = cos.where(~calm, 0.0)

    df["wind_dir_sin"] = sin
    df["wind_dir_cos"] = cos
    return df


def _add_derived_atmospherics(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap physics: dew-point depression is a fog proxy."""
    df["dew_depression"] = df["temp_c"] - df["dew_c"]
    return df


def _add_grouped_lags(df: pd.DataFrame, fields: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    """Add `<field>_lag<k>` columns. Grouped by airport — NEVER ungrouped."""
    grouped = df.groupby("airport", sort=False, group_keys=False)
    for field in fields:
        if field not in df.columns:
            continue
        shifts = {f"{field}_lag{k}": grouped[field].shift(k) for k in lags}
        df = df.assign(**shifts)
    return df


def _add_grouped_rollings(
    df: pd.DataFrame,
    fields: Iterable[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    """Add causal rolling mean+std. Grouped by airport.

    Causality: we do `.shift(1).rolling(window)` so the window ends at t-1.
    That way the current row never peeks at itself.
    """
    grouped = df.groupby("airport", sort=False, group_keys=False)
    new_cols: dict[str, pd.Series] = {}
    for field in fields:
        if field not in df.columns:
            continue
        shifted = grouped[field].shift(1)
        # rolling must also be done per-group so windows don't span airports
        roll = shifted.groupby(df["airport"], sort=False, group_keys=False)
        for w in windows:
            new_cols[f"{field}_rollmean{w}"] = roll.rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
            new_cols[f"{field}_rollstd{w}"] = roll.rolling(w, min_periods=w).std().reset_index(level=0, drop=True)
    if new_cols:
        df = df.assign(**new_cols)
    return df


def _add_pressure_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Pressure change over 3h and 6h — the classic storm signal."""
    grouped = df.groupby("airport", sort=False, group_keys=False)["slp_hpa"]
    df["slp_delta_3h"] = df["slp_hpa"] - grouped.shift(3)
    df["slp_delta_6h"] = df["slp_hpa"] - grouped.shift(6)
    return df


def _add_wind_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Wind rate-of-change + volatility spike indicator.

    - `wind_accel_1h` : wind(t) - wind(t-1)      (grouped)
    - `wind_spike`   : 1 if wind(t) exceeds its own 12h causal mean by >2x
                       causal std; else 0. (grouped, causal.)
    """
    grouped = df.groupby("airport", sort=False, group_keys=False)["wind_knots"]
    df["wind_accel_1h"] = df["wind_knots"] - grouped.shift(1)

    # causal reference mean/std over last 12h (excluding current)
    shifted = grouped.shift(1)
    roll = shifted.groupby(df["airport"], sort=False, group_keys=False)
    ref_mean = roll.rolling(12, min_periods=6).mean().reset_index(level=0, drop=True)
    ref_std = roll.rolling(12, min_periods=6).std().reset_index(level=0, drop=True)

    spike = (df["wind_knots"] > (ref_mean + 2.0 * ref_std)).astype("int8")
    # Where ref stats are undefined (warm-up), spike is 0 rather than NA,
    # but only for rows where wind_knots is itself present.
    spike = spike.where(df["wind_knots"].notna(), other=pd.NA).astype("Int8")
    df["wind_spike"] = spike
    return df


def _add_physics_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-driven generalizable features.

    Universal (no per-airport hacks). They happen to fire at coastal
    airports because the physics fires there.

    1.  `dew_depression_min_3h`, `dew_depression_min_6h` -- causal rolling
        min of dew depression. Persistent low dew depression = saturated
        air = fog ingredient.
    2.  `calm_and_saturated` -- binary fog-formation indicator:
        wind_knots < 5 AND dew_depression < 2.
    3.  `dew_depression_x_hour_sin` -- diurnal interaction. Dawn / late
        night is when radiation fog forms.
    4.  `pressure_drop_x_wind` -- 3h pressure delta (inverted so 'drop' is
        positive) times wind speed. Frontal-passage / squall proxy.

    The `coastal` interactions are added AFTER geography is attached.
    """
    grouped = df.groupby("airport", sort=False, group_keys=False)["dew_depression"]
    shifted = grouped.shift(1)
    roll = shifted.groupby(df["airport"], sort=False, group_keys=False)
    df["dew_depression_min_3h"] = (
        roll.rolling(3, min_periods=3).min().reset_index(level=0, drop=True)
    )
    df["dew_depression_min_6h"] = (
        roll.rolling(6, min_periods=6).min().reset_index(level=0, drop=True)
    )

    # Calm-and-saturated: only well-defined when both inputs are present.
    calm = df["wind_knots"] < 5.0
    saturated = df["dew_depression"] < 2.0
    both_present = df["wind_knots"].notna() & df["dew_depression"].notna()
    cs = (calm & saturated).astype("int8")
    df["calm_and_saturated"] = cs.where(both_present, other=pd.NA).astype("Int8")

    # Diurnal fog interaction. hour_sin is bounded in [-1, 1].
    df["dew_depression_x_hour_sin"] = df["dew_depression"] * df["hour_sin"]

    # Pressure-drop * wind. slp_delta_3h is current minus 3h ago, so a
    # *drop* is negative. Negate so 'drop' is positive and the product
    # reads naturally as 'falling-pressure-with-wind'.
    df["pressure_drop_x_wind"] = (-df["slp_delta_3h"]) * df["wind_knots"]
    return df


def _add_coastal_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Geography-aware interactions, added AFTER `_attach_geography`.

    The `coastal` flag here is 0/1, so multiplying gates the feature on
    coastal airports only -- but the model still has the un-gated copy
    (`wind_dir_sin`, `wind_dir_cos`) so it can learn the inland role too.

    This lets the model discover an onshore-wind arc on coastal airports
    from the data, instead of us hand-coding the arc for KSFO vs KBOS.
    """
    if "coastal" not in df.columns:
        return df
    coastal = df["coastal"].astype("float64")
    df["wind_dir_sin_coastal"] = df["wind_dir_sin"] * coastal
    df["wind_dir_cos_coastal"] = df["wind_dir_cos"] * coastal
    df["calm_and_saturated_coastal"] = (
        df["calm_and_saturated"].astype("float") * coastal
    )
    return df


def _attach_geography(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add lat/lon/elev/coastal features from config.yaml."""
    rows = []
    for role in ("train", "test"):
        for code, meta in cfg["stations"][role].items():
            rows.append({
                "airport": code,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "elev_m": meta["elev_m"],
                "coastal": 1 if meta["geography"] == "coastal" else 0,
            })
    geo = pd.DataFrame(rows)
    return df.merge(geo, on="airport", how="left")



# Public entry point                                                   
def build_feature_matrix(
    labelled: pd.DataFrame,
    cfg: dict | None = None,
    drop_warmup: bool = True,
    include_physics: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix from a labelled hourly frame.

    Input: output of `src.label.load_and_label(role, cfg)` — rows are hourly,
    grouped by airport, and already contain y_risk / y_vis_future /
    y_wind_future.

    Output: same rows, plus all engineered features. If `drop_warmup` is
    True (default), rows where any lag-12 / roll-12 feature is NaN are
    dropped — those are the per-airport warm-up periods.
    """
    if cfg is None:
        cfg = load_config()

    df = labelled.sort_values(["airport", "timestamp"]).reset_index(drop=True)

    df = _clip_and_impute(df)
    df = _add_calendar_features(df)
    df = _add_wind_direction(df)
    df = _add_derived_atmospherics(df)

    # Include dew_depression in lags/rollings as well
    lag_fields = list(DYNAMIC_FIELDS) + ["dew_depression"]
    df = _add_grouped_lags(df, lag_fields, LAG_HOURS)
    df = _add_grouped_rollings(df, lag_fields, ROLL_WINDOWS)

    df = _add_pressure_deltas(df)
    df = _add_wind_dynamics(df)
    df = _attach_geography(df, cfg)
    if include_physics:
        df = _add_physics_interactions(df)
        df = _add_coastal_interactions(df)

    if drop_warmup:
    
        warmup_n = max(max(LAG_HOURS), max(ROLL_WINDOWS))
        df = df.sort_values(["airport", "timestamp"]).reset_index(drop=True)
        df["_row_in_group"] = df.groupby("airport", sort=False).cumcount()
        df = df[df["_row_in_group"] >= warmup_n].drop(columns="_row_in_group").reset_index(drop=True)

    # Defragment: many .assign() calls above leave the block manager fragmented.
    # .copy() consolidates blocks and silences PerformanceWarning downstream.
    return df.copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns that should be fed to a model."""
    never_feature = {
        "timestamp", "airport", "usaf", "wban", "month",
        "y_risk", "y_vis_future", "y_wind_future",
        # wind_dir_deg is kept as raw for debugging but shouldn't be a feature —
        # we use wind_dir_sin / wind_dir_cos instead.
        "wind_dir_deg",
    }
    return [c for c in df.columns if c not in never_feature]

# CLI                                         
def _main() -> None:
    """Build train + test feature matrices and persist to data/processed/."""
    from .label import load_and_label

    parser = argparse.ArgumentParser(description="Build processed feature matrices")
    parser.add_argument(
        "--variant",
        choices=["phase4_clean", "phase5_physics"],
        default="phase5_physics",
        help="phase4_clean excludes Phase 5 physics interactions; phase5_physics includes them",
    )
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    include_physics = args.variant == "phase5_physics"

    for role in ("train", "test"):
        print(f"[features] building {role} ({args.variant}) ...")
        labelled = load_and_label(role, cfg)
        feats = build_feature_matrix(labelled, cfg, include_physics=include_physics)
        out = processed_dir / f"{role}_features_{args.variant}.parquet"
        feats.to_parquet(out, index=False)
        print(f"[features]   rows={len(feats):,}  cols={feats.shape[1]}  -> {out}")


if __name__ == "__main__":
    _main()
