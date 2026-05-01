
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from .ingest import ROOT, load_config

parser = argparse.ArgumentParser(description="Validate a processed feature variant")
parser.add_argument(
    "--variant",
    choices=["phase4_clean", "phase5_physics"],
    default="phase5_physics",
    help="feature variant to validate",
)
args = parser.parse_args()

cfg = load_config()
proc = ROOT / cfg["paths"]["processed_dir"]
train = pd.read_parquet(proc / f"train_features_{args.variant}.parquet")
test = pd.read_parquet(proc / f"test_features_{args.variant}.parquet")

print(f"variant: {args.variant}")
print(f"train: {train.shape}, airports: {sorted(train['airport'].unique())}")
print(f"test:  {test.shape}, airports: {sorted(test['airport'].unique())}")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not condition:
        failures.append(name)


print("\n== 1. Causality of lag features ==")
# For a row at timestamp t, wind_knots_lag1 should equal the wind_knots
# of the *same airport's* row at t-1h in the full labelled hourly grid.
from src.label import load_and_label
full = load_and_label("train", cfg)
full = full.sort_values(["airport", "timestamp"]).reset_index(drop=True)
for code, g in train.groupby("airport", sort=False):
    g = g[["timestamp", "wind_knots_lag1"]].copy()
    f = full[full["airport"] == code][["timestamp", "wind_knots"]].copy()
    # Expected lag1 at t = wind_knots at t-1h from the full grid
    f["t_plus_1"] = f["timestamp"] + pd.Timedelta(hours=1)
    joined = g.merge(f[["t_plus_1", "wind_knots"]],
                     left_on="timestamp", right_on="t_plus_1", how="left")
    a = joined["wind_knots_lag1"]
    e = joined["wind_knots"]
    mism = ((a != e) & ~(a.isna() & e.isna())).sum()
    check(f"wind_knots_lag1 causal @ {code}", mism == 0, f"{mism} mismatches")


print("\n== 2. Causality of rolling means (must NOT include current row) ==")
# For a row at timestamp t, wind_knots_rollmean3 should equal the mean of
# wind_knots at t-1h, t-2h, t-3h from the full grid (excluding t itself).
for code, g in train.groupby("airport", sort=False):
    gsub = g[["timestamp", "wind_knots_rollmean3"]].copy()
    f = full[full["airport"] == code][["timestamp", "wind_knots"]].set_index("timestamp")
    # sample 500 rows 
    sample = gsub.sample(min(500, len(gsub)), random_state=0)
    bad = 0
    checked = 0
    for _, row in sample.iterrows():
        t = row["timestamp"]
        try:
            past = f.loc[[t - pd.Timedelta(hours=k) for k in (1, 2, 3)], "wind_knots"]
        except KeyError:
            continue
        checked += 1
    
        expected = np.mean(past.values)
        got = row["wind_knots_rollmean3"]
        if pd.isna(expected) and pd.isna(got):
            continue
        if pd.isna(expected) or pd.isna(got):
            bad += 1
        elif abs(expected - got) < 1e-9:
            continue
        else:
            bad += 1
    check(f"wind_knots_rollmean3 causal @ {code}", bad == 0, f"{bad}/{checked} bad")


print("\n== 3. No cross-airport leakage in lags ==")
for code, g in train.groupby("airport", sort=False):
    g = g.sort_values("timestamp").reset_index(drop=True)
    n_nan = g["wind_knots_lag12"].isna().sum()
    check(f"no warmup leakage @ {code}", n_nan <= g["wind_knots"].isna().sum() + 20,
          f"{n_nan} NaN in lag12 (raw NaN: {g['wind_knots'].isna().sum()})")


print("\n== 4. Cyclic encodings ==")
# sin^2 + cos^2 should be ~1
for col_pair in [("hour_sin", "hour_cos"), ("doy_sin", "doy_cos")]:
    s, c = col_pair
    mag = (train[s] ** 2 + train[c] ** 2)
    check(f"{s}/{c} unit circle", np.allclose(mag, 1.0, atol=1e-9), f"min={mag.min():.4f} max={mag.max():.4f}")

mag = train["wind_dir_sin"] ** 2 + train["wind_dir_cos"] ** 2
valid = np.isclose(mag, 0.0, atol=1e-9) | np.isclose(mag, 1.0, atol=1e-9)
check("wind_dir_(sin,cos) is 0 or unit", bool(valid.all()),
      f"{(~valid).sum()} rows violate")


print("\n== 5. Imputation ==")
check("precip_mm has no NaN", train["precip_mm"].isna().sum() == 0)
check("ceiling_m has no NaN", train["ceiling_m"].isna().sum() == 0)
check("ceiling_unknown is 0/1", set(train["ceiling_unknown"].unique()).issubset({0, 1}))
check("vis_km <= 16", train["vis_km"].max() <= 16.0 + 1e-9)


print("\n== 6. Geography attached ==")
for code, g in train.groupby("airport", sort=False):
    check(f"coastal flag present @ {code}", g["coastal"].notna().all())


print("\n== 6b. Physics interaction features ==")
has_physics = "dew_depression_min_3h" in train.columns
if not has_physics:
    print("  [INFO] physics features not present in this variant; skipping Phase 5-specific checks")
else:
    for code, g in train.groupby("airport", sort=False):
        g = g.sort_values("timestamp")
        # Pull dew_depression series indexed by timestamp for this airport
        dd = g.set_index("timestamp")["dew_depression"]
        expected = dd.shift(1).rolling(3, min_periods=3).min()
        actual = g.set_index("timestamp")["dew_depression_min_3h"]
        aligned = pd.concat([expected.rename("exp"), actual.rename("act")], axis=1).dropna()
        if len(aligned) == 0:
            check(f"dew_depression_min_3h has overlap with expected @ {code}", False)
            continue
        diff = (aligned["exp"] - aligned["act"]).abs().max()
        check(f"dew_depression_min_3h causal & correct @ {code}",
              diff < 1e-9, f"max abs diff {diff:.2e}")

    cs_def = ((train["wind_knots"] < 5.0) & (train["dew_depression"] < 2.0)).astype("int8")
    cs_def = cs_def.where(train["wind_knots"].notna() & train["dew_depression"].notna(),
                          other=pd.NA).astype("Int8")
    mask = train["calm_and_saturated"].notna() & cs_def.notna()
    diff = (train.loc[mask, "calm_and_saturated"].astype(int) -
            cs_def.loc[mask].astype(int)).abs().max()
    check("calm_and_saturated matches definition", int(diff) == 0,
          f"diff={int(diff)}")

    # coastal interactions are zero on inland airports
    for code, g in train.groupby("airport", sort=False):
        if g["coastal"].iloc[0] == 0:
            for col in ("wind_dir_sin_coastal", "wind_dir_cos_coastal",
                        "calm_and_saturated_coastal"):
                if col not in g.columns:
                    continue
                non_zero = g[col].fillna(0).abs().gt(1e-12).any()
                check(f"{col} is zero on inland @ {code}", not non_zero)

    # pressure_drop_x_wind: sign-correct (positive when pressure is falling and wind blowing)
    near_zero_drop = train["slp_delta_3h"].abs() < 0.05
    sub = train.loc[near_zero_drop, "pressure_drop_x_wind"]
    check("pressure_drop_x_wind ~0 when pressure steady",
          sub.abs().median() < 1.0, f"median {sub.abs().median():.3f}")


print("\n== 7. Labels still aligned ==")
check("y_risk exists", "y_risk" in train.columns)
prev = train["y_risk"].astype("float").mean()
check("y_risk prevalence plausible (1%..10%)", 0.01 < prev < 0.10, f"{prev:.3%}")


print("\n== 7b. Redundancy check (correlation > 0.999 between distinct feature columns) ==")

from .features import feature_columns as _fc
feat_list = _fc(train)
num_feats = [c for c in feat_list if train[c].dtype.kind in "fciub" and c != "month"]
sample = train[num_feats].sample(n=min(50_000, len(train)), random_state=0)
corr = sample.corr().abs()
import numpy as _np
upper = corr.where(_np.triu(_np.ones(corr.shape, dtype=bool), k=1))
stacked = upper.stack().dropna()
# Threshold tuning: 0.9999 catches the units-bug class (vis_m vs vis_km is
# 0.99996; wind_mps vs wind_knots is 1.00000). Slow-varying atmospheric
# autocorrelations (e.g. slp_hpa_lag2 vs slp_hpa_rollmean3 = 0.9995) are
# legitimate -- pressure genuinely is that smooth at hourly cadence -- so
# the threshold has to sit between them.
REDUNDANCY_THRESHOLD = 0.9999
redundant = stacked[stacked > REDUNDANCY_THRESHOLD]
if len(redundant) > 0:
    detail = "; ".join(f"{a}~{b}={v:.5f}" for (a, b), v in redundant.head(10).items())
    check(f"no redundant feature pairs (|corr|>{REDUNDANCY_THRESHOLD})", False, detail)
else:
    check(f"no redundant feature pairs (|corr|>{REDUNDANCY_THRESHOLD})", True,
          f"checked {len(num_feats)} features, max abs corr {stacked.max():.5f}")


print("\n== 8. Feature column list ==")
from .features import feature_columns
feats = feature_columns(train)
print(f"  {len(feats)} feature columns")
# Targets must be excluded
for banned in ("y_risk", "y_vis_future", "y_wind_future", "airport", "timestamp", "wind_dir_deg"):
    check(f"'{banned}' excluded from feature list", banned not in feats)


print("\n" + ("=" * 50))
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
