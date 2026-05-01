# %% [markdown]
# # 01 — Exploratory Data Analysis
#
# Aviation Weather Risk Prediction.
#
# Goal of this notebook: understand the shape, quality, and quirks of the NOAA ISD
# data we pulled, and sanity-check the `y_risk` label before we commit to feature
# engineering.
#
# **Guiding questions:**
# 1. What does the data look like? (distributions, ranges, outliers)
# 2. Where are the holes? (missingness, time gaps, duplicates)
# 3. What's the label prevalence, and how does it vary by airport / season / hour?
# 4. Are coastal and inland airports genuinely different?
# 5. How predictable is vis/wind from its own recent history? (persistence baseline)

# %%
# Setup
from __future__ import annotations
import sys
from pathlib import Path

# Make src importable when running as a script from notebooks/
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ingest import load_config
from src.label import load_and_label

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid", context="notebook")
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

cfg = load_config()
print("config loaded, stations:",
      list(cfg["stations"]["train"]) + list(cfg["stations"]["test"]))

# %%
# Load raw + labelled train data
train = load_and_label("train", cfg)
print(f"train shape: {train.shape}")
print(train.head())
print(train.tail())

# %%
# Load test data so we can compare distributions (NOT used for decisions)
test = load_and_label("test", cfg)
print(f"test shape: {test.shape}")
print(f"Test SHape: {test.tail()}")

# %% [markdown]
# ## 1. Basic sanity: rows per airport, time coverage

# %%
summary = (train.groupby("airport")
                .agg(rows=("timestamp", "size"),
                     start=("timestamp", "min"),
                     end=("timestamp", "max"))
                .sort_index())
print("TRAIN:")
print(summary)

print("\nTEST:")
print(test.groupby("airport")
          .agg(rows=("timestamp", "size"),
               start=("timestamp", "min"),
               end=("timestamp", "max"))
          .sort_index())

# %% [markdown]
# ## 2. Duplicates & time-gap audit
#
# After resampling to an hourly grid there shouldn't be *exact* timestamp
# duplicates, but NaN-filled rows are expected where NOAA had no obs nearby.
# The real question is: how big are the gaps?

# %%
def gap_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in df.groupby("airport", sort=False):
        ts = g["timestamp"].sort_values()
        diffs_h = ts.diff().dt.total_seconds().div(3600).dropna()
        n_dup = ts.duplicated().sum()
        n_rows = len(g)
        missing_vis = g["vis_km"].isna().sum()
        rows.append({
            "airport": code,
            "rows": n_rows,
            "dup_ts": n_dup,
            "median_gap_h": diffs_h.median(),
            "max_gap_h": diffs_h.max(),
            "gap_>3h_count": (diffs_h > 3).sum(),
            "vis_nan": missing_vis,
            "vis_nan_pct": round(missing_vis / n_rows * 100, 2),
        })
    return pd.DataFrame(rows)

print("TRAIN gap audit:")
print(gap_audit(train))
print("\nTEST gap audit:")
print(gap_audit(test))

# %% [markdown]
# ## 3. Missingness heatmap per airport (train)

# %%
feat_cols = ["temp_c", "dew_c", "vis_km", "wind_knots", "wind_dir_deg",
             "slp_hpa", "ceiling_m", "precip_mm"]

miss = (train.groupby("airport")[feat_cols]
             .apply(lambda g: g.isna().mean())
             * 100)
print("Missing % per airport (train):")
print(miss.round(1))

fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(miss, annot=True, fmt=".1f", cmap="rocket_r", ax=ax,
            cbar_kws={"label": "% missing"})
ax.set_title("Missing data % — train")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_missing_train.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Target prevalence
#
# How often is `y_risk = 1`? Overall, per airport, per month, per hour.

# %%
prev_overall = train["y_risk"].mean()
print(f"Overall y_risk prevalence (train): {prev_overall:.2%}")

prev_by_airport = train.groupby("airport")["y_risk"].mean().sort_values(ascending=False)
print("\nPrevalence by airport:")
print(prev_by_airport.to_string())

# %%
# y_risk by month
train_m = train.copy()
train_m["month"] = train_m["timestamp"].dt.month
by_month = (train_m.groupby(["airport", "month"])["y_risk"].mean()
                   .unstack("airport"))

fig, ax = plt.subplots(figsize=(9, 4))
by_month.plot(marker="o", ax=ax)
ax.set_title("y_risk prevalence by month — train")
ax.set_ylabel("P(y_risk = 1)")
ax.set_xlabel("month")
ax.set_xticks(range(1, 13))
plt.tight_layout()
plt.savefig(FIG_DIR / "02_prevalence_by_month.png", dpi=120)
plt.show()

# %%
# y_risk by hour-of-day (diurnal)
train_h = train.copy()
train_h["hour"] = train_h["timestamp"].dt.hour
by_hour = (train_h.groupby(["airport", "hour"])["y_risk"].mean()
                   .unstack("airport"))

fig, ax = plt.subplots(figsize=(9, 4))
by_hour.plot(marker="o", ax=ax)
ax.set_title("y_risk prevalence by hour of day (UTC) — train")
ax.set_ylabel("P(y_risk = 1)")
ax.set_xlabel("hour (UTC)")
ax.set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(FIG_DIR / "03_prevalence_by_hour.png", dpi=120)
plt.show()

# %% [markdown]
# ### Why y_risk fires: vis vs wind breakdown
# If y_risk=1, was it the visibility rule, the wind rule, or both?

# %%
# Re-derive the two sub-rules on the future window so we can attribute
vis_thr = cfg["thresholds"]["visibility_km_lt"]
wnd_thr = cfg["thresholds"]["wind_knots_gt"]
H = cfg["thresholds"]["horizon_hours"]

def attribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in df.groupby("airport", sort=False):
        vis = g["vis_km"].reset_index(drop=True)
        wnd = g["wind_knots"].reset_index(drop=True)
        fv = pd.concat([vis.shift(-k) for k in range(1, H + 1)], axis=1)
        fw = pd.concat([wnd.shift(-k) for k in range(1, H + 1)], axis=1)
        uv = (fv < vis_thr).any(axis=1)
        uw = (fw > wnd_thr).any(axis=1)
        rows.append({
            "airport": code,
            "vis_only": ((uv & ~uw).sum()),
            "wind_only": ((uw & ~uv).sum()),
            "both": ((uv & uw).sum()),
            "total_unsafe": ((uv | uw).sum()),
        })
    return pd.DataFrame(rows).set_index("airport")

attr = attribution(train)
print("Attribution of y_risk triggers (train):")
print(attr)

ax = attr[["vis_only", "wind_only", "both"]].plot(kind="bar", stacked=True,
                                                   figsize=(7, 4))
ax.set_title("y_risk triggers — vis vs wind vs both")
ax.set_ylabel("# hours")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_risk_attribution.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Distributions of raw features — overall + by geography

# %%
# Attach geography from config
geo_map = {**{c: cfg["stations"]["train"][c]["geography"] for c in cfg["stations"]["train"]},
           **{c: cfg["stations"]["test"][c]["geography"]  for c in cfg["stations"]["test"]}}
train["geo"] = train["airport"].map(geo_map)
test["geo"]  = test["airport"].map(geo_map)

plot_feats = ["temp_c", "dew_c", "vis_km", "wind_knots",
              "slp_hpa", "ceiling_m", "precip_mm"]

fig, axes = plt.subplots(len(plot_feats), 1, figsize=(9, 2.2 * len(plot_feats)))
for ax, feat in zip(axes, plot_feats):
    for code, g in train.groupby("airport", sort=False):
        g[feat].dropna().plot.hist(bins=60, alpha=0.35, ax=ax, label=code, density=True)
    ax.set_title(feat)
    ax.legend(fontsize=8, ncol=5)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_feature_hists_by_airport.png", dpi=120)
plt.show()

# %%
# Visibility tail zoom — most aviation risk lives in the low-vis tail
fig, ax = plt.subplots(figsize=(9, 4))
for code, g in train.groupby("airport", sort=False):
    vals = g["vis_km"].dropna()
    vals = vals[vals < 10]
    vals.plot.hist(bins=60, alpha=0.45, ax=ax, label=code, density=True)
ax.set_title("Visibility distribution — zoom to <10 km (train)")
ax.set_xlabel("visibility (km)")
ax.axvline(vis_thr, color="red", linestyle="--", label=f"{vis_thr} km threshold")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "06_vis_tail.png", dpi=120)
plt.show()

# %%
# Wind tail zoom — high wind tail drives risk
fig, ax = plt.subplots(figsize=(9, 4))
for code, g in train.groupby("airport", sort=False):
    vals = g["wind_knots"].dropna()
    vals = vals[vals > 10]
    vals.plot.hist(bins=60, alpha=0.45, ax=ax, label=code, density=True)
ax.set_title("Wind distribution — zoom to >10 kt (train)")
ax.set_xlabel("wind (knots)")
ax.axvline(wnd_thr, color="red", linestyle="--", label=f"{wnd_thr} kt threshold")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "07_wind_tail.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Coastal vs inland comparison (the hypothesis check)

# %%
agg = (train.groupby("geo")[["vis_km", "wind_knots", "dew_c", "temp_c", "slp_hpa"]]
            .agg(["mean", "std", "median"]))
print(agg.round(2))

# Density plots, coastal vs inland
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, feat in zip(axes, ["vis_km", "wind_knots", "dew_c"]):
    for geo, g in train.groupby("geo"):
        vals = g[feat].dropna()
        ax.hist(vals, bins=60, alpha=0.45, label=geo, density=True)
    ax.set_title(feat)
    ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "08_coastal_vs_inland.png", dpi=120)
plt.show()

# %% [markdown]
# ## 7. Correlation matrix (per airport)

# %%
corr_feats = ["temp_c", "dew_c", "vis_km", "wind_knots", "wind_dir_deg",
              "slp_hpa", "ceiling_m", "precip_mm"]

for code in train["airport"].unique():
    sub = train.loc[train["airport"] == code, corr_feats]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(sub.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title(f"Feature correlation — {code}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"09_corr_{code}.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 8. Autocorrelation — how persistent are vis & wind?
#
# If vis at t+1 is nearly identical to vis at t, then "persistence" is a strong
# baseline and model needs to beat it by a meaningful margin.

# %%
from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for col_idx, code in enumerate(cfg["stations"]["train"]):
    sub = (train[train["airport"] == code]
           .sort_values("timestamp")
           .set_index("timestamp"))

    for row_idx, feat in enumerate(["vis_km", "wind_knots"]):
        ax = axes[row_idx, col_idx]
        series = sub[feat].dropna()
        # truncate to first 48 lags (hours) for readability
        lags = range(1, 49)
        acf_vals = [series.autocorr(lag=L) for L in lags]
        ax.plot(list(lags), acf_vals, marker="o", markersize=3)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{code} — {feat} ACF (1–48 h)")
        ax.set_xlabel("lag (hours)")
        ax.set_ylabel("autocorr")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_autocorrelation.png", dpi=120)
plt.show()

# %% [markdown]
# ## 9. Persistence baseline for y_risk
#
# "predict y_risk_future = 1 if it's unsafe RIGHT NOW".
# This is the floor.

# %%
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

def persistence_eval(df: pd.DataFrame) -> pd.DataFrame:
    """'Naive' model: predict future unsafe = current unsafe."""
    rows = []
    for code, g in df.groupby("airport", sort=False):
        g = g.dropna(subset=["y_risk"])
        current_unsafe = ((g["vis_km"] < vis_thr) | (g["wind_knots"] > wnd_thr)).astype(int)
        y = g["y_risk"].astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, current_unsafe,
                                                     average="binary",
                                                     zero_division=0)
        try:
            auc = roc_auc_score(y, current_unsafe)
            ap  = average_precision_score(y, current_unsafe)
        except Exception:
            auc, ap = np.nan, np.nan
        rows.append({"airport": code,
                     "prevalence": y.mean(),
                     "precision": p, "recall": r, "f1": f,
                     "roc_auc_persist": auc, "pr_auc_persist": ap})
    return pd.DataFrame(rows).set_index("airport")

print("Persistence-baseline performance (train):")
print(persistence_eval(train).round(3))
print("\nPersistence-baseline performance (test):")
print(persistence_eval(test).round(3))

# %% [markdown]
# ## 10. Findings summary (first pass)
#
# ### Coverage
# After dedup + resample, every airport has a clean regular hourly grid.
# - Train (KDEN/KDFW/KSFO): 52,585 hours each (2018-01-01 -> 2024-01-01).
# - Test  (KATL/KBOS):       14,501 hours each (2024-01-01 -> 2025-08-27).
# - Zero duplicate timestamps, zero gaps > 1 h on the hourly grid.
# - NOAA raw VIS nan after resample: < 0.1% train, ~0.05–0.1% test. Essentially complete.
#
# ### Missingness
# - Core fields (temp, dew, vis, wind speed): ~0% missing on hourly grid.
# - Wind direction: 7–11% missing (calm winds often have no direction coded).
# - SLP: 0.7–2.4% missing (train) — trivial.
# - Ceiling (CIG): 9–21% missing — worst at KDFW. Imputation needed
# - Precip (AA1): 16–30% missing — treat as "no precip observed" -> fill with 0 during feature eng.
#
# ### Target prevalence
# - Overall train prevalence: **3.56%** unsafe. Class imbalance is real.
# - Per airport: KDEN 5.4%, KDFW 2.6%, KSFO 2.7%. KDEN is ~2x the others.
# - Per test airport: KATL 3.6%, KBOS 5.2%.
#
# ### Attribution (what fires y_risk?)
# - KDEN: mostly visibility (fog/snow), with a small wind slice.
# - KDFW: mostly visibility (convective/storm fog) with a minority wind slice.
# - KSFO: almost equal split between vis (marine layer fog) and wind.
# - "Both at once" is rare everywhere — suggests the two rules capture distinct regimes.
#
# ### Seasonality / diurnality
# - Clear winter peak at KDEN (Feb ~12%) and KDFW (DJF high) — cold-weather fog.
# - KSFO has a spring/summer peak (marine-layer season) — opposite regime.
# - Diurnal: strong pre-dawn fog signal at all inland airports.
#   KSFO shows nighttime-UTC peak consistent with overnight local marine fog.
#
# ### Coastal vs inland
# - Coastal dew point mean is **much higher and less variable** (8.9 +/- 3.7 vs 4.95 +/- 10.9).
# - Coastal temp std is 4.1 vs inland 11.6 — coast is heavily moderated.
# - Visibility means are similar on average but wind distributions differ:
#   KSFO has a fatter middle tail (steady marine winds) while inland is more bursty.
#
# ### Persistence baseline (this is the FLOOR)
# Predicting "unsafe in next 2h = unsafe right now":
# | airport | prevalence | PR-AUC | ROC-AUC | recall |
# |---------|------------|--------|---------|--------|
# | KDEN    | 5.4%       | 0.44   | 0.78    | 0.56   |
# | KDFW    | 2.6%       | 0.27   | 0.71    | 0.42   |
# | KSFO    | 2.7%       | 0.30   | 0.72    | 0.45   |
# | KATL    | 3.6%       | 0.53   | 0.82    | 0.65   |
# | KBOS    | 5.2%       | 0.42   | 0.77    | 0.55   |
#
# 
#
# ### Autocorrelation
# - Vis and wind have strong 1-6h persistence everywhere.
# - KSFO wind shows pronounced **24h periodicity** 
#   to the model: hour-of-day + lag features should pick this up cleanly.
# - Inland airports decay faster — more random-walk-like, harder beyond 6h.
#