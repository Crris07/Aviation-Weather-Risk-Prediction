# %% [markdown]
# # 02 — Feature Engineering
#
# Aviation Weather Risk Prediction.
#
# Phase 3. EDA (01) told us:
# - Vis/wind autocorrelation decays over ~6h → lag & rolling features
# - KSFO wind has 24h periodicity → cyclic hour features
# - Seasonality (KDEN Feb peak, KSFO summer) → cyclic day-of-year features
# - Vis-only vs wind-only events ≈ disjoint → keep both signal paths alive
# - Coastal/inland are different regimes → geography features

# %%
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ingest import load_config
from src.features import feature_columns

pd.set_option("display.max_columns", 60)
sns.set_theme(style="whitegrid", context="notebook")

cfg = load_config()
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
proc = ROOT / cfg["paths"]["processed_dir"]
train = pd.read_parquet(proc / "train_features.parquet")
test = pd.read_parquet(proc / "test_features.parquet")
print(f"train: {train.shape}   test: {test.shape}")
print(f"n features: {len(feature_columns(train))}")

# %% [markdown]
# ## Sanity: prevalence preserved from EDA

# %%
print(f"train y_risk prevalence: {train['y_risk'].astype('float').mean():.3%}")
print("by airport:")
print(train.groupby("airport")["y_risk"].apply(lambda s: s.astype("float").mean()).round(4))

# %% [markdown]
# ## Which features correlate with y_risk?
#
# Point-biserial (Pearson on continuous vs binary) is a quick screen. Not a
# replacement for model feature importance — but useful to see if the EDA's
# hypotheses survived the feature build.

# %%
feats = feature_columns(train)
y = train["y_risk"].astype("float")

numeric_feats = [c for c in feats if train[c].dtype.kind in "fciu"]
corrs = (train[numeric_feats]
         .corrwith(y)
         .abs()
         .sort_values(ascending=False))
print("Top 25 features by |corr| with y_risk:")
print(corrs.head(25).round(3))

# %%
# Plot top correlations
top = corrs.head(20)
fig, ax = plt.subplots(figsize=(8, 7))
top[::-1].plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("|corr| with y_risk")
ax.set_title("Top 20 features by absolute correlation with y_risk\n(train, pooled across airports)")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_feature_corr_with_label.png", dpi=120)
plt.show()

# %% [markdown]
#
# Sanity check: plot P(y_risk=1) vs `doy_sin` — should show KDEN's February
# peak as a modulation.

# %%
for code in ["KDEN", "KDFW", "KSFO"]:
    sub = train[train.airport == code].copy()
    sub["month_bin"] = sub["timestamp"].dt.month
    by_month = sub.groupby("month_bin")["y_risk"].apply(lambda s: s.astype("float").mean())
    plt.plot(by_month.index, by_month.values, marker="o", label=code)
plt.xlabel("month")
plt.ylabel("P(y_risk=1)")
plt.title("Seasonal risk (should match EDA fig 02)")
plt.legend()
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(FIG_DIR / "12_seasonal_risk_postfeatures.png", dpi=120)
plt.show()

# %% [markdown]
# ## Wind dynamics features — do they fire where we'd expect?


# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train["wind_accel_1h"].dropna(), bins=80, color="steelblue")
axes[0].set_title("wind_accel_1h (knots/hour)")
axes[0].set_yscale("log")
axes[0].axvline(0, color="k", lw=0.5)

spike_rate = train.groupby("airport")["wind_spike"].apply(
    lambda s: s.astype("float").mean()
)
axes[1].bar(spike_rate.index, spike_rate.values, color="tomato")
axes[1].set_title("wind_spike fire rate by airport")
axes[1].set_ylabel("fraction of hours flagged")
plt.tight_layout()
plt.savefig(FIG_DIR / "13_wind_dynamics.png", dpi=120)
plt.show()

# %% [markdown]
# ## Next phase (03_baseline.py)
#
# Targets to beat:
# - **Persistence baseline:** 0.44 PR-AUC on KDEN, 0.40 average (from EDA).