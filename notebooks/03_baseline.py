# %% [markdown]
# # 03 — Baselines 
# Persistence vs LightGBM (raw + calibrated) + LORO domain robustness.

# %%
from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

FIG = ROOT / "reports" / "figures"
res = json.loads((ROOT / "reports" / "baseline_results.json").read_text())


def rows(key):
    out = []
    for r in res[key]:
        if "pr_auc" not in r:
            continue
        parts = r["label"].split("/")
        out.append({"label": r["label"], "model": parts[0],
                    "airport": parts[-1],
                    **{k: r[k] for k in ("pr_auc", "roc_auc", "recall_at_fpr10",
                                         "precision_at_recall90", "brier", "ece",
                                         "prevalence", "n")}})
    return pd.DataFrame(out)

# %% [markdown]
# ## 1. TEST-airport results only (what actually matters)
# Ordered as: persistence -> lgbm raw -> lgbm calibrated.

# %%
test_df = pd.concat([rows("persistence_test"), rows("lgbm_test")], ignore_index=True)
print(test_df.to_string(index=False))

# %% [markdown]
# ## 2. Val (sanity check — train airports, last 20%)

# %%
val_df = pd.concat([rows("persistence_train"), rows("lgbm_val")], ignore_index=True)
print(val_df.to_string(index=False))

# %% [markdown]
# ## 3. PR-AUC head-to-head (test set)

# %%
piv = (rows("lgbm_test")
       .assign(short=lambda d: d["model"].str.replace("lgbm_", ""))
       .pivot_table(index="airport", columns="short", values="pr_auc"))
piv["persistence"] = rows("persistence_test").set_index("airport")["pr_auc"]
piv = piv[["persistence", "raw", "cal"]]
fig, ax = plt.subplots(figsize=(8, 5))
piv.plot.bar(ax=ax, color=["#888", "#2a7", "#27a"])
ax.axhline(0.40, ls="--", color="red", lw=0.8, label="EDA target 0.40")
ax.set_ylabel("PR-AUC"); ax.set_title("Test-set PR-AUC"); ax.legend()
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(FIG / "14_pr_auc_test.png", dpi=120); plt.show()

# %% [markdown]
# ## 4. Reliability diagrams (calibration check)
# 

# %%
rel_raw = pd.read_csv(ROOT / "reports" / "reliability_raw.csv")
rel_cal = pd.read_csv(ROOT / "reports" / "reliability_cal.csv")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
ax.plot(rel_raw["conf"], rel_raw["acc"], "o-", color="#c22", label="raw (uncalibrated)")
ax.plot(rel_cal["conf"], rel_cal["acc"], "o-", color="#2a7", label="isotonic-calibrated")
ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
ax.set_title("Reliability diagram (pooled val)"); ax.legend()
plt.tight_layout()
plt.savefig(FIG / "16_reliability.png", dpi=120); plt.show()

# %% [markdown]
# ## 5. LORO domain robustness
# coastal→inland and inland→coastal. PR-AUC shouldn't collapse.

# %%
loro = rows("loro")
loro["direction"] = loro["model"].str.extract(r"loro_([a-z_]+?)_(?:raw|cal)")
loro["variant"] = loro["model"].str.extract(r"(raw|cal)")
loro_cal = loro[loro["variant"] == "cal"]
piv = loro_cal.pivot_table(index="airport", columns="direction", values="pr_auc")
print(piv.round(3))

fig, ax = plt.subplots(figsize=(8, 5))
piv.plot.bar(ax=ax, color=["#c88", "#8c8"])
ax.axhline(0.40, ls="--", color="red", lw=0.8)
ax.set_ylabel("PR-AUC (calibrated)")
ax.set_title("LORO: train regime -> eval regime")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(FIG / "17_loro.png", dpi=120); plt.show()

# %% [markdown]
# ## 6. Feature importance (LGBM gain)

# %%
imp = pd.read_csv(ROOT / "reports" / "baseline_feature_importance.csv").head(20)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(imp["feature"][::-1], imp["gain"][::-1], color="steelblue")
ax.set_xlabel("total gain"); ax.set_title("Top 20 LightGBM features")
plt.tight_layout()
plt.savefig(FIG / "15_feature_importance.png", dpi=120); plt.show()

# %%
