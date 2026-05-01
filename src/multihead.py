"""Phase 5: multi-head LightGBM (vis head + wind head).

EDA showed vis-events and wind-events are nearly disjoint sets of hours, with
different driving features (humidity / dew depression / ceiling for vis;
pressure deltas / wind accel / wind spike for wind). A single-head model
trained on (vis OR wind) has to learn a blurry compromise; two specialist
heads each get to focus.

This module mirrors the structure of `baseline.py` end-to-end so results are
directly comparable:

    head A:  P(vis-event in next 2h)   trained on y_vis  = y_vis_future  < 3 km
    head B:  P(wind-event in next 2h)  trained on y_wind = y_wind_future > 25 kt

Combiner:  P(risky) = 1 - (1 - P_vis_cal) * (1 - P_wind_cal)
                    = P_vis + P_wind - P_vis*P_wind         (independence)

Independence is a reasonable working assumption because vis and wind events
are almost disjoint in the data (EDA Phase 2). Ther's also report the
single-head LGBM (Phase 4) on the same splits for a clean head-to-head.

Outputs:
    reports/multihead_results.json
    reports/multihead_vis.txt        (saved booster)
    reports/multihead_wind.txt       (saved booster)
    reports/multihead_reliability_*.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from .ingest import ROOT, load_config
from .features import feature_columns
from .evaluation import score_split, reliability_curve
from .baseline import (
    INLAND, COASTAL, time_split, lgbm_params, train_lgbm,
    fit_calibrator, _print_table,
)


# Per-head label derivation                                            
def derive_head_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add y_vis (binary) and y_wind (binary) from the regression targets.

    NaN propagates: if y_vis_future is NaN, y_vis is NaN (not 0). Same for wind.
    This matters: missing future obs != safe future.
    """
    vis_thr = cfg["thresholds"]["visibility_km_lt"]
    wnd_thr = cfg["thresholds"]["wind_knots_gt"]

    out = df.copy()
    vis = out["y_vis_future"]
    wnd = out["y_wind_future"]
    # Use pandas nullable boolean to preserve NaN.
    out["y_vis"]  = pd.array(np.where(vis.notna(), vis  < vis_thr, pd.NA), dtype="boolean")
    out["y_wind"] = pd.array(np.where(wnd.notna(), wnd  > wnd_thr, pd.NA), dtype="boolean")
    return out


# Train one head (LGBM + isotonic on val)                              
def train_head(tr_split, val_split, feats, label_col):
    """Train one binary head, return (model, calibrator, val_with_scores)."""
    keep_tr = tr_split[label_col].notna()
    keep_val = val_split[label_col].notna()
    X_tr = tr_split.loc[keep_tr, feats]
    y_tr = tr_split.loc[keep_tr, label_col].astype(int)
    X_val = val_split.loc[keep_val, feats]
    y_val = val_split.loc[keep_val, label_col].astype(int)

    print(f"[head:{label_col}] train={len(X_tr):,} val={len(X_val):,} "
          f"prev_tr={y_tr.mean():.3%} prev_val={y_val.mean():.3%}")

    model = train_lgbm(X_tr, y_tr, X_val, y_val)
    s_val = model.predict(X_val)
    iso = fit_calibrator(y_val.values, s_val)

    val_out = val_split.loc[keep_val].copy()
    val_out[f"{label_col}_score_raw"] = s_val
    val_out[f"{label_col}_score_cal"] = iso.predict(s_val)
    return model, iso, val_out


def combine(p_vis: np.ndarray, p_wind: np.ndarray) -> np.ndarray:
    """Independence-assumption combiner for OR'd risk."""
    p_vis = np.clip(p_vis, 0.0, 1.0)
    p_wind = np.clip(p_wind, 0.0, 1.0)
    return 1.0 - (1.0 - p_vis) * (1.0 - p_wind)


# Standard run (train on 3 inland + held-out time slice for val,     
# eval on KATL + KBOS test airports)                          
def run_multihead_standard(train, test, feats):
    tr = train[train["y_risk"].notna()].copy()
    te = test[test["y_risk"].notna()].copy()
    tr_split, val_split = time_split(tr, 0.2)

    vis_model,  vis_iso,  _ = train_head(tr_split, val_split, feats, "y_vis")
    wind_model, wind_iso, _ = train_head(tr_split, val_split, feats, "y_wind")

    #  val scores (combined risk) 
    val_split = val_split.copy()
    val_split["p_vis_raw"]  = vis_model.predict(val_split[feats])
    val_split["p_wind_raw"] = wind_model.predict(val_split[feats])
    val_split["p_vis_cal"]  = vis_iso.predict(val_split["p_vis_raw"].values)
    val_split["p_wind_cal"] = wind_iso.predict(val_split["p_wind_raw"].values)
    val_split["score_raw"] = combine(val_split["p_vis_raw"].values, val_split["p_wind_raw"].values)
    val_split["score_cal"] = combine(val_split["p_vis_cal"].values, val_split["p_wind_cal"].values)

    #test score
    te = te.copy()
    te["p_vis_raw"]  = vis_model.predict(te[feats])
    te["p_wind_raw"] = wind_model.predict(te[feats])
    te["p_vis_cal"]  = vis_iso.predict(te["p_vis_raw"].values)
    te["p_wind_cal"] = wind_iso.predict(te["p_wind_raw"].values)
    te["score_raw"] = combine(te["p_vis_raw"].values, te["p_wind_raw"].values)
    te["score_cal"] = combine(te["p_vis_cal"].values, te["p_wind_cal"].values)

    val_results, test_results = [], []
    # combined-risk metrics (compare directly to Phase 4 lgbm rows)
    for code, g in val_split.groupby("airport", sort=False):
        val_results.append(score_split(g["y_risk"], g["score_raw"], f"mh_raw/val/{code}"))
        val_results.append(score_split(g["y_risk"], g["score_cal"], f"mh_cal/val/{code}"))
    for code, g in te.groupby("airport", sort=False):
        test_results.append(score_split(g["y_risk"], g["score_raw"], f"mh_raw/test/{code}"))
        test_results.append(score_split(g["y_risk"], g["score_cal"], f"mh_cal/test/{code}"))

    # per-head diagnostic metrics (so we can see which head carries which airport)
    head_results = []
    for code, g in te.groupby("airport", sort=False):
        if g["y_vis"].notna().any():
            head_results.append(score_split(g["y_vis"], g["p_vis_cal"], f"mh_head_vis/test/{code}"))
        if g["y_wind"].notna().any():
            head_results.append(score_split(g["y_wind"], g["p_wind_cal"], f"mh_head_wind/test/{code}"))

    rel_raw = reliability_curve(val_split["y_risk"].astype(int).values,
                                val_split["score_raw"].values)
    rel_cal = reliability_curve(val_split["y_risk"].astype(int).values,
                                val_split["score_cal"].values)

    return {
        "val": val_results, "test": test_results, "heads": head_results,
        "vis_model": vis_model, "wind_model": wind_model,
        "vis_iso": vis_iso, "wind_iso": wind_iso,
        "reliability_raw": rel_raw, "reliability_cal": rel_cal,
    }


# LORO multi-head                                                      
def run_multihead_loro(train, test, feats):
    all_df = pd.concat([train, test], ignore_index=True)
    all_df = all_df[all_df["y_risk"].notna()].copy()

    runs = []
    for name, train_regime, eval_regime in (
        ("inland_to_coastal", INLAND, COASTAL),
        ("coastal_to_inland", COASTAL, INLAND),
    ):
        tr = all_df[all_df["airport"].isin(train_regime)].copy()
        ev = all_df[all_df["airport"].isin(eval_regime)].copy()
        tr_split, val_split = time_split(tr, 0.2)

        vis_model,  vis_iso,  _ = train_head(tr_split, val_split, feats, "y_vis")
        wind_model, wind_iso, _ = train_head(tr_split, val_split, feats, "y_wind")
        print(f"[loro/{name}] eval_rows={len(ev):,}")

        ev = ev.copy()
        ev["p_vis_raw"]  = vis_model.predict(ev[feats])
        ev["p_wind_raw"] = wind_model.predict(ev[feats])
        ev["p_vis_cal"]  = vis_iso.predict(ev["p_vis_raw"].values)
        ev["p_wind_cal"] = wind_iso.predict(ev["p_wind_raw"].values)
        ev["score_raw"] = combine(ev["p_vis_raw"].values, ev["p_wind_raw"].values)
        ev["score_cal"] = combine(ev["p_vis_cal"].values, ev["p_wind_cal"].values)

        for code, g in ev.groupby("airport", sort=False):
            runs.append(score_split(g["y_risk"], g["score_raw"], f"mh_loro_{name}_raw/{code}"))
            runs.append(score_split(g["y_risk"], g["score_cal"], f"mh_loro_{name}_cal/{code}"))
    return runs



# CLI                                                                  
def _main():
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / "train_features.parquet")
    test = pd.read_parquet(proc / "test_features.parquet")
    print(f"train={train.shape} test={test.shape}")

    feats = [c for c in feature_columns(train) if train[c].dtype.kind in "fciub"]

    train = derive_head_labels(train, cfg)
    test  = derive_head_labels(test,  cfg)
    print(f"features={len(feats)}  vis-prev={train['y_vis'].mean():.3%}  "
          f"wind-prev={train['y_wind'].mean():.3%}  "
          f"any-prev={train['y_risk'].mean():.3%}")

    mh = run_multihead_standard(train, test, feats)
    loro = run_multihead_loro(train, test, feats)

    _print_table("Multi-head on val (train airports, last 20%)", mh["val"])
    _print_table("Multi-head on TEST airports (KATL/KBOS, unseen)", mh["test"])
    _print_table("Per-head diagnostics on TEST airports", mh["heads"])
    _print_table("Multi-head LORO", loro)

    # compare against Phase 4 baseline if available 
    bl_path = ROOT / "reports" / "baseline_results.json"
    compare_rows = []
    if bl_path.exists():
        bl = json.loads(bl_path.read_text())
        bl_test = {r["label"]: r for r in bl["lgbm_test"] if "pr_auc" in r}
        mh_test = {r["label"]: r for r in mh["test"] if "pr_auc" in r}
        for variant in ("raw", "cal"):
            for code in sorted({lab.split("/")[-1] for lab in bl_test} |
                               {lab.split("/")[-1] for lab in mh_test}):
                bl_key = f"lgbm_{variant}/test/{code}"
                mh_key = f"mh_{variant}/test/{code}"
                if bl_key in bl_test and mh_key in mh_test:
                    bl_pr = bl_test[bl_key]["pr_auc"]
                    mh_pr = mh_test[mh_key]["pr_auc"]
                    compare_rows.append({
                        "airport": code, "variant": variant,
                        "single_head_pr_auc": bl_pr,
                        "multi_head_pr_auc": mh_pr,
                        "delta": mh_pr - bl_pr,
                    })
        if compare_rows:
            print("\n=== Single-head vs Multi-head (test PR-AUC) ===")
            print(f"  {'airport':6s}  {'variant':7s}  {'single':>7s}  {'multi':>7s}  {'delta':>7s}")
            for r in compare_rows:
                arrow = "+" if r["delta"] >= 0 else "-"
                print(f"  {r['airport']:6s}  {r['variant']:7s}  "
                      f"{r['single_head_pr_auc']:>7.3f}  "
                      f"{r['multi_head_pr_auc']:>7.3f}  "
                      f"{arrow}{abs(r['delta']):>6.3f}")

    # persist 
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    payload = {
        "n_features": len(feats),
        "vis_prevalence_train": float(train["y_vis"].astype("float").mean()),
        "wind_prevalence_train": float(train["y_wind"].astype("float").mean()),
        "any_prevalence_train": float(train["y_risk"].astype("float").mean()),
        "mh_val": mh["val"],
        "mh_test": mh["test"],
        "mh_heads_test": mh["heads"],
        "mh_loro": loro,
        "vs_baseline": compare_rows,
    }
    (reports / "multihead_results.json").write_text(json.dumps(payload, indent=2))
    mh["reliability_raw"].to_csv(reports / "multihead_reliability_raw.csv", index=False)
    mh["reliability_cal"].to_csv(reports / "multihead_reliability_cal.csv", index=False)
    mh["vis_model"].save_model(str(reports / "multihead_vis.txt"))
    mh["wind_model"].save_model(str(reports / "multihead_wind.txt"))

    # feature importance per head
    for name, model in (("vis", mh["vis_model"]), ("wind", mh["wind_model"])):
        imp = pd.DataFrame({
            "feature": feats,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        imp.to_csv(reports / f"multihead_{name}_feature_importance.csv", index=False)

    print(f"\nsaved -> {reports / 'multihead_results.json'}")


if __name__ == "__main__":
    _main()
