"""Phase 4: baselines with proper evaluation.

Separate tables for val vs test. Calibration (isotonic) on val. Leave-one-regime-out
(LORO) domain robustness: train inland -> test coastal, and vice versa.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from .ingest import ROOT, load_config
from .features import feature_columns
from .evaluation import score_split, reliability_curve


INLAND = {"KDEN", "KDFW", "KATL"}
COASTAL = {"KSFO", "KBOS"}


def risk_now(df, vis_km_lt=3.0, wind_knots_gt=25.0):
    vis_bad = (df["vis_km"] < vis_km_lt).astype("float")
    wnd_bad = (df["wind_knots"] > wind_knots_gt).astype("float")
    return np.maximum(vis_bad, wnd_bad)


def time_split(df: pd.DataFrame, val_frac: float = 0.2):
    parts_tr, parts_val = [], []
    for _, g in df.groupby("airport", sort=False):
        g = g.sort_values("timestamp")
        cut = int(len(g) * (1 - val_frac))
        parts_tr.append(g.iloc[:cut])
        parts_val.append(g.iloc[cut:])
    return pd.concat(parts_tr), pd.concat(parts_val)


def lgbm_params():
    return {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "is_unbalance": True,
    }


def train_lgbm(X_tr, y_tr, X_val, y_val):
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    return lgb.train(
        lgbm_params(), dtr, num_boost_round=1000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )


def fit_calibrator(y_val, s_val) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s_val, y_val)
    return iso


# Runs                                                                  #

def run_persistence(train, test):
    out = []
    for role, df in (("train", train), ("test", test)):
        df = df.copy()
        df["score"] = risk_now(df)
        for code, g in df.groupby("airport", sort=False):
            out.append(score_split(g["y_risk"], g["score"], f"persistence/{role}/{code}"))
    return out


def run_lgbm_standard(train, test, feats):
    """Train on all 3 train airports, val = last 20%, test = KATL+KBOS."""
    tr = train[train["y_risk"].notna()].copy()
    te = test[test["y_risk"].notna()].copy()
    tr_split, val_split = time_split(tr, 0.2)

    X_tr = tr_split[feats]; y_tr = tr_split["y_risk"].astype(int)
    X_val = val_split[feats]; y_val = val_split["y_risk"].astype(int)
    print(f"[lgbm] train={len(X_tr):,} val={len(X_val):,}")
    model = train_lgbm(X_tr, y_tr, X_val, y_val)

    # raw scores on val
    val_split = val_split.assign(score_raw=model.predict(X_val))

    # calibrate on val
    iso = fit_calibrator(val_split["y_risk"].astype(int).values, val_split["score_raw"].values)
    val_split["score_cal"] = iso.predict(val_split["score_raw"].values)

    # apply to test
    te = te.assign(score_raw=model.predict(te[feats]))
    te["score_cal"] = iso.predict(te["score_raw"].values)

    # metrics
    val_results, test_results = [], []
    for code, g in val_split.groupby("airport", sort=False):
        val_results.append(score_split(g["y_risk"], g["score_raw"], f"lgbm_raw/val/{code}"))
        val_results.append(score_split(g["y_risk"], g["score_cal"], f"lgbm_cal/val/{code}"))
    for code, g in te.groupby("airport", sort=False):
        test_results.append(score_split(g["y_risk"], g["score_raw"], f"lgbm_raw/test/{code}"))
        test_results.append(score_split(g["y_risk"], g["score_cal"], f"lgbm_cal/test/{code}"))

    # reliability curves for plotting (val pooled)
    rel_raw = reliability_curve(val_split["y_risk"].astype(int).values, val_split["score_raw"].values)
    rel_cal = reliability_curve(val_split["y_risk"].astype(int).values, val_split["score_cal"].values)

    return {
        "val": val_results, "test": test_results,
        "model": model, "calibrator": iso,
        "reliability_raw": rel_raw, "reliability_cal": rel_cal,
    }


def run_loro(train, test, feats):
    """Leave-One-Regime-Out domain robustness.

    Pool train+test rows (we use their features only, not peek at labels) then:
      A) train on inland only -> eval on coastal
      B) train on coastal only -> eval on inland
    Uses each regime's last 20% as val for early stopping.
    """
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

        X_tr = tr_split[feats]; y_tr = tr_split["y_risk"].astype(int)
        X_val = val_split[feats]; y_val = val_split["y_risk"].astype(int)
        print(f"[loro/{name}] train={len(X_tr):,} val={len(X_val):,} eval={len(ev):,}")
        model = train_lgbm(X_tr, y_tr, X_val, y_val)

        # calibrate on the in-regime val (honest: we never saw eval regime labels)
        s_val = model.predict(X_val)
        iso = fit_calibrator(y_val.values, s_val)

        # eval
        ev = ev.assign(score_raw=model.predict(ev[feats]))
        ev["score_cal"] = iso.predict(ev["score_raw"].values)
        for code, g in ev.groupby("airport", sort=False):
            runs.append(score_split(g["y_risk"], g["score_raw"], f"loro_{name}_raw/{code}"))
            runs.append(score_split(g["y_risk"], g["score_cal"], f"loro_{name}_cal/{code}"))
    return runs


# CLI                                                                   #

def _print_table(title: str, rows: list[dict]):
    print(f"\n=== {title} ===")
    print(f"  {'label':40s}  {'n':>7}  {'prev':>6}  {'PR-AUC':>7}  {'ROC':>6}  {'R@FPR10':>8}  {'P@Rec90':>8}  {'Brier':>6}  {'ECE':>6}")
    for r in rows:
        if "pr_auc" not in r:
            print(f"  {r['label']:40s}  (degenerate)")
            continue
        print(f"  {r['label']:40s}  {r['n']:>7,}  {r['prevalence']:>6.2%}  "
              f"{r['pr_auc']:>7.3f}  {r['roc_auc']:>6.3f}  "
              f"{r['recall_at_fpr10']:>8.3f}  {r['precision_at_recall90']:>8.3f}  "
              f"{r['brier']:>6.3f}  {r['ece']:>6.3f}")


def _main():
    parser = argparse.ArgumentParser(description="Run baseline/LORO evaluation for a feature variant")
    parser.add_argument(
        "--variant",
        choices=["phase4_clean", "phase5_physics"],
        default="phase5_physics",
        help="feature variant to load and report",
    )
    args = parser.parse_args()

    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / f"train_features_{args.variant}.parquet")
    test = pd.read_parquet(proc / f"test_features_{args.variant}.parquet")
    print(f"train={train.shape} test={test.shape}")

    feats = [c for c in feature_columns(train) if train[c].dtype.kind in "fciub"]

    pers = run_persistence(train, test)
    pers_tr = [r for r in pers if "/train/" in r["label"]]
    pers_te = [r for r in pers if "/test/" in r["label"]]

    lgb_out = run_lgbm_standard(train, test, feats)
    loro_out = run_loro(train, test, feats)

    _print_table("persistence on train airports (for reference)", pers_tr)
    _print_table("persistence on TEST airports", pers_te)
    _print_table("LightGBM on val (train airports, last 20%)", lgb_out["val"])
    _print_table("LightGBM on TEST airports (KATL/KBOS, unseen)", lgb_out["test"])
    _print_table("LORO: train inland -> eval coastal / train coastal -> eval inland", loro_out)

    # Save everything
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    payload = {
        "variant": args.variant,
        "n_features": len(feats),
        "persistence_train": pers_tr,
        "persistence_test": pers_te,
        "lgbm_val": lgb_out["val"],
        "lgbm_test": lgb_out["test"],
        "loro": loro_out,
    }
    (reports / f"baseline_results_{args.variant}.json").write_text(json.dumps(payload, indent=2))
    lgb_out["reliability_raw"].to_csv(reports / f"reliability_raw_{args.variant}.csv", index=False)
    lgb_out["reliability_cal"].to_csv(reports / f"reliability_cal_{args.variant}.csv", index=False)
    lgb_out["model"].save_model(str(reports / f"baseline_lgbm_{args.variant}.txt"))

    imp = pd.DataFrame({
        "feature": feats,
        "gain": lgb_out["model"].feature_importance(importance_type="gain"),
        "split": lgb_out["model"].feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(reports / f"baseline_feature_importance_{args.variant}.csv", index=False)

    print(f"\nsaved -> {reports / f'baseline_results_{args.variant}.json'}")


if __name__ == "__main__":
    _main()
