"""Phase 5a: per-airport isotonic calibration on the Phase 4 single-head LGBM.

Phase 4 fits ONE isotonic regressor on pooled val (all 3 train airports).
Different airports have different prevalences (KDEN 5.4% vs KDFW 2.6% vs KSFO
2.7%) and different score distributions, so a pooled calibrator is a
compromise. This module fits one isotonic per airport on its own val slice
and applies it at test/LORO time.

Test airports (KATL, KBOS) have no train data, so they have no own-airport
calibrator. We pick one from the *most similar* train airport by regime:

    KATL (inland)  -> mean isotonic of {KDEN, KDFW, KATL-equivalent inland set}
                      = nearest-prevalence inland airport (KDFW, prev 2.6%)
    KBOS (coastal) -> nearest-prevalence coastal airport (KSFO, prev 2.7%)

We also report two reference settings on the same splits:

    pooled        - the original Phase 4 single isotonic on all train airports
    per_airport   - new: fit per-airport on train airports, transfer by regime

Comparison is PR-AUC, ECE, Brier on test set against Phase 4 numbers.
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
    INLAND, COASTAL, time_split, train_lgbm, fit_calibrator,
    _print_table,
)


# Regime-nearest train airport for each test airport.
# Picked by airport regime + closest prevalence.
TRANSFER_MAP = {
    "KATL": "KDFW",   # inland-inland, 3.6% -> 2.6%
    "KBOS": "KSFO",   # coastal-coastal, 5.2% -> 2.7%
}


def fit_per_airport_calibrators(val_df: pd.DataFrame) -> dict[str, IsotonicRegression]:
    """One isotonic per airport, fit on that airport's val slice."""
    out = {}
    for code, g in val_df.groupby("airport", sort=False):
        y = g["y_risk"].astype(int).values
        s = g["score_raw"].values
        if y.sum() == 0 or y.sum() == len(y):
            print(f"[per-airport-cal/{code}] degenerate, skipping")
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s, y)
        out[code] = iso
        print(f"[per-airport-cal/{code}] n={len(y):,} pos={y.sum()} prev={y.mean():.3%}")
    return out


def apply_per_airport(df: pd.DataFrame, calibrators: dict[str, IsotonicRegression],
                      transfer_map: dict[str, str]) -> pd.Series:
    """Apply per-airport calibrator. Test airports use their transfer mapping."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for code, g in df.groupby("airport", sort=False):
        target = code if code in calibrators else transfer_map.get(code)
        if target is None or target not in calibrators:
            print(f"[apply/{code}] no calibrator and no transfer rule -> raw")
            out.loc[g.index] = g["score_raw"].values
            continue
        iso = calibrators[target]
        out.loc[g.index] = iso.predict(g["score_raw"].values)
        if code != target:
            print(f"[apply/{code}] using calibrator from {target}")
    return out


# Standard run                                                          #

def run_standard(train, test, feats):
    tr = train[train["y_risk"].notna()].copy()
    te = test[test["y_risk"].notna()].copy()
    tr_split, val_split = time_split(tr, 0.2)

    X_tr = tr_split[feats]; y_tr = tr_split["y_risk"].astype(int)
    X_val = val_split[feats]; y_val = val_split["y_risk"].astype(int)
    print(f"[lgbm] train={len(X_tr):,} val={len(X_val):,}")
    model = train_lgbm(X_tr, y_tr, X_val, y_val)

    val_split = val_split.copy()
    val_split["score_raw"] = model.predict(X_val)
    te = te.copy()
    te["score_raw"] = model.predict(te[feats])

    # pooled isotonic (Phase 4 reference) 
    iso_pooled = fit_calibrator(y_val.values, val_split["score_raw"].values)
    val_split["score_pooled"] = iso_pooled.predict(val_split["score_raw"].values)
    te["score_pooled"] = iso_pooled.predict(te["score_raw"].values)

    #per-airport
    per_airport = fit_per_airport_calibrators(val_split)
    val_split["score_per_airport"] = apply_per_airport(val_split, per_airport, TRANSFER_MAP)
    te["score_per_airport"] = apply_per_airport(te, per_airport, TRANSFER_MAP)

    val_results, test_results = [], []
    for code, g in val_split.groupby("airport", sort=False):
        val_results.append(score_split(g["y_risk"], g["score_raw"],         f"raw/val/{code}"))
        val_results.append(score_split(g["y_risk"], g["score_pooled"],      f"pooled/val/{code}"))
        val_results.append(score_split(g["y_risk"], g["score_per_airport"], f"per_airport/val/{code}"))
    for code, g in te.groupby("airport", sort=False):
        test_results.append(score_split(g["y_risk"], g["score_raw"],         f"raw/test/{code}"))
        test_results.append(score_split(g["y_risk"], g["score_pooled"],      f"pooled/test/{code}"))
        test_results.append(score_split(g["y_risk"], g["score_per_airport"], f"per_airport/test/{code}"))

    return {
        "val": val_results, "test": test_results,
        "model": model,
        "iso_pooled": iso_pooled,
        "iso_per_airport": per_airport,
    }



# LORO                                                                  #

def run_loro(train, test, feats):
    """Leave-One-Regime-Out with per-airport calibration applied to eval-side."""
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

        val_split = val_split.copy()
        val_split["score_raw"] = model.predict(X_val)
        ev = ev.copy()
        ev["score_raw"] = model.predict(ev[feats])

        # pooled
        iso_pooled = fit_calibrator(y_val.values, val_split["score_raw"].values)
        ev["score_pooled"] = iso_pooled.predict(ev["score_raw"].values)

        # per-airport on the train regime, transferred by regime to eval
        per_airport = fit_per_airport_calibrators(val_split)

        # build a regime-aware transfer: each eval airport maps to nearest-prev
        # train airport in the *same* training regime. Compute by ranking
        # train-regime val prevalences.
        train_prev = (val_split.groupby("airport")["y_risk"]
                      .apply(lambda s: s.astype(int).mean())
                      .to_dict())

        def nearest_train(prev_target: float) -> str:
            return min(train_prev, key=lambda k: abs(train_prev[k] - prev_target))

        eval_prev = (ev.groupby("airport")["y_risk"]
                     .apply(lambda s: s.astype(int).mean())
                     .to_dict())
        transfer = {code: nearest_train(p) for code, p in eval_prev.items()}
        for src, dst in transfer.items():
            print(f"[loro/{name}] transfer {src} (prev {eval_prev[src]:.3%}) -> {dst} (prev {train_prev[dst]:.3%})")

        ev["score_per_airport"] = apply_per_airport(ev, per_airport, transfer)

        for code, g in ev.groupby("airport", sort=False):
            runs.append(score_split(g["y_risk"], g["score_raw"],         f"loro_{name}_raw/{code}"))
            runs.append(score_split(g["y_risk"], g["score_pooled"],      f"loro_{name}_pooled/{code}"))
            runs.append(score_split(g["y_risk"], g["score_per_airport"], f"loro_{name}_per_airport/{code}"))
    return runs



# Comparison helpers                                                    #

def _delta_table(rows: list[dict], variants: tuple[str, ...]):
    """Pivot [{label: 'pooled/test/KATL', pr_auc: ..., ece: ...}, ...] into
    a per-airport delta table across variants."""
    parsed = []
    for r in rows:
        if "pr_auc" not in r:
            continue
        parts = r["label"].split("/")
        # Two label shapes:
        #   <variant>/<split>/<airport>           e.g. pooled/test/KATL
        #   <variant_with_split>/<airport>        e.g. loro_inland_to_coastal_pooled/KSFO
        if len(parts) == 3:
            variant, split, airport = parts
        elif len(parts) == 2:
            variant, airport = parts
            split = "loro"
        else:
            continue
        parsed.append({"variant": variant, "split": split, "airport": airport,
                       "pr_auc": r["pr_auc"], "ece": r["ece"], "brier": r["brier"]})
    return pd.DataFrame(parsed)


def _main():
    cfg = load_config()
    proc = ROOT / cfg["paths"]["processed_dir"]
    train = pd.read_parquet(proc / "train_features.parquet")
    test  = pd.read_parquet(proc / "test_features.parquet")
    print(f"train={train.shape} test={test.shape}")

    feats = [c for c in feature_columns(train) if train[c].dtype.kind in "fciub"]
    print(f"features={len(feats)}")

    std = run_standard(train, test, feats)
    loro = run_loro(train, test, feats)

    _print_table("Val (train airports, last 20%)", std["val"])
    _print_table("Test (KATL/KBOS unseen)", std["test"])
    _print_table("LORO", loro)

    # Head-to-head: per_airport vs pooled (test set)
    df = _delta_table(std["test"], ("pooled", "per_airport"))
    if not df.empty:
        piv_pr = df.pivot_table(index="airport", columns="variant", values="pr_auc")
        piv_ece = df.pivot_table(index="airport", columns="variant", values="ece")
        if {"pooled", "per_airport"}.issubset(piv_pr.columns):
            piv_pr["delta_PR"] = piv_pr["per_airport"] - piv_pr["pooled"]
            piv_ece["delta_ECE"] = piv_ece["per_airport"] - piv_ece["pooled"]
            print("\n=== Test PR-AUC: pooled vs per-airport ===")
            print(piv_pr.round(4).to_string())
            print("\n=== Test ECE: pooled vs per-airport (lower=better) ===")
            print(piv_ece.round(4).to_string())

    # LORO comparison
    df_l = _delta_table(loro, ("pooled", "per_airport"))
    if not df_l.empty:
        # split label into name + variant
        # variant is like 'loro_inland_to_coastal_raw' / '_pooled' / '_per_airport'
        df_l["direction"] = df_l["variant"].str.extract(r"loro_(inland_to_coastal|coastal_to_inland)")
        df_l["calib"] = df_l["variant"].str.extract(r"_(raw|pooled|per_airport)$")
        piv = df_l.pivot_table(index=["direction", "airport"], columns="calib",
                               values="pr_auc")
        piv_ece = df_l.pivot_table(index=["direction", "airport"], columns="calib",
                                   values="ece")
        print("\n=== LORO PR-AUC by calibration ===")
        print(piv.round(4).to_string())
        print("\n=== LORO ECE by calibration (lower=better) ===")
        print(piv_ece.round(4).to_string())

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    payload = {
        "n_features": len(feats),
        "transfer_map": TRANSFER_MAP,
        "val": std["val"], "test": std["test"], "loro": loro,
    }
    (reports / "calibration_per_airport_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {reports / 'calibration_per_airport_results.json'}")


if __name__ == "__main__":
    _main()
