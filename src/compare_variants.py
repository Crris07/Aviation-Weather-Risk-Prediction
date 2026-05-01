"""Compare Phase 4 clean vs Phase 5 physics experiment artifacts.

Reads the variant-specific report bundles emitted by `src.baseline` and writes:
  - compact comparison CSVs
  - test-set PR-AUC comparison plot
  - LORO calibrated PR-AUC comparison plot
  - feature-importance plot highlighting Phase 5 physics features
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .ingest import ROOT


VARIANTS = ("phase4_clean", "phase5_physics")
PHYSICS_FEATURES = {
    "dew_depression_min_3h",
    "dew_depression_min_6h",
    "calm_and_saturated",
    "dew_depression_x_hour_sin",
    "pressure_drop_x_wind",
    "wind_dir_sin_coastal",
    "wind_dir_cos_coastal",
    "calm_and_saturated_coastal",
}


def _rows(payload: dict, key: str, variant: str) -> pd.DataFrame:
    out = []
    for r in payload[key]:
        if "pr_auc" not in r:
            continue
        parts = r["label"].split("/")
        out.append({
            "variant": variant,
            "label": r["label"],
            "model": parts[0],
            "split": parts[1] if len(parts) > 2 else "",
            "airport": parts[-1],
            **{k: r[k] for k in (
                "pr_auc", "roc_auc", "recall_at_fpr10",
                "precision_at_recall90", "brier", "ece",
                "prevalence", "n",
            )},
        })
    return pd.DataFrame(out)


def _load_payloads(reports: Path) -> dict[str, dict]:
    payloads = {}
    for variant in VARIANTS:
        path = reports / f"baseline_results_{variant}.json"
        payloads[variant] = json.loads(path.read_text())
    return payloads


def _build_metric_tables(payloads: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = []
    loro_rows = []
    for variant, payload in payloads.items():
        test_rows.append(_rows(payload, "lgbm_test", variant))
        loro_rows.append(_rows(payload, "loro", variant))
    return pd.concat(test_rows, ignore_index=True), pd.concat(loro_rows, ignore_index=True)


def _save_test_comparison(test_df: pd.DataFrame, out_dir: Path) -> None:
    cal = test_df[test_df["model"] == "lgbm_cal"].copy()
    piv = cal.pivot_table(index="airport", columns="variant", values="pr_auc")
    piv["delta_physics_minus_clean"] = piv["phase5_physics"] - piv["phase4_clean"]
    piv = piv.reset_index()
    piv.to_csv(out_dir / "phase4_vs_phase5_test_pr_auc.csv", index=False)

    plot_df = cal[["airport", "variant", "pr_auc"]].copy()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="airport", y="pr_auc", hue="variant",
                palette={"phase4_clean": "#7a8aa0", "phase5_physics": "#2a7f62"})
    plt.axhline(0.40, ls="--", color="red", lw=0.8)
    plt.ylabel("PR-AUC")
    plt.xlabel("")
    plt.title("Test-set PR-AUC: clean Phase 4 vs physics Phase 5")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_vs_phase5_test_pr_auc.png", dpi=140)
    plt.close()


def _save_loro_comparison(loro_df: pd.DataFrame, out_dir: Path) -> None:
    loro = loro_df.copy()
    loro["direction"] = loro["model"].str.extract(r"loro_([a-z_]+?)_(?:raw|cal)")
    loro["calibration"] = loro["model"].str.extract(r"(raw|cal)$")
    loro = loro[loro["calibration"] == "cal"].copy()

    piv = loro.pivot_table(
        index=["airport", "direction"],
        columns="variant",
        values="pr_auc",
    )
    piv["delta_physics_minus_clean"] = piv["phase5_physics"] - piv["phase4_clean"]
    piv = piv.reset_index()
    piv.to_csv(out_dir / "phase4_vs_phase5_loro_pr_auc.csv", index=False)

    plot_df = loro[["airport", "direction", "variant", "pr_auc"]].copy()
    plot_df["scenario"] = plot_df["airport"] + "\n" + plot_df["direction"].str.replace("_", "->")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="scenario", y="pr_auc", hue="variant",
                palette={"phase4_clean": "#7a8aa0", "phase5_physics": "#2a7f62"})
    plt.axhline(0.40, ls="--", color="red", lw=0.8)
    plt.ylabel("PR-AUC")
    plt.xlabel("")
    plt.title("LORO calibrated PR-AUC: robustness with and without physics features")
    plt.tight_layout()
    plt.savefig(out_dir / "phase4_vs_phase5_loro_pr_auc.png", dpi=140)
    plt.close()


def _save_importance_comparison(reports: Path, out_dir: Path) -> None:
    frames = []
    for variant in VARIANTS:
        imp = pd.read_csv(reports / f"baseline_feature_importance_{variant}.csv")
        imp["variant"] = variant
        imp["is_physics_feature"] = imp["feature"].isin(PHYSICS_FEATURES)
        frames.append(imp)
    all_imp = pd.concat(frames, ignore_index=True)
    all_imp.to_csv(out_dir / "phase4_vs_phase5_feature_importance_full.csv", index=False)

    phase5 = all_imp[all_imp["variant"] == "phase5_physics"].copy()
    top_phase5 = phase5.head(20).copy()
    top_phase5["label"] = top_phase5["feature"]
    top_phase5["group"] = top_phase5["is_physics_feature"].map({True: "physics", False: "non_physics"})

    plt.figure(figsize=(9, 7))
    sns.barplot(
        data=top_phase5,
        x="gain",
        y="label",
        hue="group",
        dodge=False,
        palette={"physics": "#d66b2d", "non_physics": "#4f81bd"},
    )
    plt.xlabel("LightGBM gain")
    plt.ylabel("")
    plt.title("Phase 5 top-20 feature importance\nPhysics features highlighted")
    plt.tight_layout()
    plt.savefig(out_dir / "phase5_feature_importance_top20.png", dpi=140)
    plt.close()

    physics_only = phase5[phase5["is_physics_feature"]].copy().head(10)
    physics_only.to_csv(out_dir / "phase5_physics_feature_importance.csv", index=False)


def main() -> None:
    sns.set_theme(style="whitegrid")
    reports = ROOT / "reports"
    out_dir = reports / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = _load_payloads(reports)
    test_df, loro_df = _build_metric_tables(payloads)
    _save_test_comparison(test_df, out_dir)
    _save_loro_comparison(loro_df, out_dir)
    _save_importance_comparison(reports, out_dir)

    print("saved comparison outputs to", out_dir)


if __name__ == "__main__":
    main()
