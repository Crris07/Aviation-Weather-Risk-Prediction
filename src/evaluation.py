"""Evaluation utilities: cleaner metrics, calibration, domain splits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)


def recall_at_fpr(y, s, target_fpr: float = 0.1) -> float:
    """Max recall while FPR <= target_fpr."""
    fpr, tpr, _ = roc_curve(y, s)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def precision_at_recall(y, s, target_recall: float = 0.9) -> float:
    """Max precision while recall >= target_recall."""
    prec, rec, _ = precision_recall_curve(y, s)
    mask = rec[:-1] >= target_recall
    return float(prec[:-1][mask].max()) if mask.any() else 0.0


def expected_calibration_error(y, s, n_bins: int = 15) -> float:
    """ECE: weighted mean gap between confidence and accuracy across bins."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s >= lo) & (s < hi) if hi < 1 else (s >= lo) & (s <= hi)
        if mask.sum() == 0:
            continue
        conf = s[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def score_split(y_true: pd.Series, y_score: pd.Series, label: str) -> dict:
    """Full metric bundle for one (airport, split) combo."""
    mask = y_true.notna() & y_score.notna()
    y = y_true[mask].astype(int).values
    s = y_score[mask].values
    if y.sum() == 0 or y.sum() == len(y):
        return {"label": label, "n": int(len(y)), "note": "degenerate"}
    return {
        "label": label,
        "n": int(len(y)),
        "pos": int(y.sum()),
        "prevalence": float(y.mean()),
        # ranking
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        # operating points (aviation: asymmetric cost favours recall)
        "recall_at_fpr10": recall_at_fpr(y, s, 0.1),
        "precision_at_recall90": precision_at_recall(y, s, 0.9),
        # calibration
        "brier": float(brier_score_loss(y, s)),
        "ece": expected_calibration_error(y, s),
    }


def reliability_curve(y, s, n_bins: int = 15) -> pd.DataFrame:
    """Binned confidence vs accuracy for plotting."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s >= lo) & (s < hi) if hi < 1 else (s >= lo) & (s <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(mask.sum()),
                     "conf": float(s[mask].mean()), "acc": float(y[mask].mean())})
    return pd.DataFrame(rows)
