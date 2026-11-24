"""Metric definitions (documented in the export Methodology sheet) and the deterministic bootstrap.

RMSE = sqrt(mean squared error) (scikit-learn >= 1.3 safe); R2 = 1 - SSE/SST (never clipped); MAPE only when no
actual value is zero; macro metrics average over the classes of the TRAINING portion with zero_division = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from src.predictive.contracts import BOOTSTRAP_N


def regression_metrics(y_true, y_pred) -> dict:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    sse = float(((yt - yp) ** 2).sum())
    sst = float(((yt - yt.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt(mean_squared_error(yt, yp))), "mae": float(mean_absolute_error(yt, yp)),
            "r2": (1.0 - sse / sst) if sst > 0 else None,
            "mape": float(np.mean(np.abs((yt - yp) / yt)) * 100) if len(yt) and not np.any(yt == 0) else None}


def _cm(y_true_idx, y_pred_idx, k: int) -> np.ndarray:
    return np.bincount(y_true_idx * k + y_pred_idx, minlength=k * k).reshape(k, k)


def cm_macro(cm: np.ndarray) -> dict:
    """Macro precision/recall/F1 and accuracy from a confusion matrix (rows = actual, cols = predicted)."""
    tp = np.diag(cm).astype(float)
    pred, act = cm.sum(0).astype(float), cm.sum(1).astype(float)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, act, out=np.zeros_like(tp), where=act > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    present = act > 0
    return {"macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()), "macro_f1": float(f1.mean()),
            "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
            "balanced_accuracy": float(rec[present].mean()) if present.any() else 0.0,
            "_prec": prec, "_rec": rec, "_f1": f1, "_support": act}


def _encode(y, classes):
    lut = {str(c): i for i, c in enumerate(classes)}
    return np.array([lut.get(str(v), -1) for v in y])


def classification_metrics(y_true, y_pred, classes, proba=None) -> dict:
    """Metrics with ``classes`` fixed to the training classes (so an unseen label counts as an error)."""
    k = len(classes)
    t, p = _encode(y_true, classes), _encode(y_pred, classes)
    keep = (t >= 0) & (p >= 0)
    cm = _cm(t[keep], p[keep], k)
    m = cm_macro(cm)
    out = {key: m[key] for key in ("macro_f1", "macro_precision", "macro_recall", "accuracy", "balanced_accuracy")}
    out["n_unseen_labels"] = int((~keep).sum())
    out["_cm"] = cm
    out["_table"] = pd.DataFrame({"class": [str(c) for c in classes], "precision": m["_prec"], "recall": m["_rec"],
                                  "f1": m["_f1"], "support": m["_support"].astype(int)})
    out["roc_auc"] = roc_auc(y_true, proba, classes)
    return out


def roc_auc(y_true, proba, classes):
    """ROC AUC only when mathematically valid; otherwise None (binary: second sorted class is positive)."""
    if proba is None:
        return None
    t = _encode(y_true, classes)
    proba = np.asarray(proba, float)
    present = np.unique(t[t >= 0])
    if len(present) < 2 or proba.shape[1] != len(classes):
        return None
    if len(classes) == 2:
        return float(roc_auc_score((t == 1).astype(int), proba[:, 1]))
    if len(present) != len(classes):
        return None
    return float(roc_auc_score(t, proba, multi_class="ovr", average="macro", labels=list(range(len(classes)))))


def selection_score(task: str, y_true, y_pred, classes=None) -> float:
    """The selection metric: RMSE (lower is better) or macro-F1 (higher is better)."""
    if task == "regression":
        return regression_metrics(y_true, y_pred)["rmse"]
    t, p = _encode(y_true, classes), _encode(y_pred, classes)
    keep = (t >= 0) & (p >= 0)
    return cm_macro(_cm(t[keep], p[keep], len(classes)))["macro_f1"]


def bootstrap_ci(task, y_true, y_pred, classes=None, proba=None, seed=42, n_boot=BOOTSTRAP_N) -> tuple:
    """Percentile 95% bootstrap of held-out metrics: resamples (actual, predicted) pairs; NO refitting."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    ci, notes = {}, []
    if task == "regression":
        yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
        e = (yt - yp)
        rm = np.sqrt((e[idx] ** 2).mean(1)); ma = np.abs(e[idx]).mean(1)
        ytb = yt[idx]; sst = ((ytb - ytb.mean(1, keepdims=True)) ** 2).sum(1)
        sse = (e[idx] ** 2).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where(sst > 0, 1 - sse / sst, np.nan)
        for name, arr in (("rmse", rm), ("mae", ma), ("r2", r2)):
            a = arr[~np.isnan(arr)]
            if len(a):
                ci[name] = (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
        return ci, notes
    k = len(classes)
    t, p = _encode(y_true, classes), _encode(y_pred, classes)
    ok = (t >= 0) & (p >= 0)
    t, p = t[ok], p[ok]
    pr = proba[ok] if proba is not None else None
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    vals = {"macro_f1": [], "macro_precision": [], "macro_recall": [], "accuracy": []}
    for row in idx:
        m = cm_macro(_cm(t[row], p[row], k))
        for key in vals:
            vals[key].append(m[key])
    for key, arr in vals.items():
        ci[key] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    if pr is not None and k == 2:
        auc, skipped = [], 0
        for row in idx:
            tb = (t[row] == 1).astype(int)
            if tb.min() == tb.max():
                skipped += 1
                continue
            auc.append(roc_auc_score(tb, pr[row, 1]))
        if auc:
            ci["roc_auc"] = (float(np.percentile(auc, 2.5)), float(np.percentile(auc, 97.5)))
        if skipped:
            notes.append(f"{skipped} bootstrap resamples contained a single class and were skipped for ROC AUC.")
    return ci, notes
