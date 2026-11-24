"""Permutation importance of the ALREADY-SELECTED model on CV validation folds of the training portion only.

Held-out test rows are never passed in. Each fold's model is the selected model fitted on that fold's ``T_f``
(screening inside ``fit``); a feature that was not eligible in a fold has no importance for that fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.predictive.contracts import IMPORTANCE_REPEATS
from src.predictive.metrics import selection_score


def fold_importance(model, X_val: pd.DataFrame, y_val: pd.Series, task: str, classes, seed: int, fold: int,
                    repeats: int = IMPORTANCE_REPEATS) -> list:
    """Per-feature importance on one validation fold: regression = RMSE increase, classification = macro-F1 drop."""
    base = selection_score(task, y_val, model.predict(X_val), classes)
    rng = np.random.default_rng(seed + 1000 + fold)
    rows = []
    for col in model.cols_:
        vals = []
        for _ in range(repeats):
            Xp = X_val.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            sc = selection_score(task, y_val, model.predict(Xp), classes)
            vals.append(sc - base if task == "regression" else base - sc)
        rows.append({"feature": col, "fold": fold, "importance": float(np.mean(vals))})
    return rows


def summarise(fold_rows: list, n_folds: int) -> pd.DataFrame:
    if not fold_rows:
        return pd.DataFrame(columns=["feature", "mean", "sd_across_folds", "folds_eligible", "folds_total"])
    f = pd.DataFrame(fold_rows)
    g = f.groupby("feature")["importance"]
    out = pd.DataFrame({"mean": g.mean(), "sd_across_folds": g.std(ddof=1), "folds_eligible": g.count()})
    out["folds_total"] = n_folds
    return out.reset_index().sort_values("mean", ascending=False).reset_index(drop=True)
