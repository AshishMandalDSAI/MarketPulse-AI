"""Time-series evaluation shared by the forecasting engine and the generic MMM.

* ``ts_metrics`` — MAE, RMSE, MAPE, sMAPE, WAPE and R² with explicit rules for
  zeros: MAPE skips periods whose actual value is 0 (and reports how many were
  skipped; ``None`` if every actual is 0), sMAPE skips periods where actual and
  prediction are both 0, WAPE is ``None`` when the actuals sum to 0, and R² is
  ``None`` for fewer than 2 points or a constant actual series. Nothing is
  silently turned into 0 — a 0 error would read as a perfect model.
* ``rolling_origin_backtest`` — walk-forward evaluation: the model is refitted
  on data up to each origin and scored on the next ``horizon`` points it has
  never seen. Origins step backwards from the end, so every fold respects time
  order and no fold ever trains on its own test points.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


def ts_metrics(actual, predicted) -> Dict[str, Optional[float]]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    n = int(len(a))
    if n == 0:
        return {"MAE": None, "RMSE": None, "MAPE": None, "sMAPE": None, "WAPE": None, "R2": None,
                "n": 0, "n_zero_actuals_excluded_from_MAPE": 0}
    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = a != 0
    mape = float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else None
    denom = np.abs(a) + np.abs(p)
    ok = denom > 0
    smape = float(np.mean(2.0 * np.abs(err[ok]) / denom[ok]) * 100) if ok.any() else None
    total = float(np.sum(np.abs(a)))
    wape = float(np.sum(np.abs(err)) / total * 100) if total > 0 else None
    sst = float(np.sum((a - a.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / sst) if (n >= 2 and sst > 0) else None
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WAPE": wape, "R2": r2,
            "n": n, "n_zero_actuals_excluded_from_MAPE": int((~nz).sum())}


def rolling_origin_backtest(values, fit_predict: Callable[[np.ndarray, int], np.ndarray], *,
                            horizon: int, n_folds: int = 3, min_train: int = 4) -> dict:
    """Walk-forward evaluation.

    ``fit_predict(train, h)`` must fit on ``train`` only and return ``h``
    predictions for the periods immediately after it.
    Returns per-fold metrics, pooled metrics over all test points, the pooled
    errors (actual - predicted) and the fold boundaries (as integer positions).
    """
    y = np.asarray(values, dtype=float)
    n = len(y)
    h = max(1, int(horizon))
    folds: List[dict] = []
    all_a: List[float] = []
    all_p: List[float] = []
    for k in range(n_folds, 0, -1):
        origin = n - k * h
        if origin < min_train:
            continue
        test = y[origin:origin + h]
        if len(test) == 0:
            continue
        pred = np.asarray(fit_predict(y[:origin].copy(), len(test)), dtype=float)[:len(test)]
        folds.append({"train_end": int(origin), "test_start": int(origin), "test_end": int(origin + len(test)),
                      "metrics": ts_metrics(test, pred)})
        all_a.extend(test.tolist())
        all_p.extend(pred.tolist())
    pooled = ts_metrics(all_a, all_p) if all_a else ts_metrics([], [])
    errors = (np.asarray(all_a) - np.asarray(all_p)).tolist() if all_a else []
    return {"n_folds": len(folds), "horizon": h, "folds": folds, "metrics": pooled, "errors": errors}
