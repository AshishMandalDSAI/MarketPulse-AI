"""Phase 4 -- General Forecasting Engine.

Consumes an AnalysisContext (from the canonical data contract) and produces a
ForecastResult.  Trains multiple simple models, selects the best via
time-aware validation, and generates point forecasts with confidence intervals.

No hardcoded column names.  Confidence derives from actual validation metrics.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction.contracts import ForecastResult, ModelCandidate, FeatureImportance
from src.data_engine.canonical import AnalysisContext
from src.prediction.task_detector import MAX_ROWS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ROWS = 10
_TRAIN_FRACTION = 0.80
_CONFIDENCE_Z = 1.96  # 95 % prediction interval


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_time_attr(time_info, attr: str, default=None):
    """Read a TimeInfo attribute regardless of whether it is a dataclass or dict."""
    if time_info is None:
        return default
    if hasattr(time_info, attr):
        return getattr(time_info, attr)
    if isinstance(time_info, dict):
        return time_info.get(attr, default)
    return default


def _detect_frequency(time_info) -> str:
    """Map time_info.time_granularity to a normalised frequency label."""
    raw = _safe_time_attr(time_info, "time_granularity", "irregular")
    mapping = {
        "daily": "daily",
        "weekly": "weekly",
        "monthly": "monthly",
        "quarterly": "monthly",   # treat quarterly as monthly for offset logic
        "yearly": "yearly",
    }
    return mapping.get(raw, "irregular")


def _generate_future_dates(last_date: pd.Timestamp, horizon: int,
                           frequency: str) -> list[str]:
    """Generate *horizon* future date strings starting after *last_date*."""
    offset_map = {
        "daily": pd.DateOffset(days=1),
        "weekly": pd.DateOffset(weeks=1),
        "monthly": pd.DateOffset(months=1),
        "yearly": pd.DateOffset(years=1),
    }
    offset = offset_map.get(frequency, pd.DateOffset(days=1))
    dates: list[str] = []
    current = last_date
    for _ in range(horizon):
        current = current + offset
        dates.append(current.strftime("%Y-%m-%d"))
    return dates


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE, RMSE, MAPE, sMAPE, WAPE and R² (see src/analytics/ts_evaluation).
    Undefined metrics are None, never 0 or infinity."""
    from src.analytics.ts_evaluation import ts_metrics
    m = ts_metrics(actual, predicted)
    return {k: m[k] for k in ("MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "R2")}


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def _train_naive(train: np.ndarray, val: np.ndarray,
                 horizon: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray]:
    """Naive forecast: repeat the last training value."""
    t0 = time.time()
    last = train[-1]
    val_pred = np.full(len(val), last)
    future_pred = np.full(horizon, last)
    metrics = _compute_metrics(val, val_pred)
    elapsed = time.time() - t0
    candidate = ModelCandidate(
        name="Naive (Last Value)",
        model_type="naive",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred, future_pred


def _train_seasonal_naive(train: np.ndarray, val: np.ndarray,
                          horizon: int,
                          season_length: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray]:
    """Seasonal naive: repeat the last seasonal cycle."""
    t0 = time.time()
    # Validation predictions
    val_pred = np.empty(len(val))
    for i in range(len(val)):
        idx = len(train) - season_length + (i % season_length)
        val_pred[i] = train[idx] if 0 <= idx < len(train) else train[-1]

    # Future predictions
    full = np.concatenate([train, val])
    future_pred = np.empty(horizon)
    for i in range(horizon):
        idx = len(full) - season_length + (i % season_length)
        future_pred[i] = full[idx] if 0 <= idx < len(full) else full[-1]

    metrics = _compute_metrics(val, val_pred)
    elapsed = time.time() - t0
    candidate = ModelCandidate(
        name="Seasonal Naive",
        model_type="seasonal_naive",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred, future_pred


def _train_moving_average(train: np.ndarray, val: np.ndarray,
                          horizon: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray]:
    """Moving-average forecast using a rolling window."""
    t0 = time.time()
    window = min(12, max(2, len(train) // 3))

    # Validation: multi-step, recursive on its own predictions — exactly how
    # the future forecast is produced. (Feeding the actual validation values
    # back in, as earlier versions did, made this candidate look far more
    # accurate than its real multi-step forecasts.)
    series = list(train)
    val_pred = np.empty(len(val))
    for i in range(len(val)):
        val_pred[i] = np.mean(series[-window:])
        series.append(val_pred[i])

    # Future: keep extending with predictions
    series_full = list(np.concatenate([train, val]))
    future_pred = np.empty(horizon)
    for i in range(horizon):
        pred = np.mean(series_full[-window:])
        future_pred[i] = pred
        series_full.append(pred)

    metrics = _compute_metrics(val, val_pred)
    elapsed = time.time() - t0
    candidate = ModelCandidate(
        name=f"Moving Average (window={window})",
        model_type="moving_average",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred, future_pred


def _train_linear_trend(train: np.ndarray, val: np.ndarray,
                        horizon: int) -> tuple[ModelCandidate, np.ndarray, np.ndarray, float]:
    """Linear trend via sklearn LinearRegression on time index.

    Returns the candidate, validation predictions, future predictions,
    and the trend coefficient.
    """
    from sklearn.linear_model import LinearRegression

    t0 = time.time()
    X_train = np.arange(len(train)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train, train)

    # Validation
    X_val = np.arange(len(train), len(train) + len(val)).reshape(-1, 1)
    val_pred = model.predict(X_val)

    # Future (retrained on all data)
    full = np.concatenate([train, val])
    X_full = np.arange(len(full)).reshape(-1, 1)
    model.fit(X_full, full)
    X_future = np.arange(len(full), len(full) + horizon).reshape(-1, 1)
    future_pred = model.predict(X_future)

    coef = float(model.coef_[0])

    # Metrics on validation set (from the train-only fit)
    model_train_only = LinearRegression().fit(X_train, train)
    val_pred_eval = model_train_only.predict(X_val)
    metrics = _compute_metrics(val, val_pred_eval)
    elapsed = time.time() - t0

    candidate = ModelCandidate(
        name="Linear Trend",
        model_type="linear_trend",
        metrics=metrics,
        score=metrics["MAE"],
        training_time_seconds=elapsed,
    )
    return candidate, val_pred_eval, future_pred, coef


def _train_exponential_smoothing(
    train: np.ndarray, val: np.ndarray, horizon: int, frequency: str,
) -> Optional[tuple[ModelCandidate, np.ndarray, np.ndarray]]:
    """Exponential smoothing via statsmodels.  Returns None if unavailable."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return None

    t0 = time.time()
    try:
        # Choose seasonal periods if enough data
        seasonal_periods_map = {
            "daily": 7,
            "weekly": 52,
            "monthly": 12,
            "yearly": 1,
        }
        sp = seasonal_periods_map.get(frequency, 1)
        use_seasonal = sp > 1 and len(train) >= 2 * sp

        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=sp if use_seasonal else None,
        ).fit(optimized=True)

        val_pred = model.forecast(len(val))

        # Retrain on all data for future predictions
        full = np.concatenate([train, val])
        model_full = ExponentialSmoothing(
            full,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=sp if use_seasonal else None,
        ).fit(optimized=True)
        future_pred = model_full.forecast(horizon)

        # Replace any NaN predictions with the last known value
        fallback = full[-1]
        val_pred = np.where(np.isfinite(val_pred), val_pred, fallback)
        future_pred = np.where(np.isfinite(future_pred), future_pred, fallback)

        metrics = _compute_metrics(val, val_pred)
        elapsed = time.time() - t0

        candidate = ModelCandidate(
            name="Exponential Smoothing",
            model_type="exponential_smoothing",
            metrics=metrics,
            score=metrics["MAE"],
            training_time_seconds=elapsed,
        )
        return candidate, val_pred, future_pred

    except Exception as exc:
        logger.debug("Exponential smoothing failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Season-length heuristic
# ---------------------------------------------------------------------------

def _guess_season_length(frequency: str, n_obs: int) -> int:
    """Return a reasonable seasonal period length based on the frequency."""
    defaults = {
        "daily": 7,
        "weekly": 52,
        "monthly": 12,
        "yearly": 1,
    }
    sp = defaults.get(frequency, 1)
    # Only use seasonality if we have at least two full cycles
    if sp > 1 and n_obs < 2 * sp:
        sp = 1
    return max(sp, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_forecast(
    context: AnalysisContext,
    target_column: str = None,
    horizon: int = None,
    aggregate: str = "auto",
) -> ForecastResult:
    """Run the general forecasting engine.

    Parameters
    ----------
    context : AnalysisContext
        The canonical analysis context containing the dataset, time info,
        target candidates, and data quality metadata.
    target_column : str, optional
        The column to forecast.  If not provided, the first target candidate
        from ``context.dataset.target_candidates`` is used.
    horizon : int, optional
        Number of future periods to forecast.  Defaults to
        ``min(12, n_observations // 5)``.

    Returns
    -------
    ForecastResult
        A fully populated forecast result with predictions, confidence
        intervals, model candidates, and validation metrics.
    """
    ds = context.dataset
    df = ds.dataframe

    # ------------------------------------------------------------------
    # Guard: no DataFrame
    # ------------------------------------------------------------------
    if df is None or df.empty:
        return ForecastResult(
            status="unavailable",
            explanation="No data available for forecasting.",
            limitations=["DataFrame is empty or missing."],
        )

    # ------------------------------------------------------------------
    # Time column detection
    # ------------------------------------------------------------------
    time_info = ds.time_info
    time_col = _safe_time_attr(time_info, "time_column") if hasattr(time_info, "time_column") or isinstance(time_info, dict) else None

    if not time_col or time_col not in df.columns:
        return ForecastResult(
            status="unavailable",
            explanation="No time column detected in the dataset.",
            limitations=["Forecasting requires a datetime column for temporal ordering."],
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    if target_column and target_column in df.columns:
        target_col = target_column
    elif ds.target_candidates:
        first_candidate = ds.target_candidates[0]
        target_col = first_candidate.column if hasattr(first_candidate, "column") else None
    else:
        target_col = None

    if not target_col or target_col not in df.columns:
        return ForecastResult(
            status="unavailable",
            time_column=time_col,
            explanation="No suitable numeric target column found for forecasting.",
            limitations=["No numeric column qualifies as a forecast target."],
        )

    # Verify the target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return ForecastResult(
            status="unavailable",
            target_column=target_col,
            time_column=time_col,
            explanation=f"Target column '{target_col}' is not numeric.",
            limitations=["Forecasting requires a numeric target column."],
        )

    # ------------------------------------------------------------------
    # Data preparation: parse time, drop missing time/target, aggregate
    # repeated timestamps into one value per period, keep time order.
    # ------------------------------------------------------------------
    working_df = df[[time_col, target_col]].copy()
    n_input_rows = len(working_df)
    working_df[time_col] = pd.to_datetime(working_df[time_col], errors="coerce")
    n_bad_time = int(working_df[time_col].isna().sum())
    working_df[target_col] = pd.to_numeric(working_df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_bad_target = int(working_df[target_col].isna().sum())
    working_df = working_df.dropna(subset=[time_col, target_col])

    agg_note = None
    n_dup_times = int(working_df[time_col].duplicated().sum())
    if n_dup_times:
        # Several rows per timestamp (transactions, stores, regions): forecast
        # the per-period total for additive measures, the mean otherwise.
        how = aggregate
        if how == "auto":
            how = "sum" if (working_df[target_col] >= 0).all() else "mean"
        working_df = (working_df.groupby(time_col, as_index=False)[target_col]
                      .agg("sum" if how == "sum" else "mean"))
        agg_note = (f"{n_dup_times} repeated timestamp(s): values were combined per period "
                    f"({'summed' if how == 'sum' else 'averaged'}) instead of treating rows as separate periods.")
    working_df = working_df.sort_values(time_col).reset_index(drop=True)

    sampled_note = None
    if len(working_df) > MAX_ROWS:
        # Keep the most recent contiguous periods (random sampling would punch
        # holes in the series).
        sampled_note = (f"Series has {len(working_df):,} periods; the most recent {MAX_ROWS:,} were used.")
        working_df = working_df.iloc[-MAX_ROWS:].reset_index(drop=True)

    n_obs = len(working_df)

    # ------------------------------------------------------------------
    # Edge case: too few rows
    # ------------------------------------------------------------------
    if n_obs < _MIN_ROWS:
        return ForecastResult(
            status="insufficient_data",
            target_column=target_col,
            time_column=time_col,
            explanation=(
                f"Only {n_obs} valid observations after cleaning. "
                f"Forecasting requires at least {_MIN_ROWS}."
            ),
            limitations=[
                f"Row count ({n_obs}) is below the minimum ({_MIN_ROWS}).",
                "Consider providing more data for reliable forecasting.",
            ],
        )

    # ------------------------------------------------------------------
    # Edge case: all-constant target
    # ------------------------------------------------------------------
    target_values = working_df[target_col].values.astype(float)
    if np.std(target_values) == 0:
        return ForecastResult(
            status="failed",
            target_column=target_col,
            time_column=time_col,
            explanation=(
                f"Target column '{target_col}' has zero variance "
                f"(constant value {target_values[0]:.4g}). "
                "Forecasting is not meaningful."
            ),
            limitations=["Target column is constant; no pattern to learn."],
        )

    # ------------------------------------------------------------------
    # Frequency and horizon
    # ------------------------------------------------------------------
    frequency = _detect_frequency(time_info)
    if horizon is None:
        horizon = min(12, max(1, n_obs // 5))

    # ------------------------------------------------------------------
    # Train / validation split (time-aware)
    # ------------------------------------------------------------------
    split_idx = max(2, int(n_obs * _TRAIN_FRACTION))
    train_vals = target_values[:split_idx]
    val_vals = target_values[split_idx:]

    if len(val_vals) < 1:
        # Fall back: use last observation as single-point validation
        split_idx = n_obs - 1
        train_vals = target_values[:split_idx]
        val_vals = target_values[split_idx:]

    # ------------------------------------------------------------------
    # Train model candidates
    # ------------------------------------------------------------------
    candidates: list[ModelCandidate] = []
    val_predictions: dict[str, np.ndarray] = {}
    future_predictions: dict[str, np.ndarray] = {}
    trend_coef: Optional[float] = None

    # 1. Naive
    cand, vp, fp = _train_naive(train_vals, val_vals, horizon)
    candidates.append(cand)
    val_predictions[cand.name] = vp
    future_predictions[cand.name] = fp

    # 2. Seasonal Naive
    season_len = _guess_season_length(frequency, n_obs)
    if season_len > 1:
        cand, vp, fp = _train_seasonal_naive(train_vals, val_vals, horizon, season_len)
        candidates.append(cand)
        val_predictions[cand.name] = vp
        future_predictions[cand.name] = fp

    # 3. Moving Average
    cand, vp, fp = _train_moving_average(train_vals, val_vals, horizon)
    candidates.append(cand)
    val_predictions[cand.name] = vp
    future_predictions[cand.name] = fp

    # 4. Linear Trend
    try:
        cand, vp, fp, coef = _train_linear_trend(train_vals, val_vals, horizon)
        candidates.append(cand)
        val_predictions[cand.name] = vp
        future_predictions[cand.name] = fp
        trend_coef = coef
    except Exception as exc:
        logger.debug("Linear trend training failed: %s", exc)

    # 5. Exponential Smoothing (optional dependency)
    es_result = _train_exponential_smoothing(train_vals, val_vals, horizon, frequency)
    if es_result is not None:
        cand, vp, fp = es_result
        candidates.append(cand)
        val_predictions[cand.name] = vp
        future_predictions[cand.name] = fp

    # ------------------------------------------------------------------
    # Walk-forward (rolling-origin) evaluation of every candidate type.
    # Each fold refits on data before the origin only and is scored on the
    # next periods it has never seen.
    # ------------------------------------------------------------------
    from src.analytics.ts_evaluation import rolling_origin_backtest

    bt_h = max(1, min(horizon, len(val_vals), max(1, n_obs // 6)))
    fit_fns = {
        "naive": lambda tr, h: _train_naive(tr, np.zeros(h), h)[1],
        "moving_average": lambda tr, h: _train_moving_average(tr, np.zeros(h), h)[1],
        "linear_trend": lambda tr, h: _train_linear_trend(tr, np.zeros(h), h)[1],
    }
    if season_len > 1:
        fit_fns["seasonal_naive"] = lambda tr, h: _train_seasonal_naive(tr, np.zeros(h), h, season_len)[1]
    if es_result is not None:
        fit_fns["exponential_smoothing"] = (
            lambda tr, h: (_train_exponential_smoothing(tr, np.zeros(h), h, frequency) or (None, np.full(h, tr[-1])))[1])

    backtests = {}
    for cand in candidates:
        fn = fit_fns.get(cand.model_type)
        if fn is None:
            continue
        try:
            bt = rolling_origin_backtest(target_values, fn, horizon=bt_h, n_folds=3,
                                         min_train=max(3, season_len + 1 if season_len > 1 else 3))
        except Exception as exc:  # a candidate that cannot be backtested is not selectable
            logger.debug("Backtest failed for %s: %s", cand.name, exc)
            continue
        if bt["n_folds"] and bt["metrics"]["MAE"] is not None:
            backtests[cand.name] = bt
            cand.metrics = {**cand.metrics, **{f"backtest_{k}": v for k, v in bt["metrics"].items()
                                               if k in ("MAE", "RMSE", "MAPE", "sMAPE", "WAPE")}}
            cand.score = bt["metrics"]["MAE"]

    # ------------------------------------------------------------------
    # Model selection: lowest walk-forward MAE, but a more complex model is
    # chosen over the best naive baseline only if it is at least 2% better.
    # ------------------------------------------------------------------
    MIN_IMPROVEMENT = 0.02
    baseline_types = ("naive", "seasonal_naive")

    def _score(c):
        bt = backtests.get(c.name)
        return bt["metrics"]["MAE"] if bt else c.metrics.get("MAE") if c.metrics.get("MAE") is not None else float("inf")

    best_candidate: Optional[ModelCandidate] = min(candidates, key=_score) if candidates else None
    baselines = [c for c in candidates if c.model_type in baseline_types]
    best_baseline = min(baselines, key=_score) if baselines else None
    selection_note = ""
    if best_candidate is not None and best_baseline is not None and best_candidate.model_type not in baseline_types:
        base_mae, best_mae_bt = _score(best_baseline), _score(best_candidate)
        if not (best_mae_bt <= base_mae * (1 - MIN_IMPROVEMENT)):
            selection_note = (f"{best_candidate.name} was not at least {MIN_IMPROVEMENT:.0%} better than "
                              f"{best_baseline.name} in walk-forward testing, so the simpler baseline is used.")
            best_candidate = best_baseline
    best_mae = _score(best_candidate) if best_candidate is not None else float("inf")

    if best_candidate is None or not np.isfinite(best_mae):
        return ForecastResult(
            status="failed",
            target_column=target_col,
            time_column=time_col,
            horizon=horizon,
            frequency=frequency,
            explanation="All model candidates failed during training.",
            model_candidates=candidates,
            limitations=["No model produced valid predictions."],
        )

    best_candidate.selected = True
    selected_name = best_candidate.name
    preds = future_predictions[selected_name]
    val_preds = val_predictions[selected_name]

    # ------------------------------------------------------------------
    # Prediction intervals from out-of-sample (walk-forward) errors; fall
    # back to holdout residuals when too few folds exist. Approximate: they
    # assume errors are roughly normal and stable over time.
    # ------------------------------------------------------------------
    sel_bt = backtests.get(selected_name)
    oos_errors = np.asarray(sel_bt["errors"]) if sel_bt else (val_vals - val_preds)
    residual_std = float(np.std(oos_errors, ddof=1)) if len(oos_errors) > 1 else 0.0

    lower_bound = (preds - _CONFIDENCE_Z * residual_std).tolist()
    upper_bound = (preds + _CONFIDENCE_Z * residual_std).tolist()

    times = working_df[time_col]
    def _rng(i0, i1):
        i1 = max(i0, min(i1, n_obs)) - 1
        return [str(pd.Timestamp(times.iloc[i0]).date()), str(pd.Timestamp(times.iloc[i1]).date())] if n_obs else None

    naive_bt = backtests.get(best_baseline.name) if best_baseline is not None else None
    evaluation = {
        "method": "rolling-origin (walk-forward) backtest",
        "folds": sel_bt["n_folds"] if sel_bt else 0,
        "fold_horizon": bt_h,
        "data_range": _rng(0, n_obs),
        "training_range_final_model": _rng(0, n_obs),
        "holdout_range": _rng(split_idx, n_obs),
        "fold_test_ranges": [_rng(f["test_start"], f["test_end"]) for f in (sel_bt["folds"] if sel_bt else [])],
        "selected_backtest_metrics": sel_bt["metrics"] if sel_bt else None,
        "baseline": best_baseline.name if best_baseline is not None else None,
        "baseline_backtest_metrics": naive_bt["metrics"] if naive_bt else None,
        "improvement_vs_baseline_pct": (
            round((1 - sel_bt["metrics"]["MAE"] / naive_bt["metrics"]["MAE"]) * 100, 2)
            if (sel_bt and naive_bt and naive_bt["metrics"]["MAE"]) else None),
        "selection_rule": (f"Lowest walk-forward MAE; a non-baseline model must beat the best naive "
                           f"baseline by at least {MIN_IMPROVEMENT:.0%}."),
        "selection_note": selection_note,
        "candidates": {name: bt["metrics"] for name, bt in backtests.items()},
        "interval_method": ("95% interval = forecast ± 1.96 × std of out-of-sample walk-forward errors "
                            "(approximate; assumes roughly normal, stable errors)."),
        "rows_input": n_input_rows, "rows_bad_time": n_bad_time, "rows_bad_target": n_bad_target,
        "aggregation": agg_note, "sampling": sampled_note,
    }

    # ------------------------------------------------------------------
    # Future dates
    # ------------------------------------------------------------------
    last_date = working_df[time_col].iloc[-1]
    future_dates = _generate_future_dates(last_date, horizon, frequency)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    feature_importance: list[FeatureImportance] = []
    if best_candidate.model_type == "linear_trend" and trend_coef is not None:
        feature_importance.append(
            FeatureImportance(feature="time_index", importance=trend_coef, rank=1)
        )

    # ------------------------------------------------------------------
    # Confidence score from validation R-squared
    # ------------------------------------------------------------------
    best_r2 = best_candidate.metrics.get("R2")
    # Clamp between 0 and 1 (holdout R²; undefined R² gives zero confidence)
    confidence_score = max(0.0, min(1.0, best_r2)) if best_r2 is not None else 0.0

    # ------------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------------
    limitations: list[str] = []
    if n_obs < 30:
        limitations.append(
            f"Only {n_obs} observations; forecasts may be unreliable."
        )
    if frequency == "irregular":
        limitations.append(
            "Irregular time intervals detected; forecasts assume regular spacing."
        )
    if len(val_vals) < 5:
        limitations.append(
            f"Validation set has only {len(val_vals)} point(s); "
            "error estimates may be imprecise."
        )
    if residual_std > 0 and best_mae > 0:
        mean_val = float(np.mean(np.abs(target_values)))
        if mean_val > 0 and best_mae / mean_val > 0.3:
            limitations.append(
                "Model error is high relative to the target's magnitude; "
                "treat predictions as rough estimates."
            )
    if agg_note:
        limitations.append(agg_note)
    if sampled_note:
        limitations.append(sampled_note)
    if n_bad_time or n_bad_target:
        limitations.append(f"{n_bad_time} row(s) without a valid date and {n_bad_target} row(s) without a "
                           f"valid '{target_col}' value were left out (never filled in).")
    if selection_note:
        limitations.append(selection_note)
    if evaluation["improvement_vs_baseline_pct"] is not None and evaluation["improvement_vs_baseline_pct"] <= 0 \
            and best_candidate.model_type in baseline_types:
        limitations.append("No model beat the naive baseline; the forecast is a simple carry-forward and "
                           "should be read as 'no detectable pattern'.")
    if not feature_importance:
        limitations.append(
            "Selected model does not produce feature-importance scores."
        )

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    r2_txt = f"{best_r2:.4f}" if best_r2 is not None else "n/a"
    imp = evaluation["improvement_vs_baseline_pct"]
    imp_txt = (f" {imp:+.1f}% MAE vs {evaluation['baseline']}." if imp is not None and evaluation["baseline"]
               and best_candidate.name != evaluation["baseline"] else "")
    explanation = (
        f"Trained {len(candidates)} model(s) on '{target_col}' over "
        f"{n_obs} observations ({frequency} frequency). "
        f"Best model: {selected_name} "
        f"(walk-forward MAE={best_mae:.4f} over {evaluation['folds']} fold(s), holdout R2={r2_txt}).{imp_txt} "
        f"Forecasting {horizon} step(s) ahead with approximate 95% prediction intervals."
    )

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    return ForecastResult(
        status="success",
        target_column=target_col,
        time_column=time_col,
        horizon=horizon,
        frequency=frequency,
        model_name=selected_name,
        predictions=preds.tolist(),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        future_dates=future_dates,
        metrics=best_candidate.metrics,
        confidence=confidence_score,
        model_candidates=candidates,
        feature_importance=feature_importance,
        limitations=limitations,
        explanation=explanation,
        evaluation=evaluation,
    )
