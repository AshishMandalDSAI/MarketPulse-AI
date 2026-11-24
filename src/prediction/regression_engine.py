"""Phase 4 -- General Regression Engine.

Trains multiple regression models on a dataset-agnostic basis, selects
the best performer, and returns a fully-populated RegressionResult with
predictions, residuals, feature importance, and model candidates.

Never hardcodes column names.  Confidence derives from actual R-squared,
never fabricated.
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction.contracts import RegressionResult, ModelCandidate, FeatureImportance
from src.prediction.preprocessing import preprocess_for_prediction
from src.prediction.task_detector import get_feature_columns, MAX_ROWS, MIN_ROWS_SUPERVISED
from src.data_engine.canonical import AnalysisContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, safe against zero actuals."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R-squared, and MAPE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = _compute_mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def _extract_feature_importance(
    model, model_name: str, feature_names: list[str],
) -> list[FeatureImportance]:
    """Extract and rank feature importances from a fitted model."""
    if model_name == "Ridge":
        raw = np.abs(model.coef_)
    else:
        # Tree-based models expose .feature_importances_
        raw = model.feature_importances_

    # Normalise so importances sum to 1
    total = raw.sum()
    if total > 0:
        normalised = raw / total
    else:
        normalised = raw

    # Pair with names, sort descending
    pairs = sorted(
        zip(feature_names, normalised),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        FeatureImportance(feature=name, importance=float(imp), rank=rank + 1)
        for rank, (name, imp) in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_regression(
    context: AnalysisContext,
    target_column: str = None,
    feature_columns: list = None,
) -> RegressionResult:
    """Train baseline regression models, select the best, and return results.

    Parameters
    ----------
    context : AnalysisContext
        Wraps the canonical dataset produced by Phase 2.
    target_column : str | None
        Column to predict.  If *None*, the first target candidate is used.
    feature_columns : list[str] | None
        Predictor columns.  If *None*, auto-detected via task_detector.

    Returns
    -------
    RegressionResult
        Fully populated result contract.
    """
    ds = context.dataset
    df = ds.dataframe

    # ------------------------------------------------------------------
    # 1. Target selection
    # ------------------------------------------------------------------
    if target_column is None:
        candidates = ds.target_candidates or []
        if not candidates:
            return RegressionResult(
                status="unavailable",
                explanation="No target candidates detected in the dataset.",
                limitations=["Cannot run regression without a numeric target column."],
            )
        target_column = candidates[0].column

    # Validate target exists and is numeric
    if df is None or df.empty:
        return RegressionResult(
            status="unavailable",
            target_column=target_column,
            explanation="Dataset is empty.",
            limitations=["No data available for regression."],
        )

    if target_column not in df.columns:
        return RegressionResult(
            status="unavailable",
            target_column=target_column,
            explanation=f"Target column '{target_column}' not found in dataset.",
            limitations=[f"Column '{target_column}' does not exist."],
        )

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return RegressionResult(
            status="unavailable",
            target_column=target_column,
            explanation=f"Target column '{target_column}' is not numeric.",
            limitations=["Regression requires a numeric target column."],
        )

    # ------------------------------------------------------------------
    # 2. Row-count gate
    # ------------------------------------------------------------------
    row_count = len(df)
    if row_count < MIN_ROWS_SUPERVISED:
        return RegressionResult(
            status="insufficient_data",
            target_column=target_column,
            explanation=(
                f"Only {row_count} rows available; regression requires "
                f"at least {MIN_ROWS_SUPERVISED}."
            ),
            limitations=[f"Row count ({row_count}) below minimum ({MIN_ROWS_SUPERVISED})."],
        )

    # ------------------------------------------------------------------
    # 3. Sample down if above MAX_ROWS
    # ------------------------------------------------------------------
    work_df = df
    limitations: list[str] = []
    if row_count > MAX_ROWS:
        work_df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        limitations.append(
            f"Dataset sampled from {row_count} to {MAX_ROWS} rows for training."
        )

    # ------------------------------------------------------------------
    # 4. Feature selection
    # ------------------------------------------------------------------
    if feature_columns is None:
        feature_columns = get_feature_columns(context, target_column, "regression")

    if not feature_columns:
        return RegressionResult(
            status="failed",
            target_column=target_column,
            explanation="No usable feature columns detected after filtering.",
            limitations=["All columns were excluded (identifiers, constants, or non-numeric)."],
        )

    # ------------------------------------------------------------------
    # 5. Preprocessing
    # ------------------------------------------------------------------
    try:
        X, y, feature_names, prep_report = preprocess_for_prediction(
            work_df, target_column, feature_columns, "regression",
        )
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        return RegressionResult(
            status="failed",
            target_column=target_column,
            explanation=f"Preprocessing error: {exc}",
            limitations=["Data could not be prepared for modelling."],
        )

    if X.empty or len(feature_names) == 0:
        return RegressionResult(
            status="failed",
            target_column=target_column,
            explanation="Preprocessing produced zero usable features.",
            limitations=["No features remained after cleaning and encoding."],
        )

    # ------------------------------------------------------------------
    # 6. Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    n_train = len(X_train)
    n_test = len(X_test)

    # ------------------------------------------------------------------
    # 7. Model training and evaluation
    # ------------------------------------------------------------------
    model_specs = [
        ("Ridge", "Ridge", Ridge(alpha=1.0)),
        ("RandomForest", "RandomForestRegressor",
         RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("GradientBoosting", "GradientBoostingRegressor",
         GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ]

    candidates: list[ModelCandidate] = []
    trained_models: dict[str, object] = {}
    candidate_preds: dict[str, np.ndarray] = {}

    for name, model_type, model in model_specs:
        try:
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0
            # PART 06 performance: parallel tree building, sequential (bit-identical) prediction.
            if getattr(model, "n_jobs", None) == -1:
                model.set_params(n_jobs=None)

            preds = model.predict(X_test)
            metrics = _regression_metrics(y_test.values, preds)

            candidates.append(ModelCandidate(
                name=name,
                model_type=model_type,
                metrics=metrics,
                score=metrics["R2"],
                training_time_seconds=round(elapsed, 4),
                selected=False,
            ))
            trained_models[name] = model
            candidate_preds[name] = preds

        except Exception as exc:
            logger.warning("Model '%s' failed: %s", name, exc)
            candidates.append(ModelCandidate(
                name=name,
                model_type=model_type,
                metrics={},
                score=float("-inf"),
                training_time_seconds=0.0,
                selected=False,
            ))

    # ------------------------------------------------------------------
    # 8. Model selection
    # ------------------------------------------------------------------
    valid = [c for c in candidates if c.metrics]
    if not valid:
        return RegressionResult(
            status="failed",
            target_column=target_column,
            feature_columns=list(feature_names),
            explanation="All models failed during training.",
            limitations=["No model could be fitted to this data."],
            model_candidates=candidates,
            n_train=n_train,
            n_test=n_test,
        )

    # Prefer highest R2; fall back to lowest MAE when all R2 are negative
    all_negative_r2 = all(c.score < 0 for c in valid)
    if all_negative_r2:
        best = min(valid, key=lambda c: c.metrics.get("MAE", float("inf")))
    else:
        best = max(valid, key=lambda c: c.score)

    best.selected = True
    best_model = trained_models[best.name]
    best_preds = candidate_preds[best.name]
    best_metrics = best.metrics

    # ------------------------------------------------------------------
    # 9. Feature importance
    # ------------------------------------------------------------------
    try:
        fi = _extract_feature_importance(best_model, best.name, feature_names)
    except Exception as exc:
        logger.warning("Feature importance extraction failed: %s", exc)
        fi = []

    # ------------------------------------------------------------------
    # 10. Predictions, actuals, residuals (capped at 200)
    # ------------------------------------------------------------------
    actuals = y_test.values
    residuals = actuals - best_preds
    cap = 200

    pred_list = [float(v) for v in best_preds[:cap]]
    actual_list = [float(v) for v in actuals[:cap]]
    resid_list = [float(v) for v in residuals[:cap]]

    # ------------------------------------------------------------------
    # 11. Confidence from actual R-squared
    # ------------------------------------------------------------------
    r2_val = best_metrics.get("R2", 0.0)
    confidence = max(0.0, min(r2_val, 1.0))

    # ------------------------------------------------------------------
    # 12. Limitations
    # ------------------------------------------------------------------
    if prep_report.get("leakage_columns_removed"):
        limitations.append(
            f"Removed {len(prep_report['leakage_columns_removed'])} "
            f"near-duplicate/leakage column(s)."
        )
    if prep_report.get("high_cardinality_dropped"):
        limitations.append(
            f"Dropped {len(prep_report['high_cardinality_dropped'])} "
            f"high-cardinality categorical column(s)."
        )
    if all_negative_r2:
        limitations.append(
            "All models produced negative R-squared; predictions may be worse than "
            "the mean baseline."
        )
    # Check for constant predictions
    if len(set(pred_list)) <= 1:
        limitations.append(
            "Best model produced constant predictions -- it may not have learned "
            "meaningful patterns."
        )

    # ------------------------------------------------------------------
    # 13. Explanation
    # ------------------------------------------------------------------
    explanation = (
        f"Trained {len(model_specs)} regression models on '{target_column}' "
        f"using {len(feature_names)} feature(s).  "
        f"Best model: {best.name} "
        f"(R2={best_metrics['R2']:.4f}, MAE={best_metrics['MAE']:.4f}, "
        f"RMSE={best_metrics['RMSE']:.4f}).  "
        f"Train set: {n_train} rows, test set: {n_test} rows."
    )

    # ------------------------------------------------------------------
    # 14. Build and return result
    # ------------------------------------------------------------------
    return RegressionResult(
        status="success",
        target_column=target_column,
        feature_columns=list(feature_names),
        model_name=best.name,
        metrics=best_metrics,
        predictions=pred_list,
        actuals=actual_list,
        residuals=resid_list,
        feature_importance=fi,
        model_candidates=candidates,
        confidence=round(confidence, 4),
        limitations=limitations,
        explanation=explanation,
        n_train=n_train,
        n_test=n_test,
    )
