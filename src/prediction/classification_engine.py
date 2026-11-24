"""Phase 4 -- General Classification Engine.

Dataset-agnostic supervised classification: auto-detects categorical or
binary-numeric targets, trains multiple classifiers, evaluates with
stratified holdout, and returns a fully populated ClassificationResult.

Never hardcodes column names.
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction.contracts import (
    ClassificationResult,
    ModelCandidate,
    FeatureImportance,
)
from src.prediction.preprocessing import preprocess_for_prediction
from src.prediction.task_detector import (
    get_feature_columns,
    MAX_ROWS,
    MIN_ROWS_SUPERVISED,
    MAX_CLASSES,
)
from src.data_engine.canonical import AnalysisContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target auto-detection
# ---------------------------------------------------------------------------

def _detect_classification_target(context: AnalysisContext) -> str | None:
    """Find the best classification target column automatically.

    Looks for categorical columns with 2-20 unique values (excluding
    identifiers), then binary numeric columns (0/1).  Returns the column
    with lowest cardinality (most likely a label).
    """
    ds = context.dataset
    df = ds.dataframe
    if df is None or df.empty:
        return None

    identifier_set = set(ds.identifier_columns or [])
    candidates: list[tuple[str, int]] = []  # (column, cardinality)

    # Categorical columns with 2-MAX_CLASSES unique values
    for col in (ds.categorical_columns or []):
        if col in identifier_set:
            continue
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= MAX_CLASSES:
            candidates.append((col, nunique))

    # Boolean columns
    for col in (ds.boolean_columns or []):
        if col in identifier_set:
            continue
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= MAX_CLASSES:
            candidates.append((col, nunique))

    # Binary numeric columns (only 0/1)
    for col in (ds.numeric_columns or []):
        if col in identifier_set:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        unique_vals = set(non_null.unique())
        if unique_vals <= {0, 1, 0.0, 1.0}:
            candidates.append((col, len(unique_vals)))

    if not candidates:
        return None

    # Pick lowest cardinality
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_classification(
    context: AnalysisContext,
    target_column: str = None,
    feature_columns: list = None,
) -> ClassificationResult:
    """Run a complete classification pipeline on the dataset.

    Parameters
    ----------
    context : AnalysisContext
        The canonical analysis context wrapping the dataset.
    target_column : str, optional
        Column to predict.  Auto-detected if not provided.
    feature_columns : list, optional
        Feature columns to use.  Auto-detected if not provided.

    Returns
    -------
    ClassificationResult
        Full result with metrics, confusion matrix, feature importance,
        and model candidates.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    ds = context.dataset
    df = ds.dataframe

    # --- Guard: empty data ---
    if df is None or df.empty:
        return ClassificationResult(
            status="unavailable",
            explanation="No data available for classification.",
        )

    # --- 1. Resolve target column ---
    if target_column is None:
        target_column = _detect_classification_target(context)

    if target_column is None:
        return ClassificationResult(
            status="unavailable",
            explanation=(
                "No suitable classification target found. Need a categorical "
                "column with 2-20 unique values or a binary numeric column."
            ),
        )

    if target_column not in df.columns:
        return ClassificationResult(
            status="unavailable",
            target_column=target_column,
            explanation=f"Target column '{target_column}' not found in the dataset.",
        )

    # --- 2. Row-count guard ---
    if len(df) < MIN_ROWS_SUPERVISED:
        return ClassificationResult(
            status="insufficient_data",
            target_column=target_column,
            explanation=(
                f"Only {len(df)} rows available; classification requires "
                f"at least {MIN_ROWS_SUPERVISED}."
            ),
        )

    # --- 3. Class analysis ---
    y_raw = df[target_column].dropna()
    classes_unique = y_raw.unique()
    n_classes = len(classes_unique)

    if n_classes < 2:
        return ClassificationResult(
            status="insufficient_data",
            target_column=target_column,
            classes=[str(c) for c in classes_unique],
            explanation="Only 1 class found in the target; classification requires at least 2.",
        )

    if n_classes > MAX_CLASSES:
        return ClassificationResult(
            status="failed",
            target_column=target_column,
            explanation=(
                f"Target '{target_column}' has {n_classes} classes, exceeding "
                f"the maximum of {MAX_CLASSES}. Consider grouping rare classes "
                f"or using regression instead."
            ),
        )

    # --- 4. Get feature columns ---
    if feature_columns is None:
        feature_columns = get_feature_columns(
            context, target_column=target_column, task_type="classification"
        )

    if not feature_columns:
        return ClassificationResult(
            status="failed",
            target_column=target_column,
            explanation="No usable feature columns after excluding identifiers and constants.",
        )

    # --- 5. Preprocess ---
    try:
        X, y_series, final_features, prep_report = preprocess_for_prediction(
            df, target_column, feature_columns, task_type="classification"
        )
    except Exception as exc:
        logger.warning("Preprocessing failed: %s", exc)
        return ClassificationResult(
            status="failed",
            target_column=target_column,
            explanation=f"Preprocessing failed: {exc}",
        )

    if X.empty or len(final_features) == 0:
        return ClassificationResult(
            status="failed",
            target_column=target_column,
            explanation="0 features remain after preprocessing; cannot train classifiers.",
        )

    # --- 6. Encode target as integers ---
    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str))
    class_labels = [str(c) for c in le.classes_]
    n_classes = len(class_labels)
    is_binary = n_classes == 2

    # --- 7. Class imbalance check ---
    limitations: list[str] = []
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) >= 2:
        ratio = class_counts.max() / max(class_counts.min(), 1)
        if ratio > 10:
            limitations.append(
                f"Severe class imbalance detected (ratio {ratio:.0f}:1). "
                f"Results may be biased toward the majority class."
            )

    # --- 8. Train/test split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Stratified split can fail if a class has too few samples
        logger.info("Stratified split failed; falling back to non-stratified.")
        limitations.append("Stratified split failed due to small class sizes; used random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # --- 9. Define and train models ---
    model_defs = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ]

    candidates: list[ModelCandidate] = []
    trained_models: dict[str, object] = {}

    for name, model in model_defs:
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            # PART 06 performance: trees are built in parallel (identical trees for a fixed
            # random_state); prediction is sequential so the averaged outputs are bit-identical.
            if getattr(model, "n_jobs", None) == -1:
                model.set_params(n_jobs=None)
            y_pred = model.predict(X_test)

            metrics: dict[str, float] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            }

            # ROC AUC for binary classification only
            if is_binary:
                try:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        y_prob = model.decision_function(X_test)
                    else:
                        y_prob = None

                    if y_prob is not None:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                except Exception:
                    pass  # skip roc_auc if it fails

            candidate = ModelCandidate(
                name=name,
                model_type=name,
                metrics=metrics,
                score=metrics["f1"],
                training_time_seconds=elapsed,
                selected=False,
            )
            candidates.append(candidate)
            trained_models[name] = model

        except Exception as exc:
            logger.warning("Model %s failed: %s", name, exc)
            candidates.append(ModelCandidate(
                name=name,
                model_type=name,
                metrics={},
                score=0.0,
                training_time_seconds=time.time() - t0,
                selected=False,
            ))

    # --- 10. Select best model by F1 ---
    if not candidates or all(c.score == 0.0 for c in candidates):
        return ClassificationResult(
            status="failed",
            target_column=target_column,
            classes=class_labels,
            feature_columns=final_features,
            model_candidates=candidates,
            limitations=limitations,
            explanation="All models failed to train.",
            n_train=len(X_train),
            n_test=len(X_test),
        )

    best_candidate = max(candidates, key=lambda c: c.score)
    best_candidate.selected = True
    best_model = trained_models.get(best_candidate.name)

    # --- 11. Confusion matrix from best model ---
    cm: list[list[int]] = []
    if best_model is not None:
        y_pred_best = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best).tolist()

    # --- 12. Feature importance ---
    feature_importance: list[FeatureImportance] = []
    if best_model is not None:
        importances = None

        if hasattr(best_model, "feature_importances_"):
            # Tree-based models
            importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            # LogisticRegression: abs(coef_) averaged across classes for multiclass
            coef = best_model.coef_
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                importances = np.abs(coef).mean(axis=0)

        if importances is not None and len(importances) == len(final_features):
            # Sort by importance descending
            sorted_indices = np.argsort(importances)[::-1]
            for rank, idx in enumerate(sorted_indices, start=1):
                feature_importance.append(FeatureImportance(
                    feature=final_features[idx],
                    importance=float(importances[idx]),
                    rank=rank,
                ))

    # --- 13. Build result ---
    best_metrics = best_candidate.metrics
    confidence = best_candidate.score  # F1 as confidence

    explanation_parts = [
        f"Classification on '{target_column}' ({n_classes} classes: {', '.join(class_labels)}).",
        f"Best model: {best_candidate.name} with F1={best_candidate.score:.4f}.",
        f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows "
        f"using {len(final_features)} features.",
    ]
    if limitations:
        explanation_parts.append(f"Limitations: {'; '.join(limitations)}")

    return ClassificationResult(
        status="success",
        target_column=target_column,
        classes=class_labels,
        feature_columns=final_features,
        model_name=best_candidate.name,
        metrics=best_metrics,
        confusion_matrix=cm,
        feature_importance=feature_importance,
        model_candidates=candidates,
        confidence=confidence,
        limitations=limitations,
        explanation=" ".join(explanation_parts),
        n_train=len(X_train),
        n_test=len(X_test),
    )
