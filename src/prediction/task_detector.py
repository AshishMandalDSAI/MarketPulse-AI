"""Phase 4 -- Predictive Task Detection.

Inspects a CanonicalDataset (via AnalysisContext) and determines which
predictive tasks -- forecasting, regression, classification, segmentation --
are feasible, returning a ranked list of PredictiveTask objects with
confidence scores, feature lists, and human-readable explanations.

No hardcoded column names.  All decisions derive from the dataset's
structural metadata (column types, cardinality, row count, time info).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction.contracts import PredictiveTask
from src.data_engine.canonical import AnalysisContext, CanonicalDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource / threshold constants
# ---------------------------------------------------------------------------

MAX_ROWS = 100_000
MAX_FEATURES = 200
MAX_CARDINALITY_FOR_ENCODING = 50
MIN_ROWS_FORECAST = 10
MIN_ROWS_SUPERVISED = 20
MIN_ROWS_SEGMENTATION = 10
MIN_FEATURES_SEGMENTATION = 2
MAX_CLASSES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_time_attr(time_info, attr: str, default=None):
    """Access a TimeInfo attribute with dual dataclass/dict support."""
    if time_info is None:
        return default
    if hasattr(time_info, attr):
        return getattr(time_info, attr)
    if isinstance(time_info, dict):
        return time_info.get(attr, default)
    return default


def get_feature_columns(
    context: AnalysisContext,
    target_column: Optional[str] = None,
    task_type: str = "regression",
) -> list[str]:
    """Get valid feature columns excluding target, identifiers, and constants.

    Parameters
    ----------
    context : AnalysisContext
        The analysis context wrapping the canonical dataset.
    target_column : str | None
        Column to exclude as the prediction target.
    task_type : str
        One of "forecasting", "regression", "classification", "segmentation".
        For segmentation only numeric columns are returned.

    Returns
    -------
    list[str]
        Ordered list of usable feature column names.
    """
    ds = context.dataset

    identifier_set = set(ds.identifier_columns or [])
    exclude = set(identifier_set)
    if target_column:
        exclude.add(target_column)

    # Exclude constant columns
    constant_cols: set[str] = set()
    if ds.data_quality is not None:
        constant_cols = set(
            ds.data_quality.constant_columns
            if hasattr(ds.data_quality, "constant_columns")
            else (ds.data_quality.get("constant_columns", [])
                  if isinstance(ds.data_quality, dict) else [])
        )
    exclude |= constant_cols

    # Exclude empty columns
    empty_cols = set(ds.empty_columns or [])
    exclude |= empty_cols

    # Exclude the time column for non-forecasting tasks
    time_info = ds.time_info
    time_col = _safe_time_attr(time_info, "time_column")
    if time_col and task_type != "forecasting":
        exclude.add(time_col)

    if task_type == "segmentation":
        # Segmentation uses only numeric features
        features = [c for c in (ds.numeric_columns or []) if c not in exclude]
    else:
        # Supervised tasks: numeric + encodable categorical
        numeric_feats = [c for c in (ds.numeric_columns or []) if c not in exclude]

        cat_feats = []
        df = ds.dataframe
        for col in (ds.categorical_columns or []):
            if col in exclude:
                continue
            # Only include low-cardinality categoricals suitable for encoding
            if df is not None and col in df.columns:
                nunique = df[col].nunique(dropna=True)
                if nunique <= MAX_CARDINALITY_FOR_ENCODING:
                    cat_feats.append(col)
            else:
                cat_feats.append(col)

        features = numeric_feats + cat_feats

    # Cap at MAX_FEATURES
    if len(features) > MAX_FEATURES:
        features = features[:MAX_FEATURES]

    return features


# ---------------------------------------------------------------------------
# Individual task detectors
# ---------------------------------------------------------------------------

def _detect_forecasting(ds: CanonicalDataset) -> Optional[PredictiveTask]:
    """Detect whether a time-series forecasting task is feasible."""
    time_info = ds.time_info
    time_col = _safe_time_attr(time_info, "time_column")

    if not time_col:
        return PredictiveTask(
            task_type="forecasting",
            confidence=0.0,
            reason="No time column detected in the dataset.",
            requirements_met=False,
            limitations=["Dataset has no datetime column for temporal ordering."],
        )

    # Need at least one numeric target candidate
    numeric_targets = [
        tc for tc in (ds.target_candidates or [])
        if hasattr(tc, "column") and tc.column in (ds.numeric_columns or [])
    ]

    if not numeric_targets:
        return PredictiveTask(
            task_type="forecasting",
            target_column=None,
            confidence=0.0,
            reason="Time column found but no numeric target candidate available.",
            requirements_met=False,
            limitations=["No numeric column qualifies as a forecast target."],
        )

    if ds.row_count < MIN_ROWS_FORECAST:
        return PredictiveTask(
            task_type="forecasting",
            target_column=numeric_targets[0].column,
            confidence=0.0,
            reason=(
                f"Only {ds.row_count} rows; forecasting needs "
                f"at least {MIN_ROWS_FORECAST}."
            ),
            requirements_met=False,
            limitations=[f"Row count ({ds.row_count}) below minimum ({MIN_ROWS_FORECAST})."],
        )

    # Requirements met -- compute confidence
    best_target = numeric_targets[0]
    is_regular = bool(_safe_time_attr(time_info, "is_regular_time_series", False))
    granularity = _safe_time_attr(time_info, "time_granularity", "irregular")
    time_range = _safe_time_attr(time_info, "time_range")
    n_periods = time_range.get("periods", ds.row_count) if isinstance(time_range, dict) else ds.row_count

    confidence = 0.0

    # Regularity bonus
    if is_regular:
        confidence += 0.30
    elif granularity and granularity != "irregular":
        confidence += 0.15

    # Observation count bonus (more data = more confidence, up to 0.30)
    obs_ratio = min(n_periods / 100, 1.0)
    confidence += obs_ratio * 0.30

    # Target quality bonus
    confidence += min(best_target.score, 1.0) * 0.25

    # Known granularity bonus
    if granularity and granularity not in ("irregular", None):
        confidence += 0.10

    confidence = round(min(confidence, 1.0), 4)

    limitations = []
    if not is_regular:
        limitations.append("Time series has irregular intervals; resampling may be needed.")
    if n_periods < 30:
        limitations.append(f"Only {n_periods} time periods; forecasts may be unreliable.")
    if ds.row_count > MAX_ROWS:
        limitations.append(
            f"Dataset has {ds.row_count} rows; will be sampled to {MAX_ROWS} for training."
        )

    return PredictiveTask(
        task_type="forecasting",
        target_column=best_target.column,
        feature_columns=[],  # populated later by the caller / runner
        confidence=confidence,
        reason=(
            f"Time column '{time_col}' with {n_periods} periods "
            f"({granularity or 'unknown'} granularity). "
            f"Target: '{best_target.column}' (score {best_target.score:.2f})."
        ),
        requirements_met=True,
        limitations=limitations,
    )


def _detect_regression(
    ds: CanonicalDataset, context: AnalysisContext
) -> Optional[PredictiveTask]:
    """Detect whether a regression task is feasible."""
    numeric_targets = [
        tc for tc in (ds.target_candidates or [])
        if hasattr(tc, "column") and tc.column in (ds.numeric_columns or [])
    ]

    if not numeric_targets:
        return PredictiveTask(
            task_type="regression",
            confidence=0.0,
            reason="No numeric target candidate detected.",
            requirements_met=False,
            limitations=["No numeric column qualifies as a regression target."],
        )

    best_target = numeric_targets[0]
    features = get_feature_columns(context, target_column=best_target.column, task_type="regression")

    # Need at least one numeric feature (excluding the target)
    numeric_features = [f for f in features if f in (ds.numeric_columns or [])]
    if not numeric_features:
        return PredictiveTask(
            task_type="regression",
            target_column=best_target.column,
            confidence=0.0,
            reason="Target found but no numeric feature columns available after exclusions.",
            requirements_met=False,
            limitations=["No usable numeric predictor columns."],
        )

    if ds.row_count < MIN_ROWS_SUPERVISED:
        return PredictiveTask(
            task_type="regression",
            target_column=best_target.column,
            feature_columns=features,
            confidence=0.0,
            reason=(
                f"Only {ds.row_count} rows; regression needs "
                f"at least {MIN_ROWS_SUPERVISED}."
            ),
            requirements_met=False,
            limitations=[f"Row count ({ds.row_count}) below minimum ({MIN_ROWS_SUPERVISED})."],
        )

    # Requirements met -- compute confidence
    confidence = 0.0

    # Target quality
    confidence += min(best_target.score, 1.0) * 0.30

    # Feature count bonus (more features up to a point = better)
    feat_ratio = min(len(features) / 10, 1.0)
    confidence += feat_ratio * 0.25

    # Row count bonus
    row_ratio = min(ds.row_count / 200, 1.0)
    confidence += row_ratio * 0.25

    # Data quality bonus
    if ds.data_quality is not None:
        q_score = (
            ds.data_quality.score
            if hasattr(ds.data_quality, "score")
            else (ds.data_quality.get("score", 0.5) if isinstance(ds.data_quality, dict) else 0.5)
        )
        confidence += q_score * 0.15

    confidence = round(min(confidence, 1.0), 4)

    limitations = []
    if ds.row_count > MAX_ROWS:
        limitations.append(
            f"Dataset has {ds.row_count} rows; will be sampled to {MAX_ROWS}."
        )
    if len(features) > 50:
        limitations.append(
            f"{len(features)} feature columns; dimensionality reduction may help."
        )

    return PredictiveTask(
        task_type="regression",
        target_column=best_target.column,
        feature_columns=features,
        confidence=confidence,
        reason=(
            f"Target: '{best_target.column}' (score {best_target.score:.2f}) "
            f"with {len(features)} feature column(s) and {ds.row_count} rows."
        ),
        requirements_met=True,
        limitations=limitations,
    )


def _detect_classification(
    ds: CanonicalDataset, context: AnalysisContext
) -> Optional[PredictiveTask]:
    """Detect whether a classification task is feasible."""
    df = ds.dataframe
    if df is None or df.empty:
        return PredictiveTask(
            task_type="classification",
            confidence=0.0,
            reason="No data available.",
            requirements_met=False,
            limitations=["DataFrame is empty or missing."],
        )

    # Find classification target candidates:
    # 1. Categorical columns with 2-MAX_CLASSES unique values (non-identifier)
    # 2. Binary numeric columns (only 0/1 values)
    identifier_set = set(ds.identifier_columns or [])
    candidates: list[tuple[str, int, str]] = []  # (column, cardinality, source)

    for col in (ds.categorical_columns or []):
        if col in identifier_set:
            continue
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= MAX_CLASSES:
            candidates.append((col, nunique, "categorical"))

    for col in (ds.numeric_columns or []):
        if col in identifier_set:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        unique_vals = set(non_null.unique())
        if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
            candidates.append((col, len(unique_vals), "binary_numeric"))

    # Also check boolean columns
    for col in (ds.boolean_columns or []):
        if col in identifier_set:
            continue
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= MAX_CLASSES:
            candidates.append((col, nunique, "boolean"))

    if not candidates:
        return PredictiveTask(
            task_type="classification",
            confidence=0.0,
            reason="No suitable classification target found (need categorical with 2-20 classes).",
            requirements_met=False,
            limitations=["No column has the right cardinality for classification."],
        )

    # Pick the best candidate: prefer lower cardinality, prefer categorical source
    def _sort_key(item):
        col, card, source = item
        source_rank = {"binary_numeric": 0, "boolean": 1, "categorical": 2}
        return (source_rank.get(source, 3), card)

    candidates.sort(key=_sort_key)
    best_col, best_card, best_source = candidates[0]

    features = get_feature_columns(context, target_column=best_col, task_type="classification")

    if not features:
        return PredictiveTask(
            task_type="classification",
            target_column=best_col,
            confidence=0.0,
            reason=f"Target '{best_col}' found but no feature columns available.",
            requirements_met=False,
            limitations=["No usable predictor columns after exclusions."],
        )

    if ds.row_count < MIN_ROWS_SUPERVISED:
        return PredictiveTask(
            task_type="classification",
            target_column=best_col,
            feature_columns=features,
            confidence=0.0,
            reason=(
                f"Only {ds.row_count} rows; classification needs "
                f"at least {MIN_ROWS_SUPERVISED}."
            ),
            requirements_met=False,
            limitations=[f"Row count ({ds.row_count}) below minimum ({MIN_ROWS_SUPERVISED})."],
        )

    # Requirements met -- compute confidence
    confidence = 0.0

    # Binary is easier / more reliable
    if best_card == 2:
        confidence += 0.30
    elif best_card <= 5:
        confidence += 0.20
    else:
        confidence += 0.10

    # Feature count
    feat_ratio = min(len(features) / 10, 1.0)
    confidence += feat_ratio * 0.20

    # Row count
    row_ratio = min(ds.row_count / 200, 1.0)
    confidence += row_ratio * 0.25

    # Rows per class -- need reasonable balance
    min_class_size = df[best_col].value_counts(dropna=True).min()
    if min_class_size >= 10:
        confidence += 0.15
    elif min_class_size >= 5:
        confidence += 0.08

    # Data quality
    if ds.data_quality is not None:
        q_score = (
            ds.data_quality.score
            if hasattr(ds.data_quality, "score")
            else (ds.data_quality.get("score", 0.5) if isinstance(ds.data_quality, dict) else 0.5)
        )
        confidence += q_score * 0.10

    confidence = round(min(confidence, 1.0), 4)

    limitations = []
    if min_class_size < 10:
        limitations.append(
            f"Smallest class has only {min_class_size} samples; results may be unreliable."
        )
    if best_card > 10:
        limitations.append(f"{best_card} classes detected; multi-class performance may vary.")
    if ds.row_count > MAX_ROWS:
        limitations.append(
            f"Dataset has {ds.row_count} rows; will be sampled to {MAX_ROWS}."
        )

    return PredictiveTask(
        task_type="classification",
        target_column=best_col,
        feature_columns=features,
        confidence=confidence,
        reason=(
            f"Target: '{best_col}' ({best_card} classes, source: {best_source}) "
            f"with {len(features)} feature column(s) and {ds.row_count} rows."
        ),
        requirements_met=True,
        limitations=limitations,
    )


def _detect_segmentation(
    ds: CanonicalDataset, context: AnalysisContext
) -> Optional[PredictiveTask]:
    """Detect whether a segmentation/clustering task is feasible."""
    features = get_feature_columns(context, target_column=None, task_type="segmentation")

    if len(features) < MIN_FEATURES_SEGMENTATION:
        return PredictiveTask(
            task_type="segmentation",
            confidence=0.0,
            reason=(
                f"Only {len(features)} numeric non-identifier column(s); "
                f"segmentation needs at least {MIN_FEATURES_SEGMENTATION}."
            ),
            requirements_met=False,
            limitations=["Insufficient numeric features for meaningful clustering."],
        )

    if ds.row_count < MIN_ROWS_SEGMENTATION:
        return PredictiveTask(
            task_type="segmentation",
            feature_columns=features,
            confidence=0.0,
            reason=(
                f"Only {ds.row_count} rows; segmentation needs "
                f"at least {MIN_ROWS_SEGMENTATION}."
            ),
            requirements_met=False,
            limitations=[f"Row count ({ds.row_count}) below minimum ({MIN_ROWS_SEGMENTATION})."],
        )

    # Requirements met -- compute confidence
    confidence = 0.0

    # Feature count bonus
    feat_ratio = min(len(features) / 8, 1.0)
    confidence += feat_ratio * 0.30

    # Row count bonus
    row_ratio = min(ds.row_count / 100, 1.0)
    confidence += row_ratio * 0.30

    # Data quality bonus
    if ds.data_quality is not None:
        q_score = (
            ds.data_quality.score
            if hasattr(ds.data_quality, "score")
            else (ds.data_quality.get("score", 0.5) if isinstance(ds.data_quality, dict) else 0.5)
        )
        confidence += q_score * 0.20

    # Variance check -- at least some features should have meaningful spread
    df = ds.dataframe
    if df is not None:
        varying = 0
        for col in features[:20]:  # check up to 20
            if col in df.columns:
                std = df[col].std(skipna=True)
                mean = df[col].mean(skipna=True)
                if pd.notna(std) and pd.notna(mean) and abs(mean) > 1e-9:
                    cv = std / abs(mean)
                    if cv > 0.05:
                        varying += 1
        if features:
            var_ratio = varying / min(len(features), 20)
            confidence += var_ratio * 0.20

    confidence = round(min(confidence, 1.0), 4)

    limitations = []
    if ds.row_count > MAX_ROWS:
        limitations.append(
            f"Dataset has {ds.row_count} rows; will be sampled to {MAX_ROWS}."
        )
    if len(features) > 50:
        limitations.append(
            f"{len(features)} features; dimensionality reduction recommended."
        )

    return PredictiveTask(
        task_type="segmentation",
        target_column=None,
        feature_columns=features,
        confidence=confidence,
        reason=(
            f"{len(features)} numeric feature column(s) across {ds.row_count} rows "
            f"available for clustering."
        ),
        requirements_met=True,
        limitations=limitations,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_predictive_tasks(context: AnalysisContext) -> list[PredictiveTask]:
    """Detect which predictive tasks are feasible for the given dataset.

    Examines the CanonicalDataset wrapped by *context* and returns a list of
    PredictiveTask objects -- one per task type -- sorted by confidence
    (highest first).  Tasks whose requirements are not met still appear in the
    list with ``requirements_met=False`` so callers can explain why a task is
    unavailable.

    Parameters
    ----------
    context : AnalysisContext
        The analysis context produced by Phase 2's canonical pipeline.

    Returns
    -------
    list[PredictiveTask]
        Detected tasks, sorted by confidence descending.
    """
    ds = context.dataset
    tasks: list[PredictiveTask] = []

    # Run each detector; they always return a PredictiveTask (possibly unmet)
    try:
        task = _detect_forecasting(ds)
        if task is not None:
            tasks.append(task)
    except Exception as exc:
        logger.warning("Forecasting detection failed: %s", exc)
        tasks.append(PredictiveTask(
            task_type="forecasting", confidence=0.0,
            reason=f"Detection error: {exc}", requirements_met=False,
        ))

    try:
        task = _detect_regression(ds, context)
        if task is not None:
            tasks.append(task)
    except Exception as exc:
        logger.warning("Regression detection failed: %s", exc)
        tasks.append(PredictiveTask(
            task_type="regression", confidence=0.0,
            reason=f"Detection error: {exc}", requirements_met=False,
        ))

    try:
        task = _detect_classification(ds, context)
        if task is not None:
            tasks.append(task)
    except Exception as exc:
        logger.warning("Classification detection failed: %s", exc)
        tasks.append(PredictiveTask(
            task_type="classification", confidence=0.0,
            reason=f"Detection error: {exc}", requirements_met=False,
        ))

    try:
        task = _detect_segmentation(ds, context)
        if task is not None:
            tasks.append(task)
    except Exception as exc:
        logger.warning("Segmentation detection failed: %s", exc)
        tasks.append(PredictiveTask(
            task_type="segmentation", confidence=0.0,
            reason=f"Detection error: {exc}", requirements_met=False,
        ))

    # Sort: requirements_met first, then by confidence descending
    tasks.sort(key=lambda t: (t.requirements_met, t.confidence), reverse=True)

    return tasks
