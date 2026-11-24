"""Phase 5 — Categorical Anomaly Detection.

Dataset-agnostic anomaly detection for categorical columns.  Detects rare
categories, categories with unusual numeric metrics, and columns with
anomalous missingness.  Consumes Phase 2 CanonicalDataset / AnalysisContext
contracts — no hardcoded column names.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.contracts import Anomaly, AnomalySeverity, AnomalyType
from src.data_engine.canonical import AnalysisContext, CanonicalDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------
MAX_CATEGORICAL_COLS = 20
MAX_CATEGORIES_PER_COL = 100
_MIN_OBS = 5
_MAX_MEASURE_COLS = 3
_MIN_CATEGORIES_FOR_METRIC = 3
_MIN_OBS_PER_CATEGORY = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_categorical_columns(canonical: CanonicalDataset) -> list[str]:
    """Return categorical columns eligible for analysis, respecting limits."""
    candidates: list[str] = []

    # Prefer dimension_candidates; fall back to is_categorical metadata
    if canonical.dimension_candidates:
        candidates = list(canonical.dimension_candidates)
    else:
        for name, meta in canonical.column_metadata.items():
            if meta.is_categorical:
                candidates.append(name)

    return candidates[:MAX_CATEGORICAL_COLS]


def _should_skip_column(
    col: str,
    series: pd.Series,
    canonical: CanonicalDataset,
) -> bool:
    """Return True if the column should be skipped for categorical analysis."""
    meta = canonical.column_metadata.get(col)

    # Skip identifier columns
    if meta and meta.is_identifier:
        return True

    # Skip columns with < 5 non-null observations
    non_null = series.dropna()
    if len(non_null) < _MIN_OBS:
        return True

    # Skip high-cardinality columns
    if non_null.nunique() > MAX_CATEGORIES_PER_COL:
        return True

    return False


def _get_measure_columns(canonical: CanonicalDataset, df: pd.DataFrame) -> list[str]:
    """Return up to _MAX_MEASURE_COLS measure columns for metric analysis."""
    candidates = list(canonical.measure_candidates) if canonical.measure_candidates else []
    # Keep only columns actually present and numeric in the DataFrame
    valid = [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return valid[:_MAX_MEASURE_COLS]


# ---------------------------------------------------------------------------
# Detection: rare categories
# ---------------------------------------------------------------------------

def _detect_rare_categories(
    col: str,
    series: pd.Series,
    dataset_id: str,
) -> list[Anomaly]:
    """Flag categories whose frequency is < 1% AND count < 3, but only when
    at least 5 other categories have significantly higher frequency."""
    anomalies: list[Anomaly] = []
    non_null = series.dropna()
    if len(non_null) == 0:
        return anomalies

    counts = non_null.value_counts()
    total = counts.sum()
    n_categories = len(counts)

    if n_categories < 2:
        return anomalies

    avg_pct = 100.0 / n_categories  # uniform average

    for value, count in counts.items():
        pct = 100.0 * count / total

        # Must be < 1% AND count < 3
        if pct >= 1.0 or count >= 3:
            continue

        # Only flag if at least 5 other categories have significantly higher
        # frequency (> 2x this category's frequency)
        higher_count = (counts.drop(value) > count * 2).sum()
        if higher_count < 5:
            continue

        score = max(0.0, min(1.0, 1.0 - pct / avg_pct)) if avg_pct > 0 else 0.0

        anomalies.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            anomaly_type=AnomalyType.CATEGORY_ANOMALY.value,
            severity=AnomalySeverity.LOW.value,
            score=round(score, 4),
            method="rare_category",
            observed_value=str(value),
            expected_value=f"avg frequency {avg_pct:.1f}%",
            deviation=round(avg_pct - pct, 4) if avg_pct > 0 else None,
            explanation=(
                f"Category '{value}' in column '{col}' appears only {count} "
                f"time(s) ({pct:.1f}%), significantly less than the average "
                f"category frequency of {avg_pct:.1f}%."
            ),
        ))

    return anomalies


# ---------------------------------------------------------------------------
# Detection: unusual numeric metric per category
# ---------------------------------------------------------------------------

def _detect_unusual_metric(
    dim_col: str,
    measure_col: str,
    df: pd.DataFrame,
    dataset_id: str,
) -> list[Anomaly]:
    """Flag categories whose mean for a measure column deviates > 2 std from
    the overall category-level mean distribution."""
    anomalies: list[Anomaly] = []

    subset = df[[dim_col, measure_col]].dropna()
    if len(subset) == 0:
        return anomalies

    grouped = subset.groupby(dim_col)[measure_col]

    # Filter to categories with enough observations
    sizes = grouped.size()
    valid_cats = sizes[sizes >= _MIN_OBS_PER_CATEGORY].index
    if len(valid_cats) < _MIN_CATEGORIES_FOR_METRIC:
        return anomalies

    means = grouped.mean().loc[valid_cats]
    overall_mean = means.mean()
    overall_std = means.std(ddof=1)

    if overall_std == 0 or np.isnan(overall_std):
        return anomalies

    for cat, cat_mean in means.items():
        z = abs(cat_mean - overall_mean) / overall_std
        if z < 2.0:
            continue

        direction = "high" if cat_mean > overall_mean else "low"

        if z >= 3.0:
            severity = AnomalySeverity.HIGH.value
        else:
            severity = AnomalySeverity.MEDIUM.value

        score = min(1.0, z / 5.0)

        anomalies.append(Anomaly(
            dataset_id=dataset_id,
            column=dim_col,
            anomaly_type=AnomalyType.CATEGORY_ANOMALY.value,
            severity=severity,
            score=round(score, 4),
            method="category_metric_deviation",
            observed_value=round(cat_mean, 2),
            expected_value=round(overall_mean, 2),
            deviation=round(z, 4),
            explanation=(
                f"Category '{cat}' in column '{dim_col}' has an unusually "
                f"{direction} average {measure_col} of {cat_mean:.2f} compared "
                f"to the overall mean of {overall_mean:.2f}."
            ),
        ))

    return anomalies


# ---------------------------------------------------------------------------
# Detection: missingness anomalies
# ---------------------------------------------------------------------------

def _detect_missingness_anomalies(
    df: pd.DataFrame,
    canonical: CanonicalDataset,
    dataset_id: str,
) -> list[Anomaly]:
    """Flag columns whose missing-value rate is disproportionately high
    compared to the dataset average.  Covers both categorical and numeric
    columns."""
    anomalies: list[Anomaly] = []
    n_rows = len(df)
    if n_rows == 0:
        return anomalies

    col_missing_pcts: dict[str, float] = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        col_missing_pcts[col] = 100.0 * missing_count / n_rows

    avg_missing = np.mean(list(col_missing_pcts.values())) if col_missing_pcts else 0.0

    # Only flag when the dataset average is reasonably low (< 10%)
    if avg_missing >= 10.0:
        return anomalies

    for col, pct in col_missing_pcts.items():
        if pct <= 20.0:
            continue

        # Skip identifier columns
        meta = canonical.column_metadata.get(col)
        if meta and meta.is_identifier:
            continue

        if pct > 50.0:
            severity = AnomalySeverity.HIGH.value
            score = min(1.0, pct / 100.0)
        else:
            severity = AnomalySeverity.MEDIUM.value
            score = pct / 100.0

        anomalies.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            anomaly_type=AnomalyType.MISSINGNESS_ANOMALY.value,
            severity=severity,
            score=round(score, 4),
            method="missingness_rate",
            observed_value=f"{pct:.1f}%",
            expected_value=f"dataset avg {avg_missing:.1f}%",
            deviation=round(pct - avg_missing, 4),
            explanation=(
                f"Column '{col}' has {pct:.1f}% missing values, significantly "
                f"higher than the dataset average of {avg_missing:.1f}%."
            ),
        ))

    return anomalies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_categorical_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 30,
) -> list[Anomaly]:
    """Run all categorical anomaly detection methods and return findings.

    Parameters
    ----------
    context : AnalysisContext
        The canonical analysis context wrapping the dataset.
    dataset_id : str
        Unique identifier for the dataset being analyzed.
    max_anomalies : int
        Maximum number of anomalies to return (most severe first).

    Returns
    -------
    list[Anomaly]
        Detected anomalies sorted by severity (CRITICAL > HIGH > MEDIUM > LOW),
        then by score descending.
    """
    canonical = context.dataset
    df = context.get_dataframe()

    # Edge case: no dataframe or empty
    if df is None or df.empty:
        return []

    anomalies: list[Anomaly] = []

    # --- Categorical column analysis ---
    cat_cols = _get_categorical_columns(canonical)
    measure_cols = _get_measure_columns(canonical, df)

    for col in cat_cols:
        if col not in df.columns:
            continue

        series = df[col]
        if _should_skip_column(col, series, canonical):
            continue

        # (a) Rare category detection
        anomalies.extend(_detect_rare_categories(col, series, dataset_id))

        # (b) Category with unusual numeric metric
        for measure_col in measure_cols:
            anomalies.extend(
                _detect_unusual_metric(col, measure_col, df, dataset_id)
            )

    # --- Missingness anomalies (all columns, not just categorical) ---
    anomalies.extend(_detect_missingness_anomalies(df, canonical, dataset_id))

    # --- Sort by severity (descending) then score (descending) ---
    severity_order = {
        AnomalySeverity.CRITICAL.value: 0,
        AnomalySeverity.HIGH.value: 1,
        AnomalySeverity.MEDIUM.value: 2,
        AnomalySeverity.LOW.value: 3,
    }
    anomalies.sort(key=lambda a: (severity_order.get(a.severity, 99), -a.score))

    return anomalies[:max_anomalies]
