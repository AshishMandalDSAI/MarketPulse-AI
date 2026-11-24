"""Phase 5 — Numeric Anomaly Detection.

Dataset-agnostic numeric outlier detection using IQR, Z-score, and
Modified Z-score (MAD) methods.  Consumes the Phase 2
CanonicalDataset / AnalysisContext contracts — no hardcoded column names.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

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
MAX_ROWS_ANALYZE = 50_000
MAX_NUMERIC_COLS = 50
_MIN_OBS_IQR = 10
_MIN_OBS_ZSCORE = 30
_MIN_OBS_MAD = 20
_MAX_SKEW_FOR_ZSCORE = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_series(series: pd.Series) -> pd.Series:
    """Convert to numeric, replace inf with NaN, drop NaN."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.dropna()


def _severity_from_iqr(value: float, q1: float, q3: float, iqr: float) -> str:
    """Return severity string based on IQR distance."""
    extreme_lo = q1 - 3 * iqr
    extreme_hi = q3 + 3 * iqr
    high_lo = q1 - 2 * iqr
    high_hi = q3 + 2 * iqr

    if value < extreme_lo or value > extreme_hi:
        return AnomalySeverity.CRITICAL.value
    if value < high_lo or value > high_hi:
        return AnomalySeverity.HIGH.value
    return AnomalySeverity.MEDIUM.value


def _severity_from_z(z: float) -> str:
    """Return severity string based on absolute z-score."""
    az = abs(z)
    if az > 4:
        return AnomalySeverity.CRITICAL.value
    if az > 3.5:
        return AnomalySeverity.HIGH.value
    return AnomalySeverity.MEDIUM.value


_SEVERITY_RANK = {
    AnomalySeverity.LOW.value: 0,
    AnomalySeverity.MEDIUM.value: 1,
    AnomalySeverity.HIGH.value: 2,
    AnomalySeverity.CRITICAL.value: 3,
}


# ---------------------------------------------------------------------------
# Per-method detectors
# ---------------------------------------------------------------------------

def _detect_iqr(
    clean: pd.Series, col: str, dataset_id: str
) -> list[Anomaly]:
    """IQR-based outlier detection.  Requires n >= 10."""
    n = len(clean)
    if n < _MIN_OBS_IQR:
        return []

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return []

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    median = float(clean.median())

    outlier_mask = (clean < lower) | (clean > upper)
    outlier_idx = clean.index[outlier_mask]
    results: list[Anomaly] = []

    for idx in outlier_idx:
        value = float(clean.loc[idx])
        score = min(1.0, abs(value - median) / (3 * iqr))
        severity = _severity_from_iqr(value, q1, q3, iqr)
        deviation = (value - median) / iqr

        explanation = (
            f"Value {value} in column '{col}' is {abs(deviation):.1f} IQR "
            f"from the median ({median:.2f}). Detected by IQR method."
        )
        results.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            row_index=int(idx),
            anomaly_type=AnomalyType.NUMERIC_OUTLIER.value,
            severity=severity,
            score=round(score, 4),
            method="iqr",
            observed_value=value,
            expected_value=median,
            deviation=round(deviation, 4),
            explanation=explanation,
        ))

    return results


def _detect_zscore(
    clean: pd.Series, col: str, dataset_id: str
) -> list[Anomaly]:
    """Z-score outlier detection.  Requires n >= 30 and |skewness| < 2."""
    n = len(clean)
    if n < _MIN_OBS_ZSCORE:
        return []

    skewness = float(clean.skew())
    if abs(skewness) >= _MAX_SKEW_FOR_ZSCORE:
        return []

    mean = float(clean.mean())
    std = float(clean.std())
    if std == 0:
        return []

    z_scores = (clean - mean) / std
    outlier_mask = z_scores.abs() > 3
    outlier_idx = clean.index[outlier_mask]
    results: list[Anomaly] = []

    for idx in outlier_idx:
        value = float(clean.loc[idx])
        z = float(z_scores.loc[idx])
        score = min(1.0, abs(z) / 5)
        severity = _severity_from_z(z)

        explanation = (
            f"Value {value} in column '{col}' is {abs(z):.1f} standard "
            f"deviations from the mean ({mean:.2f}). Detected by Z-score method."
        )
        results.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            row_index=int(idx),
            anomaly_type=AnomalyType.NUMERIC_OUTLIER.value,
            severity=severity,
            score=round(score, 4),
            method="zscore",
            observed_value=value,
            expected_value=mean,
            deviation=round(z, 4),
            explanation=explanation,
        ))

    return results


def _detect_mad(
    clean: pd.Series, col: str, dataset_id: str
) -> list[Anomaly]:
    """Modified Z-score (MAD) outlier detection.  Requires n >= 20."""
    n = len(clean)
    if n < _MIN_OBS_MAD:
        return []

    median = float(clean.median())
    mad = float((clean - median).abs().median())
    if mad == 0:
        return []

    modified_z = 0.6745 * (clean - median) / mad
    outlier_mask = modified_z.abs() > 3.5
    outlier_idx = clean.index[outlier_mask]
    results: list[Anomaly] = []

    for idx in outlier_idx:
        value = float(clean.loc[idx])
        mz = float(modified_z.loc[idx])
        score = min(1.0, abs(mz) / 5)
        severity = _severity_from_z(mz)  # same thresholds

        explanation = (
            f"Value {value} in column '{col}' is {abs(mz):.1f} modified "
            f"Z-scores from the median ({median:.2f}). Detected by MAD method."
        )
        results.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            row_index=int(idx),
            anomaly_type=AnomalyType.NUMERIC_OUTLIER.value,
            severity=severity,
            score=round(score, 4),
            method="mad",
            observed_value=value,
            expected_value=median,
            deviation=round(mz, 4),
            explanation=explanation,
        ))

    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(anomalies: list[Anomaly]) -> list[Anomaly]:
    """If the same (row_index, column) is flagged by multiple methods,
    keep the highest-severity entry and note all methods in the explanation."""

    grouped: dict[tuple[Optional[int], str], list[Anomaly]] = {}
    for a in anomalies:
        key = (a.row_index, a.column)
        grouped.setdefault(key, []).append(a)

    deduplicated: list[Anomaly] = []
    for _key, group in grouped.items():
        if len(group) == 1:
            deduplicated.append(group[0])
            continue

        # Sort by severity descending, then score descending
        group.sort(
            key=lambda x: (_SEVERITY_RANK.get(x.severity, 0), x.score),
            reverse=True,
        )
        best = group[0]
        all_methods = sorted({a.method for a in group})
        methods_str = ", ".join(all_methods)

        # Rebuild explanation to note all methods
        best.explanation = (
            f"Value {best.observed_value} in column '{best.column}' flagged "
            f"as outlier by {methods_str}. Severity: {best.severity}, "
            f"score: {best.score:.4f}."
        )
        best.method = methods_str
        deduplicated.append(best)

    return deduplicated


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

def _select_columns(
    canonical: CanonicalDataset,
    df: pd.DataFrame,
) -> list[str]:
    """Choose which numeric columns to analyze, respecting exclusions and limits."""
    skip_ids = set(canonical.identifier_columns or [])

    constant_cols: set[str] = set()
    if canonical.data_quality is not None:
        constant_cols = set(canonical.data_quality.constant_columns or [])

    candidates: list[str] = []
    for col in canonical.numeric_columns or []:
        if col in skip_ids:
            continue
        if col in constant_cols:
            continue
        if col not in df.columns:
            continue
        candidates.append(col)

    if not candidates:
        return []

    # If too many columns, pick top MAX_NUMERIC_COLS by variance
    if len(candidates) > MAX_NUMERIC_COLS:
        variances = {}
        for col in candidates:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            variances[col] = float(s.var()) if len(s) > 1 else 0.0
        candidates.sort(key=lambda c: variances.get(c, 0.0), reverse=True)
        candidates = candidates[:MAX_NUMERIC_COLS]

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_numeric_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 50,
) -> list[Anomaly]:
    """Detect numeric outliers in the canonical dataset.

    Applies IQR, Z-score, and Modified Z-score (MAD) methods to every
    eligible numeric column.  Returns at most *max_anomalies* results,
    deduplicated and sorted by severity then score.

    Parameters
    ----------
    context : AnalysisContext
        The Phase 2 analysis context wrapping a CanonicalDataset.
    dataset_id : str
        Identifier to stamp on each Anomaly.
    max_anomalies : int
        Maximum number of anomalies to return.

    Returns
    -------
    list[Anomaly]
    """
    canonical = context.dataset
    df = context.get_dataframe()

    # ---- Edge case: no data at all ----
    if df is None or df.empty:
        logger.info("No data available for numeric anomaly detection.")
        return []

    # ---- Sampling for resource limits ----
    sampled = False
    original_rows = len(df)
    if original_rows > MAX_ROWS_ANALYZE:
        df = df.sample(n=MAX_ROWS_ANALYZE, random_state=42).copy()
        df.reset_index(drop=True, inplace=True)
        sampled = True
        logger.info(
            "Sampled %d rows from %d for numeric anomaly detection.",
            MAX_ROWS_ANALYZE,
            original_rows,
        )

    # ---- Column selection ----
    columns = _select_columns(canonical, df)
    if not columns:
        logger.info("No eligible numeric columns for anomaly detection.")
        return []

    # ---- Run detectors on each column ----
    all_anomalies: list[Anomaly] = []

    for col in columns:
        clean = _prepare_series(df[col])
        if len(clean) < _MIN_OBS_IQR:
            continue

        # IQR — always when n >= 10
        all_anomalies.extend(_detect_iqr(clean, col, dataset_id))

        # Z-score — when n >= 30 and roughly normal
        all_anomalies.extend(_detect_zscore(clean, col, dataset_id))

        # MAD — when n >= 20
        all_anomalies.extend(_detect_mad(clean, col, dataset_id))

    # ---- Deduplicate ----
    all_anomalies = _deduplicate(all_anomalies)

    # ---- Sort: highest severity first, then highest score ----
    all_anomalies.sort(
        key=lambda a: (_SEVERITY_RANK.get(a.severity, 0), a.score),
        reverse=True,
    )

    # ---- Cap at max_anomalies ----
    if len(all_anomalies) > max_anomalies:
        all_anomalies = all_anomalies[:max_anomalies]

    # ---- Add sampling metadata to explanations if sampled ----
    if sampled:
        note = (
            f" [Analysis based on {MAX_ROWS_ANALYZE:,} sampled rows "
            f"out of {original_rows:,} total.]"
        )
        for a in all_anomalies:
            a.explanation += note

    logger.info(
        "Numeric anomaly detection complete: %d anomalies across %d columns.",
        len(all_anomalies),
        len(columns),
    )
    return all_anomalies
