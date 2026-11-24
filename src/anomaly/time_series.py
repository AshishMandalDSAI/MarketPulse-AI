"""Phase 5 — Time-Series Anomaly Detection.

Dataset-agnostic time-series anomaly detection using rolling deviation,
spike/drop detection, and residual-from-trend methods.  Consumes the
Phase 2 CanonicalDataset / AnalysisContext contracts — no hardcoded
column names.
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
MAX_ROWS = 50_000
MAX_TIME_COLS = 10

_MIN_OBS_ROLLING = 10
_MIN_OBS_SPIKE = 10
_MIN_OBS_TREND = 20

_ROLLING_DEVIATION_THRESHOLD = 2.5
_SPIKE_PCT_THRESHOLD = 0.5       # 50% change
_TREND_RESIDUAL_THRESHOLD = 2.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_time_column(time_info) -> Optional[str]:
    """Extract time column name from TimeInfo (dataclass or dict)."""
    if time_info is None:
        return None
    time_col = (
        time_info.time_column
        if hasattr(time_info, "time_column")
        else (
            time_info.get("time_column")
            if isinstance(time_info, dict)
            else None
        )
    )
    return time_col


def _prepare_series(series: pd.Series) -> pd.Series:
    """Convert to numeric, replace inf with NaN, drop NaN."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.dropna()


def _select_measure_columns(
    canonical: CanonicalDataset,
    time_col: str,
) -> list[str]:
    """Select numeric measure columns, excluding time/identifier/constant columns.

    Returns at most MAX_TIME_COLS columns, sorted by descending variance.
    """
    # Use measure_candidates (numeric, non-identifier) when available
    candidates: list[str] = []
    if hasattr(canonical, "measure_candidates") and canonical.measure_candidates:
        candidates = list(canonical.measure_candidates)
    elif hasattr(canonical, "numeric_columns") and canonical.numeric_columns:
        # Fallback: numeric minus identifiers
        identifiers = set(
            canonical.identifier_columns
            if hasattr(canonical, "identifier_columns")
            else []
        )
        candidates = [c for c in canonical.numeric_columns if c not in identifiers]
    else:
        return []

    # Exclude the time column itself
    candidates = [c for c in candidates if c != time_col]

    if not candidates:
        return []

    # Exclude constant columns
    df = canonical.dataframe
    if df is None:
        return []

    valid = []
    for col in candidates:
        if col not in df.columns:
            continue
        s = _prepare_series(df[col])
        if len(s) < _MIN_OBS_ROLLING:
            continue
        if s.std() == 0:
            continue
        valid.append((col, s.var()))

    # Sort by variance descending, keep top MAX_TIME_COLS
    valid.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in valid[:MAX_TIME_COLS]]


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def _detect_rolling_deviation(
    series: pd.Series,
    col: str,
    dataset_id: str,
    original_indices: pd.Index,
) -> list[Anomaly]:
    """Rolling window deviation detection (n >= 10)."""
    n = len(series)
    if n < _MIN_OBS_ROLLING:
        return []

    window = max(3, min(n // 5, 12))
    min_periods = max(2, window // 2)

    rolling_mean = series.rolling(window, center=True, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, center=True, min_periods=min_periods).std()

    deviation = (series - rolling_mean) / (rolling_std + 1e-10)

    anomalies: list[Anomaly] = []
    mask = deviation.abs() > _ROLLING_DEVIATION_THRESHOLD

    for idx in series.index[mask]:
        dev_val = float(deviation.loc[idx])
        obs_val = float(series.loc[idx])
        exp_val = float(rolling_mean.loc[idx])
        score = min(1.0, abs(dev_val) / 4.0)

        # Severity based on score
        if score >= 0.9:
            severity = AnomalySeverity.CRITICAL.value
        elif score >= 0.7:
            severity = AnomalySeverity.HIGH.value
        elif score >= 0.4:
            severity = AnomalySeverity.MEDIUM.value
        else:
            severity = AnomalySeverity.LOW.value

        row_idx = int(original_indices[idx]) if idx < len(original_indices) else idx

        anomalies.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            row_index=row_idx,
            anomaly_type=AnomalyType.TIME_SERIES_ANOMALY.value,
            severity=severity,
            score=round(score, 4),
            method="rolling_deviation",
            observed_value=obs_val,
            expected_value=round(exp_val, 4),
            deviation=round(dev_val, 4),
            explanation=(
                f"'{col}' value {obs_val} deviates {abs(dev_val):.1f} "
                f"standard deviations from the rolling mean ({exp_val:.4g}) "
                f"using a window of {window}."
            ),
        ))

    return anomalies


def _detect_spikes_drops(
    series: pd.Series,
    col: str,
    dataset_id: str,
    original_indices: pd.Index,
) -> list[Anomaly]:
    """Spike/drop detection via percent change (n >= 10)."""
    n = len(series)
    if n < _MIN_OBS_SPIKE:
        return []

    pct_change = series.pct_change()
    series_std = series.std()

    anomalies: list[Anomaly] = []

    # PART 06 performance: positional numpy access instead of one label lookup per row (the
    # loop used ~730k pandas .loc calls at 100k rows). Same values, same order, same output.
    pct_arr = pct_change.to_numpy()
    val_arr = series.to_numpy()
    labels = series.index
    with np.errstate(invalid="ignore"):
        candidates = np.flatnonzero(np.abs(pct_arr) > _SPIKE_PCT_THRESHOLD)
    for pos in candidates:
        if pos == 0:                # first row: no previous value (pct_change is NaN)
            continue
        idx = labels[pos]
        pct = pct_arr[pos]
        if pd.isna(pct) or not np.isfinite(pct):
            continue

        abs_pct = abs(pct)
        if abs_pct <= _SPIKE_PCT_THRESHOLD:
            continue

        # Check that absolute change is meaningful (> 1 std)
        abs_change = abs(val_arr[pos] - val_arr[pos - 1])
        if abs_change <= series_std:
            continue

        # Determine severity
        if abs_pct > 2.0:
            severity = AnomalySeverity.CRITICAL.value
        elif abs_pct > 1.0:
            severity = AnomalySeverity.HIGH.value
        else:
            severity = AnomalySeverity.MEDIUM.value

        direction = "spike" if pct > 0 else "drop"
        score = min(1.0, abs_pct / 2.0)

        obs_val = float(val_arr[pos])
        prev_val = float(val_arr[pos - 1])
        row_idx = int(original_indices[idx]) if idx < len(original_indices) else idx

        anomalies.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            row_index=row_idx,
            anomaly_type=AnomalyType.SUDDEN_CHANGE.value,
            severity=severity,
            score=round(score, 4),
            method="spike_drop",
            observed_value=obs_val,
            expected_value=round(prev_val, 4),
            deviation=round(pct, 4),
            explanation=(
                f"'{col}' shows a {direction} of {abs_pct:.0%} "
                f"(from {prev_val:.4g} to {obs_val:.4g}), "
                f"exceeding the 50% change threshold."
            ),
        ))

    return anomalies


def _detect_trend_residual(
    series: pd.Series,
    col: str,
    dataset_id: str,
    original_indices: pd.Index,
) -> list[Anomaly]:
    """Residual-from-trend detection via linear fit (n >= 20)."""
    n = len(series)
    if n < _MIN_OBS_TREND:
        return []

    values = series.values.astype(float)
    x = np.arange(n, dtype=float)

    try:
        coeffs = np.polyfit(x, values, 1)
    except (np.linalg.LinAlgError, ValueError):
        logger.debug("polyfit failed for column '%s'; skipping trend residual.", col)
        return []

    trend = np.polyval(coeffs, x)
    residuals = values - trend
    residual_std = np.std(residuals)

    if residual_std < 1e-10:
        return []

    anomalies: list[Anomaly] = []

    for i in range(n):
        abs_res = abs(residuals[i])
        if abs_res <= _TREND_RESIDUAL_THRESHOLD * residual_std:
            continue

        score = min(1.0, abs_res / (3.0 * residual_std + 1e-10))
        idx = series.index[i]

        if score >= 0.9:
            severity = AnomalySeverity.CRITICAL.value
        elif score >= 0.7:
            severity = AnomalySeverity.HIGH.value
        elif score >= 0.4:
            severity = AnomalySeverity.MEDIUM.value
        else:
            severity = AnomalySeverity.LOW.value

        obs_val = float(values[i])
        exp_val = float(trend[i])
        row_idx = int(original_indices[idx]) if idx < len(original_indices) else idx

        anomalies.append(Anomaly(
            dataset_id=dataset_id,
            column=col,
            row_index=row_idx,
            anomaly_type=AnomalyType.TIME_SERIES_ANOMALY.value,
            severity=severity,
            score=round(score, 4),
            method="trend_residual",
            observed_value=obs_val,
            expected_value=round(exp_val, 4),
            deviation=round(float(residuals[i]), 4),
            explanation=(
                f"'{col}' value {obs_val} deviates {abs_res:.4g} from "
                f"the linear trend ({exp_val:.4g}), which is "
                f"{abs_res / residual_std:.1f}x the residual standard deviation."
            ),
        ))

    return anomalies


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(anomalies: list[Anomaly]) -> list[Anomaly]:
    """Keep highest-severity anomaly per (row_index, column) pair."""
    severity_rank = {
        AnomalySeverity.CRITICAL.value: 4,
        AnomalySeverity.HIGH.value: 3,
        AnomalySeverity.MEDIUM.value: 2,
        AnomalySeverity.LOW.value: 1,
    }

    best: dict[tuple, Anomaly] = {}
    for a in anomalies:
        key = (a.row_index, a.column)
        existing = best.get(key)
        if existing is None:
            best[key] = a
        else:
            existing_rank = severity_rank.get(existing.severity, 0)
            new_rank = severity_rank.get(a.severity, 0)
            if new_rank > existing_rank or (
                new_rank == existing_rank and a.score > existing.score
            ):
                best[key] = a

    return list(best.values())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_time_series_anomalies(
    context: AnalysisContext,
    dataset_id: str,
    max_anomalies: int = 30,
) -> list[Anomaly]:
    """Detect time-series anomalies in the dataset.

    Returns a list of Anomaly objects, at most *max_anomalies*,
    sorted by descending score.  Returns an empty list when the
    dataset has no usable time column.
    """
    canonical = context.dataset

    # --- 1. Check for a valid time column ---
    time_info = canonical.time_info if hasattr(canonical, "time_info") else None
    time_col = _get_time_column(time_info)

    if not time_col:
        logger.debug("No time column detected; skipping time-series anomaly detection.")
        return []

    df = canonical.dataframe
    if df is None or df.empty:
        logger.debug("DataFrame is None or empty; skipping time-series anomaly detection.")
        return []

    if time_col not in df.columns:
        logger.debug("Time column '%s' not found in DataFrame; skipping.", time_col)
        return []

    # --- 2. Resource-limit the DataFrame ---
    if len(df) > MAX_ROWS:
        logger.info(
            "Sampling %d rows from %d for time-series anomaly detection.",
            MAX_ROWS, len(df),
        )
        df = df.sort_values(time_col).tail(MAX_ROWS).copy()
    else:
        df = df.copy()

    # Sort by time column
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)
    except Exception:
        logger.debug("Could not parse/sort time column '%s'; skipping.", time_col)
        return []

    # Drop rows where time is NaT after conversion
    df = df.dropna(subset=[time_col]).reset_index(drop=True)
    if len(df) < _MIN_OBS_ROLLING:
        return []

    # --- 3. Select measure columns ---
    # Temporarily set the sorted dataframe so _select_measure_columns uses it
    original_df = canonical.dataframe
    canonical.dataframe = df
    try:
        measure_cols = _select_measure_columns(canonical, time_col)
    finally:
        canonical.dataframe = original_df

    if not measure_cols:
        logger.debug("No valid measure columns found for time-series analysis.")
        return []

    # --- 4. Run detectors on each column ---
    all_anomalies: list[Anomaly] = []
    original_indices = df.index  # after reset_index, these are 0..n-1

    for col in measure_cols:
        series = _prepare_series(df[col])
        if len(series) < _MIN_OBS_ROLLING or series.std() == 0:
            continue

        # Re-index to positional for alignment
        series = series.reset_index(drop=True)
        col_indices = pd.RangeIndex(len(series))

        # a) Rolling deviation
        all_anomalies.extend(
            _detect_rolling_deviation(series, col, dataset_id, col_indices)
        )

        # b) Spike/drop
        all_anomalies.extend(
            _detect_spikes_drops(series, col, dataset_id, col_indices)
        )

        # c) Trend residual
        all_anomalies.extend(
            _detect_trend_residual(series, col, dataset_id, col_indices)
        )

    # --- 5. Deduplicate and limit ---
    all_anomalies = _deduplicate(all_anomalies)
    all_anomalies.sort(key=lambda a: a.score, reverse=True)

    if len(all_anomalies) > max_anomalies:
        all_anomalies = all_anomalies[:max_anomalies]

    logger.info(
        "Time-series anomaly detection found %d anomalies across %d columns.",
        len(all_anomalies), len(measure_cols),
    )

    return all_anomalies
