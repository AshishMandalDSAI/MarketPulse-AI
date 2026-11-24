"""Phase 4 — Generic Preprocessing & Feature Engineering for ML Tasks.

Provides dataset-agnostic preprocessing pipelines for supervised
(regression, classification) and unsupervised (segmentation) tasks.

Steps: target-NaN removal, numeric imputation (median), one-hot encoding
(low-cardinality categoricals), StandardScaler, datetime feature
extraction, ID/constant/leakage column removal.

Never hardcodes column names.

Usage:
    from src.prediction.preprocessing import preprocess_for_prediction
    X, y, features, report = preprocess_for_prediction(df, "revenue", ["clicks", "region"])
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Pandas compatibility — dtype lists for select_dtypes()
# ---------------------------------------------------------------------------
# In Pandas 2.x, string columns have dtype "object" and "str" is NOT a valid
# dtype for select_dtypes().  In Pandas 3.x, a dedicated StringDtype ("string")
# was introduced.  We build include-lists that work with both versions.

def _string_and_object_dtypes() -> list[str]:
    """Return dtype strings for selecting text/object columns across Pandas versions."""
    dtypes = ["object"]
    try:
        # "string" is recognised by select_dtypes in Pandas >= 1.0 (StringDtype).
        # In Pandas 2.x it exists but columns rarely use it unless explicitly set.
        # In Pandas 3.x it is the default for string columns.
        pd.DataFrame({"_t": pd.array(["a"], dtype="string")}).select_dtypes(include=["string"])
        dtypes.append("string")
    except TypeError:
        pass  # Pandas version does not recognise "string" in select_dtypes
    return dtypes


def _categorical_dtypes() -> list[str]:
    """Return dtype strings for selecting categorical-like columns across Pandas versions."""
    return _string_and_object_dtypes() + ["category", "bool"]


# Resource-limit constants — imported from task_detector when available,
# otherwise sensible defaults.
try:
    from src.prediction.task_detector import (
        MAX_CARDINALITY_FOR_ENCODING,
        MAX_FEATURES,
        MAX_ROWS,
    )
except ImportError:  # task_detector not yet created
    MAX_ROWS = 50_000
    MAX_FEATURES = 200
    MAX_CARDINALITY_FOR_ENCODING = 50


# ---------------------------------------------------------------------------
# Identifier-column heuristics
# ---------------------------------------------------------------------------

_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^index$|^uuid$|^guid$|^key$|^pk$|^row_?num)",
    re.IGNORECASE,
)


def _is_id_column(series: pd.Series, col_name: str) -> bool:
    """Return True when *col_name* / *series* looks like an identifier."""
    if _ID_PATTERNS.search(col_name):
        return True
    # Numeric column where every value is unique ⇒ likely an ID
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique == len(series.dropna()) and n_unique > 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Datetime feature extraction
# ---------------------------------------------------------------------------

def extract_datetime_features(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Extract year, month, day_of_week, quarter from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    time_column : str
        Column name that contains datetime values (or can be coerced).

    Returns
    -------
    pd.DataFrame
        New DataFrame with the extracted datetime feature columns.  The
        original *time_column* is **not** included in the output.
    """
    if time_column not in df.columns:
        return pd.DataFrame(index=df.index)

    ts = pd.to_datetime(df[time_column], errors="coerce")
    prefix = time_column
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_year"] = ts.dt.year.astype("Int64")
    out[f"{prefix}_month"] = ts.dt.month.astype("Int64")
    out[f"{prefix}_day_of_week"] = ts.dt.day_of_week.astype("Int64")
    out[f"{prefix}_quarter"] = ts.dt.quarter.astype("Int64")
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enforce_resource_limits(
    df: pd.DataFrame,
    report: dict[str, Any],
) -> pd.DataFrame:
    """Down-sample rows and cap features to stay within resource limits."""
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        report["rows_sampled_to"] = MAX_ROWS
    if df.shape[1] > MAX_FEATURES:
        # Keep the first MAX_FEATURES columns (arbitrary but deterministic)
        dropped = list(df.columns[MAX_FEATURES:])
        df = df.iloc[:, :MAX_FEATURES]
        report.setdefault("dropped_columns", []).extend(dropped)
    return df


def _remove_constant_columns(
    df: pd.DataFrame,
    report: dict[str, Any],
) -> pd.DataFrame:
    """Drop columns that have a single unique value (after imputation)."""
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols)
        report.setdefault("dropped_columns", []).extend(const_cols)
        report["constant_columns_removed"] = const_cols
    return df


def _detect_leakage_columns(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.99,
) -> list[str]:
    """Return feature names with |correlation| > *threshold* to the target."""
    leakage: list[str] = []
    numeric_cols = X.select_dtypes(include="number").columns
    if numeric_cols.empty or y.empty:
        return leakage
    for col in numeric_cols:
        try:
            corr = X[col].corr(y)
            if corr is not None and abs(corr) > threshold:
                leakage.append(col)
        except Exception:
            continue
    return leakage


# ---------------------------------------------------------------------------
# Main supervised preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_for_prediction(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    task_type: str = "regression",
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    """Full preprocessing pipeline for supervised ML tasks.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    target_column : str
        Name of the column to predict.
    feature_columns : list[str]
        Candidate feature column names.  Identifier columns and columns
        identical/nearly-identical to *target_column* are removed
        automatically.
    task_type : str
        ``"regression"`` or ``"classification"``.

    Returns
    -------
    X_processed : pd.DataFrame
        Processed feature matrix.
    y : pd.Series
        Target values (rows aligned with *X_processed*).
    feature_names : list[str]
        Final feature column names after all transformations.
    preprocessing_report : dict
        Details of every step taken.
    """
    report: dict[str, Any] = {
        "rows_before": len(df),
        "features_before": len(feature_columns),
        "imputed_columns": [],
        "encoded_columns": [],
        "dropped_columns": [],
        "scaled": False,
        "datetime_features_added": [],
        "leakage_columns_removed": [],
    }

    # --- Edge case: empty DataFrame ---
    if df.empty or not feature_columns:
        report["rows_after"] = 0
        report["features_after"] = 0
        return pd.DataFrame(), pd.Series(dtype="float64"), [], report

    # --- Ensure target exists ---
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # --- Work on a copy ---
    work = df.copy()

    # --- 1. Drop rows where target is NaN ---
    before_drop = len(work)
    work = work.dropna(subset=[target_column]).reset_index(drop=True)
    report["target_nan_rows_dropped"] = before_drop - len(work)

    if work.empty:
        report["rows_after"] = 0
        report["features_after"] = 0
        return pd.DataFrame(), pd.Series(dtype="float64"), [], report

    # --- Separate target ---
    y = work[target_column].copy()

    # --- Build feature DataFrame (exclude target) ---
    safe_features = [
        c for c in feature_columns
        if c in work.columns and c != target_column
    ]
    X = work[safe_features].copy()

    # --- 2. Exclude ID columns ---
    id_cols = [c for c in X.columns if _is_id_column(X[c], c)]
    if id_cols:
        X = X.drop(columns=id_cols)
        report["dropped_columns"].extend(id_cols)
        report["id_columns_removed"] = id_cols

    # --- Enforce resource limits early ---
    X = _enforce_resource_limits(X, report)

    # --- 3. Datetime feature extraction ---
    datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    # Also detect object/str columns that look like dates
    import warnings as _warnings
    for col in X.select_dtypes(include=_string_and_object_dtypes()).columns:
        try:
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
                parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
            if parsed.notna().sum() > 0.5 * len(parsed):
                X[col] = parsed
                datetime_cols.append(col)
        except Exception:
            continue

    dt_frames: list[pd.DataFrame] = []
    for dtcol in datetime_cols:
        dt_feats = extract_datetime_features(X, dtcol)
        if not dt_feats.empty:
            dt_frames.append(dt_feats)
            report["datetime_features_added"].extend(dt_feats.columns.tolist())

    # Drop original datetime columns (not usable directly by models)
    if datetime_cols:
        X = X.drop(columns=datetime_cols, errors="ignore")

    # Append extracted datetime features
    for dtf in dt_frames:
        X = pd.concat([X, dtf], axis=1)

    # --- 4. Split numeric vs categorical ---
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=_categorical_dtypes()).columns.tolist()

    # --- 5. Numeric imputation (median) ---
    for col in numeric_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            report["imputed_columns"].append(col)

    # --- 6. Categorical encoding (one-hot, low cardinality only) ---
    encoded_parts: list[pd.DataFrame] = []
    high_card_dropped: list[str] = []
    for col in cat_cols:
        n_unique = X[col].nunique(dropna=True)
        if n_unique <= MAX_CARDINALITY_FOR_ENCODING and n_unique > 0:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=int)
            encoded_parts.append(dummies)
            report["encoded_columns"].append(col)
        else:
            high_card_dropped.append(col)

    # Drop original categoricals
    if cat_cols:
        X = X.drop(columns=cat_cols, errors="ignore")
    # Append dummies
    for ep in encoded_parts:
        X = pd.concat([X, ep], axis=1)
    if high_card_dropped:
        report["dropped_columns"].extend(high_card_dropped)
        report["high_cardinality_dropped"] = high_card_dropped

    # --- 7. Drop all-NaN columns ---
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        report["dropped_columns"].extend(all_nan_cols)

    # --- 8. Remove constant columns ---
    X = _remove_constant_columns(X, report)

    # --- 9. Target leakage protection ---
    if not X.empty:
        leakage = _detect_leakage_columns(X, y, threshold=0.99)
        if leakage:
            X = X.drop(columns=leakage)
            report["leakage_columns_removed"] = leakage

    # --- 10. Scaling (StandardScaler on numeric features) ---
    if not X.empty:
        num_to_scale = X.select_dtypes(include="number").columns.tolist()
        if num_to_scale:
            scaler = StandardScaler()
            X[num_to_scale] = scaler.fit_transform(X[num_to_scale])
            report["scaled"] = True

    # --- Final bookkeeping ---
    feature_names = X.columns.tolist()
    report["rows_after"] = len(X)
    report["features_after"] = len(feature_names)

    return X, y, feature_names, report


# ---------------------------------------------------------------------------
# Segmentation (unsupervised) preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_segmentation(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Preprocessing for unsupervised clustering.

    Only numeric columns are kept.  Imputed with median, then scaled.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    feature_columns : list[str]
        Candidate feature column names.

    Returns
    -------
    X_processed : pd.DataFrame
        Processed numeric feature matrix.
    feature_names : list[str]
        Final feature column names.
    preprocessing_report : dict
        Details of every step taken.
    """
    report: dict[str, Any] = {
        "rows_before": len(df),
        "features_before": len(feature_columns),
        "imputed_columns": [],
        "encoded_columns": [],
        "dropped_columns": [],
        "scaled": False,
        "datetime_features_added": [],
        "leakage_columns_removed": [],
    }

    if df.empty or not feature_columns:
        report["rows_after"] = 0
        report["features_after"] = 0
        return pd.DataFrame(), [], report

    work = df.copy()

    # Keep only requested columns that exist
    safe_features = [c for c in feature_columns if c in work.columns]
    X = work[safe_features].copy()

    # --- Exclude ID columns ---
    id_cols = [c for c in X.columns if _is_id_column(X[c], c)]
    if id_cols:
        X = X.drop(columns=id_cols)
        report["dropped_columns"].extend(id_cols)

    # --- Enforce resource limits ---
    X = _enforce_resource_limits(X, report)

    # --- Datetime feature extraction ---
    datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    for col in X.select_dtypes(include=_string_and_object_dtypes()).columns:
        try:
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
                parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
            if parsed.notna().sum() > 0.5 * len(parsed):
                X[col] = parsed
                datetime_cols.append(col)
        except Exception:
            continue

    for dtcol in datetime_cols:
        dt_feats = extract_datetime_features(X, dtcol)
        if not dt_feats.empty:
            X = pd.concat([X, dt_feats], axis=1)
            report["datetime_features_added"].extend(dt_feats.columns.tolist())

    if datetime_cols:
        X = X.drop(columns=datetime_cols, errors="ignore")

    # --- Keep numeric only (segmentation uses numeric features) ---
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        report["dropped_columns"].extend(non_numeric)

    # --- Drop all-NaN columns ---
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        report["dropped_columns"].extend(all_nan_cols)

    # --- Imputation (median) ---
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
            report["imputed_columns"].append(col)

    # --- Remove constant columns ---
    X = _remove_constant_columns(X, report)

    # --- Scaling ---
    if not X.empty:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
        report["scaled"] = True

    feature_names = X.columns.tolist()
    report["rows_after"] = len(X)
    report["features_after"] = len(feature_names)

    return X, feature_names, report
