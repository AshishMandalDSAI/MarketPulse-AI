"""Generic dataset profiling — no hardcoded column names."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import IDENTIFIER_CARDINALITY_RATIO


def _infer_column_type(series: pd.Series, n_rows: int) -> str:
    """Best-effort dtype classification: numeric / categorical / datetime / text / identifier."""
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        ratio = series.nunique(dropna=True) / n_rows if n_rows else 0
        is_integer_like = bool((non_null.dropna() % 1 == 0).all())
        if ratio > IDENTIFIER_CARDINALITY_RATIO and n_rows > 20 and is_integer_like:
            return "identifier"
        return "numeric"

    # try datetime parse for object/string columns
    # Use is_string_dtype + is_object_dtype for Pandas 2.x/3.x compat
    # (Pandas 3.x uses StringDtype by default; Pandas 2.x uses object)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = non_null.head(30)
        try:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format",
                    category=UserWarning,
                )
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.85:
                return "datetime"
        except Exception:
            pass

        ratio = series.nunique(dropna=True) / n_rows if n_rows else 0
        if ratio > IDENTIFIER_CARDINALITY_RATIO and n_rows > 20:
            return "identifier"

        avg_len = non_null.astype(str).str.len().mean()
        if avg_len is not None and avg_len > 40:
            return "text"
        return "categorical"

    return "categorical"


def profile_dataset(df: pd.DataFrame) -> dict:
    """Row/col counts, per-column dtype inference, missing-value %,
    cardinality, and basic stats. No hardcoded column names."""
    n_rows, n_cols = df.shape
    columns = {}
    for col in df.columns:
        series = df[col]
        inferred = _infer_column_type(series, n_rows)
        missing_pct = float(series.isna().mean() * 100) if n_rows else 0.0
        entry = {
            "dtype": str(series.dtype),
            "inferred_type": inferred,
            "missing_pct": round(missing_pct, 2),
            "cardinality": int(series.nunique(dropna=True)),
        }
        if inferred == "numeric":
            non_null = pd.to_numeric(series, errors="coerce").dropna()
            # Summary stats are computed over finite values only: a column
            # holding both +inf and -inf would otherwise yield a NaN mean
            # (and a NumPy RuntimeWarning). Non-finite values are counted.
            finite = non_null[np.isfinite(non_null.to_numpy(dtype=float))]
            if not finite.empty:
                entry["stats"] = {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "std": float(finite.std()) if len(finite) > 1 else 0.0,
                }
                n_non_finite = int(len(non_null) - len(finite))
                if n_non_finite:
                    entry["stats"]["non_finite_count"] = n_non_finite
        columns[col] = entry

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": columns,
        "overall_missing_pct": round(float(df.isna().mean().mean() * 100) if n_cols else 0.0, 2),
    }
