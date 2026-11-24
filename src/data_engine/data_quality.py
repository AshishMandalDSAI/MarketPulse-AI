"""Data-quality assessment and user-controlled cleaning.

Three steps, kept separate so nothing changes without the user seeing it:

1. ``assess_quality(df)``   — detect problems and score the data (read-only).
2. ``recommend_plan(df)``   — propose a per-column treatment, each with a reason.
3. ``apply_cleaning(df, plan)`` — apply an (edited) plan to a *copy* of the data
   and return the cleaned frame plus a complete audit trail.

Design rules
------------
* The input frame is never modified; the original upload is stored separately
  by the caller (raw bytes).
* Missing means: NaN/None, empty strings, whitespace-only strings, and common
  placeholder tokens (``NA``, ``N/A``, ``null``, ``-``...). Infinities and
  unparseable values in numeric/date columns are *invalid*, counted separately.
* Dates are never invented. Invalid or missing dates stay missing.
* Revenue/target columns are never imputed by default (inventing revenue would
  bias every model); rows without a target are simply not used for training.
* Time-ordered imputation only uses *earlier* observations (forward fill,
  expanding median of prior values), so no future information leaks into the past.
* Duplicates are flagged, not dropped, by default: identical rows can be
  legitimate (e.g. two equal transactions). The user decides.
* Quality is reported for the original data and, separately, after cleaning.
  Imputed cells are counted as *imputed*, so a cleaned dataset is never presented
  as naturally complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PLACEHOLDER_TOKENS = {"na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a", "missing"}

# Weights of the quality dimensions (sum to 1). Documented in the UI and report.
SCORE_WEIGHTS = {
    "completeness": 0.30,
    "validity": 0.25,
    "uniqueness": 0.15,
    "consistency": 0.15,
    "type_correctness": 0.15,
}

STRATEGIES = {
    "none": "Leave as is",
    "leave": "Leave missing (excluded where a value is required)",
    "mean": "Fill with column mean",
    "median": "Fill with column median",
    "zero": "Fill with 0",
    "ffill": "Forward fill (carry last earlier value)",
    "past_median": "Fill with median of earlier rows only",
    "interpolate": "Linear interpolation between neighbours",
    "mode": "Fill with most frequent value",
    "unknown": "Fill with explicit 'Unknown' category",
    "drop_rows": "Drop rows where this column is missing",
}

_NUM_CLEAN = re.compile(r"[,\s₹$€£%]")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _is_blank(series: pd.Series) -> pd.Series:
    """True for empty / whitespace-only / placeholder strings (object columns)."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return pd.Series(False, index=series.index)
    s = series.where(series.notna(), None)
    as_str = s.map(lambda v: v.strip().lower() if isinstance(v, str) else None)
    return as_str.map(lambda v: v is not None and (v == "" or v in PLACEHOLDER_TOKENS)).astype(bool)


def _to_numeric_lenient(series: pd.Series) -> pd.Series:
    """Parse numbers stored as text ('1,234', '₹500', '12%'). Non-parsable -> NaN."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    cleaned = series.map(lambda v: _NUM_CLEAN.sub("", v) if isinstance(v, str) else v)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # One day/month order for the whole column (per-value 'mixed' parsing reads
    # 03/01/2022 as 1 March but 17/01/2022 as 17 January in the same column).
    from src.data_engine.date_parsing import parse_date_column
    return parse_date_column(series)[0]


def _looks_like_date_column(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(50)
    if len(sample) < 3:
        return False
    parsed = _parse_dates(sample)
    hint = any(k in str(name).lower() for k in ("date", "time", "day", "week", "month", "period"))
    ratio = parsed.notna().mean()
    return ratio >= 0.8 or (hint and ratio >= 0.5)


@dataclass
class ColumnQuality:
    column: str
    kind: str                     # numeric | numeric_text | date | categorical | boolean | text | empty
    n_missing: int = 0            # NaN/None + blank/placeholder strings
    n_blank_strings: int = 0      # subset of n_missing that were '', whitespace or placeholders
    n_invalid: int = 0            # unparseable numbers/dates, +/-inf
    n_inf: int = 0
    n_unique: int = 0
    mixed_types: bool = False     # e.g. mostly numbers with some free text
    storage_matches_content: bool = True
    missing_pct: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------
# 1. assessment
# --------------------------------------------------------------------------

def _column_quality(series: pd.Series, name: str, n_rows: int) -> ColumnQuality:
    blank = _is_blank(series)
    is_na = series.isna() | blank
    n_missing = int(is_na.sum())
    present = series[~is_na]
    cq = ColumnQuality(column=str(name), kind="empty", n_missing=n_missing,
                       n_blank_strings=int(blank.sum()),
                       missing_pct=round(100.0 * n_missing / n_rows, 2) if n_rows else 0.0)
    if present.empty:
        cq.notes.append("Column is entirely empty.")
        return cq
    cq.n_unique = int(present.astype(str).nunique())

    lowered = set(present.astype(str).str.strip().str.lower().unique())
    if pd.api.types.is_bool_dtype(series) or (cq.n_unique <= 2 and lowered <= {"true", "false", "yes", "no"}):
        cq.kind = "boolean"
        return cq

    if pd.api.types.is_numeric_dtype(series):
        vals = series.astype(float)
        inf = np.isinf(vals.to_numpy())
        cq.kind = "numeric"
        cq.n_inf = int(inf.sum())
        cq.n_invalid = cq.n_inf
        if cq.n_inf:
            cq.notes.append(f"{cq.n_inf} infinite value(s).")
        return cq

    if _looks_like_date_column(present, name):
        parsed = _parse_dates(present.astype(str).str.strip())
        cq.kind = "date"
        cq.n_invalid = int(parsed.isna().sum())
        if cq.n_invalid:
            cq.notes.append(f"{cq.n_invalid} value(s) are not valid dates; they will stay missing (dates are never invented).")
        return cq

    numeric = _to_numeric_lenient(present)
    parse_ratio = float(numeric.notna().mean())
    if parse_ratio >= 0.75:
        cq.kind = "numeric_text"
        cq.storage_matches_content = False
        cq.n_invalid = int(numeric.isna().sum()) + int(np.isinf(numeric.dropna().to_numpy()).sum())
        cq.mixed_types = cq.n_invalid > 0
        cq.notes.append("Numbers are stored as text; they can be converted to numeric.")
        if cq.n_invalid:
            cq.notes.append(f"{cq.n_invalid} value(s) could not be read as numbers.")
        return cq
    if 0.3 <= parse_ratio < 0.75:
        cq.mixed_types = True
        cq.notes.append(f"Mixed content: {parse_ratio:.0%} of values look numeric, the rest are text.")

    avg_len = present.astype(str).str.len().mean()
    cq.kind = "text" if (avg_len > 40 and cq.n_unique > 0.5 * len(present)) else "categorical"
    return cq


def assess_quality(df: pd.DataFrame, *, imputed_cells: int = 0) -> dict:
    """Read-only quality report. ``imputed_cells`` lets a post-cleaning report
    count filled cells as imputed rather than naturally complete."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    raw_names = [str(c) for c in df.columns]
    dup_col_names = sorted({c for c in raw_names if raw_names.count(c) > 1})

    work = df.copy()
    if dup_col_names:
        work.columns = [f"{c}__{i}" for i, c in enumerate(raw_names)]

    columns: Dict[str, ColumnQuality] = {}
    missing_matrix = pd.DataFrame(index=work.index)
    for col in work.columns:
        cq = _column_quality(work[col], col, n_rows)
        columns[str(col)] = cq
        missing_matrix[col] = work[col].isna() | _is_blank(work[col])

    missing_cells = int(missing_matrix.to_numpy().sum()) if total_cells else 0
    invalid_cells = sum(c.n_invalid for c in columns.values())
    present_cells = max(total_cells - missing_cells, 0)

    per_row = missing_matrix.sum(axis=1) if n_cols else pd.Series(dtype=int)
    rows_any_missing = int((per_row > 0).sum())
    rows_all_missing = int((per_row == n_cols).sum()) if n_cols else 0
    row_hist = {
        "0": int((per_row == 0).sum()),
        "1": int((per_row == 1).sum()),
        "2": int((per_row == 2).sum()),
        "3+": int((per_row >= 3).sum()),
    }

    try:
        dup_rows = int(df.duplicated().sum()) if n_rows else 0
    except TypeError:  # unhashable cells
        dup_rows = int(df.astype(str).duplicated().sum())

    # --- dimension scores (0-100) ---
    completeness = 100.0 * (1 - (missing_cells + imputed_cells) / total_cells) if total_cells else 0.0
    validity = 100.0 * (1 - invalid_cells / present_cells) if present_cells else 0.0
    uniqueness = 100.0 * (1 - dup_rows / n_rows) if n_rows else 0.0
    n_mixed = sum(1 for c in columns.values() if c.mixed_types)
    consistency = 100.0 * (1 - (n_mixed + len(dup_col_names)) / n_cols) if n_cols else 0.0
    n_bad_storage = sum(1 for c in columns.values() if not c.storage_matches_content)
    type_correctness = 100.0 * (1 - n_bad_storage / n_cols) if n_cols else 0.0

    completeness_with_imputed = 100.0 * (1 - missing_cells / total_cells) if total_cells else 0.0
    dims = {
        "completeness": round(max(completeness, 0.0), 1),
        "validity": round(max(validity, 0.0), 1),
        "uniqueness": round(max(uniqueness, 0.0), 1),
        "consistency": round(max(consistency, 0.0), 1),
        "type_correctness": round(max(type_correctness, 0.0), 1),
    }
    overall = round(sum(dims[k] * w for k, w in SCORE_WEIGHTS.items()), 1)

    kinds: Dict[str, int] = {}
    for c in columns.values():
        if c.n_missing:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "total_cells": int(total_cells),
        "missing_cells": missing_cells,
        "missing_pct": round(100.0 * missing_cells / total_cells, 2) if total_cells else 0.0,
        "imputed_cells": int(imputed_cells),
        # cells holding a value (original or imputed); shown next to, never
        # instead of, the natural completeness used in the score
        "filled_pct": round(max(completeness_with_imputed, 0.0), 1),
        "invalid_cells": int(invalid_cells),
        "infinite_cells": int(sum(c.n_inf for c in columns.values())),
        "blank_string_cells": int(sum(c.n_blank_strings for c in columns.values())),
        "duplicate_rows": dup_rows,
        "duplicate_column_names": dup_col_names,
        "rows_with_missing": rows_any_missing,
        "rows_all_missing": rows_all_missing,
        "missing_per_row_histogram": row_hist,
        "column_kinds_with_missing": kinds,
        "columns": {k: v.to_dict() for k, v in columns.items()},
        "dimensions": dims,
        "score": overall,
        "score_method": (
            "Weighted average of five 0–100 dimensions: completeness 30% (non-missing, non-imputed cells), "
            "validity 25% (present cells that are valid — no infinities, unreadable numbers or invalid dates), "
            "uniqueness 15% (rows that are not exact duplicates), consistency 15% (columns without mixed "
            "content or duplicate names), type correctness 15% (columns stored in a type matching their content)."
        ),
    }


# --------------------------------------------------------------------------
# 2. recommendations
# --------------------------------------------------------------------------

def recommend_plan(df: pd.DataFrame, report: Optional[dict] = None, *, date_column: Optional[str] = None,
                   roles: Optional[Dict[str, str]] = None) -> dict:
    """Propose a treatment per column with a plain-language reason.

    ``roles`` maps column -> semantic role ('revenue', 'channel', 'date', ...)
    when known; it lets the plan avoid inventing revenue and explain spend gaps.
    """
    report = report or assess_quality(df)
    roles = roles or {}
    time_ordered = bool(date_column) and date_column in df.columns
    plan: Dict[str, dict] = {}
    for col, cq in report["columns"].items():
        if col not in df.columns:
            continue
        role = roles.get(col)
        n_missing, kind = cq["n_missing"], cq["kind"]
        convert = kind == "numeric_text"
        entry = {"strategy": "none", "reason": "No missing or invalid values.", "convert_numeric": convert,
                 "options": ["none"]}
        if kind == "date" or role == "date":
            entry.update(strategy="leave" if (n_missing or cq["n_invalid"]) else "none",
                         reason="Dates are never invented. Missing or invalid dates stay empty; those rows are "
                                "left out of time-based analyses.",
                         options=["none", "leave", "drop_rows"])
        elif kind == "empty":
            entry.update(strategy="leave", reason="Column is entirely empty; nothing to impute.",
                         options=["leave"])
        elif n_missing == 0 and cq["n_inf"] == 0 and not (convert and cq["n_invalid"]):
            if convert:
                entry["reason"] = "Numbers stored as text will be converted to numeric."
        elif kind in ("numeric", "numeric_text"):
            opts = ["leave", "mean", "median", "zero", "drop_rows"]
            if time_ordered:
                opts += ["ffill", "past_median", "interpolate"]
            entry["options"] = opts
            pct = cq["missing_pct"]
            values = _to_numeric_lenient(df[col]).replace([np.inf, -np.inf], np.nan)
            skew = float(values.skew()) if values.notna().sum() > 2 else 0.0
            if role in ("revenue", "target"):
                entry.update(strategy="leave",
                             reason="This is the revenue/target column. Inventing revenue would bias every "
                                    "model, so rows without it are excluded from training instead.")
            elif pct > 40:
                entry.update(strategy="leave",
                             reason=f"{pct:.0f}% missing — filling most of a column would mostly be invention.")
            elif role == "channel":
                entry.update(strategy="zero",
                             reason="Marketing spend column: a blank usually means nothing was spent that period. "
                                    "This is an assumption — choose 'leave' if blanks mean 'unknown'.")
            elif time_ordered:
                entry.update(strategy="past_median",
                             reason="Time-ordered data: fill with the median of earlier rows only, so no future "
                                    "information leaks into the past.")
            elif abs(skew) > 1:
                entry.update(strategy="median", reason=f"Skewed distribution (skew {skew:.1f}); the median is "
                                                       "robust to extreme values.")
            else:
                entry.update(strategy="mean", reason="Roughly symmetric distribution; the mean is representative.")
            if cq["n_inf"]:
                entry["reason"] += f" {cq['n_inf']} infinite value(s) are treated as missing first."
        elif kind in ("categorical", "boolean", "text"):
            entry["options"] = ["leave", "mode", "unknown", "drop_rows"]
            if cq["missing_pct"] <= 5 and kind != "text":
                entry.update(strategy="mode", reason="Few values missing; the most frequent value is a safe fill.")
            else:
                entry.update(strategy="unknown",
                             reason="Keeps missingness visible as its own 'Unknown' category instead of hiding it.")
        plan[col] = entry
    return {
        "columns": plan,
        "drop_duplicate_rows": False,
        "duplicate_rows_found": report["duplicate_rows"],
        "duplicate_reason": ("Exact duplicate rows are flagged but kept by default: identical rows can be "
                             "legitimate (e.g. two equal transactions). Tick removal if they are data-entry errors."),
        "date_column": date_column,
    }


# --------------------------------------------------------------------------
# 3. application
# --------------------------------------------------------------------------

def _past_median_fill(values: pd.Series) -> pd.Series:
    """Fill NaN at position i with the median of non-missing values before i.

    ``expanding().median()`` skips NaN, so at a missing position it equals the
    median of the earlier observed values exactly (O(n log n); a loop over a
    growing history list was O(n^2) and stalled on large uploads)."""
    values = pd.to_numeric(values, errors="coerce")
    past = values.expanding(min_periods=1).median()
    return values.fillna(past)


def _past_median_fill_reference(values: pd.Series) -> pd.Series:
    """Straightforward reference implementation (used by tests only)."""
    out = values.copy()
    history: List[float] = []
    for idx, v in values.items():
        if pd.isna(v):
            if history:
                out.at[idx] = float(np.median(history))
        else:
            history.append(float(v))
    return out


def apply_cleaning(df: pd.DataFrame, plan: dict) -> tuple:
    """Apply ``plan`` to a copy of ``df``. Returns (cleaned_df, audit)."""
    original_shape = df.shape
    out = df.copy()
    audit = {
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "original_rows": int(original_shape[0]),
        "original_cols": int(original_shape[1]),
        "blank_strings_normalised": 0,
        "infinite_values_to_missing": 0,
        "columns_converted_to_numeric": [],
        "invalid_numbers_to_missing": 0,
        "imputed_cells": {},
        "rows_removed": {"duplicates": 0, "missing_required": 0},
        "strategies": {},
        "transformed_columns": [],
    }

    # 1. normalise blanks/placeholders to real missing values (no information lost)
    for col in out.columns:
        blank = _is_blank(out[col])
        if blank.any():
            audit["blank_strings_normalised"] += int(blank.sum())
            out.loc[blank, col] = np.nan

    col_plans = plan.get("columns", {})

    # 2. type fixes: numbers stored as text, infinities
    for col, cp in col_plans.items():
        if col not in out.columns:
            continue
        if cp.get("convert_numeric"):
            before_missing = int(out[col].isna().sum())
            out[col] = _to_numeric_lenient(out[col])
            audit["invalid_numbers_to_missing"] += int(out[col].isna().sum()) - before_missing
            audit["columns_converted_to_numeric"].append(col)
            audit["transformed_columns"].append(col)
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            inf = np.isinf(out[col].astype(float).to_numpy())
            if inf.any():
                audit["infinite_values_to_missing"] += int(inf.sum())
                out.loc[inf, col] = np.nan

    # 3. duplicates (only when the user asked)
    if plan.get("drop_duplicate_rows"):
        before = len(out)
        out = out.drop_duplicates()
        audit["rows_removed"]["duplicates"] = before - len(out)

    # 4. time order for past-only strategies (restored afterwards)
    date_col = plan.get("date_column")
    order = None
    if date_col and date_col in out.columns:
        parsed = _parse_dates(out[date_col])
        order = parsed.sort_values(kind="stable", na_position="last").index

    # 5. missing-value strategies
    drop_mask = pd.Series(False, index=out.index)
    for col, cp in col_plans.items():
        if col not in out.columns:
            continue
        strategy = cp.get("strategy", "none")
        audit["strategies"][col] = strategy
        missing = out[col].isna()
        n_miss = int(missing.sum())
        if strategy in ("none", "leave") or n_miss == 0:
            continue
        if strategy == "drop_rows":
            drop_mask |= missing
            continue
        series = out[col]
        if strategy in ("mean", "median", "zero", "ffill", "past_median", "interpolate"):
            series = pd.to_numeric(series, errors="coerce")
            if strategy == "mean":
                filled = series.fillna(series.mean())
            elif strategy == "median":
                filled = series.fillna(series.median())
            elif strategy == "zero":
                filled = series.fillna(0.0)
            else:
                ordered = series.loc[order] if order is not None else series
                if strategy == "ffill":
                    f = ordered.ffill()
                elif strategy == "past_median":
                    f = _past_median_fill(ordered)
                else:  # interpolate: inside known values only, never extrapolate
                    f = ordered.interpolate(method="linear", limit_area="inside")
                filled = f.reindex(series.index)
        elif strategy == "mode":
            modes = series.mode(dropna=True)
            filled = series.fillna(modes.iloc[0]) if not modes.empty else series
        elif strategy == "unknown":
            filled = series.astype(object).where(series.notna(), "Unknown")
        else:
            continue
        n_filled = int((missing & filled.notna()).sum())
        out[col] = filled
        if n_filled:
            audit["imputed_cells"][col] = n_filled
            if col not in audit["transformed_columns"]:
                audit["transformed_columns"].append(col)

    if drop_mask.any():
        audit["rows_removed"]["missing_required"] = int(drop_mask.sum())
        out = out[~drop_mask]

    audit["final_rows"] = int(len(out))
    audit["final_cols"] = int(out.shape[1])
    audit["total_imputed_cells"] = int(sum(audit["imputed_cells"].values()))
    audit["total_rows_removed"] = int(sum(audit["rows_removed"].values()))
    audit["plan"] = plan
    return out, audit


def quality_summary(df_original: pd.DataFrame, df_clean: Optional[pd.DataFrame] = None,
                    audit: Optional[dict] = None) -> dict:
    """Original and post-cleaning quality side by side."""
    before = assess_quality(df_original)
    result = {"original": before}
    if df_clean is not None:
        imputed = int((audit or {}).get("total_imputed_cells", 0))
        result["cleaned"] = assess_quality(df_clean, imputed_cells=imputed)
        result["audit"] = audit
    return result
