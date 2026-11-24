"""P0: target eligibility, task detection and dtype-level candidate columns.

Only declared pandas dtypes, label counts and the target's own value counts are read here (assumption
A10). No feature-target relationship and no learned parameter is computed; identifier / constant / empty /
value-based typing of FEATURES is done fold-locally in ``leakage.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from src.predictive.contracts import MAX_CLASSES, MAX_FEATURES


@dataclass
class TargetInfo:
    column: str
    eligible: bool
    task: str = ""                 # regression | classification
    ambiguous: bool = False        # numeric integer target with few levels: user may switch
    n_classes: int = 0
    reason: str = ""
    non_missing: int = 0


def _integer_valued(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return len(v) > 0 and bool(np.all(np.isfinite(v))) and bool(np.all(np.floor(v) == v))


def target_info(df: pd.DataFrame, column: str) -> TargetInfo:
    if column not in df.columns:
        return TargetInfo(column, False, reason="Column not found.")
    s = df[column]
    s = s.replace([np.inf, -np.inf], np.nan) if pdt.is_numeric_dtype(s) else s
    nn = s.dropna()
    info = TargetInfo(column, False, non_missing=int(len(nn)))
    if len(nn) == 0:
        info.reason = "The column is empty."
        return info
    if pdt.is_datetime64_any_dtype(s):
        info.reason = "Date columns cannot be predicted by these models."
        return info
    nd = int(nn.nunique())
    if nd < 2:
        info.reason = "The column has a single value, so there is nothing to predict."
        return info
    if pdt.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pdt.is_string_dtype(s):
        if nd > MAX_CLASSES:
            info.reason = (f"The column has {nd} distinct values, more than {MAX_CLASSES}; it looks like free text "
                           "or an identifier rather than a category.")
            return info
        return TargetInfo(column, True, "classification", False, nd, "Categorical target.", int(len(nn)))
    if pdt.is_numeric_dtype(s):
        n = len(nn)
        if _integer_valued(nn) and nd >= 0.99 * n and n >= 10 and bool((np.diff(np.sort(nn.values)) == 1).all()):
            info.reason = "The column is a running number (an identifier), not an outcome."
            return info
        if _integer_valued(nn) and nd <= MAX_CLASSES:
            task = "classification" if nd <= 10 else "regression"
            return TargetInfo(column, True, task, True, nd,
                              "Whole-number target with few distinct values: it can be treated as a category "
                              "or as a number.", int(n))
        return TargetInfo(column, True, "regression", False, 0, "Continuous numeric target.", int(n))
    info.reason = "Unsupported column type."
    return info


def eligible_targets(df: pd.DataFrame) -> list:
    """Eligible target columns, ranked by data quality (share of non-missing values), never by name."""
    rows = []
    for c in df.columns:
        info = target_info(df, c)
        if info.eligible:
            rows.append((-info.non_missing / max(len(df), 1), list(df.columns).index(c), c))
    return [c for _, _, c in sorted(rows)]


def candidate_features(df: pd.DataFrame, target: str, time_col: Optional[str] = None) -> tuple:
    """Dtype-level candidate predictors. Returns (candidates, excluded{col: reason}).

    Only the declared pandas dtype is used (no value-based inference, no platform semantic types).
    """
    cands, excluded = [], {}
    for c in df.columns:
        if c == target:
            continue
        if time_col and c == time_col:
            excluded[c] = "Used only to order the rows (never a predictor)."
            continue
        s = df[c]
        if pdt.is_datetime64_any_dtype(s):
            excluded[c] = "Date columns are never used as raw predictors."
        elif pdt.is_numeric_dtype(s) or pdt.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) \
                or s.dtype == object or pdt.is_string_dtype(s):
            cands.append(c)
        else:
            excluded[c] = f"Unsupported column type ({s.dtype})."
    if len(cands) > MAX_FEATURES:
        ranked = sorted(cands, key=lambda k: (-(df[k].notna().mean()), list(df.columns).index(k)))
        for c in ranked[MAX_FEATURES:]:
            excluded[c] = f"More than {MAX_FEATURES} candidate columns; this one had the most missing values."
        keep = set(ranked[:MAX_FEATURES])
        cands = [c for c in cands if c in keep]
    return cands, excluded


def date_like_columns(df: pd.DataFrame) -> list:
    """Declared date/time columns available as the ordering column in chronological mode."""
    return [c for c in df.columns if pdt.is_datetime64_any_dtype(df[c])]
