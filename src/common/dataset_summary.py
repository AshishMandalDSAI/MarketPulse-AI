"""Dataset-agnostic column typing and summary used by every Phase 8 page.

Reuses the platform's existing column-type detection
(``src.data_engine.canonical.build_column_metadata``) rather than adding a
second classifier, so Phase 8 pages agree with the rest of the platform on
what is numeric, categorical, a date or an identifier. No column name is
ever assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.data_engine.canonical import build_column_metadata


@dataclass
class DatasetSummary:
    n_rows: int = 0
    n_cols: int = 0
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    datetime_columns: list = field(default_factory=list)
    boolean_columns: list = field(default_factory=list)
    identifier_columns: list = field(default_factory=list)
    text_columns: list = field(default_factory=list)
    constant_columns: list = field(default_factory=list)
    empty_columns: list = field(default_factory=list)
    missing_cells: int = 0
    missing_pct: float = 0.0
    duplicate_rows: int = 0
    memory_mb: float = 0.0
    columns: list = field(default_factory=list)  # list[dict], one per column

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0 or self.n_cols == 0

    def column_table(self) -> pd.DataFrame:
        """One row per column: name, detected type, missing %, unique count."""
        return pd.DataFrame(
            self.columns,
            columns=["column", "detected_type", "dtype", "missing_pct", "unique_values"],
        )

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_cols": self.n_cols,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "datetime_columns": list(self.datetime_columns),
            "boolean_columns": list(self.boolean_columns),
            "identifier_columns": list(self.identifier_columns),
            "text_columns": list(self.text_columns),
            "constant_columns": list(self.constant_columns),
            "empty_columns": list(self.empty_columns),
            "missing_cells": self.missing_cells,
            "missing_pct": self.missing_pct,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": self.memory_mb,
        }


def _stringify_unhashable(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` where object columns holding lists/dicts/sets are turned
    into text, so the shared column-type detector (which needs hashable
    values) can classify them. The caller's frame is never modified."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object and out[col].map(
                lambda v: isinstance(v, (list, dict, set, tuple))).any():
            out[col] = out[col].map(lambda v: str(v) if isinstance(v, (list, dict, set, tuple)) else v)
    return out


def summarize_dataset(df: Optional[pd.DataFrame]) -> DatasetSummary:
    """Summarize any DataFrame. Never raises on empty/None input."""
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0 or df.shape[1] == 0:
        return DatasetSummary()

    n_rows, n_cols = df.shape
    summary = DatasetSummary(n_rows=int(n_rows), n_cols=int(n_cols))

    try:
        metadata = build_column_metadata(df)
    except TypeError:  # unhashable cell values (lists/dicts) break uniqueness counts
        df = _stringify_unhashable(df)
        metadata = build_column_metadata(df)

    for meta in metadata:
        st = meta.semantic_type
        if st == "numeric":
            summary.numeric_columns.append(meta.name)
        elif st == "categorical":
            summary.categorical_columns.append(meta.name)
        elif st == "datetime":
            summary.datetime_columns.append(meta.name)
        elif st == "boolean":
            summary.boolean_columns.append(meta.name)
        elif st == "identifier":
            summary.identifier_columns.append(meta.name)
        elif st == "text":
            summary.text_columns.append(meta.name)
        elif st == "empty":
            summary.empty_columns.append(meta.name)
        if meta.is_constant and st != "empty":
            summary.constant_columns.append(meta.name)
        summary.columns.append({
            "column": meta.name,
            "detected_type": st,
            "dtype": meta.dtype,
            "missing_pct": round(float(meta.missing_pct), 2),
            "unique_values": int(meta.unique_count),
        })

    summary.missing_cells = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    summary.missing_pct = round(summary.missing_cells / total_cells * 100, 2) if total_cells else 0.0
    summary.duplicate_rows = int(df.duplicated().sum())
    summary.memory_mb = round(float(df.memory_usage(deep=True).sum()) / 1_048_576, 3)
    return summary
