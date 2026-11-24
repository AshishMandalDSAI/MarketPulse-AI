"""Canonical Data Contract — the universal internal representation for any
CSV/XLSX dataset.

This module defines the contracts that sit between the upload/data engine and
every downstream analysis module.  No module may assume a particular business
schema exists; every analysis module consumes this contract instead.

Architecture:
    Upload → profile_and_map → standardize → CanonicalDataset.from_standardized()
    CanonicalDataset → AnalysisContext → capability-aware modules

The canonical contract works for marketing, sales, finance, operations,
education, HR, customer, healthcare-like, time-series, cross-sectional,
no-date, unusual-column, missing-value, and minimal datasets.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import IDENTIFIER_CARDINALITY_RATIO, MAX_DIMENSION_CARDINALITY


# ---------------------------------------------------------------------------
# 1. Semantic Roles
# ---------------------------------------------------------------------------

class SemanticRole(str, Enum):
    """Universal semantic roles for columns.  Marketing-specific roles are
    optional extensions — never mandatory."""
    TIME = "TIME"
    TARGET = "TARGET"
    MEASURE = "MEASURE"
    DIMENSION = "DIMENSION"
    CATEGORY = "CATEGORY"
    IDENTIFIER = "IDENTIFIER"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    UNKNOWN = "UNKNOWN"

    # Optional marketing extensions
    MARKETING_SPEND = "MARKETING_SPEND"
    MARKETING_CHANNEL = "MARKETING_CHANNEL"


# ---------------------------------------------------------------------------
# 2. Column Metadata
# ---------------------------------------------------------------------------

@dataclass
class ColumnMetadata:
    """Standardized per-column metadata.  Provides enough information for
    any future module to avoid directly inspecting raw schema assumptions."""

    name: str
    dtype: str                          # pandas dtype string
    semantic_type: str                  # "numeric" | "categorical" | "datetime" | "text" | "identifier" | "boolean" | "empty"
    role_candidates: list = field(default_factory=list)  # list[SemanticRole value strings]

    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    unique_pct: float = 0.0

    is_numeric: bool = False
    is_categorical: bool = False
    is_datetime: bool = False
    is_identifier: bool = False
    is_boolean: bool = False
    is_constant: bool = False
    is_high_cardinality: bool = False
    is_text: bool = False

    sample_values: list = field(default_factory=list)  # up to 5 sample values

    # Numeric stats (populated only for numeric columns)
    stats: Optional[dict] = None  # {"min", "max", "mean", "std", "median"}

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type,
            "role_candidates": list(self.role_candidates),
            "missing_count": self.missing_count,
            "missing_pct": round(self.missing_pct, 4),
            "unique_count": self.unique_count,
            "unique_pct": round(self.unique_pct, 4),
            "is_numeric": self.is_numeric,
            "is_categorical": self.is_categorical,
            "is_datetime": self.is_datetime,
            "is_identifier": self.is_identifier,
            "is_boolean": self.is_boolean,
            "is_constant": self.is_constant,
            "is_high_cardinality": self.is_high_cardinality,
            "is_text": self.is_text,
            "sample_values": [_safe_serialize(v) for v in self.sample_values],
        }
        if self.stats:
            d["stats"] = {k: _safe_serialize(v) for k, v in self.stats.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ColumnMetadata:
        stats = d.get("stats")
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            semantic_type=d["semantic_type"],
            role_candidates=d.get("role_candidates", []),
            missing_count=d.get("missing_count", 0),
            missing_pct=d.get("missing_pct", 0.0),
            unique_count=d.get("unique_count", 0),
            unique_pct=d.get("unique_pct", 0.0),
            is_numeric=d.get("is_numeric", False),
            is_categorical=d.get("is_categorical", False),
            is_datetime=d.get("is_datetime", False),
            is_identifier=d.get("is_identifier", False),
            is_boolean=d.get("is_boolean", False),
            is_constant=d.get("is_constant", False),
            is_high_cardinality=d.get("is_high_cardinality", False),
            is_text=d.get("is_text", False),
            sample_values=d.get("sample_values", []),
            stats=stats,
        )


def _safe_serialize(val: Any) -> Any:
    """Convert numpy/pandas types to JSON-safe Python natives."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return val


def build_column_metadata(df: pd.DataFrame) -> list[ColumnMetadata]:
    """Build ColumnMetadata for every column in the DataFrame."""
    n_rows = len(df)
    result = []

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)

        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / n_rows * 100) if n_rows else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_pct = (unique_count / n_rows * 100) if n_rows else 0.0

        non_null = series.dropna()
        is_constant = unique_count <= 1
        is_high_cardinality = unique_count > MAX_DIMENSION_CARDINALITY

        # Semantic type detection
        semantic_type = _detect_semantic_type(series, n_rows, unique_count, column_name=col)

        is_numeric = semantic_type == "numeric"
        is_categorical = semantic_type == "categorical"
        is_datetime = semantic_type == "datetime"
        is_identifier = semantic_type == "identifier"
        is_boolean = semantic_type == "boolean"
        is_text = semantic_type == "text"
        is_empty = semantic_type == "empty"

        # Sample values (up to 5, non-null)
        sample_values = []
        if not non_null.empty:
            sample = non_null.head(5)
            sample_values = [_safe_serialize(v) for v in sample.tolist()]

        # Numeric stats
        stats = None
        if is_numeric:
            numeric_vals = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric_vals.empty:
                # A column holding both +inf and -inf has a NaN mean/std. That
                # NaN is intentional downstream (it denies such a column the
                # variance bonus in target scoring), so the values are kept as
                # they were; only NumPy's "invalid value" warning is silenced,
                # and only for these two reductions.
                with np.errstate(invalid="ignore"):
                    mean_val = float(numeric_vals.mean())
                    std_val = float(numeric_vals.std()) if len(numeric_vals) > 1 else 0.0
                stats = {
                    "min": float(numeric_vals.min()),
                    "max": float(numeric_vals.max()),
                    "mean": mean_val,
                    "std": std_val,
                    "median": float(numeric_vals.median()),
                }

        meta = ColumnMetadata(
            name=col,
            dtype=dtype_str,
            semantic_type=semantic_type,
            role_candidates=[],  # populated later by _assign_roles
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            is_datetime=is_datetime,
            is_identifier=is_identifier,
            is_boolean=is_boolean,
            is_constant=is_constant,
            is_high_cardinality=is_high_cardinality,
            is_text=is_text,
            sample_values=sample_values,
            stats=stats,
        )
        result.append(meta)

    return result


_ID_NAME_PATTERN = re.compile(
    r'(?:^|[_\s])(?:id|key|uuid|guid|code|sku|ean|upc|isbn)(?:[_\s]|$)',
    re.IGNORECASE,
)


def _column_name_suggests_identifier(name: str) -> bool:
    """Return True if the column name strongly suggests an identifier."""
    return bool(_ID_NAME_PATTERN.search(name))


def _detect_semantic_type(series: pd.Series, n_rows: int, unique_count: int,
                          column_name: str = "") -> str:
    """Detect the semantic type of a series."""
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    # Boolean detection
    if series.dtype == bool or (unique_count <= 2 and set(non_null.unique()) <= {True, False, 0, 1, "True", "False", "true", "false", "yes", "no", "Yes", "No", "Y", "N", "y", "n"}):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        ratio = unique_count / n_rows if n_rows else 0
        is_integer_like = bool((non_null.dropna() % 1 == 0).all()) if not non_null.empty else False
        if ratio > IDENTIFIER_CARDINALITY_RATIO and n_rows > 20 and is_integer_like:
            return "identifier"
        return "numeric"

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        sample = non_null.head(30)
        try:
            import warnings
            with warnings.catch_warnings():
                # Intentional: we probe without a known format to detect date columns
                warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.85:
                return "datetime"
        except Exception:
            pass

        ratio = unique_count / n_rows if n_rows else 0
        # String-column identifier detection uses both cardinality ratio AND
        # column name hints. A column whose name suggests an identifier (e.g.
        # "Student_ID", "order_code") and has very high cardinality ratio is
        # classified as identifier. Otherwise, columns with unique_count ≤
        # MAX_DIMENSION_CARDINALITY are kept as categorical dimensions so
        # they can reach recommend_charts() for visualization.
        if ratio > IDENTIFIER_CARDINALITY_RATIO and n_rows > 20:
            if unique_count > MAX_DIMENSION_CARDINALITY:
                return "identifier"
            if _column_name_suggests_identifier(column_name):
                return "identifier"

        avg_len = non_null.astype(str).str.len().mean()
        if avg_len is not None and avg_len > 40:
            return "text"
        return "categorical"

    return "categorical"


# ---------------------------------------------------------------------------
# 3. Data Quality Contract
# ---------------------------------------------------------------------------

@dataclass
class DataQuality:
    """Standardized dataset quality information."""

    row_count: int = 0
    column_count: int = 0
    missing_cells: int = 0
    missing_cells_pct: float = 0.0
    duplicate_rows: int = 0
    duplicate_rows_pct: float = 0.0
    constant_columns: list = field(default_factory=list)
    high_missing_columns: list = field(default_factory=list)  # >50% missing
    empty_columns: list = field(default_factory=list)
    high_cardinality_columns: list = field(default_factory=list)
    mixed_type_columns: list = field(default_factory=list)
    infinite_value_columns: list = field(default_factory=list)
    score: float = 1.0  # 0.0-1.0, computed from actual issues
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_cells": self.missing_cells,
            "missing_cells_pct": round(self.missing_cells_pct, 4),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_rows_pct": round(self.duplicate_rows_pct, 4),
            "constant_columns": list(self.constant_columns),
            "high_missing_columns": list(self.high_missing_columns),
            "empty_columns": list(self.empty_columns),
            "high_cardinality_columns": list(self.high_cardinality_columns),
            "mixed_type_columns": list(self.mixed_type_columns),
            "infinite_value_columns": list(self.infinite_value_columns),
            "score": round(self.score, 4),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataQuality:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def compute_data_quality(df: pd.DataFrame) -> DataQuality:
    """Compute a DataQuality report from a DataFrame."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    missing_cells = int(df.isna().sum().sum())
    missing_cells_pct = (missing_cells / total_cells * 100) if total_cells else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_rows_pct = (duplicate_rows / n_rows * 100) if n_rows else 0.0

    constant_columns = []
    empty_columns = []
    high_missing_columns = []
    high_cardinality_columns = []
    mixed_type_columns = []
    infinite_value_columns = []
    warnings = []

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        missing_pct = series.isna().mean() * 100

        if n_unique <= 1:
            if series.isna().all():
                empty_columns.append(col)
            else:
                constant_columns.append(col)

        if missing_pct > 50:
            high_missing_columns.append(col)

        if n_unique > MAX_DIMENSION_CARDINALITY and not pd.api.types.is_numeric_dtype(series):
            high_cardinality_columns.append(col)

        # Mixed-type detection: string/object columns with a mix of numeric and non-numeric
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if not non_null.empty:
                numeric_mask = pd.to_numeric(non_null, errors="coerce").notna()
                pct_numeric = numeric_mask.mean()
                if 0.1 < pct_numeric < 0.9:
                    mixed_type_columns.append(col)

        # Infinite value detection
        if pd.api.types.is_numeric_dtype(series):
            n_inf = int(np.isinf(series.dropna().values).sum()) if not series.dropna().empty else 0
            if n_inf > 0:
                infinite_value_columns.append(col)

    # Warnings
    if empty_columns:
        warnings.append(f"{len(empty_columns)} completely empty column(s): {empty_columns}")
    if constant_columns:
        warnings.append(f"{len(constant_columns)} constant column(s): {constant_columns}")
    if high_missing_columns:
        warnings.append(f"{len(high_missing_columns)} column(s) with >50% missing: {high_missing_columns}")
    if duplicate_rows > 0:
        warnings.append(f"{duplicate_rows} duplicate row(s) ({duplicate_rows_pct:.1f}%)")
    if mixed_type_columns:
        warnings.append(f"{len(mixed_type_columns)} column(s) with mixed types: {mixed_type_columns}")
    if infinite_value_columns:
        warnings.append(f"{len(infinite_value_columns)} column(s) with infinite values: {infinite_value_columns}")

    # Quality score: start at 1.0, deduct for each issue
    score = 1.0
    if n_cols > 0:
        score -= min(0.3, missing_cells_pct / 100 * 0.5)  # up to 0.3 for missing data
        score -= min(0.15, duplicate_rows_pct / 100 * 0.3)  # up to 0.15 for duplicates
        score -= min(0.1, len(constant_columns) / n_cols * 0.2)  # up to 0.1 for constant cols
        score -= min(0.1, len(empty_columns) / n_cols * 0.2)  # up to 0.1 for empty cols
        score -= min(0.1, len(mixed_type_columns) / n_cols * 0.2)  # up to 0.1 for mixed types
        score -= min(0.05, len(infinite_value_columns) / n_cols * 0.1)  # up to 0.05 for inf
    score = max(0.0, round(score, 4))

    return DataQuality(
        row_count=n_rows,
        column_count=n_cols,
        missing_cells=missing_cells,
        missing_cells_pct=missing_cells_pct,
        duplicate_rows=duplicate_rows,
        duplicate_rows_pct=duplicate_rows_pct,
        constant_columns=constant_columns,
        high_missing_columns=high_missing_columns,
        empty_columns=empty_columns,
        high_cardinality_columns=high_cardinality_columns,
        mixed_type_columns=mixed_type_columns,
        infinite_value_columns=infinite_value_columns,
        score=score,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# 4. Target Detection Contract
# ---------------------------------------------------------------------------

@dataclass
class TargetCandidate:
    """A candidate target/outcome column."""
    column: str
    score: float
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"column": self.column, "score": round(self.score, 4), "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, d: dict) -> TargetCandidate:
        return cls(column=d["column"], score=d["score"], reasons=d.get("reasons", []))


def detect_target_candidates(df: pd.DataFrame, column_metadata: list[ColumnMetadata],
                              semantic_roles: dict) -> list[TargetCandidate]:
    """Detect target/outcome candidates from column metadata and semantic roles.
    Never hardcodes Revenue/Sales — uses structural and semantic signals."""
    import re

    candidates = []
    n_rows = len(df)

    # Target-like name hints (generic, not tied to any specific schema)
    target_hints = {
        "revenue", "sales", "income", "profit", "turnover", "outcome",
        "target", "result", "amount", "value", "total", "score", "grade",
        "marks", "rating", "performance", "output", "yield", "return",
        "price", "cost", "expense", "salary", "wage", "payment",
        "count", "quantity", "volume",
    }

    for meta in column_metadata:
        if not meta.is_numeric:
            continue
        if meta.is_identifier or meta.is_constant:
            continue

        score = 0.0
        reasons = []

        # Check if column has TARGET role from semantic analysis
        roles = semantic_roles.get(meta.name, [])
        if SemanticRole.TARGET.value in roles:
            score += 0.35
            reasons.append("Assigned TARGET semantic role")

        if SemanticRole.MEASURE.value in roles:
            score += 0.10
            reasons.append("Assigned MEASURE semantic role")

        # Name-based hints
        norm_name = re.sub(r"[_\-./\s]+", " ", meta.name.lower()).strip()
        name_tokens = set(norm_name.split())
        matched_hints = name_tokens & target_hints
        if matched_hints:
            score += 0.20
            reasons.append(f"Name contains target-like terms: {matched_hints}")

        # Statistical signals
        if meta.stats:
            # Variance: good targets have meaningful variance
            cv = (meta.stats.get("std", 0) / (abs(meta.stats.get("mean", 0)) + 1e-9))
            if cv > 0.05:
                score += 0.10
                reasons.append(f"Has meaningful variance (CV={cv:.2f})")

            # Non-negative: many targets are non-negative
            if meta.stats.get("min", -1) >= 0:
                score += 0.05
                reasons.append("Non-negative values")

        # Cardinality: targets tend to have moderate-to-high cardinality
        if n_rows > 10 and meta.unique_count > 5:
            score += 0.05
            reasons.append(f"Adequate cardinality ({meta.unique_count} unique values)")

        # Low missing: good targets usually aren't mostly missing
        if meta.missing_pct < 10:
            score += 0.05
            reasons.append("Low missing rate")

        if score >= 0.15 and reasons:
            candidates.append(TargetCandidate(column=meta.name, score=min(score, 1.0), reasons=reasons))

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 5. Time Detection Contract
# ---------------------------------------------------------------------------

@dataclass
class TimeInfo:
    """Standardized time detection result."""
    time_column: Optional[str] = None
    time_granularity: Optional[str] = None  # "daily" | "weekly" | "monthly" | "yearly" | "irregular" | None
    is_regular_time_series: bool = False
    time_range: Optional[dict] = None  # {"start": ..., "end": ..., "periods": int}

    def to_dict(self) -> dict:
        return {
            "time_column": self.time_column,
            "time_granularity": self.time_granularity,
            "is_regular_time_series": self.is_regular_time_series,
            "time_range": self.time_range,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TimeInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def detect_time_info(df: pd.DataFrame, date_col: Optional[str]) -> TimeInfo:
    """Detect time characteristics from the identified date column."""
    if not date_col or date_col not in df.columns:
        return TimeInfo()

    series = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if series.empty or len(series) < 2:
        return TimeInfo(time_column=date_col)

    series = series.sort_values()
    start = series.iloc[0]
    end = series.iloc[-1]
    n_periods = len(series)

    # Detect granularity from median gap
    diffs = series.diff().dropna()
    if diffs.empty:
        return TimeInfo(
            time_column=date_col,
            time_range={"start": start.isoformat(), "end": end.isoformat(), "periods": n_periods},
        )

    median_days = diffs.dt.days.median()

    if median_days <= 1.5:
        granularity = "daily"
    elif median_days <= 9:
        granularity = "weekly"
    elif median_days <= 35:
        granularity = "monthly"
    elif median_days <= 100:
        granularity = "quarterly"
    elif median_days <= 400:
        granularity = "yearly"
    else:
        granularity = "irregular"

    # Regularity check: is the CV of gaps small?
    gap_cv = (diffs.dt.days.std() / (abs(diffs.dt.days.mean()) + 1e-9)) if len(diffs) > 1 else 0
    is_regular = gap_cv < 0.3 and granularity != "irregular"

    return TimeInfo(
        time_column=date_col,
        time_granularity=granularity,
        is_regular_time_series=is_regular,
        time_range={"start": start.isoformat(), "end": end.isoformat(), "periods": n_periods},
    )


# ---------------------------------------------------------------------------
# 6. User Overrides Contract
# ---------------------------------------------------------------------------

@dataclass
class UserOverrides:
    """Contract for user overrides of auto-detected roles.  Future phases
    will expose these through the dashboard UI."""

    target_column: Optional[str] = None
    date_column: Optional[str] = None
    ignore_columns: list = field(default_factory=list)
    force_categorical: list = field(default_factory=list)
    force_measure: list = field(default_factory=list)
    force_marketing_spend: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_column": self.target_column,
            "date_column": self.date_column,
            "ignore_columns": list(self.ignore_columns),
            "force_categorical": list(self.force_categorical),
            "force_measure": list(self.force_measure),
            "force_marketing_spend": list(self.force_marketing_spend),
        }

    @classmethod
    def from_dict(cls, d: dict) -> UserOverrides:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def has_overrides(self) -> bool:
        return bool(
            self.target_column or self.date_column or self.ignore_columns
            or self.force_categorical or self.force_measure or self.force_marketing_spend
        )

    def to_pipeline_overrides(self) -> dict:
        """Convert to the format expected by pipeline._apply_overrides()."""
        result = {}
        if self.date_column is not None:
            result["date_column"] = self.date_column
        if self.target_column is not None:
            result["revenue_column"] = self.target_column
        if self.force_marketing_spend:
            result["channel_columns"] = self.force_marketing_spend
        return result


# ---------------------------------------------------------------------------
# 7. Canonical Dataset — the main contract object
# ---------------------------------------------------------------------------

@dataclass
class CanonicalDataset:
    """The universal internal data contract.

    Every analysis module should eventually consume this instead of
    independently loading pd.read_csv() or making schema assumptions.
    The DataFrame lives in runtime memory; all metadata is safely serializable.

    This object is always associated with a dataset_id — never a global singleton.
    """

    # Identity
    dataset_id: str
    filename: str = ""

    # Shape
    row_count: int = 0
    column_count: int = 0

    # The DataFrame (in-memory, not serialized)
    dataframe: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Column metadata
    columns: list = field(default_factory=list)       # list[str] — column names in order
    column_metadata: dict = field(default_factory=dict)  # name -> ColumnMetadata

    # Column type groups
    date_columns: list = field(default_factory=list)
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    boolean_columns: list = field(default_factory=list)
    text_columns: list = field(default_factory=list)
    identifier_columns: list = field(default_factory=list)
    empty_columns: list = field(default_factory=list)

    # Semantic roles: column_name -> list[SemanticRole values]
    semantic_roles: dict = field(default_factory=dict)

    # Detection results
    target_candidates: list = field(default_factory=list)  # list[TargetCandidate]
    dimension_candidates: list = field(default_factory=list)  # column names
    measure_candidates: list = field(default_factory=list)  # column names
    time_candidates: list = field(default_factory=list)  # column names

    # Contracts
    data_quality: Optional[DataQuality] = None
    time_info: Optional[TimeInfo] = None
    capabilities: Optional[dict] = None  # from CapabilityReport.to_dict()

    # User overrides
    overrides: Optional[UserOverrides] = None

    # Back-references to the existing architecture
    _standardized: Optional[Any] = field(default=None, repr=False)  # StandardizedDataset

    @classmethod
    def from_standardized(cls, dataset_id: str, std_dataset,
                           filename: str = "",
                           profile: Optional[dict] = None,
                           capability_report=None,
                           overrides: Optional[UserOverrides] = None) -> CanonicalDataset:
        """Build a CanonicalDataset from an existing StandardizedDataset.
        This is the primary construction path — bridges the existing architecture."""
        df = std_dataset.df
        n_rows, n_cols = df.shape if df is not None else (0, 0)

        # Build column metadata
        col_meta_list = build_column_metadata(df) if df is not None and not df.empty else []
        col_meta_dict = {m.name: m for m in col_meta_list}

        # Classify columns by type
        date_columns = [m.name for m in col_meta_list if m.is_datetime]
        numeric_columns = [m.name for m in col_meta_list if m.is_numeric]
        categorical_columns = [m.name for m in col_meta_list if m.is_categorical]
        boolean_columns = [m.name for m in col_meta_list if m.is_boolean]
        text_columns = [m.name for m in col_meta_list if m.is_text]
        identifier_columns = [m.name for m in col_meta_list if m.is_identifier]
        empty_columns = [m.name for m in col_meta_list if m.semantic_type == "empty"]

        # Build semantic roles from StandardizedDataset
        semantic_roles = _build_semantic_roles(std_dataset, col_meta_dict)

        # Assign role_candidates to each ColumnMetadata
        for meta in col_meta_list:
            meta.role_candidates = semantic_roles.get(meta.name, [])

        # Data quality
        quality = compute_data_quality(df) if df is not None and not df.empty else DataQuality()

        # Time info
        time_info = detect_time_info(df, std_dataset.date_col)

        # Target candidates
        target_candidates = detect_target_candidates(df, col_meta_list, semantic_roles) if df is not None and not df.empty else []

        # Capability dict
        cap_dict = capability_report.to_dict() if capability_report else None

        # Time candidates = all datetime-type columns
        time_candidates = list(date_columns)
        if std_dataset.date_col and std_dataset.date_col not in time_candidates:
            time_candidates.insert(0, std_dataset.date_col)

        # Measure candidates = all numeric non-identifier columns
        measure_candidates = [c for c in numeric_columns if c not in identifier_columns]

        # Dimension candidates = all categorical columns
        dimension_candidates = list(categorical_columns)

        return cls(
            dataset_id=dataset_id,
            filename=filename,
            row_count=n_rows,
            column_count=n_cols,
            dataframe=df,
            columns=list(df.columns) if df is not None else [],
            column_metadata=col_meta_dict,
            date_columns=date_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            boolean_columns=boolean_columns,
            text_columns=text_columns,
            identifier_columns=identifier_columns,
            empty_columns=empty_columns,
            semantic_roles=semantic_roles,
            target_candidates=target_candidates,
            dimension_candidates=dimension_candidates,
            measure_candidates=measure_candidates,
            time_candidates=time_candidates,
            data_quality=quality,
            time_info=time_info,
            capabilities=cap_dict,
            overrides=overrides,
            _standardized=std_dataset,
        )

    def to_metadata_dict(self) -> dict:
        """Serialize ALL metadata (everything except the DataFrame) to a
        JSON-safe dict. Safe for Streamlit session state, caching, API, testing."""
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "column_metadata": {name: m.to_dict() for name, m in self.column_metadata.items()},
            "date_columns": list(self.date_columns),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "boolean_columns": list(self.boolean_columns),
            "text_columns": list(self.text_columns),
            "identifier_columns": list(self.identifier_columns),
            "empty_columns": list(self.empty_columns),
            "semantic_roles": {k: list(v) for k, v in self.semantic_roles.items()},
            "target_candidates": [t.to_dict() for t in self.target_candidates],
            "dimension_candidates": list(self.dimension_candidates),
            "measure_candidates": list(self.measure_candidates),
            "time_candidates": list(self.time_candidates),
            "data_quality": self.data_quality.to_dict() if self.data_quality else None,
            "time_info": self.time_info.to_dict() if self.time_info else None,
            "capabilities": self.capabilities,
            "overrides": self.overrides.to_dict() if self.overrides else None,
        }

    @classmethod
    def from_metadata_dict(cls, d: dict, df: Optional[pd.DataFrame] = None) -> CanonicalDataset:
        """Reconstruct a CanonicalDataset from a serialized metadata dict."""
        col_meta = {
            name: ColumnMetadata.from_dict(m)
            for name, m in d.get("column_metadata", {}).items()
        }
        targets = [TargetCandidate.from_dict(t) for t in d.get("target_candidates", [])]
        quality = DataQuality.from_dict(d["data_quality"]) if d.get("data_quality") else None
        time_info = TimeInfo.from_dict(d["time_info"]) if d.get("time_info") else None
        overrides = UserOverrides.from_dict(d["overrides"]) if d.get("overrides") else None

        return cls(
            dataset_id=d["dataset_id"],
            filename=d.get("filename", ""),
            row_count=d.get("row_count", 0),
            column_count=d.get("column_count", 0),
            dataframe=df,
            columns=d.get("columns", []),
            column_metadata=col_meta,
            date_columns=d.get("date_columns", []),
            numeric_columns=d.get("numeric_columns", []),
            categorical_columns=d.get("categorical_columns", []),
            boolean_columns=d.get("boolean_columns", []),
            text_columns=d.get("text_columns", []),
            identifier_columns=d.get("identifier_columns", []),
            empty_columns=d.get("empty_columns", []),
            semantic_roles=d.get("semantic_roles", {}),
            target_candidates=targets,
            dimension_candidates=d.get("dimension_candidates", []),
            measure_candidates=d.get("measure_candidates", []),
            time_candidates=d.get("time_candidates", []),
            data_quality=quality,
            time_info=time_info,
            capabilities=d.get("capabilities"),
            overrides=overrides,
        )

    def get_standardized(self):
        """Access the underlying StandardizedDataset for backward compatibility."""
        return self._standardized


def _build_semantic_roles(std_dataset, col_meta_dict: dict) -> dict:
    """Build semantic role assignments from a StandardizedDataset + column metadata."""
    roles = {}

    def _add(col, role_val):
        if col:
            roles.setdefault(col, [])
            if role_val not in roles[col]:
                roles[col].append(role_val)

    # Time
    if std_dataset.date_col:
        _add(std_dataset.date_col, SemanticRole.TIME.value)

    # Target (revenue is the primary target in the existing architecture)
    if std_dataset.revenue_col:
        _add(std_dataset.revenue_col, SemanticRole.TARGET.value)
        _add(std_dataset.revenue_col, SemanticRole.MEASURE.value)

    # Marketing spend channels
    for ch in std_dataset.channels:
        _add(ch, SemanticRole.MEASURE.value)
        _add(ch, SemanticRole.MARKETING_SPEND.value)

    # Control cols = generic measures
    for col in std_dataset.control_cols:
        _add(col, SemanticRole.MEASURE.value)

    # Dimension cols = categories/dimensions
    for col in std_dataset.dimension_cols:
        _add(col, SemanticRole.DIMENSION.value)
        _add(col, SemanticRole.CATEGORY.value)

    # Identifier cols
    for col in std_dataset.identifier_cols:
        _add(col, SemanticRole.IDENTIFIER.value)

    # Fill UNKNOWN for any column without a role
    if std_dataset.df is not None:
        for col in std_dataset.df.columns:
            if col not in roles:
                meta = col_meta_dict.get(col)
                if meta:
                    if meta.is_datetime:
                        _add(col, SemanticRole.TIME.value)
                    elif meta.is_boolean:
                        _add(col, SemanticRole.BOOLEAN.value)
                    elif meta.is_text:
                        _add(col, SemanticRole.TEXT.value)
                    elif meta.is_numeric:
                        _add(col, SemanticRole.MEASURE.value)
                    elif meta.is_categorical:
                        _add(col, SemanticRole.CATEGORY.value)
                    else:
                        _add(col, SemanticRole.UNKNOWN.value)

    return roles


# ---------------------------------------------------------------------------
# 8. Analysis Context
# ---------------------------------------------------------------------------

@dataclass
class AnalysisContext:
    """The standardized context that future modules consume.

    Wraps a CanonicalDataset with its capabilities, quality, and semantic roles.
    Modules receive this instead of independently loading pd.read_csv().
    """
    dataset: CanonicalDataset
    capabilities: dict = field(default_factory=dict)  # from CapabilityReport.to_dict()
    semantic_roles: dict = field(default_factory=dict)
    quality: Optional[DataQuality] = None

    @classmethod
    def from_canonical(cls, canonical: CanonicalDataset) -> AnalysisContext:
        return cls(
            dataset=canonical,
            capabilities=canonical.capabilities or {},
            semantic_roles=canonical.semantic_roles,
            quality=canonical.data_quality,
        )

    def has_capability(self, key: str) -> bool:
        entry = self.capabilities.get(key)
        if entry is None:
            return False
        return bool(entry.get("available", False))

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self.dataset.dataframe

    def to_metadata_dict(self) -> dict:
        """Serialize the full context metadata (no DataFrame)."""
        return {
            "dataset": self.dataset.to_metadata_dict(),
            "capabilities": self.capabilities,
            "semantic_roles": self.semantic_roles,
            "quality": self.quality.to_dict() if self.quality else None,
        }


# ---------------------------------------------------------------------------
# 9. Builder — integrates with existing pipeline
# ---------------------------------------------------------------------------

def build_canonical(dataset_id: str, df: pd.DataFrame,
                     filename: str = "",
                     confirmed_roles: Optional[set] = None,
                     overrides: Optional[UserOverrides] = None) -> tuple[CanonicalDataset, AnalysisContext]:
    """Build a CanonicalDataset + AnalysisContext from a raw DataFrame.

    This is the primary entry point for creating canonical objects.  It runs
    the existing pipeline (profile → classify → standardize → capabilities)
    and wraps the result.
    """
    from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
    from src.data_engine.capability_engine import detect_capabilities

    profile, mapping = profile_and_map(df)

    pipeline_overrides = overrides.to_pipeline_overrides() if overrides and overrides.has_overrides() else None
    std_dataset, report = confirm_and_standardize(
        df, mapping,
        confirmed_roles=confirmed_roles,
        overrides=pipeline_overrides,
    )

    capability_report = detect_capabilities(std_dataset)

    canonical = CanonicalDataset.from_standardized(
        dataset_id=dataset_id,
        std_dataset=std_dataset,
        filename=filename,
        profile=profile,
        capability_report=capability_report,
        overrides=overrides,
    )

    context = AnalysisContext.from_canonical(canonical)
    return canonical, context
