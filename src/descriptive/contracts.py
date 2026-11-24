"""Typed, serializable result structures for Descriptive Statistics (Phase 8 Part 2).

Every statistic that cannot be calculated is stored as ``None`` together with a
plain-English reason in ``undefined`` - it is never replaced by 0 or NaN.
No Streamlit import: this package is pure computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

DEFAULT_PERCENTILES = (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0)


def _clean(value: Any) -> Any:
    """JSON-friendly copy: NaN/inf -> None, dataclasses/dicts/lists recursed."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _clean(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean(value.item())
        except Exception:
            return str(value)
    return value


class _Serializable:
    def to_dict(self) -> dict:
        return _clean(self)


@dataclass
class DescriptiveSettings(_Serializable):
    percentiles: tuple = DEFAULT_PERCENTILES
    outlier_multiplier: float = 1.5
    correlation_method: str = "pearson"          # "pearson" | "spearman"
    top_n: int = 5                                # top / least frequent categories shown
    max_frequency_rows: int = 50                  # rows kept per frequency table
    max_correlation_columns: int = 60             # numeric columns used in the matrices
    max_outlier_examples: int = 20
    high_cardinality_threshold: int = 50

    def validate(self) -> "DescriptiveSettings":
        pcts = sorted({float(p) for p in self.percentiles})
        if any(not (0.0 <= p <= 100.0) or math.isnan(p) for p in pcts):
            raise ValueError("Percentiles must be between 0 and 100.")
        if not (self.outlier_multiplier > 0 and math.isfinite(self.outlier_multiplier)):
            raise ValueError("The outlier multiplier must be a positive number.")
        if self.correlation_method not in ("pearson", "spearman"):
            raise ValueError("Correlation method must be 'pearson' or 'spearman'.")
        if self.top_n < 1 or self.max_frequency_rows < 1 or self.max_correlation_columns < 2:
            raise ValueError("top_n, max_frequency_rows and max_correlation_columns are too small.")
        return DescriptiveSettings(tuple(pcts), float(self.outlier_multiplier), self.correlation_method,
                                   int(self.top_n), int(self.max_frequency_rows),
                                   int(self.max_correlation_columns), int(self.max_outlier_examples),
                                   int(self.high_cardinality_threshold))


@dataclass
class NumericSummary(_Serializable):
    column: str
    kind: str = "numeric"
    dtype: str = ""
    sample_size: int = 0            # rows in the dataset (denominator of the missing %)
    valid_count: int = 0            # finite numeric observations used in every statistic
    missing_count: int = 0          # NaN / None / blank text
    blank_count: int = 0            # part of missing_count that was blank text
    missing_pct: float = 0.0        # missing_count / sample_size * 100
    infinite_count: int = 0         # +/-inf values (excluded, not counted as missing)
    zero_count: int = 0
    negative_count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    modes: list = field(default_factory=list)       # ascending, deterministic
    mode_count: Optional[int] = None
    mode_note: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    value_range: Optional[float] = None
    total: Optional[float] = None
    variance_sample: Optional[float] = None
    std_sample: Optional[float] = None
    variance_population: Optional[float] = None
    std_population: Optional[float] = None
    q1: Optional[float] = None
    q2: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    percentiles: dict = field(default_factory=dict)  # {percentile (float): value}
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None                 # excess kurtosis
    is_constant: bool = False
    converted_from_text: bool = False
    undefined: dict = field(default_factory=dict)    # statistic -> reason
    warnings: list = field(default_factory=list)


@dataclass
class CategoricalSummary(_Serializable):
    column: str
    kind: str = "categorical"
    dtype: str = ""
    sample_size: int = 0
    valid_count: int = 0
    missing_count: int = 0
    blank_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    modes: list = field(default_factory=list)
    mode_count: Optional[int] = None
    mode_pct: Optional[float] = None
    mode_note: str = ""
    top_categories: list = field(default_factory=list)      # [{"value","count","pct"}]
    least_frequent: list = field(default_factory=list)
    frequency: list = field(default_factory=list)            # first max_frequency_rows rows
    frequency_truncated: bool = False
    high_cardinality: bool = False
    undefined: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class BooleanSummary(_Serializable):
    column: str
    kind: str = "boolean"
    dtype: str = ""
    sample_size: int = 0
    valid_count: int = 0
    missing_count: int = 0
    missing_pct: float = 0.0
    true_count: int = 0
    false_count: int = 0
    true_pct: Optional[float] = None          # of valid observations
    false_pct: Optional[float] = None
    undefined: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class DatetimeSummary(_Serializable):
    column: str
    kind: str = "datetime"
    dtype: str = ""
    sample_size: int = 0
    valid_count: int = 0
    missing_count: int = 0
    blank_count: int = 0
    missing_pct: float = 0.0
    invalid_count: int = 0                    # text that could not be read as a date
    earliest: Optional[str] = None
    latest: Optional[str] = None
    range_days: Optional[float] = None
    unique_count: int = 0
    timezone: Optional[str] = None
    period: Optional[str] = None              # "day" | "month" | "year" | None
    period_counts: list = field(default_factory=list)   # [{"period","count"}]
    undefined: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class IdentifierSummary(_Serializable):
    column: str
    kind: str = "identifier"
    dtype: str = ""
    sample_size: int = 0
    valid_count: int = 0
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    note: str = ("Looks like an identifier (almost every value is different). It is described "
                 "by counts only and is not treated as a measurement.")


@dataclass
class MissingSummary(_Serializable):
    rows: list = field(default_factory=list)   # one dict per column
    total_cells: int = 0
    missing_cells: int = 0
    missing_cell_pct: float = 0.0
    complete_rows: int = 0
    complete_row_pct: float = 0.0
    rows_with_missing: int = 0


@dataclass
class OutlierSummary(_Serializable):
    column: str
    available: bool = False
    reason: str = ""
    valid_count: int = 0
    multiplier: float = 1.5
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    n_low: int = 0
    n_high: int = 0
    n_outliers: int = 0
    outlier_pct: Optional[float] = None       # of valid observations
    examples: list = field(default_factory=list)   # [{"row","value","side"}]
    examples_truncated: bool = False
    warnings: list = field(default_factory=list)


@dataclass
class PairwiseMatrix(_Serializable):
    """Correlation or covariance of numeric columns on pairwise complete observations."""
    kind: str                                  # "correlation" | "covariance_sample" | "covariance_population"
    method: str                                # "pearson" | "spearman" | "pearson (covariance)"
    columns: list = field(default_factory=list)
    values: list = field(default_factory=list)       # list of rows; None where undefined
    n_pairs: list = field(default_factory=list)      # pairwise complete observation counts
    reasons: dict = field(default_factory=dict)      # "a || b" -> why undefined
    excluded: dict = field(default_factory=dict)     # column -> why it was left out
    notes: list = field(default_factory=list)

    def value(self, a: str, b: str) -> Optional[float]:
        i, j = self.columns.index(a), self.columns.index(b)
        return self.values[i][j]


@dataclass
class DescriptiveReport(_Serializable):
    n_rows: int = 0
    n_cols: int = 0
    settings: dict = field(default_factory=dict)
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    boolean: list = field(default_factory=list)
    datetime: list = field(default_factory=list)
    identifiers: list = field(default_factory=list)
    empty_columns: list = field(default_factory=list)
    column_kinds: dict = field(default_factory=dict)          # column -> kind
    quality: dict = field(default_factory=dict)
    missing: Optional[MissingSummary] = None
    outliers: list = field(default_factory=list)
    correlation: Optional[PairwiseMatrix] = None
    covariance_sample: Optional[PairwiseMatrix] = None
    covariance_population: Optional[PairwiseMatrix] = None
    conversions: list = field(default_factory=list)           # every type decision, never silent
    warnings: list = field(default_factory=list)
