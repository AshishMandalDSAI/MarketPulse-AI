"""Data-availability gate for the Phase 8 modules.

This is a coarse, honest first gate: it only asks "is there enough of the right
kind of data for this module to be meaningful at all?". Each module applies its
own stricter checks when it runs (e.g. a specific test's assumptions, a model's
minimum rows, the optimizer's need for a fitted response model) and those are
added by the part that delivers the module.

Descriptive availability is deliberately NOT about variation: a constant
column still has a valid count, mean, median, mode, minimum, maximum and
frequency table. Analyses that genuinely need variation (correlation, skewness
and kurtosis, ...) are decided by the Descriptive module itself when it is
delivered; they are not gated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.dataset_summary import DatasetSummary

KNOWN_MODULES = ("descriptive", "inferential", "predictive", "prescriptive")


@dataclass
class Availability:
    available: bool
    reasons: list[str] = field(default_factory=list)   # why it is unavailable
    missing: list[str] = field(default_factory=list)   # what the user would need to provide


def _usable_columns(s: DatasetSummary) -> int:
    """Columns that are neither entirely empty nor constant (can act as a
    target or a feature)."""
    excluded = set(s.empty_columns) | set(s.constant_columns)
    return sum(1 for c in s.columns if c["column"] not in excluded)


def _non_empty_columns(s: DatasetSummary) -> int:
    """Columns holding at least one real (non-missing) value, constant or not."""
    empty = set(s.empty_columns)
    return sum(1 for c in s.columns
               if c["column"] not in empty and c["missing_pct"] < 100.0)


def _varying_numeric_columns(s: DatasetSummary) -> int:
    constant = set(s.constant_columns)
    return sum(1 for c in s.numeric_columns if c not in constant)


def check_availability(module_key: str, summary: DatasetSummary | None) -> Availability:
    """Decide whether ``module_key`` can meaningfully run on ``summary``.

    An unknown module key always yields an "unknown module" result, whatever the
    dataset looks like. ``None`` or an empty summary means "no usable dataset"
    and yields an unavailable result. Any other non-``DatasetSummary`` input is a
    programming error and raises ``TypeError``.
    """
    if not isinstance(module_key, str) or module_key not in KNOWN_MODULES:
        return Availability(False, [f"Unknown module '{module_key}'."], [])

    if summary is not None and not isinstance(summary, DatasetSummary):
        raise TypeError("check_availability needs a DatasetSummary (from summarize_dataset) "
                        f"or None, not {type(summary).__name__}.")

    if summary is None or summary.is_empty:
        return Availability(False, ["The active dataset has no rows or no columns."],
                            ["A dataset with at least one row and one column."])

    reasons: list[str] = []
    missing: list[str] = []

    if module_key == "descriptive":
        if _non_empty_columns(summary) < 1:
            reasons.append("Every column is entirely empty (no values at all), so there is "
                           "nothing to describe.")
            missing.append("At least one column that contains values.")
    elif module_key == "inferential":
        if summary.n_rows < 3:
            reasons.append(f"Only {summary.n_rows} row(s) found; hypothesis tests need at least 3.")
            missing.append("At least 3 rows (most tests need considerably more).")
        if not (summary.numeric_columns or summary.categorical_columns or summary.boolean_columns):
            reasons.append("No numeric, categorical or yes/no column was detected.")
            missing.append("At least one numeric, categorical or yes/no column.")
    elif module_key == "predictive":
        if summary.n_rows < 10:
            reasons.append(f"Only {summary.n_rows} row(s) found; a model cannot be trained and "
                           "tested reliably on so little data.")
            missing.append("At least 10 rows (far more for dependable models).")
        if _usable_columns(summary) < 2:
            reasons.append("Prediction needs a target column plus at least one varying feature column.")
            missing.append("At least two columns with varying values.")
    elif module_key == "prescriptive":
        if summary.n_rows < 10:
            reasons.append(f"Only {summary.n_rows} row(s) found; a response model cannot be fitted.")
            missing.append("At least 10 rows.")
        if _varying_numeric_columns(summary) < 2:
            reasons.append("Optimization needs an outcome to improve and at least one controllable "
                           "input, both numeric columns whose values vary (a constant column "
                           "cannot serve as either).")
            missing.append("At least two numeric columns whose values vary: an outcome and one "
                           "controllable input.")

    return Availability(not reasons, reasons, missing)
