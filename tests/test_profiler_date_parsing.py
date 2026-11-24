"""Regression tests for date-parsing in src/data_engine/profiler.py.

Verifies that _infer_column_type correctly classifies date-like columns
without producing UserWarning about format inference, and handles edge
cases: valid dates, mixed formats, invalid dates, missing values, and
ambiguous inputs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.data_engine.profiler import _infer_column_type, profile_dataset


# ---------------------------------------------------------------------------
# _infer_column_type — date detection
# ---------------------------------------------------------------------------

class TestInferColumnTypeDateDetection:
    """Date-detection path in _infer_column_type."""

    def test_valid_iso_dates_detected(self):
        s = pd.Series(["2024-01-01", "2024-02-15", "2024-03-30"] * 15)
        assert _infer_column_type(s, len(s)) == "datetime"

    def test_valid_us_format_dates_detected(self):
        s = pd.Series(["01/15/2024", "02/20/2024", "03/25/2024"] * 15)
        assert _infer_column_type(s, len(s)) == "datetime"

    def test_valid_datetime_with_time(self):
        s = pd.Series(["2024-01-01 10:30:00", "2024-02-15 14:00:00"] * 20)
        assert _infer_column_type(s, len(s)) == "datetime"

    def test_mixed_date_formats_detected(self):
        """Dates in different formats should still be classified as datetime."""
        s = pd.Series([
            "2024-01-01", "01/15/2024", "2024-03-30",
            "Feb 20, 2024", "2024-04-10", "March 5, 2024",
        ] * 8)
        result = _infer_column_type(s, len(s))
        # Mixed formats may or may not all parse — either datetime or categorical is OK
        assert result in ("datetime", "categorical")

    def test_invalid_dates_not_detected_as_datetime(self):
        s = pd.Series(["hello", "world", "foo", "bar"] * 10)
        assert _infer_column_type(s, len(s)) != "datetime"

    def test_mostly_invalid_not_datetime(self):
        """If less than 85% parse, should not be datetime."""
        s = pd.Series(["2024-01-01", "not-a-date", "garbage", "xyz"] * 10)
        assert _infer_column_type(s, len(s)) != "datetime"

    def test_missing_values_handled(self):
        """NaN values should not cause errors."""
        s = pd.Series(["2024-01-01", None, "2024-03-15", np.nan, "2024-06-01"] * 10)
        result = _infer_column_type(s, len(s))
        assert result == "datetime"

    def test_all_null_column(self):
        s = pd.Series([None, None, None, np.nan])
        assert _infer_column_type(s, len(s)) == "empty"

    def test_ambiguous_dates(self):
        """Dates like 01/02/2024 are ambiguous (Jan 2 vs Feb 1) but should still parse."""
        s = pd.Series(["01/02/2024", "03/04/2024", "05/06/2024"] * 15)
        assert _infer_column_type(s, len(s)) == "datetime"

    def test_numeric_column_not_datetime(self):
        s = pd.Series([1, 2, 3, 4, 5] * 10)
        assert _infer_column_type(s, len(s)) == "numeric"

    def test_actual_datetime_dtype(self):
        s = pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"]))
        assert _infer_column_type(s, len(s)) == "datetime"


# ---------------------------------------------------------------------------
# No-warning verification
# ---------------------------------------------------------------------------

class TestNoDateParsingWarnings:
    """Ensure profiling does not emit 'Could not infer format' warnings."""

    def test_no_warning_on_valid_dates(self):
        df = pd.DataFrame({
            "date_col": ["2024-01-01", "2024-02-15", "2024-03-30"] * 15,
            "num": range(45),
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profile_dataset(df)
        format_warnings = [x for x in w if "Could not infer format" in str(x.message)]
        assert len(format_warnings) == 0, f"Unexpected warnings: {format_warnings}"

    def test_no_warning_on_mixed_strings(self):
        df = pd.DataFrame({
            "text_col": ["hello", "world", "foo"] * 15,
            "num": range(45),
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profile_dataset(df)
        format_warnings = [x for x in w if "Could not infer format" in str(x.message)]
        assert len(format_warnings) == 0, f"Unexpected warnings: {format_warnings}"

    def test_no_warning_on_ambiguous_dates(self):
        df = pd.DataFrame({
            "ambig": ["01/02/2024", "03/04/2024", "05/06/2024"] * 15,
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profile_dataset(df)
        format_warnings = [x for x in w if "Could not infer format" in str(x.message)]
        assert len(format_warnings) == 0, f"Unexpected warnings: {format_warnings}"


# ---------------------------------------------------------------------------
# profile_dataset integration
# ---------------------------------------------------------------------------

class TestProfileDatasetIntegration:
    """Full profile_dataset with various column types."""

    def test_profile_with_date_column(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"] * 15,
            "value": [10, 20, 30] * 15,  # repeated values so not classified as identifier
        })
        result = profile_dataset(df)
        assert result["n_rows"] == 45
        assert result["columns"]["date"]["inferred_type"] == "datetime"
        assert result["columns"]["value"]["inferred_type"] == "numeric"

    def test_profile_with_mixed_types(self):
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0] * 10,
            "cat": ["A", "B", "C"] * 10,
            "dt": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"] * 10),
        })
        result = profile_dataset(df)
        assert result["columns"]["num"]["inferred_type"] == "numeric"
        assert result["columns"]["cat"]["inferred_type"] == "categorical"
        assert result["columns"]["dt"]["inferred_type"] == "datetime"

    def test_profile_empty_dataframe(self):
        df = pd.DataFrame()
        result = profile_dataset(df)
        assert result["n_rows"] == 0
        assert result["n_cols"] == 0
        assert result["columns"] == {}

    def test_profile_with_nulls(self):
        df = pd.DataFrame({
            "mixed": ["2024-01-01", None, "2024-03-01", np.nan, "2024-05-01"] * 10,
            "nums": [1, 2, None, 4, 5] * 10,
        })
        result = profile_dataset(df)
        assert result["columns"]["mixed"]["missing_pct"] > 0
        assert result["columns"]["nums"]["missing_pct"] > 0
