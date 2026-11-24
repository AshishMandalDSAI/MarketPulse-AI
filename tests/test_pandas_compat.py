"""Regression tests for Pandas 2.x / 3.x compatibility in preprocessing.

These tests verify that ``select_dtypes``-based column detection works
correctly regardless of the installed Pandas version.  The original code
used ``include=["object", "str"]`` which raises ``TypeError`` on Pandas 2.x
because ``"str"`` is not a recognised dtype there.

The fix introduced ``_string_and_object_dtypes()`` and
``_categorical_dtypes()`` helpers that build the include list dynamically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.preprocessing import (
    _categorical_dtypes,
    _string_and_object_dtypes,
    preprocess_for_prediction,
    preprocess_for_segmentation,
)


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------

class TestDtypeHelpers:
    """Verify the compatibility helper functions themselves."""

    def test_string_and_object_always_includes_object(self):
        dtypes = _string_and_object_dtypes()
        assert "object" in dtypes

    def test_string_and_object_returns_list(self):
        dtypes = _string_and_object_dtypes()
        assert isinstance(dtypes, list)
        assert len(dtypes) >= 1

    def test_categorical_includes_object(self):
        dtypes = _categorical_dtypes()
        assert "object" in dtypes

    def test_categorical_includes_category_and_bool(self):
        dtypes = _categorical_dtypes()
        assert "category" in dtypes
        assert "bool" in dtypes

    def test_categorical_is_superset_of_string_and_object(self):
        string_obj = set(_string_and_object_dtypes())
        cat = set(_categorical_dtypes())
        assert string_obj.issubset(cat)

    def test_select_dtypes_with_string_and_object_no_error(self):
        """select_dtypes must not raise TypeError with the helper output."""
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        # Should not raise
        result = df.select_dtypes(include=_string_and_object_dtypes())
        assert "a" in result.columns
        assert "b" not in result.columns

    def test_select_dtypes_with_categorical_no_error(self):
        """select_dtypes must not raise TypeError with the helper output."""
        df = pd.DataFrame({
            "txt": ["x", "y"],
            "cat": pd.Categorical(["a", "b"]),
            "flag": [True, False],
            "num": [1.0, 2.0],
        })
        result = df.select_dtypes(include=_categorical_dtypes())
        assert "txt" in result.columns
        assert "cat" in result.columns
        assert "flag" in result.columns
        assert "num" not in result.columns


# ---------------------------------------------------------------------------
# Dtype detection in select_dtypes
# ---------------------------------------------------------------------------

class TestDtypeDetection:
    """Ensure different column types are correctly detected."""

    def test_object_columns_detected(self):
        df = pd.DataFrame({"s": ["hello", "world"], "n": [1, 2]})
        cols = df.select_dtypes(include=_string_and_object_dtypes()).columns.tolist()
        assert "s" in cols

    def test_numeric_columns_not_detected_as_text(self):
        df = pd.DataFrame({"n": [1, 2, 3], "f": [1.1, 2.2, 3.3]})
        cols = df.select_dtypes(include=_string_and_object_dtypes()).columns.tolist()
        assert cols == []

    def test_datetime_columns_not_detected_as_text(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01", "2024-02-01"])})
        cols = df.select_dtypes(include=_string_and_object_dtypes()).columns.tolist()
        assert cols == []

    def test_categorical_column_detected(self):
        df = pd.DataFrame({"c": pd.Categorical(["a", "b", "c"])})
        cols = df.select_dtypes(include=_categorical_dtypes()).columns.tolist()
        assert "c" in cols

    def test_bool_column_detected(self):
        df = pd.DataFrame({"b": [True, False, True]})
        cols = df.select_dtypes(include=_categorical_dtypes()).columns.tolist()
        assert "b" in cols

    def test_mixed_dtypes_separated_correctly(self):
        df = pd.DataFrame({
            "text": ["a", "b"],
            "num": [1, 2],
            "cat": pd.Categorical(["x", "y"]),
            "dt": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "flag": [True, False],
        })
        text_cols = df.select_dtypes(include=_string_and_object_dtypes()).columns.tolist()
        cat_cols = df.select_dtypes(include=_categorical_dtypes()).columns.tolist()

        assert "text" in text_cols
        assert "num" not in text_cols
        assert "dt" not in text_cols

        assert "text" in cat_cols
        assert "cat" in cat_cols
        assert "flag" in cat_cols
        assert "num" not in cat_cols
        assert "dt" not in cat_cols


# ---------------------------------------------------------------------------
# Integration: preprocessing pipelines with various dtypes
# ---------------------------------------------------------------------------

class TestPreprocessingPandasCompat:
    """End-to-end tests that the preprocessing pipelines work with the fix."""

    @pytest.fixture()
    def mixed_df(self):
        """DataFrame with string, numeric, datetime-like, and categorical cols."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            "revenue": np.random.uniform(100, 1000, n),
            "clicks": np.random.randint(10, 500, n),
            "region": np.random.choice(["US", "EU", "APAC"], n),
            "date_str": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "is_active": np.random.choice([True, False], n),
            "segment": pd.Categorical(np.random.choice(["A", "B", "C"], n)),
        })

    def test_prediction_no_type_error(self, mixed_df):
        """preprocess_for_prediction must not raise TypeError on select_dtypes."""
        X, y, features, report = preprocess_for_prediction(
            mixed_df,
            target_column="revenue",
            feature_columns=["clicks", "region", "date_str", "is_active", "segment"],
        )
        assert len(X) > 0
        assert len(y) > 0
        assert len(features) > 0

    def test_segmentation_no_type_error(self, mixed_df):
        """preprocess_for_segmentation must not raise TypeError on select_dtypes."""
        X, features, report = preprocess_for_segmentation(
            mixed_df,
            feature_columns=["clicks", "region", "date_str", "is_active", "segment"],
        )
        assert len(X) > 0
        assert len(features) > 0

    def test_prediction_detects_datetime_strings(self, mixed_df):
        """Date-like string columns should be detected and converted."""
        X, y, features, report = preprocess_for_prediction(
            mixed_df,
            target_column="revenue",
            feature_columns=["clicks", "date_str"],
        )
        # date_str should have been parsed → datetime features added
        assert len(report["datetime_features_added"]) > 0

    def test_prediction_encodes_categorical(self, mixed_df):
        """Categorical and string columns should be one-hot encoded."""
        X, y, features, report = preprocess_for_prediction(
            mixed_df,
            target_column="revenue",
            feature_columns=["clicks", "region", "segment"],
        )
        assert len(report["encoded_columns"]) > 0

    def test_segmentation_handles_string_columns(self, mixed_df):
        """String columns should not cause TypeError in segmentation."""
        X, features, report = preprocess_for_segmentation(
            mixed_df,
            feature_columns=["clicks", "date_str", "region"],
        )
        # Segmentation keeps only numeric columns, so string cols are dropped.
        # The key assertion is that no TypeError was raised.
        assert len(X) > 0
        assert "clicks" in features or any("date_str" in f for f in features)

    def test_all_numeric_df_no_error(self):
        """A purely numeric DataFrame should not trigger dtype issues."""
        np.random.seed(99)
        n = 100
        df = pd.DataFrame({
            "target": np.random.randn(n),
            "f1": np.random.choice([1, 2, 3, 4, 5], n),
            "f2": np.random.choice([10, 20, 30, 40, 50], n),
        })
        X, y, features, report = preprocess_for_prediction(
            df, target_column="target", feature_columns=["f1", "f2"],
        )
        assert len(X) == n
        # Features may be reduced by leakage/constant detection, but should not error
        assert len(features) >= 1

    def test_empty_df_no_error(self):
        """An empty DataFrame should not crash."""
        df = pd.DataFrame(columns=["target", "f1", "f2"])
        X, y, features, report = preprocess_for_prediction(
            df, target_column="target", feature_columns=["f1", "f2"],
        )
        assert len(X) == 0
