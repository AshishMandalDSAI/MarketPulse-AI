"""Tests for M-1 fix: duplicate column name handling in analysis_runner.

Covers:
- Simple duplicate pairs
- Multiple occurrences of the same name
- Collision with suffix-generated names (e.g. column 'A_1' already exists)
- Column order and data preservation
- No-op when no duplicates exist
- Integration with run_full_analysis via CSV upload path
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.analysis_runner import normalize_duplicate_columns, run_full_analysis


class TestNormalizeDuplicateColumns:
    """Unit tests for normalize_duplicate_columns()."""

    def test_no_duplicates_returns_none_map(self):
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        result_df, rename_map = normalize_duplicate_columns(df)
        assert rename_map is None
        assert list(result_df.columns) == ["A", "B", "C"]

    def test_simple_duplicate_pair(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "A"])
        result_df, rename_map = normalize_duplicate_columns(df)
        assert list(result_df.columns) == ["A", "B", "A_1"]
        assert rename_map == {"A_1": "A"}
        # Data preserved
        assert result_df.iloc[0].tolist() == [1, 2, 3]

    def test_triple_duplicate(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["X", "X", "X"])
        result_df, rename_map = normalize_duplicate_columns(df)
        assert list(result_df.columns) == ["X", "X_1", "X_2"]
        assert rename_map == {"X_1": "X", "X_2": "X"}
        assert result_df.iloc[0].tolist() == [1, 2, 3]

    def test_multiple_different_duplicates(self):
        df = pd.DataFrame([[1, 2, 3, 4]], columns=["A", "B", "A", "B"])
        result_df, rename_map = normalize_duplicate_columns(df)
        assert list(result_df.columns) == ["A", "B", "A_1", "B_1"]
        assert rename_map == {"A_1": "A", "B_1": "B"}

    def test_collision_with_existing_suffix_name(self):
        """If 'A_1' already exists as a real column, the generated name
        must skip to 'A_2' to avoid collision."""
        df = pd.DataFrame([[1, 2, 3]], columns=["A", "A_1", "A"])
        result_df, rename_map = normalize_duplicate_columns(df)
        cols = list(result_df.columns)
        assert cols[0] == "A"
        assert cols[1] == "A_1"  # original A_1 keeps its name
        assert cols[2] == "A_2"  # duplicate A skips A_1 → becomes A_2
        assert rename_map == {"A_2": "A"}
        assert result_df.iloc[0].tolist() == [1, 2, 3]

    def test_deep_collision_chain(self):
        """A, A_1, A_2 all exist → duplicate A becomes A_3."""
        df = pd.DataFrame([[1, 2, 3, 4]], columns=["A", "A_1", "A_2", "A"])
        result_df, rename_map = normalize_duplicate_columns(df)
        assert list(result_df.columns) == ["A", "A_1", "A_2", "A_3"]
        assert rename_map == {"A_3": "A"}

    def test_column_order_preserved(self):
        df = pd.DataFrame([[10, 20, 30, 40, 50]], columns=["Z", "M", "Z", "A", "M"])
        result_df, _ = normalize_duplicate_columns(df)
        assert list(result_df.columns) == ["Z", "M", "Z_1", "A", "M_1"]
        assert result_df.iloc[0].tolist() == [10, 20, 30, 40, 50]

    def test_data_values_preserved_multirow(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Manually create duplicates
        df2 = pd.concat([df, df[["A"]]], axis=1)
        assert list(df2.columns) == ["A", "B", "A"]
        result_df, rename_map = normalize_duplicate_columns(df2)
        assert result_df["A"].tolist() == [1, 2, 3]
        assert result_df["B"].tolist() == [4, 5, 6]
        assert result_df["A_1"].tolist() == [1, 2, 3]

    def test_original_df_not_mutated(self):
        df = pd.DataFrame([[1, 2]], columns=["A", "A"])
        original_cols = list(df.columns)
        normalize_duplicate_columns(df)
        assert list(df.columns) == original_cols

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result_df, rename_map = normalize_duplicate_columns(df)
        assert rename_map is None
        assert list(result_df.columns) == []


class TestDuplicateColumnsIntegration:
    """Integration tests: duplicate-column CSV through run_full_analysis."""

    @pytest.fixture
    def dup_csv(self, tmp_path):
        """Write a CSV with duplicate column names to a temp dir."""
        csv_path = tmp_path / "raw" / "upload.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Write raw so pandas reads duplicate names
        csv_path.write_text("A,B,A\n1,2,3\n4,5,6\n7,8,9\n")
        return tmp_path, csv_path

    @patch("src.data_engine.analysis_runner.ensure_dataset_dirs")
    def test_duplicate_col_csv_runs_successfully(self, mock_dirs, dup_csv):
        """CSV with duplicate columns completes pipeline without error.

        Note: pd.read_csv auto-suffixes duplicates (A, A.1), so pandas
        handles the CSV-layer dedup. Our guard catches the case where a
        DataFrame is passed directly with true duplicate column objects.
        """
        tmp_path, csv_path = dup_csv
        processed = tmp_path / "processed"
        processed.mkdir(exist_ok=True)
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)

        with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=processed), \
             patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models), \
             patch("src.data_engine.analysis_runner.dataset_raw_path") as mock_raw:
            mock_raw.return_value = csv_path
            result = run_full_analysis("test_dup_cols", df=pd.read_csv(csv_path))

        assert result is not None
        assert result["dataset"] is not None

    @patch("src.data_engine.analysis_runner.ensure_dataset_dirs")
    def test_duplicate_col_df_direct_reports_normalization(self, mock_dirs, tmp_path):
        """When a DataFrame with true duplicate column names is passed directly,
        the report should mention normalization."""
        processed = tmp_path / "processed"
        processed.mkdir(exist_ok=True)
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)

        # Build a DataFrame with actual duplicate column names
        df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "A"])

        with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=processed), \
             patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models):
            result = run_full_analysis("test_dup_direct", df=df)

        assert result is not None
        combined_report = " ".join(result["report"])
        assert "Duplicate column names detected" in combined_report

    @patch("src.data_engine.analysis_runner.ensure_dataset_dirs")
    def test_no_duplicate_csv_no_rename_report(self, mock_dirs, tmp_path):
        """Clean CSV produces no duplicate-column report entry."""
        processed = tmp_path / "processed"
        processed.mkdir(exist_ok=True)
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)

        df = pd.DataFrame({"Sales": [100, 200], "Region": ["A", "B"], "Month": [1, 2]})

        with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=processed), \
             patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models):
            result = run_full_analysis("test_no_dup", df=df)

        combined_report = " ".join(result["report"])
        assert "Duplicate column names" not in combined_report

    @patch("src.data_engine.analysis_runner.ensure_dataset_dirs")
    def test_standardized_csv_has_unique_columns(self, mock_dirs, tmp_path):
        """After normalization, the persisted standardized_data.csv has unique columns."""
        processed = tmp_path / "processed"
        processed.mkdir(exist_ok=True)
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)

        df = pd.DataFrame([[10, 20, 30]], columns=["Val", "Val", "Val"])

        with patch("src.data_engine.analysis_runner.dataset_processed_dir", return_value=processed), \
             patch("src.data_engine.analysis_runner.dataset_models_dir", return_value=models):
            run_full_analysis("test_dup_persist", df=df)

        saved = pd.read_csv(processed / "standardized_data.csv")
        assert len(saved.columns) == len(set(saved.columns)), "Persisted CSV still has duplicate columns"
