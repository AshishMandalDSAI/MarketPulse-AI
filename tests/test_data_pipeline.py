"""Tests for dataset generation, cleaning, and validation.

Run with pytest (pip install -r requirements.txt) or standalone via
tests/run_all_tests.py in environments without pytest installed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.generate_dataset import generate_dataset, CHANNELS as GEN_CHANNELS
from src.data.data_cleaning import validate, clean, DataValidationError


def test_generate_dataset_shape():
    df = generate_dataset(seed=1, n_weeks=52)
    assert len(df) == 52
    assert "Revenue" in df.columns
    for ch in GEN_CHANNELS:
        assert f"{ch}_Spend" in df.columns


def test_generate_dataset_no_negative_core_values():
    df = generate_dataset(seed=2, n_weeks=52)
    assert (df["Revenue"] >= 0).all()
    assert (df["Orders"] >= 0).all()
    for ch in GEN_CHANNELS:
        assert (df[f"{ch}_Spend"] >= 0).all()


def test_generate_dataset_reproducible_with_seed():
    df1 = generate_dataset(seed=42, n_weeks=52)
    df2 = generate_dataset(seed=42, n_weeks=52)
    pd.testing.assert_frame_equal(df1, df2)


def test_validate_detects_missing_columns():
    df = pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=5)})
    try:
        validate(df)
        assert False, "Expected DataValidationError"
    except DataValidationError:
        pass


def test_validate_reports_missing_values():
    df = generate_dataset(seed=3, n_weeks=52)
    df.loc[0, "Competitor_Index"] = np.nan
    report = validate(df)
    assert report["missing_values"].get("Competitor_Index", 0) >= 1


def test_clean_fills_missing_values():
    df = generate_dataset(seed=4, n_weeks=52)
    df.loc[0, "Competitor_Index"] = np.nan
    cleaned, report = clean(df)
    assert cleaned["Competitor_Index"].isna().sum() == 0


def test_clean_removes_duplicate_rows():
    df = generate_dataset(seed=5, n_weeks=52)
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned, report = clean(df_with_dupe)
    assert len(cleaned) == len(df)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}")
