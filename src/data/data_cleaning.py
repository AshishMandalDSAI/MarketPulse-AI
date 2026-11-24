"""
MarketPulse AI - Data Cleaning & Validation
=============================================
Loads the raw synthetic marketing dataset, validates it, fixes issues, and
writes an intermediate cleaned file. Feature engineering happens separately
in src/features/feature_engineering.py.

Run:
    python src/data/data_cleaning.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "marketing_data_raw.csv"

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]
SPEND_COLS = [f"{c}_Spend" for c in CHANNELS]


class DataValidationError(Exception):
    pass


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run src/data/generate_dataset.py first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def validate(df: pd.DataFrame) -> dict:
    """Run a battery of validation checks and return a report dict.
    Raises DataValidationError on unrecoverable structural problems.
    """
    report = {}

    required_cols = SPEND_COLS + [
        "Date", "Revenue", "Orders", "Average_Price", "Holiday", "Season",
        "Promotion_Spend", "Competitor_Index", "Economic_Index",
        "Website_Traffic", "New_Customers", "Returning_Customers",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    report["missing_columns"] = missing_cols

    # Date checks
    if df["Date"].isna().any():
        raise DataValidationError("Null dates found.")
    # Duplicate dates are reported (and resolved during clean() by dropping
    # duplicate rows), not treated as a hard structural failure — a duplicated
    # *row* for the same date is a data-quality issue to fix, not a reason to
    # abort the whole pipeline.
    report["duplicate_dates"] = int(df["Date"].duplicated().sum())
    if not df["Date"].is_monotonic_increasing:
        report["date_order_fixed"] = True
    else:
        report["date_order_fixed"] = False

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    report["duplicate_rows"] = dup_count

    # Missing values per column
    na_counts = df.isna().sum()
    report["missing_values"] = {k: int(v) for k, v in na_counts[na_counts > 0].items()}

    # Negative spend checks
    negative_spend = {}
    for col in SPEND_COLS + ["Promotion_Spend"]:
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            negative_spend[col] = n_neg
    report["negative_spend_rows"] = negative_spend

    # Negative revenue / orders
    report["negative_revenue_rows"] = int((df["Revenue"] < 0).sum())
    report["negative_orders_rows"] = int((df["Orders"] < 0).sum())

    # Outlier check via z-score on Revenue (informational only, not dropped)
    z = (df["Revenue"] - df["Revenue"].mean()) / df["Revenue"].std()
    report["revenue_outliers_z3"] = int((z.abs() > 3).sum())

    return report


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = validate(df)

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates()

    # Impute missing numeric values with forward-fill then median (time series
    # appropriate: carry last known value, fall back to column median)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Clip any negative spend/revenue/orders to zero (defensive; shouldn't
    # trigger on the synthetic generator output, but protects real-world reuse)
    for col in SPEND_COLS + ["Promotion_Spend", "Revenue", "Orders"]:
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            df[col] = df[col].clip(lower=0)
            report.setdefault("clipped_negative", {})[col] = n_neg

    report["rows_after_cleaning"] = len(df)
    return df, report


def main():
    df = load_raw()
    cleaned, report = clean(df)

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "marketing_data_cleaned_stage1.csv"
    cleaned.to_csv(out_path, index=False)

    print("Data Validation Report")
    print("=" * 50)
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"\nCleaned stage-1 file written to: {out_path}")


if __name__ == "__main__":
    main()
