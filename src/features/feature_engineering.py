"""
MarketPulse AI - Feature Engineering
=======================================
Takes the cleaned stage-1 dataset and derives analysis-ready features:
time features, aggregate spend metrics, spend shares, growth rates, and
average order value. Writes the final processed dataset used by every
downstream module (analytics, MMM, optimization, dashboard, API).

Run:
    python src/features/feature_engineering.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE1_PATH = ROOT / "data" / "processed" / "marketing_data_cleaned_stage1.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "marketing_data_clean.csv"

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]
SPEND_COLS = [f"{c}_Spend" for c in CHANNELS]
DIGITAL_CHANNELS = ["Search", "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing"]
TRADITIONAL_CHANNELS = ["TV", "Radio", "Print"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Time features ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfYear"] = df["Week"]

    # --- Aggregate spend metrics ---
    df["Total_Marketing_Spend"] = df[SPEND_COLS].sum(axis=1)
    df["Digital_Spend"] = df[[f"{c}_Spend" for c in DIGITAL_CHANNELS]].sum(axis=1)
    df["Traditional_Spend"] = df[[f"{c}_Spend" for c in TRADITIONAL_CHANNELS]].sum(axis=1)
    df["Digital_Spend_Share"] = df["Digital_Spend"] / df["Total_Marketing_Spend"].replace(0, np.nan)
    df["Traditional_Spend_Share"] = df["Traditional_Spend"] / df["Total_Marketing_Spend"].replace(0, np.nan)

    # --- Per-channel spend share of total ---
    for ch in CHANNELS:
        df[f"{ch}_Spend_Share"] = df[f"{ch}_Spend"] / df["Total_Marketing_Spend"].replace(0, np.nan)

    # --- Growth rates (week-over-week) ---
    df["Revenue_Growth_WoW"] = df["Revenue"].pct_change()
    df["Marketing_Growth_WoW"] = df["Total_Marketing_Spend"].pct_change()
    df["Customer_Growth_WoW"] = (df["New_Customers"] + df["Returning_Customers"]).pct_change()

    # --- Average order value & customer metrics ---
    df["Total_Customers"] = df["New_Customers"] + df["Returning_Customers"]
    df["AOV"] = df["Revenue"] / df["Orders"].replace(0, np.nan)
    df["New_Customer_Share"] = df["New_Customers"] / df["Total_Customers"].replace(0, np.nan)

    # --- Naive overall ROAS (Revenue / Total Marketing Spend) ---
    df["Naive_ROAS"] = df["Revenue"] / df["Total_Marketing_Spend"].replace(0, np.nan)

    # Fill any NaNs introduced by pct_change / division on first row or zero-division
    growth_cols = ["Revenue_Growth_WoW", "Marketing_Growth_WoW", "Customer_Growth_WoW"]
    df[growth_cols] = df[growth_cols].fillna(0)

    return df


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "Date": "Week start date (Monday)",
        "Revenue": "Total weekly revenue (INR), the target variable for MMM",
        "Orders": "Total weekly orders",
        "Average_Price": "Average selling price for the week (INR)",
        "Website_Traffic": "Estimated weekly website sessions",
        "New_Customers": "New customers acquired in the week",
        "Returning_Customers": "Returning customers in the week",
        "Holiday": "1 if the week overlaps a holiday/sale period, else 0",
        "Season": "Calendar season label",
        "Competitor_Index": "Synthetic index of competitor pricing/activity intensity",
        "Economic_Index": "Synthetic macroeconomic conditions index",
        "Promotion_Spend": "Weekly promotional/discount spend (INR)",
        "Total_Marketing_Spend": "Sum of all 8 channel spends (INR)",
        "Digital_Spend": "Sum of Search, Social, Display, Email, Influencer spend",
        "Traditional_Spend": "Sum of TV, Radio, Print spend",
        "AOV": "Average Order Value = Revenue / Orders",
        "Naive_ROAS": "Revenue / Total Marketing Spend (undecomposed, observational)",
    }
    for ch in CHANNELS:
        descriptions[f"{ch}_Spend"] = f"Weekly spend on {ch.replace('_',' ')} (INR)"
        descriptions[f"{ch}_Spend_Share"] = f"{ch.replace('_',' ')} spend as a share of total marketing spend"

    rows = []
    for col in df.columns:
        rows.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Description": descriptions.get(col, "Derived analytical feature"),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(STAGE1_PATH, parse_dates=["Date"])
    engineered = engineer_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(OUTPUT_PATH, index=False)

    data_dict = build_data_dictionary(engineered)
    dict_path = ROOT / "docs" / "data_dictionary.csv"
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    data_dict.to_csv(dict_path, index=False)

    print(f"Feature-engineered dataset: {engineered.shape[0]} rows x {engineered.shape[1]} cols")
    print(f"Written to: {OUTPUT_PATH}")
    print(f"Data dictionary written to: {dict_path}")


if __name__ == "__main__":
    main()
