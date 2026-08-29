"""
MarketPulse AI - Marketing Analytics Engine
==============================================
Computes all descriptive/observational business analytics from the cleaned
dataset: KPIs, channel spend & share, naive ROAS, correlations, and
monthly/quarterly/yearly/holiday/promotion breakdowns.

These are OBSERVATIONAL metrics only (no causal model). Model-based
contribution/attribution lives in src/models/channel_contribution.py.

Run:
    python src/analytics/marketing_analytics.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "marketing_data_clean.csv"

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]
SPEND_COLS = [f"{c}_Spend" for c in CHANNELS]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    total_revenue = float(df["Revenue"].sum())
    total_spend = float(df["Total_Marketing_Spend"].sum())
    n_weeks = len(df)
    half = n_weeks // 2

    first_half_rev = df["Revenue"].iloc[:half].sum()
    second_half_rev = df["Revenue"].iloc[half:].sum()
    revenue_growth = (second_half_rev - first_half_rev) / first_half_rev if first_half_rev else 0.0

    first_half_spend = df["Total_Marketing_Spend"].iloc[:half].sum()
    second_half_spend = df["Total_Marketing_Spend"].iloc[half:].sum()
    marketing_growth = (second_half_spend - first_half_spend) / first_half_spend if first_half_spend else 0.0

    kpis = {
        "total_revenue": total_revenue,
        "total_marketing_spend": total_spend,
        "avg_weekly_revenue": float(df["Revenue"].mean()),
        "avg_weekly_marketing_spend": float(df["Total_Marketing_Spend"].mean()),
        "revenue_growth_h1_to_h2": float(revenue_growth),
        "marketing_growth_h1_to_h2": float(marketing_growth),
        "total_orders": float(df["Orders"].sum()),
        "total_customers": float(df["Total_Customers"].sum()),
        "avg_order_value": float(df["Revenue"].sum() / df["Orders"].sum()),
        "overall_naive_roas": float(total_revenue / total_spend) if total_spend else 0.0,
        "n_weeks": n_weeks,
        "date_range": {"start": str(df["Date"].min().date()), "end": str(df["Date"].max().date())},
    }
    return kpis


def compute_channel_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_spend_all = df["Total_Marketing_Spend"].sum()
    for ch in CHANNELS:
        spend_col = f"{ch}_Spend"
        ch_spend = df[spend_col].sum()
        corr = df[spend_col].corr(df["Revenue"])
        rows.append({
            "Channel": ch.replace("_", " "),
            "Total_Spend": float(ch_spend),
            "Spend_Share_Pct": float(ch_spend / total_spend_all * 100) if total_spend_all else 0.0,
            "Avg_Weekly_Spend": float(df[spend_col].mean()),
            "Revenue_Correlation": float(corr) if not np.isnan(corr) else 0.0,
            "Naive_ROAS": float(df["Revenue"].sum() / ch_spend) if ch_spend else 0.0,
        })
    perf = pd.DataFrame(rows).sort_values("Total_Spend", ascending=False).reset_index(drop=True)
    perf["Rank_By_Spend"] = perf.index + 1
    return perf


def monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby([df["Date"].dt.to_period("M")]).agg(
        Revenue=("Revenue", "sum"),
        Marketing_Spend=("Total_Marketing_Spend", "sum"),
        Orders=("Orders", "sum"),
    ).reset_index()
    g["Date"] = g["Date"].astype(str)
    return g


def quarterly_performance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby([df["Date"].dt.to_period("Q")]).agg(
        Revenue=("Revenue", "sum"),
        Marketing_Spend=("Total_Marketing_Spend", "sum"),
        Orders=("Orders", "sum"),
    ).reset_index()
    g["Date"] = g["Date"].astype(str)
    return g


def yearly_performance(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df["Date"].dt.year).agg(
        Revenue=("Revenue", "sum"),
        Marketing_Spend=("Total_Marketing_Spend", "sum"),
        Orders=("Orders", "sum"),
    ).reset_index().rename(columns={"Date": "Year"})
    return g


def holiday_analysis(df: pd.DataFrame) -> dict:
    hol = df[df["Holiday"] == 1]
    non_hol = df[df["Holiday"] == 0]
    return {
        "holiday_weeks": int(len(hol)),
        "non_holiday_weeks": int(len(non_hol)),
        "avg_revenue_holiday": float(hol["Revenue"].mean()) if len(hol) else 0.0,
        "avg_revenue_non_holiday": float(non_hol["Revenue"].mean()) if len(non_hol) else 0.0,
        "holiday_revenue_lift_pct": float(
            (hol["Revenue"].mean() - non_hol["Revenue"].mean()) / non_hol["Revenue"].mean() * 100
        ) if len(non_hol) and non_hol["Revenue"].mean() else 0.0,
    }


def promotion_analysis(df: pd.DataFrame) -> dict:
    promo = df[df["Promotion_Spend"] > df["Promotion_Spend"].median()]
    low_promo = df[df["Promotion_Spend"] <= df["Promotion_Spend"].median()]
    return {
        "high_promo_weeks": int(len(promo)),
        "low_promo_weeks": int(len(low_promo)),
        "avg_revenue_high_promo": float(promo["Revenue"].mean()) if len(promo) else 0.0,
        "avg_revenue_low_promo": float(low_promo["Revenue"].mean()) if len(low_promo) else 0.0,
        "promo_revenue_lift_pct": float(
            (promo["Revenue"].mean() - low_promo["Revenue"].mean()) / low_promo["Revenue"].mean() * 100
        ) if len(low_promo) and low_promo["Revenue"].mean() else 0.0,
    }


def main():
    df = load_data()

    kpis = compute_kpis(df)
    channel_perf = compute_channel_performance(df)
    monthly = monthly_performance(df)
    quarterly = quarterly_performance(df)
    yearly = yearly_performance(df)
    holiday = holiday_analysis(df)
    promo = promotion_analysis(df)

    out_dir = ROOT / "data" / "processed"
    channel_perf.to_csv(out_dir / "channel_performance.csv", index=False)
    monthly.to_csv(out_dir / "monthly_performance.csv", index=False)
    quarterly.to_csv(out_dir / "quarterly_performance.csv", index=False)
    yearly.to_csv(out_dir / "yearly_performance.csv", index=False)

    kpi_bundle = {
        "kpis": kpis,
        "holiday_analysis": holiday,
        "promotion_analysis": promo,
    }
    with open(out_dir / "marketing_kpis.json", "w") as f:
        json.dump(kpi_bundle, f, indent=2)

    print("Marketing Analytics Summary")
    print("=" * 50)
    for k, v in kpis.items():
        print(f"{k}: {v}")
    print("\nTop channel by spend:")
    print(channel_perf.iloc[0][["Channel", "Total_Spend", "Naive_ROAS"]])
    print(f"\nExports written to: {out_dir}")


if __name__ == "__main__":
    main()
