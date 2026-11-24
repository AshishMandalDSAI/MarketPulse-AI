"""Generates 5 deterministic fixture CSVs exercising different schema shapes
for the dataset-first platform's tests. Fixed random seed -> reproducible.

Run:
    python tests/fixtures/generate_generic_schemas.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "generic_schemas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


def gen_dataset_a_marketing(n=90):
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    google_ads = rng.uniform(500, 5000, n).round(2)
    facebook_ads = rng.uniform(300, 4000, n).round(2)
    tv_spend = rng.uniform(1000, 8000, n).round(2)
    impressions = rng.integers(1000, 50000, n)
    clicks = (impressions * rng.uniform(0.01, 0.05, n)).round(0).astype(int)
    revenue = (google_ads * 2.1 + facebook_ads * 1.8 + tv_spend * 1.2 + rng.normal(2000, 500, n)).round(2)
    df = pd.DataFrame({
        "date": dates, "revenue": revenue,
        "google_ads": google_ads, "facebook_ads": facebook_ads, "tv_spend": tv_spend,
        "impressions": impressions, "clicks": clicks,
    })
    return df


def gen_dataset_b_interaction(n=200):
    start = pd.Timestamp("2024-01-01")
    interaction_date = start + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    duration = rng.uniform(5, 600, n).round(1)
    interaction_type = rng.choice(["call", "chat", "email", "in-person"], n)
    user_id = [f"U{100000 + i}" for i in rng.integers(0, 5000, n)]
    df = pd.DataFrame({
        "interaction_date": interaction_date, "duration": duration,
        "interaction_type": interaction_type, "user_id": user_id,
    })
    return df


def gen_dataset_c_sales(n=100):
    dates = pd.date_range("2024-02-01", periods=n, freq="D")
    order_value = rng.uniform(20, 500, n).round(2)
    category = rng.choice(["Electronics", "Home", "Apparel", "Toys", "Books"], n)
    quantity = rng.integers(1, 10, n)
    df = pd.DataFrame({
        "date": dates, "order_value": order_value,
        "product_category": category, "quantity": quantity,
    })
    return df


def gen_dataset_d_arbitrary(n=80):
    dates = pd.date_range("2023-11-01", periods=n, freq="W")
    total_amount_generated_this_period = rng.uniform(100, 9000, n).round(2)
    duration_minutes = rng.uniform(1, 120, n).round(1)
    region = rng.choice(["North", "South", "East", "West"], n)
    satisfaction_score = rng.integers(1, 6, n)
    df = pd.DataFrame({
        "activity_date": dates,
        "total_amount_generated_this_period": total_amount_generated_this_period,
        "duration_minutes": duration_minutes, "region": region,
        "satisfaction_score": satisfaction_score,
    })
    return df


def gen_dataset_e_no_marketing(n=150):
    employee_id = [f"E{1000 + i}" for i in range(n)]
    department = rng.choice(["Engineering", "Sales", "HR", "Finance", "Ops"], n)
    hours_logged = rng.uniform(10, 45, n).round(1)
    projects_completed = rng.integers(0, 8, n)
    df = pd.DataFrame({
        "employee_id": employee_id, "department": department,
        "hours_logged": hours_logged, "projects_completed": projects_completed,
    })
    return df


def main():
    datasets = {
        "dataset_a_marketing.csv": gen_dataset_a_marketing(),
        "dataset_b_interaction.csv": gen_dataset_b_interaction(),
        "dataset_c_sales.csv": gen_dataset_c_sales(),
        "dataset_d_arbitrary.csv": gen_dataset_d_arbitrary(),
        "dataset_e_no_marketing.csv": gen_dataset_e_no_marketing(),
    }
    for filename, df in datasets.items():
        df.to_csv(OUT_DIR / filename, index=False)
        print(f"wrote {filename}: {df.shape}")


if __name__ == "__main__":
    main()
