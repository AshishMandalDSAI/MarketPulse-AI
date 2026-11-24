"""Generates 11 deterministic fixture CSVs covering diverse dataset types
for Phase 0 migration testing. Fixed random seed -> reproducible.

Fixtures A-K cover: marketing, sales, finance, operations, education,
employee, no-date, unusual column names, missing values, duplicate rows,
and minimal datasets.

Run:
    python tests/fixtures/generate_diverse_schemas.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "diverse_schemas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2024)


def gen_a_marketing(n=52):
    """A: Marketing dataset with date, revenue, and 3 ad spend channels."""
    dates = pd.date_range("2024-01-01", periods=n, freq="W")
    google_ads = rng.uniform(1000, 8000, n).round(2)
    facebook_ads = rng.uniform(500, 6000, n).round(2)
    youtube_ads = rng.uniform(200, 3000, n).round(2)
    orders = rng.integers(50, 500, n)
    customers = rng.integers(30, 300, n)
    revenue = (google_ads * 2.0 + facebook_ads * 1.5 + youtube_ads * 0.8
               + orders * 10 + rng.normal(5000, 1000, n)).round(2)
    return pd.DataFrame({
        "Date": dates, "Revenue": revenue,
        "Google_Ads": google_ads, "Facebook_Ads": facebook_ads,
        "YouTube_Ads": youtube_ads, "Orders": orders, "Customers": customers,
    })


def gen_b_sales(n=100):
    """B: Sales dataset with transaction date, sales amount, categories, regions."""
    dates = pd.date_range("2024-03-01", periods=n, freq="D")
    sales_amount = rng.uniform(50, 2000, n).round(2)
    orders = rng.integers(1, 20, n)
    customers = rng.integers(1, 15, n)
    product_category = rng.choice(["Electronics", "Clothing", "Food", "Furniture", "Books"], n)
    region = rng.choice(["North", "South", "East", "West"], n)
    return pd.DataFrame({
        "Transaction_Date": dates, "Sales_Amount": sales_amount,
        "Orders": orders, "Customers": customers,
        "Product_Category": product_category, "Region": region,
    })


def gen_c_finance(n=60):
    """C: Finance dataset with income, expense, profit, department."""
    dates = pd.date_range("2024-01-01", periods=n, freq="MS")
    income = rng.uniform(50000, 200000, n).round(2)
    expense = rng.uniform(30000, 150000, n).round(2)
    profit = (income - expense).round(2)
    department = rng.choice(["Engineering", "Marketing", "Sales", "HR", "Operations"], n)
    return pd.DataFrame({
        "Date": dates, "Income": income, "Expense": expense,
        "Profit": profit, "Department": department,
    })


def gen_d_operations(n=90):
    """D: Operations dataset with production metrics."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    units_produced = rng.integers(100, 1000, n)
    processing_time = rng.uniform(0.5, 8.0, n).round(2)
    defect_count = rng.integers(0, 50, n)
    machine = rng.choice(["Machine_A", "Machine_B", "Machine_C"], n)
    shift = rng.choice(["Morning", "Afternoon", "Night"], n)
    return pd.DataFrame({
        "Date": dates, "Units_Produced": units_produced,
        "Processing_Time": processing_time, "Defect_Count": defect_count,
        "Machine": machine, "Shift": shift,
    })


def gen_e_education(n=200):
    """E: Education dataset with student performance metrics (no date)."""
    student_id = [f"STU{1000 + i}" for i in range(n)]
    attendance = rng.uniform(40, 100, n).round(1)
    assignment_score = rng.uniform(0, 100, n).round(1)
    study_hours = rng.uniform(0, 40, n).round(1)
    exam_score = (0.3 * attendance + 0.4 * assignment_score + 0.3 * study_hours
                  + rng.normal(0, 5, n)).clip(0, 100).round(1)
    department = rng.choice(["CS", "Math", "Physics", "Biology", "English"], n)
    return pd.DataFrame({
        "Student_ID": student_id, "Attendance": attendance,
        "Assignment_Score": assignment_score, "Study_Hours": study_hours,
        "Exam_Score": exam_score, "Department": department,
    })


def gen_f_employee(n=150):
    """F: Employee dataset with HR metrics (no date)."""
    employee_id = [f"EMP{2000 + i}" for i in range(n)]
    department = rng.choice(["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"], n)
    experience = rng.integers(0, 30, n)
    salary = (30000 + experience * 2000 + rng.normal(0, 5000, n)).round(2)
    performance_rating = rng.uniform(1, 5, n).round(1)
    return pd.DataFrame({
        "Employee_ID": employee_id, "Department": department,
        "Experience": experience, "Salary": salary,
        "Performance_Rating": performance_rating,
    })


def gen_g_no_date(n=80):
    """G: Genuine non-time-series dataset with no date column at all."""
    product_id = [f"P{3000 + i}" for i in range(n)]
    weight_kg = rng.uniform(0.1, 50, n).round(2)
    price = rng.uniform(5, 500, n).round(2)
    rating = rng.uniform(1, 5, n).round(1)
    category = rng.choice(["A", "B", "C", "D"], n)
    warehouse = rng.choice(["WH1", "WH2", "WH3"], n)
    return pd.DataFrame({
        "product_id": product_id, "weight_kg": weight_kg,
        "price": price, "rating": rating,
        "category": category, "warehouse": warehouse,
    })


def gen_h_unusual_columns(n=60):
    """H: Dataset with completely non-standard column names that share NO
    overlap with NovaMart schema aliases."""
    return pd.DataFrame({
        "observation_timestamp": pd.date_range("2024-06-01", periods=n, freq="D"),
        "primary_outcome_metric": rng.uniform(100, 10000, n).round(2),
        "treatment_alpha_dosage": rng.uniform(0, 100, n).round(2),
        "treatment_beta_dosage": rng.uniform(0, 80, n).round(2),
        "control_variable_gamma": rng.uniform(10, 50, n).round(2),
        "cohort_label": rng.choice(["cohort_x", "cohort_y", "cohort_z"], n),
        "trial_site_identifier": [f"SITE-{i % 5}" for i in range(n)],
    })


def gen_i_missing_values(n=100):
    """I: Dataset with significant missing values (10-30% per column)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    revenue = rng.uniform(1000, 5000, n).round(2)
    spend_a = rng.uniform(100, 1000, n).round(2)
    spend_b = rng.uniform(50, 800, n).round(2)
    region = rng.choice(["North", "South", "East", "West"], n)

    # Introduce missing values
    revenue[rng.choice(n, size=int(n * 0.10), replace=False)] = np.nan
    spend_a[rng.choice(n, size=int(n * 0.20), replace=False)] = np.nan
    spend_b[rng.choice(n, size=int(n * 0.15), replace=False)] = np.nan
    region_list = list(region)
    for idx in rng.choice(n, size=int(n * 0.30), replace=False):
        region_list[idx] = None

    return pd.DataFrame({
        "Date": dates, "Revenue": revenue,
        "Spend_Channel_A": spend_a, "Spend_Channel_B": spend_b,
        "Region": region_list,
    })


def gen_j_duplicate_rows(n=80):
    """J: Dataset with intentional duplicate rows (~15% duplicates)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    value = rng.uniform(100, 5000, n).round(2)
    category = rng.choice(["X", "Y", "Z"], n)
    df = pd.DataFrame({
        "Date": dates, "Value": value, "Category": category,
    })
    # Duplicate ~15% of rows
    dup_indices = rng.choice(n, size=int(n * 0.15), replace=True)
    dups = df.iloc[dup_indices].copy()
    df = pd.concat([df, dups], ignore_index=True)
    return df.sort_values("Date").reset_index(drop=True)


def gen_k_minimal(n=5):
    """K: Minimal dataset for edge-case testing (only 5 rows)."""
    return pd.DataFrame({
        "id": range(1, n + 1),
        "measure": rng.uniform(10, 100, n).round(2),
        "label": ["alpha", "beta", "gamma", "delta", "epsilon"],
    })


def main():
    generators = {
        "a_marketing.csv": gen_a_marketing,
        "b_sales.csv": gen_b_sales,
        "c_finance.csv": gen_c_finance,
        "d_operations.csv": gen_d_operations,
        "e_education.csv": gen_e_education,
        "f_employee.csv": gen_f_employee,
        "g_no_date.csv": gen_g_no_date,
        "h_unusual_columns.csv": gen_h_unusual_columns,
        "i_missing_values.csv": gen_i_missing_values,
        "j_duplicate_rows.csv": gen_j_duplicate_rows,
        "k_minimal.csv": gen_k_minimal,
    }
    for filename, gen_fn in generators.items():
        df = gen_fn()
        df.to_csv(OUT_DIR / filename, index=False)
        print(f"wrote {filename}: {df.shape}")


if __name__ == "__main__":
    main()
