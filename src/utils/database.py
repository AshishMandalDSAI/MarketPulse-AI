"""
MarketPulse AI - Lightweight SQLite Database
================================================
Provides a simple, dependency-free persistence layer for the project using
Python's built-in sqlite3 module. Tables:

  marketing_data          - the full weekly feature-engineered dataset
  channel_metrics          - channel_performance + channel_attribution merged
  model_results             - MMM comparison metrics (one row per model variant)
  budget_recommendations  - latest budget optimizer output
  scenarios                 - what-if scenario comparison rows

Run:
    python src/utils/database.py
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "marketpulse.db"
DATA_DIR = ROOT / "data" / "processed"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def load_table_from_csv(conn: sqlite3.Connection, csv_name: str, table_name: str):
    path = DATA_DIR / csv_name
    if not path.exists():
        print(f"  [skip] {csv_name} not found (run the relevant pipeline step first)")
        return
    df = pd.read_csv(path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"  [ok] {table_name} <- {csv_name} ({len(df)} rows)")


def load_model_results(conn: sqlite3.Connection):
    metrics_path = ROOT / "models" / "model_metrics.json"
    if not metrics_path.exists():
        print("  [skip] model_metrics.json not found")
        return
    with open(metrics_path) as f:
        metrics = json.load(f)

    rows = []
    for model_name, m in metrics["model_comparison"].items():
        rows.append({
            "model_name": model_name,
            "selected": model_name == metrics["selected_model"],
            "test_r2": m["test"]["r2"],
            "test_adjusted_r2": m["test"]["adjusted_r2"],
            "test_rmse": m["test"]["rmse"],
            "test_mae": m["test"]["mae"],
            "test_mape": m["test"]["mape"],
        })
    df = pd.DataFrame(rows)
    df.to_sql("model_results", conn, if_exists="replace", index=False)
    print(f"  [ok] model_results ({len(df)} rows)")


def build_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()

    print("Building MarketPulse SQLite database...")
    load_table_from_csv(conn, "marketing_data_clean.csv", "marketing_data")
    load_table_from_csv(conn, "channel_attribution.csv", "channel_metrics")
    load_model_results(conn)
    load_table_from_csv(conn, "budget_optimization_result.csv", "budget_recommendations")
    load_table_from_csv(conn, "scenario_comparison.csv", "scenarios")

    conn.commit()
    conn.close()
    print(f"\nDatabase written to: {DB_PATH}")


def query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    # relies on ../../data/processed/*.csv from the pipeline having run first
    import sys
    sys.path.insert(0, str(ROOT))
    build_database()
    print("\nSample query: SELECT * FROM channel_metrics LIMIT 3")
    print(query("SELECT * FROM channel_metrics LIMIT 3"))
