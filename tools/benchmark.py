"""Benchmark the upload -> quality -> mapping -> cleaning -> analysis path.

Usage: python tools/benchmark.py ROWS [--skip-analysis]
Runs in one process for one size and prints a JSON line with per-stage wall
time (s) and peak resident memory (MB). Run each size in a fresh process so
peak memory is not inflated by a previous run.
"""
import io
import json
import resource
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_csv(n: int, seed: int = 0) -> bytes:
    """Retail-style data: daily rows for many stores (repeated dates), spend
    columns, ~2% missing cells, some blank strings and numbers stored as text."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2021-01-01", periods=max(30, n // 200), freq="D")
    d = rng.choice(days, n)
    df = pd.DataFrame({
        "Order_Date": pd.to_datetime(d).strftime("%Y-%m-%d"),
        "Store": rng.choice([f"S{i:03d}" for i in range(200)], n),
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Net_Sales": rng.gamma(4, 250, n).round(2),
        "Meta_Ads": rng.uniform(0, 400, n).round(2),
        "Google_Advertising": rng.uniform(0, 300, n).round(2),
        "Units": rng.integers(1, 40, n).astype(float),
        "Discount_Pct": rng.uniform(0, 30, n).round(1),
    })
    for c in ("Net_Sales", "Meta_Ads", "Units", "Discount_Pct"):
        df.loc[rng.random(n) < 0.02, c] = np.nan
    df["Region"] = df["Region"].where(rng.random(n) > 0.01, " ")
    df["Google_Advertising"] = df["Google_Advertising"].map(lambda v: f"{v:,.2f}")
    return df.to_csv(index=False).encode()


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    n = int(sys.argv[1])
    skip_analysis = "--skip-analysis" in sys.argv
    from src.data_engine.data_quality import apply_cleaning, assess_quality, recommend_plan
    from src.data_engine.pipeline import profile_and_map
    from src.data_engine.analysis_runner import run_full_analysis
    from config.settings import dataset_upload_dir

    data = make_csv(n)
    out = {"rows": n, "csv_mb": round(len(data) / 1e6, 1)}
    t = time.perf_counter(); df = pd.read_csv(io.BytesIO(data)); out["read_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter(); rep = assess_quality(df); out["quality_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    norm, _ = apply_cleaning(df, {"columns": {c: {"strategy": "none", "convert_numeric": v["kind"] == "numeric_text"}
                                             for c, v in rep["columns"].items()}})
    _, mapping = profile_and_map(norm)
    out["mapping_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    plan = recommend_plan(df, rep, date_column="Order_Date",
                          roles={"Net_Sales": "revenue", "Meta_Ads": "channel", "Google_Advertising": "channel"})
    cleaned, audit = apply_cleaning(df, plan)
    out["cleaning_s"] = round(time.perf_counter() - t, 2)
    out["quality_score"] = rep["score"]
    out["cells_imputed"] = audit["total_imputed_cells"]
    out["peak_mb_before_analysis"] = round(peak_mb(), 0)
    if not skip_analysis:
        ds = f"bench_{uuid.uuid4().hex[:8]}"
        t = time.perf_counter()
        try:
            run_full_analysis(ds, df=cleaned, mapping=mapping, confirmed_roles={"date", "revenue", "channels"})
            out["analysis_s"] = round(time.perf_counter() - t, 2)
        finally:
            shutil.rmtree(dataset_upload_dir(ds), ignore_errors=True)
    out["peak_mb"] = round(peak_mb(), 0)
    out["total_s"] = round(sum(v for k, v in out.items() if k.endswith("_s")), 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
