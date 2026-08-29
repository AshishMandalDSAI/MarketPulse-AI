"""Tests for KPI calculation and channel analytics."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data.generate_dataset import generate_dataset
from src.data.data_cleaning import clean
from src.features.feature_engineering import engineer_features
from src.analytics.marketing_analytics import (
    compute_kpis, compute_channel_performance, holiday_analysis, promotion_analysis,
)


def _sample_df():
    raw = generate_dataset(seed=10, n_weeks=104)
    raw["Date"] = pd.to_datetime(raw["Date"])
    cleaned, _ = clean(raw)
    return engineer_features(cleaned)


def test_compute_kpis_basic_sanity():
    df = _sample_df()
    kpis = compute_kpis(df)
    assert kpis["total_revenue"] > 0
    assert kpis["total_marketing_spend"] > 0
    assert kpis["n_weeks"] == 104
    assert kpis["overall_naive_roas"] > 0


def test_channel_performance_shares_sum_to_100():
    df = _sample_df()
    perf = compute_channel_performance(df)
    assert abs(perf["Spend_Share_Pct"].sum() - 100) < 1e-6
    assert len(perf) == 8


def test_holiday_analysis_keys():
    df = _sample_df()
    result = holiday_analysis(df)
    assert "holiday_revenue_lift_pct" in result
    assert result["holiday_weeks"] + result["non_holiday_weeks"] == len(df)


def test_promotion_analysis_keys():
    df = _sample_df()
    result = promotion_analysis(df)
    assert "promo_revenue_lift_pct" in result


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}")
