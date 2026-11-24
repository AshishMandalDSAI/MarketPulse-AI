"""Integration tests for the full MMM -> attribution -> optimization -> simulation
pipeline, using a small synthetic dataset generated on the fly (does not depend
on data/processed/ having been populated already, so this test is self-contained).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.generate_dataset import generate_dataset
from src.data.data_cleaning import clean
from src.features.feature_engineering import engineer_features
from src.models.marketing_mix_model import (
    select_channel_params, build_feature_sets, time_aware_split, fit_and_evaluate, CHANNELS,
)


def _prepared_df(n_weeks=104):
    raw = generate_dataset(seed=99, n_weeks=n_weeks)
    raw["Date"] = pd.to_datetime(raw["Date"])
    cleaned, _ = clean(raw)
    return engineer_features(cleaned)


def test_select_channel_params_returns_all_channels():
    df = _prepared_df()
    params = select_channel_params(df)
    assert set(params.keys()) == set(CHANNELS)
    for ch, p in params.items():
        assert 0 <= p["decay"] < 1
        assert p["shape"] > 0


def test_build_feature_sets_shapes_match():
    df = _prepared_df()
    params = select_channel_params(df)
    baseline, adstock, adstock_sat = build_feature_sets(df, params)
    assert len(baseline) == len(df)
    assert len(adstock) == len(df)
    assert len(adstock_sat) == len(df)


def test_time_aware_split_is_chronological_no_shuffle():
    train_idx, test_idx = time_aware_split(100, test_fraction=0.2)
    assert train_idx.max() < test_idx.min()  # strictly chronological, no leakage
    assert len(test_idx) == 20


def test_fit_and_evaluate_produces_finite_metrics():
    df = _prepared_df()
    params = select_channel_params(df)
    _, adstock_X, _ = build_feature_sets(df, params)
    y = df["Revenue"].values
    train_idx, test_idx = time_aware_split(len(df))
    model, scaler, train_m, test_m, test_pred = fit_and_evaluate(adstock_X, y, train_idx, test_idx)
    assert np.isfinite(test_m["rmse"])
    assert np.isfinite(test_m["mae"])
    assert len(test_pred) == len(test_idx)


def test_optimizer_allocation_sums_to_budget():
    from src.optimization.budget_optimizer import optimize_budget
    df = _prepared_df()
    params = select_channel_params(df)
    _, adstock_X, _ = build_feature_sets(df, params)
    y = df["Revenue"].values
    train_idx, test_idx = time_aware_split(len(df))
    model, scaler, _, _, _ = fit_and_evaluate(adstock_X, y, train_idx, test_idx)

    bundle = {
        "model": model, "scaler": scaler, "selected_model_name": "adstock",
        "channel_params": params, "control_cols": ["Promotion_Spend", "Average_Price", "Holiday",
                                                      "Competitor_Index", "Economic_Index"],
        "feature_names": list(adstock_X.columns), "channels": CHANNELS,
    }
    from src.optimization.budget_optimizer import get_recent_carry_in
    carry_in = get_recent_carry_in(df, params)
    recent = df.tail(8)
    control_values = {c: float(recent[c].mean()) for c in bundle["control_cols"]}
    current_allocation = {ch: float(recent[f"{ch}_Spend"].mean()) for ch in CHANNELS}
    total_budget = sum(current_allocation.values())

    result = optimize_budget(total_budget, bundle, carry_in, control_values, current_allocation)
    allocated_total = sum(result["optimized_allocation"].values())
    assert abs(allocated_total - total_budget) < 1.0  # numerical tolerance
    assert all(v >= 0 for v in result["optimized_allocation"].values())  # no negative budgets


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}")
