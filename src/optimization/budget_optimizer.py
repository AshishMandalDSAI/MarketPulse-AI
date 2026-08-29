"""
MarketPulse AI - Budget Optimization Engine
==============================================
Given the fitted Marketing Mix Model, find the channel allocation of a fixed
total marketing budget that MAXIMIZES predicted weekly revenue, subject to
per-channel minimum/maximum spend bounds and a total-budget equality
constraint.

We use scipy.optimize.minimize with SLSQP (Sequential Least Squares
Programming), which natively supports bound + equality constraints and is
well suited to this smooth, differentiable (adstock/saturation are smooth)
non-linear objective.

Because adstock is inherently a TIME-SERIES transform (depends on the whole
spend history), the optimizer here optimizes a single REPRESENTATIVE WEEK'S
allocation: it holds the adstocked carry-in from prior weeks fixed at each
channel's recent average level, and evaluates how re-allocating this week's
incremental spend changes predicted revenue through the fitted saturation
curves. This is a standard simplification for weekly budget-reallocation
recommendations and is documented here rather than silently assumed.

Run (after marketing_mix_model.py and channel_contribution.py):
    python src/optimization/budget_optimizer.py
"""

import json
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.adstock import geometric_adstock
from src.models.saturation import hill_saturation

MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "marketing_data_clean.csv"
CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]


def load_bundle():
    with open(MODELS_DIR / "mmm_model.pkl", "rb") as f:
        return pickle.load(f)


def get_recent_carry_in(df: pd.DataFrame, channel_params: dict, lookback_weeks: int = 8) -> dict:
    """Estimate each channel's adstock 'carry-in' from recent history (the
    portion of adstocked effect NOT explained by this week's own spend),
    used to hold prior-weeks' momentum constant during optimization.
    """
    carry_in = {}
    for ch in CHANNELS:
        spend_hist = df[f"{ch}_Spend"].values[-lookback_weeks:]
        decay = channel_params[ch]["decay"]
        full_adstock = geometric_adstock(spend_hist, decay)
        # carry-in = adstock value the week BEFORE the most recent week's own spend is added
        carry_in[ch] = float(full_adstock[-1] - spend_hist[-1])
    return carry_in


def predict_revenue_for_allocation(
    allocation: np.ndarray, bundle: dict, carry_in: dict, control_values: dict
) -> float:
    """Predict weekly revenue for a given channel spend allocation vector
    (ordered as CHANNELS), holding controls at their supplied (e.g. recent
    average) values and prior adstock carry-in fixed.
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    model_name = bundle["selected_model_name"]
    channel_params = bundle["channel_params"]
    feature_names = bundle["feature_names"]

    feat = {}
    for i, ch in enumerate(CHANNELS):
        spend = allocation[i]
        p = channel_params[ch]
        if model_name == "baseline":
            feat[f"{ch}_Spend"] = spend
        elif model_name == "adstock":
            adstocked = spend + p["decay"] * carry_in[ch]
            feat[f"{ch}_Spend_adstock"] = adstocked
        else:
            adstocked = spend + p["decay"] * carry_in[ch]
            feat[f"{ch}_Spend_adstock_sat"] = hill_saturation(np.array([adstocked]), p["k"], p["shape"])[0]

    for c, v in control_values.items():
        feat[c] = v

    X = pd.DataFrame([feat])[feature_names]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return float(pred)


def optimize_budget(
    total_budget: float,
    bundle: dict,
    carry_in: dict,
    control_values: dict,
    current_allocation: dict,
    min_pct: float = 0.05,
    max_pct: float = 0.55,
) -> dict:
    """Find the revenue-maximizing allocation of `total_budget` across channels.

    min_pct / max_pct: fractional bounds (of total budget) applied per channel,
    preventing degenerate all-in-one-channel solutions and reflecting realistic
    operational constraints (e.g. you can't sensibly put 100% of budget into Email).
    """
    n = len(CHANNELS)
    x0 = np.array([current_allocation[ch] for ch in CHANNELS], dtype=float)
    if x0.sum() <= 0:
        x0 = np.full(n, total_budget / n)
    else:
        x0 = x0 / x0.sum() * total_budget  # scale current mix to new total budget

    lower = total_budget * min_pct
    upper = total_budget * max_pct
    bounds = [(lower, upper) for _ in range(n)]

    def objective(alloc):
        # minimize negative revenue == maximize revenue
        return -predict_revenue_for_allocation(alloc, bundle, carry_in, control_values)

    constraints = [{"type": "eq", "fun": lambda alloc: np.sum(alloc) - total_budget}]

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    optimized_allocation = {ch: float(result.x[i]) for i, ch in enumerate(CHANNELS)}
    optimized_revenue = predict_revenue_for_allocation(result.x, bundle, carry_in, control_values)
    current_revenue = predict_revenue_for_allocation(
        np.array([current_allocation[ch] for ch in CHANNELS]), bundle, carry_in, control_values
    )

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "total_budget": float(total_budget),
        "current_allocation": current_allocation,
        "optimized_allocation": optimized_allocation,
        "current_predicted_revenue": float(current_revenue),
        "optimized_predicted_revenue": float(optimized_revenue),
        "expected_improvement_pct": float(
            (optimized_revenue - current_revenue) / current_revenue * 100
        ) if current_revenue else 0.0,
        "current_efficiency": float(current_revenue / total_budget) if total_budget else 0.0,
        "optimized_efficiency": float(optimized_revenue / total_budget) if total_budget else 0.0,
    }


def build_comparison_table(opt_result: dict) -> pd.DataFrame:
    rows = []
    for ch in CHANNELS:
        cur = opt_result["current_allocation"][ch]
        rec = opt_result["optimized_allocation"][ch]
        change = rec - cur
        change_pct = (change / cur * 100) if cur else 0.0
        rows.append({
            "Channel": ch.replace("_", " "),
            "Current_Budget": round(cur, 2),
            "Recommended_Budget": round(rec, 2),
            "Change": round(change, 2),
            "Change_Pct": round(change_pct, 2),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    bundle = load_bundle()
    channel_params = bundle["channel_params"]

    recent = df.tail(8)
    current_allocation = {ch: float(recent[f"{ch}_Spend"].mean()) for ch in CHANNELS}
    total_budget = sum(current_allocation.values())

    control_values = {c: float(recent[c].mean()) for c in bundle["control_cols"]}
    carry_in = get_recent_carry_in(df, channel_params)

    result = optimize_budget(total_budget, bundle, carry_in, control_values, current_allocation)
    table = build_comparison_table(result)

    out_dir = ROOT / "data" / "processed"
    table.to_csv(out_dir / "budget_optimization_result.csv", index=False)
    with open(out_dir / "budget_optimization_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Budget Optimization Result")
    print("=" * 60)
    print(table.to_string(index=False))
    print(f"\nCurrent predicted revenue:   {result['current_predicted_revenue']:,.2f}")
    print(f"Optimized predicted revenue: {result['optimized_predicted_revenue']:,.2f}")
    print(f"Expected improvement:        {result['expected_improvement_pct']:.2f}%")
    print(f"Optimizer success: {result['success']} ({result['message']})")


if __name__ == "__main__":
    main()
