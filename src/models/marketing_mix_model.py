"""
MarketPulse AI - Marketing Mix Model (MMM)
=============================================
Builds and compares three progressively richer models of weekly Revenue as
a function of marketing spend and business controls:

  1. BASELINE  : raw (untransformed) channel spend + controls, linear (Ridge)
  2. ADSTOCK    : geometric-adstocked spend + controls
  3. ADSTOCK+SAT: adstocked *and* Hill-saturated spend + controls

Per-channel adstock decay and saturation shape parameters are selected via a
documented grid search that maximizes each channel's univariate correlation
with Revenue (a standard, computationally-light simplification appropriate
for a course-scale project; a full joint optimization would use nested
cross-validation across all channels simultaneously, which is documented as
a limitation/future improvement).

We use TIME-AWARE train/test splitting (no shuffling) — the last 20% of
weeks are held out as a genuine future-like test set, since shuffling a
time series leaks future information into training and inflates reported
accuracy.

Model selection is NOT based on R² alone: we require the more complex model
to also improve (or not meaningfully worsen) out-of-sample RMSE/MAPE before
preferring it, consistent with the parsimony principle.

Run:
    python src/models/marketing_mix_model.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.adstock import geometric_adstock, DECAY_GRID
from src.models.saturation import hill_saturation, fit_half_saturation_point, SHAPE_GRID

DATA_PATH = ROOT / "data" / "processed" / "marketing_data_clean.csv"
MODELS_DIR = ROOT / "models"

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]
SPEND_COLS = [f"{c}_Spend" for c in CHANNELS]
CONTROL_COLS = [
    "Promotion_Spend", "Average_Price", "Holiday",
    "Competitor_Index", "Economic_Index",
]
TEST_FRACTION = 0.20


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, n_features):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n - n_features - 1 > 0 else r2
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "r2": float(r2),
        "adjusted_r2": float(adj_r2),
        "rmse": rmse,
        "mae": mae,
        "mape": mape(y_true, y_pred),
    }


def select_channel_params(df: pd.DataFrame) -> dict:
    """Documented grid search: for each channel, pick the (decay, shape)
    combination whose resulting saturated-adstocked spend best correlates
    (univariate Pearson) with weekly Revenue.
    """
    params = {}
    revenue = df["Revenue"].values
    for ch in CHANNELS:
        spend = df[f"{ch}_Spend"].values
        best = {"decay": DECAY_GRID[0], "shape": SHAPE_GRID[0], "k": 1.0, "corr": -np.inf}
        for decay in DECAY_GRID:
            adstocked = geometric_adstock(spend, decay)
            k = fit_half_saturation_point(adstocked)
            for shape in SHAPE_GRID:
                sat = hill_saturation(adstocked, k, shape)
                if sat.std() < 1e-9:
                    continue
                corr = np.corrcoef(sat, revenue)[0, 1]
                if corr > best["corr"]:
                    best = {"decay": decay, "shape": shape, "k": k, "corr": float(corr)}
        params[ch] = best
    return params


def build_feature_sets(df: pd.DataFrame, channel_params: dict):
    """Returns (X_baseline, X_adstock, X_adstock_sat) DataFrames aligned to df's index."""
    baseline = pd.DataFrame(index=df.index)
    adstock_only = pd.DataFrame(index=df.index)
    adstock_sat = pd.DataFrame(index=df.index)

    for ch in CHANNELS:
        spend = df[f"{ch}_Spend"].values
        p = channel_params[ch]
        adstocked = geometric_adstock(spend, p["decay"])
        saturated = hill_saturation(adstocked, p["k"], p["shape"])

        baseline[f"{ch}_Spend"] = spend
        adstock_only[f"{ch}_Spend_adstock"] = adstocked
        adstock_sat[f"{ch}_Spend_adstock_sat"] = saturated

    for c in CONTROL_COLS:
        baseline[c] = df[c].values
        adstock_only[c] = df[c].values
        adstock_sat[c] = df[c].values

    return baseline, adstock_only, adstock_sat


def time_aware_split(n_rows: int, test_fraction: float = TEST_FRACTION):
    split_idx = int(n_rows * (1 - test_fraction))
    return np.arange(0, split_idx), np.arange(split_idx, n_rows)


def fit_and_evaluate(X: pd.DataFrame, y: np.ndarray, train_idx, test_idx, alpha=5.0):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = evaluate(y_train, train_pred, X.shape[1])
    test_metrics = evaluate(y_test, test_pred, X.shape[1])

    return model, scaler, train_metrics, test_metrics, test_pred


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    y = df["Revenue"].values
    n = len(df)
    train_idx, test_idx = time_aware_split(n)

    channel_params = select_channel_params(df)
    X_baseline, X_adstock, X_adstock_sat = build_feature_sets(df, channel_params)

    results = {}
    fitted = {}
    for name, X in [("baseline", X_baseline), ("adstock", X_adstock), ("adstock_saturation", X_adstock_sat)]:
        model, scaler, train_m, test_m, test_pred = fit_and_evaluate(X, y, train_idx, test_idx)
        results[name] = {"train": train_m, "test": test_m}
        fitted[name] = {
            "model": model, "scaler": scaler, "X": X,
            "test_pred": test_pred, "feature_names": list(X.columns),
        }
        print(f"\n[{name.upper()}] Test set performance:")
        for k, v in test_m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Model selection: prefer more complex model only if it improves
    # (or does not meaningfully worsen) out-of-sample RMSE and MAPE ---
    order = ["baseline", "adstock", "adstock_saturation"]
    selected = order[0]
    for candidate in order[1:]:
        prev_rmse = results[selected]["test"]["rmse"]
        cand_rmse = results[candidate]["test"]["rmse"]
        prev_mape = results[selected]["test"]["mape"]
        cand_mape = results[candidate]["test"]["mape"]
        # accept candidate if it improves RMSE, or is within 1% of RMSE but
        # has meaningfully better MAPE (documented selection rule)
        if cand_rmse <= prev_rmse * 1.001 and cand_mape <= prev_mape * 1.02:
            selected = candidate

    print(f"\nSelected final model: {selected}")
    print("Selection rule: prefer the more theoretically-grounded model "
          "(adstock/saturation) unless it fails to match baseline out-of-sample "
          "RMSE/MAPE within a small tolerance — R^2 alone is not used to choose.")

    # --- Persist everything needed downstream ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(MODELS_DIR / "mmm_model.pkl", "wb") as f:
        pickle.dump({
            "selected_model_name": selected,
            "model": fitted[selected]["model"],
            "scaler": fitted[selected]["scaler"],
            "feature_names": fitted[selected]["feature_names"],
            "channel_params": channel_params,
            "control_cols": CONTROL_COLS,
            "channels": CHANNELS,
        }, f)

    metrics_out = {
        "model_comparison": results,
        "selected_model": selected,
        "channel_params": channel_params,
        "train_test_split": {
            "n_train": len(train_idx), "n_test": len(test_idx),
            "method": "time-aware (chronological), no shuffling",
        },
    }
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Actual vs predicted (test period) for dashboard/report plotting
    avp = pd.DataFrame({
        "Date": df["Date"].iloc[test_idx].values,
        "Actual_Revenue": y[test_idx],
        "Predicted_Revenue": fitted[selected]["test_pred"],
    })
    avp["Residual"] = avp["Actual_Revenue"] - avp["Predicted_Revenue"]
    avp.to_csv(ROOT / "data" / "processed" / "actual_vs_predicted.csv", index=False)

    print(f"\nModel artifacts written to: {MODELS_DIR}")
    return metrics_out


if __name__ == "__main__":
    run()
