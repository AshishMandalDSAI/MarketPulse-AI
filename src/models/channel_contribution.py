"""
MarketPulse AI - Channel Contribution & Attribution
=======================================================
Estimates each channel's contribution to Revenue using the SELECTED,
fitted Marketing Mix Model. This is a MODEL-ESTIMATED decomposition, not an
observed/audited figure — it depends on the model's functional form and
fitted coefficients, and should be read as "the model attributes this much
of revenue to this channel," not as ground truth causal proof.

Method: for each channel, we zero out that channel's transformed spend
feature, re-predict revenue with the fitted model, and take the difference
from the full-model prediction as that channel's estimated incremental
contribution ("leave-one-channel-out" contribution). This is a standard,
easily-explained MMM attribution approach and is more defensible than
raw regression coefficients, which are hard for non-technical audiences to
interpret directly.

Run (after marketing_mix_model.py):
    python src/models/channel_contribution.py
"""

import pickle
import sys
import pandas as pd
from pathlib import Path

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


def build_full_feature_matrix(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    model_name = bundle["selected_model_name"]
    channel_params = bundle["channel_params"]
    control_cols = bundle["control_cols"]

    X = pd.DataFrame(index=df.index)
    for ch in CHANNELS:
        spend = df[f"{ch}_Spend"].values
        p = channel_params[ch]
        if model_name == "baseline":
            X[f"{ch}_Spend"] = spend
        elif model_name == "adstock":
            X[f"{ch}_Spend_adstock"] = geometric_adstock(spend, p["decay"])
        else:  # adstock_saturation
            adstocked = geometric_adstock(spend, p["decay"])
            X[f"{ch}_Spend_adstock_sat"] = hill_saturation(adstocked, p["k"], p["shape"])

    for c in control_cols:
        X[c] = df[c].values

    # Ensure column order matches training
    X = X[bundle["feature_names"]]
    return X


def channel_feature_name(ch: str, model_name: str) -> str:
    if model_name == "baseline":
        return f"{ch}_Spend"
    elif model_name == "adstock":
        return f"{ch}_Spend_adstock"
    else:
        return f"{ch}_Spend_adstock_sat"


def compute_contributions(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    model = bundle["model"]
    scaler = bundle["scaler"]
    model_name = bundle["selected_model_name"]

    X_full = build_full_feature_matrix(df, bundle)
    X_full_scaled = scaler.transform(X_full)
    full_pred = model.predict(X_full_scaled)

    total_spend_all = df[[f"{c}_Spend" for c in CHANNELS]].sum().sum()
    total_revenue = df["Revenue"].sum()

    rows = []
    total_incremental = 0.0
    contributions_raw = {}

    for ch in CHANNELS:
        feat_name = channel_feature_name(ch, model_name)
        X_ablated = X_full.copy()
        X_ablated[feat_name] = 0.0
        X_ablated_scaled = scaler.transform(X_ablated)
        ablated_pred = model.predict(X_ablated_scaled)

        incremental = (full_pred - ablated_pred).clip(min=0).sum()
        contributions_raw[ch] = float(incremental)
        total_incremental += incremental

    for ch in CHANNELS:
        spend = df[f"{ch}_Spend"].sum()
        contribution = contributions_raw[ch]
        contribution_pct = (contribution / total_incremental * 100) if total_incremental > 0 else 0.0
        observed_roas = total_revenue and (df["Revenue"].sum() * (spend / total_spend_all)) / spend if spend else 0.0
        # Observed ROAS here uses the channel's naive (undecomposed) ratio for
        # comparison purposes — kept separate & clearly labeled from model efficiency.
        naive_roas = (df["Revenue"].sum() / spend) if spend else 0.0
        model_efficiency = (contribution / spend) if spend else 0.0

        rows.append({
            "Channel": ch.replace("_", " "),
            "Spend": float(spend),
            "Estimated_Contribution": float(contribution),
            "Contribution_Pct": float(contribution_pct),
            "Observed_Naive_ROAS": float(naive_roas),
            "Model_Estimated_Efficiency": float(model_efficiency),
            "Estimated_Incremental_Revenue": float(contribution),
        })

    result = pd.DataFrame(rows).sort_values("Estimated_Contribution", ascending=False).reset_index(drop=True)
    result["Rank"] = result.index + 1
    return result


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    bundle = load_bundle()
    attribution = compute_contributions(df, bundle)

    out_path = ROOT / "data" / "processed" / "channel_attribution.csv"
    attribution.to_csv(out_path, index=False)

    print(f"Channel Attribution (model: {bundle['selected_model_name']})")
    print("=" * 70)
    print(attribution.to_string(index=False))
    print(f"\nWritten to: {out_path}")
    print("\nNOTE: 'Estimated_Contribution' is a MODEL-BASED decomposition using "
          "leave-one-channel-out ablation on the fitted MMM. It is not an audited "
          "or causally-proven figure.")


if __name__ == "__main__":
    main()
