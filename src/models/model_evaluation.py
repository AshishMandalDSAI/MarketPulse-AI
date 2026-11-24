"""
MarketPulse AI - Model Evaluation & Diagnostic Curves
========================================================
Generates supporting artifacts for the dashboard's "Marketing Mix Model"
page: residual diagnostics plus adstock / saturation / response curve data
for each channel, based on the SELECTED model's fitted parameters.

Run (after marketing_mix_model.py):
    python src/models/model_evaluation.py
"""

import json
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.adstock import geometric_adstock
from src.models.saturation import hill_saturation

MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "marketing_data_clean.csv"


def load_model_bundle():
    with open(MODELS_DIR / "mmm_model.pkl", "rb") as f:
        return pickle.load(f)


def residual_diagnostics() -> dict:
    avp = pd.read_csv(ROOT / "data" / "processed" / "actual_vs_predicted.csv")
    resid = avp["Residual"].values
    return {
        "mean_residual": float(resid.mean()),
        "std_residual": float(resid.std()),
        "min_residual": float(resid.min()),
        "max_residual": float(resid.max()),
        "pct_within_10pct_of_actual": float(
            (np.abs(resid) <= 0.10 * avp["Actual_Revenue"].abs()).mean() * 100
        ),
    }


def generate_curve_data(bundle: dict) -> dict:
    """Produce adstock curve, saturation curve, and spend-response curve
    sample points for each channel, driven by the actual fitted per-channel
    decay/shape/k parameters (not illustrative placeholders).
    """
    channel_params = bundle["channel_params"]
    curves = {}
    max_spend_probe = 800000  # spend range probed for response curves (INR)
    spend_grid = np.linspace(0, max_spend_probe, 60)

    for ch, p in channel_params.items():
        # Adstock decay curve: how a single week's impulse fades over time
        impulse = np.zeros(12)
        impulse[0] = 1.0
        adstock_decay_curve = geometric_adstock(impulse, p["decay"])

        # Saturation / response curve: saturated effect across a spend range
        saturation_curve = hill_saturation(spend_grid, p["k"], p["shape"])

        curves[ch] = {
            "decay_rate": p["decay"],
            "shape": p["shape"],
            "half_saturation_k": p["k"],
            "adstock_decay_curve": adstock_decay_curve.tolist(),
            "spend_grid": spend_grid.tolist(),
            "saturation_response_curve": saturation_curve.tolist(),
        }
    return curves


def main():
    bundle = load_model_bundle()
    diagnostics = residual_diagnostics()
    curves = generate_curve_data(bundle)

    out = {
        "selected_model": bundle["selected_model_name"],
        "residual_diagnostics": diagnostics,
        "curves": curves,
    }
    out_path = MODELS_DIR / "model_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("Residual Diagnostics")
    print("=" * 40)
    for k, v in diagnostics.items():
        print(f"{k}: {v}")
    print(f"\nCurve data for {len(curves)} channels written to: {out_path}")


if __name__ == "__main__":
    main()
