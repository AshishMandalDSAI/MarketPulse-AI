"""
MarketPulse AI - Revenue Forecasting
========================================
A deliberately SIMPLE, reliable forecasting approach rather than an unstable
advanced model: a linear-trend + seasonal-harmonic regression (Fourier terms
for a 52-week annual cycle), fit with Ridge regression on time-aware
features. This is easy to explain to a business audience, hard to overfit
with ~200 weekly observations, and produces sane, boundable forecasts with
an approximate 90% prediction interval derived from historical residual
standard deviation.

Run:
    python src/forecasting/forecast.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "marketing_data_clean.csv"

N_FORECAST_WEEKS = 12
Z_90 = 1.645  # ~90% prediction interval


def build_time_features(week_idx: np.ndarray) -> np.ndarray:
    """Trend + 52-week annual Fourier harmonics (first 2 harmonics)."""
    feats = [week_idx]
    for k in (1, 2):
        feats.append(np.sin(2 * np.pi * k * week_idx / 52.0))
        feats.append(np.cos(2 * np.pi * k * week_idx / 52.0))
    return np.column_stack(feats)


def fit_forecast_model(df: pd.DataFrame):
    week_idx = np.arange(len(df))
    X = build_time_features(week_idx)
    y = df["Revenue"].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    in_sample_pred = model.predict(X)
    residual_std = float(np.std(y - in_sample_pred))
    return model, residual_std, week_idx[-1]


def forecast_future(model, residual_std, last_week_idx, n_weeks=N_FORECAST_WEEKS, last_date=None):
    future_idx = np.arange(last_week_idx + 1, last_week_idx + 1 + n_weeks)
    X_future = build_time_features(future_idx)
    preds = model.predict(X_future)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq="W-MON")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast_Revenue": preds,
        "Lower_90": preds - Z_90 * residual_std,
        "Upper_90": preds + Z_90 * residual_std,
    })
    return forecast_df


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    model, residual_std, last_week_idx = fit_forecast_model(df)
    forecast_df = forecast_future(model, residual_std, last_week_idx, last_date=df["Date"].max())

    out_dir = ROOT / "data" / "processed"
    forecast_df.to_csv(out_dir / "revenue_forecast.csv", index=False)

    print(f"Forecast for next {N_FORECAST_WEEKS} weeks (Ridge trend + seasonal harmonics)")
    print(forecast_df.to_string(index=False))
    print(f"\nResidual std (historical fit): {residual_std:,.2f}")
    print(f"Written to: {out_dir / 'revenue_forecast.csv'}")


if __name__ == "__main__":
    main()
