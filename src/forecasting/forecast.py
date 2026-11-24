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

import json
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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


def evaluate_forecast(df: pd.DataFrame, horizon: int = N_FORECAST_WEEKS, n_folds: int = 4) -> dict:
    """Walk-forward evaluation of this forecasting method (fit on weeks before
    each origin, score on the next ``horizon`` unseen weeks), compared with
    naive (last value) and seasonal-naive (same week last year) baselines.
    Does not change the published forecast."""
    from src.analytics.ts_evaluation import rolling_origin_backtest

    y = df["Revenue"].to_numpy(dtype=float)

    def ridge_fn(train, h):
        idx = np.arange(len(train))
        m = Ridge(alpha=1.0).fit(build_time_features(idx), train)
        return m.predict(build_time_features(np.arange(len(train), len(train) + h)))

    def naive_fn(train, h):
        return np.full(h, train[-1])

    def seasonal_naive_fn(train, h):
        return np.array([train[len(train) - 52 + (i % 52)] for i in range(h)])

    min_train = 104 if len(y) >= 104 + horizon * n_folds else 60
    out = {"method": "rolling-origin (walk-forward) backtest", "horizon_weeks": horizon}
    for name, fn in (("Ridge trend + seasonal (published model)", ridge_fn),
                     ("Naive (last value)", naive_fn),
                     ("Seasonal naive (same week last year)", seasonal_naive_fn)):
        bt = rolling_origin_backtest(y, fn, horizon=horizon, n_folds=n_folds, min_train=max(min_train, 53))
        out[name] = {"metrics": bt["metrics"], "folds": bt["n_folds"],
                     "oos_error_std": float(np.std(bt["errors"], ddof=1)) if len(bt["errors"]) > 1 else None}
    model_key = "Ridge trend + seasonal (published model)"
    best_base = min(("Naive (last value)", "Seasonal naive (same week last year)"),
                    key=lambda k: out[k]["metrics"]["MAE"] if out[k]["metrics"]["MAE"] is not None else np.inf)
    m_mae, b_mae = out[model_key]["metrics"]["MAE"], out[best_base]["metrics"]["MAE"]
    out["baseline"] = best_base
    out["improvement_vs_baseline_pct"] = round((1 - m_mae / b_mae) * 100, 2) if (m_mae and b_mae) else None
    dates = pd.to_datetime(df["Date"])
    out["data_range"] = [str(dates.min().date()), str(dates.max().date())]
    _, in_sample_std, _ = fit_forecast_model(df)
    out["in_sample_residual_std"] = in_sample_std
    oos = out[model_key]["oos_error_std"]
    out["interval_note"] = (
        "The published 90% interval uses the in-sample residual std "
        f"({in_sample_std:,.0f}). Out-of-sample walk-forward errors have std {oos:,.0f}"
        + ("; the interval is therefore likely too narrow and should be read as optimistic."
           if oos and oos > in_sample_std * 1.1 else "; the interval width is consistent with out-of-sample error.")
    ) if oos else "Not enough history for an out-of-sample interval check."
    return out


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

    evaluation = evaluate_forecast(df)
    with open(out_dir / "forecast_evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2, default=str)
    print(f"Walk-forward evaluation written to: {out_dir / 'forecast_evaluation.json'}")


if __name__ == "__main__":
    main()
