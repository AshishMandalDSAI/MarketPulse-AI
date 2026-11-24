"""Orchestrates the full dataset-first pipeline for one dataset_id:

  load raw -> profile+map -> standardize -> detect capabilities
  -> persist standardized data + capabilities.json + generic_kpis.json
     + universal_kpis.json to data/uploads/<dataset_id>/{processed,models}/
  -> conditionally run a lightweight generic MMM stage (if capability available)
  -> conditionally run a lightweight generic forecast stage (if capability available)
  -> conditionally run ROI stage (if capability available)

Every skipped stage is recorded with an explicit reason — never silently
skipped, never fabricated.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import ensure_dataset_dirs, dataset_processed_dir, dataset_models_dir, dataset_raw_path
from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
from src.data_engine.capability_engine import detect_capabilities
from src.data_engine.canonical import CanonicalDataset, AnalysisContext, UserOverrides
from src.kpi.generic_kpis import compute_generic_kpis

DATASET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def normalize_duplicate_columns(df: pd.DataFrame) -> tuple:
    """Detect and rename duplicate column names deterministically.

    Returns (df_with_unique_columns, rename_map_or_None).
    - If no duplicates, returns (df, None).
    - If duplicates exist, renames them as ``Name``, ``Name_1``, ``Name_2``, …,
      avoiding collisions with existing column names.
    - Column order and all data values are preserved.
    """
    cols = list(df.columns)
    if len(cols) == len(set(cols)):
        return df, None

    existing_names = set(cols)
    seen: dict[str, int] = {}
    new_cols = []
    rename_map = {}

    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            candidate = f"{col}_{seen[col]}"
            # Avoid collisions with any existing or already-assigned name
            while candidate in existing_names or candidate in set(new_cols):
                seen[col] += 1
                candidate = f"{col}_{seen[col]}"
            new_cols.append(candidate)
            existing_names.add(candidate)
            rename_map[candidate] = col  # new_name -> original_name

    df = df.copy()
    df.columns = new_cols
    return df, rename_map


def validate_dataset_id(dataset_id: str) -> str:
    if not dataset_id or not DATASET_ID_PATTERN.match(dataset_id):
        raise ValueError(
            f"Invalid dataset_id {dataset_id!r} — must match {DATASET_ID_PATTERN.pattern}."
        )
    return dataset_id


def _run_generic_mmm(dataset, models_dir: Path, processed_dir: Path, report: list) -> None:
    """A lightweight, genuinely-computed (not fabricated) linear
    channel-contribution model: Ridge regression of revenue on channel
    spend + numeric controls, with proper train/test chronological split.

    **Attribution methodology**: each channel's estimated contribution is
    ``coefficient × total_channel_spend``, preserving the sign of the
    coefficient.  A negative coefficient means the model estimates a
    negative association between that channel's spend and revenue — this
    is reported honestly rather than converted to a positive contribution.

    **Important caveat written into the output**: linear regression
    coefficients measure statistical *association*, not proof of causal
    marketing impact.  Confounders, multicollinearity and reverse
    causality can distort the estimates.

    Outputs are written in the SAME schema the dashboard pages expect
    (channel_attribution.csv with Rank, Channel, Spend, Estimated_Contribution,
    Contribution_Pct, Coefficient, Coefficient_Direction, Spend_Revenue_Ratio,
    Model_Estimated_Efficiency; model_metrics.json with selected_model +
    model_comparison structure containing both train and test metrics)
    so pages 3, 4, and 9 can consume them without adapters."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

        from src.analytics.ts_evaluation import ts_metrics

        df = dataset.df.copy()
        channel_cols = list(dataset.channels)
        control_cols = [c for c in dataset.control_cols if c in df.columns]
        X_cols = channel_cols + control_cols
        X = df[X_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(df[dataset.revenue_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        # Rows without revenue are excluded — never treated as zero revenue.
        keep = y.notna()
        n_missing_target = int((~keep).sum())
        # Rows without a valid date cannot be placed in time for the
        # chronological split, so they are excluded too.
        n_bad_dates = 0
        order_key = None
        if dataset.has_date and dataset.date_col in df.columns:
            parsed = pd.to_datetime(df[dataset.date_col], errors="coerce")
            n_bad_dates = int((parsed.isna() & keep).sum())
            keep &= parsed.notna()
            order_key = parsed
        X, y = X[keep], y[keep]
        if order_key is not None:
            order = order_key[keep].sort_values(kind="stable").index
            X, y = X.loc[order], y.loc[order]
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        n = len(y)
        if n < 4:
            report.append(f"Generic MMM skipped: only {n} usable observation(s) — need at least 4.")
            return

        split_point = max(2, int(n * 0.8))  # 80/20 chronological split, at least 2 in train
        # Missing spend = no spend recorded (documented assumption); missing
        # controls are filled with the TRAINING median only (no test leakage).
        n_spend_filled = int(X[channel_cols].isna().sum().sum()) if channel_cols else 0
        X[channel_cols] = X[channel_cols].fillna(0.0)
        n_control_filled = 0
        for c in control_cols:
            med = X[c].iloc[:split_point].median()
            n_control_filled += int(X[c].isna().sum())
            X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        has_holdout = len(X_test) > 0

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Training metrics
        train_preds = model.predict(X_train)
        train_r2 = float(r2_score(y_train, train_preds)) if len(y_train) > 1 else 0.0
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds))) if len(y_train) > 1 else 0.0
        train_mae = float(mean_absolute_error(y_train, train_preds)) if len(y_train) > 1 else 0.0

        # Holdout metrics + naive baseline (predict the training mean)
        holdout = baseline = None
        if has_holdout:
            test_preds = model.predict(X_test)
            holdout = ts_metrics(y_test.to_numpy(), test_preds)
            baseline = ts_metrics(y_test.to_numpy(), np.full(len(y_test), float(y_train.mean())))
            test_r2 = holdout["R2"] if holdout["R2"] is not None else 0.0
            test_rmse, test_mae, test_mape = holdout["RMSE"], holdout["MAE"], holdout["MAPE"]
        else:
            test_r2 = test_rmse = test_mae = 0.0
            test_mape = None

        # --- Build channel_attribution.csv in the dashboard-expected schema ---
        # Use full-data coefficients for attribution (fitted on train only)
        coefs = {col: float(coef) for col, coef in zip(X_cols, model.coef_)}
        total_revenue = float(y.sum()) or 1.0

        attribution_rows = []
        for ch in channel_cols:
            spend_total = float(df[ch].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum())
            coef = coefs[ch]

            # Estimated contribution: coefficient * total spend (signed — preserves direction)
            estimated_contribution = coef * spend_total

            # Direction label for dashboard consumers
            if coef > 0.001:
                direction = "positive"
            elif coef < -0.001:
                direction = "negative"
            else:
                direction = "neutral"

            # Spend-to-revenue ratio (NOT attributed ROAS — this is total revenue / channel spend)
            spend_revenue_ratio = (total_revenue / spend_total) if spend_total > 0 else 0.0

            # Model efficiency: signed coefficient (revenue-per-unit-spend association)
            efficiency = coef

            attribution_rows.append({
                "Channel": ch,
                "Spend": round(spend_total, 2),
                "Estimated_Contribution": round(estimated_contribution, 2),
                "Coefficient": round(coef, 6),
                "Coefficient_Direction": direction,
                "Spend_Revenue_Ratio": round(spend_revenue_ratio, 2),
                "Model_Estimated_Efficiency": round(efficiency, 4),
            })

        # Contribution_Pct: use absolute values for sizing but preserve sign info
        total_abs_contribution = sum(abs(r["Estimated_Contribution"]) for r in attribution_rows) or 1.0
        for row in attribution_rows:
            # Signed percentage — negative channels get negative share
            row["Contribution_Pct"] = round(
                row["Estimated_Contribution"] / total_abs_contribution * 100, 2
            )

        # Sort by efficiency descending and rank
        attribution_rows.sort(key=lambda r: r["Model_Estimated_Efficiency"], reverse=True)
        for i, row in enumerate(attribution_rows, 1):
            row["Rank"] = i

        attr_df = pd.DataFrame(attribution_rows)
        # Write to BOTH models/ and processed/ so loaders find it
        attr_df.to_csv(models_dir / "channel_attribution.csv", index=False)
        attr_df.to_csv(processed_dir / "channel_attribution.csv", index=False)

        # --- Build model_metrics.json in the dashboard-expected schema ---
        model_name = "generic_ridge_linear"
        _r = lambda v, d: round(v, d) if v is not None else None
        test_metrics = {
            "r2": _r(holdout["R2"], 4) if holdout else None,
            "rmse": round(test_rmse, 2),
            "mae": round(test_mae, 2),
            # None when undefined (all actuals zero) — never a fabricated 0% error
            "mape": _r(test_mape, 2),
            "smape": _r(holdout["sMAPE"], 2),
            "wape": _r(holdout["WAPE"], 2),
        } if has_holdout else None
        baseline_metrics = {
            "model": "mean of training revenue",
            "mae": _r(baseline["MAE"], 2), "rmse": _r(baseline["RMSE"], 2),
            "wape": _r(baseline["WAPE"], 2), "r2": _r(baseline["R2"], 4),
        } if baseline else None
        beats_baseline = (holdout["MAE"] < baseline["MAE"]) if (holdout and baseline) else None

        metrics_dict = {
            "selected_model": model_name,
            "model_comparison": {
                model_name: {
                    "train": {
                        "r2": round(train_r2, 4),
                        "rmse": round(train_rmse, 2),
                        "mae": round(train_mae, 2),
                    },
                    **({"test": test_metrics} if test_metrics else {}),
                }
            },
            # Extra metadata for transparency
            "n_observations": int(n),
            "n_train": int(split_point),
            "n_test": int(n - split_point),
            "has_holdout": has_holdout,
            "baseline": baseline_metrics,
            "beats_baseline": beats_baseline,
            "rows_excluded_missing_revenue": n_missing_target,
            "rows_excluded_invalid_date": n_bad_dates,
            "spend_cells_filled_zero": n_spend_filled,
            "control_cells_filled_train_median": n_control_filled,
            "features": X_cols,
            "model_type": model_name,
            "note": ("Ridge regression coefficients measure statistical association, "
                     "not proof of causal marketing impact. Confounders, "
                     "multicollinearity and reverse causality can distort estimates."),
        }
        with open(models_dir / "model_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)

        holdout_msg = f", holdout R2={test_r2:.3f}" if has_holdout else " (no holdout — insufficient data)"
        if beats_baseline is False:
            holdout_msg += " — does NOT beat a naive mean-revenue baseline on the holdout; treat attributions with caution"
        if n_missing_target or n_bad_dates:
            holdout_msg += (f"; {n_missing_target} row(s) without revenue and {n_bad_dates} without a valid date "
                            f"were excluded")
        report.append(
            f"Generic MMM (Ridge regression) trained on {split_point} rows: "
            f"train R2={train_r2:.3f}{holdout_msg}, {len(X_cols)} feature(s)."
        )
    except Exception as e:
        report.append(f"Marketing mix modeling was available but failed to train: {e}")


def _infer_frequency(dates_series: pd.Series) -> tuple:
    """Infer the time-series frequency from a sorted datetime Series.

    Returns (freq_label, timedelta) where freq_label is one of
    'daily', 'weekly', 'monthly', 'quarterly', 'yearly', or 'irregular',
    and timedelta is the corresponding pd.Timedelta (or median gap for irregular).
    """
    if len(dates_series) < 2:
        return "unknown", pd.Timedelta(weeks=1)

    gaps = dates_series.diff().dropna()
    if gaps.empty:
        return "unknown", pd.Timedelta(weeks=1)

    median_days = gaps.median().days

    # Match to standard calendar periods
    if 0 <= median_days <= 1:
        return "daily", pd.Timedelta(days=1)
    elif 5 <= median_days <= 9:
        return "weekly", pd.Timedelta(weeks=1)
    elif 25 <= median_days <= 35:
        return "monthly", pd.Timedelta(days=30)
    elif 85 <= median_days <= 95:
        return "quarterly", pd.Timedelta(days=91)
    elif 350 <= median_days <= 380:
        return "yearly", pd.Timedelta(days=365)
    else:
        return "irregular", gaps.median()


def _run_generic_forecast(dataset, models_dir: Path, processed_dir: Path, report: list) -> None:
    """Revenue forecast for an uploaded dataset, evaluated honestly.

    * Rows without a valid date or target are left out (never filled in).
    * Several rows per date (stores, transactions) are summed per period.
    * Missing periods are detected and reported (the series is not padded with
      invented values).
    * Candidates (naive, seasonal naive, moving average, linear trend) are
      compared with rolling-origin backtesting; a non-baseline model is used
      only if it beats the best naive baseline by at least 2%.
    * The 90% interval comes from out-of-sample walk-forward errors.

    Outputs revenue_forecast.csv (Date, Forecast_Revenue, Lower_90, Upper_90 —
    the columns page 7 expects) and forecast_evaluation.json."""
    try:
        from src.analytics.ts_evaluation import rolling_origin_backtest, ts_metrics
        from src.prediction import forecasting_engine as fe

        target_col = dataset.revenue_col or (dataset.control_cols[0] if dataset.control_cols else None)
        if not target_col:
            report.append("Forecasting skipped: no numeric target column available.")
            return

        df = dataset.df
        frame = pd.DataFrame({
            "date": pd.to_datetime(df[dataset.date_col], errors="coerce"),
            "y": pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan),
        })
        n_input = len(frame)
        n_bad_date = int(frame["date"].isna().sum())
        n_bad_y = int(frame["y"].isna().sum())
        frame = frame.dropna()
        n_dup = int(frame["date"].duplicated().sum())
        series = frame.groupby("date")["y"].sum().sort_index()
        n = len(series)
        if n < 6:
            report.append(f"Forecasting skipped: only {n} usable period(s) after removing rows without a valid "
                          f"date or '{target_col}' — need at least 6 for an honest backtest.")
            return

        dates_sorted = pd.Series(series.index)
        freq_label, freq_delta = _infer_frequency(dates_sorted)
        span = series.index[-1] - series.index[0]
        expected_periods = None
        if freq_label in ("daily", "weekly"):
            expected_periods = int(round(span / freq_delta)) + 1
        elif freq_label == "monthly":
            expected_periods = (series.index[-1].year - series.index[0].year) * 12 + \
                (series.index[-1].month - series.index[0].month) + 1
        missing_periods = max(0, expected_periods - n) if expected_periods else None

        y = series.to_numpy(dtype=float)
        horizon = min(12, max(1, n // 4))
        season_len = {"daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4}.get(freq_label, 1)
        if n < 2 * season_len:
            season_len = 1
        bt_h = max(1, min(horizon, n // 6))
        fit_fns = {
            "Naive (Last Value)": ("naive", lambda tr, h: fe._train_naive(tr, np.zeros(h), h)[1]),
            "Moving Average": ("moving_average", lambda tr, h: fe._train_moving_average(tr, np.zeros(h), h)[1]),
            "Linear Trend": ("linear_trend", lambda tr, h: fe._train_linear_trend(tr, np.zeros(h), h)[1]),
        }
        if season_len > 1:
            fit_fns["Seasonal Naive"] = ("seasonal_naive",
                                         lambda tr, h: fe._train_seasonal_naive(tr, np.zeros(h), h, season_len)[1])
        results = {}
        for name, (kind, fn) in fit_fns.items():
            bt = rolling_origin_backtest(y, fn, horizon=bt_h, n_folds=3,
                                         min_train=max(3, season_len + 1 if season_len > 1 else 3))
            if bt["n_folds"] and bt["metrics"]["MAE"] is not None:
                results[name] = (kind, fn, bt)
        if not results:
            report.append("Forecasting skipped: not enough history for any walk-forward test fold.")
            return

        best = min(results, key=lambda k: results[k][2]["metrics"]["MAE"])
        baselines = [k for k, v in results.items() if v[0] in ("naive", "seasonal_naive")]
        best_base = min(baselines, key=lambda k: results[k][2]["metrics"]["MAE"]) if baselines else None
        note = ""
        if best_base and results[best][0] not in ("naive", "seasonal_naive"):
            if not results[best][2]["metrics"]["MAE"] <= results[best_base][2]["metrics"]["MAE"] * 0.98:
                note = (f"{best} was not at least 2% better than {best_base} in walk-forward testing, so the "
                        f"simpler baseline is used.")
                best = best_base
        kind, fn, bt = results[best]
        forecast_vals = np.asarray(fn(y, horizon), dtype=float)[:horizon]
        errors = np.asarray(bt["errors"])
        err_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

        last_date = series.index[-1]
        if freq_label == "monthly":
            future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(horizon)]
        elif freq_label == "quarterly":
            future_dates = [last_date + pd.DateOffset(months=3 * (i + 1)) for i in range(horizon)]
        elif freq_label == "yearly":
            future_dates = [last_date + pd.DateOffset(years=i + 1) for i in range(horizon)]
        else:
            future_dates = [last_date + freq_delta * (i + 1) for i in range(horizon)]

        out = pd.DataFrame({
            "Date": future_dates,
            "Forecast_Revenue": forecast_vals,
            "Lower_90": forecast_vals - 1.645 * err_std,
            "Upper_90": forecast_vals + 1.645 * err_std,
        })
        out.to_csv(processed_dir / "revenue_forecast.csv", index=False)

        base_m = results[best_base][2]["metrics"] if best_base else None
        sel_m = bt["metrics"]
        improvement = (round((1 - sel_m["MAE"] / base_m["MAE"]) * 100, 2)
                       if base_m and base_m["MAE"] else None)
        idx = series.index
        evaluation = {
            "target": target_col, "frequency": freq_label, "periods": n, "horizon": horizon,
            "selected_model": best, "selection_note": note,
            "selection_rule": "Lowest walk-forward MAE; a non-baseline model must beat the best naive baseline by >= 2%.",
            "method": "rolling-origin (walk-forward) backtest", "folds": bt["n_folds"], "fold_horizon": bt_h,
            "data_range": [str(idx[0].date()), str(idx[-1].date())],
            "fold_test_ranges": [[str(idx[f["test_start"]].date()), str(idx[f["test_end"] - 1].date())]
                                 for f in bt["folds"]],
            "selected_metrics": sel_m, "baseline": best_base, "baseline_metrics": base_m,
            "improvement_vs_baseline_pct": improvement,
            "candidates": {k: v[2]["metrics"] for k, v in results.items()},
            "interval": "90% = forecast ± 1.645 × std of out-of-sample walk-forward errors (approximate)",
            "rows_input": n_input, "rows_without_valid_date": n_bad_date,
            "rows_without_valid_target": n_bad_y, "rows_combined_same_date": n_dup,
            "missing_periods": missing_periods,
        }
        with open(processed_dir / "forecast_evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2, default=str)

        parts = [f"Forecast for '{target_col}' ({freq_label}, {n} periods): {best} selected by walk-forward "
                 f"backtest over {bt['n_folds']} fold(s), MAE={sel_m['MAE']:.1f}"]
        if improvement is not None and best != best_base:
            parts.append(f"{improvement:+.1f}% vs {best_base}")
        if note:
            parts.append(note)
        if n_dup:
            parts.append(f"{n_dup} row(s) sharing a date were summed per period")
        if n_bad_date or n_bad_y:
            parts.append(f"{n_bad_date} row(s) without a valid date and {n_bad_y} without a valid target were left out")
        if missing_periods:
            parts.append(f"{missing_periods} missing period(s) detected in the series (not filled in)")
        report.append("; ".join(parts) + ". 90% intervals come from out-of-sample errors and are approximate.")
    except Exception as e:
        report.append(f"Forecasting was available but failed to run: {e}")


def run_full_analysis(dataset_id: str, df: pd.DataFrame = None, mapping=None, confirmed_roles=None, overrides=None,
                      data_quality: dict = None):
    """Runs the entire dataset-first pipeline for `dataset_id`. If `df` is
    not given, reads the previously-saved raw file for this dataset_id.
    Returns a dict with dataset, capabilities, generic_kpis, report."""
    validate_dataset_id(dataset_id)
    ensure_dataset_dirs(dataset_id)
    processed_dir = dataset_processed_dir(dataset_id)
    models_dir = dataset_models_dir(dataset_id)

    if df is None:
        raw_dir = dataset_raw_path(dataset_id, "").parent
        candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError(f"No raw upload found for dataset_id={dataset_id!r}.")
        path = candidates[0]
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)

    # --- Duplicate column guard (M-1) ---
    df, dup_rename_map = normalize_duplicate_columns(df)

    # +/-infinity is never a valid business value; treat it as missing so no
    # downstream sum, mean or model silently turns into inf/NaN. (The data
    # quality report below still records how many there were.)
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[num_cols].to_numpy(dtype=float)).sum()) if len(num_cols) else 0
    if n_inf:
        original_df_for_quality = df
        df = df.copy()
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    else:
        original_df_for_quality = df

    if mapping is None:
        _, mapping = profile_and_map(df)

    dataset, map_report = confirm_and_standardize(df, mapping, confirmed_roles=confirmed_roles, overrides=overrides)
    capabilities = detect_capabilities(dataset)

    report = list(map_report)

    if n_inf:
        report.insert(0, f"{n_inf} infinite value(s) were treated as missing.")
    if dup_rename_map:
        report.insert(0, f"Duplicate column names detected and normalized: "
                         f"{', '.join(f'{v!r} -> {k!r}' for k, v in dup_rename_map.items())}.")

    # Persist standardized data
    dataset.df.to_csv(processed_dir / "standardized_data.csv", index=False)

    # Data-quality report: original vs cleaned + audit trail. When the caller
    # did no cleaning (e.g. the API), assess the data as received.
    try:
        from src.data_engine.data_quality import assess_quality
        if data_quality is None:
            data_quality = {"original": assess_quality(original_df_for_quality), "cleaned": None, "audit": None}
        with open(processed_dir / "data_quality_report.json", "w") as f:
            json.dump(data_quality, f, indent=2, default=str)
        audit = data_quality.get("audit") or {}
        if audit:
            report.append(
                f"Data cleaning applied: {audit.get('total_imputed_cells', 0)} cell(s) imputed, "
                f"{audit.get('total_rows_removed', 0)} row(s) removed, "
                f"{audit.get('blank_strings_normalised', 0)} blank/placeholder cell(s) marked missing. "
                f"Quality score {data_quality['original']['score']} → {data_quality['cleaned']['score']}."
            )
        else:
            report.append(f"Data quality score {data_quality['original']['score']} / 100 (no cleaning applied).")
    except Exception as e:  # quality reporting must never block the analysis
        report.append(f"Data-quality report could not be produced: {type(e).__name__}.")

    # Generic KPIs (always computed — never fabricated)
    generic_kpis = compute_generic_kpis(dataset)
    with open(processed_dir / "generic_kpis.json", "w") as f:
        json.dump([k.to_dict() for k in generic_kpis], f, indent=2)
    with open(processed_dir / "universal_kpis.json", "w") as f:
        json.dump({
            "has_date": dataset.has_date, "has_revenue": dataset.has_revenue,
            "n_channels": len(dataset.channels), "n_rows": len(dataset.df),
            "n_controls": len(dataset.control_cols), "n_dimensions": len(dataset.dimension_cols),
            "date_column": dataset.date_col, "revenue_column": dataset.revenue_col,
            "channel_columns": dataset.channels, "dimension_columns": dataset.dimension_cols,
            "control_columns": dataset.control_cols, "identifier_columns": dataset.identifier_cols,
        }, f, indent=2)

    with open(processed_dir / "capabilities.json", "w") as f:
        json.dump(capabilities.to_dict(), f, indent=2)

    # Legacy marketing_analytics stage: only meaningful when the dataset has
    # date + revenue (the minimum the legacy KPI shape needs).
    if dataset.has_date and dataset.has_revenue:
        report.append(f"Date+revenue present — generic KPI/trend analytics are computed for '{dataset.revenue_col}'.")
    else:
        report.append("Legacy-style revenue/date analytics skipped: date and/or revenue column not identified.")

    # Marketing Mix Modeling stage
    if capabilities.is_available("marketing_mix_modeling"):
        _run_generic_mmm(dataset, models_dir, processed_dir, report)
    else:
        report.append(f"Marketing mix modeling stage skipped: {capabilities.get('marketing_mix_modeling').reason}")

    # Forecast stage
    if capabilities.is_available("forecasting"):
        _run_generic_forecast(dataset, models_dir, processed_dir, report)
    else:
        report.append(f"Forecasting stage skipped: {capabilities.get('forecasting').reason}")

    # ROI stage — compute spend-to-revenue ratio (NOT attributed ROAS)
    if capabilities.is_available("roi_roas"):
        try:
            df2 = dataset.df
            revenue_total = pd.to_numeric(df2[dataset.revenue_col], errors="coerce").fillna(0.0).sum()
            rows = []
            for ch in dataset.channels:
                spend = pd.to_numeric(df2[ch], errors="coerce").fillna(0.0).sum()
                # This is total_revenue / channel_spend — a naive ratio, NOT
                # attributed ROAS.  True attributed ROAS requires a causal model
                # isolating each channel's incremental revenue contribution.
                spend_revenue_ratio = (float(revenue_total / spend)) if spend else None
                rows.append({
                    "channel": ch,
                    "total_spend": float(spend),
                    "spend_revenue_ratio": spend_revenue_ratio,
                    "note": "Naive total-revenue-to-channel-spend ratio, not attributed ROAS",
                })
            pd.DataFrame(rows).to_csv(processed_dir / "channel_roi.csv", index=False)
            report.append(f"Spend-revenue ratios computed for {len(rows)} channel(s).")
        except Exception as e:
            report.append(f"ROI/ROAS stage was available but failed: {e}")
    else:
        report.append(f"ROI/ROAS stage skipped: {capabilities.get('roi_roas').reason}")

    if not capabilities.is_available("budget_optimization"):
        report.append(f"Budget optimization skipped: {capabilities.get('budget_optimization').reason}")
    if not capabilities.is_available("what_if_simulation"):
        report.append(f"What-if simulation skipped: {capabilities.get('what_if_simulation').reason}")

    # Phase 2: Build canonical dataset + analysis context (backward-compatible addition)
    canonical = CanonicalDataset.from_standardized(
        dataset_id=dataset_id,
        std_dataset=dataset,
        filename="",
        capability_report=capabilities,
    )
    context = AnalysisContext.from_canonical(canonical)

    # Persist canonical metadata alongside existing artifacts
    with open(processed_dir / "canonical_metadata.json", "w") as f:
        json.dump(canonical.to_metadata_dict(), f, indent=2, default=str)

    # Phase 3: Run dynamic analytics engines and persist results
    analytics_result_dict = None
    try:
        from src.analytics.contracts import AnalyticsResult
        from src.analytics.kpi_engine import discover_kpis
        from src.analytics.visualization_engine import recommend_charts
        from src.analytics.insight_engine import discover_insights
        from src.analytics.relationship_engine import analyze_relationships

        analytics_result = AnalyticsResult(
            dataset_id=dataset_id,
            kpis=discover_kpis(context),
            charts=recommend_charts(context),
            insights=discover_insights(context),
            relationships=analyze_relationships(context),
        )
        analytics_result_dict = analytics_result.to_dict()
        with open(processed_dir / "dynamic_analytics.json", "w") as f:
            json.dump(analytics_result_dict, f, indent=2, default=str)
        report.append(
            f"Phase 3 dynamic analytics: {len(analytics_result.kpis)} KPI(s), "
            f"{len(analytics_result.charts)} chart(s), "
            f"{len(analytics_result.insights)} insight(s), "
            f"{len(analytics_result.relationships)} relationship(s)."
        )
    except Exception as e:
        report.append(f"Phase 3 dynamic analytics failed (non-fatal): {e}")

    # Phase 4: Run predictive analytics engines and persist results
    prediction_report_dict = None
    try:
        from src.prediction.task_detector import detect_predictive_tasks
        from src.prediction.contracts import PredictionReport
        from src.prediction.forecasting_engine import run_forecast
        from src.prediction.regression_engine import run_regression
        from src.prediction.classification_engine import run_classification
        from src.prediction.segmentation_engine import run_segmentation

        tasks = detect_predictive_tasks(context)
        pred_report = PredictionReport(dataset_id=dataset_id, tasks=tasks)

        for task in tasks:
            if not task.requirements_met:
                continue
            try:
                if task.task_type == "forecasting":
                    pred_report.forecast = run_forecast(context, target_column=task.target_column)
                elif task.task_type == "regression":
                    pred_report.regression = run_regression(context, target_column=task.target_column)
                elif task.task_type == "classification":
                    pred_report.classification = run_classification(context, target_column=task.target_column)
                elif task.task_type == "segmentation":
                    pred_report.segmentation = run_segmentation(context)
            except Exception as task_err:
                report.append(f"Phase 4 {task.task_type} failed (non-fatal): {task_err}")

        prediction_report_dict = pred_report.to_dict()
        with open(processed_dir / "prediction_report.json", "w") as f:
            json.dump(prediction_report_dict, f, indent=2, default=str)

        completed = [t for t in ["forecast", "regression", "classification", "segmentation"]
                     if getattr(pred_report, t) is not None]
        report.append(
            f"Phase 4 prediction: {len(tasks)} task(s) detected, "
            f"{len(completed)} completed ({', '.join(completed) if completed else 'none'})."
        )
    except Exception as e:
        report.append(f"Phase 4 prediction failed (non-fatal): {e}")

    # Phase 5: Run anomaly detection + driver intelligence and persist results
    phase5_result_dict = None
    try:
        from src.anomaly.detector import detect_anomalies
        from src.anomaly.driver_engine import analyze_drivers
        from src.anomaly.contracts import Phase5Result

        anomaly_report = detect_anomalies(context, dataset_id)
        driver_report = analyze_drivers(context, dataset_id)

        phase5_result = Phase5Result(
            dataset_id=dataset_id,
            anomaly_report=anomaly_report,
            driver_report=driver_report,
        )
        phase5_result_dict = phase5_result.to_dict()
        with open(processed_dir / "phase5_analysis.json", "w") as f:
            json.dump(phase5_result_dict, f, indent=2, default=str)

        n_anomalies = len(anomaly_report.anomalies) if anomaly_report else 0
        n_drivers = len(driver_report.drivers) if driver_report else 0
        target_info = f"target='{driver_report.target_column}'" if driver_report and driver_report.target_column else "no target"
        report.append(
            f"Phase 5 anomaly+driver analysis: {n_anomalies} anomaly(ies), "
            f"{n_drivers} driver(s) ({target_info})."
        )
    except Exception as e:
        report.append(f"Phase 5 anomaly+driver analysis failed (non-fatal): {e}")

    # Phase 6: Run marketing intelligence pipeline and persist results
    phase6_result_dict = None
    try:
        from src.marketing.orchestrator import run_marketing_analysis

        phase6_result = run_marketing_analysis(
            canonical=canonical,
            context=context,
            dataset_id=dataset_id,
        )
        phase6_result_dict = phase6_result.to_dict()
        with open(processed_dir / "phase6_marketing.json", "w") as f:
            json.dump(phase6_result_dict, f, indent=2, default=str)

        readiness = "NOT_READY"
        n_channels = 0
        if phase6_result.marketing_report:
            readiness = phase6_result.marketing_report.readiness.value
            n_channels = len(phase6_result.marketing_report.channel_fields)
        has_mmm = phase6_result.mmm_result is not None and phase6_result.mmm_result._model is not None
        report.append(
            f"Phase 6 marketing intelligence: readiness={readiness}, "
            f"{n_channels} channel(s), MMM={'fitted' if has_mmm else 'not fitted'}."
        )
    except Exception as e:
        report.append(f"Phase 6 marketing intelligence failed (non-fatal): {e}")

    # Phase 7: Run Decision Intelligence orchestrator and persist results
    decision_report_dict = None
    phase7_path = processed_dir / "phase7_decisions.json"
    # Stale-result protection: remove any previous report before running
    # so a failure cannot leave an old report masquerading as current.
    if phase7_path.exists():
        phase7_path.unlink()
    try:
        from src.decision.orchestrator import DecisionOrchestrator, OrchestrationRequest

        request = OrchestrationRequest(
            dataset_id=dataset_id,
            analytics_result=analytics_result_dict,
            prediction_report=prediction_report_dict,
            phase5_result=phase5_result_dict,
            marketing_result=phase6_result_dict,
        )
        orchestrator = DecisionOrchestrator()
        orch_report = orchestrator.run(request)
        decision_report_dict = orch_report.to_dict()
        with open(phase7_path, "w") as f:
            json.dump(decision_report_dict, f, indent=2, default=str)

        n_decisions = len(orch_report.decisions())
        report.append(
            f"Phase 7 decision intelligence: status={orch_report.status}, "
            f"{n_decisions} decision(s) generated."
        )
    except Exception as e:
        report.append(f"Phase 7 decision intelligence failed (non-fatal): {e}")
        # Persist a failure marker so consumers can distinguish
        # "failed" from "never run".
        try:
            with open(phase7_path, "w") as f:
                json.dump({
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "error": str(e),
                }, f, indent=2)
        except Exception:
            pass  # Best-effort — filesystem errors don't crash the pipeline

    return {
        "dataset": dataset,
        "capabilities": capabilities,
        "generic_kpis": generic_kpis,
        "report": report,
        # Phase 2 additions (backward-compatible — new keys only)
        "canonical": canonical,
        "context": context,
        # Phase 4 addition
        "prediction_report": prediction_report_dict,
        # Phase 5 addition
        "phase5_result": phase5_result_dict,
        # Phase 6 addition
        "phase6_result": phase6_result_dict,
        # Phase 7 addition
        "decision_report": decision_report_dict,
    }
