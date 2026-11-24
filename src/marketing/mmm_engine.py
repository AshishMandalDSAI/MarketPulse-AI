"""Phase 6 — Dataset-Agnostic Marketing Mix Model Engine.

Wraps the proven adstock/saturation math from ``src/models/adstock.py`` and
``src/models/saturation.py`` with CanonicalDataset-driven inputs.  No column
names are hardcoded — channel lists, target, controls, and time column all
come from a ``MarketingCapabilityReport`` produced by field discovery.

Algorithm (preserved from the original marketing_mix_model.py):
  1. Per-channel grid search over DECAY_GRID x SHAPE_GRID — select the
     (decay, shape, k) combination whose saturated-adstocked spend best
     correlates (univariate Pearson) with the target.
  2. Three progressively richer model variants:
       - BASELINE : raw spend + controls (Ridge)
       - ADSTOCK  : geometric-adstocked spend + controls
       - ADSTOCK_SATURATION : adstocked + Hill-saturated spend + controls
  3. Time-aware chronological train/test split (no shuffling).
  4. Model selection: prefer the more theoretically grounded (complex) model
     only if it also improves (or does not meaningfully worsen) out-of-sample
     RMSE/MAPE.
  5. Returns an ``MMMResult`` contract — no pickle serialization.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.adstock import geometric_adstock, DECAY_GRID
from src.models.saturation import hill_saturation, fit_half_saturation_point, SHAPE_GRID
from src.marketing.contracts import MMMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------
MAX_ROWS_FOR_MMM = 50_000
MIN_ROWS_FOR_MMM = 12
TEST_FRACTION = 0.20
RIDGE_ALPHA = 5.0


# ---------------------------------------------------------------------------
# Evaluation helpers (preserved from original)
# ---------------------------------------------------------------------------

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — excludes zeros in denominator."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray,
              n_features: int) -> dict:
    """Compute R2, adjusted R2, RMSE, MAE, MAPE."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y_true)
    denom = n - n_features - 1
    adj_r2 = (1 - (1 - r2) * (n - 1) / denom) if denom > 0 else r2
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "r2": float(r2),
        "adjusted_r2": float(adj_r2),
        "rmse": rmse,
        "mae": mae,
        "mape": _mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Time-aware split (preserved from original)
# ---------------------------------------------------------------------------

def _time_aware_split(n_rows: int,
                      test_fraction: float = TEST_FRACTION
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train/test split — NO shuffling."""
    split_idx = int(n_rows * (1 - test_fraction))
    return np.arange(0, split_idx), np.arange(split_idx, n_rows)


# ---------------------------------------------------------------------------
# Per-channel parameter selection via grid search
# ---------------------------------------------------------------------------

def _select_channel_params(df: pd.DataFrame,
                           channels: list[str],
                           target_col: str) -> dict:
    """For each channel pick the (decay, shape, k) combination whose
    saturated-adstocked spend best correlates (univariate Pearson) with
    the target.  Documented grid search — standard MMM practice.
    """
    target = df[target_col].values
    params: dict = {}

    for ch in channels:
        spend = df[ch].values
        best = {
            "decay": DECAY_GRID[0],
            "shape": SHAPE_GRID[0],
            "k": 1.0,
            "corr": -np.inf,
        }
        for decay in DECAY_GRID:
            adstocked = geometric_adstock(spend, decay)
            k = fit_half_saturation_point(adstocked)
            for shape in SHAPE_GRID:
                sat = hill_saturation(adstocked, k, shape)
                if sat.std() < 1e-9:
                    continue
                corr = np.corrcoef(sat, target)[0, 1]
                if np.isnan(corr):
                    continue
                if corr > best["corr"]:
                    best = {
                        "decay": decay,
                        "shape": shape,
                        "k": float(k),
                        "corr": float(corr),
                    }
        params[ch] = best
    return params


# ---------------------------------------------------------------------------
# Feature-set builders (dataset-agnostic)
# ---------------------------------------------------------------------------

def _build_feature_sets(df: pd.DataFrame,
                        channels: list[str],
                        control_cols: list[str],
                        channel_params: dict,
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build three feature DataFrames: baseline, adstock-only, and
    adstock+saturation, aligned to df's index."""
    baseline = pd.DataFrame(index=df.index)
    adstock_only = pd.DataFrame(index=df.index)
    adstock_sat = pd.DataFrame(index=df.index)

    for ch in channels:
        spend = df[ch].values
        p = channel_params[ch]
        adstocked = geometric_adstock(spend, p["decay"])
        saturated = hill_saturation(adstocked, p["k"], p["shape"])

        baseline[f"{ch}__raw"] = spend
        adstock_only[f"{ch}__adstock"] = adstocked
        adstock_sat[f"{ch}__adstock_sat"] = saturated

    for c in control_cols:
        if c in df.columns:
            baseline[c] = df[c].values
            adstock_only[c] = df[c].values
            adstock_sat[c] = df[c].values

    return baseline, adstock_only, adstock_sat


# ---------------------------------------------------------------------------
# Single model variant fitting
# ---------------------------------------------------------------------------

def _fit_variant(X: pd.DataFrame, y: np.ndarray,
                 train_idx: np.ndarray, test_idx: np.ndarray,
                 alpha: float = RIDGE_ALPHA,
                 ) -> tuple:
    """Fit Ridge, return (model, scaler, train_metrics, test_metrics)."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = _evaluate(y_train, train_pred, X.shape[1])
    test_metrics = _evaluate(y_test, test_pred, X.shape[1])

    return model, scaler, train_metrics, test_metrics


# ---------------------------------------------------------------------------
# Model selection (preserved from original)
# ---------------------------------------------------------------------------

def _select_best_model(comparison: dict) -> str:
    """Prefer the more theoretically-grounded (complex) model only if it
    also improves (or does not meaningfully worsen) OOS RMSE/MAPE.
    """
    order = ["baseline", "adstock", "adstock_saturation"]
    available = [m for m in order if m in comparison]
    if not available:
        return "baseline"

    selected = available[0]
    for candidate in available[1:]:
        prev_rmse = comparison[selected]["test"]["rmse"]
        cand_rmse = comparison[candidate]["test"]["rmse"]
        prev_mape = comparison[selected]["test"]["mape"]
        cand_mape = comparison[candidate]["test"]["mape"]
        # Accept candidate if RMSE is within 0.1% and MAPE within 2%
        if cand_rmse <= prev_rmse * 1.001 and cand_mape <= prev_mape * 1.02:
            selected = candidate
    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_mmm(
    df: pd.DataFrame,
    channels: list[str],
    target_col: str,
    control_cols: Optional[list[str]] = None,
    time_col: Optional[str] = None,
    test_fraction: float = TEST_FRACTION,
    alpha: float = RIDGE_ALPHA,
) -> MMMResult:
    """Fit a Marketing Mix Model on any dataset.

    Parameters
    ----------
    df : DataFrame — the dataset (rows should be in time order if
        ``time_col`` is provided).
    channels : list of column names containing channel spend data.
    target_col : the outcome/target column (e.g. Revenue, Sales).
    control_cols : optional list of control variable columns.
    time_col : optional date/time column name — used for sorting.
    test_fraction : fraction of rows held out for testing.
    alpha : Ridge regularisation strength.

    Returns
    -------
    MMMResult contract with model diagnostics, parameters, and fitted
    objects (model + scaler kept in memory, NOT pickled).
    """
    result = MMMResult()
    result.target_column = target_col
    result.channels = list(channels)
    result.control_cols = list(control_cols or [])

    # --- Input validation ---
    if len(channels) < 1:
        result.limitations.append("No channels provided.")
        return result

    if target_col not in df.columns:
        result.limitations.append(f"Target column '{target_col}' not in DataFrame.")
        return result

    missing_channels = [c for c in channels if c not in df.columns]
    if missing_channels:
        result.limitations.append(
            f"Channel columns not found: {missing_channels}")
        channels = [c for c in channels if c in df.columns]
        if not channels:
            return result

    # --- Prepare working DataFrame ---
    work_cols = list(channels) + [target_col] + list(control_cols or [])
    if time_col and time_col in df.columns:
        work_cols.append(time_col)
    work_cols = [c for c in dict.fromkeys(work_cols)]  # deduplicate preserving order
    work_df = df[work_cols].copy()

    # Sort by time if available
    if time_col and time_col in work_df.columns:
        work_df = work_df.sort_values(time_col).reset_index(drop=True)

    # Sampling if too large
    n_rows = len(work_df)
    if n_rows > MAX_ROWS_FOR_MMM:
        result.limitations.append(
            f"Dataset sampled from {n_rows} to {MAX_ROWS_FOR_MMM} rows.")
        work_df = work_df.tail(MAX_ROWS_FOR_MMM).reset_index(drop=True)
        n_rows = len(work_df)

    if n_rows < MIN_ROWS_FOR_MMM:
        result.limitations.append(
            f"Only {n_rows} rows — need >= {MIN_ROWS_FOR_MMM} for MMM.")
        return result

    # Fill NaN with 0 for spend columns, median for controls
    for ch in channels:
        work_df[ch] = pd.to_numeric(work_df[ch], errors="coerce").fillna(0)

    # --- Time-aware split FIRST, so nothing below can see the test period ---
    train_idx, test_idx = _time_aware_split(n_rows, test_fraction)

    # Missing target values are NOT filled in (inventing revenue biases the
    # model). Their rows keep their place in the time series — adstock
    # carry-over needs the full spend sequence — but are left out of fitting
    # and scoring.
    target_series = pd.to_numeric(work_df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work_df[target_col] = target_series
    target_ok = target_series.notna().to_numpy()
    n_missing_target = int((~target_ok).sum())
    if n_missing_target:
        result.limitations.append(
            f"{n_missing_target} row(s) without a valid '{target_col}' value were excluded from "
            "fitting and evaluation (not filled in).")

    valid_controls = []
    for c in (control_cols or []):
        if c not in work_df.columns:
            continue
        s = pd.to_numeric(work_df[c], errors="coerce")
        med = s.iloc[train_idx].median()          # training rows only
        work_df[c] = s.fillna(med if pd.notna(med) else 0)
        # Skip constant controls
        if work_df[c].iloc[train_idx].std() > 1e-9:
            valid_controls.append(c)
    result.control_cols = valid_controls

    y = work_df[target_col].values

    if len(train_idx) < 5 or len(test_idx) < 2:
        result.limitations.append(
            f"Train ({len(train_idx)}) or test ({len(test_idx)}) set too small "
            f"for reliable evaluation.")
        # Try to fit anyway with larger train split
        train_idx = np.arange(0, max(n_rows - 2, 1))
        test_idx = np.arange(len(train_idx), n_rows)

    train_idx = train_idx[target_ok[train_idx]]
    test_idx = test_idx[target_ok[test_idx]]
    result.n_train = len(train_idx)
    result.n_test = len(test_idx)
    if len(train_idx) < 3 or len(test_idx) < 1:
        result.limitations.append("Too few rows with a valid target to fit and evaluate the model.")
        return result

    # --- Grid search for per-channel parameters (TRAINING rows only) ---
    channel_params = _select_channel_params(work_df.iloc[train_idx], channels, target_col)
    result.channel_params = channel_params

    # --- Build feature sets (adstock is causal: each value uses past spend only) ---
    X_baseline, X_adstock, X_adstock_sat = _build_feature_sets(
        work_df, channels, valid_controls, channel_params)

    # --- Fit all three variants ---
    comparison: dict = {}
    fitted: dict = {}
    variants = [
        ("baseline", X_baseline),
        ("adstock", X_adstock),
        ("adstock_saturation", X_adstock_sat),
    ]

    for name, X in variants:
        try:
            model, scaler, train_m, test_m = _fit_variant(
                X, y, train_idx, test_idx, alpha)
            comparison[name] = {"train": train_m, "test": test_m}
            fitted[name] = {
                "model": model,
                "scaler": scaler,
                "feature_names": list(X.columns),
            }
        except Exception as exc:
            logger.warning("MMM variant '%s' failed: %s", name, exc)
            result.limitations.append(f"Variant '{name}' failed: {exc}")

    if not comparison:
        result.limitations.append("All model variants failed to fit.")
        return result

    # --- Model selection ---
    selected = _select_best_model(comparison)
    result.selected_model = selected
    result.model_comparison = comparison
    result.train_metrics = comparison[selected]["train"]
    result.test_metrics = comparison[selected]["test"]
    result.feature_names = fitted[selected]["feature_names"]
    result._model = fitted[selected]["model"]
    result._scaler = fitted[selected]["scaler"]

    # --- Assumptions & limitations ---
    result.assumptions = [
        "Linear relationship between (transformed) spend and target "
        "(Ridge regression).",
        "Per-channel adstock/saturation parameters selected via univariate "
        "correlation on the training period only (a standard simplification; "
        "joint optimisation would use nested cross-validation).",
        "Time-aware train/test split (chronological, no shuffling) — the "
        "last ~20% of observations are held out.",
        "Model selection prefers the more theoretically grounded variant "
        "only if it also improves out-of-sample RMSE/MAPE.",
    ]
    if not time_col:
        result.assumptions.append(
            "No time column detected — rows treated in their original order.")
    result.limitations.append(
        "Model-estimated effects are correlational, not causal. "
        "Results depend on the model's functional form and the data provided.")

    return result


# ---------------------------------------------------------------------------
# Helpers for downstream modules (contribution, optimizer, simulator)
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, mmm: MMMResult) -> pd.DataFrame:
    """Rebuild the feature matrix for the selected model variant from raw data.

    This is the dataset-agnostic equivalent of
    ``channel_contribution.build_full_feature_matrix()``.
    """
    selected = mmm.selected_model
    channel_params = mmm.channel_params
    channels = mmm.channels
    control_cols = mmm.control_cols

    X = pd.DataFrame(index=df.index)
    for ch in channels:
        spend = pd.to_numeric(df[ch], errors="coerce").fillna(0).values
        p = channel_params[ch]
        if selected == "baseline":
            X[f"{ch}__raw"] = spend
        elif selected == "adstock":
            X[f"{ch}__adstock"] = geometric_adstock(spend, p["decay"])
        else:  # adstock_saturation
            adstocked = geometric_adstock(spend, p["decay"])
            X[f"{ch}__adstock_sat"] = hill_saturation(
                adstocked, p["k"], p["shape"])

    for c in control_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).values

    # Ensure column order matches training
    for col in mmm.feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[mmm.feature_names]
    return X


def channel_feature_name(channel: str, selected_model: str) -> str:
    """Return the feature name for a channel in the selected model variant."""
    if selected_model == "baseline":
        return f"{channel}__raw"
    elif selected_model == "adstock":
        return f"{channel}__adstock"
    else:
        return f"{channel}__adstock_sat"


def predict_with_mmm(X: pd.DataFrame, mmm: MMMResult) -> np.ndarray:
    """Predict target values using the fitted MMM."""
    if mmm._model is None or mmm._scaler is None:
        return np.zeros(len(X))
    X_scaled = mmm._scaler.transform(X)
    return mmm._model.predict(X_scaled)
