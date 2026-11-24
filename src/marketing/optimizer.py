"""Phase 6 — Dataset-Agnostic Budget Optimizer.

Finds the channel allocation of a fixed total marketing budget that
maximises predicted target, using scipy SLSQP.  Preserved algorithm
from ``src/optimization/budget_optimizer.py`` but with dynamic
channel/target inputs from ``MMMResult``.

Because adstock is a time-series transform (depends on the whole spend
history), the optimiser works on a single REPRESENTATIVE PERIOD's
allocation: it holds the adstocked carry-in from prior periods fixed
at each channel's recent average level and evaluates how re-allocating
this period's incremental spend changes predicted target through the
fitted saturation curves.  This is a standard simplification, documented
here rather than silently assumed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.adstock import geometric_adstock
from src.models.saturation import hill_saturation
from src.marketing.contracts import BudgetOptimizationResult, MMMResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carry-in estimation (preserved from original)
# ---------------------------------------------------------------------------

def _get_recent_carry_in(df: pd.DataFrame, mmm: MMMResult,
                         lookback: int = 8) -> dict:
    """Estimate each channel's adstock carry-in from recent history."""
    carry_in: dict = {}
    for ch in mmm.channels:
        if ch not in df.columns:
            carry_in[ch] = 0.0
            continue
        spend_hist = pd.to_numeric(
            df[ch], errors="coerce").fillna(0).values[-lookback:]
        decay = mmm.channel_params.get(ch, {}).get("decay", 0.0)
        if len(spend_hist) == 0:
            carry_in[ch] = 0.0
            continue
        full_adstock = geometric_adstock(spend_hist, decay)
        carry_in[ch] = float(full_adstock[-1] - spend_hist[-1])
    return carry_in


# ---------------------------------------------------------------------------
# Prediction for a single allocation vector
# ---------------------------------------------------------------------------

def _predict_for_allocation(
    allocation: np.ndarray,
    mmm: MMMResult,
    carry_in: dict,
    control_values: dict,
) -> float:
    """Predict target for a given channel spend allocation vector."""
    if mmm._model is None or mmm._scaler is None:
        return 0.0

    channels = mmm.channels
    selected = mmm.selected_model
    channel_params = mmm.channel_params
    feature_names = mmm.feature_names

    feat: dict = {}
    for i, ch in enumerate(channels):
        spend = allocation[i]
        p = channel_params.get(ch, {"decay": 0.0, "k": 1.0, "shape": 1.0})
        if selected == "baseline":
            feat[f"{ch}__raw"] = spend
        elif selected == "adstock":
            adstocked = spend + p["decay"] * carry_in.get(ch, 0.0)
            feat[f"{ch}__adstock"] = adstocked
        else:  # adstock_saturation
            adstocked = spend + p["decay"] * carry_in.get(ch, 0.0)
            feat[f"{ch}__adstock_sat"] = hill_saturation(
                np.array([adstocked]), p["k"], p["shape"])[0]

    for c, v in control_values.items():
        feat[c] = v

    # Build DataFrame with correct column order
    X = pd.DataFrame([feat])
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]

    X_scaled = mmm._scaler.transform(X)
    return float(mmm._model.predict(X_scaled)[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_budget(
    df: pd.DataFrame,
    mmm: MMMResult,
    total_budget: Optional[float] = None,
    min_pct: float = 0.05,
    max_pct: float = 0.55,
    lookback: int = 8,
) -> BudgetOptimizationResult:
    """Find the target-maximising allocation of ``total_budget`` across channels.

    Parameters
    ----------
    df : the dataset (for computing carry-in and current allocation).
    mmm : fitted ``MMMResult``.
    total_budget : total budget to allocate. If None, uses the sum of
        recent average spend across channels.
    min_pct / max_pct : fractional bounds per channel (of total budget).
    lookback : number of recent periods for carry-in and current allocation.

    Returns
    -------
    ``BudgetOptimizationResult`` contract.
    """
    result = BudgetOptimizationResult()
    channels = mmm.channels
    result.channels = list(channels)
    result.target_column = mmm.target_column

    if mmm._model is None or mmm._scaler is None:
        result.message = "No fitted model available for optimisation."
        return result

    n = len(channels)
    if n == 0:
        result.message = "No channels available for optimisation."
        return result

    # Current allocation (recent average)
    recent = df.tail(lookback)
    current_allocation: dict = {}
    for ch in channels:
        if ch in recent.columns:
            current_allocation[ch] = float(
                pd.to_numeric(recent[ch], errors="coerce").fillna(0).mean())
        else:
            current_allocation[ch] = 0.0

    result.current_allocation = current_allocation

    if total_budget is None:
        total_budget = sum(current_allocation.values())

    if total_budget <= 0:
        result.message = "Total budget is zero or negative."
        return result

    result.total_budget = total_budget
    result.constraints = {"min_pct": min_pct, "max_pct": max_pct}

    # Carry-in and control values
    carry_in = _get_recent_carry_in(df, mmm, lookback)
    control_values: dict = {}
    for c in mmm.control_cols:
        if c in recent.columns:
            control_values[c] = float(
                pd.to_numeric(recent[c], errors="coerce").fillna(0).mean())

    # Initial allocation
    x0 = np.array([current_allocation[ch] for ch in channels], dtype=float)
    if x0.sum() <= 0:
        x0 = np.full(n, total_budget / n)
    else:
        x0 = x0 / x0.sum() * total_budget

    lower = total_budget * min_pct
    upper = total_budget * max_pct
    bounds = [(lower, upper) for _ in range(n)]

    def objective(alloc):
        return -_predict_for_allocation(alloc, mmm, carry_in, control_values)

    constraints = [{"type": "eq", "fun": lambda alloc: np.sum(alloc) - total_budget}]

    try:
        opt = minimize(
            objective, x0, method="SLSQP", bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9},
        )
        result.success = bool(opt.success)
        result.message = str(opt.message)
        result.optimized_allocation = {
            ch: float(opt.x[i]) for i, ch in enumerate(channels)
        }
    except Exception as exc:
        logger.warning("Optimisation failed: %s", exc)
        result.message = f"Optimisation error: {exc}"
        result.optimized_allocation = dict(current_allocation)
        return result

    # Predicted targets
    current_vec = np.array([current_allocation[ch] for ch in channels])
    result.current_predicted_target = _predict_for_allocation(
        current_vec, mmm, carry_in, control_values)
    result.optimized_predicted_target = _predict_for_allocation(
        opt.x, mmm, carry_in, control_values)

    if result.current_predicted_target > 0:
        result.expected_improvement_pct = float(
            (result.optimized_predicted_target - result.current_predicted_target)
            / result.current_predicted_target * 100
        )

    return result
