"""Phase 6 — Dataset-Agnostic What-If Simulator.

Lets users hypothesise a total budget and/or individual channel allocations
and see the fitted model's predicted target change.  All predictions come
from the actual fitted MMM — never hardcoded.

Non-causal language:  Results use phrases like "the model estimates" and
"predicted outcome," never "X causes Y" or "increasing X leads to Y."
The disclaimer is embedded in every ``WhatIfResult``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.marketing.contracts import (
    MMMResult, WhatIfScenario, WhatIfResult,
)
from src.marketing.optimizer import (
    _get_recent_carry_in, _predict_for_allocation, optimize_budget,
)

logger = logging.getLogger(__name__)


def _build_scenario(
    name: str,
    allocation: dict,
    channels: list[str],
    mmm: MMMResult,
    carry_in: dict,
    control_values: dict,
    baseline_pred: float,
) -> WhatIfScenario:
    """Build a single what-if scenario."""
    alloc_vec = np.array([allocation.get(ch, 0.0) for ch in channels])
    total_budget = float(alloc_vec.sum())
    predicted = _predict_for_allocation(
        alloc_vec, mmm, carry_in, control_values)
    change = predicted - baseline_pred
    change_pct = (change / baseline_pred * 100) if baseline_pred != 0 else 0.0
    roi = predicted / total_budget if total_budget > 0 else 0.0

    return WhatIfScenario(
        name=name,
        allocation=allocation,
        total_budget=total_budget,
        predicted_target=predicted,
        target_change=change,
        target_change_pct=change_pct,
        roi=roi,
    )


def simulate_custom_allocation(
    df: pd.DataFrame,
    mmm: MMMResult,
    allocation: dict,
    lookback: int = 8,
) -> WhatIfResult:
    """Simulate a fully custom user-specified channel allocation.

    Parameters
    ----------
    df : the dataset.
    mmm : fitted ``MMMResult``.
    allocation : dict mapping channel name -> spend amount.
    lookback : periods for carry-in estimation.

    Returns
    -------
    ``WhatIfResult`` with the custom scenario and the current baseline.
    """
    channels = mmm.channels
    carry_in = _get_recent_carry_in(df, mmm, lookback)

    recent = df.tail(lookback)
    control_values: dict = {}
    for c in mmm.control_cols:
        if c in recent.columns:
            control_values[c] = float(
                pd.to_numeric(recent[c], errors="coerce").fillna(0).mean())

    # Baseline (current) allocation
    current_alloc: dict = {}
    for ch in channels:
        if ch in recent.columns:
            current_alloc[ch] = float(
                pd.to_numeric(recent[ch], errors="coerce").fillna(0).mean())
        else:
            current_alloc[ch] = 0.0

    current_vec = np.array([current_alloc[ch] for ch in channels])
    baseline_pred = _predict_for_allocation(
        current_vec, mmm, carry_in, control_values)

    baseline_scenario = _build_scenario(
        "Current", current_alloc, channels, mmm, carry_in,
        control_values, baseline_pred)

    custom_scenario = _build_scenario(
        "Custom", allocation, channels, mmm, carry_in,
        control_values, baseline_pred)

    return WhatIfResult(
        scenarios=[custom_scenario],
        baseline_scenario=baseline_scenario,
        target_column=mmm.target_column,
        channels=list(channels),
    )


def simulate_total_budget(
    df: pd.DataFrame,
    mmm: MMMResult,
    total_budget: float,
    keep_current_mix: bool = True,
    lookback: int = 8,
) -> WhatIfResult:
    """Scale the current channel mix proportionally to a new total budget.

    Parameters
    ----------
    df : the dataset.
    mmm : fitted ``MMMResult``.
    total_budget : the hypothesised total spend.
    keep_current_mix : if True, scales proportionally; if False, equal split.
    lookback : periods for carry-in estimation.
    """
    channels = mmm.channels
    recent = df.tail(lookback)

    current_alloc: dict = {}
    for ch in channels:
        if ch in recent.columns:
            current_alloc[ch] = float(
                pd.to_numeric(recent[ch], errors="coerce").fillna(0).mean())
        else:
            current_alloc[ch] = 0.0

    current_total = sum(current_alloc.values())
    if keep_current_mix and current_total > 0:
        scale = total_budget / current_total
        new_alloc = {ch: v * scale for ch, v in current_alloc.items()}
    else:
        n = len(channels)
        new_alloc = {ch: total_budget / n for ch in channels} if n > 0 else {}

    return simulate_custom_allocation(df, mmm, new_alloc, lookback)


def simulate_standard_scenarios(
    df: pd.DataFrame,
    mmm: MMMResult,
    lookback: int = 8,
) -> WhatIfResult:
    """Generate standard comparison: Current, Optimised, +20%, -20%.

    These are the same four scenarios as the original simulator, now
    dataset-agnostic.
    """
    channels = mmm.channels
    carry_in = _get_recent_carry_in(df, mmm, lookback)

    recent = df.tail(lookback)
    control_values: dict = {}
    for c in mmm.control_cols:
        if c in recent.columns:
            control_values[c] = float(
                pd.to_numeric(recent[c], errors="coerce").fillna(0).mean())

    current_alloc: dict = {}
    for ch in channels:
        if ch in recent.columns:
            current_alloc[ch] = float(
                pd.to_numeric(recent[ch], errors="coerce").fillna(0).mean())
        else:
            current_alloc[ch] = 0.0

    current_vec = np.array([current_alloc[ch] for ch in channels])
    baseline_pred = _predict_for_allocation(
        current_vec, mmm, carry_in, control_values)

    baseline_scenario = _build_scenario(
        "Current", current_alloc, channels, mmm, carry_in,
        control_values, baseline_pred)

    scenarios = [baseline_scenario]

    # Optimised scenario
    try:
        opt_result = optimize_budget(df, mmm, lookback=lookback)
        if opt_result.success and opt_result.optimized_allocation:
            opt_scenario = _build_scenario(
                "Optimised", opt_result.optimized_allocation,
                channels, mmm, carry_in, control_values, baseline_pred)
            scenarios.append(opt_scenario)
    except Exception as exc:
        logger.warning("Optimisation scenario failed: %s", exc)

    # Aggressive (+20%) and Conservative (-20%)
    for label, mult in [("Aggressive (+20%)", 1.20),
                        ("Conservative (-20%)", 0.80)]:
        total = sum(current_alloc.values()) * mult
        if total > 0 and sum(current_alloc.values()) > 0:
            scale = mult
            scaled = {ch: v * scale for ch, v in current_alloc.items()}
        else:
            n = len(channels)
            scaled = {ch: total / n for ch in channels} if n > 0 else {}
        scenario = _build_scenario(
            label, scaled, channels, mmm, carry_in,
            control_values, baseline_pred)
        scenarios.append(scenario)

    return WhatIfResult(
        scenarios=scenarios,
        baseline_scenario=baseline_scenario,
        target_column=mmm.target_column,
        channels=list(channels),
    )
