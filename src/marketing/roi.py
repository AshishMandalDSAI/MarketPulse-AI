"""Phase 6 — Dataset-Agnostic ROI / ROAS Calculation.

Computes both naive (ratio-based) and model-estimated ROI for each
marketing channel.  No column names are hardcoded — all field names
come from the ``MMMResult`` and ``MarketingCapabilityReport``.

Naive ROAS = total_target / total_spend (simple ratio, no modeling).
Model-estimated ROI = model-attributed contribution / total_spend.
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

from src.marketing.contracts import ROIResult, MMMResult, ChannelContribution

logger = logging.getLogger(__name__)


def compute_roi_naive(
    df: pd.DataFrame,
    channels: list[str],
    target_col: str,
) -> list[ROIResult]:
    """Compute naive ROAS for each channel — simple ratio, no modeling.

    naive_roas = total_target / channel_spend
    (How much target was generated per unit of spend, assuming all target
    is attributable proportionally — which is almost certainly wrong, but
    is a useful baseline.)
    """
    results: list[ROIResult] = []

    total_target = float(
        pd.to_numeric(df[target_col], errors="coerce").fillna(0).sum())

    for ch in channels:
        if ch not in df.columns:
            results.append(ROIResult(
                channel=ch, method="naive_ratio"))
            continue

        spend = float(
            pd.to_numeric(df[ch], errors="coerce").fillna(0).sum())

        naive_roas = total_target / spend if spend > 0 else None

        results.append(ROIResult(
            channel=ch,
            total_spend=spend,
            total_target=total_target,
            naive_roas=naive_roas,
            method="naive_ratio",
        ))

    return results


def compute_roi_model_based(
    df: pd.DataFrame,
    channels: list[str],
    target_col: str,
    contributions: list[ChannelContribution],
) -> list[ROIResult]:
    """Compute model-estimated ROI using channel contributions.

    model_estimated_roi = model_contribution / channel_spend
    """
    contrib_map = {c.channel: c for c in contributions}
    total_target = float(
        pd.to_numeric(df[target_col], errors="coerce").fillna(0).sum())

    results: list[ROIResult] = []
    for ch in channels:
        spend = 0.0
        if ch in df.columns:
            spend = float(
                pd.to_numeric(df[ch], errors="coerce").fillna(0).sum())

        naive_roas = total_target / spend if spend > 0 else None

        model_roi = None
        if ch in contrib_map and spend > 0:
            model_roi = contrib_map[ch].contribution / spend

        results.append(ROIResult(
            channel=ch,
            total_spend=spend,
            total_target=total_target,
            naive_roas=naive_roas,
            model_estimated_roi=model_roi,
            method="model_based",
        ))

    return results
