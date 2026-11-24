"""Phase 6 — Dataset-Agnostic Channel Contribution & Attribution.

Model-estimated decomposition using leave-one-channel-out ablation on the
fitted MMM.  Preserved algorithm from ``src/models/channel_contribution.py``
but with dynamic channel/target inputs from ``MMMResult``.

This is a MODEL-ESTIMATED decomposition — it depends on the model's
functional form and fitted coefficients.  Results should be read as
"the model attributes this much of the target to this channel," not as
ground-truth causal proof.
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

from src.marketing.contracts import ChannelContribution, MMMResult
from src.marketing.mmm_engine import (
    build_feature_matrix, channel_feature_name, predict_with_mmm,
)

logger = logging.getLogger(__name__)


def compute_contributions(
    df: pd.DataFrame,
    mmm: MMMResult,
) -> list[ChannelContribution]:
    """Compute each channel's contribution via leave-one-channel-out ablation.

    Parameters
    ----------
    df : the dataset (same or overlapping with what the MMM was fitted on).
    mmm : a fitted ``MMMResult`` from ``mmm_engine.fit_mmm()``.

    Returns
    -------
    List of ``ChannelContribution`` contracts, sorted by contribution (desc).
    """
    if mmm._model is None or mmm._scaler is None:
        logger.warning("MMM has no fitted model — cannot compute contributions.")
        return []

    channels = mmm.channels
    selected = mmm.selected_model
    target_col = mmm.target_column

    X_full = build_feature_matrix(df, mmm)
    full_pred = predict_with_mmm(X_full, mmm)

    # Total spend across all channels
    total_spend_all = 0.0
    for ch in channels:
        if ch in df.columns:
            total_spend_all += pd.to_numeric(
                df[ch], errors="coerce").fillna(0).sum()

    # Leave-one-channel-out ablation
    contributions_raw: dict[str, float] = {}
    total_incremental = 0.0

    for ch in channels:
        feat_name = channel_feature_name(ch, selected)
        if feat_name not in X_full.columns:
            contributions_raw[ch] = 0.0
            continue
        X_ablated = X_full.copy()
        X_ablated[feat_name] = 0.0
        ablated_pred = predict_with_mmm(X_ablated, mmm)
        incremental = float((full_pred - ablated_pred).clip(min=0).sum())
        contributions_raw[ch] = incremental
        total_incremental += incremental

    # Build result contracts
    results: list[ChannelContribution] = []
    for ch in channels:
        spend = 0.0
        if ch in df.columns:
            spend = float(pd.to_numeric(
                df[ch], errors="coerce").fillna(0).sum())

        contribution = contributions_raw.get(ch, 0.0)
        contribution_share = (
            contribution / total_incremental
            if total_incremental > 0 else 0.0
        )

        # Transformed spend (for reference)
        feat_name = channel_feature_name(ch, selected)
        transformed_spend = 0.0
        if feat_name in X_full.columns:
            transformed_spend = float(X_full[feat_name].sum())

        roi = contribution / spend if spend > 0 else 0.0

        results.append(ChannelContribution(
            channel=ch,
            spend=spend,
            transformed_spend=transformed_spend,
            contribution=contribution,
            contribution_share=contribution_share,
            roi=roi,
        ))

    # Sort by contribution descending
    results.sort(key=lambda c: c.contribution, reverse=True)
    return results
