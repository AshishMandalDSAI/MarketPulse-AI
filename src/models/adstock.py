"""
MarketPulse AI - Adstock (Marketing Carryover) Transformation
=================================================================
Marketing effects don't disappear the instant spend stops — a TV campaign
this week still influences purchases next week. Adstock encodes this
carryover effect.

We use GEOMETRIC ADSTOCK, the standard, well-documented approach in Marketing
Mix Modeling literature (Broadbent, 1979; widely used in Google's Robyn,
Meta's Robyn/PyMC-Marketing, etc.):

    adstock_t = spend_t + decay * adstock_{t-1}

where `decay` in [0, 1) is the fraction of last week's adstocked effect that
carries into this week. decay=0 means no carryover; decay close to 1 means
long, slow-fading carryover (typical of TV/brand channels). We use
per-channel decay rates because different channels realistically have very
different carryover lengths (e.g., Email has almost no carryover; TV/Radio
carry over several weeks).
"""

import numpy as np
import pandas as pd

# Documented decay rate search grid used during model fitting.
# Chosen to span "almost no carryover" (0.05) to "long carryover" (0.75),
# which covers realistic ranges reported in MMM literature for weekly data.
DECAY_GRID = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]


def geometric_adstock(series: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock to a 1D array of weekly spend.

    Parameters
    ----------
    series : array-like of weekly spend values, in time order.
    decay : float in [0, 1). Fraction of previous week's adstocked
        value that carries forward.

    Returns
    -------
    np.ndarray of the same length, adstocked.
    """
    series = np.asarray(series, dtype=float)
    out = np.zeros_like(series)
    carry = 0.0
    for t in range(len(series)):
        carry = series[t] + decay * carry
        out[t] = carry
    return out


def apply_adstock_to_dataframe(df: pd.DataFrame, spend_cols: list, decay_map: dict) -> pd.DataFrame:
    """Return a copy of df with `{col}_adstocked` columns added for each spend col.

    decay_map: dict mapping spend column name -> decay rate.
    """
    out = df.copy()
    for col in spend_cols:
        decay = decay_map.get(col, 0.3)
        out[f"{col}_adstocked"] = geometric_adstock(out[col].values, decay)
    return out


def normalize(series: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] for numerically stable regression fitting."""
    series = np.asarray(series, dtype=float)
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return np.zeros_like(series)
    return (series - lo) / (hi - lo)
