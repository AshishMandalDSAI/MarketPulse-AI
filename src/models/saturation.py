"""
MarketPulse AI - Saturation (Diminishing Returns) Transformation
====================================================================
Beyond a certain spend level, additional marketing dollars produce
progressively smaller incremental revenue — audiences get saturated, ad
inventory gets exhausted, and frequency capping sets in. This is modeled
with a HILL (S-curve) SATURATION function, a standard, well-documented
transform in MMM literature (used in Google's Meridian, Meta's Robyn):

    saturation(x) = x^s / (k^s + x^s)

- k (half-saturation point): the spend level at which the channel reaches
  50% of its maximum effect. Larger k = channel needs more spend before
  it saturates.
- s (shape/slope parameter): controls how sharply the S-curve bends.
  s > 1 gives a classic S-shaped diminishing-returns curve.

We chose the Hill function over a simple log-transform because it (a) passes
through the origin (zero spend -> zero effect, avoiding the -infinity
problem of log(0)), and (b) has an interpretable half-saturation parameter
that maps directly to a business question managers ask ("how much do we
need to spend on Channel X before it starts flattening out?").
"""

import numpy as np
import pandas as pd

# Documented shape-parameter search grid used during model fitting.
SHAPE_GRID = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def hill_saturation(x: np.ndarray, k: float, s: float) -> np.ndarray:
    """Hill-function saturation transform.

    Parameters
    ----------
    x : array-like, adstocked spend (must be >= 0).
    k : half-saturation constant (spend level producing 50% of max effect).
    s : shape parameter controlling curve steepness.

    Returns
    -------
    np.ndarray in [0, 1) representing the saturated (diminishing-returns)
    response.
    """
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    k = max(k, 1e-6)
    return x ** s / (k ** s + x ** s)


def fit_half_saturation_point(x: np.ndarray, quantile: float = 0.5) -> float:
    """Heuristic: set k to a chosen quantile of the observed adstocked-spend
    distribution as a reasonable starting point for the half-saturation
    constant (later refined via grid search against holdout error).
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return 1.0
    return float(np.quantile(x, quantile))


def apply_saturation_to_dataframe(df: pd.DataFrame, adstocked_cols: list, k_map: dict, s_map: dict) -> pd.DataFrame:
    """Return a copy of df with `{col}_saturated` columns for each adstocked col."""
    out = df.copy()
    for col in adstocked_cols:
        k = k_map.get(col, fit_half_saturation_point(out[col].values))
        s = s_map.get(col, 1.2)
        out[f"{col}_saturated"] = hill_saturation(out[col].values, k, s)
    return out
