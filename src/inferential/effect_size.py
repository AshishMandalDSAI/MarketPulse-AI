"""Effect-size measures and their plain-English labels (Phase 8 Part 3).

Thresholds are Cohen's (1988) conventions. They are rules of thumb, not business
rules: a "small" effect can matter a lot (or not at all) depending on the decision.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.inferential.contracts import EffectSizeResult

LIMITATION_RULE_OF_THUMB = ("The labels (negligible / small / medium / large) are general rules of thumb. "
                            "Judge importance against costs, benefits and context, not the label alone.")


def _label(value: float, cuts: tuple) -> str:
    a = abs(value)
    if a < cuts[0]:
        return "negligible"
    if a < cuts[1]:
        return "small"
    if a < cuts[2]:
        return "medium"
    return "large"


def hedges_correction(df: float) -> float:
    """Small-sample bias correction factor J = 1 - 3 / (4*df - 1)."""
    return 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0.25 else float("nan")


def cohens_d_one_sample(mean: float, mu0: float, sd: float) -> Optional[float]:
    return (mean - mu0) / sd if sd and np.isfinite(sd) and sd > 0 else None


def cohens_d_two_sample(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """(mean_x - mean_y) / pooled sample sd."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return None
    sp2 = ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2)
    return float((np.mean(x) - np.mean(y)) / math.sqrt(sp2)) if sp2 > 0 else None


def eta_squared(groups) -> Optional[float]:
    """SS_between / SS_total."""
    allv = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand = allv.mean()
    sst = float(((allv - grand) ** 2).sum())
    if sst <= 0:
        return None
    ssb = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups))
    return ssb / sst


def omega_squared(groups) -> Optional[float]:
    allv = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    k, n = len(groups), len(allv)
    grand = allv.mean()
    sst = float(((allv - grand) ** 2).sum())
    ssb = float(sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups))
    msw = (sst - ssb) / (n - k) if n > k else float("nan")
    denom = sst + msw
    return (ssb - (k - 1) * msw) / denom if denom > 0 and np.isfinite(denom) else None


def cramers_v(chi2_uncorrected: float, n: int, n_rows: int, n_cols: int) -> Optional[float]:
    k = min(n_rows - 1, n_cols - 1)
    return math.sqrt(chi2_uncorrected / (n * k)) if n > 0 and k > 0 else None


def cohens_w(chi2: float, n: int) -> Optional[float]:
    return math.sqrt(chi2 / n) if n > 0 else None


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def describe_d(d: Optional[float], kind: str, extra: Optional[dict] = None, reason: str = "") -> EffectSizeResult:
    name = "Cohen's d" if kind != "paired" else "Cohen's d (paired, d_z)"
    if d is None:
        return EffectSizeResult(name=name, reason=reason or "The standard deviation is zero, so d is undefined.")
    lab = _label(d, (0.2, 0.5, 0.8))
    where = {"one": "the sample mean and the hypothesised value",
             "z": "the sample mean and the hypothesised value (in units of the known population SD)",
             "two": "the two group means (in pooled standard deviations)",
             "paired": "the two paired measurements (in standard deviations of the differences)"}[kind]
    practical = (f"The difference between {where} is about {abs(d):.2f} standard deviations ({lab}). "
                 f"A value near 0.2 is small, 0.5 medium and 0.8 large.")
    lim = ("d describes size, not importance, and does not show cause. " + LIMITATION_RULE_OF_THUMB)
    return EffectSizeResult(name=name, value=float(d), label=lab, practical=practical, limitations=lim,
                            extra=dict(extra or {}))


def describe_eta(eta2: Optional[float], omega2: Optional[float] = None, reason: str = "") -> EffectSizeResult:
    if eta2 is None:
        return EffectSizeResult(name="Eta-squared (η²)", reason=reason or "Total variation is zero.")
    lab = _label(eta2, (0.01, 0.06, 0.14))
    practical = (f"About {eta2 * 100:.1f}% of the variation in the outcome is associated with group membership ({lab}). "
                 "The rest is unexplained by the grouping.")
    lim = ("η² tends to over-state the effect in small samples (omega-squared, shown alongside, corrects for that). "
           "It says the groups differ, not why. " + LIMITATION_RULE_OF_THUMB)
    return EffectSizeResult(name="Eta-squared (η²)", value=float(eta2), label=lab, practical=practical, limitations=lim,
                            extra={"omega_squared": omega2})


def describe_association(name: str, v: Optional[float], dfstar: int, reason: str = "") -> EffectSizeResult:
    """Cramér's V (dfstar = min(rows-1, cols-1)) or Cohen's w (dfstar = 1)."""
    if v is None:
        return EffectSizeResult(name=name, reason=reason or "Not defined for this table.")
    scale = math.sqrt(max(dfstar, 1))
    lab = _label(v, (0.1 / scale, 0.3 / scale, 0.5 / scale))
    practical = (f"{name} = {v:.2f} ({lab}) on a 0-to-1 scale where 0 is no association and 1 is a perfect one.")
    lim = ("It measures the strength of association between categories, not its direction or cause. "
           + LIMITATION_RULE_OF_THUMB)
    return EffectSizeResult(name=name, value=float(v), label=lab, practical=practical, limitations=lim)


def describe_r(r: Optional[float], method: str, reason: str = "") -> EffectSizeResult:
    name = "Pearson r" if method == "pearson" else "Spearman rho"
    if r is None:
        return EffectSizeResult(name=name, reason=reason or "Correlation is undefined.")
    lab = _label(r, (0.1, 0.3, 0.5))
    direction = "positive (both tend to rise together)" if r > 0 else "negative (one tends to fall as the other rises)" if r < 0 else "none"
    share = (f" About {r * r * 100:.1f}% of the variation in one is shared in a straight-line way with the other."
             if method == "pearson" else
             f" The squared value ({r * r:.2f}) is the share of rank variation the two columns have in common.")
    practical = f"The association is {direction}, strength {abs(r):.2f} ({lab}).{share}"
    lim = ("Correlation does not establish causation. "
           + ("Pearson r only captures straight-line association and is sensitive to outliers. " if method == "pearson"
              else "Spearman rho captures any steadily rising or falling pattern, not only straight lines. ")
           + LIMITATION_RULE_OF_THUMB)
    return EffectSizeResult(name=name, value=float(r), label=lab, practical=practical, limitations=lim)
