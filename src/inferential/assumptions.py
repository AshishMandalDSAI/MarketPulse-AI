"""Assumption checks (Phase 8 Part 3). Pure functions, no Streamlit.

Every check returns an ``AssumptionCheck`` that says what was found, why it matters,
how serious it is (``consequence``) and what to do instead. Nothing here runs a test.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from src.inferential.contracts import CONSEQUENCE_ORDER, AssumptionCheck

ROBUST_N = 30                 # t-tests/ANOVA are fairly robust to non-normality at about 30+ per group
SHAPIRO_MAX_N = 5000          # scipy's Shapiro-Wilk p-value is only reliable up to this size
CHI_MIN_EXPECTED = 5.0
CHI_HARD_MIN_EXPECTED = 1.0
CHI_MAX_SHARE_BELOW_5 = 0.20  # Cochran's rule: at least 80% of cells need an expected count of 5 or more


def worst(checks: Sequence[AssumptionCheck]) -> str:
    """Overall validity from the checks: valid / usable_with_caution / unreliable / invalid."""
    rank = max((CONSEQUENCE_ORDER.index(c.consequence) for c in checks), default=0)
    return {0: "valid", 1: "usable_with_caution", 2: "unreliable", 3: "invalid"}[rank]


def check_sample_size(n: int, minimum: int, label: str, recommended: Optional[int] = None) -> AssumptionCheck:
    why = "Very small samples give unstable estimates and tests with little power to detect real effects."
    if n < minimum:
        return AssumptionCheck(f"Sample size ({label})", "violated",
                               f"Only {n} valid observation(s); at least {minimum} are required.", why, "invalid",
                               "Collect or select more data.")
    if recommended and n < recommended:
        return AssumptionCheck(f"Sample size ({label})", "caution",
                               f"{n} valid observations; {recommended} or more is recommended.", why,
                               "usable_with_caution", "Treat the result as tentative.")
    return AssumptionCheck(f"Sample size ({label})", "met", f"{n} valid observations.", why)


def check_not_constant(x: np.ndarray, label: str, what: str = "this test") -> AssumptionCheck:
    why = "If every value is identical there is no variation to analyse, so the statistic is undefined (not zero)."
    if len(x) and np.ptp(x) == 0:
        return AssumptionCheck(f"Variation ({label})", "violated",
                               f"All {len(x)} values are identical ({x[0]:g}), so the standard deviation is 0.", why,
                               "invalid", f"Choose a column that varies; {what} cannot be calculated.")
    return AssumptionCheck(f"Variation ({label})", "met", "Values vary.", why)


def check_normality(x: np.ndarray, label: str, alpha: float = 0.05, robust_n: int = ROBUST_N,
                    what: str = "the test") -> AssumptionCheck:
    name = f"Normality ({label})"
    why = (f"{what} assumes the values (or the means) are roughly bell-shaped. With small samples a strong departure "
           "can make p-values wrong; with large samples the test is much more forgiving.")
    n = len(x)
    if n < 3:
        return AssumptionCheck(name, "not_checked", f"Normality cannot be tested with {n} value(s) (need 3).", why,
                               "usable_with_caution", "Interpret with great care.")
    if np.ptp(x) == 0:
        return AssumptionCheck(name, "not_checked", "All values are identical, so normality cannot be assessed.", why)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n <= SHAPIRO_MAX_N:
            stat, p = stats.shapiro(x)
            method = "Shapiro-Wilk"
        elif n >= 8:
            stat, p = stats.normaltest(x)
            method = "D'Agostino-Pearson"
        else:  # pragma: no cover - unreachable
            return AssumptionCheck(name, "not_checked", "Too few values.", why)
    if p >= alpha:
        return AssumptionCheck(name, "met", f"{method} p = {p:.4g} (no clear departure from normality).", why)
    if n >= robust_n:
        return AssumptionCheck(name, "violated",
                               f"{method} p = {p:.4g}: the data are not normal, but n = {n} is large enough "
                               "for the test to be fairly robust.", why, "usable_with_caution",
                               "Check the histogram; consider a rank-based test if the data are very skewed.")
    return AssumptionCheck(name, "violated",
                           f"{method} p = {p:.4g}: the data are not normal and n = {n} is small.", why, "unreliable",
                           "Use a rank-based (non-parametric) alternative such as Mann-Whitney U, Wilcoxon signed-rank "
                           "or Kruskal-Wallis (not available in this module), or collect more data.")


def check_equal_variances(groups: Sequence[np.ndarray], alpha: float = 0.05, required: bool = True) -> AssumptionCheck:
    name = "Equal variances (Levene / Brown-Forsythe)"
    why = ("The pooled-variance t-test and ANOVA assume groups are equally spread out; with unequal spreads and unequal "
           "group sizes their p-values can be badly wrong.")
    usable = [g for g in groups if len(g) >= 2]
    if len(usable) < 2 or all(np.ptp(g) == 0 for g in usable):
        return AssumptionCheck(name, "not_checked", "Not enough varying groups to compare spreads.", why)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = stats.levene(*usable, center="median")
    if p >= alpha:
        return AssumptionCheck(name, "met", f"Levene p = {p:.4g} (no clear difference in spread).", why)
    sizes = [len(g) for g in groups]
    unequal_n = max(sizes) / max(min(sizes), 1) > 1.5
    if not required:
        return AssumptionCheck(name, "caution", f"Levene p = {p:.4g}: spreads differ, but Welch's t-test does not "
                               "assume equal variances.", why, "none")
    return AssumptionCheck(name, "violated", f"Levene p = {p:.4g}: the groups have different spreads"
                           + (" and unequal sizes." if unequal_n else "."), why,
                           "unreliable" if unequal_n else "usable_with_caution",
                           "Use Welch's t-test (independent t-test) or a Welch ANOVA / Kruskal-Wallis alternative.")


def check_independence(values: Optional[np.ndarray] = None, time_ordered: bool = False, paired: bool = False) -> AssumptionCheck:
    name = "Independence of observations"
    why = ("Tests treat each observation as unrelated to the others. Repeated measurements, clusters or time series "
           "make p-values too small and intervals too narrow.")
    base = ("Independence cannot be verified from the data; it depends on how the data were collected. "
            + ("Pairs are assumed independent of each other (only the two values inside a pair are related). " if paired else ""))
    if not time_ordered:
        return AssumptionCheck(name, "not_checked", base.strip(), why, "none",
                               "Confirm each row is a separate, unrelated unit (customer, order, ...).")
    extra, consequence = "The rows are time-ordered, so neighbouring observations may be related. ", "usable_with_caution"
    if values is not None and len(values) >= 8 and np.ptp(values) > 0:
        v = np.asarray(values, dtype=float)
        r1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
        bound = 1.96 / np.sqrt(len(v))
        extra += f"Lag-1 autocorrelation in row order = {r1:.2f} (a clear problem if beyond ±{bound:.2f}). "
        if abs(r1) > bound:
            consequence = "unreliable"
    return AssumptionCheck(name, "caution", (base + extra).strip(), why, consequence,
                           "Aggregate to one value per period, use time-series methods, or treat the result as exploratory.")


# Lag-1 serial-dependence screen for categorical sequences ---------------------------------
SERIAL_MIN_N = 20             # fewest observations for which the transition table is even attempted


def serial_dependence_screen(values, alpha: float = 0.05) -> dict:
    """Screen a categorical sequence (in row order) for lag-1 serial dependence.

    Cross-tabulates each observation against the one before it and applies the ordinary
    chi-square test of independence to that transition table (the Anderson-Goodman /
    Bartlett test of a first-order dependence, df = (k-1)^2 for k categories). It is a
    SCREEN only: it looks at lag 1, has limited power, and is itself an approximation
    (overlapping pairs). A non-significant result never shows that observations are independent.

    Returns ``{"assessable": bool, "reason": str, "chi2", "p_value", "dof", "n_pairs", "dependent": bool}``.
    """
    x = np.asarray(list(values), dtype=object)
    n = len(x)
    out = {"assessable": False, "reason": "", "chi2": None, "p_value": None, "dof": None, "n_pairs": max(n - 1, 0),
           "dependent": False}
    if n < SERIAL_MIN_N:
        out["reason"] = f"only {n} observation(s); at least {SERIAL_MIN_N} are needed to screen for serial dependence"
        return out
    prev, nxt = x[:-1].astype(str), x[1:].astype(str)
    rows, row_idx = np.unique(prev, return_inverse=True)
    cols, col_idx = np.unique(nxt, return_inverse=True)
    if len(rows) < 2 or len(cols) < 2:
        out["reason"] = "the sequence has too little variation to form a transition table"
        return out
    table = np.zeros((len(rows), len(cols)), dtype=float)
    np.add.at(table, (row_idx, col_idx), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    if float(expected.min()) < CHI_HARD_MIN_EXPECTED or float((expected < CHI_MIN_EXPECTED).mean()) > CHI_MAX_SHARE_BELOW_5:
        out["reason"] = ("the transition table is too sparse (expected counts too small) for a reliable screen; "
                         "combine categories or use more data")
        return out
    out.update(assessable=True, chi2=float(chi2), p_value=float(p), dof=int(dof), dependent=bool(p < alpha))
    return out


def check_chi_square_independence(sequences: Optional[dict] = None, time_ordered: bool = False, alpha: float = 0.05,
                                  procedure: str = "chi-square test") -> AssumptionCheck:
    """Independence-of-observations check for the two chi-square procedures.

    ``sequences`` maps a column label to its categorical values in row order (missing rows removed).

    Policy
    * Not flagged as time-ordered: independence is reported as NOT CHECKED (it cannot be verified from data).
      It is never reported as met.
    * Flagged as time-ordered: a caution that the ordinary chi-square p-value assumes independent
      observations and is NOT validated by this test. Never refused outright, because categorical
      records can be (nearly) independent over time. A lag-1 screen is run per column (Bonferroni-adjusted
      alpha): if it detects serial dependence the result is downgraded to ``unreliable``; otherwise
      (not detected, or not assessable) it stays ``usable_with_caution`` and independence remains unestablished.
    """
    name = "Independence of observations"
    why = ("The chi-square p-value assumes every observation is an independent draw. Serial dependence - a category tending "
           "to follow itself or another category - makes the counts less informative than they look, so the test statistic is "
           "inflated, p-values are too small and 'significant' results appear too often.")
    base = ("Independence cannot be verified from the data; it depends on how the data were collected. The chi-square test "
            "assumes independence of observations; it does not establish it. ")
    if not time_ordered:
        return AssumptionCheck(name, "not_checked", base.strip(), why, "none",
                               "Confirm each row is a separate, unrelated unit (customer, order, ...).")
    detail = (base + f"The rows were declared time-ordered, so ordinary chi-square inference for the {procedure} may be invalid because neighbouring "
              "observations can be serially dependent. This result must not be read as showing that independence holds. ")
    rec = ("Aggregate to one count per period, compare periods with time-series or repeated-measures methods, or treat the "
           "result as exploratory.")
    seqs = sequences or {}
    a_each = alpha / max(len(seqs), 1)
    screens = {k: serial_dependence_screen(v, a_each) for k, v in seqs.items()}
    found = {k: s for k, s in screens.items() if s["assessable"] and s["dependent"]}
    parts = []
    for k, s in screens.items():
        if s["assessable"]:
            parts.append(f"'{k}': lag-1 chi-square = {s['chi2']:.4g}, df = {s['dof']}, p = {s['p_value']:.4g}")
        else:
            parts.append(f"'{k}': not assessable ({s['reason']})")
    screen_txt = ("Screening diagnostic (lag-1 transitions in row order, alpha = " + f"{a_each:.4g} per column): " + "; ".join(parts) + ". ") if parts else ""
    if found:
        return AssumptionCheck(name, "violated", (detail + screen_txt + "Serial dependence WAS detected in "
                               + ", ".join(f"'{k}'" for k in found) + ", so the ordinary p-value is not trustworthy.").strip(),
                               why, "unreliable", rec)
    tail = ("No lag-1 serial dependence was detected, but this screen has limited power and only looks at lag 1, so independence "
            "is still not established." if any(s["assessable"] for s in screens.values()) else
            "Serial dependence could not be screened, so independence is not established.")
    return AssumptionCheck(name, "caution", (detail + screen_txt + tail).strip(), why, "usable_with_caution", rec)


def check_expected_counts(expected: np.ndarray, two_by_two: bool = False) -> AssumptionCheck:
    name = "Expected cell counts (chi-square)"
    why = ("The chi-square p-value relies on a large-sample approximation. When expected counts are small the approximation "
           "fails and the p-value can be far from the truth.")
    e = np.asarray(expected, dtype=float)
    below5 = float((e < CHI_MIN_EXPECTED).mean())
    mn = float(e.min())
    alt = ("Combine sparse categories, collect more data" + (", or use Fisher's exact test for a 2x2 table (not available "
           "in this module)." if two_by_two else "."))
    if mn < CHI_HARD_MIN_EXPECTED or below5 > CHI_MAX_SHARE_BELOW_5:
        return AssumptionCheck(name, "violated",
                               f"Smallest expected count = {mn:.2f}; {below5 * 100:.0f}% of cells are below 5. "
                               "At least 80% of cells need an expected count of 5 or more and none may be below 1.",
                               why, "invalid", alt)
    if below5 > 0:
        return AssumptionCheck(name, "caution", f"{below5 * 100:.0f}% of cells have an expected count below 5 "
                               f"(smallest {mn:.2f}).", why, "usable_with_caution", alt)
    return AssumptionCheck(name, "met", f"All expected counts are 5 or more (smallest {mn:.2f}).", why)


def check_paired(n_a: int, n_b: int, n_pairs: int) -> AssumptionCheck:
    why = "A paired test compares two measurements of the same unit, so each unit needs both values."
    if n_pairs == 0:
        return AssumptionCheck("Paired observations", "violated", "No row has a value in both columns.", why, "invalid",
                               "Choose two columns that were measured on the same rows.")
    lost = max(n_a, n_b) - n_pairs
    detail = f"{n_pairs} complete pairs" + (f"; {lost} row(s) with only one of the two values were left out." if lost else ".")
    return AssumptionCheck("Paired observations", "met", detail, why)


def check_group_sizes(sizes: dict, minimum: int = 2, recommended: int = 5) -> AssumptionCheck:
    why = "Each group needs several observations to estimate its own mean and spread."
    too_small = [g for g, n in sizes.items() if n < minimum]
    if too_small:
        return AssumptionCheck("Group sizes", "violated", f"Group(s) with fewer than {minimum} observations: "
                               f"{', '.join(map(str, too_small))}.", why, "invalid", "Combine or drop tiny groups.")
    small = [g for g, n in sizes.items() if n < recommended]
    if small:
        return AssumptionCheck("Group sizes", "caution", f"Small group(s) (under {recommended}): {', '.join(map(str, small))}.",
                               why, "usable_with_caution", "Treat the result as tentative.")
    return AssumptionCheck("Group sizes", "met", "Every group has several observations.", why)
