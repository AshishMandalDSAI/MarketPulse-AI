"""The hypothesis tests (Phase 8 Part 3). Pure pandas / numpy / SciPy, no Streamlit.

Every public function takes pandas Series (aligned by position, i.e. columns of one
table), applies listwise deletion, checks assumptions BEFORE calculating, and returns a
``TestResult``. A test that cannot be run validly returns ``status == "refused"`` with the
reason and a recommended alternative; it never returns a number it cannot justify.
All statistics come from SciPy (``ttest_*``, ``f_oneway``, ``chisquare``,
``chi2_contingency``, ``pearsonr``, ``spearmanr``, ``norm``, ``t``); only closed-form
bookkeeping (means, sums of squares, Cohen's d, Fisher-z intervals) is written here.
"""

from __future__ import annotations

import math
import warnings
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.inferential import assumptions as A
from src.inferential import effect_size as E
from src.inferential import interpret as I
from src.inferential.contracts import (
    AssumptionCheck, ConfidenceInterval, EffectSizeResult, MissingInfo, TestResult,
)

ALTERNATIVES = ("two-sided", "greater", "less")
MAX_CATEGORIES = 50
MAX_CATEGORICAL_NUMERIC_LEVELS = 12
MAX_GROUPS = 30
Z_MIN_N_NO_NORMALITY = 30
MAX_PLOT_POINTS = 20_000

TEST_SPECS = {
    "z": "Z-test (one sample, known population SD)",
    "t1": "One-sample t-test",
    "t_ind": "Independent two-sample t-test",
    "t_paired": "Paired t-test",
    "anova": "One-way ANOVA",
    "chi_gof": "Chi-square goodness-of-fit test",
    "chi_ind": "Chi-square test of independence",
    "pearson": "Pearson correlation test",
    "spearman": "Spearman rank correlation test",
    "ci_mean": "Confidence interval for a mean",
}
TEST_ORDER = tuple(TEST_SPECS)


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def _check_params(alpha: float, level: float, alternative: str = "two-sided") -> None:
    if not (isinstance(alpha, (int, float)) and 0 < alpha < 1):
        raise ValueError("The significance level must be between 0 and 1 (exclusive).")
    if not (isinstance(level, (int, float)) and 0 < level < 1):
        raise ValueError("The confidence level must be between 0 and 1 (exclusive).")
    if alternative not in ALTERNATIVES:
        raise ValueError(f"The alternative must be one of {ALTERNATIVES}.")


def _as_series(values) -> pd.Series:
    return pd.Series(values).reset_index(drop=True)


def _as_numeric(values, label: str):
    """(float Series with NaN for missing/blank, None) or (None, reason)."""
    s = _as_series(values)
    if pd.api.types.is_bool_dtype(s):
        return None, f"'{label}' holds yes/no values, not numbers. Choose a numeric column."
    if pd.api.types.is_datetime64_any_dtype(s):
        return None, f"'{label}' holds dates, not numbers. Choose a numeric column."
    if pd.api.types.is_numeric_dtype(s):
        try:
            v = s.astype("float64")
        except (TypeError, ValueError):
            v = pd.to_numeric(s, errors="coerce").astype("float64")
    else:
        text = s.astype("object")
        blank = text.map(lambda x: isinstance(x, str) and x.strip() == "")
        text = text.mask(blank)
        v = pd.to_numeric(text, errors="coerce").astype("float64")
        bad = v.isna() & text.notna()
        if bad.any():
            example = str(text[bad].iloc[0])
            return None, (f"'{label}' contains {int(bad.sum()):,} value(s) that are not numbers (for example "
                          f"'{example}'). Choose a numeric column or clean the data first.")
    if np.isinf(v.to_numpy()).any():
        return None, f"'{label}' contains {int(np.isinf(v.to_numpy()).sum()):,} infinite value(s), which cannot be analysed."
    return v, None


def _label_of(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v)


def _as_categories(values, label: str):
    """(object Series of text labels with NaN for missing/blank, None) or (None, reason)."""
    s = _as_series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        return None, f"'{label}' holds dates; group dates into categories first."
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        finite = v.dropna()
        if len(finite) and (not np.isfinite(finite.to_numpy(dtype=float)).all()
                            or (np.mod(finite.to_numpy(dtype=float), 1) != 0).any()
                            or finite.nunique() > MAX_CATEGORICAL_NUMERIC_LEVELS):
            return None, (f"'{label}' is a numeric column with many or non-whole values, so it is not a category. "
                          "Choose a categorical or yes/no column.")
    obj = s.astype("object")
    blank = obj.map(lambda x: isinstance(x, str) and x.strip() == "")
    out = obj.mask(blank).map(lambda x: x if (x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NA)
                              else _label_of(x))
    out = out.where(~(obj.isna() | blank), np.nan)
    return out, None


def _missing_info(cols: dict, n_total: int) -> tuple:
    masks = {k: v.isna().to_numpy() for k, v in cols.items()}
    any_missing = reduce(np.logical_or, masks.values()) if masks else np.zeros(n_total, bool)
    removed = int(any_missing.sum())
    info = MissingInfo(original_n=int(n_total), removed_n=removed, final_n=int(n_total - removed),
                       missing_pct=(removed / n_total * 100.0) if n_total else 0.0,
                       per_variable={k: int(m.sum()) for k, m in masks.items()})
    return ~any_missing, info


def _new(key: str, alpha: float, level: float, alternative: str = "two-sided", **inputs) -> TestResult:
    return TestResult(test_key=key, test_name=TEST_SPECS[key], alternative=alternative, alpha=float(alpha),
                      inputs={"confidence_level": level, "significance_level": alpha, "alternative": alternative, **inputs},
                      independence_status="time_ordered_unverified" if inputs.get("time_ordered") else "not_verifiable")


def _refuse(res: TestResult, reason: str, recommendation: str = "") -> TestResult:
    res.status = "refused"
    res.refusal_reason = reason.strip()
    res.recommendation = recommendation.strip()
    res.validity = "not_run"
    res.statistic = res.p_value = res.significant = None
    res.interpretation = I.refusal(res)
    return res


def _gate(res: TestResult, checks: list) -> bool:
    """Record the checks; refuse the test if any is blocking. True means 'refused'."""
    res.assumptions = list(checks)
    blocking = [c for c in checks if c.consequence == "invalid"]
    if blocking:
        _refuse(res, " ".join(c.detail for c in blocking),
                " ".join(c.recommendation for c in blocking if c.recommendation))
        return True
    return False


def _independence_status(res: TestResult, checks: list) -> str:
    """Independence is an assumption, never a finding: the status only records what was declared and screened."""
    if not res.inputs.get("time_ordered"):
        return "not_verifiable"
    ind = next((c for c in checks if c.name == "Independence of observations"), None)
    return "serial_dependence_detected" if ind is not None and ind.consequence == "unreliable" else "time_ordered_unverified"


def _complete(res: TestResult, checks: list) -> TestResult:
    res.assumptions = list(checks)
    res.validity = A.worst(checks)
    res.independence_status = _independence_status(res, checks)
    res.interpretation = I.build(res)
    return res


def _prepare_numeric(res: TestResult, named: dict, n_total: int):
    """Coerce every named column; listwise delete. Returns (frame, ok)."""
    cols = {}
    for label, values in named.items():
        v, err = _as_numeric(values, label)
        if err:
            _refuse(res, err, "Choose a numeric column.")
            return None, False
        cols[label] = v
    keep, res.missing = _missing_info(cols, n_total)
    frame = pd.DataFrame(cols)[keep].reset_index(drop=True)
    return frame, True


def _ci(level: float, lo, hi, parameter: str, reason: str = "") -> ConfidenceInterval:
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return ConfidenceInterval(level=level, parameter=parameter, reason=reason or "The interval could not be calculated.")
    return ConfidenceInterval(level=level, lower=float(lo), upper=float(hi), parameter=parameter)


def _hyp(alt: str, left: str, right: str, h0op: str = "=") -> tuple:
    h0 = f"{left} {h0op} {right}"
    h1 = {"two-sided": f"{left} ≠ {right}", "greater": f"{left} > {right}", "less": f"{left} < {right}"}[alt]
    return h0, h1


def _n_check(n: int, minimum: int, label: str, recommended: Optional[int] = None) -> AssumptionCheck:
    return A.check_sample_size(n, minimum, label, recommended)


def _empty_refusal(res: TestResult, n_total: int) -> Optional[TestResult]:
    if n_total == 0:
        res.missing = MissingInfo()
        return _refuse(res, "The dataset has no rows, so there is nothing to test.", "Upload a dataset with data.")
    return None


# --------------------------------------------------------------------------
# One-sample procedures
# --------------------------------------------------------------------------

def z_test(values, mu0: float, sigma: Optional[float], *, alpha: float = 0.05, confidence_level: float = 0.95,
           alternative: str = "two-sided", column: str = "value", time_ordered: bool = False) -> TestResult:
    """One-sample Z-test. Only runs with a user-supplied population SD and (n >= 30 or normal data)."""
    _check_params(alpha, confidence_level, alternative)
    res = _new("z", alpha, confidence_level, alternative, column=column, hypothesised_mean=mu0,
               population_sd=sigma, time_ordered=time_ordered)
    res.statistic_name = "z"
    s = _as_series(values)
    if (r := _empty_refusal(res, len(s))) is not None:
        return r
    if not (isinstance(mu0, (int, float)) and np.isfinite(mu0)):
        return _refuse(res, "The hypothesised mean must be a finite number.")
    res.h0, res.h1 = _hyp(alternative, f"the population mean of '{column}' (μ)", f"{mu0:g}")
    frame, ok = _prepare_numeric(res, {column: s}, len(s))
    if not ok:
        return res
    x = frame[column].to_numpy(dtype=float)
    n = len(x)
    res.sample_sizes = {"n": n}
    # Rule 1: the population SD must be supplied by the user and be a positive finite number.
    if sigma is None:
        res.assumptions = [AssumptionCheck("Population standard deviation supplied", "violated",
                                           "No population standard deviation was supplied.",
                                           "The Z-test needs the true population SD as a known value; the sample's own SD "
                                           "is only an estimate and is not accepted as a substitute.", "invalid",
                                           "Use the one-sample t-test, which estimates the SD from the sample.")]
        return _refuse(res, "The Z-test needs the population standard deviation to be supplied by you, and none was "
                       "supplied. The sample standard deviation is not used as a substitute.",
                       "Use the one-sample t-test instead.")
    if not (isinstance(sigma, (int, float)) and np.isfinite(sigma) and sigma > 0):
        res.assumptions = [AssumptionCheck("Population standard deviation supplied", "violated",
                                           f"The supplied value ({sigma!r}) is not a positive number.", "", "invalid",
                                           "Enter a positive population standard deviation.")]
        return _refuse(res, "The supplied population standard deviation must be a positive number.",
                       "Enter a valid value, or use the one-sample t-test.")
    checks = [AssumptionCheck("Population standard deviation supplied", "met",
                              f"σ = {sigma:g} was supplied by the user and is used as the known population value.",
                              "The Z-test is only valid when σ is known, not estimated from this sample.")]
    checks.append(_n_check(n, 1, column))
    # Rule 2: n >= 30, or the sample must pass the normality check.
    if n >= Z_MIN_N_NO_NORMALITY:
        checks.append(AssumptionCheck(f"Sample size or normality ({column})", "met",
                                      f"n = {n} is at least {Z_MIN_N_NO_NORMALITY}, so the sample mean is approximately "
                                      "normal (central limit theorem).", "The Z-test needs a roughly normal sample mean."))
    else:
        nm = A.check_normality(x, column, alpha, what="The Z-test")
        if nm.status != "met":
            detail = (f"n = {n} is below {Z_MIN_N_NO_NORMALITY} and normality is not established ({nm.detail})")
            checks.append(AssumptionCheck(f"Sample size or normality ({column})", "violated", detail, nm.why_it_matters,
                                          "invalid", "Use the one-sample t-test if the data are roughly normal, or collect "
                                          f"at least {Z_MIN_N_NO_NORMALITY} observations."))
        else:
            checks.append(AssumptionCheck(f"Sample size or normality ({column})", "met",
                                          f"n = {n} is below {Z_MIN_N_NO_NORMALITY} but the data pass the normality check "
                                          f"({nm.detail})", nm.why_it_matters))
    checks.append(A.check_independence(x, time_ordered))
    if _gate(res, checks):
        return res
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n >= 2 else float("nan")
    if n >= 2 and np.isfinite(sd) and (sd / sigma > 2 or sd / sigma < 0.5):
        checks.append(AssumptionCheck("Supplied σ versus the sample's spread", "caution",
                                      f"The sample SD ({sd:.4g}) is far from the supplied σ ({sigma:g}).",
                                      "If the supplied σ is wrong, every Z-test result is wrong.", "usable_with_caution",
                                      "Double-check where σ came from."))
    se = sigma / math.sqrt(n)
    z = (mean - mu0) / se
    p = {"two-sided": 2 * stats.norm.sf(abs(z)), "greater": stats.norm.sf(z), "less": stats.norm.cdf(z)}[alternative]
    lo, hi = stats.norm.interval(confidence_level, loc=mean, scale=se)
    res.statistic, res.p_value = float(z), float(min(max(p, 0.0), 1.0))
    res.df = None
    res.undefined["degrees_of_freedom"] = "The Z-test uses the normal distribution, which has no degrees of freedom."
    res.ci = _ci(confidence_level, lo, hi, f"the population mean of '{column}'")
    res.effect_size = E.describe_d(E.cohens_d_one_sample(mean, mu0, sigma), "z")
    res.details["Sample summary"] = pd.DataFrame({"Measure": ["n", "Sample mean", "Known population SD (σ)", "Standard error σ/√n",
                                                              "Sample SD (for reference only)"],
                                                  "Value": [n, mean, sigma, se, sd]})
    res.plot_data = {"kind": "hist", "x": x, "ref": mu0, "ref_label": "hypothesised mean"}
    return _complete(res, checks)


def one_sample_t(values, mu0: float, *, alpha: float = 0.05, confidence_level: float = 0.95,
                 alternative: str = "two-sided", column: str = "value", time_ordered: bool = False) -> TestResult:
    _check_params(alpha, confidence_level, alternative)
    res = _new("t1", alpha, confidence_level, alternative, column=column, hypothesised_mean=mu0, time_ordered=time_ordered)
    res.statistic_name = "t"
    s = _as_series(values)
    if (r := _empty_refusal(res, len(s))) is not None:
        return r
    if not (isinstance(mu0, (int, float)) and np.isfinite(mu0)):
        return _refuse(res, "The hypothesised mean must be a finite number.")
    res.h0, res.h1 = _hyp(alternative, f"the population mean of '{column}' (μ)", f"{mu0:g}")
    frame, ok = _prepare_numeric(res, {column: s}, len(s))
    if not ok:
        return res
    x = frame[column].to_numpy(dtype=float)
    n = len(x)
    res.sample_sizes = {"n": n}
    checks = [_n_check(n, 2, column, 10), A.check_not_constant(x, column, "the t-test")]
    if _gate(res, checks):
        return res
    checks.append(A.check_normality(x, column, alpha, what="The t-test"))
    checks.append(A.check_independence(x, time_ordered))
    out = stats.ttest_1samp(x, mu0, alternative=alternative)
    ci2 = stats.ttest_1samp(x, mu0).confidence_interval(confidence_level)
    mean, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    res.statistic, res.p_value, res.df = float(out.statistic), float(out.pvalue), float(n - 1)
    res.ci = _ci(confidence_level, ci2.low, ci2.high, f"the population mean of '{column}'")
    d = E.cohens_d_one_sample(mean, mu0, sd)
    res.effect_size = E.describe_d(d, "one", {"hedges_g": d * E.hedges_correction(n - 1) if d is not None else None})
    res.details["Sample summary"] = pd.DataFrame({"Measure": ["n", "Sample mean", "Sample SD", "Standard error"],
                                                  "Value": [n, mean, sd, sd / math.sqrt(n)]})
    res.plot_data = {"kind": "hist", "x": x, "ref": mu0, "ref_label": "hypothesised mean"}
    return _complete(res, checks)


def mean_confidence_interval(values, *, confidence_level: float = 0.95, column: str = "value",
                             alpha: float = 0.05, time_ordered: bool = False) -> TestResult:
    """Standalone t-based confidence interval for a mean (an estimate, not a hypothesis test)."""
    _check_params(alpha, confidence_level)
    res = _new("ci_mean", alpha, confidence_level, "two-sided", column=column, time_ordered=time_ordered)
    res.statistic_name = "mean"
    res.h0 = res.h1 = ""
    res.undefined.update({"p_value": "This is an interval estimate, not a hypothesis test, so there is no p-value.",
                          "effect_size": "Not applicable to an interval estimate."})
    s = _as_series(values)
    if (r := _empty_refusal(res, len(s))) is not None:
        return r
    frame, ok = _prepare_numeric(res, {column: s}, len(s))
    if not ok:
        return res
    x = frame[column].to_numpy(dtype=float)
    n = len(x)
    res.sample_sizes = {"n": n}
    checks = [_n_check(n, 2, column, 10), A.check_not_constant(x, column, "a confidence interval")]
    if _gate(res, checks):
        return res
    checks.append(A.check_normality(x, column, alpha, what="The interval"))
    checks.append(A.check_independence(x, time_ordered))
    mean, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    sem = sd / math.sqrt(n)
    lo, hi = stats.t.interval(confidence_level, n - 1, loc=mean, scale=sem)
    res.statistic, res.df = mean, float(n - 1)
    res.ci = _ci(confidence_level, lo, hi, f"the population mean of '{column}'")
    res.details["Sample summary"] = pd.DataFrame({"Measure": ["n", "Sample mean", "Sample SD", "Standard error"],
                                                  "Value": [n, mean, sd, sem]})
    res.plot_data = {"kind": "hist", "x": x, "ref": mean, "ref_label": "sample mean"}
    return _complete(res, checks)


# --------------------------------------------------------------------------
# Two-sample procedures
# --------------------------------------------------------------------------

def _group_split(res: TestResult, values, groups, value_label: str, group_label: str):
    """Coerce both columns, delete incomplete rows, return (values array, labels array) or None if refused."""
    v, err = _as_numeric(values, value_label)
    if err:
        _refuse(res, err, "Choose a numeric column for the values.")
        return None
    g, err = _as_categories(groups, group_label)
    if err:
        _refuse(res, err, "Choose a categorical or yes/no column for the groups.")
        return None
    if len(v) != len(g):
        _refuse(res, "The value and group columns have different lengths.")
        return None
    keep, res.missing = _missing_info({value_label: v, group_label: g}, len(v))
    return v[keep].to_numpy(dtype=float), g[keep].to_numpy(dtype=object)


def independent_t(values, groups, level_a, level_b, *, equal_var: bool = False, alpha: float = 0.05,
                  confidence_level: float = 0.95, alternative: str = "two-sided", value_label: str = "value",
                  group_label: str = "group", time_ordered: bool = False) -> TestResult:
    """Independent two-sample t-test. Welch (unequal variances) by default; equal_var=True gives Student's pooled test."""
    _check_params(alpha, confidence_level, alternative)
    a_name, b_name = _label_of(level_a), _label_of(level_b)
    res = _new("t_ind", alpha, confidence_level, alternative, value_column=value_label, group_column=group_label,
               group_a=a_name, group_b=b_name, variant="Student (pooled variance)" if equal_var else "Welch (unequal variances)",
               time_ordered=time_ordered)
    res.statistic_name = "t"
    s = _as_series(values)
    if (r := _empty_refusal(res, len(s))) is not None:
        return r
    res.h0, res.h1 = _hyp(alternative, f"mean of '{value_label}' for {group_label} = {a_name} (μ_A)",
                          f"mean for {group_label} = {b_name} (μ_B)")
    if a_name == b_name:
        return _refuse(res, "The two groups to compare must be different.", "Choose two different groups.")
    out = _group_split(res, s, groups, value_label, group_label)
    if out is None:
        return res
    v, g = out
    xa, xb = v[g == a_name], v[g == b_name]
    res.missing.not_used_n = int(len(v) - len(xa) - len(xb))
    res.missing.final_n = int(len(xa) + len(xb))
    if res.missing.not_used_n:
        res.missing.note = f"{res.missing.not_used_n:,} complete row(s) belong to other groups and were not used."
    res.sample_sizes = {a_name: len(xa), b_name: len(xb)}
    checks = [A.check_group_sizes(res.sample_sizes, 2, 10)]
    for nm, arr in ((a_name, xa), (b_name, xb)):
        if len(arr) == 0:
            checks.append(AssumptionCheck(f"Group present ({nm})", "violated", f"No usable observations for group '{nm}'.",
                                          "Both groups need data.", "invalid", "Choose groups that appear in the data."))
    if len(xa) >= 2 and len(xb) >= 2 and np.ptp(xa) == 0 and np.ptp(xb) == 0:
        checks.append(AssumptionCheck("Variation (both groups)", "violated",
                                      "Within both groups every value is identical, so the standard errors are 0.",
                                      "With no spread the t statistic is undefined.", "invalid",
                                      "Choose a column that varies within groups."))
    if _gate(res, checks):
        return res
    checks.append(A.check_normality(xa, f"{a_name}", alpha, what="The t-test"))
    checks.append(A.check_normality(xb, f"{b_name}", alpha, what="The t-test"))
    checks.append(A.check_equal_variances([xa, xb], alpha, required=equal_var))
    checks.append(A.check_independence(v, time_ordered))
    out_t = stats.ttest_ind(xa, xb, equal_var=equal_var, alternative=alternative)
    ci2 = stats.ttest_ind(xa, xb, equal_var=equal_var).confidence_interval(confidence_level)
    res.statistic, res.p_value = float(out_t.statistic), float(out_t.pvalue)
    res.df = float(out_t.df)
    res.ci = _ci(confidence_level, ci2.low, ci2.high, f"the difference in means ({a_name} − {b_name})")
    d = E.cohens_d_two_sample(xa, xb)
    res.effect_size = E.describe_d(d, "two", {"hedges_g": d * E.hedges_correction(len(xa) + len(xb) - 2) if d is not None else None})
    res.details["Group summary"] = pd.DataFrame({
        "Group": [a_name, b_name], "n": [len(xa), len(xb)], "Mean": [xa.mean(), xb.mean()],
        "SD": [np.std(xa, ddof=1), np.std(xb, ddof=1)]})
    res.plot_data = {"kind": "groups", "groups": {a_name: xa, b_name: xb}}
    return _complete(res, checks)


def paired_t(values_a, values_b, *, alpha: float = 0.05, confidence_level: float = 0.95,
             alternative: str = "two-sided", label_a: str = "A", label_b: str = "B",
             time_ordered: bool = False) -> TestResult:
    _check_params(alpha, confidence_level, alternative)
    res = _new("t_paired", alpha, confidence_level, alternative, column_a=label_a, column_b=label_b, time_ordered=time_ordered)
    res.statistic_name = "t"
    a, b = _as_series(values_a), _as_series(values_b)
    if (r := _empty_refusal(res, len(a))) is not None:
        return r
    res.h0, res.h1 = _hyp(alternative, f"the mean difference ('{label_a}' − '{label_b}')", "0")
    if len(a) != len(b):
        return _refuse(res, "The two columns have different numbers of rows, so observations cannot be paired.",
                       "Choose two columns from the same table.")
    if label_a == label_b:
        return _refuse(res, "A paired test needs two different columns.", "Choose two different columns.")
    frame, ok = _prepare_numeric(res, {label_a: a, label_b: b}, len(a))
    if not ok:
        return res
    xa, xb = frame[label_a].to_numpy(float), frame[label_b].to_numpy(float)
    n = len(xa)
    res.sample_sizes = {"pairs": n}
    diff = xa - xb
    checks = [A.check_paired(int(a.notna().sum()), int(b.notna().sum()), n), _n_check(n, 2, "pairs", 10)]
    if n >= 2:
        checks.append(A.check_not_constant(diff, "differences", "the paired t-test"))
    if _gate(res, checks):
        return res
    checks.append(A.check_normality(diff, "differences", alpha, what="The paired t-test"))
    checks.append(A.check_independence(diff, time_ordered, paired=True))
    out = stats.ttest_rel(xa, xb, alternative=alternative)
    ci2 = stats.ttest_rel(xa, xb).confidence_interval(confidence_level)
    md, sdd = float(diff.mean()), float(diff.std(ddof=1))
    res.statistic, res.p_value, res.df = float(out.statistic), float(out.pvalue), float(n - 1)
    res.ci = _ci(confidence_level, ci2.low, ci2.high, f"the mean difference ('{label_a}' − '{label_b}')")
    dz = md / sdd
    res.effect_size = E.describe_d(dz, "paired", {"hedges_g": dz * E.hedges_correction(n - 1)})
    res.details["Difference summary"] = pd.DataFrame({"Measure": ["Pairs", f"Mean of {label_a}", f"Mean of {label_b}",
                                                                  "Mean difference", "SD of differences"],
                                                      "Value": [n, xa.mean(), xb.mean(), md, sdd]})
    res.plot_data = {"kind": "hist", "x": diff, "ref": 0.0, "ref_label": "no difference"}
    return _complete(res, checks)


# --------------------------------------------------------------------------
# ANOVA
# --------------------------------------------------------------------------

def _normality_each(groups: dict, alpha: float) -> AssumptionCheck:
    per = {k: A.check_normality(v, k, alpha, what="ANOVA") for k, v in groups.items()}
    failing = {k: c for k, c in per.items() if c.status == "violated"}
    unchecked = [k for k, c in per.items() if c.status == "not_checked"]
    why = next(iter(per.values())).why_it_matters if per else ""
    if failing:
        rank = max(A.CONSEQUENCE_ORDER.index(c.consequence) for c in failing.values())
        return AssumptionCheck("Normality (each group)", "violated",
                               "Not normal: " + "; ".join(f"{k} ({c.detail})" for k, c in failing.items()), why,
                               A.CONSEQUENCE_ORDER[rank], next(iter(failing.values())).recommendation)
    if unchecked:
        return AssumptionCheck("Normality (each group)", "not_checked",
                               f"Could not be assessed for: {', '.join(unchecked)} (too few or identical values).",
                               why, "usable_with_caution", "Interpret with care.")
    return AssumptionCheck("Normality (each group)", "met", "No clear departure from normality in any group.", why)


def one_way_anova(values, groups, *, alpha: float = 0.05, confidence_level: float = 0.95,
                  value_label: str = "value", group_label: str = "group", time_ordered: bool = False) -> TestResult:
    _check_params(alpha, confidence_level)
    res = _new("anova", alpha, confidence_level, "two-sided", value_column=value_label, group_column=group_label,
               time_ordered=time_ordered)
    res.statistic_name = "F"
    s = _as_series(values)
    if (r := _empty_refusal(res, len(s))) is not None:
        return r
    res.h0 = f"The mean of '{value_label}' is the same in every '{group_label}' group (μ_1 = μ_2 = … = μ_k)."
    res.h1 = "At least one group mean is different."
    out = _group_split(res, s, groups, value_label, group_label)
    if out is None:
        return res
    v, g = out
    levels = sorted(set(g.tolist()))
    if len(levels) > MAX_GROUPS:
        return _refuse(res, f"'{group_label}' has {len(levels)} groups; at most {MAX_GROUPS} are supported.",
                       "Choose a column with fewer categories.")
    sizes_all = {lv: int((g == lv).sum()) for lv in levels}
    tiny = [lv for lv, n in sizes_all.items() if n < 2]
    keep_levels = [lv for lv in levels if sizes_all[lv] >= 2]
    res.missing.not_used_n = int(sum(sizes_all[lv] for lv in tiny))
    res.missing.final_n = int(sum(sizes_all[lv] for lv in keep_levels))
    if tiny:
        res.missing.note = (f"Group(s) with a single observation were left out: {', '.join(tiny)} "
                            f"({res.missing.not_used_n:,} row(s)).")
        res.warnings.append(res.missing.note)
    res.sample_sizes = {lv: sizes_all[lv] for lv in keep_levels}
    checks = []
    if len(keep_levels) < 2:
        checks.append(AssumptionCheck("Number of groups", "violated",
                                      f"Only {len(keep_levels)} group(s) have 2 or more observations; ANOVA needs at least 2.",
                                      "ANOVA compares group means.", "invalid", "Choose a column that splits the data into "
                                      "at least two groups with several rows each."))
        _gate(res, checks)
        return res
    arrays = {lv: v[g == lv] for lv in keep_levels}
    allv = np.concatenate(list(arrays.values()))
    checks.append(A.check_group_sizes(res.sample_sizes, 2, 5))
    checks.append(A.check_not_constant(allv, value_label, "ANOVA"))
    if all(np.ptp(a) == 0 for a in arrays.values()) and np.ptp(allv) > 0:
        checks.append(AssumptionCheck("Variation within groups", "violated",
                                      "Every value inside each group is identical, so there is no within-group variation.",
                                      "F divides by the within-group variation, which would be 0.", "invalid",
                                      "Choose a column with variation inside the groups."))
    if _gate(res, checks):
        return res
    checks.append(_normality_each(arrays, alpha))
    checks.append(A.check_equal_variances(list(arrays.values()), alpha, required=True))
    checks.append(A.check_independence(v, time_ordered))
    k, N = len(arrays), len(allv)
    out_f = stats.f_oneway(*arrays.values())
    res.statistic, res.p_value = float(out_f.statistic), float(out_f.pvalue)
    res.df, res.df2 = float(k - 1), float(N - k)
    eta = E.eta_squared(list(arrays.values()))
    res.effect_size = E.describe_eta(eta, E.omega_squared(list(arrays.values())))
    res.undefined["confidence_interval"] = ("The overall F-test compares several means at once, so it has no single "
                                            "interval; the group table shows an interval for each group mean.")
    res.ci = ConfidenceInterval(level=confidence_level, parameter="overall test", reason=res.undefined["confidence_interval"])
    grand = allv.mean()
    ssb = float(sum(len(a) * (a.mean() - grand) ** 2 for a in arrays.values()))
    ssw = float(sum(((a - a.mean()) ** 2).sum() for a in arrays.values()))
    rows = []
    for lv, a in arrays.items():
        sem = a.std(ddof=1) / math.sqrt(len(a))
        lo, hi = (stats.t.interval(confidence_level, len(a) - 1, loc=a.mean(), scale=sem) if sem > 0 else (np.nan, np.nan))
        rows.append({"Group": lv, "n": len(a), "Mean": a.mean(), "SD": a.std(ddof=1),
                     f"{confidence_level * 100:.0f}% CI lower": lo, f"{confidence_level * 100:.0f}% CI upper": hi})
    res.details["Group summary"] = pd.DataFrame(rows)
    res.details["ANOVA table"] = pd.DataFrame({
        "Source": ["Between groups", "Within groups", "Total"], "Sum of squares": [ssb, ssw, ssb + ssw],
        "df": [k - 1, N - k, N - 1], "Mean square": [ssb / (k - 1), ssw / (N - k), np.nan],
        "F": [res.statistic, np.nan, np.nan], "p-value": [res.p_value, np.nan, np.nan]})
    res.plot_data = {"kind": "groups", "groups": arrays}
    return _complete(res, checks)


# --------------------------------------------------------------------------
# Chi-square
# --------------------------------------------------------------------------

CHI_TIME_ORDER_WARNING = ("Time-ordered data: the chi-square test assumes independent observations and does NOT establish "
                          "independence. Serial dependence can invalidate ordinary chi-square inference (p-values too small, "
                          "false 'significant' results). Treat this result as exploratory unless independence is justified "
                          "by how the data were collected.")


def chi_square_gof(values, *, expected_proportions: Optional[dict] = None, alpha: float = 0.05,
                   column: str = "category", time_ordered: bool = False) -> TestResult:
    """Do the category counts follow expected proportions (equal by default)?

    ``time_ordered=True`` declares that the rows are in time order. The test is still run (time-ordered
    categorical data is not rejected outright), but the result carries an explicit independence warning,
    is never reported as having met the independence assumption, and is downgraded to ``unreliable`` when a
    lag-1 serial-dependence screen detects dependence.
    """
    _check_params(alpha, 0.95)
    res = _new("chi_gof", alpha, 0.95, "two-sided", column=column,
               expected="equal frequencies" if not expected_proportions else "user-supplied proportions",
               time_ordered=bool(time_ordered))
    res.statistic_name = "chi-square"
    s = _as_series(values)
    if (r := _empty_refusal(res, len(s))) is not None:
        return r
    res.h0 = f"The categories of '{column}' occur with the expected proportions."
    res.h1 = "At least one category occurs more or less often than expected."
    cat, err = _as_categories(s, column)
    if err:
        return _refuse(res, err, "Choose a categorical or yes/no column.")
    keep, res.missing = _missing_info({column: cat}, len(cat))
    cat = cat[keep]
    counts = cat.value_counts()
    levels = sorted(counts.index.tolist())
    n = int(counts.sum())
    res.sample_sizes = {"n": n, "categories": len(levels)}
    checks = []
    if len(levels) < 2:
        checks.append(AssumptionCheck("Number of categories", "violated", f"Only {len(levels)} category in '{column}'.",
                                      "A goodness-of-fit test compares at least two categories.", "invalid",
                                      "Choose a column with at least two categories."))
    if len(levels) > MAX_CATEGORIES:
        checks.append(AssumptionCheck("Number of categories", "violated", f"'{column}' has {len(levels)} categories; at most "
                                      f"{MAX_CATEGORIES} are supported.", "Too many sparse categories make the test meaningless.",
                                      "invalid", "Combine categories or pick another column."))
    if _gate(res, checks):
        return res
    if expected_proportions:
        try:
            props = np.array([float(expected_proportions[lv]) for lv in levels], dtype=float)
        except (KeyError, TypeError, ValueError):
            return _refuse(res, "A proportion is needed for every category (" + ", ".join(levels) + ").",
                           "Supply one number for every category, or use equal frequencies.")
        extra = set(expected_proportions) - set(levels)
        if extra:
            return _refuse(res, f"Proportions were given for categories that are not in the data: {', '.join(map(str, extra))}.")
        if (props <= 0).any() or not np.isfinite(props).all() or abs(props.sum() - 1.0) > 1e-6:
            return _refuse(res, f"The expected proportions must be positive and add up to 1 (they add up to {props.sum():g}).",
                           "Correct the proportions.")
    else:
        props = np.full(len(levels), 1.0 / len(levels))
    observed = np.array([counts[lv] for lv in levels], dtype=float)
    expected = props * n
    checks += [_n_check(n, 2, column), A.check_expected_counts(expected, two_by_two=False),
               A.check_chi_square_independence({column: cat.tolist()}, time_ordered, alpha, "chi-square goodness-of-fit test")]
    if time_ordered:
        res.warnings.append(CHI_TIME_ORDER_WARNING)
    res.independence_status = _independence_status(res, checks)
    if _gate(res, checks):
        res.details["Observed vs expected"] = pd.DataFrame({"Category": levels, "Observed": observed, "Expected": expected})
        return res
    out = stats.chisquare(observed, f_exp=expected)
    res.statistic, res.p_value, res.df = float(out.statistic), float(out.pvalue), float(len(levels) - 1)
    w = E.cohens_w(res.statistic, n)
    res.effect_size = E.describe_association("Cohen's w", w, 1)
    res.undefined["confidence_interval"] = "A goodness-of-fit test compares a whole distribution, so it has no single interval."
    res.ci = ConfidenceInterval(level=0.95, parameter="overall test", reason=res.undefined["confidence_interval"])
    res.details["Observed vs expected"] = pd.DataFrame({
        "Category": levels, "Observed": observed, "Expected proportion": props, "Expected": expected,
        "Standardised residual": (observed - expected) / np.sqrt(expected)})
    res.plot_data = {"kind": "bars", "labels": levels, "observed": observed, "expected": expected}
    return _complete(res, checks)


def chi_square_independence(values_a, values_b, *, alpha: float = 0.05, label_a: str = "A", label_b: str = "B",
                            time_ordered: bool = False) -> TestResult:
    """Chi-square test of independence between two categorical columns.

    ``time_ordered=True`` follows the same policy as :func:`chi_square_gof`: run, warn explicitly, never claim
    that independence of observations holds, and downgrade to ``unreliable`` if serial dependence is detected.
    (The null hypothesis is independence of the two VARIABLES; the test's own assumption is independence of the
    OBSERVATIONS, which is a different thing and is not established by the test.)
    """
    _check_params(alpha, 0.95)
    res = _new("chi_ind", alpha, 0.95, "two-sided", column_a=label_a, column_b=label_b, time_ordered=bool(time_ordered))
    res.statistic_name = "chi-square"
    a, b = _as_series(values_a), _as_series(values_b)
    if (r := _empty_refusal(res, len(a))) is not None:
        return r
    res.h0 = f"'{label_a}' and '{label_b}' are independent (no association)."
    res.h1 = f"'{label_a}' and '{label_b}' are associated."
    if len(a) != len(b):
        return _refuse(res, "The two columns have different lengths.")
    if label_a == label_b:
        return _refuse(res, "Choose two different columns.", "Choose two different columns.")
    ca, err = _as_categories(a, label_a)
    if err:
        return _refuse(res, err, "Choose categorical or yes/no columns.")
    cb, err = _as_categories(b, label_b)
    if err:
        return _refuse(res, err, "Choose categorical or yes/no columns.")
    keep, res.missing = _missing_info({label_a: ca, label_b: cb}, len(ca))
    table = pd.crosstab(ca[keep].to_numpy(), cb[keep].to_numpy())
    table.index.name, table.columns.name = label_a, label_b
    n = int(table.to_numpy().sum())
    r_, c_ = table.shape
    res.sample_sizes = {"n": n, "rows": r_, "columns": c_}
    checks = []
    if r_ < 2 or c_ < 2:
        checks.append(AssumptionCheck("Number of categories", "violated",
                                      f"After removing missing values the table is {r_} x {c_}; each column needs at least 2 categories.",
                                      "Independence cannot be tested when a variable never varies.", "invalid",
                                      "Choose columns that each have at least two categories."))
    if max(r_, c_) > MAX_CATEGORIES:
        checks.append(AssumptionCheck("Number of categories", "violated", f"The table is {r_} x {c_}; at most {MAX_CATEGORIES} "
                                      "categories per column are supported.", "Sparse tables make the test unreliable.", "invalid",
                                      "Combine categories or choose other columns."))
    if _gate(res, checks):
        return res
    checks.append(_n_check(n, 2, "rows"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2c, pc, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=True)
        chi2u = stats.chi2_contingency(table.to_numpy(), correction=False)[0]
    checks.append(A.check_expected_counts(expected, two_by_two=(r_ == 2 and c_ == 2)))
    checks.append(A.check_chi_square_independence({label_a: ca[keep].tolist(), label_b: cb[keep].tolist()}, time_ordered,
                                                  alpha, "chi-square test of independence"))
    if time_ordered:
        res.warnings.append(CHI_TIME_ORDER_WARNING)
    res.independence_status = _independence_status(res, checks)
    res.details["Observed counts"] = table.reset_index()
    res.details["Expected counts"] = pd.DataFrame(expected, index=table.index, columns=table.columns).round(4).reset_index()
    if _gate(res, checks):
        return res
    res.statistic, res.p_value, res.df = float(chi2c), float(pc), float(dof)
    v = E.cramers_v(float(chi2u), n, r_, c_)
    res.effect_size = E.describe_association("Cramér's V", v, min(r_ - 1, c_ - 1))
    if r_ == 2 and c_ == 2:
        res.warnings.append("For a 2 x 2 table the reported chi-square uses Yates' continuity correction; Cramér's V "
                            "is computed from the uncorrected chi-square.")
    res.undefined["confidence_interval"] = "No confidence interval is defined for the chi-square test of independence."
    res.ci = ConfidenceInterval(level=0.95, parameter="overall test", reason=res.undefined["confidence_interval"])
    res.plot_data = {"kind": "heat", "table": table}
    return _complete(res, checks)


# --------------------------------------------------------------------------
# Correlation tests
# --------------------------------------------------------------------------

SPEARMAN_DF_NOTE = ("Approximate. Spearman's rho has no exact degrees of freedom. SciPy's p-value refers rho to a Student t "
                    "distribution via t = rho*sqrt((n-2)/(1-rho^2)); 'n - 2' is the df of that approximating t distribution "
                    "(borrowed from Pearson's r), not an exact property of rank correlation. The approximation is poorest for "
                    "small n (under about 10) and with many ties.")
SPEARMAN_DF_WARNING = ("Spearman degrees of freedom (n - 2) are APPROXIMATE: they belong to the Student-t approximation SciPy uses "
                       "for the p-value, not to an exact distribution of Spearman's rho, so the p-value is approximate too.")


def _correlation(method: str, x_vals, y_vals, alpha, level, alternative, x_label, y_label, time_ordered) -> TestResult:
    _check_params(alpha, level, alternative)
    key = "pearson" if method == "pearson" else "spearman"
    res = _new(key, alpha, level, alternative, column_x=x_label, column_y=y_label, time_ordered=time_ordered)
    sym = "ρ" if method == "pearson" else "ρ_s"
    res.statistic_name = "r" if method == "pearson" else "rho"
    x, y = _as_series(x_vals), _as_series(y_vals)
    if (r := _empty_refusal(res, len(x))) is not None:
        return r
    kind = "linear" if method == "pearson" else "monotonic (rank-based)"
    left = f"the {kind} correlation between '{x_label}' and '{y_label}' ({sym})"
    res.h0, res.h1 = _hyp(alternative, left, "0")
    if len(x) != len(y):
        return _refuse(res, "The two columns have different numbers of rows.")
    if x_label == y_label:
        return _refuse(res, "A column is perfectly correlated with itself; choose two different columns.")
    frame, ok = _prepare_numeric(res, {x_label: x, y_label: y}, len(x))
    if not ok:
        return res
    xa, ya = frame[x_label].to_numpy(float), frame[y_label].to_numpy(float)
    n = len(xa)
    res.sample_sizes = {"pairs": n}
    checks = [_n_check(n, 3, "pairs", 10)]
    if n >= 1:
        checks.append(A.check_not_constant(xa, x_label, "a correlation"))
        checks.append(A.check_not_constant(ya, y_label, "a correlation"))
    if _gate(res, checks):
        return res
    if method == "pearson":
        checks.append(A.check_normality(xa, x_label, alpha, what="The Pearson test"))
        checks.append(A.check_normality(ya, y_label, alpha, what="The Pearson test"))
    checks.append(A.check_independence(xa, time_ordered))
    if method == "pearson":
        out = stats.pearsonr(xa, ya, alternative=alternative)
        ci2 = stats.pearsonr(xa, ya).confidence_interval(level)
        r_val, p_val = float(out.statistic), float(out.pvalue)
        lo, hi, why = ci2.low, ci2.high, "The interval could not be calculated (perfect correlation or too few pairs)."
    else:
        out = stats.spearmanr(xa, ya, alternative=alternative)
        r_val, p_val = float(out.statistic), float(out.pvalue)
        lo = hi = None
        why = ""
        if n < 10:
            why = "An interval for Spearman's rho needs at least 10 pairs."
        elif abs(r_val) >= 1:
            why = "The correlation is perfect (±1), so the interval has no width."
        else:
            zc = stats.norm.ppf(1 - (1 - level) / 2)
            se = math.sqrt((1 + r_val ** 2 / 2) / (n - 3))
            lo, hi = math.tanh(math.atanh(r_val) - zc * se), math.tanh(math.atanh(r_val) + zc * se)
            res.warnings.append("The Spearman confidence interval is an approximation (Fisher z with the Bonett-Wright "
                                "standard error); SciPy has no exact version.")
    res.statistic, res.p_value, res.df = r_val, p_val, float(n - 2)
    if method == "spearman":
        res.df_is_approximate = True
        res.df_note = SPEARMAN_DF_NOTE
        res.warnings.append(SPEARMAN_DF_WARNING)
    res.ci = _ci(level, lo, hi, f"the correlation between '{x_label}' and '{y_label}'", why)
    res.effect_size = E.describe_r(r_val, method)
    res.warnings.append(I.CORRELATION_NOTICE)
    res.plot_data = {"kind": "scatter", "x": xa[:MAX_PLOT_POINTS], "y": ya[:MAX_PLOT_POINTS], "n_total": n,
                     "x_label": x_label, "y_label": y_label}
    return _complete(res, checks)


def pearson_test(x, y, *, alpha: float = 0.05, confidence_level: float = 0.95, alternative: str = "two-sided",
                 x_label: str = "x", y_label: str = "y", time_ordered: bool = False) -> TestResult:
    return _correlation("pearson", x, y, alpha, confidence_level, alternative, x_label, y_label, time_ordered)


def spearman_test(x, y, *, alpha: float = 0.05, confidence_level: float = 0.95, alternative: str = "two-sided",
                  x_label: str = "x", y_label: str = "y", time_ordered: bool = False) -> TestResult:
    return _correlation("spearman", x, y, alpha, confidence_level, alternative, x_label, y_label, time_ordered)
