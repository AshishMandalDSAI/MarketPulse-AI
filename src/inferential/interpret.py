"""Plain-English interpretation of inferential results (Phase 8 Part 3).

The same words are used on the page and in the exports. The layer never says that a
null hypothesis is "proven", that significance means importance, or that a
correlation shows cause.
"""

from __future__ import annotations

from typing import Optional

from src.inferential.contracts import CONSEQUENCE_TEXT, TestResult

CORRELATION_NOTICE = "Correlation does not establish causation."
CORRELATION_TESTS = ("pearson", "spearman")
CHI_TESTS = ("chi_gof", "chi_ind")


def is_significant(p_value: Optional[float], alpha: float) -> Optional[bool]:
    """Reject H0 only when p is strictly below alpha (p == alpha is NOT significant)."""
    if p_value is None:
        return None
    return bool(p_value < alpha)


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "not available"
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def _ci_text(r: TestResult) -> str:
    ci = r.ci
    if ci is None:
        return ""
    if not ci.defined:
        return f"No {ci.level * 100:.0f}% confidence interval is available: {ci.reason}"
    return (f"The {ci.level * 100:.0f}% confidence interval for {ci.parameter} is {ci.lower:.4g} to {ci.upper:.4g}. "
            "If the study were repeated many times, about that share of such intervals would contain the true value; "
            "it is not a 95% chance for this one interval.")


def refusal(r: TestResult) -> dict:
    text = {
        "headline": f"{r.test_name} was not run.",
        "reason": r.refusal_reason,
        "alternative": r.recommendation or "",
        "does_not_prove": "No statistical conclusion can be drawn because no test was calculated.",
    }
    text["summary"] = " ".join(v for v in (text["headline"], text["reason"], text["alternative"]) if v)
    return text


def build(r: TestResult) -> dict:
    """Interpretation for a completed (or refused) result; also fills ``significant``."""
    if r.status != "ok":
        return refusal(r)
    out: dict = {}
    if r.p_value is None:                       # a plain estimate, e.g. a confidence interval for a mean
        out["headline"] = f"{r.test_name}: an estimate, not a hypothesis test."
        out["confidence_interval"] = _ci_text(r)
        out["does_not_prove"] = ("The interval describes uncertainty about the average in the wider population, "
                                 "assuming the sample is representative. It does not say anything about cause.")
        out["assumptions"] = _assumption_summary(r)
        out["summary"] = " ".join(v for v in (out["headline"], out["confidence_interval"], out["does_not_prove"],
                                              out["assumptions"]) if v)
        return out
    sig = is_significant(r.p_value, r.alpha)
    r.significant = sig
    p = fmt_p(r.p_value)
    out["p_value_meaning"] = (f"The p-value is {p}. It is the probability of seeing a result at least this extreme "
                              "if the null hypothesis were true. It is not the probability that the null hypothesis "
                              "is true, and not the probability that the result happened by chance alone.")
    if sig:
        out["decision"] = (f"Because p ({p}) is below the significance level α = {r.alpha:g}, the null hypothesis is "
                           "rejected: the data are statistically significant evidence against it.")
        out["headline"] = f"{r.test_name}: statistically significant (p {('= ' + p) if not p.startswith('<') else p})."
        out["does_not_prove"] = ("A significant result does not prove the alternative hypothesis, does not show what "
                                 "caused the difference, and does not mean the effect is large or valuable. Roughly "
                                 f"{r.alpha * 100:g}% of tests on pure noise would still come out significant.")
    else:
        out["decision"] = (f"Because p ({p}) is not below α = {r.alpha:g}, the null hypothesis is not rejected: the data "
                           "do not provide enough evidence against it.")
        out["headline"] = f"{r.test_name}: not statistically significant (p = {p})."
        out["does_not_prove"] = ("Not rejecting the null hypothesis does not prove it true. The effect may be real but "
                                 "too small, or the sample too small or too noisy, to detect it.")
    out["significance"] = f"Statistical significance at α = {r.alpha:g}: {'yes' if sig else 'no'}."
    es = r.effect_size
    if es and es.value is not None:
        out["effect"] = f"Effect size ({es.name}) = {es.value:.3g} ({es.label}). {es.practical}"
        if sig and es.label == "negligible":
            out["practical"] = ("The result is statistically significant but the effect is negligible, so it may "
                                "be too small to matter in practice. Significance only says the effect is unlikely to be "
                                "exactly zero; large samples make tiny effects significant.")
        elif sig and es.label == "small":
            out["practical"] = ("The result is statistically significant and the effect is small. Whether that matters "
                                "depends on the cost and benefit of a change of this size.")
        elif sig:
            out["practical"] = (f"The result is statistically significant and the effect is {es.label}. Statistical "
                                "significance still does not guarantee business success; weigh it against cost and context.")
        elif es.label in ("medium", "large"):
            out["practical"] = (f"The effect looks {es.label} but is not statistically significant, which usually means "
                                "the sample is too small to be sure. It is worth collecting more data, not concluding "
                                "that there is no effect.")
        else:
            out["practical"] = ("The effect is small and not statistically significant: there is no evidence of a "
                                "meaningful effect in this data.")
    else:
        why = es.reason if es else "no effect size is available"
        out["practical"] = f"The practical size of the effect cannot be judged: {why}"
    ci = _ci_text(r)
    if ci:
        out["confidence_interval"] = ci
    out["assumptions"] = _assumption_summary(r)
    if r.test_key in CORRELATION_TESTS:
        out["causation"] = CORRELATION_NOTICE + " Two columns can move together because of a third factor, by coincidence, or in the reverse direction."
    ind = independence_text(r)
    if ind:
        out["independence"] = ind
    approx = approximation_text(r)
    if approx:
        out["approximation"] = approx
    parts = [out["headline"], out.get("independence", ""), out["p_value_meaning"], out["decision"], out.get("effect", ""),
             out["practical"], out.get("confidence_interval", ""), out["does_not_prove"], out["assumptions"],
             out.get("approximation", ""), out.get("causation", "")]
    out["summary"] = " ".join(x for x in parts if x)
    return out


def independence_text(r: TestResult) -> str:
    """Explicit independence disclosure for the chi-square tests when the rows were declared time-ordered."""
    if r.test_key not in CHI_TESTS or not r.inputs.get("time_ordered"):
        return ""
    text = ("Independence warning: the rows were declared time-ordered. The chi-square test ASSUMES independent observations and "
            "does NOT establish independence. Serial dependence may invalidate ordinary chi-square inference, typically making "
            "p-values too small and significant results too frequent.")
    if r.independence_status == "serial_dependence_detected":
        return text + " A lag-1 screen detected serial dependence, so this p-value is unreliable."
    return text + (" No serial dependence was detected by the lag-1 screen (or it could not be screened), but that screen has "
                   "limited power and cannot show that independence holds; treat the result as exploratory.")


def approximation_text(r: TestResult) -> str:
    """Disclosure that a displayed degrees-of-freedom value belongs to an approximating distribution."""
    return r.df_note if r.df_is_approximate else ""


def _assumption_summary(r: TestResult) -> str:
    verdict = {"valid": "No assumption problems were found.",
               "usable_with_caution": "The result is usable with caution:",
               "unreliable": "The result is unreliable because of these problems:",
               "invalid": "The result is invalid because of these problems:"}.get(r.validity, "")
    issues = [f"{a.name} - {a.detail}" + (f" Suggested: {a.recommendation}" if a.recommendation else "")
              for a in r.assumptions if a.consequence != "none"]
    return " ".join([verdict] + issues) if issues else verdict


def validity_text(validity: str) -> str:
    return CONSEQUENCE_TEXT.get({"valid": "none"}.get(validity, validity), validity.replace("_", " "))
