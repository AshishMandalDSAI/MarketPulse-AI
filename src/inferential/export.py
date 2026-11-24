"""Tables and CSV / Excel downloads for inferential results (Phase 8 Part 3).

Only values that were calculated are exported. An undefined value is an empty cell and
its reason appears in the ``Undefined statistics`` column; nothing is replaced by 0.
Spreadsheet formula injection is neutralised for values (``src.common.export``) and, here,
for text that starts with a formula character in column names.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Sequence

import numpy as np
import pandas as pd

from src.common.export import _FORMULA_PREFIXES, dataframe_to_csv_bytes, sheets_to_excel_bytes
from src.inferential.contracts import TestResult

SHEET_ORDER = ("Test Summary", "Assumption Checks", "Details", "Methodology")

METHODOLOGY = {
    "Hypothesis testing": "Every test states H0 and H1, computes a test statistic and p-value, and rejects H0 only when "
                          "p is strictly below the significance level α (default 0.05). A non-significant result does not "
                          "prove H0.",
    "Z-test": "z = (x̄ − μ0) / (σ/√n) with σ supplied by the user. Refused when σ is missing/invalid, or when n < 30 and the "
              "sample does not pass the Shapiro-Wilk normality check. CI: x̄ ± z·σ/√n. Effect size: d = (x̄ − μ0)/σ.",
    "One-sample t-test": "t = (x̄ − μ0) / (s/√n), df = n − 1 (SciPy ttest_1samp). CI from the t distribution. d = (x̄ − μ0)/s; "
                         "Hedges' g = d·J, J = 1 − 3/(4·df − 1).",
    "Independent t-test": "Welch's test by default (no equal-variance assumption, Welch-Satterthwaite df); Student's pooled test "
                          "on request (SciPy ttest_ind). CI for the difference (A − B). d uses the pooled sample SD.",
    "Paired t-test": "t-test on the differences (A − B) of complete pairs, df = pairs − 1 (SciPy ttest_rel). d_z = mean "
                     "difference / SD of differences.",
    "One-way ANOVA": "F = MS_between / MS_within, df = (k − 1, N − k) (SciPy f_oneway). η² = SS_between / SS_total; ω² also "
                     "reported. Groups with a single observation are excluded and disclosed. No post-hoc tests.",
    "Chi-square goodness-of-fit": "χ² = Σ (O − E)² / E, df = categories − 1 (SciPy chisquare). Expected frequencies equal "
                                  "unless proportions are supplied. Effect size: Cohen's w = √(χ²/n).",
    "Chi-square independence": "χ² on the contingency table (SciPy chi2_contingency; Yates' correction on 2 x 2 tables). "
                               "Cramér's V = √(χ²_uncorrected / (n · min(rows − 1, cols − 1))).",
    "Chi-square time-ordered data": "Both chi-square tests ASSUME independent observations and never establish it. If the rows are "
                                    "declared time-ordered the test still runs but carries an explicit independence warning "
                                    "(never reported as 'valid'): serial dependence can invalidate ordinary chi-square inference. A "
                                    "lag-1 screen (chi-square on the row-order transition table, Bonferroni-adjusted α per column) "
                                    "downgrades the result to 'unreliable' when it detects dependence; a non-significant screen only "
                                    "leaves it 'usable with caution' because the screen has limited power and checks lag 1 only.",
    "Chi-square validity rule": "The test is REFUSED when any expected count is below 1 or more than 20% of cells have an "
                                "expected count below 5 (Cochran's rule); the expected counts are shown instead of a p-value.",
    "Pearson correlation": "r with a t-based p-value, df = n − 2, CI by Fisher z (SciPy pearsonr). Measures straight-line "
                           "association only. Correlation does not establish causation.",
    "Spearman correlation": "Spearman's rho on ranks (average ranks for ties). The p-value is SciPy's Student-t APPROXIMATION: "
                            "t = ρ_s·√((n − 2)/(1 − ρ_s²)) referred to a t distribution; the reported 'degrees of freedom' n − 2 "
                            "are therefore APPROXIMATE (the df of that t distribution, borrowed from Pearson's r), not an exact "
                            "property of rank correlation, and the p-value is approximate (poorest for n < 10 or many ties). CI by "
                            "Fisher z with the Bonett-Wright standard error (also approximate, n ≥ 10). Correlation does not "
                            "establish causation.",
    "Confidence interval for a mean": "x̄ ± t(1−α/2, n−1) · s/√n. An estimate, not a hypothesis test.",
    "Confidence intervals": "Always two-sided at the chosen confidence level, whatever the alternative hypothesis. An undefined "
                            "interval is blank with a reason, never 0.",
    "Missing data": "Listwise deletion over the columns each test uses. Original, removed and final counts and the "
                    "missing percentage (removed / original × 100) are reported. Blank text counts as missing; missing "
                    "values are never replaced by 0.",
    "Normality": "Shapiro-Wilk (n ≤ 5000) or D'Agostino-Pearson (larger n). A failed check is serious only when n is small.",
    "Equal variances": "Levene's test with the median (Brown-Forsythe).",
    "Independence": "Cannot be verified from data. For time-ordered datasets a warning and the lag-1 autocorrelation are shown.",
    "Effect sizes": "Labels follow Cohen's rules of thumb. They describe size, not business importance.",
}


def _neutral(name) -> str:
    text = str(name)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def slug(label: str, default: str = "dataset") -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label or "")).strip("_")[:40]
    return s or default


def export_stem(label: str, table: str = None) -> str:
    stem = "inferential_statistics_" + slug(label)
    return stem + "_" + slug(table).lower() if table else stem


def _num(v):
    return np.nan if v is None else v


def _summary_row(r: TestResult) -> dict:
    ci, es, m = r.ci, r.effect_size, r.missing
    undefined = dict(r.undefined)
    if ci is not None and not ci.defined and ci.reason:
        undefined.setdefault("confidence_interval", ci.reason)
    if es is not None and es.value is None and es.reason:
        undefined.setdefault("effect_size", es.reason)
    if r.status == "refused":
        undefined = {"test": r.refusal_reason}
    sizes = "; ".join(f"{k}: {v}" for k, v in r.sample_sizes.items())
    ordinary = {k: v for k, v in r.inputs.items() if k not in ("confidence_level", "significance_level", "alternative")}
    return {
        "Test": r.test_name, "Status": "Run" if r.status == "ok" else "Not run (refused)",
        "Refusal reason": r.refusal_reason, "Recommended alternative": r.recommendation,
        "H0": r.h0, "H1": r.h1, "Alternative": r.alternative,
        "Test statistic name": r.statistic_name, "Test statistic": _num(r.statistic),
        "P-value": _num(r.p_value), "Significance level (α)": r.alpha,
        "Significant": "" if r.significant is None else ("Yes" if r.significant else "No"),
        "Degrees of freedom": _num(r.df), "Degrees of freedom (2nd)": _num(r.df2),
        "Degrees of freedom basis": ("" if r.df is None else ("Approximate" if r.df_is_approximate else "Exact")),
        "Degrees of freedom note": r.df_note,
        "Confidence level": ci.level if ci else np.nan,
        "CI lower": _num(ci.lower) if ci else np.nan, "CI upper": _num(ci.upper) if ci else np.nan,
        "CI estimates": ci.parameter if ci else "",
        "Effect size measure": es.name if es else "", "Effect size": _num(es.value) if es else np.nan,
        "Effect size label": es.label if es else "",
        "Assumption status": r.validity.replace("_", " "),
        "Independence of observations": {"not_verifiable": "Not verifiable (assumed, not tested)",
                                         "time_ordered_unverified": "Time-ordered: NOT established",
                                         "serial_dependence_detected": "Time-ordered: serial dependence detected"
                                         }.get(r.independence_status, r.independence_status),
        "Sample size(s)": sizes,
        "Original observations": m.original_n if m else np.nan, "Removed (missing)": m.removed_n if m else np.nan,
        "Final observations used": m.final_n if m else np.nan,
        "Missing %": m.missing_pct if m else np.nan,
        "Complete rows not used": m.not_used_n if m else np.nan,
        "Missing-data note": m.note if m else "",
        "Inputs": "; ".join(f"{k}: {v}" for k, v in ordinary.items()),
        "Interpretation": r.interpretation.get("summary", ""),
        "Warnings": " | ".join(r.warnings),
        "Undefined statistics": "; ".join(f"{k}: {v}" for k, v in undefined.items()),
    }


def build_tables(results: Sequence[TestResult], dataset_label: str = "", dataset_id: str = "") -> dict:
    """Named tables for one or more results of the ACTIVE dataset."""
    summary = pd.DataFrame([_summary_row(r) for r in results])
    if summary.empty:
        summary = pd.DataFrame({"Note": ["No test has been run yet."]})
    else:
        summary.insert(0, "Dataset", dataset_label)
        summary.insert(1, "Dataset id", dataset_id)
    arows, drows = [], []
    for r in results:
        for a in r.assumptions:
            arows.append({"Test": r.test_name, "Assumption": a.name, "Status": a.status.replace("_", " "),
                          "Consequence": a.consequence.replace("_", " "), "What was found": a.detail,
                          "Why it matters": a.why_it_matters, "Recommendation": a.recommendation})
        for name, frame in r.details.items():
            for i, rec in frame.reset_index(drop=True).iterrows():
                for col, val in rec.items():
                    drows.append({"Test": r.test_name, "Table": name, "Row": i + 1, "Measure": _neutral(col),
                                  "Value": "" if (isinstance(val, float) and np.isnan(val)) else val})
    assumptions = pd.DataFrame(arows) if arows else pd.DataFrame({"Note": ["No assumption checks were recorded."]})
    details = pd.DataFrame(drows) if drows else pd.DataFrame({"Note": ["No detail tables were produced."]})
    meth = pd.DataFrame({"Topic": list(METHODOLOGY), "Description": list(METHODOLOGY.values())})
    return {"Test Summary": summary, "Assumption Checks": assumptions, "Details": details, "Methodology": meth}


def tables_to_excel(tables: dict) -> bytes:
    return sheets_to_excel_bytes(tables)


def tables_to_csv_zip(tables: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in tables.items():
            zf.writestr(f"{slug(name).lower()}.csv", dataframe_to_csv_bytes(frame))
    return buf.getvalue()


def table_to_csv(tables: dict, name: str) -> bytes:
    return dataframe_to_csv_bytes(tables[name])
