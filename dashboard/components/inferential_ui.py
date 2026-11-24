"""Page body for Module 15 - Inferential Statistics (Phase 8 Part 3).

The shared scaffold (``stats_ui.render_module_page``) draws the overview and dataset
information; this module draws test selection, inputs, assumption checks, results, effect
size, confidence intervals, plain-English explanation and downloads.

Session safety: every widget key carries the ``mp_page_`` prefix and the active dataset
id, so a dataset switch can never reuse another dataset's selections. Results are
calculated from the ACTIVE table on every run (tests are cheap) and are never stored in a
cache or module-level variable, so a stale result cannot be shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components import session_paths
from dashboard.components.stats_ui import (
    apply_dark_layout, render_limitations, render_plain_english, render_section, widget_key,
)
from src.common.dataset_summary import DatasetSummary
from src.common.formatting import escape_text, fmt_int
from src.inferential import export as ix
from src.inferential import interpret as itp
from src.inferential import tests as T
from src.inferential.contracts import TestResult
from src.inferential.export import METHODOLOGY

MODULE_KEY = "inferential"
ALPHA_CHOICES = (0.01, 0.05, 0.10)
LEVEL_CHOICES = (0.90, 0.95, 0.99)
ALT_LABELS = {"two-sided": "Two-sided (different from)", "greater": "One-sided (greater than)",
              "less": "One-sided (less than)"}
GUIDE = {
    "z": "Compares a sample mean with a target when you KNOW the population standard deviation. Rarely applicable in practice.",
    "t1": "Compares the mean of one numeric column with a target value.",
    "t_ind": "Compares the mean of a numeric column between two groups (e.g. two regions).",
    "t_paired": "Compares two numeric columns measured on the same rows (e.g. before vs after).",
    "anova": "Compares the mean of a numeric column across three or more groups.",
    "chi_gof": "Checks whether the categories of one column occur equally often, or in proportions you specify.",
    "chi_ind": "Checks whether two categorical columns are associated.",
    "pearson": "Measures and tests the straight-line relationship between two numeric columns.",
    "spearman": "Measures and tests a steadily rising or falling relationship (rank-based) between two numeric columns.",
    "ci_mean": "Estimates a range of plausible values for the true mean of one numeric column (an estimate, not a test).",
}


def _key(name: str) -> str:
    return widget_key(MODULE_KEY, f"{name}_{session_paths.get_dataset_id()}")


# --------------------------------------------------------------------------
# Column pools (from the dataset's own metadata; nothing is hard-coded)
# --------------------------------------------------------------------------

def _pools(df: pd.DataFrame, summary: DatasetSummary) -> dict:
    skip = set(summary.identifier_columns) | set(summary.empty_columns)
    numeric = [c for c in summary.numeric_columns if c in df.columns and c not in skip]
    cats = [c for c in list(summary.categorical_columns) + list(summary.boolean_columns)
            if c in df.columns and c not in skip]
    return {"numeric": numeric, "categorical": cats,
            "n_levels": {c: int(df[c].nunique(dropna=True)) for c in cats}}


def _availability(pools: dict) -> dict:
    """test key -> None if available, otherwise the plain-English reason it is not."""
    nn, cats, lv = len(pools["numeric"]), pools["categorical"], pools["n_levels"]
    two_level = [c for c in cats if 2 <= lv[c] <= T.MAX_CATEGORIES]
    multi = [c for c in cats if 2 <= lv[c] <= T.MAX_GROUPS]
    need_num = "needs at least one numeric column"
    out = {
        "z": None if nn >= 1 else need_num, "t1": None if nn >= 1 else need_num, "ci_mean": None if nn >= 1 else need_num,
        "t_ind": None if nn >= 1 and two_level else "needs a numeric column and a categorical column with at least 2 groups",
        "t_paired": None if nn >= 2 else "needs at least two numeric columns",
        "anova": None if nn >= 1 and multi else "needs a numeric column and a categorical column with 2 to 30 groups",
        "chi_gof": None if two_level else "needs a categorical or yes/no column with at least 2 categories",
        "chi_ind": None if len(two_level) >= 2 else "needs at least two categorical or yes/no columns with 2+ categories",
        "pearson": None if nn >= 2 else "needs at least two numeric columns",
        "spearman": None if nn >= 2 else "needs at least two numeric columns",
    }
    return out


def _levels(df: pd.DataFrame, col: str) -> list:
    cat, err = T._as_categories(df[col], col)
    return [] if err else sorted(cat.dropna().unique().tolist())


def _parse(text: str, what: str, required: bool = True):
    text = (text or "").strip()
    if not text:
        if required:
            raise ValueError(f"Enter a number for {what}.")
        return None
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number (for {what}).") from None
    if not np.isfinite(v):
        raise ValueError(f"{what} must be a finite number.")
    return v


# --------------------------------------------------------------------------
# Controls -> result
# --------------------------------------------------------------------------

def _common_controls(test: str) -> tuple:
    c1, c2, c3 = st.columns(3)
    alpha = c1.selectbox("Significance level (α)", ALPHA_CHOICES, index=1, key=_key("alpha"),
                         help="How strong the evidence must be before the null hypothesis is rejected.")
    level = c2.selectbox("Confidence level", LEVEL_CHOICES, index=1, key=_key("conf"),
                         format_func=lambda v: f"{v * 100:.0f}%")
    if test in ("z", "t1", "t_ind", "t_paired", "pearson", "spearman"):
        alt = c3.selectbox("Alternative hypothesis", list(ALT_LABELS), key=_key("alt"), format_func=ALT_LABELS.get)
    else:
        alt = "two-sided"
        c3.caption("This test is always two-sided.")
    return float(alpha), float(level), alt


def _switch_to_t1() -> None:
    st.session_state[_key("test")] = "t1"


def _z_inputs(df: pd.DataFrame, col: str, mu0: float, alpha: float, level: float, alt: str,
              time_ordered: bool) -> "TestResult | None":
    """Z-test inputs (PART 03 fix). The known population SD is a required, validated input; the sample SD
    is shown for reference only and is never used in its place."""
    sigma_txt = st.text_input("Known population standard deviation (σ) — required", "", key=_key(f"sigma_{col}"),
                              help="σ must come from outside this sample (for example a long-run process value or "
                                   "a published population figure). It must be greater than zero.")
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(x) >= 2:
        st.caption(f"For reference only: the sample standard deviation of '{escape_text(col)}' is "
                   f"{float(x.std(ddof=1)):.4g}. It is **not** used as σ — using it would make this a t-test.")
    if not sigma_txt.strip():
        st.info("**Enter the known population standard deviation (σ) to run the Z-test.** The Z-test is only valid "
                "when σ is known from outside your data. If you do not know σ, use the **one-sample t-test**, which "
                "estimates the spread from the sample and accounts for that uncertainty.")
        st.button("Use the one-sample t-test instead", key=_key("to_t1"), on_click=_switch_to_t1)
        # The engine records the explicit refusal (no statistic is computed and nothing is substituted).
        return T.z_test(df[col], mu0, None, alpha=alpha, confidence_level=level, alternative=alt, column=col,
                        time_ordered=time_ordered)
    try:
        sigma = _parse(sigma_txt, "σ")
    except ValueError as exc:
        st.error(str(exc))
        return None
    if sigma <= 0:
        st.error(f"σ must be greater than zero (you entered {sigma:g}). A standard deviation of zero or less is not "
                 "possible for a population with any spread.")
        st.button("Use the one-sample t-test instead", key=_key("to_t1_bad"), on_click=_switch_to_t1)
        return T.z_test(df[col], mu0, sigma, alpha=alpha, confidence_level=level, alternative=alt, column=col,
                        time_ordered=time_ordered)
    return T.z_test(df[col], mu0, sigma, alpha=alpha, confidence_level=level, alternative=alt, column=col,
                    time_ordered=time_ordered)


def _zn(v: float) -> str:
    """Readable number for formulas: thousands separators for large values, never 4e+06 style."""
    v = float(v)
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        return f"{v:,.2f}" if abs(v) >= 1e4 else f"{v:.3e}"
    return f"{v:.6g}"


def _zp(p: float) -> str:
    return "< 1 \\times 10^{-16}" if p < 1e-16 else f"{p:.6g}"


def _z_formula_section(res: TestResult) -> None:
    """Formula, the calculation with the actual numbers, and a plain-language conclusion."""
    tbl = res.details.get("Sample summary")
    if tbl is None:
        return
    vals = dict(zip(tbl["Measure"], tbl["Value"]))
    n, mean, sigma, se = int(vals["n"]), float(vals["Sample mean"]), float(vals["Known population SD (σ)"]), \
        float(vals["Standard error σ/√n"])
    mu0 = float(res.inputs.get("hypothesised_mean", 0.0))
    render_section("Formula and Calculation")
    st.latex(r"Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}")
    st.latex(rf"Z = \frac{{{_zn(mean)} - {_zn(mu0)}}}{{{_zn(sigma)} / \sqrt{{{n}}}}} = "
             rf"\frac{{{_zn(mean - mu0)}}}{{{_zn(se)}}} = {res.statistic:.4f}")
    pf = {"two-sided": r"p = 2\,P(Z' \ge |z|)", "greater": r"p = P(Z' \ge z)", "less": r"p = P(Z' \le z)"}[res.alternative]
    st.latex(pf + (rf" {_zp(res.p_value)}" if res.p_value < 1e-16 else rf" = {_zp(res.p_value)}"))
    reject = res.p_value < res.alpha
    st.markdown(f"**Z-statistic:** {res.statistic:.4f} · **p-value:** {"< 1e-16" if res.p_value < 1e-16 else format(res.p_value, ".4g")} · **α:** {res.alpha:g} · "
                f"**Conclusion:** {'reject H0' if reject else 'do not reject H0'} "
                f"(p {'<' if reject else '≥'} α).")
    direction = {"two-sided": "different from", "greater": "greater than", "less": "less than"}[res.alternative]
    if reject:
        msg = (f"The sample mean ({_zn(mean)}) is {abs(res.statistic):.2f} standard errors away from {_zn(mu0)}. With the "
               f"known σ = {_zn(sigma)}, a gap this large would be very unlikely (p = {res.p_value:.3g}) if the true mean "
               f"were {_zn(mu0)}, so the data support a population mean {direction} {_zn(mu0)} at the {res.alpha:g} level.")
    else:
        msg = (f"The sample mean ({_zn(mean)}) is {abs(res.statistic):.2f} standard errors away from {_zn(mu0)}. With the "
               f"known σ = {_zn(sigma)}, a gap like this is plausible by chance (p = {res.p_value:.3g}), so the data do "
               f"not show that the population mean is {direction} {_zn(mu0)}. This is not proof that it equals {_zn(mu0)}.")
    st.info(msg)


def _run(test: str, df: pd.DataFrame, pools: dict, summary: DatasetSummary) -> "TestResult | None":
    """Draw the input controls for ``test`` and calculate. Returns None if inputs are incomplete."""
    num, cats = pools["numeric"], pools["categorical"]
    lv = pools["n_levels"]
    alpha, level, alt = _common_controls(test)
    default_time = bool(summary.datetime_columns)
    time_ordered = st.checkbox("My rows are in time order (for example daily or weekly records)", value=default_time,
                               key=_key("time_ordered"),
                               help="Time-ordered observations are often not independent, which tests assume.")
    if time_ordered and test in ("chi_gof", "chi_ind"):
        st.warning("**Independence warning (time-ordered rows).** " + T.CHI_TIME_ORDER_WARNING)
    try:
        if test in ("z", "t1", "ci_mean"):
            col = st.selectbox("Numeric column", num, key=_key("col1"))
            if test == "ci_mean":
                return T.mean_confidence_interval(df[col], confidence_level=level, column=col, alpha=alpha, time_ordered=time_ordered)
            mu0 = _parse(st.text_input("Hypothesised mean (μ0)", "0", key=_key(f"mu0_{col}")), "the hypothesised mean")
            if test == "t1":
                return T.one_sample_t(df[col], mu0, alpha=alpha, confidence_level=level, alternative=alt, column=col,
                                      time_ordered=time_ordered)
            return _z_inputs(df, col, mu0, alpha, level, alt, time_ordered)
        if test in ("t_ind", "anova"):
            pool = [c for c in cats if 2 <= lv[c] <= (T.MAX_CATEGORIES if test == "t_ind" else T.MAX_GROUPS)]
            vcol = st.selectbox("Numeric column (the values to compare)", num, key=_key("vcol"))
            gcol = st.selectbox("Group column", pool, key=_key("gcol"))
            if test == "anova":
                return T.one_way_anova(df[vcol], df[gcol], alpha=alpha, confidence_level=level, value_label=vcol,
                                       group_label=gcol, time_ordered=time_ordered)
            levels = _levels(df, gcol)
            if len(levels) < 2:
                st.info("The chosen group column does not have two usable groups.")
                return None
            a, b = st.columns(2)
            la = a.selectbox("First group (A)", levels, index=0, key=_key(f"la_{gcol}"))
            lb = b.selectbox("Second group (B)", levels, index=1, key=_key(f"lb_{gcol}"))
            variant = st.radio("Variance assumption", ["Welch (recommended: does not assume equal variances)",
                                                       "Student (assumes equal variances)"], key=_key("variant"))
            return T.independent_t(df[vcol], df[gcol], la, lb, equal_var=variant.startswith("Student"), alpha=alpha,
                                   confidence_level=level, alternative=alt, value_label=vcol, group_label=gcol,
                                   time_ordered=time_ordered)
        if test in ("t_paired", "pearson", "spearman"):
            a, b = st.columns(2)
            ca = a.selectbox("First numeric column", num, index=0, key=_key("ca"))
            cb = b.selectbox("Second numeric column", num, index=min(1, len(num) - 1), key=_key("cb"))
            if test == "t_paired":
                return T.paired_t(df[ca], df[cb], alpha=alpha, confidence_level=level, alternative=alt, label_a=ca,
                                  label_b=cb, time_ordered=time_ordered)
            fn = T.pearson_test if test == "pearson" else T.spearman_test
            return fn(df[ca], df[cb], alpha=alpha, confidence_level=level, alternative=alt, x_label=ca, y_label=cb,
                      time_ordered=time_ordered)
        pool = [c for c in cats if 2 <= lv[c] <= T.MAX_CATEGORIES]
        if test == "chi_gof":
            col = st.selectbox("Categorical column", pool, key=_key("catcol"))
            mode = st.radio("Expected frequencies", ["Equal for every category", "My own percentages"], key=_key(f"gofmode_{col}"))
            props = None
            if mode.startswith("My"):
                levels = _levels(df, col)
                if len(levels) > 20:
                    st.info("Custom percentages are limited to 20 categories; use equal frequencies or another column.")
                    return None
                st.caption("Enter the expected percentage for every category (they must add up to 100).")
                props = {}
                for lvl in levels:
                    txt = st.text_input(f"Expected % for '{lvl}'", "", key=_key(f"gofp_{col}_{lvl}"))
                    p = _parse(txt, f"the expected percentage for '{lvl}'")
                    props[lvl] = p / 100.0
            return T.chi_square_gof(df[col], expected_proportions=props, alpha=alpha, column=col, time_ordered=time_ordered)
        a, b = st.columns(2)
        ca = a.selectbox("First categorical column", pool, index=0, key=_key("cha"))
        cb = b.selectbox("Second categorical column", pool, index=min(1, len(pool) - 1), key=_key("chb"))
        return T.chi_square_independence(df[ca], df[cb], alpha=alpha, label_a=ca, label_b=cb, time_ordered=time_ordered)
    except ValueError as exc:
        st.warning(str(exc))
        return None


# --------------------------------------------------------------------------
# Result sections
# --------------------------------------------------------------------------

def _fmt(v, digits: int = 4) -> str:
    return "—" if v is None else f"{v:.{digits}g}"


def _chart(res: TestResult) -> None:
    """Chart of the data behind the test (PART 03: every chart has a title and named axes)."""
    import plotly.graph_objects as go
    pdata = res.plot_data
    if not pdata:
        return
    kind = pdata["kind"]
    inp = res.inputs
    fig = go.Figure()
    title = res.test_name
    if kind == "hist":
        col = inp.get("column") or (f"{inp.get('column_a')} − {inp.get('column_b')}" if res.test_key == "t_paired" else "value")
        what = "Paired differences" if res.test_key == "t_paired" else f"Distribution of {col}"
        fig.add_histogram(x=pdata["x"], name="Observed values", marker_color="#4C9AFF")
        fig.add_vline(x=pdata["ref"], line_dash="dash", line_color="#FF8B00", annotation_text=pdata["ref_label"])
        fig.update_layout(xaxis_title=("Difference " + col) if res.test_key == "t_paired" else col,
                          yaxis_title="Number of rows")
        title = (f"Paired differences ({col}) with zero difference marked" if res.test_key == "t_paired"
                 else f"{what} with the {pdata['ref_label']} marked")
    elif kind == "groups":
        for name, arr in pdata["groups"].items():
            fig.add_box(y=arr, name=str(name), boxmean=True)
        fig.update_layout(xaxis_title=inp.get("group_column", "Group"), yaxis_title=inp.get("value_column", "value"),
                          showlegend=False)
        title = f"{inp.get('value_column', 'Values')} by {inp.get('group_column', 'group')} (box = quartiles, dashed = mean)"
    elif kind == "bars":
        fig.add_bar(x=pdata["labels"], y=pdata["observed"], name="Observed")
        fig.add_bar(x=pdata["labels"], y=pdata["expected"], name="Expected")
        fig.update_layout(barmode="group", xaxis_title=inp.get("column", "Category"), yaxis_title="Number of rows",
                          legend_title_text="Frequency")
        title = f"Observed vs expected frequencies of {inp.get('column', 'the category')}"
    elif kind == "heat":
        t = pdata["table"]
        fig.add_heatmap(z=t.to_numpy(), x=[str(c) for c in t.columns], y=[str(i) for i in t.index], colorscale="Blues",
                        colorbar=dict(title="Rows"), text=t.to_numpy(), texttemplate="%{text}")
        fig.update_layout(xaxis_title=inp.get("column_b", "Second variable"), yaxis_title=inp.get("column_a", "First variable"))
        title = f"Contingency table: {inp.get('column_a', '')} × {inp.get('column_b', '')} (row counts)"
    elif kind == "scatter":
        fig.add_scatter(x=pdata["x"], y=pdata["y"], mode="markers", marker=dict(size=6, opacity=0.6))
        fig.update_layout(xaxis_title=pdata["x_label"], yaxis_title=pdata["y_label"])
        title = f"{pdata['y_label']} vs {pdata['x_label']}"
        if pdata["n_total"] > len(pdata["x"]):
            st.caption(f"The chart shows the first {len(pdata['x']):,} of {pdata['n_total']:,} pairs; the test uses all of them.")
    st.plotly_chart(apply_dark_layout(fig, title, 380, x_title=fig.layout.xaxis.title.text,
                                      y_title=fig.layout.yaxis.title.text), use_container_width=True)


def _missing_section(res: TestResult) -> None:
    m = res.missing
    if m is None:
        return
    st.markdown("**Data used and missing values** (rows with a missing value in any selected column are removed "
                "for this test only; the dataset is not changed).")
    rows = {"Original observations": fmt_int(m.original_n), "Removed (missing or blank)": fmt_int(m.removed_n),
            "Final observations used": fmt_int(m.final_n), "Missing %": f"{m.missing_pct:.2f}%"}
    if m.not_used_n:
        rows["Complete rows not used (other groups)"] = fmt_int(m.not_used_n)
    st.table(pd.DataFrame({"Measure": list(rows), "Value": list(rows.values())}).set_index("Measure"))
    if m.per_variable:
        st.caption("Missing values per selected column: " + ", ".join(f"{escape_text(k)} = {v:,}" for k, v in m.per_variable.items()))
    if m.note:
        st.caption(escape_text(m.note))
    if res.sample_sizes:
        st.caption("Sample sizes: " + ", ".join(f"{escape_text(str(k))} = {v:,}" for k, v in res.sample_sizes.items()))


def _assumption_section(res: TestResult) -> None:
    render_section("Assumption Checks")
    st.caption("Assumptions are checked before any test is calculated. A test that cannot be run validly is refused.")
    if not res.assumptions:
        st.info("No assumption checks were recorded because the inputs were rejected first (see the explanation below).")
        return
    icon = {"met": "✅", "violated": "❌", "caution": "⚠️", "not_checked": "ℹ️"}
    for a in res.assumptions:
        with st.expander(f"{icon.get(a.status, '•')} {a.name} - {a.status.replace('_', ' ')}", expanded=a.status in ("violated", "caution")):
            st.markdown(f"**What was found:** {escape_text(a.detail)}")
            if a.why_it_matters:
                st.markdown(f"**Why it matters:** {escape_text(a.why_it_matters)}")
            if a.consequence != "none":
                st.markdown(f"**Effect on the result:** {itp.validity_text(a.consequence)}")
            if a.recommendation:
                st.markdown(f"**Suggested alternative:** {escape_text(a.recommendation)}")


def _results_section(res: TestResult) -> None:
    render_section("Statistical Results")
    if res.status != "ok":
        st.warning(f"**{escape_text(res.test_name)} was not run.** {escape_text(res.refusal_reason)}")
        if res.recommendation:
            st.info(f"**What to do instead:** {escape_text(res.recommendation)}")
        for name, frame in res.details.items():
            st.markdown(f"**{name}** (shown so you can see why):")
            st.dataframe(frame, use_container_width=True, hide_index=True)
        _missing_section(res)
        return
    if res.validity == "valid":
        st.success("No assumption problems were found.")
    elif res.validity == "usable_with_caution":
        st.warning("Usable with caution - see the assumption checks above.")
    else:
        st.error(f"This result is **{res.validity}** because of the assumption problems above. Do not base a decision on it.")
    ind_text = itp.independence_text(res)
    if ind_text:
        # Explicit and unconditional for time-ordered chi-square results: never silently implies independence holds.
        (st.error if res.independence_status == "serial_dependence_detected" else st.warning)(
            "**Independence NOT established.** " + escape_text(ind_text))
    if res.h0:
        st.markdown(f"**Null hypothesis (H0):** {escape_text(res.h0)}  \n**Alternative hypothesis (H1):** {escape_text(res.h1)}")
    rows = {f"Test statistic ({res.statistic_name})": _fmt(res.statistic)}
    if res.p_value is not None:
        rows["P-value"] = itp.fmt_p(res.p_value)
        rows["Significance level (α)"] = f"{res.alpha:g}"
        rows["Statistically significant"] = "Yes" if res.significant else "No"
    if res.df is not None:
        df_label = "Degrees of freedom (approximate)" if res.df_is_approximate else "Degrees of freedom"
        rows[df_label] = _fmt(res.df) if res.df2 is None else f"{_fmt(res.df)} and {_fmt(res.df2)}"
    else:
        reason = res.undefined.get("degrees_of_freedom")
        if reason:
            rows["Degrees of freedom"] = "not applicable"
    st.table(pd.DataFrame({"Result": list(rows), "Value": list(rows.values())}).set_index("Result"))
    for k in ("degrees_of_freedom", "p_value"):
        if res.undefined.get(k):
            st.caption(escape_text(res.undefined[k]))
    if res.df_is_approximate and res.df_note:
        st.caption("**About the degrees of freedom:** " + escape_text(res.df_note))
    _missing_section(res)
    for name, frame in res.details.items():
        st.markdown(f"**{name}**")
        st.dataframe(frame, use_container_width=True, hide_index=True)
    _chart(res)
    for w in res.warnings:
        st.caption(escape_text(w))


def _effect_section(res: TestResult) -> None:
    render_section("Effect-Size Interpretation")
    es = res.effect_size
    if res.status != "ok":
        st.info("No effect size, because the test was not run.")
        return
    if es is None:
        st.info(escape_text(res.undefined.get("effect_size", "Not applicable.")))
        return
    if es.value is None:
        st.info(f"**{escape_text(es.name)}** is undefined: {escape_text(es.reason)}")
        return
    st.markdown(f"**{escape_text(es.name)} = {es.value:.3g}** - {es.label}")
    st.markdown(escape_text(es.practical))
    for k, v in es.extra.items():
        if v is not None:
            st.caption(f"{k.replace('_', ' ').capitalize()}: {v:.3g}")
    st.caption("Limitations: " + escape_text(es.limitations))
    st.caption("Statistical significance is not the same as practical or business importance: a very small p-value can "
               "go with a trivial effect, and a large effect can fail to reach significance in a small sample.")


def _ci_section(res: TestResult) -> None:
    render_section("Confidence Intervals")
    ci = res.ci
    if res.status != "ok":
        st.info("No confidence interval, because the test was not run.")
        return
    if ci is None:
        st.info("Not applicable to this test.")
    elif ci.defined:
        st.markdown(f"**{ci.level * 100:.0f}% confidence interval for {escape_text(ci.parameter)}:** "
                    f"{ci.lower:.4g} to {ci.upper:.4g}")
        st.caption("The interval is always two-sided at the chosen confidence level, whatever the alternative hypothesis.")
    else:
        st.info(f"No confidence interval is shown: {escape_text(ci.reason)}")


def _plain_english_section(res: TestResult) -> None:
    render_section("Plain-English Explanation")
    it = res.interpretation
    if res.status != "ok":
        render_plain_english(escape_text(it.get("summary", "")))
        return
    render_plain_english(escape_text(it.get("headline", "")))
    for key, title in (("independence", "Independence warning"), ("p_value_meaning", "What the p-value means"), ("decision", "Decision"),
                       ("effect", "Size of the effect"), ("practical", "Practical importance"),
                       ("confidence_interval", "Confidence interval"), ("does_not_prove", "What this does NOT prove"),
                       ("assumptions", "Reliability"), ("approximation", "About the degrees of freedom")):
        if it.get(key):
            st.markdown(f"**{title}.** {escape_text(it[key])}")
    if it.get("causation"):
        st.warning(it["causation"])


def _download_section(res: TestResult) -> None:
    render_section("Downloads")
    active = session_paths.get_active_dataset() or {}
    label = "NovaMart_demo" if active.get("is_demo") else (active.get("filename") or session_paths.get_dataset_id())
    tables = ix.build_tables([res], label, session_paths.get_dataset_id())
    st.caption("The files contain exactly the result shown above for the current dataset and inputs: test, H0/H1, statistic, "
               "p-value, degrees of freedom, confidence interval, effect size, assumption status, sample sizes, missing-value "
               "information and interpretation. Undefined values are blank with the reason beside them.")
    c1, c2, c3 = st.columns(3)
    pick = c1.selectbox("Table", list(tables), key=_key("dl_table"))
    try:
        c1.download_button("⬇️ Download this table (CSV)", data=ix.table_to_csv(tables, pick),
                           file_name=ix.export_stem(label, pick) + ".csv", mime="text/csv", key=_key("dl_csv"))
        c2.download_button("⬇️ Download all tables (CSV, zipped)", data=ix.tables_to_csv_zip(tables),
                           file_name=ix.export_stem(label) + "_csv.zip", mime="application/zip", key=_key("dl_zip"))
    except Exception:
        c1.caption("CSV downloads could not be prepared for this result.")
    try:
        c3.download_button("⬇️ Download Excel workbook", data=ix.tables_to_excel(tables),
                           file_name=ix.export_stem(label) + ".xlsx", key=_key("dl_xlsx"),
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        c3.caption(f"Excel unavailable: {exc}" if str(exc) else "Excel download could not be prepared.")


# --------------------------------------------------------------------------
# Entry point (passed to render_module_page as body_fn)
# --------------------------------------------------------------------------

def render_inferential_body(df: pd.DataFrame, summary: DatasetSummary) -> None:
    st.info("**Inferential Statistics** is delivered in Part 3 of Phase 8 and is implemented. Every result on this page is "
            "calculated from the active dataset only, using the columns you select.")
    pools = _pools(df, summary)
    avail = _availability(pools)
    usable = [k for k in T.TEST_ORDER if avail[k] is None]

    render_section("Statistical Test Selection")
    if not usable:
        st.warning("This dataset cannot support any of the available tests. " + " ".join(
            f"{T.TEST_SPECS[k]}: {avail[k]}." for k in T.TEST_ORDER))
        return
    # PART 03: open on the one-sample t-test rather than the Z-test, which cannot run until a known σ is entered.
    default_idx = usable.index("t1") if "t1" in usable else 0
    test = st.selectbox("Which test do you want to run?", usable, index=default_idx, key=_key("test"),
                        format_func=T.TEST_SPECS.get)
    st.caption(GUIDE[test])
    unavailable = [k for k in T.TEST_ORDER if avail[k] is not None]
    if unavailable:
        with st.expander("Tests that are not available for this dataset, and why"):
            for k in unavailable:
                st.markdown(f"- **{T.TEST_SPECS[k]}** - {avail[k]}.")
    if summary.identifier_columns:
        st.caption("Identifier columns are not offered as inputs: " + ", ".join(escape_text(str(c)) for c in summary.identifier_columns[:8]))

    render_section("Input Controls")
    res = _run(test, df, pools, summary)
    if res is None:
        return
    _assumption_section(res)
    _results_section(res)
    if res.ok and res.test_key == "z":
        _z_formula_section(res)
    _effect_section(res)
    _ci_section(res)
    _plain_english_section(res)
    _download_section(res)
    with st.expander("How every number is calculated"):
        for topic, text in METHODOLOGY.items():
            st.markdown(f"**{escape_text(topic)}** - {escape_text(text)}")
    render_limitations([
        "A test result describes the data you selected; it does not show cause and effect.",
        "Independence of observations cannot be verified from the data; confirm it from how the data were collected.",
        "Running many tests increases the chance of a false positive; this page does not adjust for multiple testing.",
        "Statistical significance is not business importance; look at the effect size and the confidence interval.",
    ])
