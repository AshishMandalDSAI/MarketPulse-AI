"""Page body for Module 14 - Descriptive Statistics (Phase 8 Part 2).

The layout comes from the shared scaffold (``stats_ui.render_module_page``); this
module only draws the module's own sections. All numbers come from
``src.descriptive`` and are calculated from the ACTIVE dataset's table.

Session safety: every widget key carries the active dataset id and the
``mp_page_`` prefix, so a dataset switch can never reuse another dataset's
selection. The report cache is keyed by the table's CONTENT (a file change is
a different key) and its worker never reads session state.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import session_paths
from dashboard.components.stats_ui import (
    apply_dark_layout, render_plain_english, render_section, render_limitations, widget_key,
)
from src.common.dataset_summary import DatasetSummary
from src.common.formatting import escape_text, fmt_int, fmt_number, fmt_pct
from src.descriptive import charts as ch
from src.descriptive import export as dx
from src.descriptive.contracts import DescriptiveReport, DescriptiveSettings
from src.descriptive.engine import METHODOLOGY, OUTLIER_DISCLAIMER, _unique_names, analyze_dataframe

MODULE_KEY = "descriptive"
PERCENTILE_CHOICES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
DEFAULT_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
MAX_TABLE_ROWS = 500


# --------------------------------------------------------------------------
# Calculation (cached on the table CONTENT, never on session state)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Calculating statistics...", max_entries=8)
def _report_worker(df: pd.DataFrame, percentiles: tuple, multiplier: float, method: str) -> DescriptiveReport:
    return analyze_dataframe(df, DescriptiveSettings(percentiles=percentiles, outlier_multiplier=multiplier,
                                                     correlation_method=method))


def compute_report(df: pd.DataFrame, settings: DescriptiveSettings) -> DescriptiveReport:
    s = settings.validate()
    try:
        return _report_worker(df, s.percentiles, s.outlier_multiplier, s.correlation_method)
    except Exception:   # e.g. cells Streamlit cannot hash (lists/dicts): compute without the cache; a real error re-raises
        return analyze_dataframe(df, s)


def _key(name: str) -> str:
    return widget_key(MODULE_KEY, f"{name}_{session_paths.get_dataset_id()}")


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------

def _show(df: pd.DataFrame, rows: int = MAX_TABLE_ROWS) -> None:
    if len(df) > rows:
        st.caption(f"Showing the first {rows:,} of {len(df):,} rows. The downloads contain every row.")
    st.dataframe(df.head(rows), use_container_width=True, hide_index=True)


def _numeric_table(report: DescriptiveReport) -> pd.DataFrame:
    return pd.DataFrame([{
        "Column": n.column, "Valid": fmt_int(n.valid_count), "Missing": fmt_int(n.missing_count),
        "Missing %": fmt_pct(n.missing_pct, 1), "Mean": fmt_number(n.mean), "Median": fmt_number(n.median),
        "Mode": (", ".join(fmt_number(m) for m in n.modes[:3]) + (" …" if len(n.modes) > 3 else "")) or "—",
        "Std dev (sample)": fmt_number(n.std_sample), "Min": fmt_number(n.minimum), "Max": fmt_number(n.maximum),
        "Range": fmt_number(n.value_range), "Skewness": fmt_number(n.skewness),
        "Kurtosis (excess)": fmt_number(n.kurtosis)} for n in report.numeric])


def _spread_table(report: DescriptiveReport) -> pd.DataFrame:
    rows = []
    for n in report.numeric:
        row = {"Column": n.column, "Sum": fmt_number(n.total), "Variance (sample)": fmt_number(n.variance_sample),
               "Variance (population)": fmt_number(n.variance_population),
               "Std dev (population)": fmt_number(n.std_population), "Q1": fmt_number(n.q1),
               "Q2 (median)": fmt_number(n.q2), "Q3": fmt_number(n.q3), "IQR": fmt_number(n.iqr)}
        for p, v in n.percentiles.items():
            row[f"P{p:g}"] = fmt_number(v)
        row["Zeros"] = fmt_int(n.zero_count)
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix_display(m, decimals=3) -> pd.DataFrame:
    frame = pd.DataFrame([[("—" if v is None else f"{v:.{decimals}f}" if abs(v) < 1e6 else f"{v:.4g}") for v in row]
                          for row in m.values], columns=m.columns, index=m.columns)
    return frame.reset_index().rename(columns={"index": "Column"})


def _show_undefined(items: list, title: str) -> None:
    items = [i for i in items if i]
    if items:
        with st.expander(title):
            for i in items:
                st.markdown(f"- {escape_text(i)}")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _settings_panel() -> DescriptiveSettings:
    render_section("Analysis Settings")
    with st.expander("Percentiles, outliers and correlation", expanded=False):
        c1, c2, c3 = st.columns(3)
        pcts = c1.multiselect("Percentiles to show", PERCENTILE_CHOICES, default=DEFAULT_PERCENTILES,
                              key=_key("percentiles"), help="Linear interpolation between closest ranks.")
        k = c2.selectbox("Outlier fence multiplier (k)", [1.5, 2.0, 3.0], index=0, key=_key("k"),
                         help="1.5 is the usual Tukey rule; 3.0 flags only extreme values.")
        method = c3.radio("Correlation method", ["pearson", "spearman"], key=_key("method"), horizontal=True,
                          format_func=lambda m: "Pearson (straight-line)" if m == "pearson" else "Spearman (rank-based)")
    return DescriptiveSettings(percentiles=tuple(float(p) for p in (pcts or (50,))), outlier_multiplier=float(k),
                               correlation_method=method)


def _quality_section(report: DescriptiveReport) -> None:
    render_section("Data Quality Overview")
    m, q = report.missing, report.quality
    cols = st.columns(4)
    cols[0].metric("Complete rows", fmt_pct(m.complete_row_pct, 1) if m else "—")
    cols[1].metric("Missing cells", fmt_pct(m.missing_cell_pct, 2) if m else "—")
    cols[2].metric("Duplicate rows", fmt_int(q.get("duplicate_rows")))
    cols[3].metric("Constant / empty columns", f"{len(q.get('constant_columns', []))} / {len(report.empty_columns)}")
    notes = []
    if q.get("constant_columns"):
        notes.append("Constant columns (one value only): " + ", ".join(q["constant_columns"]))
    if report.empty_columns:
        notes.append("Empty columns (no values): " + ", ".join(report.empty_columns))
    if report.identifiers:
        notes.append("Identifier-like columns (counts only, left out of correlation): "
                     + ", ".join(i.column for i in report.identifiers))
    for n in notes:
        st.markdown(f"- {escape_text(n)}")
    if report.conversions:
        with st.expander(f"Column type decisions ({len(report.conversions)})"):
            for c in report.conversions:
                st.markdown(f"- {escape_text(c)}")
    render_plain_english("Statistics use only the values that are present. A missing value is never treated as zero, "
                         "and every percentage of missing data is out of all rows in the dataset.")


def _numeric_section(report: DescriptiveReport) -> None:
    render_section("Numeric Column Statistics")
    if not report.numeric:
        st.info("This dataset has no numeric measurement columns to summarize.")
        return
    _show(_numeric_table(report))
    with st.expander("Spread, quartiles and percentiles"):
        _show(_spread_table(report))
    _show_undefined([f"{n.column} - {stat}: {why}" for n in report.numeric for stat, why in n.undefined.items()],
                    "Why some values show — (undefined, not zero)")
    _show_undefined([f"{n.column}: {w}" for n in report.numeric for w in n.warnings], "Column notes")
    ties = [f"{n.column}: {n.mode_note}" for n in report.numeric if len(n.modes) > 1 or "No mode" in n.mode_note]
    _show_undefined(ties, "Mode notes (ties and columns without a mode)")
    render_plain_english("Mean is the average, median is the middle value, and the standard deviation shows how far "
                         "values typically sit from the average. Skewness above 0 means a longer tail to the right; "
                         "excess kurtosis above 0 means heavier tails than a normal curve.")


def _categorical_section(report: DescriptiveReport) -> None:
    render_section("Categorical and Yes/No Column Statistics")
    if not report.categorical and not report.boolean:
        st.info("This dataset has no categorical or yes/no columns.")
        return
    if report.categorical:
        _show(pd.DataFrame([{
            "Column": c.column, "Valid": fmt_int(c.valid_count), "Missing": fmt_int(c.missing_count),
            "Missing %": fmt_pct(c.missing_pct, 1), "Unique values": fmt_int(c.unique_count),
            "Mode": ", ".join(c.modes[:3]) + (" …" if len(c.modes) > 3 else ""),
            "Mode share": fmt_pct(c.mode_pct, 1)} for c in report.categorical]))
        pick = st.selectbox("Show top and least frequent categories for", [c.column for c in report.categorical],
                            key=_key("cat_pick"))
        c = next(x for x in report.categorical if x.column == pick)
        left, right = st.columns(2)
        left.markdown("**Most frequent**")
        left.dataframe(pd.DataFrame([{"Category": t["value"], "Count": t["count"], "% of values": round(t["pct"], 2)}
                                     for t in c.top_categories]), hide_index=True, use_container_width=True)
        right.markdown("**Least frequent**")
        right.dataframe(pd.DataFrame([{"Category": t["value"], "Count": t["count"], "% of values": round(t["pct"], 2)}
                                      for t in c.least_frequent]), hide_index=True, use_container_width=True)
        _show_undefined([f"{x.column}: {x.mode_note}" for x in report.categorical if x.mode_note]
                        + [f"{x.column}: {w}" for x in report.categorical for w in x.warnings], "Category notes")
    if report.boolean:
        st.markdown("**Yes/No columns**")
        _show(pd.DataFrame([{
            "Column": b.column, "Valid": fmt_int(b.valid_count), "Missing": fmt_int(b.missing_count),
            "True": fmt_int(b.true_count), "False": fmt_int(b.false_count),
            "True %": fmt_pct(b.true_pct, 1), "False %": fmt_pct(b.false_pct, 1)} for b in report.boolean]))


def _datetime_section(report: DescriptiveReport) -> None:
    render_section("Date Column Summaries")
    if not report.datetime:
        st.info("This dataset has no date columns.")
        return
    _show(pd.DataFrame([{
        "Column": d.column, "Valid": fmt_int(d.valid_count), "Missing": fmt_int(d.missing_count),
        "Invalid": fmt_int(d.invalid_count), "Earliest": (d.earliest or "—")[:19], "Latest": (d.latest or "—")[:19],
        "Range (days)": fmt_number(d.range_days, 1), "Grouped by": d.period or "—"} for d in report.datetime]))
    pick = st.selectbox("Rows per period for", [d.column for d in report.datetime], key=_key("dt_pick"))
    d = next(x for x in report.datetime if x.column == pick)
    if d.period_counts:
        st.caption(f"Rows per {d.period}.")
        st.dataframe(pd.DataFrame(d.period_counts).rename(columns={"period": d.period.capitalize(), "count": "Rows"}),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("; ".join(d.undefined.values()) or "No period breakdown available.")
    _show_undefined([f"{x.column}: {w}" for x in report.datetime for w in x.warnings], "Date notes")


def _frequency_section(report: DescriptiveReport) -> None:
    render_section("Frequency Distributions")
    options = [c.column for c in report.categorical] + [b.column for b in report.boolean]
    if not options:
        st.info("No categorical or yes/no columns to count.")
        return
    pick = st.selectbox("Column", options, key=_key("freq_pick"))
    cat = next((c for c in report.categorical if c.column == pick), None)
    if cat is not None:
        st.dataframe(pd.DataFrame([{"Value": t["value"], "Count": t["count"], "% of values": round(t["pct"], 2)}
                                   for t in cat.frequency]), hide_index=True, use_container_width=True)
        if cat.frequency_truncated:
            st.caption(f"The table lists the {len(cat.frequency)} most frequent of {cat.unique_count:,} categories.")
    else:
        b = next(x for x in report.boolean if x.column == pick)
        st.dataframe(pd.DataFrame([{"Value": "True", "Count": b.true_count, "% of values": b.true_pct},
                                   {"Value": "False", "Count": b.false_count, "% of values": b.false_pct}]),
                     hide_index=True, use_container_width=True)


def _missing_section(report: DescriptiveReport) -> None:
    render_section("Missing-Value Analysis")
    m = report.missing
    if not m or not m.rows:
        st.info("No columns to analyze.")
        return
    c = st.columns(3)
    c[0].metric("Missing cells", f"{m.missing_cells:,} ({fmt_pct(m.missing_cell_pct, 2)})")
    c[1].metric("Rows with any missing value", f"{m.rows_with_missing:,}")
    c[2].metric("Complete rows", f"{m.complete_rows:,} ({fmt_pct(m.complete_row_pct, 1)})")
    _show(pd.DataFrame([{
        "Column": r["column"], "Type": r["type"], "Missing": fmt_int(r["missing_count"]),
        "Missing %": fmt_pct(r["missing_pct"], 2), "Blank text": fmt_int(r["blank_text"]),
        "Infinite": fmt_int(r["infinite_count"]), "Invalid dates": fmt_int(r["invalid_dates"]),
        "Valid": fmt_int(r["valid_count"])} for r in m.rows]))
    if m.missing_cells == 0:
        st.success("No missing values were found.")


def _outlier_section(report: DescriptiveReport) -> None:
    render_section("Outlier Analysis")
    if not report.outliers:
        st.info("This dataset has no numeric columns to check for outliers.")
        return
    k = report.settings.get("outlier_multiplier", 1.5)
    st.caption(f"Method: values below Q1 − {k:g}×IQR or above Q3 + {k:g}×IQR are flagged. {OUTLIER_DISCLAIMER}")
    rows = []
    for o in report.outliers:
        rows.append({"Column": o.column, "Lower bound": fmt_number(o.lower_bound), "Upper bound": fmt_number(o.upper_bound),
                     "Outliers": fmt_int(o.n_outliers) if o.available else "—",
                     "Outlier %": fmt_pct(o.outlier_pct, 2) if o.available else "—",
                     "Below / above": f"{o.n_low} / {o.n_high}" if o.available else "—",
                     "Note": o.reason or " ".join(o.warnings)})
    _show(pd.DataFrame(rows))
    flagged = [o.column for o in report.outliers if o.n_outliers]
    if flagged:
        pick = st.selectbox("Show flagged values for", flagged, key=_key("out_pick"))
        o = next(x for x in report.outliers if x.column == pick)
        st.dataframe(pd.DataFrame([{"Row": e["row"], "Value": e["value"], "Side": e["side"]} for e in o.examples]),
                     hide_index=True, use_container_width=True)
        if o.examples_truncated:
            st.caption(f"Listing the {len(o.examples)} values furthest outside the bounds; the download lists the same.")
    else:
        st.success("No outliers were flagged by this rule.")


def _correlation_section(report: DescriptiveReport) -> None:
    render_section("Correlation Analysis")
    m = report.correlation
    if m is None:
        st.info("Correlation needs at least two numeric, non-identifier columns.")
        return
    st.caption(f"Method: {m.method}, on rows where both columns have values. Correlation does not show cause and effect.")
    _show(_matrix_display(m))
    _show_undefined([f"{k.replace(' || ', ' vs ')}: {v}" for k, v in m.reasons.items()
                     if k.split(" || ")[0] != k.split(" || ")[1]], "Pairs with an undefined correlation (shown as —)")
    _show_undefined(m.notes, "Column limit")


def _covariance_section(report: DescriptiveReport) -> None:
    render_section("Covariance Analysis")
    if report.covariance_sample is None:
        st.info("Covariance needs at least two numeric, non-identifier columns.")
        return
    kind = st.radio("Covariance type", ["sample", "population"], horizontal=True, key=_key("cov_kind"),
                    format_func=lambda k: "Sample (divide by n − 1)" if k == "sample" else "Population (divide by n)")
    m = report.covariance_population if kind == "population" else report.covariance_sample
    st.caption("Covariance is in the units of the two columns multiplied together, so its size cannot be compared "
               "across pairs; use correlation for that.")
    _show(_matrix_display(m))
    _show_undefined([f"{k.replace(' || ', ' vs ')}: {v}" for k, v in m.reasons.items()
                     if k.split(" || ")[0] != k.split(" || ")[1]], "Pairs with an undefined covariance")


def _chart_section(df: pd.DataFrame, report: DescriptiveReport) -> None:
    render_section("Visualizations")
    names = _unique_names(df.columns)
    chart_df = df.copy(deep=False)
    chart_df.columns = names
    numeric = [n.column for n in report.numeric]
    groupable = [c.column for c in report.categorical if c.unique_count <= ch.MAX_GROUPS] + [b.column for b in report.boolean]
    categorical = [c.column for c in report.categorical] + [b.column for b in report.boolean]
    kind = st.selectbox("Chart type", ch.CHART_TYPES, key=_key("chart_type"))
    result = None
    none = "(none)"

    def need(options, what):
        if not options:
            st.info(f"This chart needs {what}, which this dataset does not have.")
            return False
        return True

    if kind == "Histogram":
        if need(numeric, "a numeric column"):
            col = st.selectbox("Column", numeric, key=_key("h_col"))
            bins = st.slider("Number of bins", ch.MIN_BINS, 100, 20, key=_key("h_bins"))
            result = ch.histogram(chart_df, col, bins)
    elif kind in ("Box plot", "Violin plot", "Density plot"):
        if need(numeric, "a numeric column"):
            col = st.selectbox("Numeric column", numeric, key=_key("d_col"))
            grp = st.selectbox("Group by (optional)", [none] + groupable, key=_key("d_grp"))
            fn = {"Box plot": ch.box_plot, "Violin plot": ch.violin_plot, "Density plot": ch.density_plot}[kind]
            result = fn(chart_df, col, None if grp == none else grp)
    elif kind == "Correlation heatmap":
        result = ch.correlation_heatmap(report.correlation)
    elif kind == "Scatter plot":
        if need(numeric if len(numeric) >= 2 else [], "at least two numeric columns"):
            x = st.selectbox("X axis", numeric, key=_key("s_x"))
            y = st.selectbox("Y axis", [c for c in numeric if c != x], key=_key("s_y"))
            grp = st.selectbox("Colour by (optional)", [none] + groupable, key=_key("s_g"))
            result = ch.scatter_plot(chart_df, x, y, None if grp == none else grp)
    elif kind == "Bar chart":
        if need(categorical, "a categorical column"):
            cat = st.selectbox("Categories", categorical, key=_key("b_cat"))
            agg = st.selectbox("Bar height", ["count", "mean", "median", "sum"], key=_key("b_agg"))
            val = None
            if agg != "count":
                if numeric:
                    val = st.selectbox("Numeric column", numeric, key=_key("b_val"))
                else:
                    st.info("A mean, median or sum needs a numeric column; showing counts instead.")
            result = ch.bar_chart(chart_df, cat, val, agg if val else "count")
    else:  # Frequency distribution
        if need(categorical, "a categorical column"):
            result = ch.frequency_chart(chart_df, st.selectbox("Column", categorical, key=_key("f_col")))

    if result is None:
        return
    if not result.available:
        st.info(result.message)
        return
    fig = result.figure
    # PART 03 chart standard: every chart names both axes (the descriptive engine leaves a few blank).
    if not (fig.layout.xaxis.title.text or "").strip():
        fig.update_xaxes(title_text="Variable" if kind == "Correlation heatmap" else
                         ("All rows (no grouping)" if kind in ("Box plot", "Violin plot") else "Value"))
    if not (fig.layout.yaxis.title.text or "").strip():
        fig.update_yaxes(title_text="Variable" if kind == "Correlation heatmap" else "Value")
    st.plotly_chart(apply_dark_layout(fig, fig.layout.title.text or "", 420), use_container_width=True)
    for note in result.notes:
        st.caption(note)


def _download_section(report: DescriptiveReport) -> None:
    render_section("Download Reports")
    active = session_paths.get_active_dataset() or {}
    label = "NovaMart_demo" if active.get("is_demo") else (active.get("filename") or session_paths.get_dataset_id())
    kind = st.session_state.get(_key("cov_kind"), "sample")
    tables = dx.build_tables(report, label, session_paths.get_dataset_id(), covariance=kind)
    st.caption("The files contain exactly the results shown on this page for the current dataset and settings "
               "(covariance type as selected above).")
    c1, c2, c3 = st.columns(3)
    pick = c1.selectbox("Table", list(tables), key=_key("dl_table"))
    try:
        c1.download_button("⬇️ Download this table (CSV)", data=dx.table_to_csv(tables, pick),
                           file_name=dx.export_stem(label, pick) + ".csv", mime="text/csv", key=_key("dl_csv"))
        c2.download_button("⬇️ Download all tables (CSV, zipped)", data=dx.tables_to_csv_zip(tables),
                           file_name=dx.export_stem(label) + "_csv.zip", mime="application/zip", key=_key("dl_zip"))
    except Exception:
        c1.caption("CSV downloads could not be prepared for this dataset.")
    try:
        c3.download_button("⬇️ Download Excel workbook", data=dx.tables_to_excel(tables),
                           file_name=dx.export_stem(label) + ".xlsx", key=_key("dl_xlsx"),
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        c3.caption(f"Excel unavailable: {exc}" if str(exc) else "Excel download could not be prepared.")


def _methodology_section(report: DescriptiveReport) -> None:
    render_section("Methodology and Statistical Warnings")
    for w in report.warnings:
        st.markdown(f"- {escape_text(w)}")
    with st.expander("How every number is calculated"):
        for topic, text in METHODOLOGY.items():
            st.markdown(f"**{escape_text(topic)}** - {escape_text(text)}")
    render_limitations([
        "Statistics describe THIS dataset only; they say nothing about a wider population (that is Module 15).",
        "Charts of very large tables are drawn from a fixed random sample and say so; tables use all rows.",
        "An outlier flag is not proof of an error.",
    ])


# --------------------------------------------------------------------------
# Entry point (passed to render_module_page as body_fn)
# --------------------------------------------------------------------------

def render_descriptive_body(df: pd.DataFrame, summary: DatasetSummary) -> None:
    st.info("**Descriptive Statistics** is delivered in Part 2 of Phase 8 and is implemented. Every figure on this "
            "page is calculated from the active dataset only.")
    settings = _settings_panel()
    try:
        report = compute_report(df, settings)
    except Exception:
        st.warning("The statistics could not be calculated for this dataset. Check that the data loaded "
                   "correctly, or try another dataset.")
        return
    _quality_section(report)
    _numeric_section(report)
    _categorical_section(report)
    _datetime_section(report)
    _frequency_section(report)
    _missing_section(report)
    _outlier_section(report)
    _correlation_section(report)
    _covariance_section(report)
    _chart_section(df, report)
    _download_section(report)
    _methodology_section(report)
