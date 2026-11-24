"""Executive Overview page — dynamic, dataset-agnostic.

Phase 3: uses dynamic KPI discovery, automatic chart recommendations,
automatic insight detection, and relationship analysis from the Phase 3
analytics engines.  Falls back to legacy marketing-specific view when
the dataset has marketing capabilities AND no canonical metadata is
available (backward compatibility).

Never hardcodes Revenue, Date, Google Ads, Total_Marketing_Spend, etc.
in the dynamic path.  Every KPI has source, calculation, confidence.
Every chart has score and reason.  Every insight has evidence.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import (
    load_marketing_data, load_kpis, load_channel_attribution,
    load_budget_optimization, load_budget_table,
    load_dynamic_analytics, load_capabilities, has_capability,
    load_generic_kpis,
)
from dashboard.components import chart_labels as CL
from dashboard.components.currency import symbol as _cur_sym
from dashboard.components.kpi_card import render_kpi_row, render_kpi_card, format_inr
from dashboard.components.ui import callout, md_inline
from dashboard.components.data_loader import load_forecast
from dashboard.components import session_paths as _sp_for_overview
from dashboard.styles.theme import apply_chart_style
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


# ---------------------------------------------------------------------------
# Legacy marketing-specific rendering (preserved for marketing datasets)
# ---------------------------------------------------------------------------

def _generate_legacy_insights(kpis, attribution, budget_opt, budget_table):
    """Short, dynamically-generated bullet insights for marketing datasets."""
    insights = []
    if not attribution.empty:
        top = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).iloc[0]
        insights.append(
            f"💡 **{top['Channel']}** is the model's most efficient channel — an estimated "
            f"{_cur_sym()}{top['Model_Estimated_Efficiency']:.2f} of model-estimated revenue per {_cur_sym()}1 spent (an association, not proof of cause)."
        )
    if budget_opt:
        imp = budget_opt.get("expected_improvement_pct", 0)
        if imp > 0.5:
            insights.append(
                f"💡 Reallocating the current budget toward the optimizer's recommendation is projected to "
                f"lift predicted weekly revenue by **{imp:+.1f}%** — with no increase in total spend."
            )
    if not budget_table.empty:
        biggest_shift = budget_table.reindex(budget_table["Change"].abs().sort_values(ascending=False).index).iloc[0]
        direction = "increasing" if biggest_shift["Change"] > 0 else "decreasing"
        insights.append(
            f"💡 The single largest recommended move is **{direction} {biggest_shift['Channel']}** by "
            f"{format_inr(abs(biggest_shift['Change']), compact=False)} ({biggest_shift['Change_Pct']:+.1f}%)."
        )
    growth = kpis.get("revenue_growth_h1_to_h2")
    if growth is not None:
        direction = "grown" if growth >= 0 else "declined"
        insights.append(f"💡 Revenue has **{direction} {abs(growth)*100:.1f}%** comparing the first half to the "
                         f"second half of the observed period.")
    return insights


def _render_legacy_overview():
    """Full legacy marketing-specific overview (unchanged from pre-Phase 3)."""
    st.markdown('<div class="mp-header-badge">MARKETPULSE AI · EXECUTIVE OVERVIEW</div>', unsafe_allow_html=True)
    st.title("What's happening, why, and what to do next")
    st.caption("Revenue, spend, model-estimated channel effects, the revenue forecast and the optimizer's "
               "recommendation for the active dataset — on one page. Active dataset: NovaMart (demo, synthetic).")

    df = load_marketing_data()
    kpis = load_kpis().get("kpis", {})
    attribution = load_channel_attribution()
    budget_opt = load_budget_optimization()
    budget_table = load_budget_table()

    overall_roas = kpis.get("overall_naive_roas", 0)
    revenue_growth = kpis.get("revenue_growth_h1_to_h2", 0) * 100
    predicted_rev = budget_opt.get("current_predicted_revenue", 0)
    optimized_rev = budget_opt.get("optimized_predicted_revenue", 0)
    improvement = budget_opt.get("expected_improvement_pct", 0)

    has_growth = "revenue_growth_h1_to_h2" in kpis
    has_mkt_growth = "marketing_growth_h1_to_h2" in kpis
    render_kpi_row([
        {"label": "Total Revenue", "value": format_inr(kpis.get("total_revenue", 0)),
         "delta": f"{revenue_growth:+.1f}% 2nd half vs 1st half" if has_growth else None,
         "positive": revenue_growth >= 0, "help": "Whole observed period"},
        {"label": "Marketing Spend", "value": format_inr(kpis.get("total_marketing_spend", 0)),
         "delta": (f"{kpis.get('marketing_growth_h1_to_h2', 0)*100:+.1f}% 2nd half vs 1st half"
                   if has_mkt_growth else None),
         "positive": kpis.get("marketing_growth_h1_to_h2", 0) >= 0, "help": "Whole observed period"},
        {"label": "Overall ROAS", "value": f"{overall_roas:.2f}x",
         "help": "Total revenue ÷ total marketing spend (not incremental)"},
        {"label": "Optimizer Uplift (est.)", "value": f"{improvement:+.1f}%",
         "help": "Model-estimated weekly revenue change at the same total spend; not guaranteed"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    if predicted_rev and optimized_rev:
        render_kpi_row([
            {"label": "Predicted weekly revenue · current mix", "value": format_inr(predicted_rev, compact=False),
             "help": "Marketing Mix Model estimate for one week"},
            {"label": "Predicted weekly revenue · recommended mix", "value": format_inr(optimized_rev, compact=False),
             "help": "Same total budget, reallocated by the optimizer — an estimate, not a guarantee"},
        ])

    st.markdown('<div class="mp-section-title">Revenue & Marketing Spend Trend</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Revenue"], name="Revenue",
                                  line=dict(color=CHART_COLORWAY[0], width=2)))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, l=10, r=10, b=10),
                           title="Weekly Revenue Trend")
        st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(df["Date"]),
                                          y_title=CL.value_axis_title("Revenue", monetary=True), currency="y"),
                        use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Total_Marketing_Spend"], name="Marketing Spend",
                                  line=dict(color=CHART_COLORWAY[1], width=2)))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, l=10, r=10, b=10),
                           title="Weekly Marketing Spend Trend")
        st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(df["Date"]),
                                          y_title=CL.value_axis_title("Total_Marketing_Spend", monetary=True),
                                          currency="y"),
                        use_container_width=True)

    st.markdown('<div class="mp-section-title">Channel Contribution & Budget Distribution</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        if not attribution.empty:
            fig = go.Figure(go.Pie(labels=attribution["Channel"], values=attribution["Estimated_Contribution"],
                                    hole=0.55, marker=dict(colors=CHART_COLORWAY), sort=True,
                                    textposition="inside", texttemplate="%{percent:.0%}",
                                    hovertemplate="%{label}<br>" + _cur_sym() + "%{value:,.0f} (%{percent})<extra></extra>"))
            fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")  # small slices: hover only
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10),
                               title="Model-Estimated Revenue Contribution by Channel",
                               legend_title_text="Marketing channel")
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)
    with col4:
        if not attribution.empty:
            spend_dist = attribution.sort_values("Spend", ascending=False)
            fig = go.Figure(go.Bar(y=spend_dist["Channel"], x=spend_dist["Spend"], orientation="h",
                                    marker_color=CHART_COLORWAY[0],
                                    hovertemplate="%{y}: " + _cur_sym() + "%{x:,.0f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10),
                               title="Current Marketing Budget Distribution")
            st.plotly_chart(apply_chart_style(fig, currency="x", y_title="Marketing channel",
                                              x_title=CL.value_axis_title("Spend", "sum", monetary=True)
                                              + " over the period"),
                            use_container_width=True)

    if not budget_table.empty:
        st.markdown('<div class="mp-section-title">Current vs Optimized Budget</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=budget_table["Channel"], y=budget_table["Current_Budget"],
                              marker_color=CHART_COLORWAY[0]))
        fig.add_trace(go.Bar(name="AI Recommended", x=budget_table["Channel"], y=budget_table["Recommended_Budget"],
                              marker_color=CHART_COLORWAY[2]))
        fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=340, margin=dict(t=20, l=10, r=10, b=10),
                          title="Current vs Optimized Weekly Budget by Channel",
                          legend=dict(orientation="h", y=1.08, x=0, title_text="Allocation"))
        st.plotly_chart(apply_chart_style(fig, currency="y", x_title="Marketing channel",
                                          y_title=CL.value_axis_title("Weekly budget", monetary=True)),
                        use_container_width=True)
        st.caption("Full channel-by-channel numbers and a live re-run of the optimizer are on the Marketing "
                   "Resource Optimizer page.")

    st.markdown('<div class="mp-section-title">Top Performing Channels</div>', unsafe_allow_html=True)
    if not attribution.empty:
        top3 = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).head(3)
        cols = st.columns(3)
        for col, (_, row) in zip(cols, top3.iterrows()):
            with col:
                st.markdown(
                    f"""<div class="mp-kpi-card">
                        <div class="mp-kpi-label">{row['Channel']}</div>
                        <div class="mp-kpi-value">{_cur_sym()}{row['Model_Estimated_Efficiency']:.2f}</div>
                        <div class="mp-kpi-delta-pos">est. revenue per {_cur_sym()}1 spent</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    _render_legacy_forecast_overview()

    st.markdown('<div class="mp-section-title">Key business insights</div>', unsafe_allow_html=True)
    st.caption("Generated from the computed results above (rule-based). Model figures are associations "
               "estimated from historical data, not proven causal effects.")
    insights = _generate_legacy_insights(kpis, attribution, budget_opt, budget_table)
    for insight in insights:
        callout(insight)

    st.markdown('<div class="mp-section-title">Data quality</div>', unsafe_allow_html=True)
    _render_legacy_data_quality(df)

    with st.expander("What does this page mean, and how should management use it?"):
        st.markdown(
            "- **What's happening:** the KPI cards and trend charts above summarize actual revenue and spend "
            "over the observed period.\n"
            "- **Why it's happening:** the pie/bar charts show how the fitted Marketing Mix Model attributes "
            "revenue across channels, and how current budget is actually distributed.\n"
            "- **What to do:** the *Potential Improvement* figure and the Current-vs-Optimized chart come "
            "from a real optimization run against the fitted model — see the Marketing Resource Optimizer page to adjust "
            "assumptions and re-run it live."
        )


def _render_legacy_forecast_overview():
    """Compact forecast overview: next periods + how the model compares with
    a naive baseline in walk-forward testing."""
    forecast = load_forecast()
    if forecast.empty or "Forecast_Revenue" not in forecast.columns:
        return
    st.markdown('<div class="mp-section-title">Forecast overview</div>', unsafe_allow_html=True)
    import json as _json
    ev_path = _sp_for_overview.get_processed_dir() / "forecast_evaluation.json"
    ev = {}
    try:
        ev = _json.loads(ev_path.read_text()) if ev_path.exists() else {}
    except (OSError, ValueError):
        ev = {}
    nxt = float(forecast["Forecast_Revenue"].iloc[0])
    total = float(forecast["Forecast_Revenue"].sum())
    lo = float(forecast["Lower_90"].sum()) if "Lower_90" in forecast.columns else None
    hi = float(forecast["Upper_90"].sum()) if "Upper_90" in forecast.columns else None
    imp = ev.get("improvement_vs_baseline_pct")
    render_kpi_row([
        {"label": "Next period (forecast)", "value": format_inr(nxt, compact=False),
         "help": "Estimate, not a guarantee"},
        {"label": f"Next {len(forecast)} periods (total)", "value": format_inr(total),
         "help": f"90% range {format_inr(lo)} – {format_inr(hi)}" if lo is not None and hi is not None else None},
        {"label": "Walk-forward accuracy vs naive", "value": f"{imp:+.1f}% MAE" if imp is not None else "—",
         "help": "Positive = the model beats a naive 'last value' forecast on unseen periods"},
    ])


def _render_legacy_data_quality(df):
    """Data-quality indicators for the active (demo) data."""
    from src.data_engine.data_quality import assess_quality
    rep = assess_quality(df)
    render_kpi_row([
        {"label": "Data quality score", "value": f"{rep['score']:.1f} / 100", "help": "Completeness, validity, "
         "uniqueness, consistency, type correctness"},
        {"label": "Missing cells", "value": f"{rep['missing_pct']:.2f}%"},
        {"label": "Duplicate rows", "value": f"{rep['duplicate_rows']:,}"},
        {"label": "Rows × columns", "value": f"{rep['n_rows']:,} × {rep['n_cols']}"},
    ])
    st.caption("The demo dataset is synthetic and complete by construction; uploaded datasets show their own "
               "quality report on the Upload Data page.")


# ---------------------------------------------------------------------------
# Dynamic dataset-agnostic rendering (Phase 3)
# ---------------------------------------------------------------------------

def _kpi_display_value(kpi: dict) -> str:
    """KPI value with the dataset currency when the KPI is a sum or average of a
    monetary column and the currency is known (PART 01); counts, rates and
    percentages never get a currency."""
    text = kpi.get("formatted_value") or str(kpi.get("value", ""))
    calc = str(kpi.get("calculation", "")).lower()
    col = kpi.get("source_column")
    from dashboard.components import currency as _cur
    if (calc.startswith(("sum(", "mean(")) and not kpi.get("unit") and col and _cur.is_monetary(col)
            and not CL.is_percent(col) and _cur.symbol()):
        return f"{_cur.symbol()}{text}"
    return text


def _render_dynamic_kpis(analytics: dict) -> None:
    """Render dynamically discovered KPIs as cards."""
    kpis = analytics.get("kpis", [])
    if not kpis:
        st.info("No KPIs could be discovered for this dataset.")
        return

    # Sort by priority (lower = more important)
    kpis_sorted = sorted(kpis, key=lambda k: k.get("priority", 999))

    # Show top 4 as primary KPI row
    primary = kpis_sorted[:4]
    cards = []
    for kpi in primary:
        card = {
            "label": kpi["label"],
            "value": _kpi_display_value(kpi),
        }
        if kpi.get("delta"):
            card["delta"] = kpi["delta"]
            card["positive"] = kpi.get("delta_positive", True)
        cards.append(card)
    render_kpi_row(cards)

    # Show remaining KPIs in a second row if present
    secondary = kpis_sorted[4:8]
    if secondary:
        st.markdown("<br>", unsafe_allow_html=True)
        cards2 = []
        for kpi in secondary:
            card = {
                "label": kpi["label"],
                "value": _kpi_display_value(kpi),
            }
            if kpi.get("delta"):
                card["delta"] = kpi["delta"]
                card["positive"] = kpi.get("delta_positive", True)
            cards2.append(card)
        render_kpi_row(cards2)

    # Expandable detail: all KPIs with explanation
    with st.expander("📊 KPI Details & Methodology"):
        for kpi in kpis_sorted:
            conf = kpi.get("confidence", 0)
            conf_label = "🟢" if conf >= 0.8 else ("🟡" if conf >= 0.5 else "🔴")
            st.markdown(
                f"**{kpi['label']}** = {_kpi_display_value(kpi)} "
                f"&nbsp;&nbsp;{conf_label} confidence {conf:.0%}\n\n"
                f"*Calculation:* `{kpi.get('calculation', 'N/A')}` &nbsp;|&nbsp; "
                f"*Source:* `{kpi.get('source_column', 'N/A')}` &nbsp;|&nbsp; "
                f"*Category:* {kpi.get('category', 'N/A')}\n\n"
                f"{kpi.get('explanation', '')}\n\n---"
            )


def _render_dynamic_charts(analytics: dict, df) -> None:
    """Render top recommended visualizations using Plotly."""
    charts = analytics.get("charts", [])
    if not charts:
        return

    # Take top 4 charts
    top_charts = sorted(charts, key=lambda c: -c.get("score", 0))[:4]

    for i in range(0, len(top_charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(top_charts):
                break
            chart = top_charts[idx]
            with col:
                _render_single_chart(chart, df)


def _per_period(df, x_col: str, y_col: str, aggregation: str = "none"):
    """One point per time value. Row-level data with several rows per date (orders,
    stores) would otherwise draw a zig-zag through every row. Sums by default;
    'mean'/'avg' aggregation hints average instead."""
    import pandas as pd
    plot_df = pd.DataFrame({x_col: df[x_col], y_col: pd.to_numeric(df[y_col], errors="coerce")}).dropna()
    if plot_df[x_col].duplicated().any():
        how = "mean" if str(aggregation).lower() in ("mean", "avg", "average") else "sum"
        plot_df = plot_df.groupby(x_col, as_index=False)[y_col].agg(how)
    return plot_df.sort_values(x_col)


def _effective_aggregation(chart_type: str, df, x_col, aggregation) -> str:
    """The aggregation the renderer below really applies (labels must match it)."""
    agg = str(aggregation or "none").lower()
    if chart_type in ("line", "area"):
        if x_col in df.columns and df[x_col].duplicated().any():
            return "mean" if agg in ("mean", "avg", "average") else "sum"
        return "none"
    if chart_type == "bar":
        return agg if agg in ("count", "sum", "mean") else "mean"
    if chart_type == "horizontal_bar":
        return "sum" if agg == "sum" else "mean"
    return "none"


def _axis_titles(chart_type, df, x_col, y_col, aggregation, title):
    """(x title, y title, chart title) for a recommended chart (upgrade PART 01:
    every chart names what each axis shows, with units only when known)."""
    agg = _effective_aggregation(chart_type, df, x_col, aggregation)
    if chart_type in ("line", "area"):
        is_time = x_col in df.columns and (pd.api.types.is_datetime64_any_dtype(df[x_col]) or
                                           pd.to_datetime(df[x_col], errors="coerce").notna().mean() > 0.9)
        xt = CL.time_axis_title(df[x_col]) if is_time else CL.category_axis_title(x_col)
        return xt, CL.value_axis_title(y_col, agg), CL.chart_title(title, agg, y_col)
    if chart_type == "bar":
        if agg == "count":
            return CL.category_axis_title(x_col), "Number of records", title
        return CL.category_axis_title(x_col), CL.value_axis_title(y_col, agg), CL.chart_title(title, agg, y_col)
    if chart_type == "horizontal_bar":
        return CL.value_axis_title(x_col, agg), CL.category_axis_title(y_col), CL.chart_title(title, agg, x_col)
    if chart_type == "scatter":
        return CL.value_axis_title(x_col), CL.value_axis_title(y_col), title
    if chart_type == "histogram":
        return CL.value_axis_title(x_col), "Number of records", title
    if chart_type == "box":
        return CL.category_axis_title(x_col), CL.value_axis_title(y_col), title
    if chart_type == "heatmap":
        return "Variable", "Variable", title
    return None, None, title           # donut: no axes


def _render_single_chart(chart: dict, df) -> None:
    """Render a single chart recommendation as a Plotly figure."""
    chart_type = chart.get("chart_type", "bar")
    title = chart.get("title", "Chart")
    x_col = chart.get("x_column")
    y_col = chart.get("y_column")
    y_cols = chart.get("y_columns")
    aggregation = chart.get("aggregation", "none")
    top_n = chart.get("top_n")

    try:
        fig = go.Figure()

        if chart_type == "line" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                plot_df = _per_period(df, x_col, y_col, aggregation)
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col], y=plot_df[y_col],
                    name=y_col, mode="lines",
                    line=dict(color=CHART_COLORWAY[0], width=2),
                ))

        elif chart_type == "area" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                plot_df = _per_period(df, x_col, y_col, aggregation)
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col], y=plot_df[y_col],
                    name=y_col, fill="tozeroy", mode="lines",
                    line=dict(color=CHART_COLORWAY[1], width=2),
                ))

        elif chart_type == "bar" and x_col:
            if x_col in df.columns:
                if aggregation == "count":
                    counts = df[x_col].value_counts(dropna=True)
                    if top_n:
                        counts = counts.head(top_n)
                    fig.add_trace(go.Bar(
                        x=counts.index.astype(str), y=counts.values,
                        marker_color=CHART_COLORWAY[2],
                    ))
                elif y_col and y_col in df.columns:
                    import pandas as pd
                    if aggregation == "mean":
                        grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                    elif aggregation == "sum":
                        grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                    else:
                        grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                    if top_n:
                        grouped = grouped.head(top_n)
                    fig.add_trace(go.Bar(
                        x=grouped.index.astype(str), y=grouped.values,
                        marker_color=CHART_COLORWAY[2],
                    ))

        elif chart_type == "horizontal_bar" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                import pandas as pd
                if aggregation == "sum":
                    grouped = df.groupby(y_col)[x_col].sum().sort_values(ascending=True)
                else:
                    grouped = df.groupby(y_col)[x_col].mean().sort_values(ascending=True)
                if top_n:
                    grouped = grouped.tail(top_n)
                fig.add_trace(go.Bar(
                    y=grouped.index.astype(str), x=grouped.values,
                    orientation="h", marker_color=CHART_COLORWAY[3],
                ))

        elif chart_type == "scatter" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                plot_df = df[[x_col, y_col]].dropna()
                fig.add_trace(go.Scatter(
                    x=plot_df[x_col], y=plot_df[y_col],
                    mode="markers", name=f"{x_col} vs {y_col}",
                    marker=dict(color=CHART_COLORWAY[4], size=6, opacity=0.7),
                ))

        elif chart_type == "histogram" and x_col:
            if x_col in df.columns:
                import pandas as pd
                series = pd.to_numeric(df[x_col], errors="coerce").dropna()
                fig.add_trace(go.Histogram(
                    x=series, nbinsx=30,
                    marker_color=CHART_COLORWAY[5],
                ))

        elif chart_type == "donut" and x_col:
            if x_col in df.columns:
                counts = df[x_col].value_counts(dropna=True).head(8)
                fig = go.Figure(go.Pie(
                    labels=counts.index.astype(str), values=counts.values,
                    hole=0.45, marker=dict(colors=CHART_COLORWAY),
                    textposition="inside", texttemplate="%{percent:.0%}",
                ))
                fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")

        elif chart_type == "box" and x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                fig.add_trace(go.Box(
                    x=df[x_col].astype(str), y=df[y_col],
                    marker_color=CHART_COLORWAY[6],
                ))

        elif chart_type == "heatmap" and y_cols:
            import pandas as pd
            valid_cols = [c for c in y_cols if c in df.columns]
            if len(valid_cols) >= 2:
                numeric_df = df[valid_cols].apply(pd.to_numeric, errors="coerce")
                corr = numeric_df.corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale="RdBu_r", zmin=-1, zmax=1,
                    text=corr.values.round(2), texttemplate="%{text}",
                ))

        x_title, y_title, title = _axis_titles(chart_type, df, x_col, y_col, aggregation, title)
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            margin=dict(t=30, l=10, r=10, b=10),
            title=dict(text=title, font=dict(size=14)),
        )
        if chart_type == "donut" and x_col:
            fig.update_layout(legend_title_text=CL.category_axis_title(x_col))
        if chart_type == "heatmap":
            fig.update_traces(colorbar=dict(title=dict(text="Correlation (r)")))
        st.plotly_chart(apply_chart_style(fig, x_title=x_title, y_title=y_title), use_container_width=True)

        # Score badge
        score = chart.get("score", 0)
        st.caption(f"Score: {score:.2f} — {chart.get('reason', '')}")

    except Exception as e:
        st.warning(f"Could not render chart '{title}': {e}")


def _render_dynamic_insights(analytics: dict) -> None:
    """Render automatically discovered insights."""
    insights = analytics.get("insights", [])
    if not insights:
        return

    severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢", "info": "ℹ️"}

    for insight in insights[:8]:
        icon = severity_icons.get(insight.get("severity", "info"), "ℹ️")
        conf = insight.get("confidence", 0)

        st.markdown(
            f'<div class="mp-callout" style="margin-bottom:8px;">'
            f'{icon} <b>{md_inline(insight.get("title", ""))}</b> (confidence: {conf:.0%})<br>'
            f'{md_inline(insight.get("description", ""))}<br>'
            f'<small style="color: var(--mp-text-muted);">Evidence: {md_inline(insight.get("evidence", "N/A"))}</small>'
            f'</div>', unsafe_allow_html=True)


def _render_dynamic_relationships(analytics: dict) -> None:
    """Render detected relationships."""
    relationships = analytics.get("relationships", [])
    if not relationships:
        return

    for rel in relationships[:5]:
        cols_str = " ↔ ".join(rel.get("columns", []))
        strength = rel.get("strength", 0)
        strength_bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))

        st.markdown(
            f"**{cols_str}** — {rel.get('relationship_type', 'unknown')} "
            f"({rel.get('direction', '')})\n\n"
            f"`{strength_bar}` strength: {strength:.3f}\n\n"
            f"{rel.get('description', '')}\n\n"
            f"*Evidence:* {rel.get('evidence', 'N/A')}\n\n---"
        )


def _load_quality_report():
    import json
    path = _sp_for_overview.get_processed_dir() / "data_quality_report.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def _render_data_quality_summary(analytics: dict) -> None:
    """Data-quality summary. Uses the data-quality engine's report written at upload
    time (same score as the Upload page); falls back to the KPI engine's estimate."""
    rep = _load_quality_report()
    if rep and isinstance(rep.get("original"), dict):
        orig, clean, audit = rep["original"], rep.get("cleaned") or {}, rep.get("audit") or {}
        render_kpi_row([
            {"label": "Data quality · uploaded file", "value": f"{orig.get('score', 0):.1f} / 100",
             "help": "Completeness 30% · validity 25% · uniqueness 15% · consistency 15% · types 15%"},
            {"label": "Data quality · after cleaning", "value": f"{clean.get('score', orig.get('score', 0)):.1f} / 100",
             "help": "Imputed cells are not counted as complete, so cleaning cannot inflate the score."},
            {"label": "Missing cells (original)", "value": f"{orig.get('missing_pct', 0):.2f}%"},
            {"label": "Cells imputed · rows removed",
             "value": f"{audit.get('total_imputed_cells', 0):,} · {audit.get('total_rows_removed', 0):,}"},
        ])
        st.caption("Full column-level report and cleaning audit: Upload Data page (shown when the file is "
                   "loaded) and data_quality_report.json in the dataset's processed folder.")
        return
    kpis = analytics.get("kpis", [])
    quality_kpi = next((k for k in kpis if k.get("category") == "quality"), None)
    if quality_kpi:
        score = quality_kpi.get("value", 100)
        color = "#16A34A" if score >= 80 else ("#D97706" if score >= 50 else "#DC2626")
        st.markdown(
            f'<div class="mp-kpi-card" style="display:inline-block; margin-right:16px;">'
            f'<div class="mp-kpi-label">Data Quality Score</div>'
            f'<div class="mp-kpi-value" style="color:{color}">{score:.1f}%</div>'
            f'<div style="color:#6B7280; font-size:12px;">{quality_kpi.get("explanation", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_capability_summary() -> None:
    """Show which analysis capabilities are available/unavailable."""
    caps = load_capabilities()
    if not caps:
        return

    available = []
    unavailable = []
    for key, entry in caps.items():
        if isinstance(entry, dict):
            name = entry.get("name", key)
            if entry.get("available", False):
                available.append(name)
            else:
                unavailable.append((name, entry.get("reason", "")))

    if available:
        st.success(f"**Available:** {', '.join(available)}")
    if unavailable:
        reasons = "; ".join(f"{n} ({r})" for n, r in unavailable[:5])
        st.info(f"**Not available for this dataset:** {reasons}")


def _render_dynamic_overview():
    """Full dynamic, dataset-agnostic executive overview."""
    analytics = load_dynamic_analytics()
    df = load_marketing_data()

    st.markdown('<div class="mp-header-badge">MARKETPULSE AI · EXECUTIVE OVERVIEW</div>', unsafe_allow_html=True)
    st.title("Executive Overview")
    st.caption("AI-driven automatic analysis — KPIs, trends, and insights discovered dynamically from your data.")

    # Section 1: Data Quality Summary
    st.markdown('<div class="mp-section-title">📋 Data Quality</div>', unsafe_allow_html=True)
    _render_data_quality_summary(analytics)

    # Section 2: Primary KPIs
    st.markdown('<div class="mp-section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    _render_dynamic_kpis(analytics)

    # Section 3: Recommended Visualizations
    st.markdown('<div class="mp-section-title">📈 Recommended Visualizations</div>', unsafe_allow_html=True)
    if df is not None and not df.empty:
        _render_dynamic_charts(analytics, df)
    else:
        st.info("No data available for visualization.")

    # Section 4: Key Insights
    st.markdown('<div class="mp-section-title">🤖 AI-Discovered Insights</div>', unsafe_allow_html=True)
    _render_dynamic_insights(analytics)

    # Section 5: Relationships
    with st.expander("🔗 Detected Relationships & Correlations"):
        _render_dynamic_relationships(analytics)

    # Section 6: Available Analysis Capabilities
    with st.expander("🧪 Analysis Capabilities"):
        _render_capability_summary()

    # Explainer
    with st.expander("What does this page show, and how was it generated?"):
        st.markdown(
            "This executive overview is **automatically generated** from your dataset. "
            "Every KPI, chart, insight, and relationship was discovered by analyzing "
            "the data's structure, types, and statistical properties — no column names "
            "were assumed or hardcoded.\n\n"
            "- **KPIs** are discovered by identifying target/measure columns, computing "
            "totals, averages, growth rates, and data quality metrics.\n"
            "- **Charts** are recommended based on column types and relationships "
            "(time + numeric → line; category + numeric → bar; etc.).\n"
            "- **Insights** detect trends, concentration, outliers, correlations, and "
            "data quality issues using statistical tests.\n"
            "- **Relationships** find Pearson correlations, group comparisons, and "
            "time-based trends.\n\n"
            "Each element includes its confidence level, evidence, and methodology."
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@safe_page("Executive Overview")
def render():
    """Route to dynamic or legacy overview based on data availability."""
    # Try dynamic analytics first (Phase 3 path)
    analytics = load_dynamic_analytics()
    if analytics and analytics.get("kpis"):
        _render_dynamic_overview()
        return

    # Check if this is a marketing dataset with legacy capabilities
    if has_capability("marketing_mix_modeling") or has_capability("channel_attribution"):
        try:
            _render_legacy_overview()
            return
        except (KeyError, Exception):
            pass

    # Final fallback: dynamic overview even with limited analytics
    _render_dynamic_overview()


# app.py imports this module and calls render() itself; rendering again at
# import time drew the page twice on its first visit (duplicate elements).
if __name__ == "__main__":
    render()
