"""Channel ROI page — attribution table, rankings, spend-vs-contribution.

Works with both legacy demo channel_attribution (full MMM with adstock/saturation)
and generic Ridge-based attribution from the dataset-first pipeline. Column
detection handles both schemas gracefully.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import load_channel_attribution
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from dashboard.styles.theme import apply_chart_style


@safe_page("Channel ROI", required_capability="roi_roas")
def render():
    st.markdown('<div class="mp-header-badge">CHANNEL ROI</div>', unsafe_allow_html=True)
    st.title("Channel ROI & Efficiency")

    attribution = load_channel_attribution()
    if attribution.empty:
        st.warning("Attribution data not found. Run the analysis pipeline first.")
        return

    # Normalize column names: handle both legacy and generic schemas
    col_map = {}
    if "channel" in attribution.columns and "Channel" not in attribution.columns:
        col_map["channel"] = "Channel"
    if "relative_contribution_pct" in attribution.columns and "Contribution_Pct" not in attribution.columns:
        col_map["relative_contribution_pct"] = "Contribution_Pct"
    if col_map:
        attribution = attribution.rename(columns=col_map)

    # Check required columns
    has_channel = "Channel" in attribution.columns
    has_efficiency = "Model_Estimated_Efficiency" in attribution.columns
    has_contribution_pct = "Contribution_Pct" in attribution.columns
    has_spend = "Spend" in attribution.columns
    has_estimated_contribution = "Estimated_Contribution" in attribution.columns
    has_naive_roas = "Observed_Naive_ROAS" in attribution.columns

    if not has_channel:
        st.warning("Channel attribution data is missing the Channel column.")
        return

    # Detect if we have the newer signed-coefficient columns
    has_direction = "Coefficient_Direction" in attribution.columns
    has_spend_rev_ratio = "Spend_Revenue_Ratio" in attribution.columns

    with st.expander("What does this page mean, and why does it matter?"):
        st.markdown(
            "- **Contribution** is the model's estimated association between each channel's spend and "
            "revenue. Positive means higher spend is associated with higher revenue; negative means the "
            "opposite. These are statistical associations from a Ridge regression, **not proof of causal "
            "marketing impact** — confounders, multicollinearity and reverse causality can distort them.\n"
            "- **Efficiency** is the model coefficient — estimated revenue change per unit of spend. A "
            "negative value means additional spend in that channel is associated with *lower* revenue in "
            "the model.\n"
            + ("- **Spend/Revenue Ratio** is total revenue divided by that channel's spend — a naive, "
               "unattributed ratio shown for context only. It is **not** channel-attributed ROAS because "
               "it does not isolate each channel's incremental contribution.\n" if (has_naive_roas or has_spend_rev_ratio) else "")
        )

    # KPI cards (top/bottom/growth)
    if has_efficiency:
        top = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).iloc[0]
        bottom = attribution.sort_values("Model_Estimated_Efficiency", ascending=True).iloc[0]

        kpi_cols = st.columns(3 if has_contribution_pct else 2)
        with kpi_cols[0]:
            st.markdown(f"""<div class="mp-kpi-card"><div class="mp-kpi-label">🏆 Top Performer</div>
                <div class="mp-kpi-value">{top['Channel']}</div>
                <div class="mp-kpi-delta-pos">{top['Model_Estimated_Efficiency']:.2f} est. efficiency</div></div>""",
                unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f"""<div class="mp-kpi-card"><div class="mp-kpi-label">⚠️ Lowest Efficiency</div>
                <div class="mp-kpi-value">{bottom['Channel']}</div>
                <div class="mp-kpi-delta-neg">{bottom['Model_Estimated_Efficiency']:.2f} est. efficiency</div></div>""",
                unsafe_allow_html=True)
        if has_contribution_pct and len(kpi_cols) > 2:
            growth_candidate = attribution.sort_values("Contribution_Pct", ascending=True).iloc[0]
            with kpi_cols[2]:
                st.markdown(f"""<div class="mp-kpi-card"><div class="mp-kpi-label">🎯 Growth Opportunity</div>
                    <div class="mp-kpi-value">{growth_candidate['Channel']}</div>
                    <div class="mp-kpi-delta-pos">Lowest current contribution share</div></div>""",
                    unsafe_allow_html=True)

    # Performance table — show available columns
    st.markdown('<div class="mp-section-title">Channel Performance Table</div>', unsafe_allow_html=True)
    display_cols = []
    rename_map = {}
    for col, label in [
        ("Rank", "Rank"), ("Channel", "Channel"), ("Spend", "Spend"),
        ("Estimated_Contribution", "Contribution"), ("Contribution_Pct", "Contribution %"),
        ("Coefficient", "Coefficient"), ("Coefficient_Direction", "Direction"),
        ("Observed_Naive_ROAS", "Spend/Revenue Ratio"), ("Spend_Revenue_Ratio", "Spend/Revenue Ratio"),
        ("Model_Estimated_Efficiency", "Model Efficiency"),
    ]:
        if col in attribution.columns:
            display_cols.append(col)
            if col != label:
                rename_map[col] = label

    if display_cols:
        display_df = attribution[display_cols].rename(columns=rename_map)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Charts
    if has_efficiency:
        col4, col5 = st.columns(2)
        with col4:
            fig = go.Figure(go.Bar(x=attribution["Channel"], y=attribution["Model_Estimated_Efficiency"],
                                    marker_color=CHART_COLORWAY))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340,
                               title="Efficiency by Channel",
                               margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)
        with col5:
            if has_contribution_pct:
                fig = go.Figure(go.Bar(x=attribution["Channel"], y=attribution["Contribution_Pct"],
                                        marker_color=CHART_COLORWAY[::-1]))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=340, title="Estimated Contribution %",
                                   margin=dict(t=40, l=10, r=10, b=10))
                st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    if has_spend and has_estimated_contribution:
        st.markdown('<div class="mp-section-title">Spend vs Contribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=attribution["Spend"], y=attribution["Estimated_Contribution"], mode="markers+text",
            text=attribution["Channel"], textposition="top center",
            marker=dict(size=14, color=CHART_COLORWAY[:len(attribution)]),
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title="Total Spend",
                           yaxis_title="Estimated Contribution", margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    st.caption("Contribution is a model-estimated decomposition, not an audited or causally-proven figure.")
