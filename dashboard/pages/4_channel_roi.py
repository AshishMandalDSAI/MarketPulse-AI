"""Channel ROI page — attribution table, rankings, spend-vs-contribution."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import load_channel_attribution
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


def render():
    st.markdown('<div class="mp-header-badge">CHANNEL ROI</div>', unsafe_allow_html=True)
    st.title("Channel ROI & Efficiency")

    attribution = load_channel_attribution()
    if attribution.empty:
        st.warning("Attribution data not found. Run `python src/models/channel_contribution.py` first.")
        return

    with st.expander("What does this page mean, and why does it matter?"):
        st.markdown(
            "- **Contribution** is how much of total revenue the fitted model attributes to each channel.\n"
            "- **Efficiency** is estimated revenue generated per ₹1 spent — the number to watch when "
            "deciding *where the next rupee should go*, not just which channel spends the most.\n"
            "- **Naive ROAS** is a simple, undecomposed ratio shown for comparison; it can diverge from "
            "model efficiency because it ignores carryover and diminishing-returns effects."
        )

    top = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).iloc[0]
    bottom = attribution.sort_values("Model_Estimated_Efficiency", ascending=True).iloc[0]
    growth_candidate = attribution.sort_values("Contribution_Pct", ascending=True).iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="mp-kpi-card"><div class="mp-kpi-label">🏆 Top Performer</div>
            <div class="mp-kpi-value">{top['Channel']}</div>
            <div class="mp-kpi-delta-pos">₹{top['Model_Estimated_Efficiency']:.2f} est. revenue / ₹1</div></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="mp-kpi-card"><div class="mp-kpi-label">⚠️ Lowest Efficiency</div>
            <div class="mp-kpi-value">{bottom['Channel']}</div>
            <div class="mp-kpi-delta-neg">₹{bottom['Model_Estimated_Efficiency']:.2f} est. revenue / ₹1</div></div>""",
            unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="mp-kpi-card"><div class="mp-kpi-label">🎯 Growth Opportunity</div>
            <div class="mp-kpi-value">{growth_candidate['Channel']}</div>
            <div class="mp-kpi-delta-pos">Lowest current contribution share</div></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="mp-section-title">Channel Performance Table</div>', unsafe_allow_html=True)
    display_df = attribution[[
        "Rank", "Channel", "Spend", "Estimated_Contribution", "Contribution_Pct",
        "Observed_Naive_ROAS", "Model_Estimated_Efficiency",
    ]].rename(columns={
        "Estimated_Contribution": "Contribution (₹)", "Contribution_Pct": "Contribution %",
        "Observed_Naive_ROAS": "Naive ROAS", "Model_Estimated_Efficiency": "Model Efficiency",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col4, col5 = st.columns(2)
    with col4:
        fig = go.Figure(go.Bar(x=attribution["Channel"], y=attribution["Model_Estimated_Efficiency"],
                                marker_color=CHART_COLORWAY))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, title="Efficiency by Channel (est. ₹ revenue per ₹1 spent)",
                           margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col5:
        fig = go.Figure(go.Bar(x=attribution["Channel"], y=attribution["Contribution_Pct"],
                                marker_color=CHART_COLORWAY[::-1]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, title="Estimated Contribution %",
                           margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Spend vs Contribution</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Scatter(
        x=attribution["Spend"], y=attribution["Estimated_Contribution"], mode="markers+text",
        text=attribution["Channel"], textposition="top center",
        marker=dict(size=14, color=CHART_COLORWAY[:len(attribution)]),
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title="Total Spend (₹)",
                       yaxis_title="Estimated Contribution (₹)", margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Contribution is a model-estimated decomposition (leave-one-channel-out ablation on the fitted "
               "MMM), not an audited or causally-proven figure. Naive ROAS is a simple, undecomposed ratio "
               "shown for comparison.")
