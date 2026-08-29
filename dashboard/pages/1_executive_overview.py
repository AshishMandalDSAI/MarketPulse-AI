"""Executive Overview page — the first thing a manager/professor sees in a demo."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import (
    load_marketing_data, load_kpis, load_channel_attribution,
    load_budget_optimization, load_budget_table,
)
from dashboard.components.kpi_card import render_kpi_row, format_inr
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


def _generate_insights(kpis, attribution, budget_opt, budget_table):
    """Short, dynamically-generated bullet insights — no hard-coded text,
    every number below is read from this session's actual computed outputs."""
    insights = []
    if not attribution.empty:
        top = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).iloc[0]
        insights.append(
            f"💡 **{top['Channel']}** is the model's most efficient channel — an estimated "
            f"₹{top['Model_Estimated_Efficiency']:.2f} of incremental revenue per ₹1 spent."
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


def render():
    st.markdown('<div class="mp-header-badge">MARKETPULSE AI · EXECUTIVE OVERVIEW</div>', unsafe_allow_html=True)
    st.title("What's happening, why, and what to do next")
    st.caption("AI-Driven Marketing Mix Modeling & Budget Optimization — Turn Marketing Spend into Smarter "
               "Business Decisions.")

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

    render_kpi_row([
        {"label": "Total Revenue", "value": format_inr(kpis.get("total_revenue", 0)),
         "delta": f"{revenue_growth:+.1f}% H1→H2", "positive": revenue_growth >= 0},
        {"label": "Marketing Spend", "value": format_inr(kpis.get("total_marketing_spend", 0)),
         "delta": f"{kpis.get('marketing_growth_h1_to_h2', 0)*100:+.1f}% H1→H2",
         "positive": kpis.get("marketing_growth_h1_to_h2", 0) >= 0},
        {"label": "Overall ROAS", "value": f"{overall_roas:.2f}x"},
        {"label": "Potential Improvement", "value": f"{improvement:+.1f}%",
         "delta": "via budget optimization", "positive": improvement >= 0},
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    if predicted_rev and optimized_rev:
        render_kpi_row([
            {"label": "Model Predicted Revenue (current mix)", "value": format_inr(predicted_rev, compact=False)},
            {"label": "Optimized Revenue (AI recommended mix)", "value": format_inr(optimized_rev, compact=False)},
        ])

    st.markdown('<div class="mp-section-title">Revenue & Marketing Spend Trend</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Revenue"], name="Revenue",
                                  line=dict(color=CHART_COLORWAY[0], width=2)))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, l=10, r=10, b=10),
                           title="Weekly Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Total_Marketing_Spend"], name="Marketing Spend",
                                  line=dict(color=CHART_COLORWAY[1], width=2)))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, l=10, r=10, b=10),
                           title="Weekly Marketing Spend Trend")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Channel Contribution & Budget Distribution</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        if not attribution.empty:
            fig = go.Figure(go.Pie(labels=attribution["Channel"], values=attribution["Estimated_Contribution"],
                                    hole=0.45, marker=dict(colors=CHART_COLORWAY)))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10),
                               title="Model-Estimated Revenue Contribution by Channel")
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if not attribution.empty:
            spend_dist = attribution.sort_values("Spend", ascending=False)
            fig = go.Figure(go.Bar(x=spend_dist["Channel"], y=spend_dist["Spend"],
                                    marker_color=CHART_COLORWAY[2]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10),
                               title="Current Marketing Budget Distribution")
            st.plotly_chart(fig, use_container_width=True)

    if not budget_table.empty:
        st.markdown('<div class="mp-section-title">Current vs Optimized Budget</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=budget_table["Channel"], y=budget_table["Current_Budget"],
                              marker_color=CHART_COLORWAY[0]))
        fig.add_trace(go.Bar(name="AI Recommended", x=budget_table["Channel"], y=budget_table["Recommended_Budget"],
                              marker_color=CHART_COLORWAY[2]))
        fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=340, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Full channel-by-channel numbers and a live re-run of the optimizer are on the Budget "
                   "Optimizer page.")

    st.markdown('<div class="mp-section-title">Top Performing Channels</div>', unsafe_allow_html=True)
    if not attribution.empty:
        top3 = attribution.sort_values("Model_Estimated_Efficiency", ascending=False).head(3)
        cols = st.columns(3)
        for col, (_, row) in zip(cols, top3.iterrows()):
            with col:
                st.markdown(
                    f"""<div class="mp-kpi-card">
                        <div class="mp-kpi-label">{row['Channel']}</div>
                        <div class="mp-kpi-value">₹{row['Model_Estimated_Efficiency']:.2f}</div>
                        <div class="mp-kpi-delta-pos">est. revenue per ₹1 spent</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="mp-section-title">🤖 AI-Driven Business Insights</div>', unsafe_allow_html=True)
    insights = _generate_insights(kpis, attribution, budget_opt, budget_table)
    for insight in insights:
        st.markdown(f'<div class="mp-callout" style="margin-bottom:8px;">{insight}</div>', unsafe_allow_html=True)

    with st.expander("What does this page mean, and how should management use it?"):
        st.markdown(
            "- **What's happening:** the KPI cards and trend charts above summarize actual revenue and spend "
            "over the observed period.\n"
            "- **Why it's happening:** the pie/bar charts show how the fitted Marketing Mix Model attributes "
            "revenue across channels, and how current budget is actually distributed.\n"
            "- **What to do:** the *Potential Improvement* figure and the Current-vs-Optimized chart come "
            "from a real optimization run against the fitted model — see the Budget Optimizer page to adjust "
            "assumptions and re-run it live."
        )
