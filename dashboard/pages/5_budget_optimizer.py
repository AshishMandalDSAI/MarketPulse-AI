"""Budget Optimizer page — current vs AI-recommended allocation, live re-run."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import load_budget_optimization, load_budget_table
from dashboard.components.kpi_card import render_kpi_row, format_inr
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


def render():
    st.markdown('<div class="mp-header-badge">BUDGET OPTIMIZER</div>', unsafe_allow_html=True)
    st.title("AI Budget Optimizer")

    default_result = load_budget_optimization()
    default_table = load_budget_table()

    if not default_result:
        st.warning("Run `python src/optimization/budget_optimizer.py` first to generate a baseline result.")
        return

    default_budget = default_result["total_budget"]
    st.markdown("Choose a total weekly marketing budget and run the optimizer — allocations are computed "
                "live from the actual fitted Marketing Mix Model, never hard-coded.")

    with st.expander("How should management use this page?"):
        st.markdown(
            "This re-allocates the **same total budget** across channels to maximize model-predicted revenue "
            "— it does not assume more money is available. The optimizer respects per-channel minimum/maximum "
            "spend bounds so no single channel is over- or under-funded. Treat the output as a "
            "**scenario-based recommendation**, not a guaranteed result — validate with a real-world pilot "
            "before a full reallocation."
        )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        budget_input = st.slider(
            "Total weekly marketing budget (₹)",
            min_value=int(default_budget * 0.5), max_value=int(default_budget * 2.0),
            value=int(default_budget), step=10000,
        )
    with col_b:
        run_clicked = st.button("🚀 Run Budget Optimization", use_container_width=True)

    if run_clicked or "budget_opt_result" not in st.session_state:
        with st.spinner("Running scipy.optimize (SLSQP) against the fitted MMM..."):
            try:
                from src.optimization.budget_optimizer import (
                    load_bundle, get_recent_carry_in, optimize_budget, build_comparison_table, CHANNELS,
                )
                import pandas as pd
                bundle = load_bundle()
                df = pd.read_csv(ROOT / "data" / "processed" / "marketing_data_clean.csv", parse_dates=["Date"])
                recent = df.tail(8)
                current_allocation = {ch: float(recent[f"{ch}_Spend"].mean()) for ch in CHANNELS}
                control_values = {c: float(recent[c].mean()) for c in bundle["control_cols"]}
                carry_in = get_recent_carry_in(df, bundle["channel_params"])
                result = optimize_budget(budget_input, bundle, carry_in, control_values, current_allocation)
                table = build_comparison_table(result)
                st.session_state["budget_opt_result"] = result
                st.session_state["budget_opt_table"] = table
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                return

    result = st.session_state.get("budget_opt_result", default_result)
    table = st.session_state.get("budget_opt_table", default_table)

    render_kpi_row([
        {"label": "Current Predicted Revenue", "value": format_inr(result["current_predicted_revenue"], compact=False)},
        {"label": "Optimized Predicted Revenue", "value": format_inr(result["optimized_predicted_revenue"], compact=False)},
        {"label": "Expected Improvement", "value": f"{result['expected_improvement_pct']:+.1f}%"},
        {"label": "Budget Efficiency (optimized)", "value": f"{result['optimized_efficiency']:.2f}x"},
    ])

    st.markdown('<div class="mp-section-title">Current vs AI-Recommended Allocation</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=table["Channel"], y=table["Current_Budget"], marker_color=CHART_COLORWAY[0]))
    fig.add_trace(go.Bar(name="Recommended", x=table["Channel"], y=table["Recommended_Budget"], marker_color=CHART_COLORWAY[2]))
    fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=380, margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Recommendation Table</div>', unsafe_allow_html=True)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption("Optimization holds recent adstock carry-in and business controls fixed and re-allocates this "
               "week's spend across channels, subject to per-channel min/max bounds, to maximize predicted "
               "revenue for the chosen total budget.")
