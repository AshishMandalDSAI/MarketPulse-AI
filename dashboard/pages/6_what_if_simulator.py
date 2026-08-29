"""What-If Simulator page — interactive channel sliders driving live model predictions."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import load_scenario_comparison
from dashboard.components.kpi_card import render_kpi_row, format_inr
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]


def _apply_preset(sim, multiplier=None, use_optimized=False):
    """Set slider session-state values to a preset scenario, then rerun so
    the sliders visibly reflect the new allocation."""
    if use_optimized:
        from src.optimization.budget_optimizer import optimize_budget
        result = optimize_budget(
            sim.current_total_budget, sim.bundle, sim.carry_in,
            sim.control_values, sim.current_allocation,
        )
        allocation = result["optimized_allocation"]
    else:
        allocation = {ch: v * multiplier for ch, v in sim.current_allocation.items()}

    for ch in CHANNELS:
        capped = min(max(allocation[ch], 0), sim.current_allocation[ch] * 3)
        st.session_state[f"slider_{ch}"] = int(capped)
    st.rerun()


def render():
    st.markdown('<div class="mp-header-badge">WHAT-IF SIMULATOR</div>', unsafe_allow_html=True)
    st.title("What-If Budget Simulator")

    try:
        from src.optimization.simulator import BudgetSimulator
        sim = BudgetSimulator()
    except FileNotFoundError:
        st.warning("Trained model not found. Run the model pipeline first.")
        return

    st.markdown("Adjust individual channel budgets and see the model's predicted revenue update live, or "
                "jump to a preset scenario below — every number is computed live from the fitted model.")

    st.markdown("**Quick scenarios:**")
    preset_cols = st.columns(5)
    presets = [
        ("Current", {"multiplier": 1.0}), ("Optimized", {"use_optimized": True}),
        ("+10% Budget", {"multiplier": 1.10}), ("+20% Budget", {"multiplier": 1.20}),
        ("-10% Budget", {"multiplier": 0.90}),
    ]
    for col, (label, kwargs) in zip(preset_cols, presets):
        with col:
            if st.button(label, use_container_width=True, key=f"preset_{label}"):
                _apply_preset(sim, **kwargs)

    allocation = {}
    cols = st.columns(2)
    for i, ch in enumerate(CHANNELS):
        current = sim.current_allocation[ch]
        with cols[i % 2]:
            allocation[ch] = st.slider(
                ch.replace("_", " "), min_value=0, max_value=int(current * 3),
                value=int(current), step=1000, key=f"slider_{ch}",
            )

    result = sim.simulate_channel_allocation(allocation)

    render_kpi_row([
        {"label": "Total Budget (this scenario)", "value": format_inr(result["total_budget"], compact=False)},
        {"label": "Predicted Revenue", "value": format_inr(result["predicted_revenue"], compact=False)},
        {"label": "Revenue Change", "value": f"{result['revenue_change_pct']:+.2f}%",
         "positive": result["revenue_change_pct"] >= 0},
        {"label": "ROI", "value": f"{result['roi']:.2f}x"},
    ])

    st.markdown('<div class="mp-section-title">Scenario Comparison</div>', unsafe_allow_html=True)
    scenario_df = load_scenario_comparison()
    if not scenario_df.empty:
        fig = go.Figure(go.Bar(x=scenario_df["Scenario"], y=scenario_df["Predicted_Revenue"],
                                marker_color=CHART_COLORWAY[:len(scenario_df)]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, title="Predicted Revenue by Scenario",
                           margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    st.caption("All figures on this page are computed live from the fitted Marketing Mix Model — moving a "
               "slider re-runs the model's saturation/adstock response for that channel, holding other "
               "channels and business controls at their recent-average levels. Preset scenarios are capped "
               "at 3× a channel's current spend to stay within the slider range shown above.")
