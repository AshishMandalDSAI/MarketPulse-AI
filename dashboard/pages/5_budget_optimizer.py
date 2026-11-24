"""Budget Optimizer page — current vs AI-recommended allocation, live re-run."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import load_budget_optimization, load_budget_table
from dashboard.components.kpi_card import render_kpi_row, format_inr
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from dashboard.styles.theme import apply_chart_style
from dashboard.components.ui import show_table


@safe_page("Budget Optimizer", required_capability="budget_optimization")
def render():
    st.markdown('<div class="mp-header-badge">BUDGET OPTIMIZER</div>', unsafe_allow_html=True)
    st.title("AI Budget Optimizer")

    default_result = load_budget_optimization()
    default_table = load_budget_table()

    if not default_result:
        from dashboard.components import session_paths as _sp
        if _sp.is_user_dataset_mode():
            st.info("Budget optimization requires a full Marketing Mix Model with adstock and saturation "
                    "parameters, which is not available for this dataset. The generic Ridge regression model "
                    "provides channel attribution and efficiency estimates — see the **Channel Performance & Contribution** and "
                    "**Recommendations** pages for actionable insights.")
        else:
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
        # Phase 1: Guard against reading demo data when an uploaded dataset is active.
        from dashboard.components import session_paths as _sp
        if _sp.is_user_dataset_mode():
            st.info("Live budget re-optimization is not yet supported for uploaded datasets. "
                    "The pre-computed results from the analysis pipeline are shown below.")
        else:
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

    if result.get("success") is False:
        st.error("The optimizer did not find a valid allocation for these inputs: "
                 f"{result.get('message', 'unknown reason')}. The figures below should not be used.")
    same_budget = abs(result["total_budget"] - sum(result["current_allocation"].values())) <= 1.0
    realloc = result.get("reallocation_improvement_pct")
    cards = [
        {"label": "Predicted weekly revenue · current mix",
         "value": format_inr(result["current_predicted_revenue"], compact=False),
         "help": "Your recent average weekly spend, run through the model"},
        {"label": "Predicted weekly revenue · recommended",
         "value": format_inr(result["optimized_predicted_revenue"], compact=False),
         "help": f"Total budget {format_inr(result['total_budget'], compact=False)}"},
    ]
    if same_budget or realloc is None:
        cards.append({"label": "Estimated uplift (same budget)", "value": f"{result['expected_improvement_pct']:+.1f}%",
                      "help": "Model estimate — not a guarantee"})
    else:
        cards.append({"label": "Uplift from reallocation", "value": f"{realloc:+.1f}%",
                      "help": "vs. your current mix scaled to this budget (like-for-like)"})
        cards.append({"label": "Total change vs today", "value": f"{result['expected_improvement_pct']:+.1f}%",
                      "help": "Includes the effect of changing the total budget"})
    cards.append({"label": "Revenue per ₹1 (recommended)", "value": f"{result['optimized_efficiency']:.2f}x",
                  "help": "Predicted revenue ÷ total budget"})
    render_kpi_row(cards)

    st.markdown('<div class="mp-section-title">Current vs AI-Recommended Allocation</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=table["Channel"], y=table["Current_Budget"], marker_color=CHART_COLORWAY[0],
                         hovertemplate="%{x}: ₹%{y:,.0f}<extra>Current</extra>"))
    fig.add_trace(go.Bar(name="Recommended", x=table["Channel"], y=table["Recommended_Budget"], marker_color=CHART_COLORWAY[1],
                         hovertemplate="%{x}: ₹%{y:,.0f}<extra>Recommended</extra>"))
    fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=380, margin=dict(t=20, l=10, r=10, b=10),
                      legend=dict(orientation="h", y=1.1, x=0), yaxis_title="Weekly spend")
    fig.update_yaxes(tickprefix="₹", tickformat="~s")
    st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    st.markdown('<div class="mp-section-title">Recommendation Table</div>', unsafe_allow_html=True)
    money_cols = [c for c in ("Current_Budget", "Recommended_Budget", "Change") if c in table.columns]
    show_table(table, currency_cols=money_cols, pct_cols=[c for c in ("Change_Pct",) if c in table.columns])
    bounds = result.get("bounds_per_channel")
    if bounds:
        st.caption(f"Each channel is kept between {format_inr(bounds['min'], compact=False)} and "
                   f"{format_inr(bounds['max'], compact=False)} per week; allocations always sum to the total budget.")

    st.caption("Optimization holds recent adstock carry-in and business controls fixed and re-allocates this "
               "week's spend across channels, subject to per-channel min/max bounds, to maximize predicted "
               "revenue for the chosen total budget. Predictions come from a model of historical association — "
               "validate large shifts with a pilot before committing.")
