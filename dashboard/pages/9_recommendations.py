"""Recommendations page — dynamically generated from actual model/analytics outputs.

Works with both legacy demo channel_attribution (full MMM) and generic
Ridge-based attribution from the dataset-first pipeline. Gracefully
handles missing columns by checking before accessing.
"""

import sys
from pathlib import Path

import pandas as pd  # was missing: the page failed with NameError on the demo (found in upgrade PART 01)
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import load_channel_attribution, load_budget_optimization, load_kpis


def generate_recommendations(attribution, budget_opt, kpis) -> list:
    recs = []
    if attribution.empty:
        return recs

    # Check which columns are available
    has_efficiency = "Model_Estimated_Efficiency" in attribution.columns
    has_channel = "Channel" in attribution.columns
    has_contribution_pct = "Contribution_Pct" in attribution.columns
    has_spend = "Spend" in attribution.columns

    # Fall back to generic column names if dashboard-expected names are missing
    if not has_channel:
        if "channel" in attribution.columns:
            attribution = attribution.rename(columns={"channel": "Channel"})
            has_channel = True
    if not has_efficiency:
        if "relative_contribution_pct" in attribution.columns:
            # Use relative contribution as a proxy for efficiency
            attribution = attribution.rename(columns={"relative_contribution_pct": "Model_Estimated_Efficiency"})
            has_efficiency = True

    if not has_channel or not has_efficiency:
        return recs

    sorted_by_eff = attribution.sort_values("Model_Estimated_Efficiency", ascending=False)
    best = sorted_by_eff.iloc[0]
    worst = sorted_by_eff.iloc[-1]

    # Check for negative-coefficient channels
    has_direction = "Coefficient_Direction" in attribution.columns
    negative_channels = attribution[attribution["Coefficient_Direction"] == "negative"] if has_direction else pd.DataFrame()

    # Only recommend increasing investment when the best channel has a positive association
    if best["Model_Estimated_Efficiency"] > 0:
        recs.append({
            "title": f"Increase investment in {best['Channel']}",
            "detail": f"{best['Channel']} shows the strongest positive association with revenue "
                       f"(coefficient: {best['Model_Estimated_Efficiency']:.4f}). "
                       f"Note: this is a statistical association from a linear model, not proof of "
                       f"causal impact — validate with a controlled test before reallocating budget.",
            "type": "opportunity",
        })

    # Flag channels with negative associations
    if not negative_channels.empty:
        for _, neg_ch in negative_channels.iterrows():
            recs.append({
                "title": f"Investigate negative association for {neg_ch['Channel']}",
                "detail": f"{neg_ch['Channel']} has a negative model coefficient "
                           f"({neg_ch['Model_Estimated_Efficiency']:.4f}), meaning higher spend is "
                           f"associated with lower revenue in the model. This may reflect "
                           f"multicollinearity, reverse causality, or genuine over-saturation — "
                           f"investigate before changing spend.",
                "type": "risk",
            })
    elif worst["Model_Estimated_Efficiency"] < best["Model_Estimated_Efficiency"] * 0.5:
        recs.append({
            "title": f"Investigate {worst['Channel']} spending efficiency",
            "detail": f"{worst['Channel']} shows the lowest model-estimated efficiency "
                       f"({worst['Model_Estimated_Efficiency']:.4f}). Consider whether this "
                       f"reflects genuine diminishing returns (over-saturation) or a targeting/creative issue "
                       f"worth investigating before cutting spend outright.",
            "type": "risk",
        })

    # Diminishing returns candidate (only when Spend and Contribution_Pct are available)
    if has_spend and has_contribution_pct:
        attribution2 = attribution.copy()
        total_spend = attribution2["Spend"].sum()
        if total_spend > 0:
            attribution2["spend_share"] = attribution2["Spend"] / total_spend
            attribution2["contribution_share"] = attribution2["Contribution_Pct"] / 100
            attribution2["gap"] = attribution2["spend_share"] - attribution2["contribution_share"]
            over_invested = attribution2.sort_values("gap", ascending=False).iloc[0]
            if over_invested["gap"] > 0.03:
                recs.append({
                    "title": f"Review possible diminishing returns on {over_invested['Channel']}",
                    "detail": f"{over_invested['Channel']} accounts for {over_invested['spend_share']*100:.1f}% of "
                               f"spend but only {over_invested['contribution_share']*100:.1f}% of estimated "
                               f"contribution — a gap consistent with the channel approaching saturation at current "
                               f"spend levels.",
                    "type": "risk",
                })

    if budget_opt:
        imp = budget_opt.get("expected_improvement_pct", 0)
        if imp > 1:
            recs.append({
                "title": "Reallocate the current budget toward the optimizer's recommendation",
                "detail": f"Without increasing total spend, the optimizer's recommended allocation is "
                           f"projected to improve predicted weekly revenue by {imp:+.1f}%.",
                "type": "opportunity",
            })

    if kpis:
        holiday_lift = kpis.get("holiday_analysis", {}).get("holiday_revenue_lift_pct", 0)
        if holiday_lift and holiday_lift > 5:
            recs.append({
                "title": "Increase investment during high-performing (holiday) periods",
                "detail": f"Holiday weeks show a {holiday_lift:+.1f}% revenue lift versus non-holiday weeks. "
                           f"Front-loading incremental spend ahead of these windows may capture additional demand.",
                "type": "opportunity",
            })

    return recs


@safe_page("Recommendations")
def render():
    st.markdown('<div class="mp-header-badge">RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.title("Management Recommendations")

    attribution = load_channel_attribution()
    budget_opt = load_budget_optimization()
    kpis = load_kpis()

    recs = generate_recommendations(attribution, budget_opt, kpis)
    if not recs:
        st.warning("Run the full analytics/model pipeline first to generate recommendations.")
        return

    st.markdown("Recommendations below are generated dynamically from this project's computed model and "
                "analytics outputs — none are hard-coded.")

    for rec in recs:
        icon = "💡" if rec["type"] == "opportunity" else "⚠️"
        border_class = "mp-kpi-delta-pos" if rec["type"] == "opportunity" else "mp-kpi-delta-neg"
        st.markdown(
            f"""<div class="mp-kpi-card" style="margin-bottom:12px;">
                <div class="mp-kpi-label">{icon} {rec['title']}</div>
                <div style="color:#374151; font-size:14px; margin-top:6px;">{rec['detail']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
