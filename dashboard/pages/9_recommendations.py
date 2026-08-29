"""Recommendations page — dynamically generated from actual model/analytics outputs."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import load_channel_attribution, load_budget_optimization, load_kpis


def generate_recommendations(attribution, budget_opt, kpis) -> list:
    recs = []
    if attribution.empty:
        return recs

    sorted_by_eff = attribution.sort_values("Model_Estimated_Efficiency", ascending=False)
    best = sorted_by_eff.iloc[0]
    worst = sorted_by_eff.iloc[-1]

    recs.append({
        "title": f"Increase investment in {best['Channel']}",
        "detail": f"{best['Channel']} shows the highest model-estimated efficiency "
                   f"(₹{best['Model_Estimated_Efficiency']:.2f} of estimated incremental revenue per ₹1 spent). "
                   f"Shifting incremental budget here is likely to improve overall marketing ROI.",
        "type": "opportunity",
    })

    if worst["Model_Estimated_Efficiency"] < best["Model_Estimated_Efficiency"] * 0.5:
        recs.append({
            "title": f"Investigate {worst['Channel']} spending efficiency",
            "detail": f"{worst['Channel']} shows the lowest model-estimated efficiency "
                       f"(₹{worst['Model_Estimated_Efficiency']:.2f} per ₹1 spent). Consider whether this "
                       f"reflects genuine diminishing returns (over-saturation) or a targeting/creative issue "
                       f"worth investigating before cutting spend outright.",
            "type": "risk",
        })

    # Diminishing returns candidate: high spend but low contribution share relative to spend share
    attribution2 = attribution.copy()
    attribution2["spend_share"] = attribution2["Spend"] / attribution2["Spend"].sum()
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
