"""Reports page — downloadable CSV/JSON exports and summary report."""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data" / "processed"


def _download_button(label, path, mime="text/csv"):
    if not path.exists():
        st.caption(f"⚪ {label} — not yet generated")
        return
    with open(path, "rb") as f:
        st.download_button(label, f.read(), file_name=path.name, mime=mime, use_container_width=True)


def render():
    st.markdown('<div class="mp-header-badge">REPORTS</div>', unsafe_allow_html=True)
    st.title("Reports & Exports")

    st.markdown('<div class="mp-section-title">Data Exports</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        _download_button("⬇️ Channel Performance (CSV)", DATA_DIR / "channel_performance.csv")
        _download_button("⬇️ Channel Attribution (CSV)", DATA_DIR / "channel_attribution.csv")
        _download_button("⬇️ Budget Recommendation (CSV)", DATA_DIR / "budget_optimization_result.csv")
    with col2:
        _download_button("⬇️ Marketing KPIs (JSON)", DATA_DIR / "marketing_kpis.json", mime="application/json")
        _download_button("⬇️ Scenario Comparison (CSV)", DATA_DIR / "scenario_comparison.csv")
        _download_button("⬇️ Revenue Forecast (CSV)", DATA_DIR / "revenue_forecast.csv")

    st.markdown('<div class="mp-section-title">Executive Summary Report</div>', unsafe_allow_html=True)

    kpi_path = DATA_DIR / "marketing_kpis.json"
    budget_path = DATA_DIR / "budget_optimization_summary.json"
    if kpi_path.exists() and budget_path.exists():
        with open(kpi_path) as f:
            kpis = json.load(f).get("kpis", {})
        with open(budget_path) as f:
            budget = json.load(f)

        summary_md = f"""# MarketPulse AI — Executive Summary Report

**Company:** NovaMart (synthetic, for academic demonstration)
**Period:** {kpis.get('date_range', {}).get('start', '—')} to {kpis.get('date_range', {}).get('end', '—')}

## Key Metrics
- Total Revenue: ₹{kpis.get('total_revenue', 0):,.0f}
- Total Marketing Spend: ₹{kpis.get('total_marketing_spend', 0):,.0f}
- Overall Naive ROAS: {kpis.get('overall_naive_roas', 0):.2f}x
- Revenue Growth (H1 → H2): {kpis.get('revenue_growth_h1_to_h2', 0)*100:+.1f}%

## Budget Optimization
- Current Predicted Weekly Revenue: ₹{budget.get('current_predicted_revenue', 0):,.0f}
- Optimized Predicted Weekly Revenue: ₹{budget.get('optimized_predicted_revenue', 0):,.0f}
- Expected Improvement: {budget.get('expected_improvement_pct', 0):+.1f}%

*All figures generated from the project's own dataset, Marketing Mix Model, and optimization engine.
Dataset is synthetic and created for academic project demonstration purposes.*
"""
        st.download_button("⬇️ Download Executive Summary (Markdown)", summary_md,
                            file_name="marketpulse_executive_summary.md", mime="text/markdown",
                            use_container_width=True)
        with st.expander("Preview summary"):
            st.markdown(summary_md)
    else:
        st.caption("Run the full pipeline to enable the executive summary report.")
