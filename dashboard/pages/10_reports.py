"""Reports page — downloadable CSV/JSON exports and summary report."""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page
from dashboard.components import session_paths
from config.settings import DEMO_DATASET_ID
DATA_DIR = ROOT / "data" / "processed"  # legacy default; render() resolves the active dataset's dir


def _download_button(label, path, mime="text/csv"):
    if not path.exists():
        st.caption(f"⚪ {label} — not yet generated")
        return
    with open(path, "rb") as f:
        st.download_button(label, f.read(), file_name=path.name, mime=mime, use_container_width=True)


@safe_page("Reports")
def render():
    from dashboard.components.currency import symbol as _cur_symbol
    _cur = _cur_symbol()  # demo: ₹; upload: detected/chosen or none (PART 01)
    st.markdown('<div class="mp-header-badge">REPORTS</div>', unsafe_allow_html=True)
    st.title("Reports & Exports")

    data_dir = session_paths.get_processed_dir()  # scoped to the active dataset — no cross-dataset leakage
    is_demo = session_paths.get_dataset_id() == DEMO_DATASET_ID

    st.markdown('<div class="mp-section-title">Data Exports</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        _download_button("⬇️ Channel Performance (CSV)", data_dir / "channel_performance.csv")
        _download_button("⬇️ Channel Attribution (CSV)", data_dir / "channel_attribution.csv")
        _download_button("⬇️ Budget Recommendation (CSV)", data_dir / "budget_optimization_result.csv")
    with col2:
        _download_button("⬇️ Marketing KPIs (JSON)", data_dir / "marketing_kpis.json", mime="application/json")
        _download_button("⬇️ Scenario Comparison (CSV)", data_dir / "scenario_comparison.csv")
        _download_button("⬇️ Revenue Forecast (CSV)", data_dir / "revenue_forecast.csv")
        _download_button("⬇️ Generic KPIs (JSON)", data_dir / "generic_kpis.json", mime="application/json")

    st.markdown('<div class="mp-section-title">Executive Summary Report</div>', unsafe_allow_html=True)

    kpi_path = data_dir / "marketing_kpis.json"
    generic_kpi_path = data_dir / "generic_kpis.json"
    budget_path = data_dir / "budget_optimization_summary.json"

    # Determine which KPI source is available
    has_legacy_kpis = kpi_path.exists()
    has_generic_kpis = generic_kpi_path.exists()

    if has_legacy_kpis or has_generic_kpis:
        if has_legacy_kpis:
            # Legacy demo format: nested dict with specific keys
            with open(kpi_path) as f:
                kpis = json.load(f).get("kpis", {})

            company_line = "**Company:** NovaMart (synthetic, for academic demonstration)" if is_demo \
                else "**Dataset:** your uploaded dataset"
            period_line = f"**Period:** {kpis.get('date_range', {}).get('start', '—')} to {kpis.get('date_range', {}).get('end', '—')}"

            metrics_lines = [
                f"- Total Revenue: {_cur}{kpis.get('total_revenue', 0):,.0f}",
                f"- Total Marketing Spend: {_cur}{kpis.get('total_marketing_spend', 0):,.0f}",
                f"- Overall Naive ROAS: {kpis.get('overall_naive_roas', 0):.2f}x",
                f"- Revenue Growth (H1 → H2): {kpis.get('revenue_growth_h1_to_h2', 0)*100:+.1f}%",
            ]
            disclaimer = ("*All figures generated from the project's own dataset, Marketing Mix Model, "
                          "and optimization engine.*")
            if is_demo:
                disclaimer += "\n*Dataset is synthetic and created for academic project demonstration purposes.*"
        else:
            # Generic format: array of {label, value, note}
            with open(generic_kpi_path) as f:
                generic_kpis = json.load(f)

            company_line = "**Dataset:** your uploaded dataset"
            # Try to find a date range entry
            date_range_entry = next((k for k in generic_kpis if "date range" in k.get("label", "").lower()), None)
            period_line = f"**Period:** {date_range_entry['value']}" if date_range_entry else ""

            metrics_lines = [f"- {k['label']}: {k['value']}" for k in generic_kpis]
            disclaimer = "*All figures generated from the project's analysis pipeline.*"

        # Budget optimization section (optional — only available for full MMM)
        budget_section = ""
        if budget_path.exists():
            with open(budget_path) as f:
                budget = json.load(f)
            budget_section = f"""
## Budget Optimization
- Current Predicted Weekly Revenue: {_cur}{budget.get('current_predicted_revenue', 0):,.0f}
- Optimized Predicted Weekly Revenue: {_cur}{budget.get('optimized_predicted_revenue', 0):,.0f}
- Expected Improvement: {budget.get('expected_improvement_pct', 0):+.1f}%
"""

        summary_md = f"""# MarketPulse AI — Executive Summary Report

{company_line}
{period_line}

## Key Metrics
{chr(10).join(metrics_lines)}
{budget_section}
{disclaimer}
"""
        st.download_button("⬇️ Download Executive Summary (Markdown)", summary_md,
                            file_name="marketpulse_executive_summary.md", mime="text/markdown",
                            use_container_width=True)
        with st.expander("Preview summary"):
            st.markdown(summary_md)
    else:
        st.caption("Run the full pipeline to enable the executive summary report.")
