"""Marketing Mix Model page — model performance, diagnostics, curves."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import (
    load_model_metrics, load_model_diagnostics, load_actual_vs_predicted,
)
from dashboard.components.kpi_card import render_kpi_row
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


def render():
    st.markdown('<div class="mp-header-badge">MARKETING MIX MODEL</div>', unsafe_allow_html=True)
    st.title("Marketing Mix Model (MMM)")

    metrics = load_model_metrics()
    diagnostics = load_model_diagnostics()
    avp = load_actual_vs_predicted()

    if not metrics:
        st.warning("Model metrics not found. Run `python src/models/marketing_mix_model.py` first.")
        return

    selected = metrics["selected_model"]
    test_m = metrics["model_comparison"][selected]["test"]

    st.markdown(f"**Selected model:** `{selected}` (chosen for the best out-of-sample RMSE/MAPE balance, not R² alone)")

    render_kpi_row([
        {"label": "R²  (test)", "value": f"{test_m['r2']*100:.1f}%"},
        {"label": "RMSE (test)", "value": f"₹{test_m['rmse']:,.0f}"},
        {"label": "MAE (test)", "value": f"₹{test_m['mae']:,.0f}"},
        {"label": "MAPE (test)", "value": f"{test_m['mape']:.1f}%"},
    ])

    st.markdown('<div class="mp-callout">'
                'How this works, in plain terms: the model assumes this week\'s marketing spend keeps '
                'influencing sales for a few weeks after (adstock / carryover), and that spending more on a '
                'channel helps less and less the more you already spend on it (saturation / diminishing '
                'returns). It combines these effects with business controls (price, promotions, holidays, '
                'competitor and economic conditions) to predict weekly revenue, and is evaluated on weeks it '
                'never saw during training.</div>', unsafe_allow_html=True)

    st.markdown('<div class="mp-section-title">Model Comparison</div>', unsafe_allow_html=True)
    comp_rows = []
    for name, m in metrics["model_comparison"].items():
        comp_rows.append({
            "Model": name, "Test R²": round(m["test"]["r2"], 3),
            "Test RMSE": round(m["test"]["rmse"], 0), "Test MAE": round(m["test"]["mae"], 0),
            "Test MAPE %": round(m["test"]["mape"], 2), "Selected": "✅" if name == selected else "",
        })
    st.dataframe(comp_rows, use_container_width=True, hide_index=True)

    st.markdown('<div class="mp-section-title">Actual vs Predicted Revenue (Test Period)</div>', unsafe_allow_html=True)
    if not avp.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=avp["Date"], y=avp["Actual_Revenue"], name="Actual", line=dict(color=CHART_COLORWAY[0])))
        fig.add_trace(go.Scatter(x=avp["Date"], y=avp["Predicted_Revenue"], name="Predicted", line=dict(color=CHART_COLORWAY[3], dash="dash")))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="mp-section-title">Residual Analysis</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(x=avp["Date"], y=avp["Residual"], marker_color=CHART_COLORWAY[5]))
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=280, title="Residuals (Actual − Predicted)", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    if diagnostics:
        st.markdown('<div class="mp-section-title">Adstock & Saturation Curves</div>', unsafe_allow_html=True)
        curves = diagnostics.get("curves", {})
        channel_choice = st.selectbox("Channel", list(curves.keys()))
        c = curves[channel_choice]

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Scatter(y=c["adstock_decay_curve"], mode="lines+markers", line=dict(color=CHART_COLORWAY[2])))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                               title=f"Adstock Decay Curve — decay={c['decay_rate']}",
                               xaxis_title="Weeks after spend", yaxis_title="Carryover effect",
                               margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Scatter(x=c["spend_grid"], y=c["saturation_response_curve"],
                                        mode="lines", line=dict(color=CHART_COLORWAY[4])))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                               title=f"Saturation / Response Curve — k={c['half_saturation_k']:.0f}",
                               xaxis_title="Weekly spend (₹)", yaxis_title="Saturated response",
                               margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
