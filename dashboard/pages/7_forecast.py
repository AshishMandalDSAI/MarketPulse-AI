"""Forecast page — historical revenue + near-term forecast with prediction interval."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import load_marketing_data, load_forecast
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY


def render():
    st.markdown('<div class="mp-header-badge">FORECAST</div>', unsafe_allow_html=True)
    st.title("Revenue Forecast")

    df = load_marketing_data()
    forecast = load_forecast()

    if forecast.empty:
        st.warning("Forecast not found. Run `python src/forecasting/forecast.py` first.")
        return

    st.markdown("A trend + seasonal (52-week Fourier) Ridge regression forecast — chosen deliberately for "
                "reliability over complexity given the dataset size, with an approximate 90% prediction interval.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Revenue"], name="Historical Revenue", line=dict(color=CHART_COLORWAY[0])))
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Forecast_Revenue"], name="Forecast", line=dict(color=CHART_COLORWAY[3])))
    fig.add_trace(go.Scatter(
        x=list(forecast["Date"]) + list(forecast["Date"][::-1]),
        y=list(forecast["Upper_90"]) + list(forecast["Lower_90"][::-1]),
        fill="toself", fillcolor="rgba(217,119,6,0.18)", line=dict(color="rgba(0,0,0,0)"),
        name="90% Prediction Interval", showlegend=True,
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Forecast Table</div>', unsafe_allow_html=True)
    st.dataframe(forecast, use_container_width=True, hide_index=True)
