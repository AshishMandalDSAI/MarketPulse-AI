"""Marketing Performance page — revenue, spend, channels, seasonality, filters."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.data_loader import (
    load_marketing_data, load_channel_performance, load_monthly_performance,
    load_quarterly_performance,
)
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]


def render():
    st.markdown('<div class="mp-header-badge">MARKETING PERFORMANCE</div>', unsafe_allow_html=True)
    st.title("Marketing Performance")

    df = load_marketing_data()
    channel_perf = load_channel_performance()
    monthly = load_monthly_performance()
    quarterly = load_quarterly_performance()

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            years = sorted(df["Year"].unique().tolist())
            selected_years = st.multiselect("Year", years, default=years)
        with col2:
            quarters = sorted(df["Quarter"].unique().tolist())
            selected_quarters = st.multiselect("Quarter", quarters, default=quarters)
        with col3:
            channel_labels = [c.replace("_", " ") for c in CHANNELS]
            selected_channels = st.multiselect("Channel (for spend chart)", channel_labels, default=channel_labels)

    filtered = df[df["Year"].isin(selected_years) & df["Quarter"].isin(selected_quarters)]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Scatter(x=filtered["Date"], y=filtered["Revenue"], line=dict(color=CHART_COLORWAY[0])))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Revenue Trend", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Scatter(x=filtered["Date"], y=filtered["Total_Marketing_Spend"], line=dict(color=CHART_COLORWAY[1])))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Marketing Spend Trend", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Channel Spend & Share</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        selected_cols = [f"{c.replace(' ', '_')}_Spend" for c in selected_channels]
        selected_cols = [c for c in selected_cols if c in filtered.columns]
        totals = filtered[selected_cols].sum().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=[c.replace("_Spend", "").replace("_", " ") for c in totals.index],
                                y=totals.values, marker_color=CHART_COLORWAY[2]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Channel Spend", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        if not channel_perf.empty:
            fig = go.Figure(go.Pie(labels=channel_perf["Channel"], values=channel_perf["Spend_Share_Pct"], hole=0.4,
                                    marker=dict(colors=CHART_COLORWAY)))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Channel Spend Share", margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Monthly & Quarterly Performance</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        if not monthly.empty:
            fig = go.Figure(go.Bar(x=monthly["Date"], y=monthly["Revenue"], marker_color=CHART_COLORWAY[0]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Monthly Revenue", margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with col6:
        if not quarterly.empty:
            fig = go.Figure(go.Bar(x=quarterly["Date"], y=quarterly["Revenue"], marker_color=CHART_COLORWAY[3]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Quarterly Revenue", margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Customer Trends</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered["New_Customers"], name="New Customers", line=dict(color=CHART_COLORWAY[4])))
    fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered["Returning_Customers"], name="Returning Customers", line=dict(color=CHART_COLORWAY[5])))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="New vs Returning Customers", margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="mp-section-title">Promotion & Holiday Impact</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        promo_avg = df.groupby(df["Promotion_Spend"] > df["Promotion_Spend"].median())["Revenue"].mean()
        fig = go.Figure(go.Bar(x=["Low Promotion Weeks", "High Promotion Weeks"], y=promo_avg.values,
                                marker_color=[CHART_COLORWAY[6], CHART_COLORWAY[0]]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Avg Revenue: Promotion Impact", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col8:
        hol_avg = df.groupby("Holiday")["Revenue"].mean()
        fig = go.Figure(go.Bar(x=["Non-Holiday Weeks", "Holiday Weeks"], y=hol_avg.values,
                                marker_color=[CHART_COLORWAY[7], CHART_COLORWAY[1]]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Avg Revenue: Holiday Impact", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
