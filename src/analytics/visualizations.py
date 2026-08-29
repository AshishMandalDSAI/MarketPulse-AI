"""
MarketPulse AI - Visualization Helpers
==========================================
Reusable Plotly figure builders shared between the dashboard, notebooks, and
any future reporting scripts, so chart styling stays consistent everywhere
rather than being redefined per-page.

These return `plotly.graph_objects.Figure` objects; callers decide whether to
`.show()` them (notebook), pass them to `st.plotly_chart()` (dashboard), or
`.write_image()` them (static report export, requires `kaleido`).
"""

import pandas as pd
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"
COLORWAY = ["#5B6CFF", "#8B5CF6", "#16A34A", "#D97706", "#DC2626", "#06B6D4", "#EC4899", "#84CC16"]


def revenue_trend_chart(df: pd.DataFrame, date_col="Date", revenue_col="Revenue") -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[date_col], y=df[revenue_col], line=dict(color=COLORWAY[0], width=2)))
    fig.update_layout(template=TEMPLATE, title="Revenue Trend", height=350)
    return fig


def spend_trend_chart(df: pd.DataFrame, date_col="Date", spend_col="Total_Marketing_Spend") -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[date_col], y=df[spend_col], line=dict(color=COLORWAY[1], width=2)))
    fig.update_layout(template=TEMPLATE, title="Marketing Spend Trend", height=350)
    return fig


def channel_spend_bar(channel_totals: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=channel_totals.index, y=channel_totals.values, marker_color=COLORWAY))
    fig.update_layout(template=TEMPLATE, title="Channel Spend", height=350)
    return fig


def channel_share_pie(channel_perf: pd.DataFrame, label_col="Channel", value_col="Spend_Share_Pct") -> go.Figure:
    fig = go.Figure(go.Pie(labels=channel_perf[label_col], values=channel_perf[value_col], hole=0.4,
                            marker=dict(colors=COLORWAY)))
    fig.update_layout(template=TEMPLATE, title="Channel Spend Share", height=350)
    return fig


def contribution_pie(attribution: pd.DataFrame, label_col="Channel", value_col="Estimated_Contribution") -> go.Figure:
    fig = go.Figure(go.Pie(labels=attribution[label_col], values=attribution[value_col], hole=0.45,
                            marker=dict(colors=COLORWAY)))
    fig.update_layout(template=TEMPLATE, title="Model-Estimated Revenue Contribution by Channel", height=350)
    return fig


def actual_vs_predicted_chart(avp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=avp["Date"], y=avp["Actual_Revenue"], name="Actual", line=dict(color=COLORWAY[0])))
    fig.add_trace(go.Scatter(x=avp["Date"], y=avp["Predicted_Revenue"], name="Predicted",
                              line=dict(color=COLORWAY[3], dash="dash")))
    fig.update_layout(template=TEMPLATE, title="Actual vs Predicted Revenue", height=380)
    return fig


def residual_bar_chart(avp: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=avp["Date"], y=avp["Residual"], marker_color=COLORWAY[5]))
    fig.update_layout(template=TEMPLATE, title="Residuals (Actual - Predicted)", height=300)
    return fig


def budget_comparison_bar(comparison_table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=comparison_table["Channel"], y=comparison_table["Current_Budget"],
                          marker_color=COLORWAY[0]))
    fig.add_trace(go.Bar(name="Recommended", x=comparison_table["Channel"], y=comparison_table["Recommended_Budget"],
                          marker_color=COLORWAY[2]))
    fig.update_layout(template=TEMPLATE, barmode="group", title="Current vs Recommended Budget", height=380)
    return fig


def scenario_comparison_bar(scenario_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=scenario_df["Scenario"], y=scenario_df["Predicted_Revenue"],
                            marker_color=COLORWAY[: len(scenario_df)]))
    fig.update_layout(template=TEMPLATE, title="Predicted Revenue by Scenario", height=350)
    return fig


def forecast_chart(historical: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical["Date"], y=historical["Revenue"], name="Historical",
                              line=dict(color=COLORWAY[0])))
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Forecast_Revenue"], name="Forecast",
                              line=dict(color=COLORWAY[3])))
    fig.add_trace(go.Scatter(
        x=list(forecast["Date"]) + list(forecast["Date"][::-1]),
        y=list(forecast["Upper_90"]) + list(forecast["Lower_90"][::-1]),
        fill="toself", fillcolor="rgba(217,119,6,0.18)", line=dict(color="rgba(0,0,0,0)"),
        name="90% Prediction Interval",
    ))
    fig.update_layout(template=TEMPLATE, title="Revenue Forecast", height=420)
    return fig
