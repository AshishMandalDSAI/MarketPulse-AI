"""Marketing Performance page — revenue, spend, channels, seasonality, filters.

Dual rendering paths:
- Legacy: uses hardcoded demo dataset columns (Year, Quarter, Revenue, etc.)
- Dynamic: uses canonical metadata to discover date, revenue, and channel columns
  for uploaded datasets.

The page routes automatically: if the demo-specific columns are present, legacy
renders; otherwise, the dynamic path runs with whatever columns are available.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import (
    load_marketing_data, load_channel_performance, load_monthly_performance,
    load_quarterly_performance, load_canonical_metadata, load_generic_kpis,
)
from dashboard.components import session_paths
from dashboard.components import chart_labels as CL
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from dashboard.styles.theme import apply_chart_style

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]


def _has_demo_columns(df: pd.DataFrame) -> bool:
    """Check if the dataframe has the demo-specific columns needed for legacy rendering."""
    required = {"Year", "Quarter", "Date", "Revenue"}
    return required.issubset(set(df.columns))


def _render_legacy(df: pd.DataFrame):
    """Full legacy marketing-specific rendering (unchanged for demo dataset)."""
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
        st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(filtered["Date"]),
                                          y_title=CL.value_axis_title("Revenue", monetary=True), currency="y"),
                        use_container_width=True)
    with col2:
        fig = go.Figure(go.Scatter(x=filtered["Date"], y=filtered["Total_Marketing_Spend"], line=dict(color=CHART_COLORWAY[1])))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Marketing Spend Trend", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(filtered["Date"]),
                                          y_title=CL.value_axis_title("Total_Marketing_Spend", monetary=True),
                                          currency="y"),
                        use_container_width=True)

    st.markdown('<div class="mp-section-title">Channel Spend & Share</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        selected_cols = [f"{c.replace(' ', '_')}_Spend" for c in selected_channels]
        selected_cols = [c for c in selected_cols if c in filtered.columns]
        totals = filtered[selected_cols].sum().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=[c.replace("_Spend", "").replace("_", " ") for c in totals.index],
                                y=totals.values, marker_color=CHART_COLORWAY[2]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Channel Spend (selected period)", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(apply_chart_style(fig, x_title="Marketing channel",
                                          y_title=CL.value_axis_title("Spend", "sum", monetary=True), currency="y"),
                        use_container_width=True)
    with col4:
        if not channel_perf.empty:
            fig = go.Figure(go.Pie(labels=channel_perf["Channel"], values=channel_perf["Spend_Share_Pct"], hole=0.4,
                                    marker=dict(colors=CHART_COLORWAY)))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Channel Spend Share (%)", margin=dict(t=40, l=10, r=10, b=10),
                              legend_title_text="Marketing channel")
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    st.markdown('<div class="mp-section-title">Monthly & Quarterly Performance</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        if not monthly.empty:
            fig = go.Figure(go.Bar(x=monthly["Date"], y=monthly["Revenue"], marker_color=CHART_COLORWAY[0]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Monthly Revenue", margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(apply_chart_style(fig, x_title="Month",
                                              y_title=CL.value_axis_title("Revenue", "sum", monetary=True),
                                              currency="y"), use_container_width=True)
    with col6:
        if not quarterly.empty:
            fig = go.Figure(go.Bar(x=quarterly["Date"], y=quarterly["Revenue"], marker_color=CHART_COLORWAY[3]))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="Quarterly Revenue", margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(apply_chart_style(fig, x_title="Quarter",
                                              y_title=CL.value_axis_title("Revenue", "sum", monetary=True),
                                              currency="y"), use_container_width=True)

    st.markdown('<div class="mp-section-title">Customer Trends</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered["New_Customers"], name="New Customers", line=dict(color=CHART_COLORWAY[4])))
    fig.add_trace(go.Scatter(x=filtered["Date"], y=filtered["Returning_Customers"], name="Returning Customers", line=dict(color=CHART_COLORWAY[5])))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=320, title="New vs Returning Customers", margin=dict(t=40, l=10, r=10, b=10),
                      legend_title_text="Customer type")
    st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(filtered["Date"]),
                                      y_title="Number of customers"), use_container_width=True)

    st.markdown('<div class="mp-section-title">Promotion & Holiday Impact</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        promo_avg = df.groupby(df["Promotion_Spend"] > df["Promotion_Spend"].median())["Revenue"].mean()
        fig = go.Figure(go.Bar(x=["Low Promotion Weeks", "High Promotion Weeks"], y=promo_avg.values,
                                marker_color=[CHART_COLORWAY[6], CHART_COLORWAY[0]]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Average Weekly Revenue: Low vs High Promotion Weeks", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(apply_chart_style(fig, x_title="Week type",
                                          y_title=CL.value_axis_title("Weekly revenue", "mean", monetary=True),
                                          currency="y"), use_container_width=True)
    with col8:
        hol_avg = df.groupby("Holiday")["Revenue"].mean()
        fig = go.Figure(go.Bar(x=["Non-Holiday Weeks", "Holiday Weeks"], y=hol_avg.values,
                                marker_color=[CHART_COLORWAY[7], CHART_COLORWAY[1]]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, title="Average Weekly Revenue: Holiday vs Non-Holiday Weeks", margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(apply_chart_style(fig, x_title="Week type",
                                          y_title=CL.value_axis_title("Weekly revenue", "mean", monetary=True),
                                          currency="y"), use_container_width=True)


def _render_dynamic(df: pd.DataFrame):
    """Dynamic rendering path for uploaded datasets. Uses canonical metadata
    to discover the actual column names instead of hardcoding demo columns."""
    meta = load_canonical_metadata()
    date_col = session_paths.get_date_column()
    revenue_col = meta.get("revenue_column") if meta else None
    channel_cols = meta.get("channel_columns", []) if meta else []
    dimension_cols = meta.get("dimension_columns", []) if meta else []

    # Generic KPIs
    kpis = load_generic_kpis()
    if kpis:
        st.subheader("Key Performance Indicators")
        cols = st.columns(min(4, len(kpis)))
        for i, kpi in enumerate(kpis[:4]):
            cols[i % len(cols)].metric(kpi.get("label", ""), kpi.get("value", ""))

    # Revenue/target trend
    if date_col and date_col in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        # Plot revenue trend if available, otherwise first numeric column
        target_col = revenue_col if revenue_col and revenue_col in df.columns else (numeric_cols[0] if numeric_cols else None)
        if target_col:
            plot_df, applied = CL.per_period(df, date_col, target_col)
            if not plot_df.empty:
                fig = go.Figure(go.Scatter(x=plot_df[date_col], y=plot_df[target_col],
                                           line=dict(color=CHART_COLORWAY[0])))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                  title=CL.chart_title(f"{target_col} Trend", applied, target_col),
                                  margin=dict(t=40, l=10, r=10, b=10))
                st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(plot_df[date_col]),
                                                  y_title=CL.value_axis_title(target_col, applied)),
                                use_container_width=True)

        # Channel spend trends (if channels identified)
        if channel_cols:
            available_channels = [c for c in channel_cols if c in df.columns]
            if available_channels:
                st.markdown('<div class="mp-section-title">Channel Spend</div>', unsafe_allow_html=True)
                totals = df[available_channels].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sort_values(ascending=False)
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(go.Bar(x=totals.index.tolist(), y=totals.values,
                                           marker_color=CHART_COLORWAY[2]))
                    fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                                      title="Total Spend by Channel",
                                      margin=dict(t=40, l=10, r=10, b=10))
                    spend_unit = CL.currency_unit(available_channels[0])
                    st.plotly_chart(apply_chart_style(fig, x_title="Marketing channel",
                                                      y_title="Total spend" + (f" ({spend_unit})" if spend_unit else "")),
                                    use_container_width=True)
                with col2:
                    total_spend = totals.sum()
                    if total_spend > 0:
                        shares = (totals / total_spend * 100).round(1)
                        fig = go.Figure(go.Pie(labels=shares.index.tolist(), values=shares.values,
                                               hole=0.4, marker=dict(colors=CHART_COLORWAY)))
                        fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                                          title="Channel Spend Share (%)",
                                          margin=dict(t=40, l=10, r=10, b=10),
                                          legend_title_text="Marketing channel")
                        st.plotly_chart(apply_chart_style(fig), use_container_width=True)

        # Additional numeric trends (up to 4 non-channel numeric columns)
        other_numeric = [c for c in numeric_cols if c != target_col and c not in channel_cols][:4]
        if other_numeric and date_col in df.columns:
            st.markdown('<div class="mp-section-title">Additional Metrics</div>', unsafe_allow_html=True)
            n_cols = min(2, len(other_numeric))
            for i in range(0, len(other_numeric), n_cols):
                batch = other_numeric[i:i + n_cols]
                cols = st.columns(len(batch))
                for j, col_name in enumerate(batch):
                    with cols[j]:
                        plot_df, applied = CL.per_period(df, date_col, col_name)
                        if not plot_df.empty:
                            fig = go.Figure(go.Scatter(x=plot_df[date_col], y=plot_df[col_name],
                                                       line=dict(color=CHART_COLORWAY[(i + j) % len(CHART_COLORWAY)])))
                            fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                              title=CL.chart_title(f"{col_name} Trend", applied, col_name),
                                              margin=dict(t=40, l=10, r=10, b=10))
                            st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(plot_df[date_col]),
                                                              y_title=CL.value_axis_title(col_name, applied)),
                                            use_container_width=True)

    # Categorical breakdowns
    if dimension_cols:
        available_dims = [c for c in dimension_cols if c in df.columns and df[c].nunique(dropna=True) <= 50]
        if available_dims:
            st.markdown('<div class="mp-section-title">Breakdowns</div>', unsafe_allow_html=True)
            for dim_col in available_dims[:2]:
                st.subheader(f"By {dim_col}")
                if revenue_col and revenue_col in df.columns:
                    # Show mean revenue by category
                    grouped = df.groupby(dim_col)[revenue_col].mean().sort_values(ascending=False).head(15)
                    fig = go.Figure(go.Bar(x=grouped.index.tolist(), y=grouped.values,
                                           marker_color=CHART_COLORWAY[0]))
                    fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                      title=f"Average {CL.humanize(revenue_col)} by {CL.humanize(dim_col)}",
                                      margin=dict(t=40, l=10, r=10, b=10))
                    st.plotly_chart(apply_chart_style(fig, x_title=CL.category_axis_title(dim_col),
                                                      y_title=CL.value_axis_title(revenue_col, "mean")),
                                    use_container_width=True)
                else:
                    counts = df[dim_col].value_counts().head(15)
                    fig = go.Figure(go.Bar(x=counts.index.astype(str).tolist(), y=counts.values,
                                           marker_color=CHART_COLORWAY[2]))
                    fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                      title=f"Records by {CL.humanize(dim_col)} (top 15)",
                                      margin=dict(t=40, l=10, r=10, b=10))
                    st.plotly_chart(apply_chart_style(fig, x_title=CL.category_axis_title(dim_col),
                                                      y_title="Number of records"), use_container_width=True)


@safe_page("Marketing Performance", required_capability="trend_analysis")
def render():
    st.markdown('<div class="mp-header-badge">MARKETING PERFORMANCE</div>', unsafe_allow_html=True)
    st.title("Marketing Performance")

    df = load_marketing_data()

    if _has_demo_columns(df):
        _render_legacy(df)
    else:
        _render_dynamic(df)
