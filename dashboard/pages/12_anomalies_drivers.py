"""Phase 5 Anomaly Detection & Driver Intelligence dashboard page.

Renders anomaly detection results and driver/relationship analysis from
the Phase 5 analysis report.  Never hardcodes column names -- every label,
metric, and chart axis comes from the report JSON.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from dashboard.components.data_loader import (
    load_phase5_analysis,
    pipeline_ready,
)
from dashboard.components import session_paths
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from dashboard.styles.theme import apply_chart_style

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
    "info": "🔵",
}

_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]


def _severity_icon(severity: str) -> str:
    return _SEVERITY_EMOJI.get(str(severity).lower(), "⚪")


def _safe_get(d, *keys, default=None):
    """Nested dict access that never raises."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def _fmt(value, decimals: int = 4) -> str:
    """Format a numeric value safely, returning '--' for None/NaN."""
    if value is None:
        return "--"
    try:
        v = float(value)
        if pd.isna(v):
            return "--"
        if abs(v) >= 1_000:
            return f"{v:,.{min(decimals, 2)}f}"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Section 1: Anomaly overview metrics
# ---------------------------------------------------------------------------

def _render_anomaly_overview(anomaly_report: dict):
    """Top-level metrics and distribution charts for anomalies."""
    st.markdown("---")
    st.subheader("Anomaly Overview")

    anomalies = anomaly_report.get("anomalies", [])
    if not anomalies:
        st.info("No anomalies were detected in this dataset.")
        return

    # -- Metric cards --
    total = len(anomalies)
    severities = [a.get("severity", "unknown").lower() for a in anomalies]
    critical_count = severities.count("critical")
    high_count = severities.count("high")
    affected_cols = list({a.get("column", "unknown") for a in anomalies})

    cols = st.columns(4)
    cols[0].metric("Total Anomalies", total)
    cols[1].metric("Critical", critical_count)
    cols[2].metric("High", high_count)
    cols[3].metric("Affected Columns", len(affected_cols))

    # -- Distribution charts --
    if HAS_PLOTLY:
        chart_left, chart_right = st.columns(2)

        # Severity distribution
        with chart_left:
            sev_counts = {}
            for s in severities:
                sev_counts[s] = sev_counts.get(s, 0) + 1
            # Order by severity
            ordered_labels = [s for s in _SEVERITY_ORDER if s in sev_counts]
            ordered_labels += [s for s in sev_counts if s not in ordered_labels]
            ordered_values = [sev_counts[s] for s in ordered_labels]

            sev_colors = {
                "critical": "#DC2626",
                "high": "#D97706",
                "medium": "#EAB308",
                "low": "#16A34A",
                "info": "#06B6D4",
            }
            bar_colors = [sev_colors.get(s, "#6B7280") for s in ordered_labels]

            fig = go.Figure(go.Bar(
                x=ordered_labels,
                y=ordered_values,
                marker_color=bar_colors,
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Severity Distribution",
                height=320,
                margin=dict(t=40, l=10, r=10, b=10),
                xaxis_title="Severity",
                yaxis_title="Count",
            )
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)

        # Anomaly type distribution
        with chart_right:
            types = [a.get("type", a.get("anomaly_type", "unknown")) for a in anomalies]
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            type_labels = [t[0] for t in sorted_types]
            type_values = [t[1] for t in sorted_types]

            fig = go.Figure(go.Bar(
                x=type_labels,
                y=type_values,
                marker_color=CHART_COLORWAY[0],
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Anomaly Type Distribution",
                height=320,
                margin=dict(t=40, l=10, r=10, b=10),
                xaxis_title="Type",
                yaxis_title="Count",
            )
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)
    else:
        # Fallback: simple text summary
        sev_counts = {}
        for s in severities:
            sev_counts[s] = sev_counts.get(s, 0) + 1
        st.markdown("**Severity breakdown:** " + ", ".join(f"{k}: {v}" for k, v in sev_counts.items()))


# ---------------------------------------------------------------------------
# Section 2: Anomaly explorer
# ---------------------------------------------------------------------------

def _render_anomaly_explorer(anomaly_report: dict):
    """Interactive table with filters for drilling into individual anomalies."""
    st.markdown("---")
    st.subheader("Anomaly Explorer")

    anomalies = anomaly_report.get("anomalies", [])
    if not anomalies:
        st.info("No anomalies to explore.")
        return

    # Build a DataFrame from the anomalies list
    display_keys = ["column", "type", "severity", "score",
                    "observed_value", "expected_value", "explanation"]
    rows = []
    for a in anomalies:
        row = {}
        for key in display_keys:
            # Try alternate key names
            val = a.get(key, a.get(f"anomaly_{key}", None))
            if val is None and key == "type":
                val = a.get("anomaly_type", "")
            row[key] = val if val is not None else ""
        rows.append(row)

    anom_df = pd.DataFrame(rows)

    # Filters
    filter_cols = st.columns(2)
    affected_columns = sorted(anom_df["column"].unique().tolist()) if "column" in anom_df.columns else []
    severity_values = sorted(anom_df["severity"].unique().tolist()) if "severity" in anom_df.columns else []

    with filter_cols[0]:
        col_filter = st.selectbox(
            "Filter by column",
            options=["All"] + affected_columns,
            index=0,
            key="anomaly_col_filter",
        )
    with filter_cols[1]:
        sev_filter = st.selectbox(
            "Filter by severity",
            options=["All"] + severity_values,
            index=0,
            key="anomaly_sev_filter",
        )

    filtered = anom_df.copy()
    if col_filter != "All":
        filtered = filtered[filtered["column"] == col_filter]
    if sev_filter != "All":
        filtered = filtered[filtered["severity"] == sev_filter]

    if filtered.empty:
        st.info("No anomalies match the selected filters.")
        return

    # Add severity indicator column
    if "severity" in filtered.columns:
        filtered = filtered.copy()
        filtered.insert(0, "", filtered["severity"].apply(_severity_icon))

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=min(400, 40 + len(filtered) * 35),
    )
    st.caption(f"Showing {len(filtered)} of {len(anom_df)} anomalies")


# ---------------------------------------------------------------------------
# Section 3: Driver analysis
# ---------------------------------------------------------------------------

def _render_driver_analysis(driver_report: dict):
    """Top drivers table and horizontal bar chart."""
    st.markdown("---")
    st.subheader("Driver Analysis")

    target = driver_report.get("target_column", driver_report.get("target", None))
    target_type = driver_report.get("target_type", driver_report.get("type", ""))
    status = driver_report.get("status", "")

    if status and status != "success":
        reason = driver_report.get("message", driver_report.get("reason", ""))
        st.info(f"Driver analysis status: {status}. {reason}".strip())
        return

    drivers = driver_report.get("drivers", driver_report.get("top_drivers", []))

    if not drivers and not target:
        st.info("No reliable target was detected. Driver ranking is unavailable.")
        return

    if target:
        st.markdown(f"**Target column:** `{target}`" + (f"  |  **Type:** {target_type}" if target_type else ""))

    if not drivers:
        st.info("No drivers were identified for the detected target.")
        return

    # Build drivers table
    driver_rows = []
    for i, d in enumerate(drivers):
        driver_rows.append({
            "Rank": i + 1,
            "Feature": d.get("feature", d.get("name", d.get("column", f"Driver {i+1}"))),
            "Importance": d.get("importance", d.get("score", d.get("weight", 0))),
            "Method": d.get("method", d.get("algorithm", "")),
            "Direction": d.get("direction", d.get("sign", "")),
            "Explanation": d.get("explanation", d.get("description", "")),
        })

    driver_df = pd.DataFrame(driver_rows)
    st.dataframe(driver_df, use_container_width=True, hide_index=True)

    # Horizontal bar chart of driver importance
    if HAS_PLOTLY and driver_rows:
        features = [r["Feature"] for r in driver_rows]
        importances = []
        for r in driver_rows:
            try:
                importances.append(float(r["Importance"]))
            except (TypeError, ValueError):
                importances.append(0)

        # Reverse so highest importance is at top
        features_rev = list(reversed(features))
        importances_rev = list(reversed(importances))

        fig = go.Figure(go.Bar(
            x=importances_rev,
            y=features_rev,
            orientation="h",
            marker_color=CHART_COLORWAY[0],
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Driver Importance",
            height=max(250, len(features) * 28 + 80),
            margin=dict(t=40, l=10, r=10, b=10),
        )
        # Upgrade PART 05: axis titles (the importance scale is method-specific, so unitless).
        st.plotly_chart(apply_chart_style(fig, x_title="Importance score", y_title="Driver"),
                        use_container_width=True)


# ---------------------------------------------------------------------------
# Section 4: Relationship explorer
# ---------------------------------------------------------------------------

def _render_relationships(analysis: dict):
    """Table of detected relationships between columns."""
    # Relationships may live in driver_report or at the top level
    relationships = (
        _safe_get(analysis, "driver_report", "relationships", default=[])
        or analysis.get("relationships", [])
    )

    if not relationships:
        return

    st.markdown("---")
    st.subheader("Relationship Explorer")

    rel_rows = []
    for r in relationships:
        rel_rows.append({
            "Type": r.get("type", r.get("relationship_type", "")),
            "Columns": r.get("columns", r.get("features", r.get("pair", ""))),
            "Strength": r.get("strength", r.get("correlation", r.get("score", ""))),
            "Direction": r.get("direction", ""),
            "Description": r.get("description", r.get("explanation", "")),
        })

    if not rel_rows:
        return

    # Convert list-type "Columns" values to readable strings
    rel_df = pd.DataFrame(rel_rows)
    if "Columns" in rel_df.columns:
        rel_df["Columns"] = rel_df["Columns"].apply(
            lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else str(x)
        )

    st.dataframe(rel_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 5: Limitations / unavailable analyses
# ---------------------------------------------------------------------------

def _render_limitations(analysis: dict):
    """Show what analyses were not available and why."""
    unavailable = []

    # Collect from anomaly_report
    anom_unavail = _safe_get(analysis, "anomaly_report", "unavailable_analyses", default=[])
    if anom_unavail:
        unavailable.extend(anom_unavail)

    # Collect from driver_report
    drv_unavail = _safe_get(analysis, "driver_report", "unavailable_analyses", default=[])
    if drv_unavail:
        unavailable.extend(drv_unavail)

    # Top-level unavailable_analyses
    top_unavail = analysis.get("unavailable_analyses", [])
    if top_unavail:
        unavailable.extend(top_unavail)

    # Top-level limitations
    limitations = analysis.get("limitations", [])
    if limitations:
        unavailable.extend(limitations)

    if not unavailable:
        return

    st.markdown("---")
    with st.expander("Dataset Limitations & Unavailable Analyses"):
        seen = set()
        for item in unavailable:
            if isinstance(item, dict):
                name = item.get("analysis", item.get("name", ""))
                reason = item.get("reason", item.get("message", ""))
                text = f"**{name}:** {reason}" if name else reason
            else:
                text = str(item)

            if text and text not in seen:
                seen.add(text)
                st.markdown(f"- {text}")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def render():
    st.set_page_config(
        page_title="Anomalies & Drivers | MarketPulse AI",
        page_icon="🔍",
        layout="wide",
    )
    st.title("Anomaly Detection & Driver Intelligence")
    st.caption("Phase 5 -- automatic anomaly detection and relationship-based driver analysis")

    # Gate: pipeline must be ready
    if not pipeline_ready():
        st.info("Upload and process a dataset to see anomaly detection and driver analysis.")
        st.stop()

    # Load Phase 5 analysis
    analysis = load_phase5_analysis()
    if not analysis:
        st.info(
            "Phase 5 analysis is not yet available for this dataset. "
            "Run the pipeline to generate results."
        )
        st.stop()

    # ------- Section 1 & 2: Anomaly overview + explorer -------
    anomaly_report = analysis.get("anomaly_report", analysis.get("anomalies", {}))

    # Handle case where anomaly_report is a list instead of dict
    if isinstance(anomaly_report, list):
        anomaly_report = {"anomalies": anomaly_report}
    elif not isinstance(anomaly_report, dict):
        anomaly_report = {}

    if anomaly_report:
        _render_anomaly_overview(anomaly_report)
        _render_anomaly_explorer(anomaly_report)
    else:
        st.info("No anomaly detection results found in the Phase 5 analysis.")

    # ------- Section 3: Driver analysis -------
    driver_report = analysis.get("driver_report", analysis.get("drivers", {}))
    if isinstance(driver_report, list):
        driver_report = {"drivers": driver_report}
    elif not isinstance(driver_report, dict):
        driver_report = {}

    if driver_report:
        _render_driver_analysis(driver_report)
    else:
        st.markdown("---")
        st.subheader("Driver Analysis")
        st.info("No reliable target was detected. Driver ranking is unavailable.")

    # ------- Section 4: Relationship explorer -------
    _render_relationships(analysis)

    # ------- Section 5: Limitations -------
    _render_limitations(analysis)


# app.py imports this module and calls render() itself; rendering again at
# import time drew the page twice on its first visit (duplicate elements).
if __name__ == "__main__":
    render()
