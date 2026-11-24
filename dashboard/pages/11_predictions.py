"""Phase 4 Predictive Analytics dashboard page.

Dynamically renders prediction results (forecast, regression, classification,
segmentation) from the Phase 4 prediction report.  Never hardcodes column
names -- every label, metric, and chart axis comes from the report JSON.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from dashboard.components.data_loader import (
    load_prediction_report,
    load_analysis_context,
    pipeline_ready,
)
from dashboard.components import session_paths
from dashboard.components import chart_labels as CL
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from dashboard.styles.theme import apply_chart_style

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    True: "#16A34A",   # green
    False: "#6B7280",  # gray
}


def _fmt(value, decimals: int = 4) -> str:
    """Format a numeric value safely, returning '--' for None/NaN."""
    if value is None:
        return "--"
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return "--"
        if abs(v) >= 1_000:
            return f"{v:,.{min(decimals, 2)}f}"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value) -> str:
    if value is None:
        return "--"
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return "--"
        return f"{v:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _safe_get(d: dict, *keys, default=None):
    """Nested dict access that never raises."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def _metric_cards(metrics: dict, col_count: int = 4):
    """Render a row of st.metric() cards from a dict of {label: value}."""
    items = list(metrics.items())
    if not items:
        return
    cols = st.columns(min(col_count, len(items)))
    for i, (label, value) in enumerate(items):
        cols[i % len(cols)].metric(label, value)


def _render_feature_importance(importance_data, title: str = "Feature Importance"):
    """Bar chart of feature importances from a list of dicts or a dict."""
    if not importance_data:
        return

    if isinstance(importance_data, dict):
        features = list(importance_data.keys())
        scores = list(importance_data.values())
    elif isinstance(importance_data, list):
        features = [item.get("feature", item.get("name", f"F{i}")) for i, item in enumerate(importance_data)]
        scores = [item.get("importance", item.get("score", 0)) for item in importance_data]
    else:
        return

    if not features:
        return

    # Sort descending and take top 15
    pairs = sorted(zip(features, scores), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)[:15]
    features, scores = zip(*pairs) if pairs else ([], [])

    fig = go.Figure(go.Bar(
        x=list(scores),
        y=list(features),
        orientation="h",
        marker_color=CHART_COLORWAY[0],
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        height=max(250, len(features) * 28 + 80),
        margin=dict(t=40, l=10, r=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(apply_chart_style(fig, x_title="Importance score", y_title="Feature"), use_container_width=True)


def _render_model_candidates(candidates):
    """Expandable comparison table of model candidates tried."""
    if not candidates:
        return
    with st.expander("Model candidates comparison"):
        rows = []
        for c in candidates:
            if isinstance(c, dict):
                rows.append(c)
            else:
                rows.append({"candidate": str(c)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_limitations(limitations):
    """Expandable list of model limitations."""
    if not limitations:
        return
    with st.expander("Limitations & caveats"):
        for lim in limitations:
            st.markdown(f"- {lim}")


def _render_methodology(explanation: str = None, model_name: str = None):
    """Expandable methodology section."""
    parts = []
    if model_name:
        parts.append(f"**Model:** {model_name}")
    if explanation:
        parts.append(explanation)
    if not parts:
        return
    with st.expander("Methodology"):
        st.markdown("\n\n".join(parts))


# ---------------------------------------------------------------------------
# Task summary section
# ---------------------------------------------------------------------------

def _render_task_summary(tasks: list):
    """Section 1: show detected predictive tasks and whether requirements are met."""
    st.markdown("---")
    st.subheader("Available Predictive Tasks")

    if not tasks:
        st.info("No predictive tasks were detected for this dataset.")
        return

    cols = st.columns(min(4, len(tasks)))
    for i, task in enumerate(tasks):
        col = cols[i % len(cols)]
        task_type = task.get("task_type", task.get("type", "Unknown"))
        met = task.get("requirements_met", task.get("feasible", False))
        target = task.get("target_column", task.get("target", ""))
        confidence = task.get("confidence", task.get("score", None))

        color = _STATUS_COLORS.get(bool(met), "#6B7280")
        status_label = "Ready" if met else "Not feasible"

        with col:
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding: 0.5rem 0.75rem; '
                f'margin-bottom: 0.5rem; border-radius: 0 6px 6px 0; '
                f'background: rgba(255,255,255,0.03);">'
                f'<strong style="text-transform: capitalize;">{task_type}</strong><br/>'
                f'<span style="color: {color}; font-size: 0.85rem;">{status_label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if target:
                st.caption(f"Target: {target}")
            if confidence is not None:
                st.caption(f"Confidence: {_fmt(confidence, 2)}")


# ---------------------------------------------------------------------------
# Forecast section
# ---------------------------------------------------------------------------

def _render_forecast(forecast: dict):
    """Section 2: time-series forecast results."""
    status = forecast.get("status", "")
    if status != "success":
        if status:
            st.warning(f"Forecast task status: {status}. {forecast.get('message', '')}")
        return

    st.markdown("---")
    st.subheader("Forecast Results")

    results = forecast.get("results", forecast)
    predictions = results.get("predictions", [])
    metrics = results.get("metrics", {})
    model_name = results.get("model_name", results.get("model", ""))
    explanation = results.get("explanation", "")
    candidates = results.get("candidates", results.get("model_candidates", []))
    importance = results.get("feature_importance", results.get("importances", None))
    limitations = results.get("limitations", [])

    # -- Metrics row --
    metric_display = {}
    for key, label in [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE"), ("r2", "R²")]:
        val = metrics.get(key, metrics.get(key.upper(), None))
        if val is not None:
            metric_display[label] = _pct(val) if key == "mape" else _fmt(val)
    _metric_cards(metric_display)

    # -- Line chart with confidence bands --
    if predictions and not isinstance(predictions[0], dict):
        # Forecasting engine output: parallel flat lists (predictions,
        # lower_bound, upper_bound, future_dates). Build a frame with named
        # columns so the chart below can detect the axes.
        n_pred = len(predictions)
        dates = results.get("future_dates") or []
        pred_df = pd.DataFrame({
            "date": dates if len(dates) == n_pred else list(range(1, n_pred + 1)),
            "forecast": predictions,
        })
        for bound in ("lower_bound", "upper_bound"):
            vals = results.get(bound) or []
            if len(vals) == n_pred:
                pred_df[bound] = vals
    elif predictions:
        pred_df = pd.DataFrame(predictions)
    elif "prediction_values" in results:
        pred_df = pd.DataFrame(results["prediction_values"])
    else:
        pred_df = pd.DataFrame()

    if not pred_df.empty:
        # Detect columns dynamically
        time_col = None
        for candidate_col in pred_df.columns:
            if str(candidate_col).lower() in ("date", "period", "time", "timestamp", "ds", "index"):
                time_col = candidate_col
                break
        if time_col is None and len(pred_df.columns) >= 1:
            # First column is likely the time axis
            time_col = pred_df.columns[0]

        pred_col = None
        for candidate_col in pred_df.columns:
            low = str(candidate_col).lower()
            if any(k in low for k in ("forecast", "prediction", "predicted", "yhat", "pred")):
                pred_col = candidate_col
                break

        lower_col = None
        upper_col = None
        for candidate_col in pred_df.columns:
            low = str(candidate_col).lower()
            if any(k in low for k in ("lower", "lo", "yhat_lower", "lower_bound")):
                lower_col = candidate_col
            if any(k in low for k in ("upper", "hi", "yhat_upper", "upper_bound")):
                upper_col = candidate_col

        if time_col and pred_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pred_df[time_col], y=pred_df[pred_col],
                name="Forecast", line=dict(color=CHART_COLORWAY[3]),
            ))
            if lower_col and upper_col:
                fig.add_trace(go.Scatter(
                    x=list(pred_df[time_col]) + list(pred_df[time_col][::-1]),
                    y=list(pred_df[upper_col]) + list(pred_df[lower_col][::-1]),
                    fill="toself",
                    fillcolor="rgba(217,119,6,0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Confidence Band",
                    showlegend=True,
                ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=400,
                margin=dict(t=20, l=10, r=10, b=10),
                title="Forecast with confidence band",
                legend_title_text="Series",
            )
            st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(pred_df[time_col]),
                                              y_title=CL.value_axis_title(pred_col)), use_container_width=True)

    # -- Methodology, importance, candidates, limitations --
    _render_methodology(explanation, model_name)
    _render_feature_importance(importance)
    _render_model_candidates(candidates)
    _render_limitations(limitations)


# ---------------------------------------------------------------------------
# Regression section
# ---------------------------------------------------------------------------

def _render_regression(regression: dict):
    """Section 3: regression results."""
    status = regression.get("status", "")
    if status != "success":
        if status:
            st.warning(f"Regression task status: {status}. {regression.get('message', '')}")
        return

    st.markdown("---")
    st.subheader("Regression Results")

    results = regression.get("results", regression)
    metrics = results.get("metrics", {})
    predictions = results.get("predictions", [])
    actuals = results.get("actuals", [])
    model_name = results.get("model_name", results.get("model", ""))
    explanation = results.get("explanation", "")
    candidates = results.get("candidates", results.get("model_candidates", []))
    importance = results.get("feature_importance", results.get("importances", None))
    limitations = results.get("limitations", [])

    # -- Metrics row --
    metric_display = {}
    for key, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²"), ("mape", "MAPE")]:
        val = metrics.get(key, metrics.get(key.upper(), None))
        if val is not None:
            metric_display[label] = _pct(val) if key == "mape" else _fmt(val)
    _metric_cards(metric_display)

    # -- Predictions vs Actuals scatter --
    pred_vals = predictions
    actual_vals = actuals

    # Might be stored as a list of dicts or a flat list
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
        pred_vals = [p.get("predicted", p.get("prediction", p.get("yhat", 0))) for p in predictions]
        actual_vals = [p.get("actual", p.get("y", 0)) for p in predictions]
    elif "prediction_values" in results:
        pv_df = pd.DataFrame(results["prediction_values"])
        for c in pv_df.columns:
            if "pred" in str(c).lower():
                pred_vals = pv_df[c].tolist()
            if "actual" in str(c).lower() or "true" in str(c).lower():
                actual_vals = pv_df[c].tolist()

    if pred_vals and actual_vals and len(pred_vals) == len(actual_vals):
        col_scatter, col_resid = st.columns(2)

        with col_scatter:
            scatter_df = pd.DataFrame({"Predicted": pred_vals, "Actual": actual_vals})
            fig = px.scatter(
                scatter_df, x="Actual", y="Predicted",
                title="Predictions vs Actuals",
                color_discrete_sequence=[CHART_COLORWAY[0]],
            )
            # Perfect-prediction line
            min_val = min(min(pred_vals), min(actual_vals))
            max_val = max(max(pred_vals), max(actual_vals))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", line=dict(dash="dash", color="#6B7280"),
                name="Perfect prediction", showlegend=True,
            ))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(apply_chart_style(fig, x_title="Actual value", y_title="Predicted value"),
                            use_container_width=True)

        with col_resid:
            residuals = [a - p for a, p in zip(actual_vals, pred_vals)]
            fig = go.Figure(go.Histogram(
                x=residuals,
                marker_color=CHART_COLORWAY[1],
                name="Residuals",
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Residual Distribution",
                height=380,
                margin=dict(t=40, l=10, r=10, b=10),
            )
            st.plotly_chart(apply_chart_style(fig, x_title="Residual (actual − predicted)",
                                              y_title="Number of records"), use_container_width=True)

    # -- Methodology, importance, candidates, limitations --
    _render_methodology(explanation, model_name)
    _render_feature_importance(importance)
    _render_model_candidates(candidates)
    _render_limitations(limitations)


# ---------------------------------------------------------------------------
# Classification section
# ---------------------------------------------------------------------------

def _render_classification(classification: dict):
    """Section 4: classification results."""
    status = classification.get("status", "")
    if status != "success":
        if status:
            st.warning(f"Classification task status: {status}. {classification.get('message', '')}")
        return

    st.markdown("---")
    st.subheader("Classification Results")

    results = classification.get("results", classification)
    metrics = results.get("metrics", {})
    model_name = results.get("model_name", results.get("model", ""))
    explanation = results.get("explanation", "")
    candidates = results.get("candidates", results.get("model_candidates", []))
    importance = results.get("feature_importance", results.get("importances", None))
    limitations = results.get("limitations", [])
    confusion = results.get("confusion_matrix", None)
    labels = results.get("labels", results.get("class_labels", None))

    # -- Metrics row --
    metric_display = {}
    for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score")]:
        val = metrics.get(key, metrics.get(key.upper(), metrics.get(f"{key}_score", None)))
        if val is not None:
            metric_display[label] = _fmt(val)
    _metric_cards(metric_display)

    # -- Confusion matrix heatmap --
    if confusion is not None:
        try:
            cm = np.array(confusion)
            if cm.ndim == 2 and cm.size > 0:
                tick_labels = labels if labels and len(labels) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]
                fig = go.Figure(go.Heatmap(
                    z=cm,
                    x=tick_labels,
                    y=tick_labels,
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(size=14),
                ))
                fig.update_layout(
                    template=PLOTLY_TEMPLATE,
                    title="Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=400,
                    margin=dict(t=40, l=10, r=10, b=10),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(apply_chart_style(fig), use_container_width=True)
        except Exception:
            st.caption("Confusion matrix could not be rendered.")

    # -- Methodology, importance, candidates, limitations --
    _render_methodology(explanation, model_name)
    _render_feature_importance(importance)
    _render_model_candidates(candidates)
    _render_limitations(limitations)


# ---------------------------------------------------------------------------
# Segmentation section
# ---------------------------------------------------------------------------

def _render_segmentation(segmentation: dict):
    """Section 5: clustering / segmentation results."""
    status = segmentation.get("status", "")
    if status != "success":
        if status:
            st.warning(f"Segmentation task status: {status}. {segmentation.get('message', '')}")
        return

    st.markdown("---")
    st.subheader("Segmentation Results")

    results = segmentation.get("results", segmentation)
    metrics = results.get("metrics", {})
    clusters = results.get("clusters", results.get("profiles", results.get("cluster_profiles", [])))
    descriptions = results.get("descriptions", results.get("cluster_descriptions", []))
    distribution = results.get("distribution", results.get("cluster_distribution", None))
    model_name = results.get("model_name", results.get("model", ""))
    explanation = results.get("explanation", "")
    limitations = results.get("limitations", [])

    # -- Quality metrics --
    metric_display = {}
    silhouette = metrics.get("silhouette_score", metrics.get("silhouette", None))
    if silhouette is not None:
        metric_display["Silhouette Score"] = _fmt(silhouette)
    n_clusters = metrics.get("n_clusters", metrics.get("num_clusters", None))
    if n_clusters is not None:
        metric_display["Clusters"] = str(n_clusters)
    inertia = metrics.get("inertia", None)
    if inertia is not None:
        metric_display["Inertia"] = _fmt(inertia, 1)
    _metric_cards(metric_display)

    # -- Cluster distribution donut --
    if distribution:
        if isinstance(distribution, dict):
            dist_labels = list(distribution.keys())
            dist_values = list(distribution.values())
        elif isinstance(distribution, list):
            dist_labels = [d.get("cluster", d.get("label", f"Cluster {i}")) for i, d in enumerate(distribution)]
            dist_values = [d.get("count", d.get("size", d.get("value", 0))) for d in distribution]
        else:
            dist_labels, dist_values = [], []

        if dist_labels and dist_values:
            fig = go.Figure(go.Pie(
                labels=dist_labels,
                values=dist_values,
                hole=0.45,
                marker=dict(colors=CHART_COLORWAY[:len(dist_labels)]),
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Cluster Distribution",
                height=380,
                margin=dict(t=40, l=10, r=10, b=10),
            )
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    # -- Cluster profiles table --
    if clusters:
        st.markdown("**Cluster Profiles**")
        if isinstance(clusters, list) and clusters:
            if isinstance(clusters[0], dict):
                st.dataframe(pd.DataFrame(clusters), use_container_width=True, hide_index=True)
            else:
                for i, cluster in enumerate(clusters):
                    st.markdown(f"- **Cluster {i}:** {cluster}")
        elif isinstance(clusters, dict):
            st.dataframe(
                pd.DataFrame.from_dict(clusters, orient="index").reset_index().rename(columns={"index": "Cluster"}),
                use_container_width=True,
                hide_index=True,
            )

    # -- Cluster descriptions --
    if descriptions:
        with st.expander("Cluster descriptions"):
            if isinstance(descriptions, list):
                for desc in descriptions:
                    if isinstance(desc, dict):
                        label = desc.get("cluster", desc.get("label", ""))
                        text = desc.get("description", desc.get("text", ""))
                        st.markdown(f"**{label}:** {text}")
                    else:
                        st.markdown(f"- {desc}")
            elif isinstance(descriptions, dict):
                for label, text in descriptions.items():
                    st.markdown(f"**{label}:** {text}")

    # -- Methodology, limitations --
    _render_methodology(explanation, model_name)
    _render_limitations(limitations)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def render():
    st.set_page_config(page_title="Predictive Analytics | MarketPulse AI", page_icon="🔮", layout="wide")
    st.title("Predictive Analytics")
    st.caption("Phase 4 -- automated ML predictions, forecasts, classification and segmentation")

    # Gate: pipeline must be ready
    if not pipeline_ready():
        st.warning("No processed data found. Please upload and process a dataset first.")
        st.stop()

    # Load prediction report
    report = load_prediction_report()
    if not report:
        st.info(
            "No prediction report found for this dataset. "
            "Run the analysis pipeline to generate predictive insights."
        )
        st.stop()

    # ------- Section 1: Task summary -------
    tasks = report.get("tasks", [])
    _render_task_summary(tasks)

    # ------- Section 2: Forecast -------
    forecast = report.get("forecast", None)
    if forecast and isinstance(forecast, dict):
        _render_forecast(forecast)

    # ------- Section 3: Regression -------
    regression = report.get("regression", None)
    if regression and isinstance(regression, dict):
        _render_regression(regression)

    # ------- Section 4: Classification -------
    classification = report.get("classification", None)
    if classification and isinstance(classification, dict):
        _render_classification(classification)

    # ------- Section 5: Segmentation -------
    segmentation = report.get("segmentation", None)
    if segmentation and isinstance(segmentation, dict):
        _render_segmentation(segmentation)

    # ------- Footer: nothing ran -------
    ran_any = any(
        report.get(key, {}).get("status") == "success"
        for key in ("forecast", "regression", "classification", "segmentation")
        if isinstance(report.get(key), dict)
    )
    if not ran_any and not tasks:
        st.info("The prediction engine did not find any feasible predictive tasks for this dataset.")


# app.py imports this module and calls render() itself; rendering again at
# import time drew the page twice on its first visit (duplicate elements).
if __name__ == "__main__":
    render()
