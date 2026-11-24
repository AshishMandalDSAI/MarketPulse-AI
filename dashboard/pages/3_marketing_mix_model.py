"""Marketing Mix Model page — model performance, diagnostics, curves.

Dual rendering: handles both the legacy demo MMM metrics format (with
model_comparison, diagnostics, actual_vs_predicted) and the generic
Ridge regression format from the dataset-first pipeline.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import (
    load_model_metrics, load_model_diagnostics, load_actual_vs_predicted,
)
from dashboard.components.kpi_card import render_kpi_row
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from dashboard.styles.theme import apply_chart_style


@safe_page("Marketing Mix Model", required_capability="marketing_mix_modeling")
def render():
    st.markdown('<div class="mp-header-badge">MARKETING MIX MODEL</div>', unsafe_allow_html=True)
    st.title("Marketing Mix Model (MMM)")

    metrics = load_model_metrics()
    diagnostics = load_model_diagnostics()
    avp = load_actual_vs_predicted()

    if not metrics:
        st.warning("Model metrics not found. Run the analysis pipeline first.")
        return

    # Detect format: legacy (has model_comparison) vs generic (flat dict)
    selected = metrics.get("selected_model")
    model_comparison = metrics.get("model_comparison")

    if selected and model_comparison and selected in model_comparison:
        # Legacy format with full model comparison
        test_m = model_comparison[selected].get("test") or {}
        model_type_label = selected
    else:
        # Generic format — build test_m from flat keys
        test_m = {
            "r2": metrics.get("r2"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
        }
        model_type_label = metrics.get("model_type", "Ridge Regression")

    def _fmt(v, kind):
        if v is None:
            return "—"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "—"
        if kind == "r2":
            return f"{v:.3f}"
        if kind == "pct":
            return f"{v:.1f}%"
        return f"{v:,.0f}"

    st.markdown(f"**Model:** `{model_type_label}`")
    if not test_m:
        st.warning("This model has no holdout period (too few observations), so out-of-sample accuracy "
                   "cannot be measured. Treat its estimates as exploratory.")
    render_kpi_row([
        {"label": "R² (holdout)", "value": _fmt(test_m.get("r2"), "r2"),
         "help": "Share of revenue variation explained on unseen weeks (1 = perfect; can be negative)"},
        {"label": "RMSE (holdout)", "value": _fmt(test_m.get("rmse"), "num")},
        {"label": "MAE (holdout)", "value": _fmt(test_m.get("mae"), "num"),
         "help": "Average absolute error on unseen periods"},
        {"label": "MAPE (holdout)", "value": _fmt(test_m.get("mape"), "pct"),
         "help": "— means undefined (actual values of 0)" if test_m.get("mape") is None else None},
    ])
    base = metrics.get("baseline")
    if base and test_m.get("mae") is not None and base.get("mae"):
        better = float(test_m["mae"]) < float(base["mae"])
        pct = (1 - float(test_m["mae"]) / float(base["mae"])) * 100
        st.markdown(
            f'<div class="mp-callout {"mp-callout-good" if better else "mp-callout-warn"}">'
            f'<div class="mp-callout-label">Baseline check</div>'
            f'Holdout MAE {_fmt(test_m["mae"], "num")} vs {_fmt(base["mae"], "num")} for a naive baseline '
            f'({base.get("model", "training mean")}): '
            f'{"<b>" + format(pct, ".1f") + "% better</b>" if better else "<b>not better</b> — the channel estimates below should be treated with caution"}.'
            f'</div>', unsafe_allow_html=True)
    excluded = (metrics.get("rows_excluded_missing_revenue") or 0) + (metrics.get("rows_excluded_invalid_date") or 0)
    if excluded:
        st.caption(f"{excluded} row(s) were excluded from training because revenue or the date was missing/invalid "
                   f"(they were not treated as zero).")

    # Explanation callout
    model_type = metrics.get("model_type", "")
    if model_type == "generic_ridge_linear":
        st.markdown(
            '<div class="mp-callout">'
            'This is a lightweight linear model (Ridge regression) that estimates how each '
            'marketing channel\'s spend contributes to the target metric. It does not model '
            'adstock (carryover) or saturation (diminishing returns) — those diagnostics are '
            'not available for this model type.'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="mp-callout">'
            'How this works, in plain terms: the model assumes this week\'s marketing spend keeps '
            'influencing sales for a few weeks after (adstock / carryover), and that spending more on a '
            'channel helps less and less the more you already spend on it (saturation / diminishing '
            'returns). It combines these effects with business controls (price, promotions, holidays, '
            'competitor and economic conditions) to predict weekly revenue, and is evaluated on weeks it '
            'never saw during training.</div>', unsafe_allow_html=True
        )

    # Model comparison table (only if multiple models were compared)
    if model_comparison and len(model_comparison) > 1:
        st.markdown('<div class="mp-section-title">Model Comparison</div>', unsafe_allow_html=True)
        comp_rows = []
        for name, m in model_comparison.items():
            t = m.get("test") or {}
            _rd = lambda v, d: round(float(v), d) if v is not None else None
            comp_rows.append({
                "Model": name, "Test R²": _rd(t.get("r2"), 3),
                "Test RMSE": _rd(t.get("rmse"), 0), "Test MAE": _rd(t.get("mae"), 0),
                "Test MAPE %": _rd(t.get("mape"), 2), "Selected": "✅" if name == selected else "",
            })
        st.dataframe(comp_rows, use_container_width=True, hide_index=True)

    # Features list (for generic models)
    features = metrics.get("features", [])
    n_obs = metrics.get("n_observations")
    if features:
        with st.expander(f"Model features ({len(features)} feature(s)" + (f", {n_obs} observations" if n_obs else "") + ")"):
            st.write(", ".join(features))

    # Actual vs Predicted (available for legacy demo)
    if not avp.empty:
        st.markdown('<div class="mp-section-title">Actual vs Predicted Revenue (Test Period)</div>', unsafe_allow_html=True)
        # Detect date column dynamically
        date_col = "Date"
        for candidate in ["Date", "date", "Week", "Period"]:
            if candidate in avp.columns:
                date_col = candidate
                break

        actual_col = next((c for c in ["Actual_Revenue", "actual", "Actual"] if c in avp.columns), None)
        pred_col = next((c for c in ["Predicted_Revenue", "predicted", "Predicted"] if c in avp.columns), None)
        resid_col = next((c for c in ["Residual", "residual"] if c in avp.columns), None)

        if actual_col and pred_col:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=avp[date_col], y=avp[actual_col], name="Actual", line=dict(color=CHART_COLORWAY[0])))
            fig.add_trace(go.Scatter(x=avp[date_col], y=avp[pred_col], name="Predicted", line=dict(color=CHART_COLORWAY[3], dash="dash")))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(apply_chart_style(fig), use_container_width=True)

        if resid_col:
            st.markdown('<div class="mp-section-title">Residual Analysis</div>', unsafe_allow_html=True)
            fig2 = go.Figure(go.Bar(x=avp[date_col], y=avp[resid_col], marker_color=CHART_COLORWAY[5]))
            fig2.update_layout(template=PLOTLY_TEMPLATE, height=280, title="Residuals (Actual − Predicted)", margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(apply_chart_style(fig2), use_container_width=True)

    # Diagnostics (adstock/saturation curves — only for legacy MMM, not generic Ridge)
    if diagnostics:
        curves = diagnostics.get("curves", {})
        if curves:
            st.markdown('<div class="mp-section-title">Adstock & Saturation Curves</div>', unsafe_allow_html=True)
            channel_choice = st.selectbox("Channel", list(curves.keys()))
            c = curves[channel_choice]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Scatter(y=c["adstock_decay_curve"], mode="lines+markers", line=dict(color=CHART_COLORWAY[2])))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                   title=f"Adstock Decay Curve — decay={c['decay_rate']}",
                                   xaxis_title="Weeks after spend", yaxis_title="Carryover effect",
                                   margin=dict(t=40, l=10, r=10, b=10))
                st.plotly_chart(apply_chart_style(fig), use_container_width=True)
            with col2:
                fig = go.Figure(go.Scatter(x=c["spend_grid"], y=c["saturation_response_curve"],
                                            mode="lines", line=dict(color=CHART_COLORWAY[4])))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                   title=f"Saturation / Response Curve — k={c['half_saturation_k']:.0f}",
                                   xaxis_title="Weekly spend", yaxis_title="Saturated response",
                                   margin=dict(t=40, l=10, r=10, b=10))
                st.plotly_chart(apply_chart_style(fig), use_container_width=True)
