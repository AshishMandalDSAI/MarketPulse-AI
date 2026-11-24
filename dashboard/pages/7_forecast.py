"""Forecast page — historical target metric + near-term forecast with prediction interval.

Dynamic column detection: uses canonical metadata to resolve date and target
column names instead of hardcoding 'Date' and 'Revenue'. Works with both
the legacy demo forecast and the generic pipeline forecast.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page

from dashboard.components.data_loader import (
    load_marketing_data, load_forecast, load_canonical_metadata,
)
from dashboard.components import session_paths
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from dashboard.components.kpi_card import render_kpi_row
from dashboard.components.ui import callout, section, show_table
import json
import pandas as pd
from dashboard.styles.theme import apply_chart_style


def _load_evaluation() -> dict:
    path = session_paths.get_processed_dir() / "forecast_evaluation.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _fmt(v, kind="num"):
    if v is None:
        return "—"
    return f"{v:.1f}%" if kind == "pct" else f"{v:,.0f}"


def _render_evaluation(ev: dict) -> None:
    section("How accurate is this forecast?")
    if not ev:
        st.caption("No out-of-sample evaluation is available for this forecast.")
        return
    if "selected_model" in ev:   # uploaded dataset (generic pipeline)
        sel_name, sel = ev["selected_model"], ev.get("selected_metrics") or {}
        base_name, base = ev.get("baseline"), ev.get("baseline_metrics") or {}
        folds, h = ev.get("folds"), ev.get("fold_horizon")
        cands = ev.get("candidates", {})
    else:                        # demo forecast script
        sel_name = "Ridge trend + seasonal (published model)"
        sel = (ev.get(sel_name) or {}).get("metrics", {})
        base_name = ev.get("baseline")
        base = (ev.get(base_name) or {}).get("metrics", {}) if base_name else {}
        folds, h = (ev.get(sel_name) or {}).get("folds"), ev.get("horizon_weeks")
        cands = {k: v["metrics"] for k, v in ev.items() if isinstance(v, dict) and "metrics" in v}
    imp = ev.get("improvement_vs_baseline_pct")
    render_kpi_row([
        {"label": "Model", "value": sel_name.split(" (")[0], "help": "Selected by walk-forward testing"},
        {"label": "Out-of-sample MAE", "value": _fmt(sel.get("MAE")),
         "help": f"Average error over {folds} fold(s) of {h} unseen period(s)"},
        {"label": "WAPE", "value": _fmt(sel.get("WAPE"), "pct"), "help": "Total absolute error ÷ total actual"},
        {"label": f"vs {base_name.split(' (')[0] if base_name else 'baseline'}",
         "value": f"{imp:+.1f}%" if imp is not None and base_name and sel_name != base_name else "baseline",
         "help": "MAE improvement over the best naive baseline (positive = better)"},
    ])
    if imp is not None and imp <= 0:
        callout("The model does not beat a naive baseline in walk-forward testing — read the forecast as "
                "'no reliable pattern beyond recent levels'.", kind="warn")
    if ev.get("selection_note"):
        callout(ev["selection_note"], kind="info")
    notes = [ev.get("interval_note") or ev.get("interval")]
    if ev.get("missing_periods"):
        notes.append(f"{ev['missing_periods']} missing period(s) were detected in the history; they were not filled in.")
    if ev.get("rows_combined_same_date"):
        notes.append(f"{ev['rows_combined_same_date']} row(s) sharing a date were summed into one value per period.")
    if ev.get("rows_without_valid_date") or ev.get("rows_without_valid_target"):
        notes.append(f"{ev.get('rows_without_valid_date', 0)} row(s) without a valid date and "
                     f"{ev.get('rows_without_valid_target', 0)} without a valid value were left out.")
    for n in [n for n in notes if n]:
        st.caption(n)
    if cands:
        with st.expander("All candidate models (walk-forward metrics)"):
            show_table(pd.DataFrame([
                {"Model": k, "MAE": m.get("MAE"), "RMSE": m.get("RMSE"), "WAPE %": m.get("WAPE"),
                 "sMAPE %": m.get("sMAPE"), "MAPE %": m.get("MAPE")}
                for k, m in cands.items()]).sort_values("MAE"), decimals=1)
            st.caption("MAPE skips periods with actual value 0; '—' means a metric is undefined, never zero error. "
                       f"Data range: {' to '.join(ev.get('data_range') or [])}.")


@safe_page("Forecast", required_capability="forecasting")
def render():
    st.markdown('<div class="mp-header-badge">FORECAST</div>', unsafe_allow_html=True)
    st.title("Revenue Forecast")

    df = load_marketing_data()
    forecast = load_forecast()

    if forecast.empty:
        if session_paths.is_user_dataset_mode():
            st.warning("Forecast not available for this dataset. Re-run the analysis pipeline from the Upload page.")
        else:
            st.warning("Forecast not found. Run `python src/forecasting/forecast.py` first.")
        return

    # Resolve date column dynamically
    date_col = session_paths.get_date_column() or "Date"

    # Resolve target/revenue column dynamically
    meta = load_canonical_metadata()
    ev = _load_evaluation()
    # The forecast evaluation records the column it forecast; canonical metadata has
    # no 'revenue_column' key for uploads, which left the history line blank.
    revenue_col = ((ev or {}).get("target") or (meta.get("revenue_column") if meta else None)
                   or (meta.get("semantic_roles", {}).get("revenue") if meta and isinstance(meta.get("semantic_roles"), dict) else None)
                   or "Revenue")

    # Detect forecast column names (handles both legacy and generic output)
    forecast_date_col = next((c for c in ["Date", "date"] if c in forecast.columns), None)
    forecast_val_col = next((c for c in ["Forecast_Revenue", "forecast"] if c in forecast.columns), None)
    upper_col = next((c for c in ["Upper_90", "upper_90"] if c in forecast.columns), None)
    lower_col = next((c for c in ["Lower_90", "lower_90"] if c in forecast.columns), None)

    if not forecast_val_col:
        st.warning("Forecast data format not recognized.")
        return

    if session_paths.is_user_dataset_mode():
        st.caption("Forecast for your dataset. Several simple models are compared by walk-forward testing and a "
                   "complex model is only used when it beats a naive baseline. Forecasts are estimates, not guarantees.")
    else:
        st.caption("Demo dataset: a trend + seasonal Ridge regression, deliberately simple, evaluated against naive "
                   "baselines with walk-forward testing. Forecasts are estimates, not guarantees.")

    fig = go.Figure()

    # Historical data overlay (use dynamic column names)
    if date_col in df.columns and revenue_col in df.columns:
        hist = pd.DataFrame({"d": pd.to_datetime(df[date_col], errors="coerce"),
                             "y": pd.to_numeric(df[revenue_col], errors="coerce")}).dropna()
        if hist["d"].duplicated().any():  # match the forecast: one value per period
            hist = hist.groupby("d", as_index=False)["y"].sum()
        hist = hist.sort_values("d")
        fig.add_trace(go.Scatter(
            x=hist["d"], y=hist["y"],
            name=f"Historical {revenue_col}", line=dict(color=CHART_COLORWAY[0]),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
        ))

    # Forecast line
    x_axis = forecast[forecast_date_col] if forecast_date_col else forecast.index
    fig.add_trace(go.Scatter(
        x=x_axis, y=forecast[forecast_val_col],
        name="Forecast", line=dict(color=CHART_COLORWAY[2], width=2.5, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>Forecast %{y:,.0f}<extra></extra>",
    ))

    # Prediction interval
    if upper_col and lower_col and upper_col in forecast.columns and lower_col in forecast.columns:
        fig.add_trace(go.Scatter(
            x=list(x_axis) + list(x_axis[::-1]),
            y=list(forecast[upper_col]) + list(forecast[lower_col][::-1]),
            fill="toself", fillcolor="rgba(245,158,11,0.16)", line=dict(color="rgba(0,0,0,0)"),
            name="90% Prediction Interval", showlegend=True,
        ))

    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, l=10, r=10, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
                      yaxis_title=revenue_col)
    fig.update_yaxes(tickformat="~s")
    # Upgrade PART 05: chart title, time-axis title and a unit-aware value-axis
    # title (currency only when the active dataset's currency is known).
    from dashboard.components.chart_labels import time_axis_title, value_axis_title
    st.plotly_chart(apply_chart_style(
        fig, title=f"{revenue_col}: history and forecast", legend_bottom=True,
        x_title=time_axis_title(x_axis), y_title=value_axis_title(revenue_col),
    ), use_container_width=True)

    _render_evaluation(ev)

    st.markdown('<div class="mp-section-title">Forecast Table</div>', unsafe_allow_html=True)
    money = [] if session_paths.is_user_dataset_mode() else [c for c in (forecast_val_col, lower_col, upper_col) if c]
    show_table(forecast, currency_cols=money)
