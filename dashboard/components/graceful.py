"""Shared graceful-degradation helpers for dashboard pages.

Rather than rewriting every legacy page's internals (which hardcode the
built-in demo dataset's columns and remain fully correct for datasets that
DO have those columns), each page's render() is wrapped so a KeyError /
EmptyDataError / missing-capability situation shows a clear
"not available for this dataset" message + a generic fallback view instead
of an unhandled traceback.
"""

import functools

import pandas as pd
import streamlit as st

from dashboard.components import session_paths
from dashboard.components.data_loader import load_marketing_data, load_generic_kpis, has_capability, capability_reason
from dashboard.styles.theme import apply_chart_style


def render_generic_fallback(feature_name: str, reason: str = "") -> None:
    st.warning(f"**{feature_name} is not available for this dataset.**" + (f"\n\n{reason}" if reason else ""))

    kpis = load_generic_kpis()
    if kpis:
        st.subheader("Generic KPIs for this dataset")
        cols = st.columns(min(4, len(kpis)))
        for i, kpi in enumerate(kpis[:4]):
            cols[i % len(cols)].metric(kpi.get("label", ""), kpi.get("value", ""))

    df = load_marketing_data()
    if df is None or df.empty:
        return

    date_col = session_paths.get_date_column()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if date_col and date_col in df.columns and numeric_cols:
        try:
            import plotly.graph_objects as go
            target = numeric_cols[0]
            from dashboard.components import chart_labels as CL
            plot_df, applied = CL.per_period(df, date_col, target)
            if not plot_df.empty:
                st.subheader(f"Trend: {target}")
                fig = go.Figure(go.Scatter(x=plot_df[date_col], y=plot_df[target], mode="lines"))
                fig.update_layout(title=CL.chart_title(f"{target} Trend", applied, target))
                st.plotly_chart(apply_chart_style(fig, x_title=CL.time_axis_title(plot_df[date_col]),
                                                  y_title=CL.value_axis_title(target, applied)),
                                use_container_width=True)
        except Exception:
            pass

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 50]
    if cat_cols:
        try:
            cat = cat_cols[0]
            st.subheader(f"Breakdown by {cat}")
            st.bar_chart(df[cat].value_counts())
        except Exception:
            pass


def safe_page(feature_name: str, required_capability: str = None):
    """Decorator for a page's render() function. Catches KeyError/
    EmptyDataError/other data-shape exceptions and shows a graceful
    fallback instead of crashing. If required_capability is given and the
    active dataset's capabilities say it's unavailable, skips straight to
    the fallback without even attempting the legacy render."""

    def decorator(render_fn):
        @functools.wraps(render_fn)
        def wrapper(*args, **kwargs):
            if required_capability and not has_capability(required_capability):
                render_generic_fallback(feature_name, capability_reason(required_capability))
                return
            try:
                return render_fn(*args, **kwargs)
            except (KeyError, pd.errors.EmptyDataError) as e:
                render_generic_fallback(feature_name, f"({type(e).__name__}: {e})")
            except Exception as e:
                render_generic_fallback(feature_name, f"Unexpected error: {e}")
        return wrapper
    return decorator
