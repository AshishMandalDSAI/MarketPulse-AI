"""Shared UI building blocks for the Phase 8 modules (14-17).

Gives the four pages one consistent layout: Module Overview, Dataset
Information, Analytical Controls, Results, Interactive Visualizations,
Business Interpretation, Downloadable Reports and Data Availability
Messages. Reuses the existing dark theme (``dashboard.styles.theme``), KPI
cards and dataset resolver; nothing here hardcodes a column name.

Session safety: everything a page keeps between reruns goes through
``cache_get`` / ``cache_set`` / ``widget_key``, which scope values to the
active dataset id and keep keys under the ``mp_page_`` prefix that
``session_paths`` wipes on every dataset switch or reset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components import session_paths
from dashboard.components.data_loader import _safe_read_csv
from dashboard.components.kpi_card import render_kpi_row
from dashboard.styles.theme import PLOTLY_TEMPLATE, CHART_COLORWAY
from src.common import session_scope
from src.common.availability import Availability, check_availability
from src.common.dataset_summary import DatasetSummary, summarize_dataset
from src.common.export import ExportError, dataframe_to_csv_bytes, sheets_to_excel_bytes
from src.common.formatting import escape_text, fmt_int, fmt_pct
from src.common.modules import ModuleSpec

__all__ = [
    "widget_key", "cache_get", "cache_set", "load_active_dataframe", "get_summary",
    "render_module_page", "render_section", "render_plain_english", "render_limitations",
    "render_unavailable", "render_download_buttons", "apply_dark_layout",
    "PLOTLY_TEMPLATE", "CHART_COLORWAY",
]


# --------------------------------------------------------------------------
# Dataset-scoped session helpers
# --------------------------------------------------------------------------

def widget_key(module_key: str, name: str) -> str:
    """Prefixed key for a Streamlit widget, so its state is cleared with the
    rest of the page state on dataset switch/reset."""
    return session_scope.scoped_key(module_key, name)


def cache_get(module_key: str, name: str, default=None):
    return session_scope.get_scoped(
        st.session_state, session_paths.get_dataset_id(), module_key, name, default)


def cache_set(module_key: str, name: str, value) -> None:
    session_scope.set_scoped(
        st.session_state, session_paths.get_dataset_id(), module_key, name, value)


def _analysis_table_path(processed_dir: Path) -> Optional[Path]:
    """Same file preference as the platform's loader: the demo's cleaned table,
    else the standardized table written for every uploaded dataset."""
    for name in ("marketing_data_clean.csv", "standardized_data.csv"):
        path = processed_dir / name
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def _read_analysis_table(dataset_id: str, path_str: str, mtime_ns: int, date_col: Optional[str]) -> pd.DataFrame:
    """Cached read keyed by dataset id AND file version.

    ``load_marketing_data`` (used by pages 00-13) is cached without any
    argument, so its cache is only correct if every dataset switch clears it.
    Streamlit's cache is shared by all browser sessions of a server process, so
    the Phase 8 pages use this loader instead: a different dataset (or a
    re-processed file) is a different cache entry by construction.
    """
    df = _safe_read_csv(Path(path_str))
    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception:
            pass
    return df


def load_active_dataframe() -> Optional[pd.DataFrame]:
    """The active dataset's analysis table, or None when unavailable."""
    path = _analysis_table_path(session_paths.get_processed_dir())
    if path is None:
        return None
    df = _read_analysis_table(session_paths.get_dataset_id(), str(path),
                              path.stat().st_mtime_ns, session_paths.get_date_column())
    if df is None or df.empty:
        return None
    return df


def get_summary(module_key: str, df: pd.DataFrame) -> DatasetSummary:
    """Dataset summary, computed once per dataset and kept in scoped state."""
    cached = cache_get(module_key, "dataset_summary")
    if isinstance(cached, DatasetSummary) and cached.n_rows == len(df) and cached.n_cols == df.shape[1]:
        return cached
    summary = summarize_dataset(df)
    cache_set(module_key, "dataset_summary", summary)
    return summary


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------

def render_section(title: str) -> None:
    st.markdown(f'<div class="mp-section-title">{escape_text(title)}</div>', unsafe_allow_html=True)


def render_plain_english(text: str, title: str = "In simple words") -> None:
    st.markdown(
        f'<div class="mp-callout"><strong>{escape_text(title)}</strong><br/>{escape_text(text)}</div>',
        unsafe_allow_html=True)


def render_limitations(items) -> None:
    items = [i for i in (items or []) if i]
    if not items:
        return
    with st.expander("Limitations & caveats"):
        for item in items:
            st.markdown(f"- {item}")


def render_unavailable(title: str, reasons, missing=None) -> None:
    """Data Availability Message: what cannot be done, why, and what is needed."""
    st.warning(f"**{title} cannot run on this dataset.**")
    for r in reasons or []:
        st.markdown(f"- {r}")
    if missing:
        st.markdown("**What would make it possible:**")
        for m in missing:
            st.markdown(f"- {m}")


def apply_dark_layout(fig, title: str = "", height: int = 380, *, x_title: Optional[str] = None,
                      y_title: Optional[str] = None):
    """Apply the platform's Plotly dark styling to a figure (with axis titles, PART 01)."""
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title, height=height,
                      margin=dict(t=48 if title else 24, l=10, r=10, b=10),
                      colorway=CHART_COLORWAY)
    from dashboard.styles.theme import apply_chart_style
    return apply_chart_style(fig, x_title=x_title, y_title=y_title)


def render_download_buttons(module_key: str, stem: str,
                            table: Optional[pd.DataFrame] = None,
                            sheets: Optional[Mapping[str, pd.DataFrame]] = None) -> None:
    """CSV (single table) and Excel (workbook of tables) download buttons.

    Only exports data the caller passes in, i.e. results that were actually
    calculated. If a format cannot be produced the reason is shown instead of
    the button silently disappearing.
    """
    cols = st.columns(2)
    if table is not None:
        try:
            cols[0].download_button(
                "⬇️ Download CSV", data=dataframe_to_csv_bytes(table),
                file_name=f"{stem}.csv", mime="text/csv", key=widget_key(module_key, f"dl_csv_{stem}"))
        except ExportError as exc:
            cols[0].caption(f"CSV unavailable: {exc}")
    workbook = sheets if sheets else ({"Summary": table} if table is not None else None)
    if workbook:
        try:
            cols[1].download_button(
                "⬇️ Download Excel", data=sheets_to_excel_bytes(workbook),
                file_name=f"{stem}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=widget_key(module_key, f"dl_xlsx_{stem}"))
        except ExportError as exc:
            cols[1].caption(f"Excel unavailable: {exc}")


# --------------------------------------------------------------------------
# Page scaffold
# --------------------------------------------------------------------------

def _header_badge(spec: ModuleSpec) -> str:
    """Menu position of the page that hosts the module (PART 01). spec.number is the
    module's permanent identity (14-17) and no longer equals its menu position."""
    try:
        from dashboard import navigation
        nav_label, tab = navigation.page_for_module(spec.key)
        pos = nav_label.split(" · ", 1)[0]
        return f"PAGE {pos} · STATISTICS" if tab else f"PAGE {pos}"
    except Exception:
        return f"MODULE {spec.number}"


def _render_header(spec: ModuleSpec) -> None:
    st.markdown(f'<span class="mp-header-badge">{_header_badge(spec)}</span>', unsafe_allow_html=True)
    st.title(f"{spec.icon} {spec.title}")
    st.caption(spec.tagline)


def _render_overview(spec: ModuleSpec) -> None:
    render_section("Module Overview")
    st.markdown(spec.overview)
    with st.expander("What this module provides"):
        for cap in spec.capabilities:
            st.markdown(f"- {cap}")


def _render_dataset_info(summary: DatasetSummary) -> None:
    render_section("Dataset Information")
    active = session_paths.get_active_dataset()
    if active:
        name = "NovaMart (demo)" if active.get("is_demo") else (active.get("filename") or "Uploaded dataset")
        st.caption(f"Active dataset: **{escape_text(name)}**")
    render_kpi_row([
        {"label": "Rows", "value": fmt_int(summary.n_rows)},
        {"label": "Columns", "value": fmt_int(summary.n_cols)},
        {"label": "Numeric", "value": fmt_int(len(summary.numeric_columns))},
        {"label": "Categorical", "value": fmt_int(len(summary.categorical_columns))},
        {"label": "Missing cells", "value": fmt_pct(summary.missing_pct, 2)},
    ])
    extras = []
    if summary.datetime_columns:
        extras.append(f"{len(summary.datetime_columns)} date column(s)")
    if summary.boolean_columns:
        extras.append(f"{len(summary.boolean_columns)} yes/no column(s)")
    if summary.identifier_columns:
        extras.append(f"{len(summary.identifier_columns)} identifier-like column(s)")
    if summary.constant_columns:
        extras.append(f"{len(summary.constant_columns)} constant column(s)")
    if summary.duplicate_rows:
        extras.append(f"{summary.duplicate_rows:,} duplicate row(s)")
    if extras:
        st.caption("Also detected: " + ", ".join(extras) + ".")
    with st.expander("Column details"):
        st.dataframe(summary.column_table(), use_container_width=True, hide_index=True)


def render_module_page(spec: ModuleSpec,
                       body_fn: Optional[Callable[[pd.DataFrame, DatasetSummary], None]] = None,
                       *, embedded: bool = False) -> None:
    """Render a Phase 8 module page.

    ``body_fn(df, summary)`` draws the module's Analytical Controls, Results,
    Visualizations, Interpretation and Downloads. While a module's analytics
    have not been delivered yet (``body_fn is None``) the page shows the
    overview and real dataset information plus an explicit status notice, and
    no results of any kind.
    """
    if embedded:
        # Inside a tab of the Statistics workspace: the workspace draws the page
        # title; each tab keeps its own heading, overview, data, gate and body.
        from dashboard import navigation
        st.subheader(navigation.stats_tab_label(spec.key))
        st.caption(spec.tagline)
    else:
        _render_header(spec)
    _render_overview(spec)

    df = load_active_dataframe()
    if df is None:
        render_section("Data Availability")
        render_unavailable(spec.title, ["No analysis table could be loaded for the active dataset."],
                           ["Upload a CSV/XLSX file on page 00, or switch to the demo dataset."])
        return

    summary = get_summary(spec.key, df)
    _render_dataset_info(summary)

    availability: Availability = check_availability(spec.key, summary)
    if not availability.available:
        render_section("Data Availability")
        render_unavailable(spec.title, availability.reasons, availability.missing)
        return

    if body_fn is None:
        render_section("Module Status")
        st.info(
            f"The analytical controls and results for **{spec.title}** are delivered in "
            f"Part {spec.delivered_in_part} of Phase 8. Nothing has been calculated for this "
            "module yet, so no results are shown.")
        return

    body_fn(df, summary)
