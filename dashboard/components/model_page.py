"""Shared frame for the four PART 02 marketing-model pages (04-07).

Every page shows, in this order: dataset name and size, the detected analytics
mode with its evidence, the column roles and target (reviewable), the target in
use, what this model can and cannot do with this dataset (with reasons and the
missing information), the model results, methodology, assumptions and
warnings. Model errors are shown on the page, never swallowed.
"""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import traceback
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import navigation
from dashboard.components import currency as CUR
from dashboard.components import role_mapping_ui as RMU
from dashboard.components import session_paths
from dashboard.components.stats_ui import apply_dark_layout, load_active_dataframe
from dashboard.components.ui import callout, empty_state, page_header
from src.common.formatting import escape_text
from src.data_engine import role_engine as RE
from src.marketing_intel.capabilities import MODELS, model_capabilities
from src.marketing_intel.context import ModelContext, build_context
from src.marketing_intel.effectiveness import EffectivenessResult, run_effectiveness

_MEMO = "mp_page_mi_memo"
UNIT_WORD = {"units": "units", "orders": "orders", "conversions": "conversions", "leads": "leads",
             "customers": "customers"}


# --------------------------------------------------------------------------- small helpers
def section(title: str) -> None:
    st.markdown(f'<div class="mp-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def bullet_list(items) -> None:
    items = [i for i in (items or []) if i]
    if items:
        st.markdown("\n".join(f"- {escape_text(str(i))}" for i in items))


def target_unit(ctx: ModelContext) -> str:
    """Unit label for the target: the dataset currency for monetary targets (none when unknown)."""
    if ctx.target_is_monetary:
        return CUR.unit_label()
    return UNIT_WORD.get(RE.ROLE_INFO[ctx.target_role][2], RE.ROLE_INFO[ctx.target_role][2])


def target_axis(ctx: ModelContext, prefix: str = "") -> str:
    u = target_unit(ctx)
    return f"{prefix}{ctx.target}" + (f" ({u})" if u else "")


def per_word(ctx: ModelContext) -> str:
    return {"daily": "day", "weekly": "week", "monthly": "month", "quarterly": "quarter"}.get(ctx.frequency, "period")


def memo(name: str, sig, fn: Callable):
    """Session memo keyed by dataset + signature (cleared with the other page caches on dataset switch)."""
    store = st.session_state.setdefault(_MEMO, {})
    key = name + ":" + session_paths.get_dataset_id() + ":" + hashlib.sha1(
        json.dumps(sig, sort_keys=True, default=str).encode(), usedforsecurity=False).hexdigest()
    if key not in store:
        if len(store) > 40:
            store.clear()
        store[key] = fn()
    return store[key]


def effectiveness_for(ctx: ModelContext, drivers: Optional[list] = None, fixed_effects: bool = True,
                      trend: bool = True) -> EffectivenessResult:
    sig = ["eff", RMU.mapping_signature(ctx.mapping), ctx.target, drivers, fixed_effects, trend, len(ctx.df)]
    return memo("eff", sig, lambda: run_effectiveness(ctx, drivers, fixed_effects=None if fixed_effects else [],
                                                      trend=trend))


def legacy_toggle(nav_key: str, legacy_module: str, legacy_name: str) -> None:
    """The previous module stays reachable, unchanged and clearly labelled (as in PART 01)."""
    title = next(p.title for p in navigation.NAV_PAGES if p.key == nav_key)
    st.markdown('<div class="mp-section-title">Previous module</div>', unsafe_allow_html=True)
    st.caption(f"The earlier **{legacy_name}** module is unchanged and is *not* the {title}.")
    if st.toggle(f"Show the previous {legacy_name} module (unchanged)", value=False, key=f"mp_interim_show_{nav_key}"):
        with st.container(border=True):
            st.caption(f"Previous module · {legacy_name} · not the {title}")
            importlib.import_module(legacy_module).render()


def render_capabilities(caps: list) -> None:
    ok = [c for c in caps if c.available]
    no = [c for c in caps if not c.available]
    a, b = st.columns(2)
    with a:
        st.markdown("**Available for this dataset**")
        if ok:
            st.markdown("\n".join(f"- ✅ {escape_text(c.label)}"
                                  + (f" — *{escape_text(c.conditional)}*" if c.conditional else "") for c in ok))
        else:
            st.caption("Nothing in this model is supported by the mapped columns.")
    with b:
        st.markdown("**Not available, and why**")
        if no:
            st.markdown("\n".join(f"- ⛔ {escape_text(c.label)} — {escape_text(c.reason)}"
                                  + (f". *Needs: {escape_text(c.needs)}*" if c.needs else "") for c in no))
        else:
            st.caption("Everything in this model is supported.")


def bar(x, y, title: str, x_title: str, y_title: str, *, color=None, error=None, text=None, horizontal=False,
        height=None) -> go.Figure:
    kw = {}
    if error is not None:
        kw["error_y" if not horizontal else "error_x"] = dict(type="data", array=error, visible=True)
    if horizontal:
        fig = go.Figure(go.Bar(x=y, y=x, orientation="h", marker_color=color, text=text, **kw))
        return apply_dark_layout(fig, title, height or max(300, 34 * len(x) + 120), x_title=y_title, y_title=x_title)
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=color, text=text, **kw))
    return apply_dark_layout(fig, title, height, x_title=x_title, y_title=y_title)


def heatmap(frame: pd.DataFrame, index_col: str, value_cols: list, labels: dict, title: str, y_title: str,
            colorbar: str = "Spearman ρ") -> go.Figure:
    z = frame[value_cols].to_numpy(float)
    fig = go.Figure(go.Heatmap(z=z, x=[labels.get(c, c) for c in value_cols], y=frame[index_col].astype(str),
                               zmin=-1, zmax=1, colorscale="RdBu", colorbar=dict(title=colorbar),
                               text=np.round(z, 2), texttemplate="%{text}"))
    return apply_dark_layout(fig, title, max(320, 40 * len(frame) + 160), x_title="Channel", y_title=y_title)


def show_error(what: str, exc: Exception) -> None:
    """Visible, specific error (never hidden behind a generic 'not available')."""
    st.error(f"{what} failed: {type(exc).__name__}: {exc}")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])


# --------------------------------------------------------------------------- frame
def render_model_page(nav_key: str, purpose: str, model_key: str,
                      body: Callable[[ModelContext, list], None], legacy: tuple) -> None:
    pos = navigation.label(nav_key).split(" · ", 1)[0]
    title = next(p.title for p in navigation.NAV_PAGES if p.key == nav_key)
    name = RMU.dataset_name()
    page_header(f"{pos} · Marketing models", title, purpose, context=f"Dataset: **{name}**")
    df = load_active_dataframe()
    if df is None:
        empty_state("No data available", "The active dataset has no analysis table yet.",
                    "Upload a file on the Upload page, or use the demo dataset.")
        legacy_toggle(nav_key, *legacy)
        return
    m = RMU.get_mapping(df)
    mode = RE.detect_mode(m)
    c = st.columns(4)
    c[0].metric("Dataset", name if len(name) <= 22 else name[:20] + "…")
    c[1].metric("Rows × columns", f"{len(df):,} × {df.shape[1]}")
    c[2].metric("Analytics mode", {"spend": "Spend-based", "exposure": "Exposure-based", "general": "General"}[mode.mode])
    c[3].metric("Mapping", f"v{m.version}" if m.version else ("demo preset" if m.source == "demo-preset" else "suggested"))
    RMU.mode_banner(mode)
    need_target = bool(m.target_candidates) and not m.targets
    RMU.render_mapping_panel(df, m, model_key, expanded=need_target)
    if not m.targets:
        caps = model_capabilities(m, None)[model_key]
        callout("Confirm the analysis target in **Column roles and target** above (and save the mapping) to run "
                f"the {title}." if m.target_candidates else
                "No outcome column is mapped. Set a column's role to an outcome above to run this model.",
                kind="warn", label="Target needed")
        section("What this model can do with this dataset")
        render_capabilities(caps)
        legacy_toggle(nav_key, *legacy)
        return
    if len(m.targets) > 1:
        target = st.selectbox("Target analysed on this page", m.targets, key=f"mp_page_target_{model_key}",
                              help="Each confirmed target is analysed independently.")
    else:
        target = m.targets[0]
        st.caption(f"Target analysed: **{escape_text(target)}** ({escape_text(m.columns[target].label)}).")
    caps = model_capabilities(m, target)[model_key]
    section("What this model can do with this dataset")
    render_capabilities(caps)
    try:
        cur_code = CUR.unit_label() or None
        ctx = build_context(df, m, target, cur_code)
    except ValueError as exc:
        callout(str(exc), kind="bad", label="Cannot analyse this target")
        legacy_toggle(nav_key, *legacy)
        return
    try:
        body(ctx, caps)
    except Exception as exc:  # shown, not hidden
        show_error(f"The {title}", exc)
    legacy_toggle(nav_key, *legacy)


def model_quality_metrics(model) -> None:
    h = model.holdout or {}
    cols = st.columns(4)
    r2 = h.get("period_r2") if np.isfinite(h.get("period_r2", np.nan)) else h.get("r2")
    cols[0].metric("Holdout R²", "—" if r2 is None or not np.isfinite(r2) else f"{r2:.2f}",
                   help="Share of variation in held-out (later) periods explained by a model fitted on earlier periods.")
    mp = h.get("period_mape") if h.get("period_mape") is not None else h.get("mape")
    cols[1].metric("Holdout MAPE", "—" if mp is None else f"{mp:.1f}%")
    cols[2].metric("Rows used", f"{model.n_rows:,}")
    cols[3].metric("Model quality", model.quality.title())
    st.caption(escape_text(f"Validation: {model.split}. {model.quality_reason}."))
