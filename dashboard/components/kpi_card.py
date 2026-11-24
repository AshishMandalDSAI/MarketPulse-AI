"""Reusable KPI card renderer for the MarketPulse AI dashboard.

A card shows: metric name, value, an optional comparison (delta) and an
optional one-line interpretation. Pass ``delta=None`` when no baseline exists:
the card then shows no comparison at all rather than an invented one.
"""

import math

import streamlit as st


def _esc(text) -> str:
    """Neutralise markup (< >) while keeping text such as '&' byte-identical."""
    return str(text).replace("<", "&lt;").replace(">", "&gt;")


def render_kpi_card(label: str, value: str, delta: str = None, positive: bool = True,
                    help: str = None, neutral: bool = False):
    delta_html = ""
    if delta is not None:
        if neutral:
            delta_html = f'<div class="mp-kpi-delta-flat">{_esc(delta)}</div>'
        else:
            cls = "mp-kpi-delta-pos" if positive else "mp-kpi-delta-neg"
            arrow = "▲" if positive else "▼"
            delta_html = f'<div class="{cls}">{arrow} {_esc(delta)}</div>'
    help_html = f'<div class="mp-kpi-help">{_esc(help)}</div>' if help else ""

    st.markdown(
        f"""
        <div class="mp-kpi-card">
            <div class="mp-kpi-label">{_esc(label)}</div>
            <div class="mp-kpi-value">{_esc(value)}</div>
            {delta_html}{help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(cards: list):
    """cards: list of dicts with keys label, value, and optional delta,
    positive, help, neutral."""
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_kpi_card(
                card["label"], card["value"],
                card.get("delta"), card.get("positive", True),
                card.get("help"), card.get("neutral", False),
            )


def format_inr(value: float, compact: bool = True) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    if compact:
        if abs(v) >= 1e7:
            return f"₹{v / 1e7:.2f}Cr"
        if abs(v) >= 1e5:
            return f"₹{v / 1e5:.2f}L"
    return f"₹{v:,.0f}"
