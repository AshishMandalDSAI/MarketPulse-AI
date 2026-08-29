"""Reusable KPI card renderer for the MarketPulse AI dashboard."""

import streamlit as st


def render_kpi_card(label: str, value: str, delta: str = None, positive: bool = True):
    delta_html = ""
    if delta is not None:
        cls = "mp-kpi-delta-pos" if positive else "mp-kpi-delta-neg"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'

    st.markdown(
        f"""
        <div class="mp-kpi-card">
            <div class="mp-kpi-label">{label}</div>
            <div class="mp-kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(cards: list):
    """cards: list of dicts with keys label, value, delta (optional), positive (optional)."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_kpi_card(
                card["label"], card["value"],
                card.get("delta"), card.get("positive", True),
            )


def format_inr(value: float, compact: bool = True) -> str:
    if compact:
        if abs(value) >= 1e7:
            return f"₹{value / 1e7:.2f}Cr"
        if abs(value) >= 1e5:
            return f"₹{value / 1e5:.2f}L"
    return f"₹{value:,.0f}"
