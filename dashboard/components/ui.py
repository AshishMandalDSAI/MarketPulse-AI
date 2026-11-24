"""Shared UI building blocks for MarketPulse AI pages.

Small, dependency-free helpers so every page formats numbers, callouts,
empty states and tables the same way. All text passed in is HTML-escaped;
only the inline markdown subset ``**bold**`` and ``*italic*`` is converted
(raw HTML blocks don't render Markdown, which previously left literal
asterisks on screen).
"""

from __future__ import annotations

import html
import math
import re
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])")
_CODE = re.compile(r"`([^`]+)`")


def md_inline(text) -> str:
    """Escape ``text`` and convert **bold**, *italic* and `code` to HTML."""
    out = html.escape(str(text), quote=False)
    out = _BOLD.sub(r"<b>\1</b>", out)
    out = _ITALIC.sub(r"<i>\1</i>", out)
    out = _CODE.sub(r"<code>\1</code>", out)
    return out


def callout(text, kind: str = "info", label: Optional[str] = None) -> None:
    """Render a design-system callout. kind: info | good | warn | bad."""
    cls = {"good": " mp-callout-good", "warn": " mp-callout-warn", "bad": " mp-callout-bad"}.get(kind, "")
    head = f'<div class="mp-callout-label">{html.escape(label)}</div>' if label else ""
    st.markdown(f'<div class="mp-callout{cls}">{head}{md_inline(text)}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str = "", action_hint: str = "") -> None:
    parts = [f'<div class="mp-empty-title">{html.escape(title)}</div>']
    if body:
        parts.append(f"<div>{md_inline(body)}</div>")
    if action_hint:
        parts.append(f'<div style="margin-top:6px;">{md_inline(action_hint)}</div>')
    st.markdown(f'<div class="mp-empty">{"".join(parts)}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="mp-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def page_header(badge: str, title: str, subtitle: str = "", context: str = "") -> None:
    """Consistent page header: badge, title, one-line purpose, dataset context."""
    sub = f'<div class="mp-page-subtitle">{md_inline(subtitle)}</div>' if subtitle else ""
    ctx = f'<div class="mp-page-context">{md_inline(context)}</div>' if context else ""
    st.markdown(
        f'<div class="mp-page-header"><span class="mp-header-badge">{html.escape(badge)}</span>'
        f'<div class="mp-page-title">{html.escape(title)}</div>{sub}{ctx}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- numbers

def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt_currency(value, symbol: str = "₹", compact: bool = True) -> str:  # kept for compatibility; prefer currency.money()
    """Indian-style compact currency (Cr / L) or full with separators.
    Non-finite / missing values render as an em dash, never as 0."""
    if not _finite(value):
        return "—"
    v = float(value)
    if compact:
        if abs(v) >= 1e7:
            return f"{symbol}{v / 1e7:,.2f} Cr"
        if abs(v) >= 1e5:
            return f"{symbol}{v / 1e5:,.2f} L"
    return f"{symbol}{v:,.0f}"


def fmt_number(value, decimals: int = 0) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def fmt_pct(value, decimals: int = 1, signed: bool = False) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):{'+' if signed else ''},.{decimals}f}%"


# ---------------------------------------------------------------- tables

def _pretty_header(name: str) -> str:
    s = str(name).replace("_", " ").strip()
    s = re.sub(r"\bPct\b", "%", s, flags=re.I)
    s = re.sub(r"\bRoi\b", "ROI", s, flags=re.I)
    s = re.sub(r"\bRoas\b", "ROAS", s, flags=re.I)
    return s[:1].upper() + s[1:] if s else s


def show_table(df: pd.DataFrame, *, currency_cols: Iterable[str] = (), pct_cols: Iterable[str] = (),
               int_cols: Iterable[str] = (), decimals: int = 2, height: Optional[int] = None,
               hide_index: bool = True) -> None:
    """Render a dataframe with readable headers and consistent number formats.
    The underlying values are unchanged (sorting and downloads stay exact)."""
    currency_cols, pct_cols, int_cols = set(currency_cols), set(pct_cols), set(int_cols)
    cfg = {}
    for col in df.columns:
        label = _pretty_header(col)
        if col in currency_cols:
            # Active dataset currency (demo: ₹); no symbol when it is unknown (PART 01).
            from dashboard.components.currency import symbol as _cur_symbol
            sym = _cur_symbol().replace("%", "%%")
            cfg[col] = st.column_config.NumberColumn(label, format=f"{sym}%,.0f")
        elif col in pct_cols:
            cfg[col] = st.column_config.NumberColumn(label, format="%.1f%%")
        elif col in int_cols:
            cfg[col] = st.column_config.NumberColumn(label, format="%,d")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            ser = df[col].dropna()
            date_only = ser.empty or bool((ser == ser.dt.normalize()).all())
            cfg[col] = st.column_config.DatetimeColumn(label, format="YYYY-MM-DD" if date_only else "YYYY-MM-DD HH:mm")
        elif pd.api.types.is_float_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(label, format=f"%,.{decimals}f")
        else:
            cfg[col] = st.column_config.Column(label)
    kwargs = {"use_container_width": True, "hide_index": hide_index, "column_config": cfg}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)
