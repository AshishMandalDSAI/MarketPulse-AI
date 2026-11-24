"""Number formatting helpers shared by the Phase 8 modules.

All helpers return a plain-text placeholder ("—") for missing, NaN or
infinite values so a page never prints ``nan`` or crashes on bad input.
"""

from __future__ import annotations

import html
import math
from typing import Any

MISSING = "—"


def _as_float(value: Any):
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def fmt_number(value: Any, decimals: int = 4) -> str:
    """Format a number: thousands separators for large values, scientific
    notation for very small non-zero values, ``—`` when not a finite number."""
    v = _as_float(value)
    if v is None:
        return MISSING
    if v == 0:
        return "0"
    if abs(v) >= 1_000:
        return f"{v:,.{min(decimals, 2)}f}"
    if abs(v) < 10 ** (-decimals):
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    """Format a value that is already a percentage (12.5 -> '12.5%')."""
    v = _as_float(value)
    if v is None:
        return MISSING
    return f"{v:.{decimals}f}%"


def fmt_int(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return MISSING
    return f"{int(round(v)):,}"


def fmt_p_value(p: Any) -> str:
    """Format a p-value. Never prints '0.000' (a p-value is not exactly 0)."""
    v = _as_float(p)
    if v is None:
        return MISSING
    if v < 0 or v > 1:
        return MISSING
    if v < 0.001:
        return "< 0.001"
    return f"{v:.3f}"


def escape_text(value: Any) -> str:
    """HTML-escape arbitrary text (column names, category labels) before it is
    placed inside ``unsafe_allow_html`` markup."""
    return html.escape(str(value), quote=True)
