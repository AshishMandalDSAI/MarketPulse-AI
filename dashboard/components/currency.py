"""Active dataset currency for display (upgrade PART 01).

* Demo dataset: INR (it is generated in INR; ``DEMO_CURRENCY``).
* Uploaded dataset: what the Upload Data page stored in ``currency.json``
  (auto-detected or chosen by the user). Missing or unidentified: no currency,
  so values are shown without any symbol. INR is never assumed for uploads.
"""
from __future__ import annotations

import math
from typing import Optional

from src.data_engine.currency import DEMO_CURRENCY, load_currency


_DEMO_MONEY = {"Revenue", "Total_Marketing_Spend", "Spend", "Current_Budget", "Recommended_Budget",
               "Estimated_Contribution", "Forecast_Revenue", "Lower_90", "Upper_90"}


def active_currency() -> Optional[dict]:
    """``{"code", "symbol", "name", "source", "monetary_columns"}`` or None."""
    from dashboard.components import session_paths
    try:
        if not session_paths.is_user_dataset_mode():
            return dict(DEMO_CURRENCY)
        info = load_currency(session_paths.get_processed_dir())
    except Exception:
        return None
    return info if info and info.get("symbol") else None


def symbol() -> str:
    info = active_currency()
    return info["symbol"] if info else ""


def unit_label() -> str:
    """Axis / column unit for monetary values: "INR", "USD", "$" (code unknown) or ""."""
    info = active_currency()
    if not info:
        return ""
    return info.get("code") or info["symbol"]


def is_monetary(column: Optional[str]) -> bool:
    """True when ``column`` holds money for the active dataset (and a currency is known)."""
    info = active_currency()
    if not info or not column:
        return False
    cols = info.get("monetary_columns")
    if cols is None:
        # Demo (NovaMart, INR): its money columns are revenue, spend and budget fields.
        name = str(column)
        return name in _DEMO_MONEY or name.endswith(("_Spend", "_Budget", "_Revenue")) or name.startswith("Revenue")
    return str(column) in set(map(str, cols))


def money(value, compact: bool = True, decimals: int = 0) -> str:
    """Format a monetary value with the active currency.

    INR keeps the Indian compact style used across the demo (₹2.22Cr / ₹4.50L);
    other currencies use K / M / B. Without a known currency no symbol is shown.
    Non-finite values render as an em dash, never as 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    info = active_currency()
    sym = info["symbol"] if info else ""
    if info and info.get("code") == "INR":
        if compact:
            if abs(v) >= 1e7:
                return f"{sym}{v / 1e7:.2f}Cr"
            if abs(v) >= 1e5:
                return f"{sym}{v / 1e5:.2f}L"
        return f"{sym}{v:,.{decimals}f}"
    if compact:
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(v) >= div:
                return f"{sym}{v / div:,.2f}{suf}"
    return f"{sym}{v:,.{decimals}f}"
