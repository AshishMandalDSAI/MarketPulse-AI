"""Axis-label builder for the universal chart standard (upgrade PART 01).

Labels describe what is actually plotted: the variable (readable name), how it was
aggregated ("Total", "Average", "Number of records"), and its unit. Currency is
added only for monetary columns of a dataset whose currency is known; percentages
get "(%)"; time axes are named after their real frequency (Date / Week / Month /
Quarter / Year). Nothing is guessed: an unknown unit is simply left out.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

_PCT_TOKENS = re.compile(r"(^|[^a-z])(pct|percent|percentage|rate|ratio|share|ctr|cvr|margin_pct)([^a-z]|$)|%",
                         re.IGNORECASE)

AGG_WORD = {"sum": "Total", "total": "Total", "mean": "Average", "avg": "Average", "average": "Average",
            "median": "Median", "max": "Maximum", "min": "Minimum"}


def humanize(name) -> str:
    """'Net_Sales' -> 'Net Sales', 'totalMarketingSpend' -> 'Total Marketing Spend'.
    Existing capitals, acronyms and non-ASCII text are kept."""
    s = str(name)
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else s


def is_percent(name, series: Optional[pd.Series] = None) -> bool:
    """True when the column is a percentage or rate, judged from its name."""
    return bool(_PCT_TOKENS.search(str(name)))


def currency_unit(column) -> str:
    """'INR', 'USD', '$' (code unknown) or '' when the column is not money or the
    currency is unknown."""
    try:
        from dashboard.components import currency
        return currency.unit_label() if currency.is_monetary(column) else ""
    except Exception:
        return ""


def value_axis_title(column, aggregation: Optional[str] = None, *, unit: Optional[str] = None,
                     monetary: Optional[bool] = None) -> str:
    """Title for an axis showing values of ``column``.

    ``unit`` overrides the detected unit; ``monetary=True`` forces the currency (used
    for demo columns whose meaning is known), ``False`` suppresses it."""
    agg = (aggregation or "").lower()
    if agg == "count":
        return "Number of records"
    base = humanize(column)
    word = AGG_WORD.get(agg)
    text = f"{word} {base}" if word and not base.lower().startswith(word.lower()) else base
    if unit is None:
        if is_percent(column):
            unit = "%"
        elif monetary is True:
            from dashboard.components import currency
            unit = currency.unit_label()
        elif monetary is None:
            unit = currency_unit(column)
        else:
            unit = ""
    if unit and unit != "%":
        # "Net_Sales_USD" + "(USD)" would repeat the currency: drop it from the name.
        stripped = re.sub(rf"(?i)(?<![A-Za-z]){re.escape(unit)}(?![A-Za-z])", "", text)
        stripped = re.sub(r"\s+", " ", stripped.replace("()", "")).strip()
        if stripped:
            text = stripped
    return f"{text} ({unit})" if unit else text


def time_axis_title(dates) -> str:
    """'Date', 'Week', 'Month', 'Quarter' or 'Year', from the spacing of the dates."""
    try:
        d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().drop_duplicates().sort_values()
    except Exception:
        return "Date"
    if len(d) < 2:
        return "Date"
    gap = d.diff().dt.days.median()
    if 6 <= gap <= 8:
        return "Week"
    if 27 <= gap <= 32:
        return "Month"
    if 88 <= gap <= 93:
        return "Quarter"
    if 360 <= gap <= 370:
        return "Year"
    return "Date"


def category_axis_title(column) -> str:
    return humanize(column)


def chart_title(title: str, aggregation: Optional[str], value_column) -> str:
    """Make the recommender's title say how values were aggregated, e.g.
    'Net_Sales by Region' with mean -> 'Average Net Sales by Region'."""
    agg = (aggregation or "").lower()
    word = AGG_WORD.get(agg)
    if not word or value_column is None:
        return title
    raw = str(value_column)
    if title.startswith(raw):
        return f"{word} {humanize(raw)}{title[len(raw):]}"
    if f"by {raw}" in title or raw not in title:
        return title
    return title.replace(raw, f"{word.lower()} {humanize(raw)}", 1)


def per_period(df: pd.DataFrame, x_col, y_col, aggregation: str = "sum") -> tuple:
    """One point per time value, in chronological order. Row-level data with several
    rows per date would otherwise draw a zig-zag through every row.

    Returns (frame, applied) where ``applied`` is the aggregation really used
    ("sum", "mean" or "none" when every date was already unique) so labels can say
    "Total ..." only when values were actually summed."""
    plot_df = pd.DataFrame({x_col: df[x_col], y_col: pd.to_numeric(df[y_col], errors="coerce")}).dropna()
    applied = "none"
    if plot_df[x_col].duplicated().any():
        applied = "mean" if str(aggregation).lower() in ("mean", "avg", "average") else "sum"
        plot_df = plot_df.groupby(x_col, as_index=False)[y_col].agg(applied)
    return plot_df.sort_values(x_col), applied
