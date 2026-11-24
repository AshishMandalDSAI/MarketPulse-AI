"""Neutralise spreadsheet-formula text in COLUMN HEADERS (the shared export helper neutralises cell values only).

Dataset column names are user data; when a result table uses them as headers (for example a per-channel scenario table),
a name such as ``=HYPERLINK(...)`` would otherwise be written as a live formula."""

from __future__ import annotations

import pandas as pd

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutral_name(name) -> str:
    text = str(name)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def neutralize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose column headers (and string row labels) cannot be read as formulas."""
    out = df.copy()
    out.columns = [neutral_name(c) for c in out.columns]
    if not isinstance(out.index, pd.MultiIndex) and (pd.api.types.is_object_dtype(out.index.dtype) or pd.api.types.is_string_dtype(out.index.dtype)):
        out.index = [neutral_name(i) for i in out.index]
    return out
