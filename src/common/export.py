"""CSV / Excel export helpers shared by the Phase 8 modules.

Exports are built only from DataFrames the caller supplies (i.e. results that
were actually calculated). Text cells that a spreadsheet program could
interpret as a formula (leading ``= + - @``) are prefixed with an apostrophe
so a value from an uploaded dataset can never execute when the file is opened.
"""

from __future__ import annotations

import io
import re
from typing import Mapping

import pandas as pd

EXCEL_MAX_ROWS = 1_048_575  # header row takes one of Excel's 1,048,576 rows
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_BAD_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


class ExportError(ValueError):
    """Raised when a requested export cannot be produced."""


def sanitize_for_spreadsheet(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with formula-like text cells neutralized."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            out[col] = series.map(
                lambda v: "'" + v if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES) else v
            )
    return out


def dataframe_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ExportError("Nothing to export: no table was provided.")
    return sanitize_for_spreadsheet(df).to_csv(index=index).encode("utf-8")


def _safe_sheet_name(name: str, used: set) -> str:
    base = _BAD_SHEET_CHARS.sub("_", str(name)).strip().strip("'") or "Sheet"
    base = base[:31]
    candidate, n = base, 2
    while candidate.lower() in used:
        suffix = f"_{n}"
        candidate = base[: 31 - len(suffix)] + suffix
        n += 1
    used.add(candidate.lower())
    return candidate


def sheets_to_excel_bytes(sheets: Mapping[str, pd.DataFrame], index: bool = False) -> bytes:
    """Write several DataFrames to one .xlsx workbook and return its bytes."""
    if not sheets:
        raise ExportError("Nothing to export: no tables were provided.")
    # Validate everything BEFORE opening the writer: raising inside the writer
    # context makes openpyxl fail with a misleading "no visible sheet" error.
    for name, frame in sheets.items():
        if not isinstance(frame, pd.DataFrame):
            raise ExportError(f"Sheet '{name}' is not a table.")
        if len(frame) > EXCEL_MAX_ROWS:
            raise ExportError(
                f"Sheet '{name}' has {len(frame):,} rows, more than Excel's limit "
                f"of {EXCEL_MAX_ROWS:,}. Use the CSV download instead.")

    used: set = set()
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                sanitize_for_spreadsheet(frame).to_excel(
                    writer, sheet_name=_safe_sheet_name(name, used), index=index)
    except ImportError as exc:  # openpyxl missing
        raise ExportError("Excel export needs the 'openpyxl' package; use CSV instead.") from exc
    return buffer.getvalue()
