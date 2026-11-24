"""Turns a raw DataFrame + ColumnMapping into a StandardizedDataset.

Never raises on missing optional roles. Columns cleared by
clear_unconfirmed_low_confidence, or never mapped in the first place, are
reclassified into control_cols / dimension_cols / identifier_cols by their
actual dtype rather than being dropped.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import IDENTIFIER_CARDINALITY_RATIO, MAX_DIMENSION_CARDINALITY
from src.data_engine.canonical import _column_name_suggests_identifier
from src.data_engine.schemas import ColumnMapping, StandardizedDataset


def _classify_leftover_column(df: pd.DataFrame, col: str) -> str:
    """Return 'control' (numeric measure), 'dimension' (categorical) or
    'identifier' (near-unique) for a column with no confirmed semantic role."""
    n_rows = len(df)
    series = df[col]

    if pd.api.types.is_numeric_dtype(series):
        ratio = series.nunique(dropna=True) / n_rows if n_rows else 0
        non_null = series.dropna()
        is_integer_like = bool((non_null % 1 == 0).all()) if not non_null.empty else False
        if ratio > IDENTIFIER_CARDINALITY_RATIO and n_rows > 20 and is_integer_like:
            return "identifier"
        return "control"

    unique_count = series.nunique(dropna=True)
    ratio = unique_count / n_rows if n_rows else 0
    # Use column name hints + cardinality to distinguish identifiers from
    # high-cardinality dimensions. Columns whose name suggests an identifier
    # (e.g. "student_id") are classified as identifiers even when unique_count
    # ≤ MAX_DIMENSION_CARDINALITY. Otherwise, columns within the dimension
    # threshold are kept as dimensions so they reach visualizations.
    if ratio > IDENTIFIER_CARDINALITY_RATIO and n_rows > 20:
        if unique_count > MAX_DIMENSION_CARDINALITY:
            return "identifier"
        if _column_name_suggests_identifier(col):
            return "identifier"
    if unique_count <= MAX_DIMENSION_CARDINALITY:
        return "dimension"
    return "identifier"  # too high-cardinality text to be a useful grouping dimension


def standardize(df: pd.DataFrame, mapping: ColumnMapping, confirmed_roles=None) -> StandardizedDataset:
    """Builds a StandardizedDataset conditionally. `confirmed_roles` is a set
    of role names (e.g. {"date", "channels"}) the caller explicitly confirmed,
    which lets MEDIUM/LOW confidence guesses for those roles survive
    clear_unconfirmed_low_confidence()."""
    confirmed_roles = confirmed_roles or set()
    mapping.clear_unconfirmed_low_confidence(confirmed_roles=confirmed_roles)

    used_columns = set()
    date_col = mapping.date.column if mapping.date.is_mapped else None
    revenue_col = mapping.revenue.column if mapping.revenue.is_mapped else None
    channels = [g.column for g in mapping.channels if g.is_mapped]

    for c in (date_col, revenue_col, *channels):
        if c:
            used_columns.add(c)

    # other confirmed single-role fields become control_cols candidates too,
    # since StandardizedDataset only has first-class slots for date/revenue.
    control_cols, dimension_cols, identifier_cols = [], [], []

    other_role_cols = []
    for role in ("sales", "cost", "impressions", "clicks", "conversions", "orders", "profit", "margin"):
        guess = getattr(mapping, role)
        if guess.is_mapped and guess.column not in used_columns:
            other_role_cols.append(guess.column)
            used_columns.add(guess.column)

    for col in other_role_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            control_cols.append(col)
        else:
            dimension_cols.append(col)

    # everything else: unmapped_columns + freed_columns (cleared low-confidence)
    leftover = list(dict.fromkeys(list(mapping.unmapped_columns) + list(mapping.freed_columns)))
    leftover = [c for c in leftover if c not in used_columns and c in df.columns]

    for col in leftover:
        kind = _classify_leftover_column(df, col)
        used_columns.add(col)
        if kind == "control":
            control_cols.append(col)
        elif kind == "dimension":
            dimension_cols.append(col)
        else:
            identifier_cols.append(col)

    date_parse = None
    if date_col and date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        # Resolve day/month order once for the whole column so every downstream
        # parse sees unambiguous dates (see src/data_engine/date_parsing.py).
        from src.data_engine.date_parsing import parse_date_column
        parsed, info = parse_date_column(df[date_col])
        df = df.copy()
        df[date_col] = parsed
        date_parse = info.to_dict()

    ds = StandardizedDataset(
        df=df,
        date_col=date_col,
        revenue_col=revenue_col,
        channels=channels,
        control_cols=control_cols,
        dimension_cols=dimension_cols,
        identifier_cols=identifier_cols,
    )
    ds.date_parse = date_parse
    return ds
