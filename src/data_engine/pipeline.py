"""High-level entry points that tie profiling, classification and
standardization together, with a human-readable decision report."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.profiler import profile_dataset
from src.data_engine.column_classifier import classify_columns
from src.data_engine.data_standardizer import standardize
from src.data_engine.schemas import ColumnMapping


def profile_and_map(df: pd.DataFrame):
    """Returns (profile: dict, mapping: ColumnMapping)."""
    profile = profile_dataset(df)
    mapping = classify_columns(df)
    return profile, mapping


def _apply_overrides(mapping: ColumnMapping, overrides: dict, confirmed_roles: set) -> None:
    """Applies explicit user overrides onto a mapping in place.

    overrides keys understood: date_column, revenue_column, channel_columns
    (list). An empty string / None means "none" (explicitly clear that
    role), not "leave default".
    """
    if "date_column" in overrides:
        val = overrides["date_column"]
        if val:
            mapping.date.column, mapping.date.confidence, mapping.date.matched_via = val, 1.0, "user-override"
            confirmed_roles.add("date")
        else:
            mapping.date.column, mapping.date.confidence, mapping.date.matched_via = None, 0.0, "none"

    if "revenue_column" in overrides:
        val = overrides["revenue_column"]
        if val:
            mapping.revenue.column, mapping.revenue.confidence, mapping.revenue.matched_via = val, 1.0, "user-override"
            confirmed_roles.add("revenue")
        else:
            mapping.revenue.column, mapping.revenue.confidence, mapping.revenue.matched_via = None, 0.0, "none"

    if "channel_columns" in overrides:
        vals = overrides["channel_columns"] or []
        from src.data_engine.schemas import FieldGuess
        mapping.channels = [FieldGuess(column=v, confidence=1.0, matched_via="user-override") for v in vals]
        confirmed_roles.add("channels")


def confirm_and_standardize(df: pd.DataFrame, mapping: ColumnMapping, confirmed_roles=None, overrides=None):
    """Runs the full confirm -> standardize step. Returns
    (StandardizedDataset, report: list[str])."""
    confirmed_roles = set(confirmed_roles or set())
    report = []

    if overrides:
        _apply_overrides(mapping, overrides, confirmed_roles)
        report.append(f"Applied explicit user overrides: {list(overrides.keys())}")

    # snapshot before clearing, to describe what happens
    pre_state = mapping.to_dict()

    dataset = standardize(df, mapping, confirmed_roles=confirmed_roles)

    if dataset.date_col:
        report.append(f"Mapped '{dataset.date_col}' -> date "
                       f"(confidence={pre_state['date']['confidence']}, via={pre_state['date']['matched_via']}).")
        dp = getattr(dataset, "date_parse", None)
        if dp:
            msg = f"Date parsing: {dp['parsed']} value(s) read."
            if dp.get("note"):
                msg += " " + dp["note"]
            if dp.get("unparseable"):
                msg += (f" {dp['unparseable']} value(s) could not be read as dates (e.g. "
                        f"{', '.join(repr(x) for x in dp['examples_unparseable'])}); those rows are excluded "
                        "from time-based analyses, never assigned an invented date.")
            report.append(msg)
    else:
        report.append("No date column identified or confirmed — date-dependent features (trends, forecasting) "
                       "will be unavailable.")

    if dataset.revenue_col:
        report.append(f"Mapped '{dataset.revenue_col}' -> revenue "
                       f"(confidence={pre_state['revenue']['confidence']}, via={pre_state['revenue']['matched_via']}).")
    else:
        report.append("No revenue/outcome column identified or confirmed — revenue-dependent features "
                       "(MMM, ROI, budget optimization) will be unavailable.")

    if dataset.channels:
        report.append(f"Mapped {len(dataset.channels)} channel/spend column(s): {dataset.channels}.")
    else:
        report.append("No marketing spend/channel columns identified or confirmed — "
                       "MMM/attribution/ROI features will be unavailable.")

    if mapping.freed_columns:
        report.append(
            f"Cleared {len(mapping.freed_columns)} low/medium-confidence guess(es) that were not explicitly "
            f"confirmed: {mapping.freed_columns}. These were reclassified as generic measures/dimensions "
            f"instead of being force-mapped or dropped."
        )

    if dataset.control_cols:
        report.append(f"{len(dataset.control_cols)} column(s) kept as generic numeric measures: {dataset.control_cols}")
    if dataset.dimension_cols:
        report.append(f"{len(dataset.dimension_cols)} column(s) kept as categorical dimensions: {dataset.dimension_cols}")
    if dataset.identifier_cols:
        report.append(f"{len(dataset.identifier_cols)} column(s) identified as identifiers (near-unique): {dataset.identifier_cols}")

    dataset.mapping_report = report
    return dataset, report
