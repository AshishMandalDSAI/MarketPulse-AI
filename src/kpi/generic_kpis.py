"""Schema-agnostic KPIs computed purely from the columns that actually
exist in the standardized dataset — never fabricated, never hardcoded."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.schemas import StandardizedDataset


@dataclass
class GenericKPI:
    label: str
    value: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "value": self.value, "note": self.note}


def compute_generic_kpis(dataset: StandardizedDataset, max_measures: int = 4, max_dimensions: int = 3) -> list:
    df = dataset.df
    if df is None or df.empty:
        return []

    kpis = []

    kpis.append(GenericKPI("Total Records", f"{len(df):,}", "Row count of the uploaded dataset."))

    if dataset.has_date:
        rng = dataset.date_range()
        if rng:
            start, end = rng
            kpis.append(GenericKPI(
                "Date Range",
                f"{start.date()} to {end.date()}",
                f"Based on '{dataset.date_col}'.",
            ))

    if dataset.identifier_cols:
        id_col = dataset.identifier_cols[0]
        n_unique = int(df[id_col].nunique(dropna=True))
        kpis.append(GenericKPI(f"Unique {id_col}", f"{n_unique:,}", f"Distinct values in '{id_col}'."))

    measures = dataset.measure_cols[:max_measures]
    for col in measures:
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        kpis.append(GenericKPI(
            f"Avg {col}", f"{numeric.mean():,.2f}", f"Average of '{col}' across {len(numeric):,} record(s).",
        ))
        kpis.append(GenericKPI(
            f"Total {col}", f"{numeric.sum():,.2f}", f"Sum of '{col}' across {len(numeric):,} record(s).",
        ))

    for col in dataset.dimension_cols[:max_dimensions]:
        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            continue
        top_value = counts.index[0]
        pct = counts.iloc[0] / counts.sum() * 100
        kpis.append(GenericKPI(
            f"Most Common {col}", f"{top_value} ({pct:.1f}%)",
            f"Most frequent value in '{col}'.",
        ))

    missing_pct = float(df.isna().mean().mean() * 100) if df.shape[1] else 0.0
    kpis.append(GenericKPI("Overall Missing Data", f"{missing_pct:.1f}%", "Average missing-value rate across all columns."))

    return kpis
