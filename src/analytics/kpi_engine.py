"""Phase 3 — Dynamic KPI Discovery Engine.

Discovers, ranks, and explains KPIs from any dataset using the Phase 2
CanonicalDataset / AnalysisContext contracts.  Never hardcodes column
names like Revenue, Date, Google Ads, etc.

Usage:
    from src.analytics.kpi_engine import discover_kpis
    kpis = discover_kpis(context)   # returns list[KPI]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.contracts import KPI, make_id
from src.data_engine.canonical import AnalysisContext, CanonicalDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_date_bound(value):
    """'2021-01-01T00:00:00' -> '2021-01-01'; keeps a time only when it is not midnight."""
    try:
        import pandas as _pd
        ts = _pd.Timestamp(value)
        if _pd.isna(ts):
            return value
        return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return value

def _fmt(value: float, unit: str = "") -> str:
    """Format a number for display."""
    if abs(value) >= 1e9:
        return f"{value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:,.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:,.1f}K"
    if isinstance(value, float) and value != int(value):
        return f"{value:,.2f}"
    return f"{value:,.0f}"


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, dropping NaN."""
    return pd.to_numeric(series, errors="coerce").dropna()


def _growth_rate(series: pd.Series) -> float | None:
    """Compute growth rate (second half vs first half)."""
    n = len(series)
    if n < 4:
        return None
    half = n // 2
    first = series.iloc[:half].sum()
    second = series.iloc[half:].sum()
    if first == 0:
        return None
    return float((second - first) / abs(first))


# ---------------------------------------------------------------------------
# KPI Discovery
# ---------------------------------------------------------------------------

def discover_kpis(context: AnalysisContext, max_primary: int = 8) -> list:
    """Discover and rank KPIs from the dataset.

    Returns a list of KPI objects sorted by priority (lower = more important).
    """
    canonical = context.dataset
    df = canonical.dataframe
    if df is None or df.empty:
        return []

    kpis: list[KPI] = []
    priority_counter = 0

    # ---- 1. Row count (always present) ----
    kpis.append(KPI(
        id=make_id("kpi_"),
        label="Total Records",
        value=canonical.row_count,
        formatted_value=f"{canonical.row_count:,}",
        calculation="count(*)",
        category="summary",
        confidence=1.0,
        explanation="Row count of the uploaded dataset.",
        priority=priority_counter,
    ))
    priority_counter += 1

    # ---- 2. Date range (if time column exists) ----
    time_info = canonical.time_info
    time_col_name = None
    if time_info:
        time_col_name = time_info.time_column if hasattr(time_info, "time_column") else (time_info.get("time_column") if isinstance(time_info, dict) else None)
    if time_col_name:
        tr = time_info.time_range if hasattr(time_info, "time_range") else (time_info.get("time_range") if isinstance(time_info, dict) else None)
        tr = tr or {}
        if isinstance(tr, dict):
            start = tr.get("start", "?")
            end = tr.get("end", "?")
            periods = tr.get("periods", "?")
        else:
            start = getattr(tr, "start", "?")
            end = getattr(tr, "end", "?")
            periods = getattr(tr, "periods", "?")
        start, end = _fmt_date_bound(start), _fmt_date_bound(end)
        granularity = (time_info.time_granularity if hasattr(time_info, "time_granularity") else (time_info.get("time_granularity") if isinstance(time_info, dict) else None)) or "unknown"
        kpis.append(KPI(
            id=make_id("kpi_"),
            label="Date Range",
            value=f"{start} to {end}",
            formatted_value=f"{start} → {end}",
            source_column=time_col_name,
            calculation="min/max of time column",
            category="summary",
            confidence=1.0,
            explanation=f"Covers {periods} {granularity} period(s) based on '{time_col_name}'.",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- 3. Target candidates — generate KPIs for top targets ----
    targets = canonical.target_candidates or []
    for i, tc in enumerate(targets[:3]):
        col = tc.get("column") if isinstance(tc, dict) else tc.column
        score = tc.get("score", 0) if isinstance(tc, dict) else tc.score
        series = _safe_numeric(df[col]) if col in df.columns else pd.Series(dtype=float)
        if series.empty:
            continue

        total = float(series.sum())
        avg = float(series.mean())

        # Total KPI
        kpis.append(KPI(
            id=make_id("kpi_"),
            label=f"Total {col}",
            value=total,
            formatted_value=_fmt(total),
            source_column=col,
            calculation=f"sum({col})",
            category="financial" if score > 0.7 else "performance",
            confidence=min(1.0, score + 0.1),
            explanation=f"'{col}' was detected as a high-confidence target/measure (score {score:.2f}).",
            priority=priority_counter,
        ))
        priority_counter += 1

        # Average KPI
        kpis.append(KPI(
            id=make_id("kpi_"),
            label=f"Average {col}",
            value=round(avg, 2),
            formatted_value=_fmt(avg),
            source_column=col,
            calculation=f"mean({col})",
            category="performance",
            confidence=min(1.0, score),
            explanation=f"Average of '{col}' across {len(series):,} record(s).",
            priority=priority_counter,
        ))
        priority_counter += 1

        # Growth rate if time column exists
        if time_col_name:
            time_col = time_col_name
            if time_col in df.columns:
                try:
                    sorted_df = df[[time_col, col]].dropna().sort_values(time_col)
                    sorted_series = _safe_numeric(sorted_df[col])
                    growth = _growth_rate(sorted_series)
                    if growth is not None:
                        direction = "grown" if growth >= 0 else "declined"
                        kpis.append(KPI(
                            id=make_id("kpi_"),
                            label=f"{col} Growth",
                            value=round(growth * 100, 1),
                            unit="%",
                            formatted_value=f"{growth * 100:+.1f}%",
                            source_column=col,
                            calculation="(second_half_sum - first_half_sum) / first_half_sum",
                            category="growth",
                            confidence=min(1.0, score * 0.9),
                            explanation=f"'{col}' has {direction} by {abs(growth)*100:.1f}% "
                                        f"comparing the first half to the second half of the observed period.",
                            priority=priority_counter,
                            delta=f"{growth*100:+.1f}% H1→H2",
                            delta_positive=growth >= 0,
                        ))
                        priority_counter += 1
                except Exception:
                    pass

    # ---- 4. Measure columns (non-target) — top stats ----
    measures = canonical.measure_candidates or []
    target_cols = {(tc.get("column") if isinstance(tc, dict) else tc.column) for tc in targets[:3]}
    non_target_measures = [m for m in measures if m not in target_cols]

    for col in non_target_measures[:4]:
        series = _safe_numeric(df[col]) if col in df.columns else pd.Series(dtype=float)
        if series.empty:
            continue
        avg = float(series.mean())
        kpis.append(KPI(
            id=make_id("kpi_"),
            label=f"Avg {col}",
            value=round(avg, 2),
            formatted_value=_fmt(avg),
            source_column=col,
            calculation=f"mean({col})",
            category="performance",
            confidence=0.6,
            explanation=f"Average of '{col}' across {len(series):,} record(s).",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- 5. Identifier uniqueness ----
    id_cols = canonical.identifier_columns or []
    for col in id_cols[:2]:
        if col in df.columns:
            n_unique = int(df[col].nunique(dropna=True))
            kpis.append(KPI(
                id=make_id("kpi_"),
                label=f"Unique {col}",
                value=n_unique,
                formatted_value=f"{n_unique:,}",
                source_column=col,
                calculation=f"count(distinct {col})",
                category="summary",
                confidence=0.9,
                explanation=f"Distinct values in identifier column '{col}'.",
                priority=priority_counter,
            ))
            priority_counter += 1

    # ---- 6. Dimension distributions (top category) ----
    dims = canonical.dimension_candidates or []
    for col in dims[:3]:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            continue
        top_val = counts.index[0]
        pct = float(counts.iloc[0] / counts.sum() * 100)
        kpis.append(KPI(
            id=make_id("kpi_"),
            label=f"Top {col}",
            value=str(top_val),
            formatted_value=f"{top_val} ({pct:.1f}%)",
            source_column=col,
            calculation=f"mode({col})",
            category="distribution",
            confidence=0.5,
            explanation=f"Most frequent value in '{col}'.",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- 7. Data quality KPI ----
    dq = canonical.data_quality
    if dq:
        score = dq.get("score", 1.0) if isinstance(dq, dict) else getattr(dq, "score", 1.0)
        missing_pct = dq.get("missing_cells_pct", 0.0) if isinstance(dq, dict) else getattr(dq, "missing_cells_pct", 0.0)
        kpis.append(KPI(
            id=make_id("kpi_"),
            label="Data Quality",
            value=round(score * 100, 1),
            unit="%",
            formatted_value=f"{score * 100:.1f}%",
            calculation="quality_score from data_quality contract",
            category="quality",
            confidence=1.0,
            explanation=f"Overall data quality score. Missing data rate: {missing_pct:.1f}%.",
            priority=priority_counter,
        ))
        priority_counter += 1

    # ---- 8. Derived KPIs where evidence exists ----
    _add_derived_kpis(kpis, canonical, df, priority_counter)

    return kpis


def _add_derived_kpis(kpis: list, canonical: CanonicalDataset, df: pd.DataFrame, priority: int) -> None:
    """Add derived business KPIs only when evidence supports them.

    Examples:
    - Ratio of two semantically related measures (e.g. revenue/orders = avg order value)
    - Rate from a boolean/binary column
    - Utilization/defect from operational measures
    """
    col_meta = canonical.column_metadata or {}
    numeric_cols = canonical.numeric_columns or []
    boolean_cols = canonical.boolean_columns or []

    # Rate from boolean columns (e.g. defect, churn, pass/fail)
    for col in boolean_cols[:3]:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        try:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                # try mapping True/False
                mapped = series.map({True: 1, False: 0, "True": 1, "False": 0,
                                     "Yes": 1, "No": 0, "yes": 1, "no": 0,
                                     1: 1, 0: 0})
                numeric = mapped.dropna()
            if not numeric.empty:
                rate = float(numeric.mean() * 100)
                kpis.append(KPI(
                    id=make_id("kpi_"),
                    label=f"{col} Rate",
                    value=round(rate, 1),
                    unit="%",
                    formatted_value=f"{rate:.1f}%",
                    source_column=col,
                    calculation=f"mean({col}) * 100",
                    category="performance",
                    confidence=0.6,
                    explanation=f"Rate derived from boolean/binary column '{col}'.",
                    priority=priority,
                ))
                priority += 1
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_primary_kpis(context: AnalysisContext, max_count: int = 8) -> list:
    """Convenience: discover + return only top-priority KPIs."""
    all_kpis = discover_kpis(context, max_primary=max_count)
    return sorted(all_kpis, key=lambda k: k.priority)[:max_count]


def get_all_kpis(context: AnalysisContext) -> list:
    """Return all discovered KPIs, sorted by priority."""
    return sorted(discover_kpis(context), key=lambda k: k.priority)
