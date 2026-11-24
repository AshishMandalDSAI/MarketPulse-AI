"""Phase 3 — Automatic Visualization Recommendation Engine.

Recommends charts from the CanonicalDataset / AnalysisContext, scored and
explained.  Never assumes specific column names like Revenue or Date.

Usage:
    from src.analytics.visualization_engine import recommend_charts
    charts = recommend_charts(context)   # returns list[ChartRecommendation]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.contracts import ChartRecommendation, make_id
from src.data_engine.canonical import AnalysisContext, CanonicalDataset

MAX_PIE_CATEGORIES = 8
MAX_TOP_N = 15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _column_cardinality(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


def _is_high_cardinality(df: pd.DataFrame, col: str, threshold: int = 20) -> bool:
    return _column_cardinality(df, col) > threshold


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


# ---------------------------------------------------------------------------
# Chart recommendation rules
# ---------------------------------------------------------------------------

def recommend_charts(context: AnalysisContext, max_charts: int = 12) -> list:
    """Recommend charts for the dataset, scored and explained."""
    canonical = context.dataset
    df = canonical.dataframe
    if df is None or df.empty:
        return []

    charts: list[ChartRecommendation] = []

    time_col = None
    time_info = canonical.time_info
    if time_info:
        time_col = time_info.time_column if hasattr(time_info, "time_column") else (time_info.get("time_column") if isinstance(time_info, dict) else None)

    numeric_cols = canonical.numeric_columns or []
    cat_cols = canonical.categorical_columns or []
    dims = canonical.dimension_candidates or []
    measures = canonical.measure_candidates or []
    targets = canonical.target_candidates or []

    # Remove identifier columns from useful numerics
    id_cols = set(canonical.identifier_columns or [])
    useful_numeric = [c for c in numeric_cols if c not in id_cols]
    useful_measures = [c for c in measures if c not in id_cols]

    # Get top target column if exists
    target_col = None
    if targets:
        tc = targets[0]
        target_col = tc.get("column") if isinstance(tc, dict) else tc.column

    # ---- Rule 1: Time + numeric target → line chart ----
    if time_col and target_col and target_col in df.columns:
        charts.append(ChartRecommendation(
            id=make_id("chart_"),
            chart_type="line",
            title=f"{target_col} Over Time",
            x_column=time_col,
            y_column=target_col,
            score=0.95,
            reason=f"Time column '{time_col}' + primary target '{target_col}' detected — "
                   f"line chart is the natural choice for time-series data.",
            category="trend",
        ))

    # ---- Rule 2: Time + other measures → area/line charts ----
    if time_col:
        other_measures = [c for c in useful_measures if c != target_col][:3]
        for col in other_measures:
            if col in df.columns:
                charts.append(ChartRecommendation(
                    id=make_id("chart_"),
                    chart_type="area",
                    title=f"{col} Trend",
                    x_column=time_col,
                    y_column=col,
                    score=0.80,
                    reason=f"Time column + numeric measure '{col}' — area chart shows trend.",
                    category="trend",
                ))

    # ---- Rule 3: Category + numeric → bar chart ----
    for dim in dims[:3]:
        if dim not in df.columns:
            continue
        for measure in useful_measures[:2]:
            if measure not in df.columns:
                continue
            cardinality = _column_cardinality(df, dim)
            if cardinality < 1:
                continue

            if cardinality > MAX_TOP_N:
                charts.append(ChartRecommendation(
                    id=make_id("chart_"),
                    chart_type="horizontal_bar",
                    title=f"Top {min(10, cardinality)} {dim} by {measure}",
                    x_column=measure,
                    y_column=dim,
                    aggregation="sum",
                    score=0.75,
                    reason=f"High-cardinality dimension '{dim}' ({cardinality} values) "
                           f"— showing top-N with horizontal bar for readability.",
                    category="comparison",
                    top_n=min(10, cardinality),
                ))
            else:
                charts.append(ChartRecommendation(
                    id=make_id("chart_"),
                    chart_type="bar",
                    title=f"{measure} by {dim}",
                    x_column=dim,
                    y_column=measure,
                    aggregation="mean",
                    score=0.82,
                    reason=f"Categorical dimension '{dim}' ({cardinality} values) + "
                           f"numeric measure '{measure}' → bar chart comparison.",
                    category="comparison",
                ))

    # ---- Rule 4: Two numeric columns → scatter ----
    if len(useful_numeric) >= 2:
        pairs_done = set()
        for i, col_a in enumerate(useful_numeric[:5]):
            for col_b in useful_numeric[i + 1:6]:
                pair = tuple(sorted([col_a, col_b]))
                if pair in pairs_done:
                    continue
                pairs_done.add(pair)
                if col_a not in df.columns or col_b not in df.columns:
                    continue
                # Check if correlation is interesting
                try:
                    a = _safe_numeric(df[col_a])
                    b = _safe_numeric(df[col_b])
                    common = a.index.intersection(b.index)
                    if len(common) < 5:
                        continue
                    corr = float(a.loc[common].corr(b.loc[common]))
                    if abs(corr) < 0.2:
                        continue
                except Exception:
                    continue

                charts.append(ChartRecommendation(
                    id=make_id("chart_"),
                    chart_type="scatter",
                    title=f"{col_a} vs {col_b}",
                    x_column=col_a,
                    y_column=col_b,
                    score=0.60 + abs(corr) * 0.3,
                    reason=f"Two numeric columns with correlation {corr:.2f} — "
                           f"scatter plot reveals relationship.",
                    category="relationship",
                ))

    # ---- Rule 5: Single numeric → histogram ----
    for col in useful_numeric[:3]:
        if col not in df.columns:
            continue
        series = _safe_numeric(df[col])
        if len(series) < 5:
            continue
        charts.append(ChartRecommendation(
            id=make_id("chart_"),
            chart_type="histogram",
            title=f"Distribution of {col}",
            x_column=col,
            score=0.55,
            reason=f"Numeric column '{col}' — histogram shows value distribution.",
            category="distribution",
        ))

    # ---- Rule 6: Category distribution → pie/donut or bar ----
    for dim in dims[:3]:
        if dim not in df.columns:
            continue
        cardinality = _column_cardinality(df, dim)
        if cardinality < 2:
            continue
        if cardinality <= MAX_PIE_CATEGORIES:
            charts.append(ChartRecommendation(
                id=make_id("chart_"),
                chart_type="donut",
                title=f"{dim} Distribution",
                x_column=dim,
                aggregation="count",
                score=0.65,
                reason=f"Categorical column '{dim}' with {cardinality} values — "
                       f"donut chart shows proportions.",
                category="distribution",
            ))
        else:
            charts.append(ChartRecommendation(
                id=make_id("chart_"),
                chart_type="bar",
                title=f"{dim} Frequency",
                x_column=dim,
                aggregation="count",
                score=0.55,
                reason=f"Categorical column '{dim}' with {cardinality} values — "
                       f"bar chart for frequency counts.",
                category="distribution",
                top_n=min(15, cardinality),
            ))

    # ---- Rule 7: Multiple numeric columns → correlation heatmap ----
    if len(useful_numeric) >= 3:
        charts.append(ChartRecommendation(
            id=make_id("chart_"),
            chart_type="heatmap",
            title="Correlation Matrix",
            y_columns=useful_numeric[:10],
            score=0.70,
            reason=f"Multiple numeric columns ({len(useful_numeric)}) — "
                   f"correlation heatmap reveals relationships at a glance.",
            category="relationship",
        ))

    # ---- Rule 8: Box plot for numeric by category ----
    if dims and useful_numeric:
        dim = dims[0]
        num_col = useful_numeric[0]
        if dim in df.columns and num_col in df.columns:
            cardinality = _column_cardinality(df, dim)
            if 2 <= cardinality <= 12:
                charts.append(ChartRecommendation(
                    id=make_id("chart_"),
                    chart_type="box",
                    title=f"{num_col} by {dim}",
                    x_column=dim,
                    y_column=num_col,
                    score=0.60,
                    reason=f"Box plot shows distribution of '{num_col}' across "
                           f"'{dim}' categories ({cardinality} groups).",
                    category="comparison",
                ))

    # Sort by score and trim
    charts.sort(key=lambda c: -c.score)
    return charts[:max_charts]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_top_charts(context: AnalysisContext, max_count: int = 6) -> list:
    """Convenience: recommend + return top charts."""
    return recommend_charts(context, max_charts=max_count)
