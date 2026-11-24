"""Phase 3 — Automatic Insight Engine.

Detects patterns (trends, growth, decline, concentration, distribution,
correlation, outliers, comparisons, data quality) from any dataset using
the Phase 2 CanonicalDataset / AnalysisContext contracts.

Every insight has evidence. Never generates unsupported statements.

Usage:
    from src.analytics.insight_engine import discover_insights
    insights = discover_insights(context)   # returns list[Insight]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.contracts import Insight, make_id
from src.data_engine.canonical import AnalysisContext, CanonicalDataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _growth_rate(series: pd.Series) -> float | None:
    n = len(series)
    if n < 4:
        return None
    half = n // 2
    first = series.iloc[:half].mean()
    second = series.iloc[half:].mean()
    if first == 0:
        return None
    return float((second - first) / abs(first))


# ---------------------------------------------------------------------------
# Insight detectors
# ---------------------------------------------------------------------------

def _detect_trends(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Detect trends in time-series data."""
    insights = []
    time_info = canonical.time_info
    time_col_name = None
    if time_info:
        time_col_name = time_info.time_column if hasattr(time_info, "time_column") else (time_info.get("time_column") if isinstance(time_info, dict) else None)
    if not time_col_name:
        return insights

    time_col = time_col_name
    if time_col not in df.columns:
        return insights

    targets = canonical.target_candidates or []
    measures = canonical.measure_candidates or []
    cols_to_check = []
    for tc in targets[:2]:
        col = tc.get("column") if isinstance(tc, dict) else tc.column
        cols_to_check.append(col)
    for m in measures[:3]:
        if m not in cols_to_check:
            cols_to_check.append(m)

    for col in cols_to_check[:4]:
        if col not in df.columns:
            continue
        try:
            sorted_df = df[[time_col, col]].dropna().sort_values(time_col)
            series = _safe_numeric(sorted_df[col])
            if len(series) < 4:
                continue
            growth = _growth_rate(series)
            if growth is None:
                continue

            abs_pct = abs(growth * 100)
            if abs_pct < 2.0:
                continue

            if growth > 0:
                insights.append(Insight(
                    id=make_id("ins_"),
                    title=f"{col} Increasing",
                    description=f"{col} increased by {abs_pct:.1f}% comparing the first half "
                                f"to the second half of the observed period.",
                    insight_type="growth",
                    severity="medium" if abs_pct > 10 else "low",
                    evidence=f"First-half average: {series.iloc[:len(series)//2].mean():,.2f}, "
                             f"second-half average: {series.iloc[len(series)//2:].mean():,.2f}.",
                    source_columns=[col, time_col],
                    confidence=min(0.9, 0.5 + abs_pct / 100),
                ))
            else:
                insights.append(Insight(
                    id=make_id("ins_"),
                    title=f"{col} Declining",
                    description=f"{col} decreased by {abs_pct:.1f}% comparing the first half "
                                f"to the second half of the observed period.",
                    insight_type="decline",
                    severity="high" if abs_pct > 20 else "medium",
                    evidence=f"First-half average: {series.iloc[:len(series)//2].mean():,.2f}, "
                             f"second-half average: {series.iloc[len(series)//2:].mean():,.2f}.",
                    source_columns=[col, time_col],
                    confidence=min(0.9, 0.5 + abs_pct / 100),
                ))
        except Exception:
            pass

    return insights


def _detect_concentration(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Detect category concentration (top N account for X%)."""
    insights = []
    dims = canonical.dimension_candidates or []
    measures = canonical.measure_candidates or []

    for dim in dims[:3]:
        if dim not in df.columns:
            continue

        # Concentration by count
        counts = df[dim].value_counts(dropna=True)
        if len(counts) < 3:
            continue
        top3_pct = float(counts.iloc[:3].sum() / counts.sum() * 100)
        if top3_pct > 50:
            top3_names = ", ".join(str(v) for v in counts.index[:3])
            insights.append(Insight(
                id=make_id("ins_"),
                title=f"Concentration in {dim}",
                description=f"Top 3 categories in '{dim}' account for {top3_pct:.0f}% of all records.",
                insight_type="concentration",
                severity="medium" if top3_pct > 70 else "low",
                evidence=f"Leading categories: {top3_names}.",
                source_columns=[dim],
                confidence=0.8,
                metric_value=round(top3_pct, 1),
            ))

        # Concentration by numeric measure
        for measure in measures[:2]:
            if measure not in df.columns:
                continue
            try:
                grouped = df.groupby(dim)[measure].sum().sort_values(ascending=False).dropna()
                if len(grouped) < 3:
                    continue
                total = grouped.sum()
                if total == 0:
                    continue
                top3_val_pct = float(grouped.iloc[:3].sum() / total * 100)
                if top3_val_pct > 50:
                    leader = grouped.index[0]
                    leader_pct = float(grouped.iloc[0] / total * 100)
                    insights.append(Insight(
                        id=make_id("ins_"),
                        title=f"{leader} Leads in {measure}",
                        description=f"'{leader}' contributes {leader_pct:.1f}% of total {measure}. "
                                    f"Top 3 categories account for {top3_val_pct:.0f}%.",
                        insight_type="comparison",
                        severity="medium",
                        evidence=f"Total {measure}: {total:,.2f}; "
                                 f"'{leader}' contributes {grouped.iloc[0]:,.2f}.",
                        source_columns=[dim, measure],
                        confidence=0.75,
                        metric_value=round(leader_pct, 1),
                    ))
            except Exception:
                pass

    return insights


def _detect_distributions(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Detect distribution characteristics of numeric columns."""
    insights = []
    measures = canonical.measure_candidates or []

    for col in measures[:4]:
        if col not in df.columns:
            continue
        series = _safe_numeric(df[col])
        if len(series) < 10:
            continue

        mean = float(series.mean())
        median = float(series.median())
        std = float(series.std())
        if mean == 0:
            continue

        skew_ratio = (mean - median) / std if std > 0 else 0

        if abs(skew_ratio) > 0.3:
            direction = "right" if skew_ratio > 0 else "left"
            insights.append(Insight(
                id=make_id("ins_"),
                title=f"{col} is {direction}-skewed",
                description=f"The distribution of '{col}' is {direction}-skewed "
                            f"(mean={mean:,.2f}, median={median:,.2f}).",
                insight_type="distribution",
                severity="info",
                evidence=f"Mean-median gap: {mean - median:,.2f}; std: {std:,.2f}.",
                source_columns=[col],
                confidence=0.6,
            ))

    return insights


def _detect_outliers(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Detect outlier presence using IQR method."""
    insights = []
    measures = canonical.measure_candidates or []

    for col in measures[:5]:
        if col not in df.columns:
            continue
        series = _safe_numeric(df[col])
        if len(series) < 10:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        pct = float(len(outliers) / len(series) * 100)

        if pct > 1.0:
            insights.append(Insight(
                id=make_id("ins_"),
                title=f"Outliers in {col}",
                description=f"{len(outliers)} observation(s) ({pct:.1f}%) in '{col}' "
                            f"are outside the interquartile range [{lower:,.2f}, {upper:,.2f}].",
                insight_type="outlier",
                severity="medium" if pct > 5 else "low",
                evidence=f"IQR={iqr:,.2f}; outlier boundaries: [{lower:,.2f}, {upper:,.2f}].",
                source_columns=[col],
                confidence=0.7,
                metric_value=round(pct, 1),
            ))

    return insights


def _detect_correlations(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Detect strong correlations between numeric columns."""
    insights = []
    numeric_cols = canonical.numeric_columns or []
    id_cols = set(canonical.identifier_columns or [])
    useful = [c for c in numeric_cols if c not in id_cols and c in df.columns]

    if len(useful) < 2:
        return insights

    pairs_seen = set()
    for i, col_a in enumerate(useful[:8]):
        for col_b in useful[i + 1:8]:
            pair = tuple(sorted([col_a, col_b]))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)

            try:
                a = _safe_numeric(df[col_a])
                b = _safe_numeric(df[col_b])
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    continue
                corr = float(a.loc[common].corr(b.loc[common]))
                if np.isnan(corr) or abs(corr) < 0.5:
                    continue

                strength = "strong" if abs(corr) >= 0.7 else "moderate"
                direction = "positive" if corr > 0 else "negative"
                insights.append(Insight(
                    id=make_id("ins_"),
                    title=f"{col_a} ↔ {col_b} Correlation",
                    description=f"{col_a} and {col_b} show a {strength} {direction} "
                                f"correlation (r = {corr:.2f}).",
                    insight_type="correlation",
                    severity="medium" if abs(corr) >= 0.7 else "low",
                    evidence=f"Pearson r = {corr:.3f} over {len(common)} observations.",
                    source_columns=[col_a, col_b],
                    confidence=min(0.95, 0.5 + abs(corr) * 0.5),
                    metric_value=round(corr, 3),
                ))
            except Exception:
                pass

    return insights


def _detect_data_quality_issues(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Detect data quality issues worth mentioning."""
    insights = []
    dq = canonical.data_quality
    if not dq:
        return insights

    # Access both dict and object forms
    def _get(key, default=None):
        if isinstance(dq, dict):
            return dq.get(key, default)
        return getattr(dq, key, default)

    missing_pct = _get("missing_cells_pct", 0.0)
    if missing_pct > 5:
        insights.append(Insight(
            id=make_id("ins_"),
            title="Significant Missing Data",
            description=f"{missing_pct:.1f}% of all data cells are missing.",
            insight_type="data_quality",
            severity="high" if missing_pct > 20 else "medium",
            evidence=f"Missing cells: {_get('missing_cells', 0):,} out of "
                     f"{_get('row_count', 0) * _get('column_count', 0):,} total.",
            source_columns=[],
            confidence=1.0,
            metric_value=round(missing_pct, 1),
        ))

    dup_rows = _get("duplicate_rows", 0)
    row_count = _get("row_count", 0)
    if dup_rows and row_count:
        dup_pct = dup_rows / row_count * 100
        if dup_pct > 1:
            insights.append(Insight(
                id=make_id("ins_"),
                title="Duplicate Rows Detected",
                description=f"{dup_rows:,} duplicate row(s) found ({dup_pct:.1f}% of the dataset).",
                insight_type="data_quality",
                severity="medium",
                evidence=f"{dup_rows} rows are exact duplicates.",
                source_columns=[],
                confidence=1.0,
                metric_value=dup_rows,
            ))

    constant_cols = _get("constant_columns", [])
    if constant_cols:
        insights.append(Insight(
            id=make_id("ins_"),
            title="Constant Column(s) Found",
            description=f"{len(constant_cols)} column(s) contain only one unique value: "
                        f"{', '.join(constant_cols[:5])}.",
            insight_type="data_quality",
            severity="low",
            evidence="These columns carry no variation and may be excluded from analysis.",
            source_columns=list(constant_cols[:5]),
            confidence=1.0,
        ))

    return insights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_insights(context: AnalysisContext, max_insights: int = 15) -> list:
    """Discover all insights from the dataset."""
    canonical = context.dataset
    df = canonical.dataframe
    if df is None or df.empty:
        return []

    insights: list[Insight] = []
    insights.extend(_detect_trends(canonical, df))
    insights.extend(_detect_concentration(canonical, df))
    insights.extend(_detect_distributions(canonical, df))
    insights.extend(_detect_outliers(canonical, df))
    insights.extend(_detect_correlations(canonical, df))
    insights.extend(_detect_data_quality_issues(canonical, df))

    # Sort: severity (high first), then confidence (high first)
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    insights.sort(key=lambda i: (severity_order.get(i.severity, 3), -i.confidence))

    return insights[:max_insights]


def get_top_insights(context: AnalysisContext, max_count: int = 8) -> list:
    """Convenience: discover + return top insights."""
    return discover_insights(context, max_insights=max_count)
