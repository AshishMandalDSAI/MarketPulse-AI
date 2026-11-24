"""Phase 3 — Relationship / Correlation Analysis Engine.

Detects numeric correlations, group comparisons, and trend relationships
from any dataset using the Phase 2 CanonicalDataset / AnalysisContext.

Never computes meaningless correlations involving IDs.
Handles missing values, constant columns, insufficient observations.

Usage:
    from src.analytics.relationship_engine import analyze_relationships
    rels = analyze_relationships(context)   # returns list[Relationship]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.contracts import Relationship, make_id
from src.data_engine.canonical import AnalysisContext, CanonicalDataset


MIN_OBSERVATIONS = 5
MIN_CORRELATION = 0.3


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


# ---------------------------------------------------------------------------
# Relationship detectors
# ---------------------------------------------------------------------------

def _compute_correlations(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Pearson correlations between numeric columns, excluding IDs."""
    relationships = []
    numeric_cols = canonical.numeric_columns or []
    id_cols = set(canonical.identifier_columns or [])
    constant_cols = set()
    dq = canonical.data_quality
    if dq:
        if isinstance(dq, dict):
            constant_cols = set(dq.get("constant_columns", []))
        else:
            constant_cols = set(getattr(dq, "constant_columns", []))

    useful = [c for c in numeric_cols
              if c not in id_cols and c not in constant_cols and c in df.columns]

    if len(useful) < 2:
        return relationships

    # Build correlation matrix
    numeric_df = df[useful].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_df.corr(method="pearson", min_periods=MIN_OBSERVATIONS)

    pairs_seen = set()
    for i, col_a in enumerate(useful):
        for j, col_b in enumerate(useful):
            if i >= j:
                continue
            pair = tuple(sorted([col_a, col_b]))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)

            try:
                corr = float(corr_matrix.loc[col_a, col_b])
            except (KeyError, TypeError):
                continue
            if np.isnan(corr) or abs(corr) < MIN_CORRELATION:
                continue

            strength = "strong" if abs(corr) >= 0.7 else "moderate"
            direction = "positive" if corr > 0 else "negative"

            relationships.append(Relationship(
                id=make_id("rel_"),
                relationship_type="correlation",
                columns=[col_a, col_b],
                strength=abs(corr),
                direction=direction,
                description=f"{strength.capitalize()} {direction} correlation between "
                            f"'{col_a}' and '{col_b}' (r = {corr:.3f}).",
                evidence=f"Pearson r = {corr:.3f}.",
            ))

    # Sort by strength
    relationships.sort(key=lambda r: -r.strength)
    return relationships


def _compute_group_comparisons(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Compare numeric measures across categorical dimensions."""
    relationships = []
    dims = canonical.dimension_candidates or []
    measures = canonical.measure_candidates or []
    id_cols = set(canonical.identifier_columns or [])

    for dim in dims[:3]:
        if dim not in df.columns:
            continue
        cardinality = int(df[dim].nunique(dropna=True))
        if cardinality < 2 or cardinality > 20:
            continue

        for measure in measures[:3]:
            if measure not in df.columns or measure in id_cols:
                continue

            try:
                grouped = df.groupby(dim)[measure].agg(["mean", "count"]).dropna()
                grouped = grouped[grouped["count"] >= 3]
                if len(grouped) < 2:
                    continue

                means = grouped["mean"]
                overall_mean = float(df[measure].mean())
                if overall_mean == 0:
                    continue

                max_group = means.idxmax()
                min_group = means.idxmin()
                spread = float((means.max() - means.min()) / abs(overall_mean)) if overall_mean else 0

                if spread < 0.05:
                    continue

                relationships.append(Relationship(
                    id=make_id("rel_"),
                    relationship_type="group_comparison",
                    columns=[dim, measure],
                    strength=min(1.0, spread),
                    direction="varies",
                    description=f"Average {measure} varies across {dim}: "
                                f"highest in '{max_group}' ({means.max():,.2f}), "
                                f"lowest in '{min_group}' ({means.min():,.2f}).",
                    evidence=f"Spread: {spread * 100:.1f}% of overall mean ({overall_mean:,.2f}). "
                             f"Groups compared: {len(grouped)}.",
                ))
            except Exception:
                pass

    return relationships


def _compute_trend_relationships(canonical: CanonicalDataset, df: pd.DataFrame) -> list:
    """Detect trend relationships between time and numeric columns."""
    relationships = []
    time_info = canonical.time_info
    time_col_name = None
    if time_info:
        time_col_name = time_info.time_column if hasattr(time_info, "time_column") else (time_info.get("time_column") if isinstance(time_info, dict) else None)
    if not time_col_name:
        return relationships

    time_col = time_col_name
    if time_col not in df.columns:
        return relationships

    measures = canonical.measure_candidates or []
    id_cols = set(canonical.identifier_columns or [])

    for col in measures[:5]:
        if col not in df.columns or col in id_cols:
            continue

        try:
            sorted_df = df[[time_col, col]].dropna().sort_values(time_col)
            series = _safe_numeric(sorted_df[col])
            if len(series) < MIN_OBSERVATIONS:
                continue

            # Simple linear trend
            x = np.arange(len(series), dtype=float)
            y = series.values.astype(float)
            if np.std(y) == 0:
                continue

            corr = float(np.corrcoef(x, y)[0, 1])
            if np.isnan(corr) or abs(corr) < MIN_CORRELATION:
                continue

            direction = "increasing" if corr > 0 else "decreasing"
            strength = "strong" if abs(corr) >= 0.7 else "moderate"

            relationships.append(Relationship(
                id=make_id("rel_"),
                relationship_type="trend",
                columns=[time_col, col],
                strength=abs(corr),
                direction=direction,
                description=f"'{col}' shows a {strength} {direction} trend over time "
                            f"(r = {corr:.3f}).",
                evidence=f"Linear correlation with time index: r = {corr:.3f} "
                         f"over {len(series)} observations.",
            ))
        except Exception:
            pass

    return relationships


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_relationships(context: AnalysisContext, max_relationships: int = 15) -> list:
    """Discover all relationships in the dataset."""
    canonical = context.dataset
    df = canonical.dataframe
    if df is None or df.empty:
        return []

    rels: list[Relationship] = []
    rels.extend(_compute_correlations(canonical, df))
    rels.extend(_compute_group_comparisons(canonical, df))
    rels.extend(_compute_trend_relationships(canonical, df))

    # Sort by strength
    rels.sort(key=lambda r: -r.strength)
    return rels[:max_relationships]


def get_correlation_matrix(context: AnalysisContext) -> pd.DataFrame | None:
    """Return the numeric correlation matrix, excluding ID/constant columns."""
    canonical = context.dataset
    df = canonical.dataframe
    if df is None or df.empty:
        return None

    numeric_cols = canonical.numeric_columns or []
    id_cols = set(canonical.identifier_columns or [])
    constant_cols = set()
    dq = canonical.data_quality
    if dq:
        if isinstance(dq, dict):
            constant_cols = set(dq.get("constant_columns", []))
        else:
            constant_cols = set(getattr(dq, "constant_columns", []))

    useful = [c for c in numeric_cols
              if c not in id_cols and c not in constant_cols and c in df.columns]

    if len(useful) < 2:
        return None

    numeric_df = df[useful].apply(pd.to_numeric, errors="coerce")
    return numeric_df.corr(method="pearson", min_periods=MIN_OBSERVATIONS)
