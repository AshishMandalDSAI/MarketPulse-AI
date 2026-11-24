"""Phase 5 — Driver / Relationship Intelligence Engine.

Extends the Phase 3 relationship engine with target-aware driver analysis.
Reuses analyze_relationships() and get_correlation_matrix() from Phase 3;
adds correlation-based ranking, mutual information, model-based feature
importance, and group comparison — all relative to a detected or user-
specified target column.

Language contract: every explanation uses associative, non-causal language.

Usage:
    from src.anomaly.driver_engine import analyze_drivers
    report = analyze_drivers(context, dataset_id="ds_001")
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.contracts import DriverResult, DriverReport, _make_id
from src.analytics.relationship_engine import analyze_relationships, get_correlation_matrix
from src.analytics.contracts import Relationship
from src.data_engine.canonical import AnalysisContext, CanonicalDataset

# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

MAX_FEATURES = 50
MAX_ROWS = 50_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_target(
    context: AnalysisContext,
    target_column: str | None,
) -> tuple[str | None, str]:
    """Return (target_column, target_type).

    target_type is one of "numeric", "categorical", or "unavailable".
    """
    canonical = context.dataset
    df = canonical.dataframe

    if target_column and target_column in df.columns:
        col = target_column
    else:
        candidates = canonical.target_candidates or []
        if not candidates:
            return None, "unavailable"
        col = candidates[0].column  # highest score
        if col not in df.columns:
            return None, "unavailable"

    series = df[col].dropna()
    if series.empty:
        return None, "unavailable"

    if pd.api.types.is_numeric_dtype(series):
        return col, "numeric"

    return col, "categorical"


def _prepare_features(
    canonical: CanonicalDataset,
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Select and preprocess numeric feature columns.

    Returns (prepared_df, feature_columns).
    """
    id_cols = set(canonical.identifier_columns or [])

    constant_cols: set[str] = set()
    dq = canonical.data_quality
    if dq:
        if isinstance(dq, dict):
            constant_cols = set(dq.get("constant_columns", []))
        else:
            constant_cols = set(getattr(dq, "constant_columns", []))

    numeric_cols = [
        c for c in (canonical.numeric_columns or [])
        if c in df.columns
        and c != target_col
        and c not in id_cols
        and c not in constant_cols
    ]

    if not numeric_cols:
        return df, []

    # Convert to numeric, fill NaN with median
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    for c in num_df.columns:
        median_val = num_df[c].median()
        if pd.isna(median_val):
            median_val = 0.0
        num_df[c] = num_df[c].fillna(median_val)

    # Drop columns that are still constant after cleaning
    variances = num_df.var()
    num_df = num_df.loc[:, variances > 0]
    feature_cols = list(num_df.columns)

    # Limit to top MAX_FEATURES by variance
    if len(feature_cols) > MAX_FEATURES:
        top = variances[feature_cols].nlargest(MAX_FEATURES).index.tolist()
        num_df = num_df[top]
        feature_cols = top

    return num_df, feature_cols


def _prepare_target(df: pd.DataFrame, target_col: str, target_type: str) -> pd.Series:
    """Return a clean target Series aligned with df's index."""
    if target_type == "numeric":
        target = pd.to_numeric(df[target_col], errors="coerce")
        median_val = target.median()
        if pd.isna(median_val):
            median_val = 0.0
        return target.fillna(median_val)
    else:
        return df[target_col].fillna("__MISSING__")


# ---------------------------------------------------------------------------
# Individual driver methods
# ---------------------------------------------------------------------------

def _correlation_drivers(
    context: AnalysisContext,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_series: pd.Series,
    target_col: str,
) -> list[DriverResult]:
    """Correlation-based driver detection (Pearson + Spearman)."""
    from scipy.stats import spearmanr

    results: list[DriverResult] = []
    if not feature_cols:
        return results

    # Pearson from the Phase 3 correlation matrix
    corr_matrix = get_correlation_matrix(context)
    pearson_lookup: dict[str, float] = {}
    if corr_matrix is not None and target_col in corr_matrix.columns:
        for feat in feature_cols:
            if feat in corr_matrix.index:
                val = corr_matrix.loc[feat, target_col]
                if not np.isnan(val):
                    pearson_lookup[feat] = float(val)

    # If target wasn't in the Phase 3 matrix, compute pairwise Pearson
    if not pearson_lookup:
        for feat in feature_cols:
            try:
                r = float(feature_df[feat].corr(target_series))
                if not np.isnan(r):
                    pearson_lookup[feat] = r
            except Exception:
                pass

    for feat in feature_cols:
        pearson_r = pearson_lookup.get(feat, np.nan)

        # Spearman
        try:
            valid = feature_df[feat].notna() & target_series.notna()
            if valid.sum() < 5:
                continue
            sp_r, _ = spearmanr(
                feature_df.loc[valid, feat].values,
                target_series.loc[valid].values,
            )
            sp_r = float(sp_r) if not np.isnan(sp_r) else 0.0
        except Exception:
            sp_r = 0.0

        # Use the stronger of the two
        if np.isnan(pearson_r):
            best_r = sp_r
            method_used = "spearman"
        elif abs(pearson_r) >= abs(sp_r):
            best_r = pearson_r
            method_used = "pearson"
        else:
            best_r = sp_r
            method_used = "spearman"

        if abs(best_r) < 0.1:
            continue

        direction = "positive" if best_r > 0 else "negative"

        # Build explanation parts
        parts = [
            f"'{feat}' is strongly associated with '{target_col}'"
            if abs(best_r) >= 0.5
            else f"'{feat}' is associated with '{target_col}'"
        ]
        if not np.isnan(pearson_r):
            parts.append(f"Pearson r = {pearson_r:.3f}")
        if sp_r != 0.0:
            parts.append(f"Spearman rho = {sp_r:.3f}")
        parts.append("This indicates predictive importance, not causation.")
        explanation = " (" + ", ".join(parts[1:-1]) + "). " + parts[-1] if len(parts) > 2 else ". ".join(parts)
        explanation = parts[0] + explanation

        results.append(DriverResult(
            feature=feat,
            target=target_col,
            relationship_type="correlation",
            strength=abs(best_r),
            direction=direction,
            importance=abs(best_r),
            method=method_used,
            confidence_basis=f"|r| = {abs(best_r):.3f}",
            explanation=explanation,
        ))

    return results


def _mutual_information_drivers(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_series: pd.Series,
    target_col: str,
    target_type: str,
) -> list[DriverResult]:
    """Mutual-information-based driver detection."""
    results: list[DriverResult] = []
    if not feature_cols:
        return results

    try:
        if target_type == "numeric":
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(
                feature_df[feature_cols].values,
                target_series.values,
                random_state=42,
            )
        else:
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(
                feature_df[feature_cols].values,
                target_series.values,
                random_state=42,
            )
    except Exception:
        return results

    max_mi = float(np.max(mi_scores)) if len(mi_scores) > 0 else 0.0
    if max_mi <= 0:
        return results

    for feat, mi_raw in zip(feature_cols, mi_scores):
        mi_norm = float(mi_raw) / max_mi
        if mi_norm < 0.05:
            continue

        results.append(DriverResult(
            feature=feat,
            target=target_col,
            relationship_type="mutual_information",
            strength=mi_norm,
            direction="non_linear",
            importance=mi_norm,
            method="mutual_information",
            confidence_basis=f"Normalized MI = {mi_norm:.3f}",
            explanation=(
                f"'{feat}' shares significant mutual information with "
                f"'{target_col}' (normalized MI = {mi_norm:.3f}), "
                f"indicating it is predictive of the target."
            ),
        ))

    return results


def _model_importance_drivers(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_series: pd.Series,
    target_col: str,
    target_type: str,
    unavailable: list[dict],
) -> list[DriverResult]:
    """Gradient-boosting feature importance driver detection."""
    results: list[DriverResult] = []
    if not feature_cols or len(feature_df) < 30:
        return results

    try:
        X = feature_df[feature_cols].values
        y = target_series.values

        if target_type == "numeric":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            # Need at least 2 classes
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                unavailable.append({
                    "analysis": "model_importance",
                    "reason": "Target column has only one class; model-based importance is unavailable.",
                })
                return results
            model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42,
            )

        model.fit(X, y)
        importances = model.feature_importances_

        max_imp = float(np.max(importances)) if len(importances) > 0 else 0.0
        if max_imp <= 0:
            return results

        for feat, imp_raw in zip(feature_cols, importances):
            imp_norm = float(imp_raw) / max_imp
            if imp_norm < 0.05:
                continue

            results.append(DriverResult(
                feature=feat,
                target=target_col,
                relationship_type="feature_importance",
                strength=imp_norm,
                direction="non_linear",
                importance=imp_norm,
                method="gradient_boosting",
                confidence_basis=f"Normalized importance = {imp_norm:.3f}",
                explanation=(
                    f"'{feat}' ranks highly in the predictive model for "
                    f"'{target_col}', suggesting it is an important predictor "
                    f"(normalized importance = {imp_norm:.3f})."
                ),
            ))

    except Exception as exc:
        unavailable.append({
            "analysis": "model_importance",
            "reason": f"Model fitting failed: {exc}",
        })

    return results


def _group_comparison_drivers(
    canonical: CanonicalDataset,
    df: pd.DataFrame,
    target_series: pd.Series,
    target_col: str,
) -> list[DriverResult]:
    """Group comparison: categorical features vs numeric target."""
    results: list[DriverResult] = []

    cat_cols = canonical.categorical_columns or []
    id_cols = set(canonical.identifier_columns or [])

    overall_mean = float(target_series.mean())
    if overall_mean == 0:
        return results

    for dim in cat_cols:
        if dim not in df.columns or dim in id_cols:
            continue

        cardinality = int(df[dim].nunique(dropna=True))
        if cardinality < 2 or cardinality > 20:
            continue

        try:
            grouped = df.groupby(dim)[target_col].agg(["mean", "count"]).dropna()
            grouped = grouped[grouped["count"] >= 3]
            if len(grouped) < 2:
                continue

            means = grouped["mean"]
            spread = float((means.max() - means.min()) / abs(overall_mean))

            if spread < 0.10:
                continue

            max_group = means.idxmax()
            min_group = means.idxmin()

            results.append(DriverResult(
                feature=dim,
                target=target_col,
                relationship_type="group_comparison",
                strength=min(1.0, spread),
                direction="varies",
                importance=min(1.0, spread),
                method="group_comparison",
                confidence_basis=(
                    f"Spread = {spread * 100:.1f}% of overall mean across "
                    f"{len(grouped)} groups"
                ),
                explanation=(
                    f"Average '{target_col}' varies across '{dim}': "
                    f"highest in '{max_group}' ({means.max():,.2f}), "
                    f"lowest in '{min_group}' ({means.min():,.2f}). "
                    f"This group-level variation is associated with "
                    f"differences in the target, not necessarily causal."
                ),
            ))
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Merge and rank
# ---------------------------------------------------------------------------

def _merge_and_rank(all_drivers: list[DriverResult]) -> list[DriverResult]:
    """Merge drivers detected by multiple methods, then rank by importance."""
    if not all_drivers:
        return []

    by_feature: dict[str, list[DriverResult]] = {}
    for d in all_drivers:
        by_feature.setdefault(d.feature, []).append(d)

    merged: list[DriverResult] = []
    for feature, group in by_feature.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Average importance across methods
        avg_importance = float(np.mean([d.importance for d in group]))

        # Keep the details from the method with highest individual importance
        best = max(group, key=lambda d: d.importance)

        methods = sorted({d.method for d in group})
        confidence_parts = [d.confidence_basis for d in group if d.confidence_basis]

        merged.append(DriverResult(
            id=best.id,
            feature=feature,
            target=best.target,
            relationship_type=best.relationship_type,
            strength=avg_importance,
            direction=best.direction,
            importance=avg_importance,
            method=", ".join(methods),
            confidence_basis="; ".join(confidence_parts),
            explanation=best.explanation,
        ))

    # Rank by importance descending
    merged.sort(key=lambda d: -d.importance)
    for rank_idx, driver in enumerate(merged, start=1):
        driver.rank = rank_idx

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_drivers(
    context: AnalysisContext,
    dataset_id: str,
    target_column: str | None = None,
) -> DriverReport:
    """Run target-aware driver analysis on the dataset.

    Extends Phase 3 relationship analysis with correlation ranking,
    mutual information, model-based importance, and group comparisons,
    all oriented toward an auto-detected or user-specified target.
    """
    canonical = context.dataset
    df = canonical.dataframe

    # --- Edge case: empty / missing DataFrame ---
    if df is None or df.empty:
        return DriverReport(
            dataset_id=dataset_id,
            target_type="unavailable",
            unavailable_analyses=[{
                "analysis": "driver_ranking",
                "reason": "Dataset is empty. No driver analysis is possible.",
            }],
        )

    # --- Sample if too large ---
    sampled = False
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
        sampled = True

    # --- Resolve target ---
    target_col, target_type = _resolve_target(context, target_column)

    unavailable: list[dict] = []

    if target_col is None or target_type == "unavailable":
        # Still run Phase 3 relationships even without a target
        phase3_rels = _safe_phase3_relationships(context)
        return DriverReport(
            dataset_id=dataset_id,
            target_column=None,
            target_type="unavailable",
            drivers=[],
            relationships=[
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in phase3_rels
            ],
            unavailable_analyses=[{
                "analysis": "driver_ranking",
                "reason": (
                    "No reliable target was detected. "
                    "Driver ranking is therefore unavailable."
                ),
            }],
        )

    # --- Check for all-NaN target ---
    target_non_null = df[target_col].dropna()
    if target_non_null.empty:
        phase3_rels = _safe_phase3_relationships(context)
        return DriverReport(
            dataset_id=dataset_id,
            target_column=target_col,
            target_type="unavailable",
            relationships=[
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in phase3_rels
            ],
            unavailable_analyses=[{
                "analysis": "driver_ranking",
                "reason": "Target column is entirely missing values.",
            }],
        )

    # --- Single-class categorical target → skip classification methods ---
    if target_type == "categorical" and target_non_null.nunique() < 2:
        phase3_rels = _safe_phase3_relationships(context)
        return DriverReport(
            dataset_id=dataset_id,
            target_column=target_col,
            target_type="unavailable",
            relationships=[
                r.to_dict() if hasattr(r, "to_dict") else r
                for r in phase3_rels
            ],
            unavailable_analyses=[{
                "analysis": "driver_ranking",
                "reason": "Target column has only one class; driver analysis requires variation.",
            }],
        )

    # --- Prepare features and target ---
    feature_df, feature_cols = _prepare_features(canonical, df, target_col)
    target_series = _prepare_target(df, target_col, target_type)

    # Align indices
    common_idx = feature_df.index.intersection(target_series.index)
    feature_df = feature_df.loc[common_idx]
    target_series = target_series.loc[common_idx]

    if feature_cols and len(common_idx) == 0:
        unavailable.append({
            "analysis": "driver_ranking",
            "reason": "No overlapping observations between features and target after cleaning.",
        })
        feature_cols = []

    if not feature_cols:
        unavailable.append({
            "analysis": "driver_ranking",
            "reason": "No suitable numeric features found for driver analysis.",
        })

    # --- Run each driver method ---
    all_drivers: list[DriverResult] = []

    # 1. Correlation (numeric target only)
    if feature_cols and target_type == "numeric":
        all_drivers.extend(
            _correlation_drivers(context, feature_df, feature_cols, target_series, target_col)
        )

    # 2. Mutual information (any target type)
    if feature_cols:
        all_drivers.extend(
            _mutual_information_drivers(feature_df, feature_cols, target_series, target_col, target_type)
        )

    # 3. Model-based importance (>= 30 rows)
    if feature_cols:
        all_drivers.extend(
            _model_importance_drivers(
                feature_df, feature_cols, target_series, target_col, target_type, unavailable,
            )
        )

    # 4. Group comparison (categorical features vs numeric target)
    if target_type == "numeric":
        all_drivers.extend(
            _group_comparison_drivers(canonical, df, target_series, target_col)
        )

    # --- Merge, rank ---
    ranked_drivers = _merge_and_rank(all_drivers)

    # --- Phase 3 relationships ---
    phase3_rels = _safe_phase3_relationships(context)

    return DriverReport(
        dataset_id=dataset_id,
        target_column=target_col,
        target_type=target_type,
        drivers=ranked_drivers,
        relationships=[
            r.to_dict() if hasattr(r, "to_dict") else r
            for r in phase3_rels
        ],
        unavailable_analyses=unavailable,
        metadata={
            "rows_analyzed": len(common_idx) if feature_cols else 0,
            "features_analyzed": len(feature_cols),
            "sampling_used": sampled,
            "methods_applied": _methods_applied(ranked_drivers),
        },
    )


def _safe_phase3_relationships(context: AnalysisContext) -> list:
    """Call Phase 3 analyze_relationships with error handling."""
    try:
        return analyze_relationships(context)
    except Exception:
        return []


def _methods_applied(drivers: list[DriverResult]) -> list[str]:
    """Collect the distinct methods that produced results."""
    methods: set[str] = set()
    for d in drivers:
        for m in d.method.split(", "):
            methods.add(m)
    return sorted(methods)
