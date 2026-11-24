"""Phase 4 -- Segmentation / Clustering Engine.

Runs KMeans clustering on any dataset exposed through an AnalysisContext.
Automatically selects the optimal number of clusters via silhouette analysis,
profiles each cluster against overall statistics, and returns a fully
populated SegmentationResult.

No hardcoded column names.  All feature selection is delegated to
``get_feature_columns`` and preprocessing to ``preprocess_for_segmentation``.

Usage::

    from src.prediction.segmentation_engine import run_segmentation
    result = run_segmentation(context)
    print(result.to_dict())
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.canonical import AnalysisContext
from src.prediction.contracts import ClusterProfile, SegmentationResult
from src.prediction.preprocessing import preprocess_for_segmentation
from src.prediction.task_detector import (
    MAX_ROWS,
    MIN_FEATURES_SEGMENTATION,
    MIN_ROWS_SEGMENTATION,
    get_feature_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster description generator
# ---------------------------------------------------------------------------

def _generate_cluster_description(
    cluster_means: dict[str, float],
    overall_means: dict[str, float],
    overall_stds: dict[str, float],
    cluster_size: int,
    total_size: int,
) -> str:
    """Build a human-readable description for one cluster.

    Compares per-feature cluster means against the dataset mean.  Features
    whose cluster mean deviates by more than 0.5 standard deviations are
    flagged as "high" or "low".

    Parameters
    ----------
    cluster_means : dict
        Feature name -> mean value within this cluster (original scale).
    overall_means : dict
        Feature name -> dataset-wide mean (original scale).
    overall_stds : dict
        Feature name -> dataset-wide standard deviation (original scale).
    cluster_size : int
        Number of points in the cluster.
    total_size : int
        Total number of points across all clusters.

    Returns
    -------
    str
        A compact natural-language description.
    """
    high_features: list[str] = []
    low_features: list[str] = []

    for feat in cluster_means:
        c_mean = cluster_means[feat]
        o_mean = overall_means.get(feat, 0.0)
        o_std = overall_stds.get(feat, 0.0)

        if o_std < 1e-9:
            # No meaningful spread -- skip
            continue

        deviation = (c_mean - o_mean) / o_std
        if deviation > 0.5:
            high_features.append(feat)
        elif deviation < -0.5:
            low_features.append(feat)

    parts: list[str] = []

    # Size commentary
    pct = cluster_size / total_size * 100 if total_size > 0 else 0
    if pct >= 40:
        parts.append("Largest cluster")
    elif pct <= 10:
        parts.append("Small cluster")

    if high_features:
        parts.append("high " + ", ".join(high_features[:5]))
    if low_features:
        parts.append("low " + ", ".join(low_features[:5]))

    if not parts:
        parts.append("Near-average across all features")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Optimal K selection
# ---------------------------------------------------------------------------

SILHOUETTE_SAMPLE = 5000


def _silhouette(X, labels) -> float:
    """Exact silhouette score (no sampling). PART 06 performance: distance chunks are computed in
    parallel with a 256 MB working chunk. Same value as the sequential default, measured at 100k
    rows: 95.6 s / 1.2 GB peak -> 75.9 s / 0.7 GB peak on 2 vCPUs."""
    import sklearn
    with sklearn.config_context(working_memory=256):
        return silhouette_score(X, labels, n_jobs=-1)


def _find_optimal_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[int, dict[int, float]]:
    """Select the best number of clusters via silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    k_min : int
        Minimum number of clusters to try.
    k_max : int
        Maximum number of clusters to try.

    Returns
    -------
    best_k : int
        The K with the highest silhouette score.
    scores : dict[int, float]
        Mapping of K -> silhouette score for every K evaluated.
    """
    scores: dict[int, float] = {}
    n_samples = X.shape[0]
    if n_samples > SILHOUETTE_SAMPLE:
        # Choose K on a fixed random sample: 7 values of K x 10 KMeans
        # initialisations on every row took minutes on large uploads. The
        # final model is still fitted on all rows with the chosen K.
        rng = np.random.default_rng(42)
        X = X[rng.choice(n_samples, SILHOUETTE_SAMPLE, replace=False)]

    # Ensure k_max is sensible relative to sample count
    k_max = min(k_max, n_samples // 5) if n_samples >= 10 else 2
    k_max = max(k_max, k_min)

    for k in range(k_min, k_max + 1):
        if k >= n_samples:
            break
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            n_unique = len(set(labels))
            if n_unique < 2:
                # Silhouette undefined for a single cluster
                continue
            # Silhouette is O(n^2); X has at most SILHOUETTE_SAMPLE rows here.
            score = _silhouette(X, labels)
            scores[k] = score
        except Exception as exc:
            logger.debug("Silhouette computation failed for K=%d: %s", k, exc)
            continue

    if not scores:
        # Fallback: couldn't compute any silhouette
        return min(3, k_max), scores

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_k, scores


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_segmentation(
    context: AnalysisContext,
    feature_columns: Optional[list[str]] = None,
    n_clusters: Optional[int] = None,
) -> SegmentationResult:
    """Run KMeans segmentation on the dataset in *context*.

    Parameters
    ----------
    context : AnalysisContext
        Analysis context wrapping a CanonicalDataset.
    feature_columns : list[str] | None
        Explicit feature columns to cluster on.  When ``None``, columns are
        selected automatically via ``get_feature_columns``.
    n_clusters : int | None
        Fixed number of clusters.  When ``None``, the optimal K is chosen
        via silhouette analysis over ``K = 2 .. min(8, n_samples // 5)``.

    Returns
    -------
    SegmentationResult
        Fully populated result including cluster profiles, quality metrics,
        and a human-readable interpretation.
    """
    t0 = time.perf_counter()
    limitations: list[str] = []

    ds = context.dataset
    df = ds.dataframe

    # ------------------------------------------------------------------
    # Guard: no data at all
    # ------------------------------------------------------------------
    if df is None or df.empty:
        return SegmentationResult(
            status="insufficient_data",
            explanation="No data available for segmentation.",
            limitations=["DataFrame is empty or missing."],
        )

    # ------------------------------------------------------------------
    # Guard: insufficient rows
    # ------------------------------------------------------------------
    if len(df) < MIN_ROWS_SEGMENTATION:
        return SegmentationResult(
            status="insufficient_data",
            explanation=(
                f"Only {len(df)} rows available; segmentation requires "
                f"at least {MIN_ROWS_SEGMENTATION}."
            ),
            limitations=[
                f"Row count ({len(df)}) below minimum ({MIN_ROWS_SEGMENTATION})."
            ],
        )

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    if feature_columns is None:
        feature_columns = get_feature_columns(
            context, target_column=None, task_type="segmentation"
        )

    if len(feature_columns) < MIN_FEATURES_SEGMENTATION:
        return SegmentationResult(
            status="unavailable",
            feature_columns=feature_columns,
            explanation=(
                f"Only {len(feature_columns)} numeric non-identifier column(s) "
                f"found; need at least {MIN_FEATURES_SEGMENTATION}."
            ),
            limitations=["Insufficient numeric features for meaningful clustering."],
        )

    # ------------------------------------------------------------------
    # Down-sample large datasets
    # ------------------------------------------------------------------
    work_df = df.copy()
    if len(work_df) > MAX_ROWS:
        work_df = work_df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        limitations.append(
            f"Dataset sampled from {len(df)} to {MAX_ROWS} rows for clustering."
        )

    # ------------------------------------------------------------------
    # Store original (unscaled) values for interpretable profiling
    # ------------------------------------------------------------------
    safe_features = [c for c in feature_columns if c in work_df.columns]
    original_numeric = work_df[safe_features].select_dtypes(include="number")
    overall_means = original_numeric.mean().to_dict()
    overall_stds = original_numeric.std().to_dict()

    # ------------------------------------------------------------------
    # Preprocessing (imputation, scaling)
    # ------------------------------------------------------------------
    X_processed, final_features, prep_report = preprocess_for_segmentation(
        work_df, feature_columns
    )

    if X_processed.empty or len(final_features) == 0:
        return SegmentationResult(
            status="failed",
            feature_columns=feature_columns,
            explanation="No features remained after preprocessing.",
            limitations=["All features were dropped during preprocessing."],
        )

    X = X_processed.values
    n_samples, n_features = X.shape

    # ------------------------------------------------------------------
    # Guard: all data identical (zero variance after scaling)
    # ------------------------------------------------------------------
    if np.allclose(X, X[0]):
        return SegmentationResult(
            status="failed",
            feature_columns=final_features,
            explanation="All data points are identical; clustering is not meaningful.",
            limitations=["Zero variance in all features after preprocessing."],
        )

    # ------------------------------------------------------------------
    # Optimal K selection
    # ------------------------------------------------------------------
    silhouette_scores: dict[int, float] = {}
    if n_clusters is not None:
        best_k = max(2, min(n_clusters, n_samples - 1))
        if best_k != n_clusters:
            limitations.append(
                f"Requested {n_clusters} clusters adjusted to {best_k} "
                f"(must be >= 2 and < n_samples)."
            )
    else:
        best_k, silhouette_scores = _find_optimal_k(X, k_min=2, k_max=8)
        if n_samples > SILHOUETTE_SAMPLE:
            limitations.append(
                f"Number of clusters chosen by silhouette score on a random sample of "
                f"{SILHOUETTE_SAMPLE:,} of {n_samples:,} rows (fixed seed); the final model "
                f"uses all rows.")
        if best_k < 2:
            best_k = 2
            limitations.append(
                "Optimal K selection returned < 2; defaulting to K=2."
            )

    # ------------------------------------------------------------------
    # Final clustering
    # ------------------------------------------------------------------
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    n_unique_labels = len(set(cluster_labels))
    if n_unique_labels < 2:
        # Edge case: KMeans collapsed to a single cluster
        limitations.append(
            "KMeans converged to a single cluster; returning K=2 with a note."
        )
        best_k = 2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------
    quality_metrics: dict[str, float] = {"inertia": float(kmeans.inertia_)}

    try:
        sil = _silhouette(X, cluster_labels)   # exact, on all rows
        quality_metrics["silhouette_score"] = float(sil)
    except Exception:
        sil = 0.0
        quality_metrics["silhouette_score"] = 0.0

    try:
        ch = calinski_harabasz_score(X, cluster_labels)
        quality_metrics["calinski_harabasz_score"] = float(ch)
    except Exception:
        pass

    if silhouette_scores:
        quality_metrics["silhouette_by_k"] = {
            str(k): round(v, 6) for k, v in silhouette_scores.items()
        }

    # ------------------------------------------------------------------
    # Cluster profiling (using ORIGINAL unscaled values)
    # ------------------------------------------------------------------
    # Build a DataFrame of original numeric features aligned with the
    # processed rows (preprocessing may have dropped rows via imputation
    # or resource limits, but preprocess_for_segmentation preserves index).
    profile_df = work_df.loc[X_processed.index, :].copy()
    profile_df["__cluster__"] = cluster_labels

    # Only profile features that exist in the original numeric set
    profile_features = [
        f for f in feature_columns
        if f in profile_df.columns and pd.api.types.is_numeric_dtype(profile_df[f])
    ]

    cluster_profiles: list[ClusterProfile] = []
    total_points = len(cluster_labels)

    for cid in range(best_k):
        mask = profile_df["__cluster__"] == cid
        cluster_data = profile_df.loc[mask, profile_features]
        size = int(mask.sum())
        pct = round(size / total_points * 100, 2) if total_points > 0 else 0.0

        centroid = cluster_data.mean().to_dict()
        # Round centroid values for readability
        centroid = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in centroid.items()
        }

        description = _generate_cluster_description(
            cluster_means=centroid,
            overall_means=overall_means,
            overall_stds=overall_stds,
            cluster_size=size,
            total_size=total_points,
        )

        cluster_profiles.append(
            ClusterProfile(
                cluster_id=cid,
                label=f"Cluster {cid + 1}",
                size=size,
                percentage=pct,
                centroid=centroid,
                description=description,
            )
        )

    # ------------------------------------------------------------------
    # Interpretation summary
    # ------------------------------------------------------------------
    elapsed = round(time.perf_counter() - t0, 2)
    profile_summaries = "; ".join(
        f"{p.label} ({p.size} pts, {p.percentage}%)" for p in cluster_profiles
    )
    interpretation = (
        f"KMeans clustering with K={best_k} on {n_features} feature(s) "
        f"and {n_samples} sample(s). "
        f"Silhouette score: {quality_metrics.get('silhouette_score', 0):.4f}. "
        f"Clusters: {profile_summaries}."
    )

    explanation = (
        f"Segmentation completed in {elapsed}s. "
        f"Features used: {', '.join(final_features[:10])}"
        + (f" (+{len(final_features) - 10} more)" if len(final_features) > 10 else "")
        + ". "
        + f"Preprocessing: {prep_report.get('features_before', 0)} input features "
        f"-> {prep_report.get('features_after', 0)} after cleaning."
    )

    # Confidence is the silhouette score clamped to [0, 1]
    confidence = max(0.0, min(1.0, quality_metrics.get("silhouette_score", 0.0)))

    logger.info(
        "Segmentation done: K=%d, silhouette=%.4f, samples=%d, features=%d, elapsed=%.2fs",
        best_k,
        quality_metrics.get("silhouette_score", 0),
        n_samples,
        n_features,
        elapsed,
    )

    return SegmentationResult(
        status="success",
        cluster_count=best_k,
        labels=[int(lbl) for lbl in cluster_labels],
        feature_columns=final_features,
        cluster_profiles=cluster_profiles,
        quality_metrics=quality_metrics,
        interpretation=interpretation,
        limitations=limitations,
        explanation=explanation,
    )
