"""K-Means segmentation (unsupervised; no target, no held-out test). Deterministic given the seed."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.predictive.contracts import (MIN_FEATURES_SEGMENTATION, MIN_ROWS_SEGMENTATION, ROW_CAP, SILHOUETTE_SAMPLE,
                                      SegmentationResult)


def numeric_candidates(df: pd.DataFrame, exclude=()) -> list:
    out = []
    for c in df.columns:
        if c in exclude or pdt.is_bool_dtype(df[c]) or not pdt.is_numeric_dtype(df[c]):
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() >= 2 and s.nunique(dropna=True) > 1:
            out.append(c)
    return out


def run_segmentation(df: pd.DataFrame, features: list, k: Optional[int] = None, k_range=(2, 8), seed: int = 42,
                     id_column: Optional[str] = None) -> SegmentationResult:
    res = SegmentationResult()
    if df is None or len(df) == 0:
        return res.refuse("The active dataset is empty.", "Upload data first.")
    feats = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        res.warnings.append(f"Columns not found and ignored: {', '.join(missing)}.")
    usable, dropped = [], {}
    for c in feats:
        if not pdt.is_numeric_dtype(df[c]) or pdt.is_bool_dtype(df[c]):
            dropped[c] = "not numeric"
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.nunique(dropna=True) <= 1:
            dropped[c] = "constant or empty"
        else:
            usable.append(c)
    res.excluded_features = dropped
    if len(usable) < MIN_FEATURES_SEGMENTATION:
        return res.refuse(f"At least {MIN_FEATURES_SEGMENTATION} varying numeric columns are needed "
                          f"(found {len(usable)}).", "Choose more numeric columns.")
    X = df[usable].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = X.notna().any(axis=1).to_numpy()
    res.data_prep = {"rows_in_table": int(len(df)), "rows_with_no_numeric_values_excluded": int((~keep).sum())}
    X = X.loc[keep]
    if len(X) < MIN_ROWS_SEGMENTATION:
        return res.refuse(f"Only {len(X)} usable rows (minimum {MIN_ROWS_SEGMENTATION}).", "Add more rows.")
    if len(X) > ROW_CAP:
        sel = np.sort(np.random.default_rng(seed).choice(len(X), ROW_CAP, replace=False))
        res.data_prep["rows_sampled_out"] = int(len(X) - ROW_CAP)
        res.warnings.append(f"{len(X) - ROW_CAP} rows were left out by a seeded random sample (cap {ROW_CAP}).")
        X = X.iloc[sel]
    res.data_prep["rows_clustered"] = int(len(X))
    imp = SimpleImputer(strategy="median")
    Z = StandardScaler().fit_transform(imp.fit_transform(X))
    if X.isna().any().any():
        res.warnings.append("Missing values were filled with column medians before clustering.")
    n = len(Z)
    kmax = min(int(k_range[1]), n - 1)
    ks = list(range(int(k_range[0]), kmax + 1))
    if not ks:
        return res.refuse("Too few rows to compare cluster counts.", "Add more rows.")
    rows, models = [], {}
    sample = None
    if n > SILHOUETTE_SAMPLE:
        sample = np.random.default_rng(seed).choice(n, SILHOUETTE_SAMPLE, replace=False)
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z)
        lab = km.labels_
        if len(set(lab)) < 2:
            sil = None
        elif sample is not None:
            sil = float(silhouette_score(Z[sample], lab[sample])) if len(set(lab[sample])) > 1 else None
        else:
            sil = float(silhouette_score(Z, lab))
        rows.append({"k": kk, "silhouette": sil, "inertia": float(km.inertia_),
                     "smallest_cluster": int(np.bincount(lab).min())})
        models[kk] = km
    res.k_table = pd.DataFrame(rows)
    valid = res.k_table.dropna(subset=["silhouette"])
    if k is None:
        if valid.empty:
            return res.refuse("No cluster count produced a valid silhouette.", "Choose different columns.")
        k = int(valid.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
        res.warnings.append(f"k = {k} was chosen because it has the highest silhouette among the tested values.")
    elif k not in models:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(Z)
        models[int(k)] = km
    km = models[int(k)]
    res.k = int(k)
    lab = km.labels_
    res.features = usable
    sil = res.k_table.loc[res.k_table.k == k, "silhouette"]
    res.metrics = {"k": int(k), "silhouette": (None if sil.empty or pd.isna(sil.iloc[0]) else float(sil.iloc[0])),
                   "inertia": float(km.inertia_), "silhouette_sampled": sample is not None}
    out = X.copy()
    out.insert(0, "source_row", X.index.to_numpy() + 1)
    out["cluster"] = [f"Cluster {i + 1}" for i in lab]
    if id_column and id_column in df.columns:
        out.insert(1, id_column, df.loc[X.index, id_column].to_numpy())
    res.assignments = out.reset_index(drop=True)
    res.profiles = profile_clusters(X, lab, usable)
    pc = PCA(n_components=2, random_state=seed).fit(Z)
    P = pc.transform(Z)
    res.pca = pd.DataFrame({"pc1": P[:, 0], "pc2": P[:, 1], "cluster": [f"Cluster {i + 1}" for i in lab]})
    res.metrics["pca_variance_explained"] = float(pc.explained_variance_ratio_.sum())
    if min(np.bincount(lab)) < max(2, 0.02 * n):
        res.warnings.append("One cluster is very small; consider a lower k.")
    res.status = "ok"
    return res


def profile_clusters(X: pd.DataFrame, labels, feats) -> pd.DataFrame:
    """Per-cluster size, means and a plain-language description from standardised differences to the overall mean."""
    overall_mean, overall_sd = X.mean(), X.std(ddof=0).replace(0, np.nan)
    rows = []
    for i in sorted(set(labels)):
        m = X[labels == i]
        d = {"cluster": f"Cluster {i + 1}", "size": int(len(m)), "share": float(len(m) / len(X))}
        z = {}
        for c in feats:
            d[f"mean_{c}"] = float(m[c].mean())
            z[c] = (m[c].mean() - overall_mean[c]) / overall_sd[c] if pd.notna(overall_sd[c]) else 0.0
        hi = sorted(((c, v) for c, v in z.items() if v >= 0.5), key=lambda t: -t[1])[:3]
        lo = sorted(((c, v) for c, v in z.items() if v <= -0.5), key=lambda t: t[1])[:3]
        parts = []
        if hi:
            parts.append("higher than average " + ", ".join(c for c, _ in hi))
        if lo:
            parts.append("lower than average " + ", ".join(c for c, _ in lo))
        d["description"] = ("; ".join(parts) if parts else "close to the overall average on every column") + \
            f" ({d['share']:.0%} of rows)"
        rows.append(d)
    return pd.DataFrame(rows)
