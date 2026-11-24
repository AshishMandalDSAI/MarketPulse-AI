"""Target-dependent screening. PURE functions of ``(X_fit, y_fit)`` only.

No function here accepts a full frame plus an index, a ``DatasetSummary`` or any validation/test row. The
caller (``procedure.ScreenedModel``) slices the training subset first: ``T_f`` in each CV fold (P1) and ``T`` in
the final review (P3). Evidence is therefore always PARTITION-LOCAL: it is computed exclusively from the
partition that was passed in (recorded in ``ScreenResult.partition``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data_engine.canonical import build_column_metadata
from src.predictive.contracts import (ID_RATIO, MAX_CARDINALITY, SEED, TIER2_DISPLAY, TIER2_MAX_CHECKED,
                                      TIER2_SINGLE_FEATURE)


@dataclass
class ScreenResult:
    partition: str
    n_rows: int
    row_ids: tuple
    x_hash: str
    y_hash: str
    eligible: list = field(default_factory=list)
    excluded: dict = field(default_factory=dict)     # col -> {tier, reason, evidence}
    tier1a: dict = field(default_factory=dict)
    tier1b: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    assoc: dict = field(default_factory=dict)
    types: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

def _hash(obj) -> str:
    return format(int(pd.util.hash_pandas_object(obj, index=True).sum()) & 0xFFFFFFFFFFFFFFFF, "x")


def _is_integer_valued(s: pd.Series) -> bool:
    if not pdt.is_numeric_dtype(s) or pdt.is_bool_dtype(s):
        return False
    v = s.dropna()
    return len(v) > 0 and bool(np.all(np.isfinite(v))) and bool(np.all(np.floor(v) == v))


def classify_columns(X_fit: pd.DataFrame) -> dict:
    """Value-based typing of the columns of ``X_fit`` ONLY (platform classifier applied to this partition)."""
    frame = X_fit
    try:
        meta = build_column_metadata(frame)
    except TypeError:  # unhashable cells
        frame = frame.astype(str)
        meta = build_column_metadata(frame)
    out = {}
    for m in meta:
        s = X_fit[m.name]
        out[m.name] = {
            "semantic": m.semantic_type, "constant": bool(m.is_constant), "n_distinct": int(m.unique_count),
            "ratio": float(m.unique_count / max(len(X_fit), 1)),
            "integer_valued": _is_integer_valued(s),
            "numeric_dtype": bool(pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s)),
            "n_rows": int(len(X_fit)),
        }
    return out


def classify_identifiers(X_fit: pd.DataFrame, types: dict, partition: str) -> tuple:
    """Return (tier1b {col: evidence}, continuous_kept {col: evidence}). Partition-local evidence only."""
    tier1b, kept = {}, {}
    for c, t in types.items():
        if t["semantic"] == "empty" or t["constant"]:
            continue
        ev = {"partition": partition, "n_rows": t["n_rows"], "n_distinct": t["n_distinct"],
              "ratio": round(t["ratio"], 4), "platform_type": t["semantic"]}
        platform = t["semantic"] == "identifier"
        continuous = t["numeric_dtype"] and not t["integer_valued"]
        if continuous:
            if platform:
                kept[c] = {**ev, "rule": "continuous numeric column classified as identifier by the platform "
                                         "classifier; kept (never excluded on distinct-value ratio alone)"}
            continue
        by_ratio = (t["ratio"] >= ID_RATIO and t["n_rows"] >= 10
                    and (t["integer_valued"] or t["semantic"] in ("categorical", "text", "identifier")
                         or not t["numeric_dtype"]))
        if platform or by_ratio:
            tier1b[c] = {**ev, "rule": ("platform classifier" if platform else "distinct-value ratio")
                         + f" on {partition} ({t['n_distinct']} distinct in {t['n_rows']} rows)"}
    return tier1b, kept


def _eq_series(a: pd.Series, b: pd.Series) -> bool:
    if pdt.is_numeric_dtype(a) and pdt.is_numeric_dtype(b) and not pdt.is_bool_dtype(a) and not pdt.is_bool_dtype(b):
        return bool(np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True))
    return bool(((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).all())


def detect_tier1a(X_fit: pd.DataFrame, y_fit: pd.Series, cols: list, partition: str) -> dict:
    """Exact copies, one-to-one recodings and perfect linear recodings of the target (never usable)."""
    out = {}
    n = len(X_fit)
    y = y_fit.reset_index(drop=True)
    for c in cols:
        x = X_fit[c].reset_index(drop=True)
        ev = {"partition": partition, "n_rows": n}
        if _eq_series(x, y):
            out[c] = {**ev, "reason": "exact copy of the target on every row of " + partition}
            continue
        nx, ny = int(x.nunique(dropna=False)), int(y.nunique(dropna=False))
        if nx >= 2 and ny >= 2 and nx <= 0.5 * n and ny <= 0.5 * n:
            f = pd.DataFrame({"x": x.astype(str), "y": y.astype(str)})
            if (f.groupby("x")["y"].nunique() == 1).all() and (f.groupby("y")["x"].nunique() == 1).all():
                out[c] = {**ev, "mapping_size": nx,
                          "reason": f"one-to-one recoding of the target ({nx} values map to {ny}) on every row of {partition}"}
                continue
        if pdt.is_numeric_dtype(x) and pdt.is_numeric_dtype(y) and not pdt.is_bool_dtype(x) and n >= 3:
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            if ok.sum() >= 3 and ok.all() and xv.std() > 0 and yv.std() > 0:
                r = float(np.corrcoef(xv, yv)[0, 1])
                if abs(r) >= 1 - 1e-12:
                    out[c] = {**ev, "r": r, "reason": f"perfect linear recoding of the target (|r| = 1) on {partition}"}
    return out


def _association(x: pd.Series, y: pd.Series) -> float:
    """Association strength in [0, 1] between one feature and the target (training partition only)."""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 5 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    xn = pdt.is_numeric_dtype(x) and not pdt.is_bool_dtype(x)
    yn = pdt.is_numeric_dtype(y) and not pdt.is_bool_dtype(y) and y.nunique() > 10
    if xn and yn:
        p = abs(float(np.corrcoef(x.astype(float), y.astype(float))[0, 1]))
        s = abs(float(np.corrcoef(x.rank(), y.rank())[0, 1]))
        return float(np.nan_to_num(max(p, s)))
    if xn and not yn:
        return _eta(x.astype(float), y.astype(str))
    if (not xn) and yn:
        return _eta(y.astype(float), x.astype(str))
    ct = pd.crosstab(x.astype(str), y.astype(str)).to_numpy(float)
    n = ct.sum()
    if n == 0 or min(ct.shape) < 2:
        return 0.0
    exp = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = float(((ct - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def _eta(values: pd.Series, groups: pd.Series) -> float:
    v = values.to_numpy(float)
    sst = float(((v - v.mean()) ** 2).sum())
    if sst <= 0:
        return 0.0
    gm = pd.Series(v).groupby(groups.to_numpy()).transform("mean").to_numpy()
    ssb = float(((gm - v.mean()) ** 2).sum())
    return float(np.sqrt(min(max(ssb / sst, 0.0), 1.0)))


def _single_feature_score(x: pd.Series, y: pd.Series, task: str, seed: int) -> float:
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 12:
        return float("nan")
    if pdt.is_numeric_dtype(x) and not pdt.is_bool_dtype(x):
        X = x.astype(float).to_numpy().reshape(-1, 1)
    else:
        if x.nunique() > MAX_CARDINALITY:
            return float("nan")
        X = pd.get_dummies(x.astype(str)).to_numpy(float)
    yv = y.to_numpy()
    scores = []
    for tr, va in KFold(3, shuffle=True, random_state=seed).split(X):
        if task == "regression":
            m = DecisionTreeRegressor(max_depth=3, random_state=seed).fit(X[tr], yv[tr].astype(float))
            p = m.predict(X[va]); t = yv[va].astype(float)
            sst = float(((t - t.mean()) ** 2).sum())
            scores.append(1 - float(((t - p) ** 2).sum()) / sst if sst > 0 else float("nan"))
        else:
            m = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(X[tr], yv[tr].astype(str))
            scores.append(float((m.predict(X[va]) == yv[va].astype(str)).mean()))
    return float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else float("nan")


def tier2_evidence(X_fit: pd.DataFrame, y_fit: pd.Series, task: str, cols: list, partition: str, seed: int) -> tuple:
    """Return (flags {col: evidence}, assoc {col: float}). Flags only warn; they never exclude."""
    y = y_fit.reset_index(drop=True)
    assoc, flags = {}, {}
    for c in cols:
        x = X_fit[c].reset_index(drop=True)
        assoc[c] = round(_association(x, y), 6)
    ranked = sorted(assoc, key=lambda k: (-assoc[k], k))
    for c in ranked:
        if assoc[c] >= TIER2_DISPLAY:
            flags[c] = {"partition": partition, "association": assoc[c],
                        "reason": f"strong association with the target ({assoc[c]:.3f}) on {partition}; "
                                  "kept unless you exclude it"}
    for c in ranked[:TIER2_MAX_CHECKED]:
        sc = _single_feature_score(X_fit[c].reset_index(drop=True), y, task, seed)
        if sc == sc and sc >= TIER2_SINGLE_FEATURE:
            ev = flags.setdefault(c, {"partition": partition, "association": assoc[c], "reason": ""})
            ev["single_feature_cv_score"] = round(sc, 4)
            ev["reason"] = (ev["reason"] + "; " if ev["reason"] else "") + \
                f"this one column alone reaches a cross-validated score of {sc:.3f} on {partition}"
    for c in cols:
        m = X_fit[c].isna().reset_index(drop=True)
        if 3 <= int(m.sum()) <= len(m) - 3:
            a = _association(m.astype(int).astype(str).astype("category"), y)
            if a >= TIER2_DISPLAY:
                ev = flags.setdefault(c, {"partition": partition, "association": assoc.get(c, 0.0), "reason": ""})
                ev["missingness_association"] = round(a, 4)
                ev["reason"] = (ev["reason"] + "; " if ev["reason"] else "") + \
                    f"whether the value is missing almost determines the target ({a:.3f}) on {partition}"
    return flags, assoc


# ---------------------------------------------------------------- the screen

def screen_features(X_fit: pd.DataFrame, y_fit: pd.Series, task: str, partition: str,
                    seed: int = SEED, with_tier2: bool = True) -> ScreenResult:
    """Run every target-dependent screen on ``(X_fit, y_fit)`` and nothing else."""
    res = ScreenResult(partition=partition, n_rows=int(len(X_fit)), row_ids=tuple(X_fit.index.tolist()),
                       x_hash=_hash(X_fit), y_hash=_hash(y_fit))
    res.types = classify_columns(X_fit)
    live = []
    for c, t in res.types.items():
        if t["semantic"] == "empty":
            res.excluded[c] = {"tier": "eligibility", "reason": f"empty on {partition}"}
        elif t["constant"]:
            res.excluded[c] = {"tier": "eligibility", "reason": f"constant on {partition}"}
        elif t["semantic"] == "datetime":
            res.excluded[c] = {"tier": "eligibility", "reason": f"date-like values on {partition}; never a raw predictor"}
        elif t["semantic"] == "text":
            res.excluded[c] = {"tier": "eligibility", "reason": f"free text on {partition}"}
        else:
            live.append(c)
    res.tier1a = detect_tier1a(X_fit, y_fit, live, partition)
    for c, ev in res.tier1a.items():
        res.excluded[c] = {"tier": "1a", "reason": ev["reason"], "evidence": ev}
    live = [c for c in live if c not in res.tier1a]
    tier1b, continuous_kept = classify_identifiers(X_fit, {c: res.types[c] for c in live}, partition)
    res.tier1b = tier1b
    for c, ev in tier1b.items():
        res.excluded[c] = {"tier": "1b", "reason": "identifier-like: " + ev["rule"], "evidence": ev}
    live = [c for c in live if c not in tier1b]
    for c in list(live):
        t = res.types[c]
        if not t["numeric_dtype"] and t["n_distinct"] > MAX_CARDINALITY:
            res.excluded[c] = {"tier": "eligibility",
                               "reason": f"{t['n_distinct']} distinct categories on {partition} (limit {MAX_CARDINALITY})"}
            live.remove(c)
    res.eligible = live
    if with_tier2 and (live or tier1b):
        flags, assoc = tier2_evidence(X_fit, y_fit, task, live + list(tier1b), partition, seed)
        res.tier2, res.assoc = flags, assoc
    for c, ev in continuous_kept.items():
        res.tier2.setdefault(c, {"partition": partition, "association": 0.0, "reason": ev["rule"]})
    return res
