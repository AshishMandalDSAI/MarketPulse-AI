"""Small, dependency-free statistics used by the PART 02 marketing models.

* ``ols_hc3`` -- ordinary least squares with HC3 heteroskedasticity-robust
  standard errors (MacKinnon & White 1985), t-tests and 95% intervals.
* ``vif`` -- variance inflation factors for multicollinearity checks.
* ``permutation_importance`` -- drop in holdout R² when one driver (or a group
  of design columns) is shuffled; deterministic for a given seed.
* metrics helpers.

Only numpy / scipy, so no new dependency is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class OLSFit:
    names: list
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    r2: float
    adj_r2: float
    n: int
    k: int
    resid_std: float
    rank_deficient: bool = False
    notes: list = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.coef[0] + X @ self.coef[1:]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names, "coef": self.coef, "std_error_hc3": self.se, "t": self.t,
                             "p_value": self.p, "ci95_low": self.ci_low, "ci95_high": self.ci_high})


def ols_hc3(X: np.ndarray, y: np.ndarray, names: list) -> OLSFit:
    """OLS with an intercept; ``X`` excludes the constant. HC3 robust covariance."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    xtx = Xc.T @ Xc
    rank = np.linalg.matrix_rank(Xc)
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ Xc.T @ y
    resid = y - Xc @ beta
    h = np.einsum("ij,jk,ik->i", Xc, xtx_inv, Xc)
    h = np.clip(h, 0, 1 - 1e-8)
    w = (resid / (1 - h)) ** 2
    meat = (Xc * w[:, None]).T @ Xc
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    dof = max(n - (k + 1), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = 2 * sps.t.sf(np.abs(t), dof)
    crit = sps.t.ppf(0.975, dof)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / dof if ss_tot > 0 else float("nan")
    notes = []
    if rank < k + 1:
        notes.append("Some drivers are exact linear combinations of others; their individual coefficients are "
                     "not identifiable (the minimum-norm solution is shown).")
    return OLSFit(names=["Intercept"] + list(names), coef=beta, se=se, t=t, p=p, ci_low=beta - crit * se,
                  ci_high=beta + crit * se, r2=float(r2), adj_r2=float(adj), n=n, k=k,
                  resid_std=float(np.sqrt(ss_res / dof)), rank_deficient=rank < k + 1, notes=notes)


def vif(frame: pd.DataFrame) -> dict:
    """Variance inflation factor of each column against all the others (inf for exact collinearity)."""
    out = {}
    cols = list(frame.columns)
    A = frame.to_numpy(dtype=float)
    for i, c in enumerate(cols):
        others = np.delete(A, i, axis=1)
        if others.shape[1] == 0:
            out[c] = 1.0
            continue
        y = A[:, i]
        if np.std(y) == 0:
            out[c] = float("inf")
            continue
        Xc = np.column_stack([np.ones(len(y)), others])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ beta
        r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
        out[c] = float("inf") if r2 >= 1 - 1e-10 else float(1.0 / (1.0 - r2))
    return out


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat) -> Optional[float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.any() else None


def permutation_importance(predict, X: pd.DataFrame, y: np.ndarray, groups: dict, n_repeats: int = 10,
                           seed: int = 0) -> pd.DataFrame:
    """Mean and std of the drop in R² when each group of design columns is shuffled (rows permuted
    together inside a group)."""
    rng = np.random.default_rng(seed)
    base = r2_score(y, predict(X))
    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[idx]
            drops.append(base - r2_score(y, predict(Xp)))
        rows.append({"driver": name, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True) if rows else \
        pd.DataFrame(columns=["driver", "importance", "std"])


def welch_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(t statistic, two-sided p-value) of Welch's unequal-variance t-test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    res = sps.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def corr_with_p(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan"), float("nan")
    res = sps.pearsonr(x[m], y[m]) if method == "pearson" else sps.spearmanr(x[m], y[m])
    return float(res[0]), float(res[1])


def bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if not m.any():
        return out
    q = p[m]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(q)
    adj[order] = np.clip(ranked, 0, 1)
    out[m] = adj
    return out
