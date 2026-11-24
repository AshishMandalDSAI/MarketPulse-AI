"""Driver response model shared by the Effectiveness model, Channel Performance,
the exposure-based Resource Optimizer and the Scenario Simulator.

A transparent linear regression of the target on the available drivers:

* marketing channels (spend or exposure) enter as ``log(1 + x)``, so each extra
  unit adds less than the one before (diminishing returns) -- a modelling
  assumption, stated in the results;
* price, promotion, discount, trade spend, distribution, competitor and
  external factors enter linearly;
* optional fixed effects for brand / geography / product / SKU absorb level
  differences between segments, and an optional linear time trend absorbs
  steady growth or decline.

Validation is chronological: the last ~20% of periods are held out, the model
is fitted on the earlier periods and scored on the held-out ones (row level and
per-period totals). Coefficients shown to users come from a refit on all rows
with HC3 robust standard errors. All results are statistical associations in
observational data, not proof of cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.marketing_intel import stats as S
from src.marketing_intel.context import ModelContext

GOOD_R2 = 0.5
MIN_R2_FOR_DECISIONS = 0.3
HOLDOUT_SHARE = 0.2
MAX_ROWS = 100_000
CHANNEL_TRANSFORM = "log1p"


@dataclass
class DriverInfo:
    column: str
    label: str
    role: str
    kind: str                 # channel | business | other
    transform: str            # log1p | linear
    unit: str = ""


@dataclass
class ResponseModel:
    target: str
    drivers: list
    fe_dims: list
    trend: bool
    design_cols: list = field(default_factory=list)
    fe_levels: dict = field(default_factory=dict)
    date_col: Optional[str] = None
    date_min: Optional[pd.Timestamp] = None
    fit: Optional[S.OLSFit] = None
    train_fit: Optional[S.OLSFit] = None
    holdout: dict = field(default_factory=dict)
    quality: str = "unavailable"          # good | moderate | weak | unavailable
    quality_reason: str = ""
    n_rows: int = 0
    n_dropped: int = 0
    n_train: int = 0
    n_test: int = 0
    split: str = ""
    vif: dict = field(default_factory=dict)
    importance: Optional[pd.DataFrame] = None
    notes: list = field(default_factory=list)
    used_rows: Optional[pd.Index] = None
    y_mean: float = 0.0
    y_std: float = 0.0

    # ------------------------------------------------------------ design / predict
    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for d in self.drivers:
            x = pd.to_numeric(df[d.column], errors="coerce").astype(float)
            X[d.column] = np.log1p(x.clip(lower=0)) if d.transform == "log1p" else x
        if self.trend and self.date_col:
            X["__trend__"] = (pd.to_datetime(df[self.date_col]) - self.date_min).dt.days / 7.0
        for dim, levels in self.fe_levels.items():
            vals = df[dim].astype(str)
            for lv in levels[1:]:
                X[f"__fe__{dim}={lv}"] = (vals == lv).astype(float)
        return X[self.design_cols] if self.design_cols else X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit.predict(self.design(df).to_numpy(dtype=float))

    @property
    def usable_for_decisions(self) -> bool:
        return self.quality in ("good", "moderate")

    def coef_table(self) -> pd.DataFrame:
        if self.fit is None:
            return pd.DataFrame()
        t = self.fit.table()
        t = t[t.term.isin([d.column for d in self.drivers] + ["__trend__"])].copy()
        info = {d.column: d for d in self.drivers}
        t["driver"] = [info[c].label if c in info else "Time trend (per week)" for c in t.term]
        t["transform"] = [info[c].transform if c in info else "linear" for c in t.term]
        t["significant_5pct"] = t.p_value < 0.05
        t["vif"] = [self.vif.get(c, np.nan) for c in t.term]
        return t.reset_index(drop=True)


def select_drivers(ctx: ModelContext, requested: Optional[list] = None, corr_limit: float = 0.98) -> tuple[list, list]:
    """Usable drivers (in order) and notes. Drops constant / missing columns and, for the default set,
    near-duplicates (|r| > ``corr_limit``) of an earlier driver."""
    cand = list(dict.fromkeys(requested if requested is not None else ctx.default_drivers()))
    notes, keep = [], []
    for c in cand:
        if c not in ctx.df.columns or c == ctx.target or c in ctx.excluded:
            if c in ctx.excluded:
                notes.append(f"'{c}' was left out: {ctx.excluded[c]}.")
            continue
        s = pd.to_numeric(ctx.df[c], errors="coerce")
        if s.notna().mean() < 0.5:
            notes.append(f"'{c}' was left out: more than half of its values are missing.")
            continue
        if s.nunique() < 2:
            notes.append(f"'{c}' was left out: it does not vary.")
            continue
        if requested is None:
            dup = next((k for k in keep if abs(np.corrcoef(*_pair(ctx.df, c, k))[0, 1]) > corr_limit), None)
            if dup:
                notes.append(f"'{c}' was left out by default: it moves almost exactly with '{dup}' "
                             f"(|r| > {corr_limit}), so their separate effects cannot be told apart. You can add it back.")
                continue
        keep.append(c)
    return keep, notes


def _pair(df, a, b):
    m = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(m) < 3 or m[a].std() == 0 or m[b].std() == 0:
        return np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 1.0])
    return m[a].to_numpy(float), m[b].to_numpy(float)


def drop_nested_dims(df: pd.DataFrame, dims: list) -> tuple[list, list]:
    """Keep the finer of two nested dimensions (e.g. SKU within Brand): the coarser one adds no
    information once the finer one has fixed effects, and keeping both makes them collinear."""
    keep = list(dims)
    notes = []
    for a in dims:
        for b in dims:
            if a == b or a not in keep or b not in keep:
                continue
            if df[a].nunique() > df[b].nunique() and (df.groupby(a, observed=True)[b].nunique() <= 1).all():
                keep.remove(b)
                notes.append(f"Segment effects use '{a}' (each {a} belongs to one {b}, so '{b}' is already covered).")
    return keep, notes


def driver_info(ctx: ModelContext, cols: list, channel_transform: str = CHANNEL_TRANSFORM) -> list:
    out = []
    for c in cols:
        role = ctx.mapping.role_of(c) or "measure"
        if c in ctx.channels:
            out.append(DriverInfo(c, ctx.channel_label.get(c, c) + f" ({ctx.channel_unit.get(c, '')})", role,
                                  "channel", channel_transform, ctx.channel_unit.get(c, "")))
        else:
            kind = "business" if role in ("price", "promotion", "discount", "trade_spend", "distribution",
                                          "competitor", "external", "quality_score") else "other"
            out.append(DriverInfo(c, c, role, kind, "linear", ctx.mapping.unit_of(c)))
    return out


def fit_response(ctx: ModelContext, drivers: Optional[list] = None, *, fixed_effects: Optional[list] = None,
                 trend: bool = True, channel_transform: str = CHANNEL_TRANSFORM, seed: int = 0,
                 importance: bool = True) -> ResponseModel:
    """Fit the driver response model with chronological holdout validation."""
    cols, notes = select_drivers(ctx, drivers)
    fe = [d for d in (fixed_effects if fixed_effects is not None else ctx.dim_cols) if d in ctx.df.columns]
    fe = [d for d in fe if 1 < ctx.df[d].nunique() <= 60]
    fe, nested_notes = drop_nested_dims(ctx.df, fe)
    notes = notes + nested_notes
    m = ResponseModel(target=ctx.target, drivers=driver_info(ctx, cols, channel_transform), fe_dims=fe,
                      trend=bool(trend and ctx.date_col), date_col=ctx.date_col, notes=notes)
    if not cols:
        m.quality, m.quality_reason = "unavailable", "No usable driver columns were found for this target."
        return m
    need = [ctx.target] + cols + fe + ([ctx.date_col] if ctx.date_col else [])
    data = ctx.df[list(dict.fromkeys(need))].copy()
    data[ctx.target] = pd.to_numeric(data[ctx.target], errors="coerce")
    before = len(data)
    data = data.dropna()
    m.n_dropped = before - len(data)
    if m.n_dropped:
        m.notes.append(f"{m.n_dropped:,} row(s) with missing values in the target or drivers were left out.")
    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=seed).sort_index()
        m.notes.append(f"The model uses a fixed random sample of {MAX_ROWS:,} rows for speed.")
    m.n_rows = len(data)
    if ctx.date_col:
        m.date_min = pd.to_datetime(data[ctx.date_col]).min()
    m.fe_levels = {d: sorted(data[d].astype(str).unique().tolist()) for d in fe}
    m.design_cols = []
    X = m.design(data)
    m.design_cols = list(X.columns)
    y = data[ctx.target].to_numpy(dtype=float)
    k = X.shape[1]
    if len(data) < max(12, 3 * (k + 1)):
        m.quality = "unavailable"
        m.quality_reason = (f"Only {len(data)} usable rows for {k} model terms; at least {max(12, 3 * (k + 1))} are "
                            "needed for a stable estimate.")
        return m
    # chronological split
    if ctx.date_col:
        dates = np.sort(pd.to_datetime(data[ctx.date_col]).unique())
        n_test = max(2, int(round(len(dates) * HOLDOUT_SHARE)))
        if len(dates) - n_test < 4:
            n_test = 0
        cut = dates[-n_test] if n_test else None
        test_mask = (pd.to_datetime(data[ctx.date_col]) >= cut).to_numpy() if n_test else np.zeros(len(data), bool)
        m.split = (f"chronological: trained on periods before {pd.Timestamp(cut).date()}, tested on the last "
                   f"{n_test} period(s)" if n_test else "no holdout (too few periods)")
    else:
        rng = np.random.default_rng(seed)
        test_mask = rng.random(len(data)) < HOLDOUT_SHARE
        m.split = "random 20% holdout (no date column, so the split cannot follow time)"
    Xa = X.to_numpy(dtype=float)
    m.fit = S.ols_hc3(Xa, y, list(X.columns))
    m.notes += m.fit.notes
    m.used_rows = data.index
    m.y_mean, m.y_std = float(np.mean(y)), float(np.std(y))
    tr, te = ~test_mask, test_mask
    m.n_train, m.n_test = int(tr.sum()), int(te.sum())
    if te.sum() >= 4 and tr.sum() > k + 1:
        m.train_fit = S.ols_hc3(Xa[tr], y[tr], list(X.columns))
        pred = m.train_fit.predict(Xa[te])
        hold = {"r2": S.r2_score(y[te], pred), "rmse": S.rmse(y[te], pred), "mape": S.mape(y[te], pred),
                "n": int(te.sum()), "train_r2": m.train_fit.r2}
        if ctx.date_col:
            per = pd.DataFrame({"d": pd.to_datetime(data[ctx.date_col]).to_numpy()[te], "y": y[te], "p": pred}) \
                .groupby("d")[["y", "p"]].sum()
            hold.update(period_r2=S.r2_score(per.y, per.p) if len(per) >= 3 else float("nan"),
                        period_rmse=S.rmse(per.y, per.p), period_mape=S.mape(per.y, per.p), n_periods=len(per),
                        period_mean=float(per.y.mean()))
        m.holdout = hold
        score = hold.get("period_r2") if ctx.is_panel and np.isfinite(hold.get("period_r2", np.nan)) else hold["r2"]
        basis = "per-period totals" if ctx.is_panel and np.isfinite(hold.get("period_r2", np.nan)) else "rows"
        if not np.isfinite(score):
            m.quality, m.quality_reason = "weak", "The holdout R² could not be computed."
        elif score >= GOOD_R2:
            m.quality = "good"
        elif score >= MIN_R2_FOR_DECISIONS:
            m.quality = "moderate"
        else:
            m.quality = "weak"
        if not m.quality_reason:
            m.quality_reason = f"holdout R² on {basis} = {score:.2f} (good ≥ {GOOD_R2}, usable ≥ {MIN_R2_FOR_DECISIONS})"
        if importance:
            groups = {d.label: [d.column] for d in m.drivers}
            fe_cols = [c for c in X.columns if c.startswith("__fe__")]
            if fe_cols:
                groups["Segment differences (fixed effects)"] = fe_cols
            if "__trend__" in X.columns:
                groups["Time trend"] = ["__trend__"]
            tf = m.train_fit
            m.importance = S.permutation_importance(lambda Z: tf.predict(Z.to_numpy(dtype=float)),
                                                    X[te].reset_index(drop=True), y[te], groups, n_repeats=10,
                                                    seed=seed)
    else:
        m.quality = "weak"
        m.quality_reason = "Too few rows to hold out a validation period, so the model is not validated."
    drv_cols = [d.column for d in m.drivers]
    if len(drv_cols) >= 2:
        m.vif = S.vif(X[drv_cols])
    return m


def standardized_effects(m: ResponseModel, ctx: ModelContext) -> pd.DataFrame:
    """Per-driver effect summary: coefficient, standardized effect, elasticity at average levels and
    a statistical contribution share for channels. Units follow the target."""
    if m.fit is None:
        return pd.DataFrame()
    data = ctx.df.loc[m.used_rows]
    X = m.design(data)
    y = pd.to_numeric(data[ctx.target], errors="coerce").to_numpy(float)
    rows = []
    coef = dict(zip(m.fit.names, m.fit.coef))
    pv = dict(zip(m.fit.names, m.fit.p))
    lo = dict(zip(m.fit.names, m.fit.ci_low))
    hi = dict(zip(m.fit.names, m.fit.ci_high))
    ymean = float(np.mean(y)) if len(y) else np.nan
    for d in m.drivers:
        b = coef[d.column]
        xd = X[d.column].to_numpy(float)
        raw = pd.to_numeric(data[d.column], errors="coerce").to_numpy(float)
        std_eff = b * np.std(xd) / np.std(y) if np.std(y) > 0 else np.nan
        if d.transform == "log1p":
            elastic = float(np.mean(b * raw / (1 + np.clip(raw, 0, None)))) / ymean if ymean else np.nan
        else:
            elastic = b * float(np.mean(raw)) / ymean if ymean else np.nan
        contrib = float(np.mean(b * xd))
        rows.append({"column": d.column, "driver": d.label, "kind": d.kind, "role": d.role, "transform": d.transform,
                     "coef": b, "ci95_low": lo[d.column], "ci95_high": hi[d.column], "p_value": pv[d.column],
                     "significant_5pct": pv[d.column] < 0.05, "standardized_effect": std_eff,
                     "elasticity_at_mean": elastic, "avg_modelled_effect": contrib, "vif": m.vif.get(d.column, np.nan)})
    t = pd.DataFrame(rows)
    ch = t[t.kind == "channel"]
    if len(ch):
        pos = ch.avg_modelled_effect.clip(lower=0)
        tot = pos.sum()
        t["association_share_pct"] = np.nan
        if tot > 0:
            t.loc[ch.index, "association_share_pct"] = 100 * pos / tot
    return t
