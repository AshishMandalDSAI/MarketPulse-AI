"""MODEL 06 -- Marketing Resource Optimizer (PART 02).

* spend-based: monetary budget allocation with the platform's existing
  prescriptive engine (``src.prescriptive.engine``) on per-period totals, with
  feasible default limits from ``src.prescriptive.limits``;
* exposure-based: reallocation of exposure *units* between channels measured
  in the same unit, only when the driver model is validated (holdout quality
  at least moderate) and at least two channels in that unit have a positive,
  significant association. Exposure units are never called a budget, and the
  result says that buying exposure has a cost the dataset does not contain;
* general: explains which inputs are missing.

Advanced feasibility handling (reserve budget, constraint diagnostics) is PART 04.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.marketing_intel.context import ModelContext, period_totals
from src.marketing_intel.response import MIN_R2_FOR_DECISIONS, ResponseModel
from src.prescriptive import feasibility as FZ
from src.prescriptive import limits as LIM

LOOKBACK = 8
SIG_P = 0.10
Z90 = 1.645


@dataclass
class ExposureGroup:
    unit: str
    channels: list
    eligible: list = field(default_factory=list)
    held: dict = field(default_factory=dict)          # column -> reason it is held at its current level
    available: bool = False
    reasons: list = field(default_factory=list)


@dataclass
class ExposureOptResult:
    status: str = "unavailable"          # ok | unavailable | infeasible | failed
    message: str = ""
    unit: str = ""
    objective: str = ""
    channels: list = field(default_factory=list)
    current: dict = field(default_factory=dict)
    recommended: dict = field(default_factory=dict)
    mins: dict = field(default_factory=dict)
    maxs: dict = field(default_factory=dict)
    total: float = 0.0
    current_outcome: Optional[float] = None
    recommended_outcome: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    solver: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    solution_status: str = ""          # PART 04: optimal | feasible_not_proven | infeasible | failed | not_supported
    solution_status_text: str = ""

    def table(self) -> pd.DataFrame:
        rows = []
        for c in self.channels:
            cur, rec = self.current.get(c, 0.0), self.recommended.get(c, np.nan)
            rows.append({"channel": c, "unit": self.unit, "current_per_period": cur, "recommended_per_period": rec,
                         "change": rec - cur, "change_pct": 100 * (rec - cur) / cur if cur else np.nan,
                         "min": self.mins.get(c), "max": self.maxs.get(c)})
        return pd.DataFrame(rows)


def exposure_groups(ctx: ModelContext, model: Optional[ResponseModel]) -> list:
    """Which exposure units can be optimised, and why not when they cannot."""
    out = []
    coefs, pv = {}, {}
    if model is not None and model.fit is not None:
        coefs = dict(zip(model.fit.names, model.fit.coef))
        pv = dict(zip(model.fit.names, model.fit.p))
    in_model = {d.column for d in model.drivers} if model is not None else set()
    for unit, cols in ctx.unit_groups().items():
        g = ExposureGroup(unit=unit, channels=list(cols))
        if len(cols) < 2:
            g.reasons.append(f"Only one channel is measured in {unit}; reallocation needs at least two channels in "
                             "the same unit (different units such as impressions and readers cannot be traded).")
        if model is None or model.fit is None:
            g.reasons.append("No validated response model is available.")
        elif not model.usable_for_decisions:
            g.reasons.append(f"The response model is too weak for optimisation ({model.quality_reason}; "
                             f"needs holdout R² ≥ {MIN_R2_FOR_DECISIONS}).")
        for c in cols:
            if c not in in_model:
                g.held[c] = "not in the model"
            elif coefs.get(c, 0) <= 0:
                g.held[c] = "no positive association with the target"
            elif pv.get(c, 1) >= SIG_P:
                g.held[c] = f"association not statistically clear (p = {pv.get(c, 1):.2f})"
            else:
                g.eligible.append(c)
        if len(g.eligible) < 2 and len(cols) >= 2:
            g.reasons.append("Fewer than two channels in this unit have a positive, statistically clear association "
                             "with the target, so there is no defensible reallocation.")
        g.available = not g.reasons
        out.append(g)
    return out


def current_levels(ctx: ModelContext, cols: list, lookback: int = LOOKBACK) -> dict:
    per = period_totals(ctx, cols)
    return {c: float(per[c].tail(lookback).mean()) for c in cols}


def optimize_exposure(ctx: ModelContext, model: ResponseModel, group: ExposureGroup, total: float,
                      mins: dict, maxs: dict, lookback: int = LOOKBACK) -> ExposureOptResult:
    """Maximise the model-estimated target per period by reallocating a fixed total of exposure
    units among the group's eligible channels."""
    res = ExposureOptResult(unit=group.unit, channels=list(group.eligible), mins=dict(mins), maxs=dict(maxs),
                            total=float(total) if total is not None else 0.0)
    res.objective = (f"Maximise the model-estimated {ctx.target} per period by reallocating a fixed total of "
                     f"{group.unit} among {', '.join(ctx.channel_label[c] for c in group.eligible)}")
    if not group.available:
        res.message = " ".join(group.reasons)
        res.solution_status, res.solution_status_text = FZ.classify("not_supported", None)
        return res
    ch = group.eligible
    feas = LIM.feasibility(total, mins, maxs, ch)
    if not feas["feasible"]:
        res.status, res.issues = "infeasible", [m for m in feas["messages"] if not m.startswith("Feasible")]
        res.message = "The limits cannot be satisfied: " + res.issues[0]
        res.solution_status, res.solution_status_text = FZ.classify("infeasible", None)
        return res
    data = ctx.df.loc[model.used_rows]
    dates = np.sort(data[ctx.date_col].unique())
    base = data[data[ctx.date_col] >= dates[-min(lookback, len(dates))]]
    n_per = base[ctx.date_col].nunique()
    cur = current_levels(ctx, ch, lookback)
    if any(cur[c] <= 0 for c in ch):
        zero = [c for c in ch if cur[c] <= 0]
        res.message = ("No recent activity for " + ", ".join(zero) + ", so their response cannot be scaled from "
                       "recent levels.")
        return res
    res.current = cur
    coef = dict(zip(model.fit.names, model.fit.coef))
    raw = {c: pd.to_numeric(base[c], errors="coerce").clip(lower=0).to_numpy(float) for c in ch}
    base_pred = float(model.predict(base).sum() / n_per)

    def outcome(x: np.ndarray) -> float:
        delta = 0.0
        for i, c in enumerate(ch):
            s = x[i] / cur[c]
            delta += coef[c] * float(np.sum(np.log1p(raw[c] * s) - np.log1p(raw[c])))
        return base_pred + delta / n_per

    lo = np.array([float(mins[c]) for c in ch])
    hi = np.array([float(maxs[c]) for c in ch])
    # optimise shares of the total (well-scaled variables) and the outcome change (well-scaled objective)
    T = float(total)
    lo_s, hi_s = lo / T, hi / T
    cur_v = np.array([cur[c] for c in ch])
    obj_scale = max(1e-9, abs(outcome(cur_v * T / cur_v.sum()) - base_pred) + 1e-3 * max(1.0, abs(base_pred)))

    def neg(sh: np.ndarray) -> float:
        return -(outcome(sh * T) - base_pred) / obj_scale

    cons = [{"type": "eq", "fun": lambda sh: float(np.sum(sh) - 1.0)}]
    starts = [np.clip(cur_v / cur_v.sum(), lo_s, hi_s), np.clip(np.full(len(ch), 1.0 / len(ch)), lo_s, hi_s),
              (lo_s + hi_s) / 2]
    best = None
    runs = 0
    vals = []
    for start in starts:
        runs += 1
        try:
            r = minimize(neg, start, method="SLSQP", bounds=list(zip(lo_s, hi_s)), constraints=cons,
                         options={"maxiter": 1000, "ftol": 1e-12})
        except (ValueError, FloatingPointError):
            continue
        x = r.x * T
        vals.append((float(outcome(np.clip(x, lo, hi))), bool(r.success)))
        ok = abs(float(np.sum(x)) - T) <= 1e-6 * max(1.0, T) and np.all(x >= lo - 1e-6 * T) and np.all(x <= hi + 1e-6 * T)
        if ok and (best is None or outcome(x) > outcome(best[0].x * T)):
            best = (r, bool(r.success))
    if best is None:
        res.status, res.message = "failed", "The optimiser did not find an allocation that satisfies the limits."
        res.solution_status, res.solution_status_text = FZ.classify("failed", None)
        return res
    r, conv = best
    convs = [v for v, ok in vals if ok]
    top = max(convs) if convs else None
    res.solver = {"method": "SLSQP (scipy), 3 starting points", "converged": conv, "message": str(r.message),
                  "iterations": int(getattr(r, "nit", 0)), "starts": runs, "starts_converged": len(convs),
                  "starts_agreeing": sum(1 for v in convs if top is not None and abs(v - top) <= 1e-6 * max(1.0, abs(top)))}
    xr = np.clip(r.x * T, lo, hi)
    res.recommended = {c: float(xr[i]) for i, c in enumerate(ch)}
    res.current_outcome = outcome(np.array([cur[c] * total / sum(cur.values()) for c in ch])) \
        if abs(sum(cur.values()) - total) > 1e-6 * max(1, total) else outcome(np.array([cur[c] for c in ch]))
    res.recommended_outcome = outcome(xr)
    res.change = res.recommended_outcome - res.current_outcome
    res.change_pct = 100 * res.change / res.current_outcome if res.current_outcome else None
    pr = (model.holdout or {}).get("period_rmse")
    if pr and np.isfinite(pr):
        res.low, res.high = res.recommended_outcome - Z90 * pr, res.recommended_outcome + Z90 * pr
    res.status = "ok"
    res.solution_status, res.solution_status_text = FZ.classify("ok", res.solver)
    hist_max = {c: float(period_totals(ctx, [c])[c].max()) for c in ch}
    for c in ch:
        if res.recommended[c] > hist_max[c] * 1.0001:
            res.warnings.append(f"{ctx.channel_label[c]}: the recommended level is above the highest level seen in "
                                "the data (extrapolation).")
    if group.held:
        res.warnings.append("Held at current levels: " + "; ".join(f"{ctx.channel_label[c]} ({why})" for c, why in group.held.items()))
    res.assumptions = [
        f"Units are {group.unit}. This is an exposure mix, not a monetary budget: the cost of one unit differs "
        "between channels and is not in this dataset, so the result does not say the new mix is affordable.",
        f"Current level = average per period over the last {min(lookback, len(dates))} periods; when the total you "
        "enter differs, 'current' is the current mix scaled to that total.",
        "Each channel's response follows the fitted log(1 + exposure) curve (diminishing returns).",
        "Other drivers stay at their recorded values in the baseline window.",
        "Associations in past data, not proven causal effects; validate with a controlled test before acting.",
    ]
    return res
