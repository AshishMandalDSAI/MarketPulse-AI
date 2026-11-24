"""MODEL 07 -- Marketing Scenario Simulator (PART 02).

Scenarios are changes to the drivers of the validated response model, applied
to a recent baseline window:

* spend / exposure / price / discount / distribution / other numeric drivers:
  percentage change (``{"type": "pct", "value": 10}``);
* 0/1 promotion flags: share of records with the promotion on
  (``{"type": "share", "value": 30}`` -- 30%), applied as its expected value.

Historical actuals and model estimates are kept apart: the baseline shows the
actual average per period *and* the model's estimate for the same periods, and
scenario results are always model estimates with an approximate range from the
holdout error. Changes beyond the observed range are flagged as extrapolation;
a weak model is flagged as unreliable. Nothing here is a forecast of the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.marketing_intel.context import ModelContext
from src.marketing_intel.response import ResponseModel

Z90 = 1.645
DEFAULT_WINDOW = 8


@dataclass
class Scenario:
    name: str
    changes: dict = field(default_factory=dict)      # column -> {"type": "pct"|"share", "value": float}


@dataclass
class ScenarioOutcome:
    name: str
    estimate: float
    change: float
    change_pct: Optional[float]
    low: Optional[float]
    high: Optional[float]
    extrapolation: list = field(default_factory=list)
    changes_text: list = field(default_factory=list)


@dataclass
class SimulationResult:
    status: str = "unavailable"
    message: str = ""
    per: str = "period"
    window_text: str = ""
    actual_baseline: Optional[float] = None
    model_baseline: Optional[float] = None
    outcomes: list = field(default_factory=list)
    reliability: str = ""
    reliability_text: str = ""
    range_basis: str = ""
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    mmm_check: list = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [{"scenario": "Historical actual (baseline window)", "kind": "actual", "estimate": self.actual_baseline,
                 "change_vs_model_baseline": np.nan, "change_pct": np.nan, "low_90": np.nan, "high_90": np.nan,
                 "extrapolation": ""},
                {"scenario": "Model baseline (no change)", "kind": "model estimate", "estimate": self.model_baseline,
                 "change_vs_model_baseline": 0.0, "change_pct": 0.0, "low_90": np.nan, "high_90": np.nan,
                 "extrapolation": ""}]
        for o in self.outcomes:
            rows.append({"scenario": o.name, "kind": "model estimate", "estimate": o.estimate,
                         "change_vs_model_baseline": o.change, "change_pct": o.change_pct, "low_90": o.low,
                         "high_90": o.high, "extrapolation": "; ".join(o.extrapolation)})
        return pd.DataFrame(rows)


def adjustable_drivers(model: ResponseModel, ctx: ModelContext) -> list:
    """(column, label, kind) of drivers a scenario may change."""
    out = []
    for d in model.drivers:
        vals = pd.to_numeric(ctx.df[d.column], errors="coerce").dropna()
        is_flag = set(vals.unique()) <= {0, 1}
        out.append({"column": d.column, "label": d.label, "kind": d.kind, "role": d.role,
                    "type": "share" if is_flag else "pct",
                    "current_share_pct": float(vals.mean() * 100) if is_flag else None})
    return out


def baseline_rows(ctx: ModelContext, model: ResponseModel, window: Optional[int] = DEFAULT_WINDOW) -> tuple[pd.DataFrame, str, int]:
    data = ctx.df.loc[model.used_rows]
    if ctx.date_col:
        dates = np.sort(data[ctx.date_col].unique())
        if window and window < len(dates):
            data = data[data[ctx.date_col] >= dates[-window]]
            return data, f"the last {window} periods ({pd.Timestamp(dates[-window]).date()} to {pd.Timestamp(dates[-1]).date()})", window
        return data, f"all {len(dates)} periods", len(dates)
    return data, f"all {len(data)} records", len(data)


def apply_changes(base: pd.DataFrame, changes: dict) -> pd.DataFrame:
    out = base.copy()
    for col, ch in (changes or {}).items():
        if col not in out.columns:
            continue
        v = float(ch.get("value", 0.0))
        if ch.get("type") == "share":
            out[col] = np.clip(v, 0, 100) / 100.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce") * (1 + v / 100.0)
    return out


def _aggregate(ctx: ModelContext, frame: pd.DataFrame, values: np.ndarray) -> float:
    if ctx.date_col:
        return float(pd.Series(values, index=frame.index).groupby(frame[ctx.date_col]).sum().mean())
    return float(np.mean(values))


def simulate(ctx: ModelContext, model: ResponseModel, scenarios: list, window: Optional[int] = DEFAULT_WINDOW,
             mmm: Optional[dict] = None) -> SimulationResult:
    res = SimulationResult()
    if model is None or model.fit is None:
        res.message = "No validated response model is available for this target, so scenarios cannot be estimated."
        return res
    base, wtxt, n = baseline_rows(ctx, model, window)
    res.per = "period" if ctx.date_col else "record"
    res.window_text = wtxt
    y = pd.to_numeric(base[ctx.target], errors="coerce").to_numpy(float)
    res.actual_baseline = _aggregate(ctx, base, y)
    pred0 = model.predict(base)
    res.model_baseline = _aggregate(ctx, base, pred0)
    h = model.holdout or {}
    if ctx.date_col and np.isfinite(h.get("period_rmse", np.nan)):
        half = Z90 * h["period_rmse"]
        res.range_basis = ("± 1.645 × holdout RMSE of per-period totals "
                           f"({h['period_rmse']:,.2f}); an approximate 90% range, not a guaranteed interval")
    elif np.isfinite(h.get("rmse", np.nan)):
        half = Z90 * h["rmse"]
        res.range_basis = f"± 1.645 × holdout RMSE ({h['rmse']:,.2f}); an approximate 90% range"
    else:
        half = None
        res.range_basis = "no holdout error is available, so no range is shown"
    res.reliability = model.quality
    res.reliability_text = {
        "good": "Good: the model explains the held-out periods well.",
        "moderate": "Moderate: usable for direction and rough size, with wide uncertainty.",
        "weak": "Low reliability: the model explains the held-out periods poorly. Do not use these estimates as forecasts.",
    }.get(model.quality, "Unvalidated model.")
    hist = ctx.df.loc[model.used_rows]
    for sc in scenarios:
        mod = apply_changes(base, sc.changes)
        p = model.predict(mod)
        est = _aggregate(ctx, mod, p)
        change = est - res.model_baseline
        extra, text = [], []
        for col, ch in sc.changes.items():
            if col not in base.columns:
                continue
            v = float(ch.get("value", 0))
            if ch.get("type") == "share":
                text.append(f"{col}: on for {v:.0f}% of records")
                continue
            text.append(f"{col}: {v:+.0f}%")
            lo, hi = pd.to_numeric(hist[col], errors="coerce").min(), pd.to_numeric(hist[col], errors="coerce").max()
            new = pd.to_numeric(mod[col], errors="coerce")
            outside = float(((new > hi * (1 + 1e-9)) | (new < lo * (1 - 1e-9))).mean())
            if outside > 0:
                extra.append(f"{col}: {100 * outside:.0f}% of values beyond the observed range (extrapolation)")
            if abs(v) > 50:
                extra.append(f"{col}: a change of {v:+.0f}% is large relative to typical variation")
        res.outcomes.append(ScenarioOutcome(
            name=sc.name, estimate=est, change=change,
            change_pct=100 * change / res.model_baseline if res.model_baseline else None,
            low=est - half if half is not None else None, high=est + half if half is not None else None,
            extrapolation=extra, changes_text=text))
    res.status = "ok"
    res.assumptions = [
        f"Baseline window: {wtxt}. Changes are applied to every record in that window.",
        "Promotion flags are applied as the share of records with the promotion on (their expected value).",
        "Drivers that a scenario does not change stay at their recorded values.",
        "Estimates come from the driver response model (associations in past data); they are not causal "
        "guarantees or forecasts of future periods.",
        "Ranges use the model's holdout error and ignore uncertainty in the estimated effects, so the true "
        "uncertainty is wider.",
    ]
    if ctx.channel_kind == "exposure":
        res.assumptions.append("Exposure changes are in exposure units (impressions, readers, listeners...), not money; "
                               "the cost of buying more exposure is unknown in this dataset.")
    if model.quality == "weak":
        res.warnings.append(res.reliability_text)
    if mmm and mmm.get("status") == "ok":
        res.mmm_check = mmm_cross_check(ctx, mmm, scenarios)
    return res


def mmm_cross_check(ctx: ModelContext, mmm: dict, scenarios: list) -> list:
    """Spend mode: the platform's marketing-mix model estimate for scenarios that only change spend."""
    from src.prescriptive.engine import LOOKBACK, what_if_custom
    per = mmm["period_data"]
    model = mmm["mmm"]
    cur = {c: float(per[c].tail(LOOKBACK).mean()) for c in model.channels}
    out = []
    for sc in scenarios:
        if not sc.changes or any(c not in model.channels or v.get("type") != "pct" for c, v in sc.changes.items()):
            continue
        alloc = {c: cur[c] * (1 + float(sc.changes.get(c, {}).get("value", 0)) / 100.0) for c in model.channels}
        w = what_if_custom(per, model, alloc)
        out.append({"scenario": sc.name, "mmm_estimate": w["scenario_outcome"], "mmm_baseline": w["current_outcome"],
                    "mmm_change_pct": w["change_pct"]})
    return out


def preset_scenarios(model: ResponseModel, ctx: ModelContext) -> list:
    """A few starting scenarios built only from drivers that exist."""
    chans = [d.column for d in model.drivers if d.kind == "channel"]
    out = []
    if chans:
        word = "spend" if ctx.channel_kind == "spend" else "exposure"
        out.append(Scenario(f"All channels +10% {word}", {c: {"type": "pct", "value": 10} for c in chans}))
        out.append(Scenario(f"All channels -10% {word}", {c: {"type": "pct", "value": -10} for c in chans}))
    prices = [d.column for d in model.drivers if d.role == "price"]
    if prices:
        out.append(Scenario("Price +5%", {prices[0]: {"type": "pct", "value": 5}}))
    flags = [a for a in adjustable_drivers(model, ctx) if a["role"] == "promotion" and a["type"] == "share"]
    if flags:
        out.append(Scenario("No promotions", {f["column"]: {"type": "share", "value": 0} for f in flags}))
    return out


def budget_change(model: ResponseModel, pct: float) -> dict:
    """Spend mode (PART 04): change the total marketing budget by ``pct`` % keeping the current mix, i.e. every
    spend channel changes by the same percentage."""
    return {d.column: {"type": "pct", "value": float(pct)} for d in model.drivers if d.kind == "channel"}


def combine(budget: dict, channel: dict) -> dict:
    """Combine a total-budget change with per-channel changes multiplicatively: (1+b)(1+c) - 1."""
    out = dict(budget)
    for col, ch in channel.items():
        if col in out and ch.get("type") == "pct" and out[col].get("type") == "pct":
            b, c = out[col]["value"] / 100.0, ch["value"] / 100.0
            out[col] = {"type": "pct", "value": 100.0 * ((1 + b) * (1 + c) - 1)}
        else:
            out[col] = ch
    return out


def sensitivity(ctx: ModelContext, model: ResponseModel, window: Optional[int] = DEFAULT_WINDOW,
                step: float = 10.0) -> pd.DataFrame:
    """One-at-a-time sensitivity (PART 04): the model-estimated outcome per period when each driver alone moves
    down and up by ``step`` % (promotion flags: share of records on moves by ``step`` points). Model estimates
    relative to the model baseline; not a forecast."""
    if model is None or model.fit is None:
        return pd.DataFrame()
    base, _, _ = baseline_rows(ctx, model, window)
    b0 = _aggregate(ctx, base, model.predict(base))
    rows = []
    for a in adjustable_drivers(model, ctx):
        col = a["column"]
        if a["type"] == "share":
            cur = float(pd.to_numeric(base[col], errors="coerce").mean() * 100)
            lo_c = {col: {"type": "share", "value": max(0.0, cur - step)}}
            hi_c = {col: {"type": "share", "value": min(100.0, cur + step)}}
            how = f"share of records on {max(0.0, cur - step):.0f}% / {min(100.0, cur + step):.0f}% (now {cur:.0f}%)"
        else:
            lo_c = {col: {"type": "pct", "value": -step}}
            hi_c = {col: {"type": "pct", "value": step}}
            how = f"-{step:.0f}% / +{step:.0f}%"
        lo = _aggregate(ctx, base, model.predict(apply_changes(base, lo_c)))
        hi = _aggregate(ctx, base, model.predict(apply_changes(base, hi_c)))
        rows.append({"driver": a["label"], "column": col, "kind": a["kind"], "change_tested": how,
                     "estimate_low": lo, "estimate_high": hi, "change_low": lo - b0, "change_high": hi - b0,
                     "change_low_pct": 100 * (lo - b0) / b0 if b0 else np.nan,
                     "change_high_pct": 100 * (hi - b0) / b0 if b0 else np.nan,
                     "range": abs(hi - lo)})
    return pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)
