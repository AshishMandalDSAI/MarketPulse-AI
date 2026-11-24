"""Budget / channel allocation optimisation with per-channel constraints, scenarios, risk and explanations.

All outcomes are model estimates from the fitted marketing-mix model for ONE representative period; nothing is
invented and nothing is a guarantee. Business logic only (no Streamlit)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.common.headers import neutralize_headers
from src.marketing.contracts import MMMResult
from src.marketing.mmm_engine import fit_mmm
from src.marketing.optimizer import _get_recent_carry_in, _predict_for_allocation
from src.marketing.roi import compute_roi_naive
from src.marketing.simulator import simulate_custom_allocation, simulate_total_budget

LOOKBACK = 8
TOL = 1e-6
MARGINAL_STEP = 0.01          # 1% of the total budget, for finite-difference marginal returns
BUDGET_STEPS = (0.8, 0.9, 1.0, 1.1, 1.2)
DISCLAIMER = ("These are model-estimated outcomes for one representative period, based on historical patterns in this "
              "dataset. They are not causal guarantees; validate with a controlled pilot before a full reallocation.")


@dataclass
class Roles:
    channels: list = field(default_factory=list)
    target: Optional[str] = None
    time_col: Optional[str] = None
    controls: list = field(default_factory=list)
    source: str = ""


@dataclass
class PrescriptiveResult:
    status: str = "unavailable"            # ok | unavailable | infeasible | failed
    message: str = ""
    issues: list = field(default_factory=list)          # constraint problems (blocking)
    notes: list = field(default_factory=list)
    objective: str = "outcome"
    target: str = ""
    channels: list = field(default_factory=list)
    total_budget: float = 0.0              # total spend of the RECOMMENDED plan (fixed mode: the requested total)
    bounds: dict = field(default_factory=dict)           # channel -> (min, max)
    current: dict = field(default_factory=dict)
    recommended: dict = field(default_factory=dict)
    current_outcome: float = 0.0
    recommended_outcome: float = 0.0
    outcome_change: float = 0.0
    outcome_change_pct: Optional[float] = None
    current_feasible: bool = True
    table: Any = None                  # per-channel comparison + explanation
    risk: dict = field(default_factory=dict)
    budget_response: Any = None
    scenarios: Any = None
    model: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    solver: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER
    # PART 04 (additive)
    requested_total: Optional[float] = None    # the total the user entered (fixed-total objective)
    budget_used: float = 0.0                   # sum of the recommended allocation
    unallocated: float = 0.0                   # requested total - budget used (only with an explicit reserve)
    allow_reserve: bool = False
    monetary: bool = True                      # False when the "channels" are not money (no ROAS-style figures)
    solution_status: str = ""                  # optimal | feasible_not_proven | infeasible | failed | not_supported
    solution_status_text: str = ""
    utilization: Any = None                    # DataFrame: per-channel constraint utilisation
    feasibility: Any = None                    # feasibility.FeasibilityReport


# ------------------------------------------------------------------ roles
def roles_from_report(path: Path) -> Roles:
    """Roles the platform already detected for this dataset (phase6_marketing.json). Missing/unreadable -> empty."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return Roles(source="none")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return Roles(source="unreadable")
    rep = data.get("marketing_report") or {}
    return Roles(channels=list(rep.get("channel_fields") or []), target=rep.get("target_field"),
                 time_col=rep.get("time_field"), controls=list(rep.get("control_fields") or []),
                 source="saved platform mapping")


def validate_roles(df: pd.DataFrame, channels: list, target: Optional[str], time_col: Optional[str] = None,
                   controls: Optional[list] = None) -> list:
    """Blocking problems with the chosen columns (empty list = usable)."""
    out = []
    if df is None or len(df) == 0:
        return ["The active dataset is empty."]
    if not target or target not in df.columns:
        out.append("Choose a numeric outcome column to improve.")
    elif not pd.api.types.is_numeric_dtype(df[target]) or pd.api.types.is_bool_dtype(df[target]):
        out.append(f"Outcome '{target}' is not numeric.")
    elif np.isinf(pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).any():
        out.append(f"Outcome '{target}' contains infinite values; clean or remove them first.")
    elif pd.to_numeric(df[target], errors="coerce").nunique() < 2:
        out.append(f"Outcome '{target}' does not vary.")
    if not channels:
        out.append("Choose at least one controllable spend / resource column.")
    for c in channels:
        if c not in df.columns:
            out.append(f"Column '{c}' is not in the dataset.")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            out.append(f"'{c}' is not numeric.")
        elif s.nunique(dropna=True) < 2:
            out.append(f"'{c}' does not vary, so its effect cannot be learned.")
        elif np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).any():
            out.append(f"'{c}' contains infinite values; clean or remove them first.")
        elif (s.dropna() < 0).any():
            out.append(f"'{c}' contains negative values; it cannot be treated as spend.")
        if c == target:
            out.append(f"'{c}' cannot be both a channel and the outcome.")
    if len(set(channels)) != len(channels):
        out.append("A channel was selected more than once.")
    return out


def fit_response_model(df: pd.DataFrame, channels: list, target: str, time_col: Optional[str] = None,
                       controls: Optional[list] = None) -> MMMResult:
    """Existing MMM engine, unchanged. Returns the MMMResult (``_model is None`` when it could not be fitted)."""
    controls = [c for c in (controls or []) if c in df.columns and c not in channels and c != target]
    return fit_mmm(df=df, channels=list(channels), target_col=target, control_cols=controls, time_col=time_col)


# ------------------------------------------------------------------ constraints
def validate_constraints(channels: list, total: Optional[float], mins: dict, maxs: dict,
                         total_range: Optional[tuple] = None) -> list:
    """Return a list of blocking problems; an empty list means the constraints are feasible."""
    issues = []

    def bad(v):
        return v is None or not isinstance(v, (int, float, np.floating, np.integer)) or isinstance(v, bool) or not math.isfinite(float(v))

    if total_range is None:
        if bad(total) or total <= 0:
            issues.append("Total budget must be a positive number.")
    else:
        lo, hi = total_range
        if bad(lo) or bad(hi) or lo < 0 or hi <= 0 or lo > hi:
            issues.append("The total budget range must be non-negative with minimum <= maximum.")
    lo_sum = hi_sum = 0.0
    for c in channels:
        a, b = mins.get(c, 0.0), maxs.get(c, None)
        if bad(a) or a < 0:
            issues.append(f"Minimum for '{c}' must be a non-negative number.")
            continue
        if bad(b):
            issues.append(f"Maximum for '{c}' must be a number.")
            continue
        if b < a:
            issues.append(f"'{c}': maximum ({b:,.2f}) is below minimum ({a:,.2f}).")
        lo_sum += a
        hi_sum += b
    if not issues:
        if total_range is None:
            if lo_sum - total > TOL * max(1.0, total):
                issues.append(f"The channel minimums add up to {lo_sum:,.2f}, more than the total budget {total:,.2f}.")
            if total - hi_sum > TOL * max(1.0, total):
                issues.append(f"The channel maximums add up to only {hi_sum:,.2f}, less than the total budget {total:,.2f}.")
        else:
            if lo_sum > total_range[1] + TOL * max(1.0, total_range[1]):
                issues.append(f"The channel minimums ({lo_sum:,.2f}) exceed the maximum total budget {total_range[1]:,.2f}.")
            if hi_sum < total_range[0] - TOL * max(1.0, total_range[0]):
                issues.append(f"The channel maximums ({hi_sum:,.2f}) are below the minimum total budget {total_range[0]:,.2f}.")
    return issues


def check_solution(alloc: dict, mins: dict, maxs: dict, total: Optional[float], total_range: Optional[tuple] = None) -> list:
    """Independent post-solve check used by the engine and by tests."""
    out = []
    s = sum(alloc.values())
    for c, v in alloc.items():
        if v < mins.get(c, 0.0) - 1e-8 * max(1.0, abs(mins.get(c, 0.0))) or v > maxs[c] + 1e-8 * max(1.0, abs(maxs[c])):
            out.append(f"{c} outside its bounds")
    if total_range is None:
        if abs(s - total) > 1e-6 * max(1.0, total):
            out.append("allocation does not sum to the total budget")
    elif s < total_range[0] - 1e-6 * max(1.0, total_range[0]) or s > total_range[1] + 1e-6 * max(1.0, total_range[1]):
        out.append("total outside the allowed budget range")
    return out


# ------------------------------------------------------------------ model helpers
class _Ctx:
    """Everything the optimiser's own objective needs (computed exactly as ``optimize_budget`` does)."""

    def __init__(self, df: pd.DataFrame, mmm: MMMResult, lookback: int = LOOKBACK):
        self.mmm, self.channels = mmm, list(mmm.channels)
        recent = df.tail(lookback)
        self.current = {c: float(pd.to_numeric(recent[c], errors="coerce").fillna(0).mean()) if c in recent.columns else 0.0
                        for c in self.channels}
        self.carry = _get_recent_carry_in(df, mmm, lookback)
        self.controls = {c: float(pd.to_numeric(recent[c], errors="coerce").fillna(0).mean())
                         for c in mmm.control_cols if c in recent.columns}
        self.hist = {c: (float(pd.to_numeric(df[c], errors="coerce").min()), float(pd.to_numeric(df[c], errors="coerce").max()))
                     for c in self.channels if c in df.columns}

    def predict(self, vec) -> float:
        return _predict_for_allocation(np.asarray(vec, dtype=float), self.mmm, self.carry, self.controls)


def _clip_project(x0, lo, hi, total):
    """A feasible start: clip into bounds, then shift the remainder within slack (deterministic water-filling)."""
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    for _ in range(100):
        gap = total - x.sum()
        if abs(gap) <= 1e-9 * max(1.0, total):
            break
        room = (hi - x) if gap > 0 else (x - lo)
        idle = room > 1e-12
        if not idle.any():
            break
        x = np.clip(x + (gap * room / room[idle].sum() if idle.any() else 0) * (room > 1e-12), lo, hi)
    return x


def _solve(ctx: _Ctx, lo, hi, total, total_range, objective, margin, starts, log: Optional[list] = None):
    n = len(ctx.channels)
    if objective == "profit":
        f = lambda a: -(margin * ctx.predict(a) - float(np.sum(a)))
        cons = [{"type": "ineq", "fun": lambda a: float(np.sum(a)) - total_range[0]},
                {"type": "ineq", "fun": lambda a: total_range[1] - float(np.sum(a))}]
    elif total_range is not None:      # PART 04: fixed-total objective with an explicit reserve (spend <= total)
        f = lambda a: -ctx.predict(a)
        cons = [{"type": "ineq", "fun": lambda a: float(np.sum(a)) - total_range[0]},
                {"type": "ineq", "fun": lambda a: total_range[1] - float(np.sum(a))}]
    else:
        f = lambda a: -ctx.predict(a)
        cons = [{"type": "eq", "fun": lambda a: float(np.sum(a)) - total}]
    best = None
    runs = 0
    for x0 in starts:
        runs += 1
        try:
            r = minimize(f, x0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                         options={"maxiter": 300, "ftol": 1e-9})
        except (ValueError, FloatingPointError) as exc:
            continue
        x = np.clip(r.x, lo, hi)
        if total_range is None and abs(x.sum() - total) > 1e-10 * max(1.0, total):
            x = _clip_project(x, lo, hi, total)
        val = -f(x)
        if log is not None:
            log.append({"value": float(val) if np.isfinite(val) else None, "success": bool(r.success),
                        "message": str(r.message)})
        if np.isfinite(val) and (best is None or val > best[0] + 1e-12):
            best = (val, x, bool(r.success), str(r.message))
    # the feasible starts themselves are candidates (the optimiser must never return something worse than them)
    for x0 in starts:
        val = -f(x0)
        if best is None or val > best[0] + 1e-12:
            best = (val, np.asarray(x0, dtype=float), True, "kept the best feasible starting allocation")
    return best, runs


def optimize_allocation(df: pd.DataFrame, mmm: MMMResult, total_budget: Optional[float], mins: dict, maxs: dict,
                        objective: str = "outcome", margin: Optional[float] = None,
                        total_range: Optional[tuple] = None, lookback: int = LOOKBACK,
                        explain: bool = True, allow_reserve: bool = False, monetary: bool = True) -> PrescriptiveResult:
    """Maximise the model-predicted outcome (fixed total) or margin x outcome - spend (budget range).

    PART 04 (additive): ``allow_reserve`` turns the fixed total into a ceiling (an explicit, reported unallocated
    reserve); every result carries budget used, unallocated amount, constraint utilisation and a solution status
    (optimal / feasible but not proven optimal / infeasible / failed / not supported). ``monetary=False`` suppresses
    ROAS-style figures when the "channels" are not money."""
    from src.prescriptive import feasibility as FZ
    log: list = []
    res = _optimize_allocation_core(df, mmm, total_budget, mins, maxs, objective, margin, total_range, lookback,
                                    explain, allow_reserve, monetary, log)
    res.allow_reserve = bool(allow_reserve and objective == "outcome")
    res.monetary = bool(monetary)
    if objective == "outcome":
        res.requested_total = float(total_budget) if total_budget is not None else res.total_budget
    if res.solver:
        conv = [e for e in log if e["success"] and e["value"] is not None]
        best_val = max((e["value"] for e in conv), default=None)
        res.solver["starts_converged"] = len(conv)
        res.solver["starts_agreeing"] = sum(1 for e in conv if best_val is not None and
                                            abs(e["value"] - best_val) <= 1e-6 * max(1.0, abs(best_val)))
        res.solver["kept_start"] = res.solver.get("message") == "kept the best feasible starting allocation"
        res.solver["start_log"] = log
    if res.status == "ok":
        res.budget_used = float(sum(res.recommended.values()))
        if objective == "outcome" and res.requested_total is not None:
            res.unallocated = max(0.0, float(res.requested_total) - res.budget_used)
        res.utilization = pd.DataFrame(FZ.utilization(res.recommended, {c: b[0] for c, b in res.bounds.items()},
                                                      {c: b[1] for c, b in res.bounds.items()}))
    res.solution_status, res.solution_status_text = FZ.classify(res.status, res.solver)
    if res.status == "unavailable" and mmm._model is not None:
        res.solution_status = "failed"
    return res


def _optimize_allocation_core(df, mmm, total_budget, mins, maxs, objective, margin, total_range, lookback, explain,
                              allow_reserve, monetary, log) -> PrescriptiveResult:
    res = PrescriptiveResult(objective=objective, target=mmm.target_column, channels=list(mmm.channels))
    res.monetary = bool(monetary)
    if mmm._model is None or mmm._scaler is None:
        res.message = "No fitted response model is available, so no recommendation can be made."
        return res
    ctx = _Ctx(df, mmm, lookback)
    ch = ctx.channels
    if objective not in ("outcome", "profit"):
        res.message = f"Unknown objective '{objective}'."
        return res
    if objective == "profit":
        if margin is None or not math.isfinite(float(margin)) or margin <= 0:
            res.status, res.issues = "infeasible", ["Profit optimisation needs a positive margin (profit per unit of outcome)."]
            res.message = res.issues[0]
            return res
        if total_range is None:
            res.status, res.issues = "infeasible", ["Profit optimisation needs a total budget range."]
            res.message = res.issues[0]
            return res
        total = None
    else:
        total = float(total_budget) if total_budget is not None else float(sum(ctx.current.values()))
        total_range = None
    mins = {c: float(mins.get(c, 0.0)) for c in ch}
    maxs = {c: float(maxs[c]) if c in maxs else float("nan") for c in ch}
    reserve_cap = None
    if objective == "outcome" and allow_reserve:
        # Explicit reserve: spend anything between the sum of minimums and min(total, sum of maximums).
        smin = sum(v for v in mins.values() if math.isfinite(v))
        smax = sum(v for v in maxs.values() if math.isfinite(v))
        if math.isfinite(total) and total > 0:
            reserve_cap = min(total, smax)
            total_range = (smin, reserve_cap)
    issues = validate_constraints(ch, total if reserve_cap is None else None, mins, maxs, total_range)
    if reserve_cap is not None and not issues and mins and sum(mins.values()) > total + TOL * max(1.0, total):
        issues.append(f"The channel minimums add up to {sum(mins.values()):,.2f}, more than the total budget {total:,.2f}.")
    if issues:
        res.status, res.issues, res.message = "infeasible", issues, "The constraints cannot be satisfied: " + issues[0]
        return res
    lo = np.array([mins[c] for c in ch]); hi = np.array([maxs[c] for c in ch])
    cur = np.array([ctx.current[c] for c in ch])
    res.total_budget = float(total if total is not None else cur.sum())
    res.bounds = {c: (mins[c], maxs[c]) for c in ch}
    res.current = dict(ctx.current)
    res.current_outcome = ctx.predict(cur)
    tot_lo = total_range[0] if total_range else total
    res.current_feasible = (not any(cur < lo - 1e-9) and not any(cur > hi + 1e-9) and
                            ((tot_lo - 1e-6 <= cur.sum() <= (total_range[1] if total_range else total) + 1e-6) if total_range
                             else abs(cur.sum() - total) <= 1e-6 * max(1.0, total)))
    tgt_total = total if (total is not None and reserve_cap is None) else (
        reserve_cap if reserve_cap is not None else float(np.clip(cur.sum(), total_range[0], total_range[1])))
    base = cur if cur.sum() > 0 else np.full(len(ch), tgt_total / len(ch))
    starts = [_clip_project(base / max(base.sum(), 1e-12) * tgt_total, lo, hi, tgt_total),
              _clip_project(np.full(len(ch), tgt_total / len(ch)), lo, hi, tgt_total),
              _clip_project((lo + hi) / 2, lo, hi, tgt_total)]
    best, runs = _solve(ctx, lo, hi, total, total_range, objective, margin, starts, log)
    if best is None:
        res.status, res.message = "failed", "The optimiser could not produce a valid allocation."
        return res
    _, x, ok, msg = best
    res.solver = {"method": "SLSQP (scipy), same objective as the platform's budget optimizer", "starts": runs,
                  "converged": ok, "message": msg}
    res.recommended = {c: float(x[i]) for i, c in enumerate(ch)}
    problems = check_solution(res.recommended, mins, maxs, total if reserve_cap is None else None, total_range)
    if problems:
        res.status, res.message = "failed", "The solution violated the constraints (" + "; ".join(problems) + "); it was discarded."
        res.recommended = {}
        return res
    if total is None or reserve_cap is not None:  # profit / reserve: report the recommended plan's total
        res.total_budget = float(sum(res.recommended.values()))
    res.recommended_outcome = ctx.predict(x)
    res.outcome_change = res.recommended_outcome - res.current_outcome
    res.outcome_change_pct = (res.outcome_change / res.current_outcome * 100.0) if res.current_outcome > 0 else None
    res.status = "ok"
    res.model = {"selected_model": mmm.selected_model, "n_train": mmm.n_train, "n_test": mmm.n_test,
                 "test_metrics": dict(mmm.test_metrics or {}), "train_metrics": dict(mmm.train_metrics or {}),
                 "channel_params": dict(mmm.channel_params or {}), "split": mmm.split_method}
    res.assumptions = [
        "Spend is evaluated for one representative period; each channel's carry-over from earlier periods is held at its "
        f"recent average (last {lookback} periods), as in the platform's budget optimizer.",
        "Current allocation = the average of the last %d periods in the data." % lookback,
        "The response model is fitted on past data; it measures association, so effects may differ in practice.",
        "Control variables are held at their recent averages.",
    ]
    if objective == "profit":
        res.assumptions.append(f"Profit = {margin:g} x predicted {mmm.target_column} - total spend (margin supplied by you).")
    if not res.current_feasible:
        res.notes.append("The current allocation does not satisfy your constraints, so the comparison is between an "
                         "infeasible current plan and the recommended feasible one.")
    if reserve_cap is not None:
        res.assumptions.append(f"You allowed an unallocated reserve: the plan may spend less than the total "
                               f"{total:,.2f}; the amount left unspent is reported, never hidden.")
    if explain:
        _explain(res, ctx, mmm, lo, hi, x)
        res.budget_response = _budget_response(df, mmm, res, mins, maxs, lookback) if objective == "outcome" else None
    return res


# ------------------------------------------------------------------ explanation, risk
def _marginals(ctx: _Ctx, x, step_amt):
    base = ctx.predict(x)
    out = []
    for i in range(len(x)):
        y = np.array(x, dtype=float); y[i] += step_amt
        out.append((ctx.predict(y) - base) / step_amt)
    return out


def _explain(res: PrescriptiveResult, ctx: _Ctx, mmm: MMMResult, lo, hi, x) -> None:
    ch = ctx.channels
    cur = np.array([ctx.current[c] for c in ch])
    step = max(res.total_budget * MARGINAL_STEP, 1e-9)
    m_cur, m_rec = _marginals(ctx, cur, step), _marginals(ctx, x, step)
    rows, outside, binding = [], [], 0
    for i, c in enumerate(ch):
        chg = x[i] - cur[i]
        pct = chg / cur[i] * 100 if cur[i] > 0 else None
        at_min, at_max = abs(x[i] - lo[i]) <= 1e-6 * max(1.0, lo[i]), abs(x[i] - hi[i]) <= 1e-6 * max(1.0, hi[i])
        binding += int(at_min or at_max)
        h = ctx.hist.get(c)
        out_rng = bool(h and (x[i] < h[0] - 1e-9 or x[i] > h[1] + 1e-9))
        if out_rng:
            outside.append(c)
        direction = "increase" if chg > 1e-9 * max(1.0, abs(cur[i])) else "decrease" if chg < -1e-9 * max(1.0, abs(cur[i])) else "keep"
        p = mmm.channel_params.get(c, {})
        why = []
        if direction == "keep":
            why.append("allocation is already at the best level the constraints allow")
        else:
            why.append(f"modelled return per extra unit of spend at the current level is {m_cur[i]:.3g} and at the "
                       f"recommended level {m_rec[i]:.3g}")
            if at_max:
                why.append("capped by your maximum for this channel")
            elif at_min:
                why.append("held at your minimum for this channel")
        if out_rng:
            why.append(f"recommended spend is outside the range seen in the data ({h[0]:,.2f} to {h[1]:,.2f}); treat as extrapolation")
        rows.append({"channel": c, "current": float(cur[i]), "recommended": float(x[i]), "change": float(chg),
                     "change_pct": pct, "min": float(lo[i]), "max": float(hi[i]), "action": direction,
                     "marginal_return_current": float(m_cur[i]), "marginal_return_recommended": float(m_rec[i]),
                     "observed_min": h[0] if h else None, "observed_max": h[1] if h else None,
                     "outside_observed_range": out_rng, "binding_constraint": "max" if at_max else "min" if at_min else "",
                     "model_decay": p.get("decay"), "model_half_saturation": p.get("k"),
                     "explanation": "; ".join(why).capitalize() + "."})
    res.table = pd.DataFrame(rows)
    tm = mmm.test_metrics or {}
    rmse, mape, r2 = tm.get("rmse"), tm.get("mape"), tm.get("r2")
    reasons = []
    level = "low"
    if outside:
        level = "high"; reasons.append("recommended spend is outside the historical range for: " + ", ".join(outside))
    if rmse is not None and abs(res.outcome_change) < rmse:
        level = "high"
        reasons.append("the estimated improvement is smaller than the model's typical test error, so it may be noise")
    if level == "low":
        if (r2 is not None and r2 < 0.5) or mmm.n_train < 30:
            level = "medium"; reasons.append("the response model is modest (test R² below 0.5 or fewer than 30 training periods)")
        if binding >= max(2, len(ch) // 2 + 1):
            level = "medium"; reasons.append("most channels sit at a constraint bound")
    if not reasons:
        reasons.append("recommended spend is within the observed range and the estimated gain exceeds typical model error")
    res.risk = {"level": level, "reasons": reasons, "model_test_rmse": rmse, "model_test_mape": mape, "model_test_r2": r2,
                "outcome_band_low": res.recommended_outcome - rmse if rmse is not None else None,
                "outcome_band_high": res.recommended_outcome + rmse if rmse is not None else None,
                "band_basis": "recommended outcome +/- the response model's test RMSE (an approximate error band, not a "
                              "statistical confidence interval)" if rmse is not None else "unavailable",
                "channels_outside_observed_range": outside, "channels_at_bound": int(binding)}
    # ROAS-style view from the platform's own ROI helper (history), alongside the model-based marginal return.
    # PART 04: only when the channels are money (never for exposure or other units).
    if not getattr(res, "monetary", True):
        return
    try:
        roi = compute_roi_naive(df=pd.DataFrame({c: [ctx.current[c]] for c in ch} | {mmm.target_column: [res.current_outcome]}),
                                channels=ch, target_col=mmm.target_column)
        res.table["historical_style_roas_current"] = [r.naive_roas for r in roi]
    except (KeyError, ValueError):
        res.table["historical_style_roas_current"] = None
    res.table["avg_return_per_unit_recommended"] = [(res.recommended_outcome / res.total_budget) if res.total_budget > 0 else None] * len(ch)


def _budget_response(df, mmm, res, mins, maxs, lookback):
    rows = []
    for f in BUDGET_STEPS:
        t = res.total_budget * f
        sub = optimize_allocation(df, mmm, t, mins, maxs, explain=False, lookback=lookback)
        rows.append({"budget_vs_current": f"{f:.0%}", "total_budget": t, "status": sub.status,
                     "predicted_outcome": sub.recommended_outcome if sub.status == "ok" else None,
                     "note": "" if sub.status == "ok" else (sub.issues[0] if sub.issues else sub.message)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what-if (existing simulator, reused)
def what_if_custom(df: pd.DataFrame, mmm: MMMResult, allocation: dict, lookback: int = LOOKBACK) -> dict:
    r = simulate_custom_allocation(df, mmm, {c: float(allocation.get(c, 0.0)) for c in mmm.channels}, lookback)
    s, b = r.scenarios[0], r.baseline_scenario
    return {"current_outcome": b.predicted_target, "scenario_outcome": s.predicted_target, "change": s.target_change,
            "change_pct": s.target_change_pct, "scenario_total": s.total_budget, "current_total": b.total_budget,
            "allocation": dict(s.allocation), "disclaimer": r.disclaimer}


def what_if_total(df: pd.DataFrame, mmm: MMMResult, total: float, keep_mix: bool = True, lookback: int = LOOKBACK) -> dict:
    r = simulate_total_budget(df, mmm, float(total), keep_mix, lookback)
    s, b = r.scenarios[0], r.baseline_scenario
    return {"current_outcome": b.predicted_target, "scenario_outcome": s.predicted_target, "change": s.target_change,
            "change_pct": s.target_change_pct, "scenario_total": s.total_budget, "allocation": dict(s.allocation)}


def scenario_table(res: PrescriptiveResult, extra: Optional[list] = None) -> pd.DataFrame:
    rows = [{"scenario": "Current", "total": sum(res.current.values()), "outcome": res.current_outcome,
             **{c: v for c, v in res.current.items()}},
            {"scenario": "Recommended", "total": sum(res.recommended.values()), "outcome": res.recommended_outcome,
             **res.recommended}]
    for e in (extra or []):
        rows.append(e)
    t = pd.DataFrame(rows)
    t["change_vs_current"] = t["outcome"] - res.current_outcome
    return t


# ------------------------------------------------------------------ report / export
def business_report(res: PrescriptiveResult) -> str:
    if res.status != "ok":
        raise ValueError("There is no completed recommendation to report.")
    pct = f" ({res.outcome_change_pct:+.1f}%)" if res.outcome_change_pct is not None else ""
    L = [f"# Allocation recommendation - {res.target}", "",
         f"**Objective:** {'maximise predicted ' + res.target + ' for a fixed budget' if res.objective == 'outcome' else 'maximise estimated profit over a budget range'}.",
         f"**Total budget (per period):** {res.total_budget:,.2f}.", "",
         f"Current plan: predicted {res.target} {res.current_outcome:,.2f}. Recommended plan: {res.recommended_outcome:,.2f}, "
         f"a model-estimated change of {res.outcome_change:+,.2f}{pct}.", "", "## Channel recommendations", ""]
    for r in res.table.itertuples():
        L.append(f"- **{r.channel}:** {r.current:,.2f} -> {r.recommended:,.2f} ({r.action}). {r.explanation}")
    rk = res.risk
    L += ["", f"## Risk: {rk['level']}", ""] + [f"- {x}" for x in rk["reasons"]]
    if rk.get("outcome_band_low") is not None:
        L.append(f"- Approximate band for the recommended outcome: {rk['outcome_band_low']:,.2f} to {rk['outcome_band_high']:,.2f} ({rk['band_basis']}).")
    L += ["", "## Assumptions", ""] + [f"- {a}" for a in res.assumptions]
    if res.notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in res.notes]
    L += ["", f"_{res.disclaimer}_"]
    return "\n".join(L)


def result_sheets(res: PrescriptiveResult, scenarios: Optional[pd.DataFrame] = None) -> dict:
    from src.common.export import ExportError
    if res is None or res.status != "ok":
        raise ExportError("There is no completed optimisation result to export.")
    kv = lambda d: pd.DataFrame({"item": list(d), "value": [str(v) for v in d.values()]})
    s = {"Summary": kv({"objective": res.objective, "outcome": res.target, "total_budget": res.total_budget,
                        "current_predicted_outcome": res.current_outcome, "recommended_predicted_outcome": res.recommended_outcome,
                        "change": res.outcome_change, "change_pct": res.outcome_change_pct, "risk_level": res.risk["level"],
                        "current_plan_feasible": res.current_feasible, "solver_converged": res.solver.get("converged")}),
         "Channel recommendations": res.table,
         "Constraints": pd.DataFrame([{"channel": c, "min": a, "max": b} for c, (a, b) in res.bounds.items()]),
         "Risk": pd.DataFrame({"item": ["level"] + [f"reason {i + 1}" for i in range(len(res.risk["reasons"]))] + ["band basis"],
                               "value": [res.risk["level"]] + res.risk["reasons"] + [res.risk["band_basis"]]}),
         "Model": kv({k: v for k, v in res.model.items() if k != "channel_params"}),
         "Assumptions": pd.DataFrame({"assumption": res.assumptions + res.notes + [res.disclaimer]})}
    if res.budget_response is not None:
        s["Budget response"] = res.budget_response
    if scenarios is not None and len(scenarios):
        s["Scenarios"] = scenarios
    return {k: neutralize_headers(v) for k, v in s.items()}
