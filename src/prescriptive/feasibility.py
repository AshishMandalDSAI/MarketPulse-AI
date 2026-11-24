"""Budget feasibility diagnostics, remedies and solution-status classification (PART 04).

``diagnose`` checks ``sum(min) <= total <= sum(max)`` (and every channel's own
``min <= max``) and, when the request is infeasible, lists the four remedies with
the exact amounts involved:

  A. increase channel maximums,
  B. change the total budget,
  C. explicitly allow an unallocated reserve (only for "maximums too low"),
  D. add further eligible channels.

Nothing here changes the user's inputs: remedies are suggestions, and a remedy
that would change an input is applied only when the user clicks it.

``classify`` turns an optimiser result into one of five statuses:
Optimal, Feasible but not proven optimal, Infeasible, Failed, Not supported.
"Optimal" is used only when the solver itself reports convergence for the
returned allocation and every starting point that converged reached the same
objective value (a local optimum confirmed from several starts; global
optimality cannot be proven for a non-concave response curve and is not claimed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

TOL = 1e-6

STATUS_LABELS = {
    "optimal": "Optimal",
    "feasible_not_proven": "Feasible but not proven optimal",
    "infeasible": "Infeasible",
    "failed": "Failed",
    "not_supported": "Not supported",
}


@dataclass
class Remedy:
    code: str                 # A | B | C | D
    title: str
    detail: str
    applicable: bool = True
    action: Optional[dict] = None     # machine-readable suggestion the UI may apply on an explicit click


@dataclass
class FeasibilityReport:
    feasible: bool
    total: Optional[float]
    sum_min: float
    sum_max: float
    problems: list = field(default_factory=list)
    remedies: list = field(default_factory=list)
    side: str = ""            # "max" (total above sum of maximums) | "min" | "channel" | "total" | ""

    @property
    def gap(self) -> float:
        if self.side == "max":
            return float(self.total - self.sum_max)
        if self.side == "min":
            return float(self.sum_min - self.total)
        return 0.0


def _fin(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def diagnose(total: Optional[float], mins: dict, maxs: dict, channels: list, *, eligible_extra: tuple = (),
             allow_reserve: bool = False, unit: str = "") -> FeasibilityReport:
    """Check feasibility and list remedies. ``eligible_extra``: channels in the data that are not selected."""
    u = f" {unit}" if unit else ""
    smin = float(sum(float(mins.get(c, 0.0) or 0.0) for c in channels))
    smax = float(sum(float(maxs.get(c, 0.0) or 0.0) for c in channels))
    rep = FeasibilityReport(feasible=True, total=total, sum_min=smin, sum_max=smax)
    if not channels:
        rep.feasible, rep.side = False, "channel"
        rep.problems.append("No channels are selected.")
        rep.remedies.append(Remedy("D", "Add eligible channels",
                                   ("Eligible channels in the data: " + ", ".join(eligible_extra)) if eligible_extra
                                   else "No eligible channels were found in the data.", bool(eligible_extra),
                                   {"add_channels": list(eligible_extra)} if eligible_extra else None))
        return rep
    if not _fin(total) or float(total) <= 0:
        rep.feasible, rep.side = False, "total"
        rep.problems.append("The total budget must be a positive number.")
        return rep
    total = float(total)
    tol = TOL * max(1.0, total)
    bad_ch = [c for c in channels if float(maxs.get(c, 0.0) or 0.0) < float(mins.get(c, 0.0) or 0.0) - tol]
    if bad_ch:
        rep.feasible, rep.side = False, "channel"
        for c in bad_ch:
            rep.problems.append(f"'{c}': the maximum ({float(maxs[c]):,.2f}{u}) is below the minimum "
                                f"({float(mins[c]):,.2f}{u}).")
        rep.remedies.append(Remedy("A", "Increase those channel maximums",
                                   "Raise each listed maximum to at least its minimum: " + "; ".join(
                                       f"{c} to {float(mins[c]):,.2f}{u}" for c in bad_ch), True,
                                   {"set_max": {c: float(mins[c]) for c in bad_ch}}))
        return rep
    if smin - total > tol:
        gap = smin - total
        rep.feasible, rep.side = False, "min"
        rep.problems.append(f"The channel minimums add up to {smin:,.2f}{u}, which is {gap:,.2f}{u} more than the total "
                            f"budget {total:,.2f}{u}. No allocation can respect every minimum.")
        rep.remedies += [
            Remedy("A", "Lower channel minimums",
                   f"Lower the minimums by {gap:,.2f}{u} in total (for example in proportion to each minimum).", True,
                   {"set_min": {c: float(mins.get(c, 0.0)) * (total / smin) for c in channels}}),
            Remedy("B", "Increase the total budget", f"Raise the total budget to at least {smin:,.2f}{u}.", True,
                   {"set_total": smin}),
            Remedy("C", "Allow an unallocated reserve",
                   "Not applicable: a reserve leaves money unspent, but here the minimums already need more than "
                   "the total.", False),
            Remedy("D", "Remove or add channels",
                   "Adding channels cannot help when minimums are too high; remove a channel or lower its minimum.",
                   False),
        ]
        return rep
    if total - smax > tol:
        gap = total - smax
        if allow_reserve:
            rep.feasible, rep.side = True, "max"
            rep.problems.append(f"The channel maximums add up to {smax:,.2f}{u}, {gap:,.2f}{u} less than the total "
                                f"{total:,.2f}{u}. You allowed a reserve, so at least {gap:,.2f}{u} will stay unallocated.")
            return rep
        rep.feasible, rep.side = False, "max"
        rep.problems.append(f"The entered channel maximums cannot accommodate the full budget: they add up to only "
                            f"{smax:,.2f}{u}, which is {gap:,.2f}{u} less than the total budget {total:,.2f}{u}.")
        rep.remedies += [
            Remedy("A", "Increase channel maximums",
                   f"Raise the maximums by {gap:,.2f}{u} in total (for example in proportion to each maximum).", True,
                   {"set_max": {c: float(maxs.get(c, 0.0)) * (total / smax) if smax > 0 else total / len(channels)
                                for c in channels}}),
            Remedy("B", "Reduce the total budget", f"Lower the total budget to at most {smax:,.2f}{u}.", True,
                   {"set_total": smax}),
            Remedy("C", "Allow an unallocated reserve",
                   f"Keep your limits and total, and explicitly allow up to {gap:,.2f}{u} (or more) to stay unallocated.",
                   True, {"allow_reserve": True}),
            Remedy("D", "Add eligible channels",
                   ("Add one of the channels you have not selected: " + ", ".join(eligible_extra)) if eligible_extra
                   else "No further eligible channels exist in this dataset.", bool(eligible_extra),
                   {"add_channels": list(eligible_extra)} if eligible_extra else None),
        ]
        return rep
    return rep


def classify(status: str, solver: Optional[dict]) -> tuple[str, str]:
    """(status key, plain explanation) for an optimiser result."""
    s = solver or {}
    if status == "infeasible":
        return "infeasible", "The limits cannot all be met; no allocation is shown."
    if status == "failed":
        return "failed", "The solver did not produce a valid allocation; nothing is shown."
    if status in ("unavailable", "not_supported"):
        return "not_supported", "Optimisation is not supported for these inputs."
    if status != "ok":
        return "failed", f"Unexpected result status '{status}'."
    if s.get("kept_start"):
        return "feasible_not_proven", ("The solver did not improve on a feasible starting allocation, so the "
                                       "allocation shown satisfies every limit but is not proven optimal.")
    if not s.get("converged"):
        return "feasible_not_proven", ("The solver did not report convergence; the allocation satisfies every limit "
                                       "but is not proven optimal.")
    agree, conv = s.get("starts_agreeing"), s.get("starts_converged")
    if conv and agree is not None and agree < conv:
        return "feasible_not_proven", (f"Only {agree} of {conv} converged starting points reached the best value, so "
                                       "the response may have several local optima; the best one found is shown.")
    return "optimal", (f"The solver converged and {agree if agree is not None else 'all'} of "
                       f"{conv if conv is not None else 'the'} converged starting points reached the same best value "
                       "(a confirmed local optimum; for a non-concave response a global optimum cannot be proven).")


def utilization(alloc: dict, mins: dict, maxs: dict) -> list:
    """Per-channel constraint utilisation rows."""
    rows = []
    for c, x in alloc.items():
        lo, hi = float(mins.get(c, 0.0)), float(maxs.get(c, 0.0))
        span = hi - lo
        at_min = abs(x - lo) <= TOL * max(1.0, abs(lo))
        at_max = abs(x - hi) <= TOL * max(1.0, abs(hi))
        rows.append({"channel": c, "recommended": float(x), "min": lo, "max": hi,
                     "share_of_max_pct": 100 * x / hi if hi > 0 else None,
                     "position_in_range_pct": 100 * (x - lo) / span if span > 0 else None,
                     "binding": "max" if at_max else "min" if at_min else ""})
    return rows
