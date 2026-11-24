"""Default per-channel limits for the budget / resource optimizers (PART 02 fix).

The previous screen gave every channel a maximum of 55% of the total budget. With
one channel the maximums then add up to 55% of the budget, so the problem was
infeasible before the user touched anything ("The channel maximums add up to
only ..., less than the total budget ...").

``default_limits`` generates limits that are always feasible for the budget and
the number of channels:

* minimum per channel: ``min(5%, 50% / n)`` of the total (so the minimums add up
  to at most half the budget), never more than half of the channel's current
  share, and 0 for a single channel;
* maximum per channel: 100% of the total for one or two channels, otherwise
  ``max(55%, 200% / n)``, raised to 125% of the channel's current share when
  that is higher (so a default never forces an existing channel to be cut) and
  capped at 100%.

Hence ``sum(min) <= total <= sum(max)`` holds by construction.

``reconcile_limits`` keeps automatically generated values in step with the total
budget while leaving any value the user typed untouched. It works on any
mutable mapping (e.g. ``st.session_state``) and is pure Python, so it is unit
tested without Streamlit.

Advanced feasibility handling (reserve budget, constraint diagnostics,
automatic repair suggestions) is PART 04 scope.
"""

from __future__ import annotations

import math
from typing import MutableMapping, Optional

AUTO_REGISTRY_SUFFIX = "__auto_limits"
_EPS = 1e-9


def _finite_pos(v) -> bool:
    try:
        return math.isfinite(float(v)) and float(v) > 0
    except (TypeError, ValueError):
        return False


def default_limits(total: float, channels: list, current: Optional[dict] = None) -> tuple[dict, dict, str]:
    """Feasible default (mins, maxs, rule text) for ``channels`` sharing ``total``."""
    n = len(channels)
    if n == 0:
        return {}, {}, "No channels."
    if not _finite_pos(total):
        return {c: 0.0 for c in channels}, {c: 0.0 for c in channels}, \
            "Enter a positive total to generate default limits."
    total = float(total)
    min_share = 0.0 if n == 1 else min(0.05, 0.5 / n)
    max_share = 1.0 if n <= 2 else min(1.0, max(0.55, 2.0 / n))
    cur = {c: max(0.0, float((current or {}).get(c, 0.0) or 0.0)) for c in channels}
    cur_tot = sum(cur.values())
    shares = {c: (cur[c] / cur_tot if cur_tot > 0 else None) for c in channels}
    mins, maxs = {}, {}
    for c in channels:
        lo, hi = min_share, max_share
        if shares[c] is not None:
            lo = min(lo, 0.5 * shares[c])
            hi = min(1.0, max(hi, 1.25 * shares[c]))
        mins[c] = math.floor(total * lo * 100) / 100
        maxs[c] = math.ceil(total * hi * 100) / 100
    # guarantee feasibility after rounding
    if sum(maxs.values()) < total:
        maxs[channels[0]] = round(maxs[channels[0]] + (total - sum(maxs.values())) + 0.01, 2)
    if sum(mins.values()) > total:
        scale = total / sum(mins.values())
        mins = {c: math.floor(v * scale * 100) / 100 for c, v in mins.items()}
    rule = (f"Defaults for {n} channel(s): minimum {100 * min_share:.1f}% and maximum {100 * max_share:.0f}% of the total "
            "per channel" + (", widened to 125% of a channel's current share where that is higher" if cur_tot > 0 and n > 2 else "")
            + ". They are recalculated when the total changes, unless you edit them.")
    return mins, maxs, rule


def reconcile_limits(state: MutableMapping, key_pairs: dict, mins: dict, maxs: dict, registry_key: str) -> dict:
    """Write default limits into ``state`` for every widget the user has not edited.

    ``key_pairs``: channel -> (min_widget_key, max_widget_key). ``registry_key`` stores the last
    automatically written value of each widget key. Returns channel -> {"min": "auto"|"manual",
    "max": "auto"|"manual"}. Must run before the widgets are created in a Streamlit run.
    """
    reg = dict(state.get(registry_key, {}) or {})
    status: dict = {}
    for ch, (kmin, kmax) in key_pairs.items():
        status[ch] = {}
        for side, key, val in (("min", kmin, mins.get(ch, 0.0)), ("max", kmax, maxs.get(ch, 0.0))):
            val = float(val)
            if key not in state:
                state[key] = val
                reg[key] = val
                status[ch][side] = "auto"
            elif key in reg and abs(float(state[key]) - float(reg[key])) <= _EPS * max(1.0, abs(float(reg[key]))):
                state[key] = val                     # untouched default -> refresh
                reg[key] = val
                status[ch][side] = "auto"
            else:
                reg.pop(key, None)                   # user value -> never overwritten
                status[ch][side] = "manual"
    state[registry_key] = reg
    return status


def reset_limits(state: MutableMapping, key_pairs: dict, mins: dict, maxs: dict, registry_key: str) -> None:
    """Explicit user action: replace every limit with the current defaults."""
    reg = {}
    for ch, (kmin, kmax) in key_pairs.items():
        state[kmin] = float(mins.get(ch, 0.0))
        state[kmax] = float(maxs.get(ch, 0.0))
        reg[kmin], reg[kmax] = state[kmin], state[kmax]
    state[registry_key] = reg


def feasibility(total: Optional[float], mins: dict, maxs: dict, channels: list) -> dict:
    """Plain summary of the core check ``sum(min) <= total <= sum(max)`` with explanations."""
    smin = float(sum(float(mins.get(c, 0.0)) for c in channels))
    smax = float(sum(float(maxs.get(c, 0.0)) for c in channels))
    msgs = []
    ok = True
    if total is None or not _finite_pos(total):
        return {"sum_min": smin, "sum_max": smax, "total": total, "feasible": False,
                "messages": ["Enter a positive total budget."]}
    tol = 1e-6 * max(1.0, float(total))
    if smin - total > tol:
        ok = False
        msgs.append(f"The minimums add up to {smin:,.2f}, which is more than the total {total:,.2f}. Lower some "
                    f"minimums by {smin - total:,.2f} in total, or raise the total.")
    if total - smax > tol:
        ok = False
        msgs.append(f"The maximums add up to only {smax:,.2f}, less than the total {total:,.2f}. Raise some "
                    f"maximums by {total - smax:,.2f} in total, or lower the total.")
    for c in channels:
        if float(maxs.get(c, 0.0)) < float(mins.get(c, 0.0)):
            ok = False
            msgs.append(f"'{c}': the maximum is below the minimum.")
    if ok:
        msgs.append(f"Feasible: minimums {smin:,.2f} ≤ total {float(total):,.2f} ≤ maximums {smax:,.2f}.")
    return {"sum_min": smin, "sum_max": smax, "total": float(total), "feasible": ok, "messages": msgs}
