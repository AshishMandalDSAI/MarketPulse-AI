"""Module 17 (Prescriptive Analytics) sections. Results are computed only on request, cached under the active dataset's
scope, and flagged when the settings change. Every number comes from the platform's own fitted response model."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import chart_labels as CL
from dashboard.components import session_paths
from dashboard.components.stats_ui import (apply_dark_layout, cache_get, cache_set, render_download_buttons,
                                           render_limitations, render_section, widget_key)
from src.common.dataset_summary import DatasetSummary
from src.common.export import ExportError, sheets_to_excel_bytes
from src.common.formatting import escape_text
from src.common.headers import neutralize_headers
from src.prescriptive import engine as E
from src.prescriptive import feasibility as FZ
from src.prescriptive import limits as LIM

MODULE_KEY = "prescriptive"


def _k(name: str) -> str:
    return widget_key(MODULE_KEY, f"{name}_{session_paths.get_dataset_id()}")


def _saved_roles() -> E.Roles:
    return E.roles_from_report(session_paths.get_processed_dir() / "phase6_marketing.json")


def _roles_section(df: pd.DataFrame, summary: DatasetSummary):
    num = [c for c in summary.numeric_columns if c in df.columns]
    saved = _saved_roles()
    if saved.source == "saved platform mapping" and saved.channels:
        st.caption("Suggested columns come from the mapping the platform already detected for this dataset. Confirm or change them.")
    elif saved.source == "unreadable":
        st.caption("The saved column mapping could not be read, so nothing is pre-selected.")
    d_t = saved.target if saved.target in num else None
    opts = [None] + num
    target = st.selectbox("Outcome to improve (numeric)", opts, index=opts.index(d_t) if d_t else 0, key=_k("target"),
                          format_func=lambda v: "(choose a column)" if v is None else v)
    pool = [c for c in num if c != target]
    d_ch = [c for c in saved.channels if c in pool]
    channels = st.multiselect("Controllable spend / resource columns (channels)", pool, default=d_ch, key=_k("channels"))
    dates = list(summary.datetime_columns)
    tc = [None] + dates
    d_tc = saved.time_col if saved.time_col in dates else None
    time_col = st.selectbox("Time column (orders the rows; optional)", tc, index=tc.index(d_tc), key=_k("time"),
                            format_func=lambda v: "(use current row order)" if v is None else v)
    ctrl_pool = [c for c in pool if c not in channels]
    controls = st.multiselect("Control columns held constant (optional)", ctrl_pool,
                              default=[c for c in saved.controls if c in ctrl_pool], key=_k("controls"))
    return target, channels, time_col, controls


def _model_for(df, channels, target, time_col, controls):
    sig = (tuple(channels), target, time_col, tuple(controls), len(df), tuple(df.columns))
    cached = cache_get(MODULE_KEY, "mmm")
    if cached and cached[0] == sig:
        return cached[1]
    with st.spinner("Fitting the response model..."):
        mmm = E.fit_response_model(df, channels, target, time_col, controls)
    cache_set(MODULE_KEY, "mmm", (sig, mmm))
    return mmm


def _apply_remedy(action: dict, keys: dict) -> None:
    """Explicit user click: apply ONE remedy to the inputs (never done automatically)."""
    reg = dict(st.session_state.get(keys["registry"], {}) or {})
    for c, v in (action.get("set_max") or {}).items():
        if c in keys["max"]:
            st.session_state[keys["max"][c]] = float(round(v, 2) + 0.01)
            reg.pop(keys["max"][c], None)              # now a user-chosen value
    for c, v in (action.get("set_min") or {}).items():
        if c in keys["min"]:
            st.session_state[keys["min"][c]] = float(max(0.0, round(v, 2) - 0.01))
            reg.pop(keys["min"][c], None)
    if "set_total" in action and keys.get("total"):
        st.session_state[keys["total"]] = float(round(action["set_total"], 2))
    if action.get("allow_reserve") and keys.get("reserve"):
        st.session_state[keys["reserve"]] = True
    if action.get("add_channels") and keys.get("channels"):
        st.session_state[keys["channels"]] = list(dict.fromkeys(list(st.session_state.get(keys["channels"], []))
                                                                + list(action["add_channels"][:1])))
    st.session_state[keys["registry"]] = reg


def render_feasibility(report, keys: dict, unit: str = "", allow_reserve_option: bool = True) -> None:
    """Feasibility status, plain explanation and the remedies A-D (each applied only when clicked)."""
    cols = st.columns(3)
    cols[0].metric("Sum of minimums" + (f" ({unit})" if unit else ""), f"{report.sum_min:,.2f}")
    cols[1].metric("Total budget" + (f" ({unit})" if unit else ""), "—" if report.total is None else f"{float(report.total):,.2f}")
    cols[2].metric("Sum of maximums" + (f" ({unit})" if unit else ""), f"{report.sum_max:,.2f}")
    if report.feasible:
        msg = (f"Feasible: minimums {report.sum_min:,.2f} ≤ total {float(report.total):,.2f} ≤ maximums "
               f"{report.sum_max:,.2f}." if report.side == "" else " ".join(report.problems))
        st.success("**Feasibility status: feasible.** " + escape_text(msg))
        return
    st.error("**Feasibility status: infeasible.** " + escape_text(" ".join(report.problems))
             + " Nothing has been changed; choose a remedy below or edit the limits yourself.")
    for r in report.remedies:
        if r.code == "C" and not allow_reserve_option:
            continue
        a, b = st.columns([4, 1])
        a.markdown(f"**{r.code}. {escape_text(r.title)}** — {escape_text(r.detail)}")
        if r.applicable and r.action:
            b.button(f"Apply {r.code}", key=f"{keys['registry']}_rem_{r.code}", on_click=_apply_remedy,
                     args=(r.action, keys))


def constraint_panel(df, channels, key_fn=None, unit: str = "", monetary: bool = True,
                     eligible_extra: tuple = (), channels_key: Optional[str] = None) -> dict:
    """Objective, total, reserve option, per-channel limits and the feasibility check (PART 02 defaults + PART 04
    diagnostics). Returns a dict with mode, total, margin, trange, mins, maxs, allow_reserve, feasible."""
    kf = key_fn or _k
    cur = {c: float(pd.to_numeric(df[c], errors="coerce").tail(E.LOOKBACK).fillna(0).mean()) for c in channels}
    tot_cur = sum(cur.values())
    objectives = ["outcome", "profit"] if monetary else ["outcome"]
    mode = st.radio("Objective", objectives, horizontal=True, key=kf("objective"),
                    format_func=lambda m: "Maximise the outcome for a fixed total budget" if m == "outcome"
                    else "Maximise estimated profit (margin x outcome - spend) over a budget range")
    if not monetary:
        st.caption("Profit optimisation is not offered: the selected columns are not money, so 'spend' cannot be "
                   "subtracted from a monetary outcome.")
    margin, total, trange, reserve = None, None, None, False
    sfx = f" ({unit})" if unit else ""
    if mode == "outcome":
        total = float(st.number_input("Total budget per period" + sfx, min_value=0.0, value=float(round(tot_cur, 2)), key=kf("total")))
        reserve = st.checkbox("Allow an unallocated reserve (the plan may spend less than the total)", key=kf("reserve"),
                              help="Off by default: the whole total is allocated. On: the total becomes a ceiling and "
                                   "any unspent amount is reported as unallocated.")
    else:
        margin = float(st.number_input("Profit per unit of outcome (margin)", min_value=0.0, value=0.0, key=kf("margin"),
                                       help="Needed for profit optimisation. Enter 0.3 for a 30% margin on a revenue outcome."))
        c1, c2 = st.columns(2)
        trange = (float(c1.number_input("Minimum total budget" + sfx, min_value=0.0, value=float(round(tot_cur * .8, 2)), key=kf("tmin"))),
                  float(c2.number_input("Maximum total budget" + sfx, min_value=0.0, value=float(round(tot_cur * 1.2, 2)), key=kf("tmax"))))
    st.markdown("**Per-channel limits** (absolute amounts per period)")
    base_total = total if total else (trange[1] if trange else tot_cur)
    d_min, d_max, rule = LIM.default_limits(base_total, channels, cur)
    pairs = {c: (kf(f"min_{c}"), kf(f"max_{c}")) for c in channels}
    registry = kf(LIM.AUTO_REGISTRY_SUFFIX)
    status = LIM.reconcile_limits(st.session_state, pairs, d_min, d_max, registry)
    st.caption(escape_text(rule))
    mins, maxs = {}, {}
    for c in channels:
        a, b = st.columns(2)
        tag_min = "" if status[c]["min"] == "auto" else " · set by you"
        tag_max = "" if status[c]["max"] == "auto" else " · set by you"
        mins[c] = float(a.number_input(f"{c} - minimum{tag_min}", min_value=0.0, key=pairs[c][0]))
        maxs[c] = float(b.number_input(f"{c} - maximum{tag_max}", min_value=0.0, key=pairs[c][1]))
    if any(v != "auto" for s in status.values() for v in s.values()):
        st.button("Reset all limits to the defaults", key=kf("reset_limits"),
                  on_click=LIM.reset_limits, args=(st.session_state, pairs, d_min, d_max, registry))
    feasible = True
    if mode == "outcome":
        rep = FZ.diagnose(total, mins, maxs, channels, eligible_extra=tuple(eligible_extra), allow_reserve=reserve,
                          unit=unit)
        keys = {"registry": registry, "max": {c: pairs[c][1] for c in channels},
                "min": {c: pairs[c][0] for c in channels}, "total": kf("total"), "reserve": kf("reserve"),
                "channels": channels_key}
        render_feasibility(rep, keys, unit)
        feasible = rep.feasible
    return {"mode": mode, "total": total, "margin": margin, "trange": trange, "mins": mins, "maxs": maxs,
            "allow_reserve": bool(reserve), "feasible": feasible}


def _constraint_inputs(df, channels, key_fn=None, unit: str = ""):
    """Backward-compatible wrapper (PART 02 signature)."""
    cp = constraint_panel(df, channels, key_fn, unit)
    return cp["mode"], cp["total"], cp["margin"], cp["trange"], cp["mins"], cp["maxs"]


STATUS_KIND = {"optimal": "good", "feasible_not_proven": "warn", "infeasible": "bad", "failed": "bad",
               "not_supported": "warn"}


def _solver_report(res: E.PrescriptiveResult, unit: str = "") -> None:
    """Solution status, solver status, objective, budget used / unallocated (PART 02 + PART 04)."""
    from dashboard.components.ui import callout
    obj = ("Maximise the model-predicted " + str(res.target) + " for a fixed total budget"
           + (" (an unallocated reserve is allowed)" if res.allow_reserve else "")
           if res.objective == "outcome" else
           "Maximise estimated profit (margin x predicted " + str(res.target) + " - spend) within a budget range")
    s = res.solver or {}
    label = FZ.STATUS_LABELS.get(res.solution_status, res.solution_status or "Not run")
    callout(f"**{label}.** " + (res.solution_status_text or ""), kind=STATUS_KIND.get(res.solution_status, "info"),
            label="Solution status")
    status = {"ok": "Solved", "infeasible": "Infeasible constraints", "failed": "Solver failed",
              "unavailable": "Not run"}.get(res.status, res.status)
    conv = s.get("converged")
    st.markdown(f"**Solver status:** {escape_text(status)}"
                + ("" if conv is None else f" · converged: {'yes' if conv else 'no'}")
                + (f" · {escape_text(str(s.get('message')))}" if s.get("message") else "")
                + (f" · {s.get('starts_agreeing')} of {s.get('starts_converged')} converged starts agree"
                   if s.get("starts_converged") else "")
                + f"  \n**Objective:** {escape_text(obj)}"
                + (f"  \n**Method:** {escape_text(str(s.get('method')))}" if s.get("method") else ""))
    if res.status == "ok":
        u = f" ({unit})" if unit else ""
        c = st.columns(4)
        c[0].metric("Total budget" + u, f"{(res.requested_total if res.requested_total is not None else res.total_budget):,.2f}")
        c[1].metric("Budget used" + u, f"{res.budget_used:,.2f}")
        c[2].metric("Unallocated" + u, f"{res.unallocated:,.2f}")
        c[3].metric("Feasibility", "Feasible")


def _results(res: E.PrescriptiveResult, df, mmm, unit: str = "") -> None:
    _solver_report(res, unit)
    if res.status != "ok":
        st.warning(f"**No recommendation:** {escape_text(res.message)}")
        for i in res.issues:
            st.caption("• " + escape_text(i))
        return
    render_section("Recommendation")
    c = st.columns(3)
    c[0].metric(f"Predicted {res.target} - current", f"{res.current_outcome:,.2f}")
    c[1].metric("Predicted - recommended", f"{res.recommended_outcome:,.2f}",
                None if res.outcome_change_pct is None else f"{res.outcome_change_pct:+.1f}%")
    c[2].metric("Risk", res.risk["level"].title())
    for n in res.notes:
        st.caption("⚠️ " + escape_text(n))
    if res.solver and not res.solver.get("converged"):
        st.caption("⚠️ The solver did not report full convergence; the best feasible allocation found is shown.")
    t = res.table
    fig = go.Figure([go.Bar(name="Current", x=t.channel, y=t.current), go.Bar(name="Recommended", x=t.channel, y=t.recommended)])
    fig.update_layout(barmode="group", legend_title_text="Allocation")
    unit = unit or (CL.currency_unit(t.channel.iloc[0]) if len(t) and res.monetary else "")
    st.plotly_chart(apply_dark_layout(fig, "Current vs recommended allocation", x_title="Channel / resource",
                                      y_title="Allocation per period" + (f" ({unit})" if unit else "")),
                    use_container_width=True)
    if res.utilization is not None and len(res.utilization):
        render_section("Constraint Utilisation")
        ut = res.utilization
        fig = go.Figure([go.Bar(name="Minimum", x=ut.channel, y=ut["min"], marker_color="#7a8599"),
                         go.Bar(name="Recommended", x=ut.channel, y=ut.recommended, marker_color="#3B82F6"),
                         go.Bar(name="Maximum", x=ut.channel, y=ut["max"], marker_color="#3ddc97")])
        fig.update_layout(barmode="group", legend_title_text="Value")
        st.plotly_chart(apply_dark_layout(fig, "Recommended allocation within each channel's limits",
                                          x_title="Channel / resource",
                                          y_title="Amount per period" + (f" ({unit})" if unit else "")),
                        use_container_width=True)
        st.dataframe(ut.round(2), hide_index=True)
        b = int((ut.binding != "").sum())
        st.caption(f"{b} of {len(ut)} channel(s) sit at a limit (binding). 'Position in range' is where the "
                   "recommendation sits between the channel's minimum (0%) and maximum (100%).")
    st.dataframe(t[["channel", "current", "recommended", "change", "change_pct", "min", "max", "action",
                    "marginal_return_current", "marginal_return_recommended", "binding_constraint",
                    "outside_observed_range"]], hide_index=True)
    render_section("Why These Recommendations")
    for r in t.itertuples():
        st.markdown(f"- **{escape_text(str(r.channel))}** - {escape_text(r.action)}: {escape_text(r.explanation)}")
    render_section("Risk and Uncertainty")
    for x in res.risk["reasons"]:
        st.markdown(f"- {escape_text(x)}")
    rk = res.risk
    if rk.get("outcome_band_low") is not None:
        st.caption(f"Approximate band for the recommended outcome: {rk['outcome_band_low']:,.2f} to {rk['outcome_band_high']:,.2f}. "
                   + escape_text(rk["band_basis"]))
    st.caption(f"Response model: {escape_text(res.model['selected_model'])}; test R² {rk['model_test_r2'] if rk['model_test_r2'] is not None else 'n/a'}; "
               f"{res.model['n_train']} training and {res.model['n_test']} test periods.")
    if res.budget_response is not None:
        render_section("Budget Sensitivity")
        st.dataframe(res.budget_response, hide_index=True)


def _what_if(res, df, mmm):
    render_section("What-if Scenario")
    st.caption("Try your own allocation. This reuses the platform's existing what-if simulator.")
    alloc = {}
    cols = st.columns(min(4, len(res.channels)))
    for i, c in enumerate(res.channels):
        alloc[c] = float(cols[i % len(cols)].number_input(c, min_value=0.0, value=float(round(res.recommended[c], 2)), key=_k(f"wi_{c}")))
    w = E.what_if_custom(df, mmm, alloc)
    a = st.columns(3)
    a[0].metric("Scenario total", f"{w['scenario_total']:,.2f}")
    a[1].metric(f"Predicted {res.target}", f"{w['scenario_outcome']:,.2f}", f"{w['change_pct']:+.1f}% vs current")
    a[2].metric("Recommended", f"{res.recommended_outcome:,.2f}")
    if w["scenario_outcome"] > res.recommended_outcome + 1e-6 and abs(w["scenario_total"] - res.total_budget) <= 1e-6 * max(1, res.total_budget):
        st.caption("This scenario scores higher than the recommendation; check it against your constraints.")
    st.caption(escape_text(w["disclaimer"]))
    row = {"scenario": "Your scenario", "total": w["scenario_total"], "outcome": w["scenario_outcome"], **alloc}
    return E.scenario_table(res, [row])


def _currency_unit() -> str:
    from dashboard.components import currency as CUR
    return CUR.unit_label()


def _channel_units(df: pd.DataFrame, channels: list) -> tuple:
    """(unit label, monetary?, problem) for the selected channel columns, from the dataset's role mapping
    (PART 04): exposure units are never treated as a monetary budget, and different units are never mixed."""
    from dashboard.components import role_mapping_ui as RMU
    from src.data_engine import role_engine as RE
    try:
        m = RMU.get_mapping(df)
    except Exception:                    # mapping unavailable: fall back to the previous generic behaviour
        return "", True, ""
    units = {}
    for c in channels:
        r = m.columns.get(c)
        role = r.role if r else "measure"
        if role in RE.MONETARY_ROLES:
            units[c] = "money"
        elif r is not None and r.group == "exposure":
            units[c] = r.unit
        else:
            units[c] = "money"           # unlabelled numeric resource: unchanged generic behaviour
    kinds = set(units.values())
    if len(kinds) > 1:
        return "", False, ("The selected columns use different units (" + ", ".join(
            f"{c}: {u}" for c, u in units.items()) + "). A budget can only be shared between columns measured in the "
            "same unit, so this combination cannot be optimised.")
    kind = next(iter(kinds)) if kinds else "money"
    return ("", True, "") if kind == "money" else (kind, False, "")


def _is_eligible_extra(df: pd.DataFrame, col: str, channels: list) -> bool:
    """A not-selected column that could be added as a channel (same kind as the selected ones)."""
    _, monetary, problem = _channel_units(df, list(channels) + [col])
    s = pd.to_numeric(df[col], errors="coerce")
    return not problem and s.nunique() > 1 and not (s.dropna() < 0).any() and _looks_like_resource(df, col)


def _looks_like_resource(df: pd.DataFrame, col: str) -> bool:
    from dashboard.components import role_mapping_ui as RMU
    try:
        r = RMU.get_mapping(df).columns.get(col)
    except Exception:
        return False
    return bool(r and r.role in ("ad_spend", "impressions", "reach", "clicks", "views", "visits", "engagement",
                                 "readership", "listenership", "grps", "audience"))


def render_prescriptive_body(df: pd.DataFrame, summary: DatasetSummary) -> None:
    st.info("**Prescriptive Analytics** is delivered in Part 4 of Phase 8 and is implemented. It recommends how to allocate a budget across channels under your "
            "constraints, using the platform's own response model. Outcomes are estimates, never guarantees.")
    render_section("Columns")
    target, channels, time_col, controls = _roles_section(df, summary)
    problems = E.validate_roles(df, channels, target, time_col, controls)
    if problems:
        st.warning("Cannot optimise yet:")
        for p in problems:
            st.caption("• " + escape_text(p))
        return
    unit, monetary, unit_problem = _channel_units(df, channels)
    if unit_problem:
        st.error(escape_text(unit_problem))
        st.caption("Status: **Not supported** for this selection — choose channels measured in one unit.")
        return
    if not monetary:
        st.info(f"The selected columns are measured in **{unit}**, not money. The allocation below is a mix of {unit}; "
                "profit and ROAS-style figures are not calculated.")
    render_section("Objective and Constraints")
    pool = [c for c in summary.numeric_columns if c in df.columns and c not in channels and c != target]
    cp = constraint_panel(df, channels, unit=unit if not monetary else _currency_unit(), monetary=monetary,
                          channels_key=_k("channels"),
                          eligible_extra=tuple(c for c in pool if _is_eligible_extra(df, c, channels)))
    mode, total, margin, trange, mins, maxs = cp["mode"], cp["total"], cp["margin"], cp["trange"], cp["mins"], cp["maxs"]
    reserve = cp["allow_reserve"]
    if mode == "outcome":
        issues = [] if cp["feasible"] else ["infeasible"]
    else:
        issues = E.validate_constraints(channels, total, mins, maxs, trange)
        for i in issues:
            st.error(escape_text(i))
    sig = (tuple(channels), target, time_col, tuple(controls), mode, total, margin, trange, tuple(sorted(mins.items())),
           tuple(sorted(maxs.items())), len(df), tuple(df.columns), reserve)
    if st.button("Find the best allocation", type="primary", key=_k("run"), disabled=bool(issues)):
        mmm = _model_for(df, channels, target, time_col, controls)
        if mmm._model is None:
            cache_set(MODULE_KEY, "res", (sig, E.PrescriptiveResult(
                message="The response model could not be fitted: " + ("; ".join(mmm.limitations) or "not enough usable data."))))
        else:
            with st.spinner("Optimising..."):
                r = E.optimize_allocation(df, mmm, total, mins, maxs, objective=mode, margin=margin, total_range=trange,
                                          allow_reserve=reserve, monetary=monetary)
            cache_set(MODULE_KEY, "res", (sig, r))
    cached = cache_get(MODULE_KEY, "res")
    if cached is None:
        st.info("Set your constraints and press **Find the best allocation**.")
        return
    if cached[0] != sig:
        st.info("Settings changed since the last run; press **Find the best allocation** to refresh. The result below is from the earlier settings.")
    res = cached[1]
    mm = cache_get(MODULE_KEY, "mmm")
    _results(res, df, mm[1] if mm else None, unit if not monetary else _currency_unit())
    if res.status != "ok" or not mm:
        return
    scen = _what_if(res, df, mm[1])
    render_section("Downloads")
    try:
        sheets = E.result_sheets(res, scen)
        render_download_buttons(MODULE_KEY, "allocation_recommendation", table=neutralize_headers(res.table), sheets=sheets)
        st.download_button("⬇️ Download business report (Markdown)", E.business_report(res).encode("utf-8"),
                           file_name="allocation_report.md", mime="text/markdown", key=_k("dl_report"))
    except (ExportError, ValueError) as exc:
        st.caption(f"Export unavailable: {exc}")
    with st.expander("Assumptions"):
        for a in res.assumptions:
            st.markdown(f"- {escape_text(a)}")
    render_limitations([
        "Results are model estimates for one representative period; they are not guarantees.",
        "The model measures association in past data, not proven cause and effect.",
        "Recommendations outside the spend range seen in the data are extrapolations and carry high risk.",
        "Validate a reallocation with a controlled pilot before committing the full budget.",
    ])
