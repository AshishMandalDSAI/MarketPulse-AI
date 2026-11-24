"""Marketing Resource Optimizer page (menu position 06, PART 02).

* Spend-based: monetary budget allocation with the platform's prescriptive
  engine and the PART 02 default-limit fix (feasible defaults that follow the
  total budget and never overwrite limits you typed), feasibility check, solver
  status and objective.
* Exposure-based: reallocation of exposure units among channels measured in
  the same unit, only when the model is validated and the channels have a
  positive, statistically clear association. Never presented as money.
* General: explains which inputs are missing.

Engines: ``src.prescriptive.engine`` / ``limits`` and ``src.marketing_intel.optimizer``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import model_page as MP
from dashboard.components import prescriptive_ui as PUI
from dashboard.components import session_paths
from dashboard.components.stats_ui import apply_dark_layout
from dashboard.components.ui import callout
from src.common.formatting import escape_text
from src.marketing_intel import optimizer as O
from src.prescriptive import engine as E
from src.prescriptive import feasibility as FZ
from src.prescriptive import limits as LIM

PURPOSE = ("Allocate a budget, or an exposure mix, across channels under your own total, minimum and maximum "
           "limits, using a validated response model.")
MODEL_KEY = "resource_optimizer"
LEGACY = ("dashboard.pages.5_budget_optimizer", "Budget Optimizer")


def _k(name: str) -> str:
    return f"mp_page_ro_{name}_{session_paths.get_dataset_id()}"


# --------------------------------------------------------------------------- spend mode
def _spend(ctx):
    eff = MP.effectiveness_for(ctx)
    mm = eff.mmm if eff.status == "ok" else {}
    if not mm or mm.get("status") != "ok":
        callout((mm or {}).get("message", eff.message or "The response model is not available."), kind="bad",
                label="Budget allocation unavailable")
        st.caption("Solution status: **Not supported** — no validated response model for this target.")
        return
    per, mmm_all = mm["period_data"], mm["mmm"]
    unit = MP.CUR.unit_label()
    all_ch = list(ctx.channels)
    chans = st.multiselect("Channels to allocate across", all_ch, default=all_ch, key=_k("channels"),
                           help="Channels you leave out keep no budget in the plan. Remedy D can add one back.")
    if not chans:
        st.warning("Select at least one channel.")
        return
    if set(chans) == set(all_ch):
        mmm = mmm_all
    else:
        from src.prescriptive.engine import fit_response_model
        mmm = MP.memo("ro_mmm", ["ro_mmm", MP.RMU.mapping_signature(ctx.mapping), ctx.target, sorted(chans)],
                      lambda: fit_response_model(per, list(chans), ctx.target, ctx.date_col, []))
        if mmm._model is None:
            callout("The response model could not be fitted for this channel selection.", kind="bad")
            return
    st.caption(f"Response model: the platform's marketing-mix model ({escape_text(mmm.selected_model)}) on "
               f"per-{MP.per_word(ctx)} totals of {len(chans)} channel(s); target **{escape_text(ctx.target)}**.")
    MP.section("Objective and constraints")
    cp = PUI.constraint_panel(per, list(chans), key_fn=_k, unit=unit, monetary=True, channels_key=_k("channels"),
                              eligible_extra=tuple(c for c in all_ch if c not in chans))
    mode, total, margin, trange, mins, maxs = cp["mode"], cp["total"], cp["margin"], cp["trange"], cp["mins"], cp["maxs"]
    issues = ([] if cp["feasible"] else ["infeasible"]) if mode == "outcome" else \
        E.validate_constraints(list(chans), total, mins, maxs, trange)
    if mode != "outcome":
        for i in issues:
            st.error(escape_text(i))
    sig = (ctx.target, tuple(chans), mode, total, margin, trange, tuple(sorted(mins.items())),
           tuple(sorted(maxs.items())), cp["allow_reserve"], MP.RMU.mapping_signature(ctx.mapping))
    if st.button("Find the best allocation", type="primary", key=_k("run"), disabled=bool(issues)):
        with st.spinner("Optimising…"):
            r = E.optimize_allocation(per, mmm, total, mins, maxs, objective=mode, margin=margin, total_range=trange,
                                      allow_reserve=cp["allow_reserve"], monetary=True)
        st.session_state[_k("res")] = (sig, r, mmm)
    cached = st.session_state.get(_k("res"))
    if cached is None:
        st.info("Set the total and limits, then press **Find the best allocation**.")
        return
    if cached[0] != sig:
        st.info("Settings changed since the last run; press **Find the best allocation** to refresh. The result "
                "below is from the earlier settings.")
    MP.section("Result")
    PUI._results(cached[1], per, cached[2], unit)
    if cached[1].status == "ok":
        with st.expander("Assumptions"):
            MP.bullet_list(cached[1].assumptions)


# --------------------------------------------------------------------------- exposure mode
def _exposure(ctx):
    eff = MP.effectiveness_for(ctx)
    model = eff.model if eff.status == "ok" else None
    groups = O.exposure_groups(ctx, model)
    callout("Exposure-based dataset: there is no channel-wise cost, so no monetary budget can be allocated. What can be "
            "optimised is the **mix of exposure units** between channels measured in the same unit (e.g. impressions "
            "across TV, YouTube, Facebook and Instagram) — and only when the model supports it.", kind="info",
            label="What is optimised")
    MP.section("Which exposure units can be optimised")
    st.dataframe(pd.DataFrame([{"unit": g.unit, "channels": ", ".join(ctx.channel_label[c] for c in g.channels),
                                "optimisable": "yes" if g.available else "no",
                                "eligible channels": ", ".join(ctx.channel_label[c] for c in g.eligible),
                                "why not / held channels": " ".join(g.reasons) + (" Held: " + "; ".join(
                                    f"{ctx.channel_label[c]} ({w})" for c, w in g.held.items()) if g.held else "")}
                               for g in groups]), hide_index=True, width="stretch")
    if model is not None:
        st.caption(escape_text(f"Response model quality: {model.quality} — {model.quality_reason}."))
    ok = [g for g in groups if g.available]
    if not ok:
        callout("No exposure unit can be optimised defensibly for this target (see the reasons above). Nothing is "
                "recommended rather than showing an unsupported allocation.", kind="warn", label="Not available")
        return
    g = ok[0] if len(ok) == 1 else next(x for x in ok if x.unit == st.selectbox(
        "Exposure unit to reallocate", [x.unit for x in ok], key=_k("unit")))
    cur = O.current_levels(ctx, g.eligible)
    cur_tot = float(sum(cur.values()))
    MP.section(f"Total and limits ({g.unit} per {MP.per_word(ctx)})")
    total = float(st.number_input(f"Total {g.unit} per {MP.per_word(ctx)} (not money)", min_value=0.0,
                                  value=float(round(cur_tot)), key=_k(f"total_{g.unit}"),
                                  help=f"Current: {cur_tot:,.0f} {g.unit} per {MP.per_word(ctx)} (recent average)."))
    d_min, d_max, rule = LIM.default_limits(total, g.eligible, cur)
    pairs = {c: (_k(f"min_{c}"), _k(f"max_{c}")) for c in g.eligible}
    status = LIM.reconcile_limits(st.session_state, pairs, d_min, d_max, _k(f"auto_{g.unit}"))
    st.caption(escape_text(rule))
    mins, maxs = {}, {}
    for c in g.eligible:
        a, b = st.columns(2)
        mins[c] = float(a.number_input(f"{ctx.channel_label[c]} - minimum {g.unit}"
                                       + ("" if status[c]["min"] == "auto" else " · set by you"), min_value=0.0,
                                       key=pairs[c][0]))
        maxs[c] = float(b.number_input(f"{ctx.channel_label[c]} - maximum {g.unit}"
                                       + ("" if status[c]["max"] == "auto" else " · set by you"), min_value=0.0,
                                       key=pairs[c][1]))
    if any(v != "auto" for s in status.values() for v in s.values()):
        st.button("Reset all limits to the defaults", key=_k("reset"), on_click=LIM.reset_limits,
                  args=(st.session_state, pairs, d_min, d_max, _k(f"auto_{g.unit}")))
    rep = FZ.diagnose(total, mins, maxs, g.eligible, unit=g.unit)
    PUI.render_feasibility(rep, {"registry": _k(f"auto_{g.unit}"), "max": {c: pairs[c][1] for c in g.eligible},
                                 "min": {c: pairs[c][0] for c in g.eligible}, "total": _k(f"total_{g.unit}"),
                                 "reserve": None, "channels": None}, g.unit, allow_reserve_option=False)
    feas = {"feasible": rep.feasible}
    sig = (ctx.target, g.unit, total, tuple(sorted(mins.items())), tuple(sorted(maxs.items())),
           MP.RMU.mapping_signature(ctx.mapping))
    if st.button("Find the best exposure mix", type="primary", key=_k("run_exp"), disabled=not feas["feasible"]):
        with st.spinner("Optimising…"):
            st.session_state[_k("exp_res")] = (sig, O.optimize_exposure(ctx, model, g, total, mins, maxs))
    cached = st.session_state.get(_k("exp_res"))
    if cached is None:
        st.info("Set the total and limits, then press **Find the best exposure mix**.")
        return
    if cached[0] != sig:
        st.info("Settings changed since the last run; press the button to refresh.")
    r = cached[1]
    MP.section("Result")
    MP.callout(f"**{FZ.STATUS_LABELS.get(r.solution_status, r.solution_status)}.** {r.solution_status_text}",
               kind=PUI.STATUS_KIND.get(r.solution_status, "info"), label="Solution status")
    st.markdown(f"**Solver status:** {escape_text({'ok': 'Solved', 'infeasible': 'Infeasible limits', 'failed': 'Solver failed', 'unavailable': 'Not run'}.get(r.status, r.status))}"
                + (f" · converged: {'yes' if r.solver.get('converged') else 'no'} · {escape_text(r.solver.get('message', ''))}" if r.solver else "")
                + f"  \n**Objective:** {escape_text(r.objective)}")
    if r.status != "ok":
        callout(r.message, kind="bad", label="No recommendation")
        MP.bullet_list(r.issues)
        return
    t = r.table()
    t["channel"] = t.channel.map(ctx.channel_label)
    k = st.columns(3)
    k[0].metric(MP.target_axis(ctx, "Estimated ") + " — current mix", f"{r.current_outcome:,.2f}")
    k[1].metric("Estimated — recommended mix", f"{r.recommended_outcome:,.2f}",
                None if r.change_pct is None else f"{r.change_pct:+.2f}%")
    k[2].metric("Approx. 90% range (±)", "—" if r.low is None else f"± {(r.high - r.low) / 2:,.0f}",
                help=None if r.low is None else f"Recommended mix: {r.low:,.0f} to {r.high:,.0f}")
    if r.low is not None:
        st.caption(f"Approximate 90% range for the recommended mix: {r.low:,.0f} to {r.high:,.0f} (holdout error of "
                   "per-period totals).")
    st.caption(f"Model estimates per {MP.per_word(ctx)}; the change is small relative to the range, so treat it as "
               "directional." if r.low is not None and r.change is not None and abs(r.change) < (r.high - r.low) / 2 else
               f"Model estimates per {MP.per_word(ctx)}.")
    fig = go.Figure([go.Bar(name="Current (recent average)", x=t.channel, y=t.current_per_period),
                     go.Bar(name="Recommended", x=t.channel, y=t.recommended_per_period)])
    fig.update_layout(barmode="group", legend_title_text="Exposure mix")
    st.plotly_chart(apply_dark_layout(fig, f"Current vs recommended {g.unit} per {MP.per_word(ctx)}",
                                      x_title="Channel", y_title=f"{g.unit[:1].upper() + g.unit[1:]} per {MP.per_word(ctx)}"),
                    width="stretch")
    st.dataframe(t.round(2), hide_index=True, width="stretch")
    for w in r.warnings:
        callout(w, kind="warn")
    with st.expander("Assumptions"):
        MP.bullet_list(r.assumptions)


def body(ctx, caps):
    if ctx.mode.mode == "spend":
        if not ctx.date_col:
            callout("Budget allocation needs a date column to fit the response model.", kind="warn")
            return
        _spend(ctx)
    elif ctx.mode.mode == "exposure":
        if not ctx.date_col:
            callout("Exposure reallocation needs a date column.", kind="warn")
            return
        _exposure(ctx)
    else:
        callout("Marketing resource optimisation is not available: the dataset has no channel-wise advertising spend "
                "(needed for budget allocation) and no marketing exposure columns (needed for an exposure mix). "
                "To enable it, add one spend column per channel — for example TV, search and social spend per "
                "period — together with a date column and the outcome.", kind="warn", label="Missing inputs")
        missing = [c for c in caps if not c.available]
        MP.bullet_list([f"{c.label}: needs {c.needs}" for c in missing if c.needs])


def render():
    MP.render_model_page("resource_optimizer", PURPOSE, MODEL_KEY, body, LEGACY)
