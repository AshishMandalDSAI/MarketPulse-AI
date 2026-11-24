"""Marketing Scenario Simulator page (menu position 07, PART 02).

Baseline and several alternative scenarios -- spend, exposure, price and
promotion changes, depending on what the dataset contains -- compared side by
side. Historical actuals and model estimates are shown separately; estimates
carry an approximate range, extrapolation flags and a reliability grade.
Engine: ``src.marketing_intel.scenarios``.
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
from dashboard.components import session_paths
from dashboard.components.stats_ui import apply_dark_layout
from dashboard.components.ui import callout
from src.common.formatting import escape_text
from src.common.export import dataframe_to_csv_bytes  # PART 06: also neutralises formula-like cells
from src.common.headers import neutralize_headers
from src.marketing_intel import scenarios as SC

PURPOSE = ("See how a validated model expects your outcome to change when spend, exposure, price, promotion or "
           "other inputs change, compared with the baseline.")
MODEL_KEY = "scenario_simulator"
LEGACY = ("dashboard.pages.6_what_if_simulator", "What-If Simulator")
WINDOWS = {"Last 8 periods": 8, "Last 13 periods": 13, "Last 26 periods": 26, "All history": None}


def _k(name: str) -> str:
    return f"mp_page_sim_{name}_{session_paths.get_dataset_id()}"


def _scenario_store(ctx, model) -> list:
    key = _k(f"list_{ctx.target}_{MP.RMU.mapping_signature(ctx.mapping)[:40]}")
    if key not in st.session_state:
        st.session_state[key] = SC.preset_scenarios(model, ctx)
    return st.session_state[key]


def _builder(ctx, model, store):
    adj = SC.adjustable_drivers(model, ctx)
    with st.form(_k("form"), clear_on_submit=False):
        st.markdown("**Add a scenario**")
        name = st.text_input("Scenario name", value=f"Scenario {len(store) + 1}", key=_k("name"))
        changes = {}
        budget_pct = 0.0
        if ctx.channel_kind == "spend":
            budget_pct = st.number_input("Total marketing budget: % change (keeps the current channel mix)",
                                         value=0.0, step=5.0, min_value=-100.0, max_value=500.0, key=_k("budget_pct"),
                                         help="Applied to every spend channel; any per-channel change below is "
                                              "applied on top of it.")
        groups = [("channel", "Marketing channels (% change)"), ("business", "Business variables"),
                  ("other", "Other drivers (% change)")]
        for kind, title in groups:
            items = [a for a in adj if a["kind"] == kind]
            if not items:
                continue
            st.caption(title)
            cols = st.columns(min(3, len(items)))
            for i, a in enumerate(items):
                with cols[i % len(cols)]:
                    if a["type"] == "share":
                        v = st.slider(f"{a['label']}: % of records with it on", 0, 100,
                                      int(round(a["current_share_pct"] or 0)), key=_k(f"sh_{a['column']}"))
                        if abs(v - (a["current_share_pct"] or 0)) >= 0.5:
                            changes[a["column"]] = {"type": "share", "value": float(v)}
                    else:
                        v = st.number_input(f"{a['label']}: % change", value=0.0, step=5.0, min_value=-100.0,
                                            max_value=500.0, key=_k(f"pc_{a['column']}"))
                        if v != 0:
                            changes[a["column"]] = {"type": "pct", "value": float(v)}
        if st.form_submit_button("Add scenario"):
            if budget_pct:
                changes = SC.combine(SC.budget_change(model, budget_pct), changes)
            if not changes:
                st.warning("Change at least one driver.")
            elif any(s.name == name for s in store):
                st.warning("A scenario with that name already exists.")
            else:
                store.append(SC.Scenario(name=name, changes=changes))
                st.rerun()


def body(ctx, caps):
    with st.spinner("Fitting and validating the response model…"):
        eff = MP.effectiveness_for(ctx)
    if eff.status != "ok":
        callout(eff.message or "No response model is available, so scenarios cannot be estimated.", kind="bad",
                label="Simulator unavailable")
        return
    model = eff.model
    if ctx.mode.mode == "general":
        callout("No marketing channels are mapped, so the simulator covers the business drivers that exist (price, "
                "promotion, discount, distribution...). Spend and exposure scenarios are unavailable.", kind="info")
    grade = {"good": "good", "moderate": "info", "weak": "bad"}.get(model.quality, "warn")
    callout({"good": "**Good** — the model explains the held-out periods well.",
               "moderate": "**Moderate** — usable for direction and rough size; uncertainty is wide.",
               "weak": "**Low reliability** — the model explains the held-out periods poorly. The estimates below "
                       "must not be used as forecasts."}.get(model.quality, "Unvalidated model.")
            + f" ({model.quality_reason})", kind=grade, label="Model reliability")
    store = _scenario_store(ctx, model)
    MP.section("Scenarios")
    wlabel = st.selectbox("Baseline window", list(WINDOWS), key=_k("window"),
                          help="Changes are applied to the records in this recent window.") if ctx.date_col else None
    window = WINDOWS[wlabel] if wlabel else None
    _builder(ctx, model, store)
    if store:
        for i, s in enumerate(list(store)):
            a, b = st.columns([5, 1])
            a.markdown(f"**{escape_text(s.name)}** — " + escape_text(", ".join(
                f"{c}: {'on for ' + format(v['value'], '.0f') + '% of records' if v['type'] == 'share' else format(v['value'], '+.0f') + '%'}"
                for c, v in s.changes.items())))
            if b.button("Remove", key=_k(f"rm_{i}_{s.name}")):
                store.remove(s)
                st.rerun()
    else:
        st.caption("No scenarios yet; add one above.")
        return
    res = SC.simulate(ctx, model, store, window=window, mmm=eff.mmm if ctx.mode.mode == "spend" else None)
    if res.status != "ok":
        callout(res.message, kind="bad")
        return
    MP.section("Comparison")
    t = res.table()
    unit = MP.target_unit(ctx)
    st.caption(escape_text(f"Values are {ctx.target} per {MP.per_word(ctx) if ctx.date_col else 'record'}"
                           + (f" ({unit})" if unit else "") + f"; baseline window: {res.window_text}."))
    st.dataframe(t.round(2), hide_index=True, width="stretch")
    fig = go.Figure()
    act = t[t.kind == "actual"]
    est = t[t.kind != "actual"]
    fig.add_trace(go.Bar(name="Historical actual", x=act.scenario, y=act.estimate, marker_color="#97A3BF"))
    err_hi = (est.high_90 - est.estimate).fillna(0).to_numpy()
    err_lo = (est.estimate - est.low_90).fillna(0).to_numpy()
    fig.add_trace(go.Bar(name="Model estimate", x=est.scenario, y=est.estimate, marker_color="#3B82F6",
                         error_y=dict(type="data", array=err_hi, arrayminus=err_lo, visible=True)))
    fig.update_layout(legend_title_text="Value type")
    st.plotly_chart(apply_dark_layout(fig, f"Scenario comparison — {ctx.target} per {MP.per_word(ctx)}",
                                      x_title="Scenario", y_title=MP.target_axis(ctx, "Estimated ")),
                    width="stretch")
    st.caption(escape_text("Error bars: " + res.range_basis + "."))
    flagged = [o for o in res.outcomes if o.extrapolation]
    for o in flagged:
        callout(f"**{o.name}**: " + "; ".join(o.extrapolation), kind="warn", label="Extrapolation")
    if res.mmm_check:
        MP.section("Cross-check with the marketing-mix model (spend-only scenarios)")
        st.dataframe(pd.DataFrame(res.mmm_check).round(2), hide_index=True, width="stretch")
        st.caption("The platform's adstock / saturation model, evaluated for one representative period. Agreement "
                   "between the two models adds confidence; disagreement means the estimate is model-dependent.")
    MP.section("Sensitivity analysis")
    step = st.select_slider("Change tested for each driver", options=[5, 10, 20], value=10, key=_k("sens_step"),
                            format_func=lambda v: f"±{v}%")
    sens = MP.memo("sens", ["sens", MP.RMU.mapping_signature(ctx.mapping), ctx.target, window, step, len(ctx.df)],
                   lambda: SC.sensitivity(ctx, model, window, float(step)))
    if len(sens):
        top = sens.head(15).iloc[::-1]
        fig = go.Figure([go.Bar(name=f"Driver −{step}%", y=top.driver, x=top.change_low, orientation="h",
                                marker_color="#EF4444"),
                         go.Bar(name=f"Driver +{step}%", y=top.driver, x=top.change_high, orientation="h",
                                marker_color="#3ddc97")])
        fig.update_layout(barmode="overlay", legend_title_text="Change tested")
        st.plotly_chart(apply_dark_layout(fig, f"Sensitivity of {ctx.target} per {MP.per_word(ctx)} to each driver "
                                               "(one at a time)", max(320, 30 * len(top) + 140),
                                          x_title=MP.target_axis(ctx, "Change in estimated "), y_title="Driver"),
                        width="stretch")
        st.dataframe(sens.drop(columns=["column"]).round(3), hide_index=True, width="stretch")
        st.caption("Each driver is moved on its own while every other driver keeps its recorded values. Model "
                   "estimates relative to the model baseline; promotion flags move by percentage points of records.")
    MP.section("Downloads")
    st.download_button("⬇️ Scenario comparison (CSV)", dataframe_to_csv_bytes(neutralize_headers(t)),
                       file_name=f"scenarios_{ctx.target}.csv", mime="text/csv", key=_k("dl"))
    with st.expander("Assumptions and limitations"):
        MP.bullet_list(res.assumptions + res.warnings)


def render():
    MP.render_model_page("scenario_simulator", PURPOSE, MODEL_KEY, body, LEGACY)
