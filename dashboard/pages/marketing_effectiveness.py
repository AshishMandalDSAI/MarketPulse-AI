"""Marketing Effectiveness Model page (menu position 04, PART 02).

How marketing (spend or exposure), price, promotion, distribution and other
business variables relate to the chosen outcome, estimated from the dataset.
Spend-based datasets add the marketing-mix model's contribution and, for a
monetary target, ROI / ROAS. Exposure-based datasets never get ROI / ROAS.
General datasets get the business-driver analysis and an explanation of why the
marketing model is unavailable. Engine: ``src.marketing_intel.effectiveness``.
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
from dashboard.components.stats_ui import apply_dark_layout
from dashboard.components.ui import callout
from src.common.formatting import escape_text
from src.common.export import dataframe_to_csv_bytes  # PART 06: also neutralises formula-like cells
from src.common.headers import neutralize_headers
from src.marketing_intel.context import period_totals
from src.marketing_intel.response import select_drivers

PURPOSE = ("How marketing, promotion, pricing, distribution and other business variables relate to your outcome, "
           "estimated from your data. Relationships are statistical associations, not proof of cause.")
MODEL_KEY = "marketing_effectiveness"
LEGACY = ("dashboard.pages.3_marketing_mix_model", "Marketing Mix Model")
SIG_COLORS = {True: "#3ddc97", False: "#7a8599"}


def _settings(ctx):
    default, _ = select_drivers(ctx)
    pool = [c for c in ctx.default_drivers() + ctx.other_numeric if c not in ctx.excluded and c != ctx.target]
    pool = list(dict.fromkeys(default + pool))
    with st.container(border=True):
        st.markdown("**Model settings**")
        drivers = st.multiselect("Drivers in the model", pool, default=default, key=f"mp_page_eff_drivers_{ctx.target}",
                                 help="Defaults: the active mode's channels plus business variables. Near-duplicate "
                                      "drivers are left out by default (see notes).")
        c1, c2 = st.columns(2)
        fe = c1.checkbox("Account for differences between segments (fixed effects)", value=True,
                         key="mp_page_eff_fe", disabled=not ctx.dim_cols,
                         help="Brand / geography / product / SKU levels.")
        trend = c2.checkbox("Include a time trend", value=True, key="mp_page_eff_trend", disabled=not ctx.date_col)
    return drivers, fe, trend


def _effects_section(ctx, res):
    e = res.effects
    MP.section("Effect estimates")
    show = e[["driver", "coef", "ci95_low", "ci95_high", "p_value", "significant_5pct", "elasticity_at_mean",
              "standardized_effect", "vif"] + (["association_share_pct"] if "association_share_pct" in e else [])]
    st.dataframe(show.rename(columns={
        "coef": f"Coefficient ({MP.target_unit(ctx) or ctx.target} per unit of driver)", "ci95_low": "95% CI low",
        "ci95_high": "95% CI high", "p_value": "p-value", "significant_5pct": "Significant (5%)",
        "elasticity_at_mean": "Elasticity at average", "standardized_effect": "Standardized effect",
        "vif": "VIF", "association_share_pct": "Association share (%)"}).round(4), hide_index=True,
        width="stretch")
    st.caption("Channel coefficients are per one-unit increase in log(1 + channel value). Elasticity: estimated % "
               f"change in {ctx.target} for a 1% change in the driver, at average levels.")
    ee = e.sort_values("elasticity_at_mean")
    st.plotly_chart(MP.bar(list(ee.driver), ee.elasticity_at_mean, "Elasticity of " + ctx.target + " by driver",
                           "Driver", "Elasticity (% change in target per 1% change)", horizontal=True,
                           color=[SIG_COLORS[bool(s)] for s in ee.significant_5pct]), width="stretch")
    st.caption("Green = statistically significant at 5%; grey = not significant.")
    ch = e[e.kind == "channel"]
    if len(ch) and "association_share_pct" in e and ch.association_share_pct.notna().any():
        c = ch.sort_values("association_share_pct")
        st.plotly_chart(MP.bar(list(c.driver), c.association_share_pct,
                               "Statistical association share by channel (not revenue attribution)", "Channel",
                               "Share of positive channel association (%)", horizontal=True), width="stretch")


def _diagnostics(ctx, res):
    model = res.model
    MP.section("Model performance")
    MP.model_quality_metrics(model)
    data = ctx.df.loc[model.used_rows]
    if ctx.date_col and model.train_fit is not None:
        pred = model.predict(data)
        per = pd.DataFrame({"d": data[ctx.date_col].to_numpy(), "actual": pd.to_numeric(data[ctx.target]).to_numpy(float),
                            "fit": pred}).groupby("d").sum()
        fig = go.Figure([go.Scatter(x=per.index, y=per.actual, name="Actual (historical)", mode="lines"),
                         go.Scatter(x=per.index, y=per.fit, name="Model estimate (fitted)", mode="lines",
                                    line=dict(dash="dash"))])
        dates = np.sort(data[ctx.date_col].unique())
        n_test = model.holdout.get("n_periods", 0)
        if n_test:
            fig.add_vrect(x0=dates[-n_test], x1=dates[-1], fillcolor="#888", opacity=0.15, line_width=0,
                          annotation_text="holdout periods", annotation_position="top left")
        st.plotly_chart(apply_dark_layout(fig, f"Actual vs model estimate — {ctx.target} per {MP.per_word(ctx)}",
                                          x_title=ctx.date_col, y_title=MP.target_axis(ctx, "Total ")),
                        width="stretch")
    if model.importance is not None and len(model.importance):
        imp = model.importance.sort_values("importance")
        st.plotly_chart(MP.bar(list(imp.driver), imp.importance, "Variable importance (permutation, holdout data)",
                               "Driver", "Drop in holdout R² when shuffled", horizontal=True, error=imp["std"]),
                        width="stretch")


def _correlations(ctx, res):
    MP.section("Correlation with the target")
    cor = res.correlations
    st.dataframe(cor.round(4), hide_index=True, width="stretch")
    c = cor.sort_values("pearson_r")
    st.plotly_chart(MP.bar(list(c.driver), c.pearson_r, f"Correlation of each driver with {ctx.target} (rows)",
                           "Driver", "Pearson correlation (r)", horizontal=True), width="stretch")
    st.caption("Correlation measures co-movement only; the regression above separates drivers that move together.")


def _mmm(ctx, res):
    mm = res.mmm or {}
    MP.section("Marketing-mix model (spend-based)")
    if mm.get("status") != "ok":
        callout(mm.get("message", "The marketing-mix model is not available."), kind="warn", label="Not available")
        return
    t = mm["table"].copy()
    cur = MP.target_unit(ctx)
    st.caption(f"Selected model: {escape_text(mm['selected_model'])}; test R² "
               f"{mm['test_metrics'].get('r2', float('nan')):.2f}; {mm['n_train']} training / {mm['n_test']} test periods. "
               + escape_text(mm["note"]))
    st.dataframe(t.round(3), hide_index=True, width="stretch")
    if "model_roas" in t:
        tt = t.sort_values("model_roas")
        st.plotly_chart(MP.bar(list(tt.channel), tt.model_roas, "Model-estimated ROAS by channel", "Channel",
                               f"{ctx.target} per 1 unit of spend" + (f" ({cur} per {cur})" if cur else ""),
                               horizontal=True), width="stretch")
    else:
        tt = t.sort_values("outcome_per_unit_spend")
        st.plotly_chart(MP.bar(list(tt.channel), tt.outcome_per_unit_spend,
                               f"Model-estimated {ctx.target} per unit of spend", "Channel",
                               f"{MP.target_unit(ctx)} per currency unit of spend", horizontal=True),
                        width="stretch")


def _segments(ctx, res):
    labels = ctx.channel_label
    if res.by_time is not None and len(res.by_time):
        MP.section("Effectiveness over time")
        chans = [c for c in res.by_time.columns if c in labels]
        st.plotly_chart(MP.heatmap(res.by_time, "window", chans, labels,
                                   f"Channel–{ctx.target} association by time window (per-{MP.per_word(ctx)} totals)",
                                   "Time window"), width="stretch")
    for dim, t in res.by_segment.items():
        MP.section(f"Effectiveness by {dim}")
        chans = [c for c in t.columns if c in labels]
        st.plotly_chart(MP.heatmap(t, dim, chans, labels, f"Channel–{ctx.target} association within each {dim}", dim),
                        width="stretch")
    if res.promotions is not None and len(res.promotions):
        MP.section("Promotion effectiveness")
        p = res.promotions
        st.dataframe(p.round(4), hide_index=True, width="stretch")
        fig = go.Figure([go.Bar(name="Promotion on", x=p.promotion, y=p.avg_target_on),
                         go.Bar(name="Promotion off", x=p.promotion, y=p.avg_target_off)])
        fig.update_layout(barmode="group", legend_title_text="Records")
        st.plotly_chart(apply_dark_layout(fig, f"Average {ctx.target} per record with and without each promotion",
                                          x_title="Promotion", y_title=MP.target_axis(ctx, "Average ")),
                        width="stretch")
        st.caption("Raw differences mix the promotion with everything that happens at the same time; the model "
                   "coefficient is the difference after accounting for the other drivers.")


def _segment_performance(ctx):
    for role in ("brand", "geography", "product"):
        for dim in ctx.dims.get(role, []):
            if ctx.df[dim].nunique() > 30:
                continue
            s = ctx.df.groupby(dim, observed=True)[ctx.target].sum().sort_values()
            st.plotly_chart(MP.bar(list(s.index.astype(str)), s.values, f"Total {ctx.target} by {dim}", dim,
                                   MP.target_axis(ctx, "Total "), horizontal=True), width="stretch")


def body(ctx, caps):
    if ctx.mode.mode == "general":
        callout("The marketing effectiveness model cannot run on this dataset: no channel-wise advertising spend and "
                "no marketing exposure columns are mapped. Add spend per channel (for ROI / ROAS) or exposure such as "
                "impressions, reach, clicks, readership or listenership to enable it. The supported alternative "
                "below analyses the business drivers that do exist.", kind="warn", label="Marketing model unavailable")
        MP.section("Supported alternative: business driver analysis")
    drivers, fe, trend = _settings(ctx)
    if not drivers:
        callout("Select at least one driver to run the model.", kind="warn")
        if ctx.mode.mode == "general":
            _segment_performance(ctx)
        return
    with st.spinner("Fitting and validating the model…"):
        res = MP.effectiveness_for(ctx, drivers, fe, trend)
    if res.status != "ok":
        callout(res.message, kind="bad", label="Model not available")
        MP.bullet_list(res.warnings)
        return
    for w in res.warnings:
        callout(w, kind="warn")
    if ctx.mode.mode == "exposure":
        callout("This is an exposure-based dataset: effects are per unit of exposure (impressions, readers, "
                "listeners...). No channel-wise cost exists, so ROI, ROAS and monetary contribution are not "
                "calculated.", kind="info", label="Exposure-based")
    _diagnostics(ctx, res)
    _effects_section(ctx, res)
    _correlations(ctx, res)
    if ctx.mode.mode == "spend":
        _mmm(ctx, res)
    _segments(ctx, res)
    if ctx.mode.mode == "general":
        MP.section("Segment performance")
        _segment_performance(ctx)
    MP.section("Downloads")
    st.download_button("⬇️ Effect estimates (CSV)", dataframe_to_csv_bytes(neutralize_headers(res.effects)),
                       file_name=f"effectiveness_{ctx.target}.csv", mime="text/csv", key="mp_page_eff_dl")
    with st.expander("Methodology"):
        MP.bullet_list(res.methodology)
    with st.expander("Assumptions and limitations"):
        MP.bullet_list(res.assumptions)
        if ctx.excluded:
            st.markdown("**Columns kept out of the model**")
            MP.bullet_list([f"{c}: {why}" for c, why in ctx.excluded.items()])


def render():
    MP.render_model_page("marketing_effectiveness", PURPOSE, MODEL_KEY, body, LEGACY)
