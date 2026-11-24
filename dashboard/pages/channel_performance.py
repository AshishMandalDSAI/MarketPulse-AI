"""Channel Performance & Contribution page (menu position 05, PART 02).

Channel comparison, ranking, trends, contribution and brand / geography
breakdowns, in each channel's own unit. Exposure channels get a statistical
association share (%), never money; spend channels also get the marketing-mix
model's contribution and, for a monetary target, ROI / ROAS.
Engine: ``src.marketing_intel.channels``.
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
from src.marketing_intel.channels import run_channel_performance

PURPOSE = ("Compare and rank the channels in your data, follow their trends and see how much of the outcome "
           "each one is associated with, in the units the data actually records.")
MODEL_KEY = "channel_performance"
LEGACY = ("dashboard.pages.4_channel_roi", "Channel ROI")


def _uw(ctx, unit: str) -> str:
    """Words for a channel unit: 'spend (USD)' for money, the exposure unit otherwise."""
    if ctx.channel_kind == "spend":
        return f"spend ({unit})" if unit else "spend"
    return unit


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _segment_alternative(ctx):
    MP.section("Supported alternative: segment performance")
    shown = False
    for role in ("brand", "geography", "product"):
        for dim in ctx.dims.get(role, []):
            if ctx.df[dim].nunique() > 30:
                continue
            s = ctx.df.groupby(dim, observed=True)[ctx.target].sum().sort_values()
            st.plotly_chart(MP.bar(list(s.index.astype(str)), s.values, f"Total {ctx.target} by {dim}", dim,
                                   MP.target_axis(ctx, "Total "), horizontal=True), width="stretch")
            shown = True
    if not shown:
        st.caption("No brand, geography or product column is mapped either.")


def body(ctx, caps):
    if not ctx.channels:
        callout("Channel performance needs channel-wise advertising spend or marketing exposure (impressions, reach, "
                "clicks, readership, listenership...). None is mapped in this dataset, so no channel is analysed "
                "and nothing is invented.", kind="warn", label="No channels")
        _segment_alternative(ctx)
        return
    with st.spinner("Analysing channels…"):
        eff = MP.effectiveness_for(ctx)
        res = MP.memo("chan", ["chan", MP.RMU.mapping_signature(ctx.mapping), ctx.target, len(ctx.df)],
                      lambda: run_channel_performance(ctx, eff.effects if eff.status == "ok" else None,
                                                      eff.mmm if eff.status == "ok" else None))
    for w in res.warnings + ([] if eff.status == "ok" else [eff.message]):
        callout(w, kind="warn")
    if eff.status == "ok" and eff.model.quality == "weak":
        callout("The driver model behind the association shares is weak on held-out periods ("
                + eff.model.quality_reason + "); treat shares and ranks as rough indications.", kind="warn")
    t = res.table
    MP.section("Channel comparison")
    st.dataframe(t.round(3), hide_index=True, width="stretch")
    st.caption(escape_text(f"Ranking basis: {res.ranking_basis or 'none available'}."))
    if "rank" in t:
        key = {"model-estimated ROAS (marketing-mix model)": ("model_roas", "Model-estimated ROAS"),
               "model-estimated outcome per unit of spend": ("outcome_per_unit_spend", f"{ctx.target} per unit of spend"),
               "statistical association share (driver model)": ("association_share_pct", "Association share (%)"),
               "correlation with the outcome per period": ("corr_with_target_per_period", "Correlation (r)")}[res.ranking_basis]
        tt = t.sort_values(key[0])
        st.plotly_chart(MP.bar(list(tt.channel), tt[key[0]], f"Channel ranking by {res.ranking_basis}", "Channel",
                               key[1], horizontal=True), width="stretch")
    for unit, cols in res.unit_groups.items():
        sub = t[t.column.isin(cols)]
        if len(sub) >= 1 and "avg_per_period" in sub:
            s = sub.sort_values("avg_per_period")
            st.plotly_chart(MP.bar(list(s.channel), s.avg_per_period,
                                   f"Average {_uw(ctx, unit)} per {MP.per_word(ctx)} by channel", "Channel",
                                   f"Average {_uw(ctx, unit)} per {MP.per_word(ctx)}", horizontal=True), width="stretch")
    if res.trends is not None:
        MP.section("Channel trends")
        for unit, cols in res.unit_groups.items():
            fig = go.Figure()
            for c in cols:
                d = res.trends[res.trends.column == c]
                fig.add_trace(go.Scatter(x=d[ctx.date_col], y=d.value, mode="lines", name=ctx.channel_label[c]))
            fig.update_layout(legend_title_text="Channel")
            st.plotly_chart(apply_dark_layout(fig, f"{_cap(_uw(ctx, unit))} per {MP.per_word(ctx)} by channel",
                                              x_title=ctx.date_col, y_title=f"Total {_uw(ctx, unit)} per {MP.per_word(ctx)}"),
                            width="stretch")
    if "model_contribution" in t and t.model_contribution.notna().any():
        MP.section("Model-estimated contribution (spend-based)")
        c = t.sort_values("model_contribution")
        st.plotly_chart(MP.bar(list(c.channel), c.model_contribution,
                               f"Model-estimated contribution to {ctx.target} by channel", "Channel",
                               MP.target_axis(ctx, "Contribution to "), horizontal=True), width="stretch")
        st.caption("From the platform's marketing-mix model (leave-one-channel-out). Model estimates, not "
                   "measured incrementality.")
    elif ctx.channel_kind == "exposure":
        MP.section("Contribution")
        st.caption("Exposure-based data: contribution is the statistical association share shown in the ranking "
                   "(in %). No monetary contribution, ROI or ROAS is calculated because no channel-wise cost exists.")
    if res.rates is not None and len(res.rates):
        MP.section("Click-through and conversion rates")
        st.dataframe(res.rates.round(4), hide_index=True, width="stretch")
        st.caption("Simple ratios of recorded totals for channels that record both measures; not model estimates.")
    if res.breakdowns:
        MP.section("Breakdowns by brand / geography")
        dims = list(res.breakdowns)
        dim = st.selectbox("Breakdown by", dims, key="mp_page_chan_dim") if len(dims) > 1 else dims[0]
        b = res.breakdowns[dim]
        for unit, cols in res.unit_groups.items():
            sub = b[b.column.isin(cols)]
            fig = go.Figure([go.Bar(name=ctx.channel_label[c], x=sub[sub.column == c][dim].astype(str),
                                    y=sub[sub.column == c].total) for c in cols])
            fig.update_layout(barmode="group", legend_title_text="Channel")
            st.plotly_chart(apply_dark_layout(fig, f"Total {_uw(ctx, unit)} by {dim} and channel", x_title=dim,
                                              y_title=f"Total {_uw(ctx, unit)}"), width="stretch")
    MP.section("Downloads")
    st.download_button("⬇️ Channel table (CSV)", dataframe_to_csv_bytes(neutralize_headers(t)),
                       file_name=f"channel_performance_{ctx.target}.csv", mime="text/csv", key="mp_page_chan_dl")
    with st.expander("Methodology and limitations"):
        MP.bullet_list(res.methodology + [
            "Associations come from observational data; they are not proof that a channel causes the outcome.",
            "Channels recorded at a coarser level (e.g. per brand and region) are counted once per recorded value."])


def render():
    MP.render_model_page("channel_performance", PURPOSE, MODEL_KEY, body, LEGACY)
