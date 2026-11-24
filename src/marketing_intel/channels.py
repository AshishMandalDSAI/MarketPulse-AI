"""MODEL 05 -- Channel Performance & Contribution (PART 02).

Per channel (advertising spend in spend mode, exposure in exposure mode):
volume, average per period, share within the same unit, activity, trend,
association with the outcome, statistical contribution share, a ranking and
brand / geography breakdowns. Channels measured in different units
(impressions vs readers vs listeners) are never added together or ranked on
volume against each other.

Monetary contribution, ROI and ROAS are shown only in spend mode, from the
platform's marketing-mix model (``effectiveness.spend_mmm``), and ROI / ROAS
only when the target is monetary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.marketing_intel import stats as S
from src.marketing_intel.context import ModelContext, channel_grains, dedup_frame, period_totals, value_grain


@dataclass
class ChannelResult:
    status: str = "unavailable"
    message: str = ""
    table: Optional[pd.DataFrame] = None
    trends: Optional[pd.DataFrame] = None           # per-period totals (long format)
    breakdowns: dict = field(default_factory=dict)  # dim -> long DataFrame (level, channel, total, unit)
    rates: Optional[pd.DataFrame] = None            # click-through / conversion rates where both measures exist
    grains: dict = field(default_factory=dict)
    unit_groups: dict = field(default_factory=dict)
    ranking_basis: str = ""
    methodology: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def run_channel_performance(ctx: ModelContext, effects: Optional[pd.DataFrame] = None,
                            mmm: Optional[dict] = None) -> ChannelResult:
    res = ChannelResult()
    if not ctx.channels:
        res.message = ("No channel-wise advertising spend or marketing exposure columns are mapped, so channel "
                       "performance cannot be analysed.")
        return res
    res.status = "ok"
    res.grains = channel_grains(ctx)
    res.unit_groups = ctx.unit_groups()
    rows = []
    per = period_totals(ctx, ctx.channels) if ctx.date_col else None
    for c in ctx.channels:
        d = dedup_frame(ctx.df, c, ctx.date_col, res.grains[c])
        v = pd.to_numeric(d[c], errors="coerce")
        row = {"channel": ctx.channel_label[c], "column": c, "unit": ctx.channel_unit[c],
               "recorded_at": " × ".join(([ctx.date_col] if ctx.date_col else []) + res.grains[c]) or "row",
               "total": float(v.sum()), "missing_pct": 100 * float(v.isna().mean())}
        if per is not None:
            s = per[c]
            row["avg_per_period"] = float(s.mean())
            row["active_periods_pct"] = 100 * float((s > 0).mean())
            half = len(s) // 2
            a, b = s.iloc[:half].mean(), s.iloc[half:].mean()
            row["change_2nd_vs_1st_half_pct"] = 100 * (b - a) / a if a else np.nan
            r, p = S.corr_with_p(s.to_numpy(float), per[ctx.target].to_numpy(float), "pearson")
            row["corr_with_target_per_period"] = r
            row["corr_p"] = p
        rows.append(row)
    t = pd.DataFrame(rows)
    for unit, cols in res.unit_groups.items():
        m = t.column.isin(cols)
        tot = t.loc[m, "total"].sum()
        t.loc[m, "share_within_unit_pct"] = 100 * t.loc[m, "total"] / tot if tot > 0 else np.nan
    if effects is not None and len(effects):
        e = effects.set_index("column")
        for col in ("elasticity_at_mean", "association_share_pct", "p_value", "standardized_effect"):
            if col in e.columns:
                t[col] = t.column.map(e[col])
    if mmm and mmm.get("status") == "ok":
        mt = mmm["table"].set_index("channel")
        for col in ("model_contribution", "contribution_share_pct", "model_roi", "model_roas", "outcome_per_unit_spend"):
            if col in mt.columns:
                t[col] = t.column.map(mt[col])
    # ranking
    if "model_roas" in t.columns and t.model_roas.notna().any():
        basis, key = "model-estimated ROAS (marketing-mix model)", "model_roas"
    elif "outcome_per_unit_spend" in t.columns and t.outcome_per_unit_spend.notna().any():
        basis, key = "model-estimated outcome per unit of spend", "outcome_per_unit_spend"
    elif "association_share_pct" in t.columns and t.association_share_pct.notna().any():
        basis, key = "statistical association share (driver model)", "association_share_pct"
    elif "corr_with_target_per_period" in t.columns:
        basis, key = "correlation with the outcome per period", "corr_with_target_per_period"
    else:
        basis, key = "", None
    res.ranking_basis = basis
    if key:
        t["rank"] = t[key].rank(ascending=False, method="min")
        t = t.sort_values("rank").reset_index(drop=True)
    res.table = t
    if per is not None:
        long = per.melt(id_vars=[ctx.date_col], value_vars=ctx.channels, var_name="column", value_name="value")
        long["channel"] = long.column.map(ctx.channel_label)
        long["unit"] = long.column.map(ctx.channel_unit)
        res.trends = long
    for role in ("brand", "geography", "product"):
        for dim in ctx.dims.get(role, []):
            if not 2 <= ctx.df[dim].nunique() <= 30:
                continue
            parts = []
            for c in ctx.channels:
                g = sorted(set(value_grain(ctx.df, c, ctx.date_col, ctx.dim_cols)) | {dim}, key=ctx.dim_cols.index)
                d = dedup_frame(ctx.df, c, ctx.date_col, g)
                s = d.groupby(dim, observed=True)[c].sum().rename("total").reset_index()
                s["channel"], s["column"], s["unit"] = ctx.channel_label[c], c, ctx.channel_unit[c]
                parts.append(s)
            res.breakdowns[dim] = pd.concat(parts, ignore_index=True)
    res.rates = funnel_rates(ctx, res.grains)
    res.methodology = [
        "Totals count each recorded value once: a channel recorded per " + ", ".join(
            sorted({" × ".join(g) or "period" for g in res.grains.values()})) + " is not multiplied by finer rows.",
        "Shares are computed only among channels measured in the same unit.",
        "Correlation uses per-period totals of the channel and the target (Pearson).",
        f"Ranking basis: {basis}." if basis else "No ranking basis is available.",
    ]
    if ctx.channel_kind == "exposure":
        res.methodology.append("Exposure channels are compared by statistical association, never by money: no "
                               "channel-wise cost exists in this dataset.")
        res.warnings.append("Contribution here is a statistical association share in %, not revenue attributed to "
                            "a channel.")
    if len(res.unit_groups) > 1:
        res.warnings.append("Channels use different units (" + ", ".join(u for u in res.unit_groups)
                            + "); volumes are only comparable within a unit.")
    return res


def funnel_rates(ctx: ModelContext, grains: dict) -> Optional[pd.DataFrame]:
    """Click-through rate (clicks / impressions) and views or reach ratios for the same channel, only
    where both measures exist for that channel. Ratios of recorded totals, not model estimates."""
    by_ch: dict = {}
    for c in ctx.channels:
        by_ch.setdefault(ctx.mapping.columns[c].channel, {})[ctx.channel_role.get(c)] = c
    rows = []
    for ch, roles in by_ch.items():
        for num, den, name in (("clicks", "impressions", "Click-through rate (clicks / impressions)"),
                               ("clicks", "reach", "Clicks per person reached"),
                               ("visits", "clicks", "Visits per click")):
            if num in roles and den in roles:
                a = dedup_frame(ctx.df, roles[num], ctx.date_col, grains[roles[num]])[roles[num]].sum()
                b = dedup_frame(ctx.df, roles[den], ctx.date_col, grains[roles[den]])[roles[den]].sum()
                rows.append({"channel": ch, "metric": name, "numerator": float(a), "denominator": float(b),
                             "rate_pct": 100 * a / b if b else np.nan})
    conv = [c for c in ctx.mapping.cols("conversions") if c == ctx.target]
    if conv:
        for c in ctx.channels:
            if ctx.channel_role.get(c) == "clicks":
                b = dedup_frame(ctx.df, c, ctx.date_col, grains[c])[c].sum()
                a = pd.to_numeric(ctx.df[conv[0]], errors="coerce").sum()
                rows.append({"channel": ctx.channel_label[c], "metric": f"{conv[0]} per click (all conversions, not "
                             "attributed to this channel)", "numerator": float(a), "denominator": float(b),
                             "rate_pct": 100 * a / b if b else np.nan})
    return pd.DataFrame(rows) if rows else None
