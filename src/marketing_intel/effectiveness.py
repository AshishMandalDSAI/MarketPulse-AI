"""MODEL 04 -- Marketing Effectiveness Model (PART 02).

Works with what the dataset actually contains:

* spend-based: the driver response model on channel spend plus the platform's
  existing marketing-mix model (adstock / saturation, ``src.marketing``) for
  model-estimated contribution and ROI / ROAS. ROI / ROAS only when the target
  is a monetary outcome (revenue / profit); for a unit outcome the result is
  "outcome per unit of spend";
* exposure-based: exposure-versus-outcome relationships, regression with HC3
  robust errors, elasticities, variable importance and a statistical
  association share -- never ROI, ROAS or monetary attribution;
* general: no marketing effectiveness (explained), but the same machinery on
  price, promotion and other business drivers.

Every estimate is an association in observational data, not proof of cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.marketing_intel import stats as S
from src.marketing_intel.context import ModelContext, period_totals
from src.marketing_intel.response import ResponseModel, fit_response, standardized_effects

MAX_SEGMENT_LEVELS = 15
MIN_PERIODS_FOR_MMM = 20
MIN_SEGMENT_PERIODS = 8          # segment / time-window correlations need at least this many periods


@dataclass
class EffectivenessResult:
    status: str = "unavailable"        # ok | unavailable
    message: str = ""
    mode: str = ""
    target: str = ""
    model: Optional[ResponseModel] = None
    effects: Optional[pd.DataFrame] = None
    correlations: Optional[pd.DataFrame] = None
    by_segment: dict = field(default_factory=dict)       # dim -> DataFrame (level x driver correlation)
    by_time: Optional[pd.DataFrame] = None
    promotions: Optional[pd.DataFrame] = None
    mmm: dict = field(default_factory=dict)               # spend mode only
    methodology: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def run_effectiveness(ctx: ModelContext, drivers: Optional[list] = None, *, fixed_effects: Optional[list] = None,
                      trend: bool = True, with_mmm: bool = True) -> EffectivenessResult:
    res = EffectivenessResult(mode=ctx.mode.mode, target=ctx.target)
    model = fit_response(ctx, drivers, fixed_effects=fixed_effects, trend=trend)
    res.model = model
    res.warnings += model.notes + ctx.notes
    if model.fit is None:
        res.message = model.quality_reason or "The model could not be fitted."
        return res
    res.status = "ok"
    res.effects = standardized_effects(model, ctx)
    res.correlations = correlations(ctx, [d.column for d in model.drivers])
    res.by_segment = segment_effectiveness(ctx, [d.column for d in model.drivers if d.kind == "channel"])
    res.by_time = time_effectiveness(ctx, [d.column for d in model.drivers if d.kind == "channel"])
    res.promotions = promotion_effects(ctx, model)
    if model.quality == "weak":
        res.warnings.append("The model explains the held-out periods poorly (" + model.quality_reason
                            + "). Treat the effects below as weak indications only.")
    high_vif = [d.label for d in model.drivers if model.vif.get(d.column, 0) > 10]
    if high_vif:
        res.warnings.append("Strong overlap between drivers (VIF > 10): " + ", ".join(high_vif)
                            + ". Their individual effects are uncertain; their combined effect is more reliable.")
    neg = res.effects[(res.effects.kind == "channel") & (res.effects.coef < 0) & res.effects.significant_5pct] \
        if len(res.effects) else []
    if len(neg):
        res.warnings.append("Negative association for " + ", ".join(neg.driver)
                            + ": this usually reflects timing or targeting (e.g. activity raised when sales are weak), "
                              "not that the channel reduces the outcome.")
    if ctx.mode.mode == "spend" and with_mmm:
        res.mmm = spend_mmm(ctx)
    res.methodology = methodology(ctx, model)
    res.assumptions = [
        "Channel effects show diminishing returns: each channel enters as log(1 + value).",
        "Other drivers enter linearly; their effect does not depend on the level of other drivers.",
        "Effects are averaged over the whole period; seasonal changes in effectiveness are only partly captured "
        "(see effectiveness over time).",
        "Observational data: an association can come from timing, targeting or omitted factors, so it is not "
        "proof that a driver causes the outcome.",
    ]
    if ctx.mode.mode == "exposure":
        res.assumptions.append("No channel-wise cost is available, so effects are per unit of exposure and are not "
                               "comparable to money: no ROI, ROAS or monetary contribution is computed.")
    return res


def methodology(ctx: ModelContext, model: ResponseModel) -> list:
    out = [f"Target: {ctx.target} ({ctx.mapping.columns[ctx.target].label}).",
           f"Linear regression of the target on {len(model.drivers)} driver(s) with HC3 heteroskedasticity-robust "
           "standard errors; 95% confidence intervals and two-sided p-values.",
           f"Validation: {model.split}. Quality: {model.quality} -- {model.quality_reason}."]
    if model.fe_dims:
        out.append("Segment fixed effects for " + ", ".join(model.fe_dims)
                   + " absorb level differences between segments, so effects are estimated within segments.")
    if model.trend:
        out.append("A linear time trend absorbs steady growth or decline.")
    out.append("Correlation p-values are also shown adjusted for multiple testing (Benjamini-Hochberg).")
    out.append(f"Segment and time-window correlations are shown only with at least {MIN_SEGMENT_PERIODS} periods.")
    out.append("Variable importance: drop in holdout R² when one driver is shuffled (10 repeats, fixed seed).")
    out.append("Multicollinearity: variance inflation factor (VIF) of each driver; above 10 means overlapping drivers.")
    if ctx.channels:
        out.append("Association share: each channel's average modelled effect as a share of the total positive "
                   "channel effect. It is a statistical decomposition, not revenue attribution.")
    return out


def correlations(ctx: ModelContext, cols: list) -> pd.DataFrame:
    """Pearson and Spearman correlation of each driver with the target (rows), plus per-period totals
    for channels in a panel."""
    y = pd.to_numeric(ctx.df[ctx.target], errors="coerce").to_numpy(float)
    rows = []
    per = None
    chans = [c for c in cols if c in ctx.channels]
    if ctx.date_col and chans:
        per = period_totals(ctx, chans)
    for c in cols:
        x = pd.to_numeric(ctx.df[c], errors="coerce").to_numpy(float)
        r, p = S.corr_with_p(x, y, "pearson")
        rs, ps = S.corr_with_p(x, y, "spearman")
        row = {"driver": c, "pearson_r": r, "pearson_p": p, "spearman_rho": rs, "spearman_p": ps}
        if per is not None and c in chans and c in per.columns:
            rp, pp = S.corr_with_p(per[c].to_numpy(float), per[ctx.target].to_numpy(float), "pearson")
            row.update(period_total_r=rp, period_total_p=pp)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["pearson_p_bh"] = S.bh_adjust(out.pearson_p)
        out["spearman_p_bh"] = S.bh_adjust(out.spearman_p)
    return out


def segment_effectiveness(ctx: ModelContext, channels: list) -> dict:
    """Per brand / geography / product level: Spearman correlation between each channel's per-period
    total and the target's per-period total within that level (association only)."""
    out = {}
    if not ctx.date_col or not channels:
        return out
    for role in ("brand", "geography", "product", "sku"):
        for dim in ctx.dims.get(role, []):
            levels = ctx.df[dim].dropna().unique()
            if not 2 <= len(levels) <= MAX_SEGMENT_LEVELS:
                continue
            per = period_totals(ctx, channels, by=[dim])
            rows = []
            for lv, g in per.groupby(dim, observed=True):
                row = {dim: lv, "periods": len(g)}
                for c in channels:      # minimum-sample gate: no estimate from fewer than MIN_SEGMENT_PERIODS
                    row[c] = (S.corr_with_p(g[c].to_numpy(float), g[ctx.target].to_numpy(float), "spearman")[0]
                              if len(g) >= MIN_SEGMENT_PERIODS else np.nan)
                rows.append(row)
            out[dim] = pd.DataFrame(rows)
    return out


def time_effectiveness(ctx: ModelContext, channels: list) -> Optional[pd.DataFrame]:
    """Spearman correlation between channel and target per-period totals in each year (or half of the
    data when it covers less than two years)."""
    if not ctx.date_col or not channels:
        return None
    per = period_totals(ctx, channels)
    years = per[ctx.date_col].dt.year
    if years.nunique() >= 2 and years.value_counts().min() >= 8:
        per["window"] = years.astype(str)
    else:
        mid = per[ctx.date_col].iloc[len(per) // 2]
        per["window"] = np.where(per[ctx.date_col] < mid, "First half", "Second half")
    rows = []
    for w, g in per.groupby("window", sort=True):
        row = {"window": w, "periods": len(g)}
        for c in channels:
            row[c] = (S.corr_with_p(g[c].to_numpy(float), g[ctx.target].to_numpy(float), "spearman")[0]
                      if len(g) >= MIN_SEGMENT_PERIODS else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def promotion_effects(ctx: ModelContext, model: ResponseModel) -> Optional[pd.DataFrame]:
    """For each 0/1 promotion flag: average target with and without it, the raw difference, a Welch
    t-test, and the model's coefficient (difference after accounting for the other drivers)."""
    flags = [c for c in ctx.business.get("promotion", []) if set(pd.to_numeric(ctx.df[c], errors="coerce").dropna().unique()) <= {0, 1}]
    if not flags:
        return None
    coef = dict(zip(model.fit.names, model.fit.coef)) if model.fit is not None else {}
    pv = dict(zip(model.fit.names, model.fit.p)) if model.fit is not None else {}
    y = pd.to_numeric(ctx.df[ctx.target], errors="coerce")
    rows = []
    for c in flags:
        f = pd.to_numeric(ctx.df[c], errors="coerce")
        on, off = y[f == 1].dropna(), y[f == 0].dropna()
        t, p = S.welch_test(on.to_numpy(), off.to_numpy())
        rows.append({"promotion": c, "share_of_rows_on_pct": 100 * float((f == 1).mean()),
                     "avg_target_on": float(on.mean()) if len(on) else np.nan,
                     "avg_target_off": float(off.mean()) if len(off) else np.nan,
                     "raw_difference_pct": 100 * (on.mean() - off.mean()) / off.mean() if len(off) and off.mean() else np.nan,
                     "welch_p": p, "model_coef": coef.get(c, np.nan), "model_p": pv.get(c, np.nan)})
    return pd.DataFrame(rows)


def spend_mmm(ctx: ModelContext) -> dict:
    """Existing marketing-mix model on per-period totals (spend mode only)."""
    from src.marketing.contribution import compute_contributions
    from src.prescriptive.engine import fit_response_model
    out: dict = {"status": "unavailable"}
    if not ctx.date_col:
        out["message"] = "The marketing-mix model needs a date column."
        return out
    per = period_totals(ctx, ctx.channels)
    if len(per) < MIN_PERIODS_FOR_MMM:
        out["message"] = f"The marketing-mix model needs at least {MIN_PERIODS_FOR_MMM} periods; this data has {len(per)}."
        return out
    per = per.dropna()
    try:
        mmm = fit_response_model(per, list(ctx.channels), ctx.target, ctx.date_col, [])
    except Exception as exc:  # reported, never hidden
        out["message"] = f"The marketing-mix model could not be fitted ({type(exc).__name__}: {exc})."
        return out
    if mmm._model is None:
        out["message"] = "The marketing-mix model could not be fitted: " + ("; ".join(mmm.limitations) or "not enough usable data.")
        return out
    contrib = compute_contributions(per, mmm)
    rows = []
    for c in contrib:
        spend = float(per[c.channel].sum())
        row = {"channel": c.channel, "total_spend": spend, "model_contribution": c.contribution,
               "contribution_share_pct": 100 * c.contribution_share if c.contribution_share <= 1 else c.contribution_share}
        if ctx.target_is_monetary:
            row["model_roi"] = (c.contribution - spend) / spend if spend > 0 else np.nan
            row["model_roas"] = c.contribution / spend if spend > 0 else np.nan
        else:
            row["outcome_per_unit_spend"] = c.contribution / spend if spend > 0 else np.nan
        rows.append(row)
    out.update(status="ok", mmm=mmm, period_data=per, table=pd.DataFrame(rows),
               selected_model=mmm.selected_model, test_metrics=dict(mmm.test_metrics or {}),
               train_metrics=dict(mmm.train_metrics or {}), n_train=mmm.n_train, n_test=mmm.n_test,
               roi_available=ctx.target_is_monetary,
               note=("ROI = (model contribution - spend) / spend; ROAS = model contribution / spend."
                     if ctx.target_is_monetary else
                     f"The target '{ctx.target}' is not money, so ROI / ROAS are not defined; the table shows the "
                     "model-estimated outcome per unit of spend instead."))
    return out
