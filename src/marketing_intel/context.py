"""Model context shared by the four PART 02 marketing models.

``build_context`` turns a table plus its confirmed ``RoleMapping`` and a chosen
target into everything the models need: parsed dates, the channel columns of
the active mode (advertising spend in spend mode, exposure in exposure mode),
business drivers, dimensions and the columns kept out of models (other
outcomes, derived ratios, aggregates, identifiers) with the reason.

It also works out the *grain* of each channel: in a Week x Geo x Brand x SKU
panel an exposure column is often recorded once per Week x Geo x Brand and
repeated on every SKU row. Totals then count each recorded value once instead
of multiplying it by the number of SKUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from src.data_engine.role_engine import (EXPOSURE_ROLES, ROLE_INFO, ModeResult, RoleMapping, _parse_dates,
                                         detect_mode)

DIM_ROLES = ("brand", "geography", "product", "sku", "channel_dimension")
BUSINESS_ROLES = ("price", "promotion", "discount", "trade_spend", "distribution", "competitor", "external",
                  "quality_score")
EXPOSURE_UNIT = {"impressions": "impressions", "reach": "people reached", "clicks": "clicks", "views": "views",
                 "visits": "visits", "engagement": "engagements", "readership": "readers",
                 "listenership": "listeners", "grps": "rating points", "audience": "audience"}
MAX_FE_LEVELS = 60


@dataclass
class ModelContext:
    df: pd.DataFrame
    mapping: RoleMapping
    mode: ModeResult
    target: str
    date_col: Optional[str]
    channels: list
    channel_kind: Optional[str]                 # "spend" | "exposure" | None
    channel_role: dict = field(default_factory=dict)    # col -> role
    channel_unit: dict = field(default_factory=dict)    # col -> unit label ("impressions", currency code...)
    channel_label: dict = field(default_factory=dict)   # col -> display label
    dims: dict = field(default_factory=dict)            # role -> [cols]
    business: dict = field(default_factory=dict)        # role -> [cols]
    other_numeric: list = field(default_factory=list)
    excluded: dict = field(default_factory=dict)        # col -> reason
    target_role: str = ""
    target_is_monetary: bool = False
    frequency: str = ""
    n_periods: int = 0
    is_panel: bool = False
    notes: list = field(default_factory=list)

    @property
    def dim_cols(self) -> list:
        return [c for cols in self.dims.values() for c in cols]

    def unit_groups(self) -> dict:
        """Channels grouped by unit -- only channels in the same unit may be compared or summed."""
        g: dict[str, list] = {}
        for c in self.channels:
            g.setdefault(self.channel_unit.get(c, ""), []).append(c)
        return g

    def default_drivers(self) -> list:
        out = list(self.channels)
        for role in BUSINESS_ROLES:
            out += self.business.get(role, [])
        return list(dict.fromkeys(out))


def parse_date_column(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return _parse_dates(s.astype(str))


def infer_frequency(dates: pd.Series) -> str:
    u = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    if len(u) < 3:
        return "irregular"
    d = u.diff().dt.days.dropna().median()
    if d <= 1.5:
        return "daily"
    if 6 <= d <= 8:
        return "weekly"
    if 27 <= d <= 32:
        return "monthly"
    if 85 <= d <= 95:
        return "quarterly"
    return f"every ~{int(d)} days"


def build_context(df: pd.DataFrame, mapping: RoleMapping, target: str, currency_code: Optional[str] = None) -> ModelContext:
    """Prepare the shared model inputs. Raises ``ValueError`` with an actionable message when the
    target is unusable."""
    if target not in df.columns:
        raise ValueError(f"The target '{target}' is not in the dataset.")
    r = mapping.columns.get(target)
    if r is None or r.group != "outcome":
        raise ValueError(f"'{target}' is not mapped as a business outcome; change its role first.")
    work = df.copy()
    mode = detect_mode(mapping)
    date_col = mapping.date_col
    notes: list[str] = []
    if date_col:
        parsed = parse_date_column(work[date_col])
        if parsed.notna().mean() < 0.9:
            notes.append(f"Only {100 * parsed.notna().mean():.0f}% of '{date_col}' parse as dates; time-based "
                         "results are not available.")
            date_col = None
        else:
            work[date_col] = parsed
    for c, cr in mapping.columns.items():
        if cr.role in ROLE_INFO and ROLE_INFO[cr.role][0] in ("outcome", "spend", "spend_total", "exposure", "business",
                                                              "measure"):
            if c in work.columns and not pd.api.types.is_numeric_dtype(work[c]):
                work[c] = pd.to_numeric(work[c], errors="coerce")
    if mode.mode == "spend":
        channels, kind = list(mode.spend_channels), "spend"
    elif mode.mode == "exposure":
        channels, kind = list(mode.exposure_columns), "exposure"
    else:
        channels, kind = [], None
    channels = [c for c in channels if c != target]
    cur = currency_code or "currency units"
    ch_role = {c: mapping.columns[c].role for c in channels}
    ch_unit = {c: (cur if kind == "spend" else EXPOSURE_UNIT.get(ch_role[c], ch_role[c])) for c in channels}
    ch_label = {c: f"{mapping.columns[c].channel or c}" for c in channels}
    # disambiguate duplicate labels (e.g. two Facebook exposure measures)
    seen: dict[str, int] = {}
    for c in channels:
        seen[ch_label[c]] = seen.get(ch_label[c], 0) + 1
    for c in channels:
        if seen[ch_label[c]] > 1:
            ch_label[c] = f"{ch_label[c]} ({ch_unit[c]})"
    dims = {role: [c for c in mapping.cols(role) if c in work.columns] for role in DIM_ROLES}
    dims = {k: v for k, v in dims.items() if v}
    business = {role: [c for c in mapping.cols(role) if c in work.columns] for role in BUSINESS_ROLES}
    business = {k: v for k, v in business.items() if v}
    if kind == "exposure" and mapping.cols("ad_spend"):
        pass  # cannot happen: ad_spend makes the mode spend-based
    excluded = {}
    for c, cr in mapping.columns.items():
        if c == target:
            continue
        if cr.group == "outcome":
            excluded[c] = "another outcome column (using it to explain the target would leak the answer)"
        elif cr.group == "derived":
            excluded[c] = "a derived ratio / share / growth of other columns"
        elif cr.role == "total_marketing_spend":
            excluded[c] = "an aggregate of channel spend (would double-count the channels)"
        elif cr.group == "identifier":
            excluded[c] = "an identifier"
        elif cr.group in ("ignore", "text"):
            excluded[c] = "ignored / free text"
        elif cr.group == "exposure" and kind == "spend":
            excluded[c] = "exposure metric (spend channels drive the spend-based models; shown as a supporting metric)"
    other_numeric = [c for c in mapping.cols("measure") if c in work.columns]
    n_periods = int(work[date_col].nunique()) if date_col else 0
    is_panel = bool(date_col and len(work) > n_periods * 1.05)
    ctx = ModelContext(df=work, mapping=mapping, mode=mode, target=target, date_col=date_col, channels=channels,
                       channel_kind=kind, channel_role=ch_role, channel_unit=ch_unit, channel_label=ch_label,
                       dims=dims, business=business, other_numeric=other_numeric, excluded=excluded,
                       target_role=r.role, target_is_monetary=r.role in ("revenue", "profit"),
                       frequency=infer_frequency(work[date_col]) if date_col else "", n_periods=n_periods,
                       is_panel=is_panel, notes=notes)
    return ctx


# --------------------------------------------------------------------------- grain & period totals
def value_grain(df: pd.DataFrame, col: str, date_col: Optional[str], dims: list) -> list:
    """Smallest set of dimensions at which ``col`` is recorded: the column is constant within every
    (date + those dimensions) group. Returns all dimensions when no smaller set fits."""
    if not date_col:
        return list(dims)
    base = [date_col]
    for k in range(0, len(dims) + 1):
        for combo in combinations(dims, k):
            keys = base + list(combo)
            nun = df.groupby(keys, dropna=False, observed=True)[col].nunique(dropna=False)
            if (nun <= 1).all():
                return list(combo)
    return list(dims)


def dedup_frame(df: pd.DataFrame, col: str, date_col: Optional[str], grain: list) -> pd.DataFrame:
    """One row per recorded value of ``col`` (date + grain)."""
    if not date_col:
        return df[[col] + grain]
    return df.drop_duplicates(subset=[date_col] + grain)[[date_col] + grain + [col]]


def period_totals(ctx: ModelContext, cols: list, by: Optional[list] = None) -> pd.DataFrame:
    """Per-period totals of the target and the given columns, optionally per extra dimension(s).
    Channel totals count each recorded value once (see ``value_grain``)."""
    if not ctx.date_col:
        raise ValueError("A date column is needed for per-period totals.")
    by = list(by or [])
    keys = [ctx.date_col] + by
    out = ctx.df.groupby(keys, observed=True)[ctx.target].sum(min_count=1).rename(ctx.target).to_frame()
    for c in cols:
        if c == ctx.target:
            continue
        grain = value_grain(ctx.df, c, ctx.date_col, ctx.dim_cols)
        g = sorted(set(grain) | set(by), key=(ctx.dim_cols + by).index)
        d = dedup_frame(ctx.df, c, ctx.date_col, g)
        s = d.groupby(keys, observed=True)[c].sum(min_count=1)
        out = out.join(s, how="left")
    return out.reset_index().sort_values(keys).reset_index(drop=True)


def channel_grains(ctx: ModelContext) -> dict:
    return {c: value_grain(ctx.df, c, ctx.date_col, ctx.dim_cols) for c in ctx.channels}
