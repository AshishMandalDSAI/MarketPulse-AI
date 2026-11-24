"""Universal business-role engine (PART 02).

Looks at every column of an uploaded table and suggests what it means in
business terms -- an outcome (revenue, units, orders, leads...), channel-wise
advertising spend, marketing exposure (impressions, reach, clicks,
readership...), another business variable (price, promotion, trade spend,
distribution, competitor, external factors), a dimension (brand, product, SKU,
geography) or time.

Every suggestion carries a confidence score, the evidence behind it,
alternative roles and ambiguity warnings. Signals used:

* the column name, split into words and matched against generic business
  vocabulary (``role_lexicon``) -- never against a fixed list of column names;
* the data type and the actual values (numeric share, whole numbers, 0/1
  flags, 0-100 bounds, negatives, uniqueness, dates that parse);
* relationships between columns (value = units x price, a spend column that is
  the sum of other spend columns, duplicated columns).

The engine also decides the analytics mode (spend-based, exposure-based or
general business analytics) and stores the user's confirmed mapping as a
versioned JSON file. Business logic only: no Streamlit here.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.data_engine.role_lexicon import (CHANNEL_WORDS, SPEND_WORDS, TRADE_WORDS, channel_label, tokens)

SCHEMA_VERSION = 1
MAPPING_FILE = "role_mapping.json"

# --------------------------------------------------------------------------- role catalogue
# role -> (group, label, unit kind)
ROLE_INFO: dict[str, tuple[str, str, str]] = {
    "revenue": ("outcome", "Revenue / sales value", "currency"),
    "sales_units": ("outcome", "Sales units / quantity", "units"),
    "orders": ("outcome", "Orders / transactions", "orders"),
    "conversions": ("outcome", "Conversions", "conversions"),
    "leads": ("outcome", "Leads", "leads"),
    "profit": ("outcome", "Profit", "currency"),
    "customers": ("outcome", "Customers", "customers"),
    "ad_spend": ("spend", "Advertising spend for one channel", "currency"),
    "total_marketing_spend": ("spend_total", "Total / aggregated marketing spend (not channel-wise)", "currency"),
    "impressions": ("exposure", "Impressions", "impressions"),
    "reach": ("exposure", "Reach", "people reached"),
    "clicks": ("exposure", "Clicks", "clicks"),
    "views": ("exposure", "Views", "views"),
    "visits": ("exposure", "Website visits / traffic", "visits"),
    "engagement": ("exposure", "Engagement", "engagements"),
    "readership": ("exposure", "Readership", "readers"),
    "listenership": ("exposure", "Listenership", "listeners"),
    "grps": ("exposure", "Rating points (GRP / TRP)", "rating points"),
    "audience": ("exposure", "Audience exposure", "audience"),
    "price": ("business", "Price", "currency per unit"),
    "discount": ("business", "Discount", "discount"),
    "promotion": ("business", "Promotion (flag or intensity)", "promotion"),
    "trade_spend": ("business", "Trade / promotion spend (not advertising)", "currency"),
    "distribution": ("business", "Distribution / availability", "distribution"),
    "competitor": ("business", "Competitor variable", "as recorded"),
    "external": ("business", "External / seasonal factor", "index"),
    # Upgrade PART 06: creative / content / quality scores (e.g. "FB_Banner_Content_Score") are
    # neither money nor exposure; they are business drivers and must never become channel spend.
    "quality_score": ("business", "Creative / content / quality score (not spend, not exposure)", "score"),
    "cost": ("business", "Non-marketing cost", "currency"),
    "brand": ("dimension", "Brand", ""),
    "product": ("dimension", "Product / category", ""),
    "sku": ("dimension", "SKU / item code", ""),
    "geography": ("dimension", "Geography / region / store", ""),
    "channel_dimension": ("dimension", "Marketing channel name (category)", ""),
    "category": ("dimension", "Other category", ""),
    "date": ("time", "Date / period", ""),
    "time_part": ("time", "Calendar part (year, month, week number)", ""),
    "identifier": ("identifier", "Identifier", ""),
    "derived": ("derived", "Derived ratio / share / growth (kept out of models)", ""),
    "measure": ("measure", "Other numeric measure", ""),
    "text": ("text", "Free text", ""),
    "ignore": ("ignore", "Ignore this column", ""),
}
ROLES = list(ROLE_INFO)
OUTCOME_ROLES = [r for r, v in ROLE_INFO.items() if v[0] == "outcome"]
EXPOSURE_ROLES = [r for r, v in ROLE_INFO.items() if v[0] == "exposure"]
MONETARY_ROLES = {"revenue", "profit", "ad_spend", "total_marketing_spend", "trade_spend", "cost", "price"}
NUMERIC_ROLES = set(OUTCOME_ROLES) | set(EXPOSURE_ROLES) | {
    "ad_spend", "total_marketing_spend", "price", "discount", "promotion", "trade_spend", "distribution",
    "competitor", "external", "cost", "measure", "derived"}
DIMENSION_ROLES = {"brand", "product", "sku", "geography", "channel_dimension", "category"}
OUTCOME_PRIORITY = ["revenue", "sales_units", "profit", "orders", "conversions", "leads", "customers"]

MODE_LABELS = {
    "spend": "Spend-Based Marketing Analytics",
    "exposure": "Exposure-Based Marketing Analytics",
    "general": "General Business Analytics",
}

# --------------------------------------------------------------------------- vocabulary
_EXPOSURE_WORDS = {
    "impressions": "impressions", "impression": "impressions", "impr": "impressions", "imps": "impressions",
    "imp": "impressions", "reach": "reach", "clicks": "clicks", "click": "clicks", "views": "views",
    "view": "views", "viewership": "views", "viewers": "views", "visits": "visits", "visit": "visits",
    "sessions": "visits", "traffic": "visits", "pageviews": "visits", "engagement": "engagement",
    "engagements": "engagement", "likes": "engagement", "shares": "engagement", "readership": "readership",
    "readers": "readership", "circulation": "readership", "listenership": "listenership",
    "listeners": "listenership", "grp": "grps", "grps": "grps", "trp": "grps", "trps": "grps",
    "audience": "audience", "exposure": "audience", "exposures": "audience",
}
_DERIVED_WORDS = {"share", "pct", "percent", "percentage", "ratio", "rate", "roas", "roi", "aov", "cpa", "cpc",
                  "cpm", "cpl", "ctr", "cvr", "growth", "wow", "mom", "yoy", "qoq", "per", "basket", "lag",
                  "lagged", "rolling", "cumulative", "cum", "ma", "zscore", "normalized", "scaled", "log"}
_QUALITY_WORDS = {"score", "scores", "rating", "ratings", "quality", "sentiment", "nps", "csat", "stars"}
_AD_WORDS = {"ad", "ads", "advert", "adverts", "advertising", "media", "marketing", "campaign", "campaigns"}
_TOTAL_WORDS = {"total", "overall", "all", "combined", "aggregate", "sum", "gross"}
_ID_WORDS = {"id", "ids", "uuid", "guid", "key"}
_DATE_WORDS = {"date", "dt", "day", "week", "wk", "month", "period", "time", "timestamp", "datetime",
               "yearweek", "yearmonth", "yyyymm", "weekstart", "weekending", "start", "ending", "end"}
_TIMEPART_SETS = [{"year"}, {"month"}, {"quarter"}, {"qtr"}, {"weekofyear"}, {"week", "of", "year"},
                  {"day", "of", "week"}, {"dayofweek"}, {"dow"}, {"week", "number"}, {"week", "no"},
                  {"month", "number"}, {"fiscal", "year"}, {"fy"}, {"year", "month"}]
_PRICE_WORDS = {"price", "prices", "mrp", "msrp", "asp", "rsp", "arp", "tariff", "ppu", "pricing"}
_DISCOUNT_WORDS = {"discount", "discounts", "disc", "markdown", "markdowns"}
_PROMO_WORDS = {"promo", "promos", "promotion", "promotions", "promotional", "tpr", "deal", "deals", "offer",
                "offers", "coupon", "coupons", "feature", "featured", "display", "bogo", "flyer", "endcap",
                "gondola", "leaflet"}
_DIST_WORDS = {"distribution", "wd", "nd", "acv", "tdp", "nos", "stores", "outlets", "doors", "coverage",
               "listings", "availability", "oos", "facings"}
_COMP_WORDS = {"competitor", "competitors", "competition", "comp", "rival", "rivals", "competitive"}
_EXT_WORDS = {"cpi", "inflation", "gdp", "unemployment", "festival", "festive", "holiday", "holidays", "rainfall",
              "rain", "temperature", "temp", "weather", "season", "seasonal", "seasonality", "economic",
              "economy", "confidence", "monsoon", "school", "macro", "fuel", "interest", "wpi", "iip", "covid",
              "lockdown", "event", "diwali", "christmas", "ramadan", "eid"}
_BRAND_WORDS = {"brand", "brands", "manufacturer", "maker", "make"}
_SKU_WORDS = {"sku", "skus", "upc", "ean", "barcode", "article", "gtin", "itemcode", "pack", "packsize"}
_PRODUCT_WORDS = {"product", "products", "item", "items", "category", "categories", "subcategory", "variant",
                  "line", "range", "segment", "family", "department", "dept"}
_GEO_WORDS = {"geo", "geography", "region", "regions", "state", "states", "city", "cities", "country",
              "countries", "market", "markets", "zone", "zones", "territory", "area", "district", "province",
              "location", "dma", "pincode", "zip", "postcode", "cluster", "store", "branch", "outlet", "town",
              "metro", "circle"}
_CHANNEL_DIM_WORDS = {"channel", "channels", "campaign", "medium", "source", "platform", "media", "network",
                      "publisher", "placement", "campaigntype"}
_CURRENCY_WORDS = {"value", "amount", "amt", "val", "inr", "usd", "eur", "gbp", "rs", "rupees", "dollars",
                   "revenue", "net", "gross", "total", "turnover"}


# --------------------------------------------------------------------------- data classes
@dataclass
class ColumnRole:
    """Suggested (or confirmed) business role of one column."""
    column: str
    role: str
    group: str
    confidence: float
    channel: Optional[str] = None           # advertising channel label for spend / exposure columns
    unit: str = ""
    evidence: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)   # [{"role", "label", "confidence", "why"}]
    warnings: list = field(default_factory=list)
    dtype: str = ""
    stats: dict = field(default_factory=dict)
    user_set: bool = False                  # True once a person chose / confirmed this role

    @property
    def label(self) -> str:
        return ROLE_INFO.get(self.role, ("", self.role, ""))[1]


@dataclass
class ModeResult:
    mode: str                                # spend | exposure | general
    label: str
    evidence: list = field(default_factory=list)
    spend_channels: list = field(default_factory=list)
    exposure_channels: dict = field(default_factory=dict)   # exposure role -> [columns]
    outcomes: list = field(default_factory=list)
    has_outcome: bool = False
    notes: list = field(default_factory=list)

    @property
    def exposure_columns(self) -> list:
        return [c for cols in self.exposure_channels.values() for c in cols]


@dataclass
class RoleMapping:
    """All column roles of one dataset, plus the chosen analysis targets."""
    dataset_id: str = ""
    dataset_name: str = ""
    fingerprint: str = ""
    schema_version: int = SCHEMA_VERSION
    version: int = 0                         # increments on every confirmed save
    confirmed: bool = False
    confirmed_at: Optional[str] = None
    source: str = "suggested"                # suggested | confirmed | demo-preset
    columns: dict = field(default_factory=dict)     # column -> ColumnRole
    target_candidates: list = field(default_factory=list)
    targets: list = field(default_factory=list)      # confirmed analysis targets
    warnings: list = field(default_factory=list)
    history: list = field(default_factory=list)
    n_rows: int = 0

    # ---- convenience views
    def role_of(self, col: str) -> Optional[str]:
        r = self.columns.get(col)
        return r.role if r else None

    def cols(self, *roles: str) -> list:
        want = set(roles)
        return [c for c, r in self.columns.items() if r.role in want]

    def group_cols(self, group: str) -> list:
        return [c for c, r in self.columns.items() if r.group == group]

    @property
    def date_col(self) -> Optional[str]:
        dates = self.cols("date")
        return dates[0] if dates else None

    @property
    def needs_target_confirmation(self) -> bool:
        """True until a person has chosen the target(s): never guess among several outcomes."""
        return not self.targets

    @property
    def primary_target(self) -> Optional[str]:
        return self.targets[0] if self.targets else None

    def unit_of(self, col: str) -> str:
        r = self.columns.get(col)
        return r.unit if r else ""

    def is_monetary(self, col: str) -> bool:
        return self.role_of(col) in MONETARY_ROLES

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RoleMapping":
        d = dict(d)
        cols = {k: ColumnRole(**v) for k, v in (d.pop("columns", {}) or {}).items()}
        known = {f for f in cls.__dataclass_fields__}
        m = cls(**{k: v for k, v in d.items() if k in known})
        m.columns = cols
        return m


# --------------------------------------------------------------------------- value profile
def _profile(s: pd.Series) -> dict:
    n = int(len(s))
    nn = s.dropna()
    p: dict[str, Any] = {"n": n, "missing_pct": round(100.0 * (n - len(nn)) / n, 2) if n else 0.0,
                         "n_unique": int(nn.nunique()) if len(nn) else 0}
    p["unique_ratio"] = (p["n_unique"] / len(nn)) if len(nn) else 0.0
    num = pd.to_numeric(nn, errors="coerce") if not pd.api.types.is_numeric_dtype(nn) else nn.astype(float)
    if pd.api.types.is_bool_dtype(nn):
        num = nn.astype(float)
    numeric_share = float(num.notna().mean()) if len(nn) else 0.0
    p["numeric_share"] = numeric_share
    p["is_numeric"] = bool(pd.api.types.is_numeric_dtype(s) or numeric_share >= 0.97) and not pd.api.types.is_datetime64_any_dtype(s)
    if p["is_numeric"] and len(num.dropna()):
        v = num.dropna().to_numpy(dtype=float)
        p.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), std=float(v.std()),
                 integer_share=float(np.mean(np.abs(v - np.round(v)) < 1e-9)),
                 negative_share=float(np.mean(v < 0)),
                 is_binary=bool(set(np.unique(v)).issubset({0.0, 1.0})),
                 within_0_100=bool(v.min() >= 0 and v.max() <= 100),
                 within_0_1=bool(v.min() >= 0 and v.max() <= 1),
                 monotonic=bool(np.all(np.diff(v) > 0)) if len(v) > 2 else False)
    p["is_constant"] = p["n_unique"] <= 1
    # dates
    p["date_parse_share"] = 0.0
    if pd.api.types.is_datetime64_any_dtype(s):
        p["date_parse_share"] = 1.0
    elif not p["is_numeric"] and len(nn):
        sample = nn.astype(str).head(400)
        parsed = _parse_dates(sample)
        p["date_parse_share"] = float(parsed.notna().mean())
    elif p["is_numeric"] and len(nn) and p.get("integer_share", 0) == 1.0:
        v = num.dropna()
        if v.between(19000101, 21001231).all():           # 20240115
            parsed = pd.to_datetime(v.astype(int).astype(str), format="%Y%m%d", errors="coerce")
            p["date_parse_share"] = float(parsed.notna().mean())
    if not p["is_numeric"] and len(nn):
        p["avg_len"] = float(nn.astype(str).str.len().mean())
    return p


def _parse_dates(sample: pd.Series) -> pd.Series:
    """Best of several unambiguous formats; never trusts a one-off parse."""
    best = pd.Series([pd.NaT] * len(sample), index=sample.index)
    for kw in ({"format": "ISO8601"}, {"dayfirst": True, "format": "mixed"}, {"format": "mixed"}):
        try:
            parsed = pd.to_datetime(sample, errors="coerce", **kw)
        except (ValueError, TypeError):
            continue
        if parsed.notna().mean() > best.notna().mean():
            best = parsed
    # pure small integers ("1".."52") are not dates
    if sample.str.fullmatch(r"\d{1,4}").mean() > 0.9:
        return pd.Series([pd.NaT] * len(sample), index=sample.index)
    return best


# --------------------------------------------------------------------------- name rules
def _name_candidates(name: str) -> list[tuple[str, float, str, Optional[str]]]:
    """(role, score, evidence, channel) candidates from the column name alone."""
    toks = tokens(name)
    ts = set(toks)
    joined = "".join(toks)
    ch = channel_label(toks)
    out: list[tuple[str, float, str, Optional[str]]] = []

    def add(role: str, score: float, why: str, channel: Optional[str] = None) -> None:
        out.append((role, score, why, channel))

    has_spend = bool(ts & SPEND_WORDS)
    has_trade = bool(ts & TRADE_WORDS) or "tradespend" in joined
    has_derived = bool(ts & _DERIVED_WORDS) or joined.startswith(("avg", "average", "mean")) and not ts & _PRICE_WORDS
    exp_word = next((_EXPOSURE_WORDS[t] for t in toks if t in _EXPOSURE_WORDS), None)
    has_quality = bool(ts & _QUALITY_WORDS) and not has_spend
    if has_quality:
        add("quality_score", 0.84, f"the name contains a score / quality word ({_hit(toks, _QUALITY_WORDS)})"
            + (f" for the channel '{ch}' -- a creative or content quality measure, not spend or exposure"
               if ch else ""), ch)
        exp_word = None          # an "engagement score" is a score, not an engagement count

    if ts & _ID_WORDS or (toks and toks[-1] in {"code", "no", "num", "number"} and len(toks) > 1
                          and not ts & {"week", "month", "year"}):
        add("identifier", 0.62, f"the name contains an identifier word ({_hit(toks, _ID_WORDS | {'code', 'no', 'num', 'number'})})")
    for parts in _TIMEPART_SETS:
        if ts == parts or (len(toks) <= 3 and parts <= ts and not ts & _ID_WORDS and len(parts) > 1):
            add("time_part", 0.62, "the name is a calendar part (year / month / quarter / week number)")
            break
    if ts & _DATE_WORDS and not ts & (_ID_WORDS | _PRICE_WORDS | SPEND_WORDS) and not exp_word:
        add("date", 0.55, f"the name contains a date word ({_hit(toks, _DATE_WORDS)})")
    if ts & _PRICE_WORDS or "unitprice" in joined:
        add("price", 0.78, f"the name contains a price word ({_hit(toks, _PRICE_WORDS) or 'unit price'})")
    if ts & _DISCOUNT_WORDS and not has_spend:
        add("discount", 0.74, f"the name contains a discount word ({_hit(toks, _DISCOUNT_WORDS)})")
    if has_spend and has_trade:
        add("trade_spend", 0.82, "a spend/cost word together with a trade / promotion word "
                                 f"({_hit(toks, TRADE_WORDS) or 'trade'}) -- trade or promotion expenditure, "
                                 "not advertising for one channel")
    elif "trade" in ts and not has_spend:
        add("trade_spend", 0.5, "the name contains 'trade' (trade investment)")
    if ts & _PROMO_WORDS and not has_spend:
        add("promotion", 0.58, f"the name contains a promotion word ({_hit(toks, _PROMO_WORDS)})")
    if has_spend and not has_trade:
        if ch and not ts & _TOTAL_WORDS:
            add("ad_spend", 0.85, f"a spend/cost word with the advertising channel '{ch}'", ch)
        elif ts & _AD_WORDS or ts & _TOTAL_WORDS or ts == {"spend"} or ts <= SPEND_WORDS:
            add("total_marketing_spend", 0.72, "a marketing spend word without one specific channel")
            add("ad_spend", 0.42, "it could be the spend of a single, unnamed channel")
        elif ch:
            add("total_marketing_spend", 0.6, f"a total/aggregate spend for '{ch}'")
            add("ad_spend", 0.55, f"spend for '{ch}'", ch)
        elif ts & {"cost", "costs", "cogs"}:
            add("cost", 0.55, "a cost word with no advertising or channel word (e.g. cost of goods)")
            add("total_marketing_spend", 0.3, "it could be marketing cost")
        else:
            add("ad_spend", 0.5, "a spend word with a qualifier that may name a channel",
                _channel_from_name(toks, SPEND_WORDS))
    if exp_word and not has_spend and not has_derived:
        cl = ch or _channel_from_name(toks, set(_EXPOSURE_WORDS))
        add(exp_word, 0.8 if ch else 0.7, f"the name contains the exposure word '{_hit(toks, set(_EXPOSURE_WORDS))}'"
            + (f" and the channel '{ch}'" if ch else ""), cl)
    if ch and not has_spend and not exp_word and not has_derived and not has_quality and not ts & _CHANNEL_DIM_WORDS:
        add("ad_spend", 0.4, f"the name is the channel '{ch}' without saying whether it is spend or exposure", ch)
        add("impressions", 0.32, f"'{ch}' could also be an exposure measure", ch)
    if ts & {"cogs"}:
        add("cost", 0.7, "cost of goods sold")

    # outcomes
    pen = 0.32 if has_derived else 0.0
    if ts & {"revenue", "revenues", "turnover", "gmv", "income", "takings", "billing", "billings"} \
            or "netsales" in joined or "salesvalue" in joined or "grosssales" in joined:
        add("revenue", 0.82 - pen, "the name contains a revenue word")
    elif "sales" in ts or "sale" in ts:
        if ts & (_CURRENCY_WORDS - {"total"}):
            add("revenue", 0.8 - pen, f"'sales' together with a money word ({_hit(toks, _CURRENCY_WORDS)})")
        elif ts & {"units", "unit", "qty", "quantity", "volume", "vol", "pieces", "pcs", "cases", "packs"}:
            add("sales_units", 0.82 - pen, "'sales' together with a quantity word")
        else:
            add("revenue", 0.55 - pen, "'sales' on its own usually means sales value")
            add("sales_units", 0.45 - pen, "'sales' on its own can also mean units sold")
    if (ts & {"units", "qty", "quantity", "volume", "vol", "pieces", "pcs", "cases", "packs", "unitssold"}
            or ("unit" in ts and not ts & _PRICE_WORDS and not has_spend)) and not ts & _PRICE_WORDS and not has_spend:
        add("sales_units", 0.78 - pen, "the name contains a quantity word")
    if ts & {"orders", "order", "transactions", "transaction", "txns", "txn", "bookings", "booking"} \
            and not ts & (_ID_WORDS | _DATE_WORDS):
        add("orders", 0.72 - pen, "the name contains an order / transaction word")
    if ts & {"conversions", "conversion", "conv", "convs", "purchases", "signups", "installs", "acquisitions",
             "registrations", "subscriptions", "sign"}:
        add("conversions", 0.76 - pen, "the name contains a conversion word")
    if ts & {"leads", "lead", "enquiries", "inquiries", "mql", "mqls", "prospects"}:
        add("leads", 0.76 - pen, "the name contains a lead word")
    if ts & {"profit", "profits", "ebit", "ebitda", "earnings"}:
        add("profit", 0.76 - pen, "the name contains a profit word")
    elif "margin" in ts:
        add("profit", 0.45, "'margin' can be a profit amount")
        add("derived", 0.42, "'margin' can also be a percentage")
    if ts & {"customers", "customer", "buyers", "buyer", "clients", "client", "subscribers", "shoppers",
             "members", "users"} and not ts & _ID_WORDS:
        add("customers", 0.7 - pen, "the name contains a customer word")

    # other business variables
    if ts & _DIST_WORDS:
        add("distribution", 0.72, f"the name contains a distribution word ({_hit(toks, _DIST_WORDS)})")
    if ts & _COMP_WORDS:
        add("competitor", 0.76, f"the name contains a competitor word ({_hit(toks, _COMP_WORDS)})")
    if ts & _EXT_WORDS:
        add("external", 0.7, f"the name contains an external / seasonal word ({_hit(toks, _EXT_WORDS)})")
    elif "index" in ts and not ts & _COMP_WORDS:
        add("external", 0.42, "an index (often an external factor)")
    if has_derived:
        add("derived", 0.4 if ts & _EXT_WORDS else 0.6, f"the name describes a ratio, share or growth ({_hit(toks, _DERIVED_WORDS) or 'average'})")

    # dimensions
    if ts & _BRAND_WORDS:
        add("brand", 0.86, "the name contains a brand word")
    if ts & _SKU_WORDS:
        add("sku", 0.86, "the name contains an SKU / item-code word")
    if ts & _PRODUCT_WORDS:
        add("product", 0.74, f"the name contains a product word ({_hit(toks, _PRODUCT_WORDS)})")
    if ts & _GEO_WORDS:
        add("geography", 0.84, f"the name contains a geography word ({_hit(toks, _GEO_WORDS)})")
    if ts & _CHANNEL_DIM_WORDS:
        add("channel_dimension", 0.66, "the name describes a channel / campaign / source")
    if ts & {"notes", "note", "comment", "comments", "description", "remarks", "text", "message", "review"}:
        add("text", 0.7, "the name describes free text")
    if ts in ({"week"}, {"month"}, {"day"}, {"wk"}):
        add("time_part", 0.5, "a bare week / month / day column is often a calendar number")
    if has_derived:
        # a share / ratio / growth of spend, exposure or an outcome is not that quantity itself
        penal = {"outcome", "spend", "spend_total", "exposure"}
        out = [(r, s - 0.35 if ROLE_INFO[r][0] in penal else s, w, c) for r, s, w, c in out]
    return out


def _hit(toks: list[str], words: Iterable[str]) -> str:
    w = set(words)
    return next((t for t in toks if t in w), "")


def _channel_from_name(toks: list[str], role_words: set) -> Optional[str]:
    rest = [t for t in toks if t not in role_words and t not in _AD_WORDS and t not in _TOTAL_WORDS]
    if not rest:
        return None
    for t in rest:
        if t in CHANNEL_WORDS:
            return CHANNEL_WORDS[t]
    return " ".join(t.capitalize() if len(t) > 3 else t.upper() for t in rest)


# --------------------------------------------------------------------------- value adjustments
def _adjust(role: str, score: float, p: dict, ev: list) -> float:
    """Raise or lower a name-based score using the actual values."""
    num = p.get("is_numeric", False)
    if p.get("is_constant"):
        return score - 0.3
    if role in NUMERIC_ROLES:
        if not num:
            ev.append("but its values are not numeric")
            return score - 0.5
        if role in set(OUTCOME_ROLES) - {"profit"} | {"ad_spend", "trade_spend", "price"} | set(EXPOSURE_ROLES):
            if p.get("negative_share", 0) > 0.01:
                ev.append(f"{100 * p['negative_share']:.0f}% of values are negative")
                score -= 0.2
        if role in set(EXPOSURE_ROLES) | {"sales_units", "orders", "conversions", "leads", "customers"}:
            if p.get("integer_share", 0) > 0.95:
                ev.append("values are whole numbers, consistent with counts")
                score += 0.08
            elif p.get("integer_share", 1) < 0.5 and role not in EXPOSURE_ROLES:
                ev.append("values have decimals, unlike counts")
                score -= 0.15
        if role in {"revenue", "ad_spend", "trade_spend", "cost", "price", "total_marketing_spend", "profit"}:
            if p.get("within_0_1") and not p.get("is_binary") and role != "price":
                ev.append("but every value lies between 0 and 1, which looks like a share, not an amount")
                score -= 0.3
            elif p.get("integer_share", 1) < 0.9:
                ev.append("values have decimals, consistent with money")
                score += 0.05
        if role == "promotion":
            if p.get("is_binary"):
                ev.append("values are 0/1 flags")
                score += 0.25
            elif p.get("within_0_1") or p.get("within_0_100"):
                ev.append("values look like a promotion intensity / share")
                score += 0.05
        if role == "discount" and (p.get("within_0_1") or p.get("within_0_100")):
            ev.append("values lie in a percentage range")
            score += 0.08
        if role == "distribution" and p.get("within_0_100"):
            ev.append("values lie between 0 and 100")
            score += 0.06
        if role == "external" and p.get("is_binary"):
            ev.append("values are 0/1 flags (calendar / event indicator)")
            score += 0.05
    if role in DIMENSION_ROLES:
        if num and p.get("n_unique", 0) > 60:
            ev.append("but it is numeric with many distinct values")
            score -= 0.35
        elif not num:
            ev.append(f"{p['n_unique']} distinct text values")
            score += 0.05
    if role == "date":
        share = p.get("date_parse_share", 0.0)
        if share >= 0.9:
            ev.append(f"{100 * share:.0f}% of values parse as dates")
            score += 0.35
        else:
            score -= 0.45
    if role == "time_part":
        if not (num and p.get("integer_share", 0) == 1.0):
            score -= 0.4
    if role == "identifier":
        if p.get("unique_ratio", 0) > 0.95:
            ev.append("almost every value is unique")
            score += 0.25
        else:
            score -= 0.3
    return score


def _value_only_candidates(p: dict) -> list[tuple[str, float, str, Optional[str]]]:
    out = []
    if p.get("is_constant"):
        out.append(("ignore", 0.9, "the column has a single value, so it cannot be analysed", None))
        return out
    if p.get("date_parse_share", 0) >= 0.9:
        out.append(("date", 0.62, f"{100 * p['date_parse_share']:.0f}% of values parse as dates", None))
    if not p.get("is_numeric"):
        if p.get("unique_ratio", 0) > 0.9 and p.get("n", 0) > 20:
            out.append(("identifier", 0.55, "text values that are almost all unique", None))
        elif p.get("avg_len", 0) > 30:
            out.append(("text", 0.55, "long text values", None))
        else:
            out.append(("category", 0.5, f"{p.get('n_unique', 0)} distinct text values", None))
    else:
        if p.get("is_binary"):
            out.append(("measure", 0.4, "0/1 values with no descriptive name", None))
            out.append(("promotion", 0.3, "0/1 values could be a promotion flag", None))
        elif p.get("monotonic") and p.get("unique_ratio", 0) == 1.0 and p.get("integer_share", 0) == 1.0:
            out.append(("identifier", 0.5, "a strictly increasing whole-number sequence", None))
        else:
            out.append(("measure", 0.38, "a numeric column with no recognised business word in its name", None))
    return out


# --------------------------------------------------------------------------- relationships
def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.nanmedian(np.abs(a))
    if not np.isfinite(denom) or denom == 0:
        return np.inf
    return float(np.nanmedian(np.abs(a - b)) / denom)


def _relationships(df: pd.DataFrame, roles: dict[str, ColumnRole], cand: dict[str, list]) -> list[str]:
    """Mutates roles in place; returns dataset-level notes."""
    notes: list[str] = []
    num = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
           for c, r in roles.items() if r.stats.get("is_numeric")}

    # 1) revenue = units x price (x (1 - discount))
    revs = [c for c, r in roles.items() if r.role == "revenue" and c in num]
    units = [c for c, r in roles.items() if r.role == "sales_units" and c in num]
    prices = [c for c, r in roles.items() if r.role == "price" and c in num]
    for rv in revs:
        best = None
        for u in units:
            for pr in prices:
                d = _rel_diff(num[rv], num[u] * num[pr])
                if d < 0.03 and (best is None or d < best[0]):
                    best = (d, u, pr)
        if best:
            d, u, pr = best
            msg = f"{rv} ≈ {u} × {pr} (median difference {100 * d:.1f}%)"
            roles[rv].confidence = min(0.99, roles[rv].confidence + 0.1)
            roles[rv].evidence.append(msg + ", so it is a sales value")
            roles[u].confidence = min(0.99, roles[u].confidence + 0.05)
            roles[u].evidence.append(msg + ", so it is a quantity")
            roles[pr].evidence.append(msg)
            notes.append(msg)

    # 2) spend columns that are sums of other spend columns -> aggregate, not a channel
    spend_cols = [c for c, r in roles.items() if r.role in ("ad_spend", "total_marketing_spend") and c in num]
    pool = sorted(spend_cols, key=lambda c: -np.nansum(num[c]))
    aggregates: set = set()
    for s in pool:
        others = [c for c in pool if c != s and c not in aggregates and np.nansum(num[c]) < np.nansum(num[s])]
        found = None
        if len(others) >= 2:
            if _rel_diff(num[s], np.nansum([num[o] for o in others], axis=0)) < 0.005:
                found = others
            elif len(others) <= 12:
                for k in range(len(others) - 1, 1, -1):
                    for combo in itertools.combinations(others, k):
                        if _rel_diff(num[s], np.nansum([num[o] for o in combo], axis=0)) < 0.005:
                            found = list(combo)
                            break
                    if found:
                        break
        if found:
            aggregates.add(s)
            r = roles[s]
            alt = {"role": r.role, "label": ROLE_INFO[r.role][1], "confidence": round(r.confidence, 2),
                   "why": "original name-based suggestion"}
            r.role, r.group = "total_marketing_spend", "spend_total"
            r.confidence = 0.93
            r.channel = None
            r.evidence.append("equals the sum of " + " + ".join(found)
                              + ", so it is an aggregate (counting it as a channel would double-count spend)")
            if alt["role"] != "total_marketing_spend":
                r.alternatives = [alt] + [a for a in r.alternatives if a["role"] != alt["role"]]
            notes.append(f"{s} is the sum of {len(found)} other spend columns")

    # 3) outcome totals made of other outcome columns (e.g. total customers = new + returning)
    for role in OUTCOME_ROLES:
        cols = [c for c, r in roles.items() if r.role == role and c in num]
        if len(cols) < 3:
            continue
        for c in cols:
            others = [o for o in cols if o != c]
            for k in range(len(others), 1, -1):
                hit = next((combo for combo in itertools.combinations(others, k)
                            if _rel_diff(num[c], np.nansum([num[o] for o in combo], axis=0)) < 0.005), None)
                if hit:
                    roles[c].evidence.append("equals " + " + ".join(hit))
                    roles[c].warnings.append("This outcome is the total of other outcome columns; choose one level "
                                             "as the target, not both.")
                    break

    # 4) exact duplicates
    seen: dict[bytes, str] = {}
    for c, arr in num.items():
        key = hashlib.sha1(np.nan_to_num(arr, nan=-1.2345e300).tobytes(), usedforsecurity=False).digest()
        if key in seen:
            roles[c].warnings.append(f"Identical to '{seen[key]}'.")
        else:
            seen[key] = c
    return notes


# --------------------------------------------------------------------------- suggestion
def _unit_for(role: str, channel: Optional[str]) -> str:
    return ROLE_INFO[role][2]


def suggest_roles(df: pd.DataFrame, dataset_id: str = "", dataset_name: str = "") -> RoleMapping:
    """Suggest a business role for every column. Never confirms anything itself."""
    roles: dict[str, ColumnRole] = {}
    cand_by_col: dict[str, list] = {}
    for col in df.columns:
        p = _profile(df[col])
        cands = []
        for role, score, why, ch in _name_candidates(str(col)):
            ev = [why]
            s = _adjust(role, score, p, ev)
            cands.append((role, s, ev, ch))
        for role, score, why, ch in _value_only_candidates(p):
            if not any(c[0] == role for c in cands):
                ev = [why]
                s = _adjust(role, score, p, ev) if role in ("date",) else score
                cands.append((role, s, ev, ch))
        # merge duplicates of one role (keep best), sort
        best: dict[str, tuple] = {}
        for c in cands:
            if c[0] not in best or c[1] > best[c[0]][1]:
                best[c[0]] = c
        ranked = sorted(best.values(), key=lambda c: -c[1])
        top = ranked[0]
        conf = float(np.clip(top[1], 0.05, 0.99))
        alts = [{"role": r, "label": ROLE_INFO[r][1], "confidence": round(float(np.clip(s, 0.05, 0.99)), 2),
                 "why": "; ".join(e)} for r, s, e, _ in ranked[1:] if s >= 0.2][:4]
        warnings = []
        if alts and conf - alts[0]["confidence"] < 0.12:
            warnings.append(f"Ambiguous: it could also be {alts[0]['label'].lower()} "
                            f"({alts[0]['confidence']:.2f} vs {conf:.2f}). Please confirm.")
        if conf < 0.5:
            warnings.append("Low confidence -- please review this role.")
        if p.get("is_constant"):
            warnings.append("The column has a single value.")
        if p.get("missing_pct", 0) > 20:
            warnings.append(f"{p['missing_pct']:.0f}% of values are missing.")
        role = top[0]
        roles[str(col)] = ColumnRole(
            column=str(col), role=role, group=ROLE_INFO[role][0], confidence=round(conf, 2),
            channel=top[3] if ROLE_INFO[role][0] in ("spend", "exposure") else None,
            unit=_unit_for(role, top[3]), evidence=list(top[2]), alternatives=alts, warnings=warnings,
            dtype=str(df[col].dtype), stats={k: v for k, v in p.items() if not isinstance(v, (list, dict))})
        cand_by_col[str(col)] = ranked
    notes = _relationships(df, roles, cand_by_col)
    for r in roles.values():
        r.confidence = round(float(np.clip(r.confidence, 0.05, 0.99)), 2)
        if r.group in ("spend", "exposure") and not r.channel:
            r.channel = _channel_from_name(tokens(r.column), SPEND_WORDS | set(_EXPOSURE_WORDS)) or r.column
    m = RoleMapping(dataset_id=dataset_id, dataset_name=dataset_name, fingerprint=fingerprint(df),
                    columns=roles, n_rows=int(len(df)))
    m.target_candidates = rank_targets(m)
    m.warnings = _dataset_warnings(m) + notes
    return m


def rank_targets(m: RoleMapping) -> list:
    """Outcome columns, most likely analysis target first (never a silent choice)."""
    outs = [c for c, r in m.columns.items() if r.group == "outcome"]
    return sorted(outs, key=lambda c: (OUTCOME_PRIORITY.index(m.columns[c].role)
                                       if m.columns[c].role in OUTCOME_PRIORITY else 99, -m.columns[c].confidence))


def _dataset_warnings(m: RoleMapping) -> list:
    w = []
    if not m.cols("date"):
        w.append("No date / period column was found: trends, time-aware validation and the optimizer need one.")
    outs = m.target_candidates
    if not outs:
        w.append("No business outcome (revenue, units, orders, leads...) was recognised. Choose a target "
                 "column manually if one exists.")
    elif len(outs) > 1:
        w.append("Several columns could be the analysis target (" + ", ".join(outs)
                 + "). Confirm which one(s) to analyse; none is chosen for you.")
    if not m.cols("ad_spend") and m.cols("total_marketing_spend"):
        w.append("Only total marketing spend is present, not spend per channel: channel ROI / ROAS cannot be "
                 "calculated.")
    if m.cols("trade_spend"):
        w.append("Trade / promotion spend (" + ", ".join(m.cols("trade_spend")) + ") is treated as a business "
                 "variable, not as advertising for a channel. Change its role if that is wrong.")
    if not m.cols("ad_spend") and not m.group_cols("exposure"):
        w.append("No channel-wise marketing spend or exposure was found: marketing models are limited to "
                 "general business analysis.")
    return w


# --------------------------------------------------------------------------- user edits
def apply_user_roles(m: RoleMapping, changes: dict, targets: Optional[list] = None,
                     channels: Optional[dict] = None) -> RoleMapping:
    """Return a copy with the person's role choices applied (``changes``: column -> role).
    Never alters a column that was not named. ``targets`` replaces the target list when given."""
    out = copy.deepcopy(m)
    for col, role in (changes or {}).items():
        if col not in out.columns or role not in ROLE_INFO:
            continue
        r = out.columns[col]
        if r.role != role:
            prev = {"role": r.role, "label": ROLE_INFO[r.role][1], "confidence": r.confidence,
                    "why": "previous suggestion"}
            r.alternatives = [prev] + [a for a in r.alternatives if a["role"] != role][:3]
            r.role, r.group, r.unit = role, ROLE_INFO[role][0], ROLE_INFO[role][2]
            r.evidence = r.evidence + ["role chosen by the user"]
            r.confidence = 1.0
            if r.group in ("spend", "exposure") and not r.channel:
                r.channel = _channel_from_name(tokens(col), SPEND_WORDS | set(_EXPOSURE_WORDS)) or col
            if r.group not in ("spend", "exposure"):
                r.channel = None
        r.user_set = True
    for col, ch in (channels or {}).items():
        if col in out.columns and ch:
            out.columns[col].channel = str(ch)
    out.target_candidates = rank_targets(out)
    if targets is not None:
        out.targets = [t for t in targets if t in out.columns and out.columns[t].group == "outcome"]
    else:
        out.targets = [t for t in out.targets if t in out.target_candidates]
    out.warnings = _dataset_warnings(out)
    return out


def validate_targets(m: RoleMapping, df: pd.DataFrame, targets: list) -> list:
    """Blocking problems with the chosen targets."""
    issues = []
    for t in targets:
        if t not in df.columns:
            issues.append(f"'{t}' is not in the dataset.")
        elif m.columns.get(t) is None or m.columns[t].group != "outcome":
            issues.append(f"'{t}' must have an outcome role to be a target.")
        elif pd.to_numeric(df[t], errors="coerce").nunique() < 2:
            issues.append(f"'{t}' does not vary, so it cannot be analysed.")
    return issues


# --------------------------------------------------------------------------- mode
def detect_mode(m: RoleMapping) -> ModeResult:
    """Spend-based when real channel-wise advertising spend exists; exposure-based when
    channel exposure exists without channel spend; otherwise general business analytics."""
    spend = m.cols("ad_spend")
    exposure: dict[str, list] = {}
    for role in EXPOSURE_ROLES:
        cols = m.cols(role)
        if cols:
            exposure[role] = cols
    outcomes = m.target_candidates
    ev: list[str] = []
    notes: list[str] = []
    if spend:
        mode = "spend"
        ev.append(f"{len(spend)} channel-wise advertising spend column(s): " + ", ".join(spend))
        if exposure:
            ev.append(f"{sum(len(v) for v in exposure.values())} exposure column(s) are also available as "
                      "supporting metrics")
    elif exposure:
        mode = "exposure"
        n = sum(len(v) for v in exposure.values())
        ev.append(f"{n} marketing exposure column(s): " + ", ".join(c for v in exposure.values() for c in v))
        ev.append("no channel-wise advertising spend was found, so monetary ROI / ROAS and budget allocation "
                  "are not available")
        if m.cols("total_marketing_spend"):
            notes.append("Total marketing spend exists but is not split by channel.")
    else:
        mode = "general"
        ev.append("no channel-wise advertising spend and no marketing exposure columns were found")
    if m.cols("trade_spend"):
        ev.append("trade / promotion spend (" + ", ".join(m.cols("trade_spend"))
                  + ") is a business variable, not channel advertising spend")
    if outcomes:
        ev.append("outcome column(s): " + ", ".join(outcomes))
    else:
        notes.append("No outcome column is mapped; choose a target to run the models.")
    return ModeResult(mode=mode, label=MODE_LABELS[mode], evidence=ev, spend_channels=spend,
                      exposure_channels=exposure, outcomes=outcomes, has_outcome=bool(outcomes), notes=notes)


# --------------------------------------------------------------------------- persistence
def fingerprint(df: pd.DataFrame) -> str:
    """Identifies the table structure a mapping belongs to: column names (in order) and row count.
    Data types are left out on purpose -- the same table read by different loaders (dates parsed or
    not, ints vs floats) must keep its mapping."""
    sig = [str(c) for c in df.columns] + [f"__rows__={len(df)}"]
    return hashlib.sha256(json.dumps(sig).encode("utf-8")).hexdigest()[:16]


def save_mapping(directory: Path, m: RoleMapping, *, note: str = "") -> RoleMapping:
    """Persist a confirmed mapping; increments ``version`` and appends a history entry."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prev = load_mapping(directory)
    out = copy.deepcopy(m)
    out.version = (prev.version if prev else m.version) + 1
    out.confirmed = True
    out.source = "confirmed"
    out.confirmed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    changed = {}
    if prev:
        for c, r in out.columns.items():
            old = prev.columns.get(c)
            if old is None or old.role != r.role:
                changed[c] = [old.role if old else None, r.role]
    out.history = (prev.history if prev else []) + [{
        "version": out.version, "at": out.confirmed_at, "targets": list(out.targets),
        "changed": changed, "note": note}]
    out.history = out.history[-50:]
    (directory / MAPPING_FILE).write_text(json.dumps(out.to_dict(), indent=2, default=str), encoding="utf-8")
    return out


def load_mapping(directory: Path, df: Optional[pd.DataFrame] = None) -> Optional[RoleMapping]:
    """The saved mapping, or None when absent, unreadable or made for a different table structure."""
    path = Path(directory) / MAPPING_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        m = RoleMapping.from_dict(data)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if m.schema_version != SCHEMA_VERSION:
        return None
    if df is not None and m.fingerprint and m.fingerprint != fingerprint(df):
        return None
    return m


# --------------------------------------------------------------------------- legacy bridge
def legacy_overrides(m: RoleMapping) -> tuple[set, dict]:
    """Translate a confirmed mapping into the (confirmed_roles, overrides) shape that the
    existing upload pipeline (``run_full_analysis``) understands."""
    confirmed, overrides = set(), {}
    if m.date_col:
        overrides["date_column"] = m.date_col
        confirmed.add("date")
    money = [t for t in m.targets if m.columns[t].role in ("revenue", "profit")]
    if money:
        overrides["revenue_column"] = money[0]
        confirmed.add("revenue")
    else:
        overrides["revenue_column"] = None
    overrides["channel_columns"] = m.cols("ad_spend")
    if overrides["channel_columns"]:
        confirmed.add("channels")
    return confirmed, overrides


def monetary_columns(m: RoleMapping) -> list:
    return [c for c, r in m.columns.items() if r.role in MONETARY_ROLES]
