"""Supported / unsupported capabilities of the four PART 02 models for one dataset.

Decided from the confirmed role mapping (and, for model-quality-dependent
items, marked as conditional). Every unavailable item says why and what
information would be needed. Used by the model pages, the tests and the
dataset compatibility report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from src.data_engine.role_engine import RoleMapping, detect_mode

MODELS = {
    "marketing_effectiveness": "Marketing Effectiveness Model",
    "channel_performance": "Channel Performance & Contribution",
    "resource_optimizer": "Marketing Resource Optimizer",
    "scenario_simulator": "Marketing Scenario Simulator",
}


@dataclass
class Capability:
    key: str
    label: str
    available: bool
    reason: str
    needs: str = ""
    conditional: str = ""      # e.g. depends on model quality, checked when the model runs

    def to_dict(self) -> dict:
        return asdict(self)


def _cap(key, label, ok, reason, needs="", conditional="") -> Capability:
    return Capability(key, label, bool(ok), reason, needs if not ok else "", conditional)


def model_capabilities(m: RoleMapping, target: Optional[str] = None) -> dict:
    """model key -> list[Capability]."""
    mode = detect_mode(m)
    target = target or m.primary_target
    t_role = m.role_of(target) if target else None
    monetary = t_role in ("revenue", "profit")
    has_t = bool(target)
    date = bool(m.date_col)
    spend = mode.spend_channels
    expo = mode.exposure_columns
    channels = spend if mode.mode == "spend" else (expo if mode.mode == "exposure" else [])
    dims = [c for c in m.cols("brand", "geography", "product", "sku")]
    price = m.cols("price")
    promo = m.cols("promotion")
    business = m.cols("price", "promotion", "discount", "trade_spend", "distribution", "competitor", "external",
                      "quality_score")
    no_t = "a confirmed target (outcome) column"
    units = {}
    for role, cols in mode.exposure_channels.items():
        units[role] = len(cols)
    multi_unit = any(n >= 2 for n in units.values())
    out: dict = {}

    out["marketing_effectiveness"] = [
        _cap("spend_effectiveness", "Channel spend effectiveness (response model)", has_t and mode.mode == "spend",
             "channel-wise advertising spend is available" if mode.mode == "spend" else "no channel-wise advertising spend",
             "channel-wise advertising spend columns" if has_t else no_t),
        _cap("mmm", "Marketing-mix model (adstock and saturation)", has_t and mode.mode == "spend" and date,
             "spend channels and a date are available" if mode.mode == "spend" and date else
             ("needs a date column" if mode.mode == "spend" else "needs channel-wise advertising spend"),
             "a date column" if mode.mode == "spend" else "channel-wise advertising spend and a date column",
             conditional="needs at least 20 periods" if mode.mode == "spend" else ""),
        _cap("roi_roas", "ROI / ROAS by channel", has_t and mode.mode == "spend" and monetary and date,
             ("spend, a monetary target and a date are available" if mode.mode == "spend" and monetary and date else
              "ROI / ROAS need channel-wise advertising spend" if mode.mode != "spend" else
              f"the target '{target}' is not money (outcome per unit of spend is shown instead)" if not monetary and has_t
              else "needs a date column"),
             "channel-wise advertising spend" if mode.mode != "spend" else "a monetary target (revenue / profit)"),
        _cap("exposure_effectiveness", "Exposure versus outcome relationships", has_t and mode.mode == "exposure",
             "exposure channels are available" if mode.mode == "exposure" else
             ("spend mode uses spend channels" if mode.mode == "spend" else "no marketing exposure columns"),
             "marketing exposure columns (impressions, reach, clicks, readership...)"),
        _cap("association", "Statistical association, elasticities and variable importance", has_t and bool(channels or business),
             "drivers are available" if channels or business else "no drivers", "driver columns" if has_t else no_t),
        _cap("by_time", "Effectiveness over time", has_t and date and bool(channels), "date and channels available"
             if date and channels else "needs a date and channels", "a date column and marketing channels"),
        _cap("by_segment", "Effectiveness by brand / geography / product", has_t and date and bool(channels) and bool(dims),
             "segment columns available" if dims else "no brand / geography / product column",
             "a brand, geography or product column"),
        _cap("promotion", "Promotion effectiveness", has_t and bool(promo), "promotion columns available" if promo
             else "no promotion column", "a promotion flag or intensity column"),
        _cap("business_drivers", "Business driver analysis (price, promotion, distribution...)", has_t and bool(business),
             "business drivers available" if business else "no business driver columns",
             "price, promotion, distribution or other business columns"),
    ]
    out["channel_performance"] = [
        _cap("comparison", "Channel comparison and ranking", has_t and bool(channels),
             f"{len(channels)} channel(s)" if channels else "no channel-wise spend or exposure",
             "channel-wise advertising spend or exposure columns"),
        _cap("trends", "Channel trends", bool(channels) and date, "date available" if date else "needs a date",
             "a date column"),
        _cap("contribution_share", "Statistical contribution share (%)", has_t and bool(channels),
             "from the driver model" if channels else "no channels", "channel columns"),
        _cap("monetary_contribution", "Model-estimated monetary contribution", has_t and mode.mode == "spend" and date,
             "from the marketing-mix model" if mode.mode == "spend" else "needs channel-wise advertising spend",
             "channel-wise advertising spend"),
        _cap("roi_roas", "ROI / ROAS by channel", has_t and mode.mode == "spend" and monetary and date,
             "spend and a monetary target" if mode.mode == "spend" and monetary else
             ("needs channel-wise advertising spend" if mode.mode != "spend" else "the target is not money"),
             "channel-wise advertising spend and a monetary target"),
        _cap("breakdown", "Brand / geography breakdowns", bool(channels) and bool(dims),
             "segment columns available" if dims else "no brand / geography / product column",
             "a brand, geography or product column"),
    ]
    out["resource_optimizer"] = [
        _cap("budget_allocation", "Monetary budget allocation", has_t and mode.mode == "spend" and date,
             "spend channels and a date" if mode.mode == "spend" and date else
             ("needs a date column" if mode.mode == "spend" else "needs channel-wise advertising spend (monetary)"),
             "channel-wise advertising spend and a date column"),
        _cap("constraints", "Minimum / maximum limits and feasibility checks", has_t and bool(channels) and date,
             "available with the allocation" if channels and date else "no channels to allocate",
             "channel-wise spend or exposure and a date"),
        _cap("exposure_allocation", "Exposure mix reallocation (same unit only)",
             has_t and mode.mode == "exposure" and date and multi_unit,
             ("at least two channels share an exposure unit" if multi_unit else
              "no exposure unit has two or more channels") if mode.mode == "exposure" else
             ("spend mode allocates money instead" if mode.mode == "spend" else "needs marketing exposure columns"),
             "two or more exposure channels in the same unit and a date",
             conditional="needs a validated model (holdout R² ≥ 0.3) and positive, significant channel effects"),
    ]
    sim_drivers = bool(channels or price or promo or business)
    out["scenario_simulator"] = [
        _cap("spend_changes", "Spend changes", has_t and mode.mode == "spend", "spend channels" if mode.mode == "spend"
             else "no channel-wise advertising spend", "channel-wise advertising spend columns"),
        _cap("exposure_changes", "Exposure changes", has_t and mode.mode == "exposure",
             "exposure channels" if mode.mode == "exposure" else "no exposure-based channels",
             "marketing exposure columns"),
        _cap("price_changes", "Price changes", has_t and bool(price), "price column" if price else "no price column",
             "a price column"),
        _cap("promotion_changes", "Promotion changes", has_t and bool(promo), "promotion column" if promo
             else "no promotion column", "a promotion column"),
        _cap("comparison", "Scenario comparison with approximate ranges", has_t and sim_drivers,
             "drivers available" if sim_drivers else "no drivers to change", "driver columns" if has_t else no_t,
             conditional="estimates are flagged as unreliable when the model is weak"),
    ]
    return out


def summary_lines(m: RoleMapping, target: Optional[str] = None) -> dict:
    """model -> (available labels, unavailable (label, reason, needs))."""
    caps = model_capabilities(m, target)
    out = {}
    for k, items in caps.items():
        out[k] = ([c.label for c in items if c.available],
                  [(c.label, c.reason, c.needs) for c in items if not c.available])
    return out
