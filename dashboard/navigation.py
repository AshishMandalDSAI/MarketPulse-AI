"""Single source of truth for dashboard navigation (upgrade PART 01).

Page labels used to be repeated as string literals in app.py, the Upload page and
tests, and the sidebar group headings were tied to list positions. Renumbering the
menu therefore silently broke navigation. Every caller now uses the stable page
*keys* below; labels, order, numbering and group headings are derived from them.

Approved navigation (PART 01):
    00 Upload Data · 01 Statistics · 02 Executive Overview · 03 Marketing Performance
    04 Marketing Effectiveness Model · 05 Channel Performance & Contribution
    06 Marketing Resource Optimizer · 07 Marketing Scenario Simulator · 08 Forecast
    09 AI Marketing Analyst · 10 Recommendations · 11 Reports · 12 Predictions
    13 Anomalies & Drivers · 14 Decision Intelligence · 15 Prescriptive Analytics

The four previous model pages (Marketing Mix Model, Channel ROI, Budget Optimizer,
What-If Simulator) and the three separate statistics pages are no longer menu
entries, but their modules are kept and stay importable (``LEGACY_PAGES``); the
API, the AI analyst and the tests still use their engines.

This module has no Streamlit dependency except inside ``go_to`` (imported lazily),
so it can be imported from tests and from the theme module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NavPage:
    key: str
    icon: str
    title: str
    module: str
    group: Optional[str] = None      # sidebar heading shown above this entry, if any


NAV_PAGES: tuple = (
    NavPage("upload", "📤", "Upload Data", "dashboard.pages.0_upload_data", "Data"),
    NavPage("statistics", "🧮", "Statistics", "dashboard.pages.statistics", "Statistics"),
    NavPage("executive_overview", "📊", "Executive Overview", "dashboard.pages.1_executive_overview", "Overview"),
    NavPage("marketing_performance", "📈", "Marketing Performance", "dashboard.pages.2_marketing_performance"),
    NavPage("marketing_effectiveness", "🧠", "Marketing Effectiveness Model",
            "dashboard.pages.marketing_effectiveness", "Marketing models"),
    NavPage("channel_performance", "📡", "Channel Performance & Contribution",
            "dashboard.pages.channel_performance"),
    NavPage("resource_optimizer", "⚖️", "Marketing Resource Optimizer", "dashboard.pages.resource_optimizer"),
    NavPage("scenario_simulator", "🧪", "Marketing Scenario Simulator", "dashboard.pages.scenario_simulator"),
    NavPage("forecast", "📅", "Forecast", "dashboard.pages.7_forecast", "Forecast & insights"),
    NavPage("ai_analyst", "🤖", "AI Marketing Analyst", "dashboard.pages.8_ai_marketing_analyst"),
    NavPage("recommendations", "💡", "Recommendations", "dashboard.pages.9_recommendations"),
    NavPage("reports", "📋", "Reports", "dashboard.pages.10_reports"),
    NavPage("predictions", "🔮", "Predictions", "dashboard.pages.11_predictions", "Advanced analytics"),
    NavPage("anomalies_drivers", "🔍", "Anomalies & Drivers", "dashboard.pages.12_anomalies_drivers"),
    NavPage("decision_intelligence", "🧠", "Decision Intelligence", "dashboard.pages.13_decision_intelligence"),
    NavPage("prescriptive", "🎯", "Prescriptive Analytics", "dashboard.pages.17_prescriptive_analytics"),
)

_BY_KEY = {p.key: p for p in NAV_PAGES}
assert len(_BY_KEY) == len(NAV_PAGES), "duplicate navigation key"


def number(key: str) -> int:
    """Position of the page in the menu (00, 01, ...)."""
    return [p.key for p in NAV_PAGES].index(key)


def label(key: str) -> str:
    """Sidebar label, e.g. ``"01 · 🧮 Statistics"``."""
    p = _BY_KEY[key]
    return f"{number(key):02d} · {p.icon} {p.title}"


def key_for_label(text: str) -> Optional[str]:
    for p in NAV_PAGES:
        if label(p.key) == text:
            return p.key
    return None


#: label -> module, in menu order (what app.py renders).
PAGES: dict = {label(p.key): p.module for p in NAV_PAGES}

#: (position, heading) pairs for the sidebar group headings.
NAV_GROUPS: list = [(number(p.key), p.group) for p in NAV_PAGES if p.group]

# Frequently used labels.
UPLOAD = label("upload")
STATISTICS = label("statistics")
EXECUTIVE_OVERVIEW = label("executive_overview")

# ---------------------------------------------------------------------------
# Statistics workspace tabs (lazy tabs on the Statistics page)
# ---------------------------------------------------------------------------

#: (module-spec key, tab label) in tab order.
STATISTICS_TABS: tuple = (
    ("descriptive", "📊 Descriptive Statistics"),
    ("inferential", "🔬 Inferential Statistics"),
    ("predictive", "🔮 Predictive Analysis"),
)
STATS_TAB_STATE_KEY = "mp_stats_tab"          # the st.tabs widget key
STATS_TAB_FORCE_KEY = "mp_force_stats_tab"    # one-shot request for a tab


def stats_tab_label(module_key: str) -> str:
    return dict(STATISTICS_TABS)[module_key]


def page_for_module(module_key: str) -> tuple:
    """(menu label, tab label or None) of the place a Phase 8 statistics module lives."""
    if module_key in dict(STATISTICS_TABS):
        return STATISTICS, stats_tab_label(module_key)
    if module_key == "prescriptive":
        return label("prescriptive"), None
    raise KeyError(module_key)


# ---------------------------------------------------------------------------
# Modules kept but no longer in the menu
# ---------------------------------------------------------------------------

LEGACY_PAGES: dict = {
    "marketing_mix_model": "dashboard.pages.3_marketing_mix_model",
    "channel_roi": "dashboard.pages.4_channel_roi",
    "budget_optimizer": "dashboard.pages.5_budget_optimizer",
    "what_if_simulator": "dashboard.pages.6_what_if_simulator",
    "descriptive_statistics": "dashboard.pages.14_descriptive_statistics",
    "inferential_statistics": "dashboard.pages.15_inferential_statistics",
    "predictive_analytics": "dashboard.pages.16_predictive_analytics",
}


def go_to(key: str, stats_tab_module: Optional[str] = None) -> None:
    """Request a jump to a page (and optionally a Statistics tab) on the next run."""
    import streamlit as st
    st.session_state["mp_force_page"] = label(key)
    if stats_tab_module is not None:
        st.session_state[STATS_TAB_FORCE_KEY] = stats_tab_label(stats_tab_module)
