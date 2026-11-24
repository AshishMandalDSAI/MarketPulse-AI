"""The Statistics page (menu entry 01): one workspace for descriptive, inferential and predictive work
(upgrade PART 01).

The three tabs reuse the existing Phase 8 modules unchanged, so every calculation,
assumption check, export and test behind them is the same as on the former
stand-alone pages 14-16 (which remain importable). The tabs are *lazy*: only the
open tab runs, so widgets of one tool never interfere with another and no work is
done for tabs nobody is looking at.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard import navigation
from dashboard.components.descriptive_ui import render_descriptive_body
from dashboard.components.graceful import safe_page
from dashboard.components.inferential_ui import render_inferential_body
from dashboard.components.predictive_ui import render_predictive_body
from dashboard.components.stats_ui import render_module_page
from dashboard.components.ui import page_header
from src.common.modules import get_module_spec

BODIES = {
    "descriptive": render_descriptive_body,
    "inferential": render_inferential_body,
    "predictive": render_predictive_body,
}


def _apply_forced_tab(tab_labels: list) -> None:
    """A one-shot request (navigation.go_to(..., stats_tab_module=...)) selects a tab
    before the tab widget is created; an unknown or stale value falls back to the
    first tab instead of raising."""
    forced = st.session_state.pop(navigation.STATS_TAB_FORCE_KEY, None)
    if forced in tab_labels:
        st.session_state[navigation.STATS_TAB_STATE_KEY] = forced
    elif st.session_state.get(navigation.STATS_TAB_STATE_KEY) not in tab_labels:
        st.session_state.pop(navigation.STATS_TAB_STATE_KEY, None)


@safe_page("Statistics")
def render():
    page_header(navigation.STATISTICS.split(" · ", 1)[0] + " · Statistics", "Statistics workspace",
                "Describe your data, test whether what you see is likely real, and train and compare "
                "predictive models, all on the active dataset. Each tab says what it needs, what it "
                "found in your data, which method it used and what the result means.")
    labels = [lbl for _, lbl in navigation.STATISTICS_TABS]
    _apply_forced_tab(labels)
    tabs = st.tabs(labels, key=navigation.STATS_TAB_STATE_KEY, on_change="rerun")
    for (module_key, _), tab in zip(navigation.STATISTICS_TABS, tabs):
        if tab.open:
            with tab:
                render_module_page(get_module_spec(module_key), BODIES[module_key], embedded=True)
