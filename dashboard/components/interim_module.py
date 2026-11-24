"""Transition content for the four new marketing-model pages (upgrade PART 01).

PART 01 moves the menu to the approved structure. The four new analytical modules
are implemented in PART 02. Until then each page:

1. says plainly that the new module is **not implemented yet**, what it will do,
   and that it arrives in PART 02;
2. keeps the previous module reachable, unchanged, inside a closed section that is
   explicitly labelled as the previous module (so nothing that worked is lost, and
   nothing from it is presented as a result of the new module).

Nothing on these pages is computed by, or labelled as, the new modules.
"""
from __future__ import annotations

import html
import importlib

import streamlit as st

from dashboard import navigation
from dashboard.components.ui import callout, page_header

DELIVERED_IN = "PART 02"


def pending_notice(title: str) -> str:
    return (f"**{title}: not implemented yet.** This module is built in {DELIVERED_IN} of the upgrade. "
            "No result on this page comes from it.")


def render_interim_module(nav_key: str, purpose: str, will_provide: list, legacy_module: str,
                          legacy_name: str) -> None:
    pos = navigation.label(nav_key).split(" · ", 1)[0]
    title = next(p.title for p in navigation.NAV_PAGES if p.key == nav_key)
    page_header(f"{pos} · Pending {DELIVERED_IN}", title, purpose)
    callout(pending_notice(title), kind="warn", label="Status")
    st.markdown(f'<div class="mp-section-title">What the {html.escape(title)} will provide</div>',
                unsafe_allow_html=True)
    st.markdown("\n".join(f"- {html.escape(item)}" for item in will_provide))
    st.markdown('<div class="mp-section-title">Previous module (kept until the new one replaces it)</div>',
                unsafe_allow_html=True)
    st.caption(f"The earlier **{legacy_name}** module is unchanged and is *not* the {title}. "
               "Switch it on below if you need it in the meantime.")
    # A toggle rather than an expander: the previous modules contain expanders of
    # their own, which Streamlit does not allow inside another expander. The module
    # only runs once the user switches it on.
    if st.toggle(f"Show the previous {legacy_name} module (unchanged)", value=False,
                 key=f"mp_interim_show_{nav_key}"):
        with st.container(border=True):
            st.caption(f"Previous module · {legacy_name} · not the {title}")
            importlib.import_module(legacy_module).render()
