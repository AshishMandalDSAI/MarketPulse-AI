"""Ultimate release — UI design system and navigation regressions."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dashboard.components.ui import fmt_currency, fmt_pct, md_inline  # noqa: E402
from dashboard.styles import theme  # noqa: E402


class TestDesignSystem:
    def test_streamlit_runs_a_real_dark_theme(self):
        cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        assert cfg["theme"]["base"] == "dark"
        assert cfg["theme"]["backgroundColor"].lower() == theme.TOKENS["bg"].lower()
        assert cfg["theme"]["primaryColor"].lower() == theme.TOKENS["primary"].lower()

    def test_no_global_text_colour_override(self):
        # Forcing light text on every element over Streamlit's light widgets made
        # button labels and alert text invisible.
        assert not re.search(r"\bp,\s*span,\s*label,\s*div\s*\{", theme.THEME_CSS)

    def test_legacy_class_names_still_styled(self):
        for cls in ("mp-kpi-card", "mp-kpi-label", "mp-kpi-value", "mp-kpi-delta-pos", "mp-kpi-delta-neg",
                    "mp-header-badge", "mp-section-title", "mp-callout"):
            assert f".{cls}" in theme.THEME_CSS

    def test_navigation_groups_cover_all_pages_in_order(self):
        from app import PAGES
        starts = [i for i, _ in theme.NAV_GROUPS]
        assert starts == sorted(starts) and starts[0] == 0 and starts[-1] < len(PAGES)
        assert 'data-testid="stRadioOption"' in theme.THEME_CSS

    def test_responsive_rules_present(self):
        assert "@media (max-width: 640px)" in theme.THEME_CSS

    def test_chart_style_keeps_wide_margins_and_adds_title_room(self):
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=[1], y=[1]))
        fig.update_layout(title="T", margin=dict(t=20, l=150, r=10, b=10))
        theme.apply_chart_style(fig)
        assert fig.layout.margin.l == 150 and fig.layout.margin.t >= 56


class TestFormatting:
    def test_md_inline_escapes_html_and_renders_bold(self):
        out = md_inline("**Display Ads** <script>alert(1)</script> is *best*")
        assert "<b>Display Ads</b>" in out and "<i>best</i>" in out
        assert "<script>" not in out and "&lt;script&gt;" in out

    @pytest.mark.parametrize("value,expected", [(None, "—"), (float("nan"), "—"), (float("inf"), "—"),
                                                (1.5e7, "₹1.50 Cr"), (2.5e5, "₹2.50 L"), (999, "₹999")])
    def test_currency_never_shows_nan_or_zero_for_missing(self, value, expected):
        assert fmt_currency(value) == expected

    def test_pct(self):
        assert fmt_pct(12.345, signed=True) == "+12.3%" and fmt_pct(None) == "—"

    def test_kpi_card_without_baseline_shows_no_comparison(self, monkeypatch):
        from dashboard.components import kpi_card
        captured = []
        monkeypatch.setattr(kpi_card.st, "markdown", lambda html, **k: captured.append(html))
        kpi_card.render_kpi_card("Revenue", "₹10", delta=None)
        assert "▲" not in captured[0] and "▼" not in captured[0]

    def test_kpi_card_markup_matches_existing_parsers(self, monkeypatch):
        from dashboard.components import kpi_card
        captured = []
        monkeypatch.setattr(kpi_card.st, "markdown", lambda html, **k: captured.append(html))
        kpi_card.render_kpi_card("R&D <x>", "1", help="note")
        m = re.search(r'mp-kpi-label">([^<]+)</div>\s*<div class="mp-kpi-value">([^<]+)<', captured[0])
        assert m and m.group(1) == "R&D &lt;x&gt;" and m.group(2) == "1"


class TestNavigation:
    def test_first_click_after_programmatic_jump_opens_the_clicked_page(self):
        """Regression: a forced jump (e.g. 'try the demo') changed the radio's
        index argument, so the next click landed on Upload Data instead."""
        from test_phase8_part1_foundation import _new_app, _demo_session
        from app import PAGES
        labels = list(PAGES)
        at = _new_app({**_demo_session(), "mp_force_page": labels[1]})
        at.run()
        assert at.sidebar.radio[0].value == labels[1]
        at.sidebar.radio[0].set_value(labels[5]).run()
        assert at.sidebar.radio[0].value == labels[5]
        assert not at.exception
        at.run()                                   # a plain rerun keeps the page
        assert at.sidebar.radio[0].value == labels[5]

    def test_landing_screen_without_dataset(self):
        from test_phase8_part1_foundation import _new_app
        from app import PAGES
        at = _new_app({})
        at.run()
        at.sidebar.radio[0].set_value(list(PAGES)[3]).run()
        assert any("No dataset is currently active" in i.value for i in at.info)
        labels = [b.label for b in at.button]
        assert any("Upload" in l for l in labels) and any("Demo" in l for l in labels)


# ---------------------------------------------------------------------------
# Regression: a button nested inside another button's `if` branch can never fire
# (the rerun its click triggers has the outer button == False). The upload page's
# "Go to Executive Overview" button had this bug; it now uses an on_click callback.
# ---------------------------------------------------------------------------
import ast as _ast
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]


def _is_st_button(test):
    return (isinstance(test, _ast.Call) and isinstance(test.func, _ast.Attribute)
            and test.func.attr in ("button", "form_submit_button")
            and isinstance(test.func.value, _ast.Name) and test.func.value.id == "st")


def _nested_buttons(tree):
    found = []

    def visit(node, inside):
        for child in _ast.iter_child_nodes(node):
            if isinstance(child, _ast.If) and _is_st_button(child.test):
                if inside:
                    found.append(child.lineno)
                for stmt in child.body:
                    visit_stmt(stmt, True)
                for stmt in child.orelse:
                    visit_stmt(stmt, inside)
            else:
                visit(child, inside)

    def visit_stmt(stmt, inside):
        wrapper = _ast.Module(body=[stmt], type_ignores=[])
        visit(wrapper, inside)

    visit(tree, False)
    return found


def test_nested_button_detector_catches_the_pattern():
    src = "import streamlit as st\nif st.button('a'):\n    if st.button('b'):\n        pass\n"
    assert _nested_buttons(_ast.parse(src)) == [3]


def test_no_dashboard_page_nests_a_button_inside_another_button():
    files = sorted((_ROOT / "dashboard").rglob("*.py")) + [_ROOT / "app.py"]
    offenders = {str(p.relative_to(_ROOT)): n for p in files
                 if (n := _nested_buttons(_ast.parse(p.read_text(encoding="utf-8"))))}
    assert offenders == {}


def test_upload_page_overview_button_uses_a_callback():
    src = (_ROOT / "dashboard/pages/0_upload_data.py").read_text(encoding="utf-8")
    assert "on_click=_go_to_overview" in src
    fn = next(n for n in _ast.parse(src).body
              if isinstance(n, _ast.FunctionDef) and n.name == "_go_to_overview")

    class _FakeSt:
        session_state = {}

    from dashboard import navigation
    ns = {"st": _FakeSt, "navigation": navigation}
    exec(compile(_ast.Module(body=[fn], type_ignores=[]), "upload_cb", "exec"), ns)
    ns["_go_to_overview"]()
    # Upgrade PART 01: Executive Overview is menu entry 02 (label from the registry).
    assert _FakeSt.session_state["mp_force_page"] == navigation.EXECUTIVE_OVERVIEW == "02 · 📊 Executive Overview"


def test_sidebar_starts_collapsed_on_small_screens():
    """'expanded' made the sidebar cover the whole page on a 390px phone at first load
    (found by the browser pass); 'auto' collapses it on small screens only."""
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert 'initial_sidebar_state="auto"' in src


def test_chart_rows_stack_full_width_on_phones():
    from dashboard.styles.theme import THEME_CSS
    block = THEME_CSS.split("@media (max-width: 640px)")[1]
    assert ':has(.stPlotlyChart' in block and "min-width: 100% !important" in block
