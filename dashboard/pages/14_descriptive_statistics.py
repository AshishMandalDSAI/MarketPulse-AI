"""Module 14 dashboard page (Phase 8). Layout comes from the shared
Phase 8 scaffold in dashboard.components.stats_ui; the module's own sections
are drawn by dashboard.components.descriptive_ui (delivered in Part 2)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.descriptive_ui import render_descriptive_body
from dashboard.components.graceful import safe_page
from dashboard.components.stats_ui import render_module_page
from src.common.modules import get_module_spec

SPEC = get_module_spec("descriptive")


@safe_page(SPEC.title)
def render():
    render_module_page(SPEC, render_descriptive_body)
