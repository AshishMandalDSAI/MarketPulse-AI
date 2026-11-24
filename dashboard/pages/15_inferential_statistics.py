"""Module 15 dashboard page (Phase 8 Part 3). Layout comes from the shared
Phase 8 scaffold in dashboard.components.stats_ui; the module's own sections
are drawn by dashboard.components.inferential_ui."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page
from dashboard.components.inferential_ui import render_inferential_body
from dashboard.components.stats_ui import render_module_page
from src.common.modules import get_module_spec

SPEC = get_module_spec("inferential")


@safe_page(SPEC.title)
def render():
    render_module_page(SPEC, render_inferential_body)
