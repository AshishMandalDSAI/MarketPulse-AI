"""Module 17 dashboard page (Phase 8). Layout comes from the shared
Phase 8 scaffold in dashboard.components.stats_ui; analytics are added to
this page's body function by the part of Phase 8 that delivers the module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page
from dashboard.components.prescriptive_ui import render_prescriptive_body
from dashboard.components.stats_ui import render_module_page
from src.common.modules import get_module_spec

SPEC = get_module_spec("prescriptive")


@safe_page(SPEC.title)
def render():
    render_module_page(SPEC, render_prescriptive_body)
