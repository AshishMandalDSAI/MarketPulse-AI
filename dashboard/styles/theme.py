"""MarketPulse AI — design system.

One set of design tokens drives the whole dashboard:

* ``.streamlit/config.toml`` sets Streamlit's own dark theme (widgets, inputs,
  buttons, alerts, charts) with the same colours, so built-in components stay
  legible. Earlier versions painted a dark background over Streamlit's *light*
  theme and forced light text on every element, which made button labels,
  alert text and the chat bar unreadable.
* ``THEME_CSS`` adds the product-specific pieces: KPI cards, page headers,
  section titles, callouts, status pills, grouped sidebar navigation and
  responsive rules. Existing class names are kept, so every page keeps working.
* ``TOKENS``, ``CHART_COLORWAY`` and ``apply_chart_style`` give charts the same
  palette, typography and number formatting.
"""

from __future__ import annotations

TOKENS = {
    "bg": "#0B1220",            # canvas
    "surface": "#131C2E",       # cards, sidebar
    "surface_2": "#1A2540",     # raised / hover
    "border": "#24314F",
    "text": "#E6EAF2",
    "text_muted": "#97A3BF",
    "primary": "#3B82F6",       # brand blue
    "primary_soft": "rgba(59,130,246,0.14)",
    "positive": "#14B8A6",      # teal — good performance
    "warning": "#F59E0B",       # amber
    "negative": "#EF4444",      # red
    "info": "#60A5FA",
}

# Colour-blind-friendlier categorical palette; first colour is the brand blue.
CHART_COLORWAY = ["#3B82F6", "#14B8A6", "#F59E0B", "#A78BFA", "#EF4444",
                  "#22D3EE", "#F472B6", "#84CC16", "#FB923C", "#94A3B8"]

# Kept for backwards compatibility with pages that pass template=PLOTLY_TEMPLATE.
PLOTLY_TEMPLATE = "plotly_dark"

# Sidebar navigation groups: (index of first page in the group, label).
# Derived from the page registry so renumbering the menu can never leave a
# heading above the wrong entry (upgrade PART 01).
from dashboard.navigation import NAV_GROUPS  # noqa: E402


def _nav_group_css() -> str:
    rules = []
    for idx, label in NAV_GROUPS:
        rules.append(
            'section[data-testid="stSidebar"] [role="radiogroup"] > div:nth-of-type(%d)::before'
            ' { content: "%s"; }' % (idx + 1, label.upper())
        )
    return "\n".join(rules)


THEME_CSS = """
<style>
:root {
    --mp-bg: %(bg)s;
    --mp-surface: %(surface)s;
    --mp-surface-2: %(surface_2)s;
    --mp-border: %(border)s;
    --mp-text: %(text)s;
    --mp-text-muted: %(text_muted)s;
    --mp-primary: %(primary)s;
    --mp-primary-soft: %(primary_soft)s;
    --mp-positive: %(positive)s;
    --mp-warning: %(warning)s;
    --mp-negative: %(negative)s;
    --mp-radius: 12px;
    /* legacy aliases used by older markup */
    --mp-accent: var(--mp-primary);
    --mp-text-light: var(--mp-text);
    --mp-green: var(--mp-positive);
    --mp-red: var(--mp-negative);
    --mp-amber: var(--mp-warning);
}

/* ---------- layout ---------- */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
h1 { font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 650 !important; letter-spacing: -0.01em; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--mp-surface);
    border-right: 1px solid var(--mp-border);
}
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 1px; }
section[data-testid="stSidebar"] [role="radiogroup"] > div { position: relative; width: 100%%; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    padding: 5px 10px 5px 8px;
    margin: 0;
    border-radius: 8px;
    border-left: 3px solid transparent;
    width: 100%%;
    transition: background 120ms ease;
}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: var(--mp-surface-2); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background: var(--mp-primary-soft);
    border-left-color: var(--mp-primary);
}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { font-weight: 650; color: #FFFFFF; }
/* hide the radio dot: the highlighted row is the active-page indicator */
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
/* group headings above the first page of each group */
section[data-testid="stSidebar"] [role="radiogroup"] > div::before {
    display: block;
    padding: 0 0 4px 11px;
    font-size: 10.5px; font-weight: 700; letter-spacing: 0.09em;
    color: var(--mp-text-muted);
}
section[data-testid="stSidebar"] [role="radiogroup"] > div:is(%(group_selectors)s) { margin-top: 14px; }
%(group_rules)s

.mp-brand { padding: 4px 0 6px 0; }
.mp-brand-name { font-size: 20px; font-weight: 750; color: var(--mp-text); letter-spacing: -0.01em; }
.mp-brand-mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 8px; margin-right: 8px;
    background: linear-gradient(135deg, var(--mp-primary), var(--mp-positive));
    color: #fff; font-size: 15px; font-weight: 800; vertical-align: -5px;
}
.mp-brand-tag { font-size: 12px; color: var(--mp-text-muted); margin-top: 6px; line-height: 1.4; }

.mp-dataset-card {
    background: var(--mp-bg); border: 1px solid var(--mp-border);
    border-radius: 10px; padding: 10px 12px; margin: 4px 0 10px 0;
}
.mp-dataset-card .mp-ds-title { font-size: 10.5px; font-weight: 700; letter-spacing: 0.09em; color: var(--mp-text-muted); }
.mp-dataset-card .mp-ds-name { font-size: 14px; font-weight: 650; color: var(--mp-text); margin-top: 4px; word-break: break-all; }
.mp-dataset-card .mp-ds-meta { font-size: 12px; color: var(--mp-text-muted); margin-top: 2px; }

/* ---------- pills ---------- */
.mp-pill {
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 11.5px; font-weight: 650; letter-spacing: 0.02em;
    border: 1px solid transparent; vertical-align: middle;
}
.mp-pill-demo { background: rgba(245,158,11,0.14); color: #FCD34D; border-color: rgba(245,158,11,0.35); }
.mp-pill-upload { background: rgba(20,184,166,0.14); color: #5EEAD4; border-color: rgba(20,184,166,0.35); }
.mp-pill-none { background: rgba(148,163,184,0.14); color: #CBD5E1; border-color: rgba(148,163,184,0.35); }
.mp-pill-good { background: rgba(20,184,166,0.14); color: #5EEAD4; }
.mp-pill-warn { background: rgba(245,158,11,0.14); color: #FCD34D; }
.mp-pill-bad { background: rgba(239,68,68,0.14); color: #FCA5A5; }

/* ---------- page header ---------- */
.mp-page-header { margin: 0 0 18px 0; }
.mp-header-badge {
    display: inline-block;
    background: var(--mp-primary-soft);
    color: #93C5FD;
    border: 1px solid rgba(59,130,246,0.35);
    padding: 3px 10px; border-radius: 999px;
    font-size: 11.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
}
.mp-page-title { font-size: clamp(26px, 3.2vw, 36px); font-weight: 750; color: var(--mp-text); margin: 10px 0 4px 0; letter-spacing: -0.02em; line-height: 1.15; }
.mp-page-subtitle { font-size: 15px; color: var(--mp-text-muted); max-width: 880px; line-height: 1.5; }
.mp-page-context { margin-top: 10px; font-size: 13px; color: var(--mp-text-muted); }

/* ---------- section titles ---------- */
.mp-section-title {
    color: var(--mp-text);
    font-size: 17px; font-weight: 700;
    margin: 22px 0 10px 0;
    padding-left: 10px;
    border-left: 3px solid var(--mp-primary);
    line-height: 1.3;
}

/* ---------- KPI cards ---------- */
.mp-kpi-card {
    background: var(--mp-surface);
    border: 1px solid var(--mp-border);
    border-radius: var(--mp-radius);
    padding: 16px 18px;
    min-height: 112px;
    height: 100%%;
    box-sizing: border-box;
}
.mp-kpi-label {
    color: var(--mp-text-muted);
    font-size: 11.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 8px; line-height: 1.3;
}
.mp-kpi-value {
    color: var(--mp-text);
    font-size: clamp(20px, 2.1vw, 27px); font-weight: 750;
    font-variant-numeric: tabular-nums; line-height: 1.15;
    overflow-wrap: anywhere;
}
.mp-kpi-delta-pos, .mp-kpi-delta-neg, .mp-kpi-delta-flat { font-weight: 650; font-size: 12.5px; margin-top: 6px; }
.mp-kpi-delta-pos { color: var(--mp-positive); }
.mp-kpi-delta-neg { color: var(--mp-negative); }
.mp-kpi-delta-flat { color: var(--mp-text-muted); }
.mp-kpi-help { color: var(--mp-text-muted); font-size: 12px; margin-top: 6px; line-height: 1.4; }

/* ---------- callouts ---------- */
.mp-callout {
    background: var(--mp-surface);
    border: 1px solid var(--mp-border);
    border-left: 3px solid var(--mp-primary);
    border-radius: 10px;
    padding: 12px 16px; margin: 0 0 10px 0;
    color: var(--mp-text); line-height: 1.55;
}
.mp-callout b, .mp-callout strong { color: #FFFFFF; }
.mp-callout-warn { border-left-color: var(--mp-warning); }
.mp-callout-good { border-left-color: var(--mp-positive); }
.mp-callout-bad { border-left-color: var(--mp-negative); }
.mp-callout-label { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--mp-text-muted); margin-bottom: 4px; }

/* ---------- empty state ---------- */
.mp-empty {
    border: 1px dashed var(--mp-border); border-radius: var(--mp-radius);
    padding: 22px; text-align: center; color: var(--mp-text-muted);
}
.mp-empty-title { color: var(--mp-text); font-weight: 650; font-size: 15px; margin-bottom: 4px; }

/* ---------- streamlit components ---------- */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    border: 1px solid var(--mp-border); border-radius: var(--mp-radius);
    padding: 6px; background: var(--mp-surface);
}
[data-testid="stExpander"] details { border-color: var(--mp-border) !important; border-radius: 10px !important; }
[data-testid="stMetric"] {
    background: var(--mp-surface); border: 1px solid var(--mp-border);
    border-radius: var(--mp-radius); padding: 12px 16px;
}
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed var(--mp-border); border-radius: var(--mp-radius); }

/* ---------- responsive ---------- */
@media (max-width: 1100px) {
    .block-container { padding-left: 1.2rem; padding-right: 1.2rem; }
}
@media (max-width: 640px) {
    .block-container { padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.4rem; }
    .mp-kpi-card { min-height: 0; padding: 12px 14px; }
    .mp-page-subtitle { font-size: 14px; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.6rem; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: calc(50%% - 0.6rem) !important; flex: 1 1 calc(50%% - 0.6rem) !important; }
    /* Rows holding charts, tables or forms stack full-width: two charts side by side
       at phone width squeeze legends over the plot (found in the 390px browser pass). */
    [data-testid="stHorizontalBlock"]:has(.stPlotlyChart, [data-testid="stDataFrame"], [data-testid="stSlider"], [data-testid="stSelectbox"]) > [data-testid="stColumn"] {
        min-width: 100%% !important; flex: 1 1 100%% !important;
    }
}
</style>
""" % {
    **TOKENS,
    "group_rules": _nav_group_css(),
    "group_selectors": ", ".join(":nth-of-type(%d)" % (i + 1) for i, _ in NAV_GROUPS),
}


def apply_chart_style(fig, *, title: str | None = None, height: int | None = None,
                      currency: str | None = None, percent_axis: str | None = None,
                      legend_bottom: bool = False, x_title: str | None = None,
                      y_title: str | None = None):
    """Apply the shared chart look: transparent background (the card supplies
    the surface), brand colourway, readable fonts, room for the title, unified
    hover, optional axis titles, and optional currency / percent tick formatting.

    ``currency`` names the monetary axis ("x" or "y"); its ticks get the *active
    dataset's* currency symbol (demo: ₹), or no symbol when the currency is
    unknown (upgrade PART 01). Build axis titles with
    ``dashboard.components.chart_labels``."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORWAY,
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=12.5, color=TOKENS["text"]),
        hoverlabel=dict(bgcolor=TOKENS["surface_2"], bordercolor=TOKENS["border"],
                        font=dict(color=TOKENS["text"], size=12.5)),
    )
    # Merge margins: add room for a title, never shrink a page's own margins
    # (some charts need wide left margins for long category labels).
    m = fig.layout.margin
    has_title = bool(title or (fig.layout.title and fig.layout.title.text))
    layout["margin"] = dict(
        t=max(m.t or 0, 56 if has_title else 28),
        l=max(m.l or 0, 12), r=max(m.r or 0, 12), b=max(m.b or 0, 12),
    )
    if title:
        layout["title"] = dict(text=title)
    if has_title:
        layout["title_x"] = 0.01
        layout["title_xanchor"] = "left"
        layout["title_font"] = dict(size=15)
    if height:
        layout["height"] = height
    if legend_bottom:
        layout["legend"] = dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0)
    fig.update_layout(**layout)
    grid = dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.25)", linecolor=TOKENS["border"])
    fig.update_xaxes(**grid)
    fig.update_yaxes(**grid)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title, automargin=True)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title, automargin=True)
    if currency:
        axis = fig.update_yaxes if currency.startswith("y") else fig.update_xaxes
        try:
            from dashboard.components.currency import symbol as _cur_symbol
            prefix = _cur_symbol()
        except Exception:
            prefix = ""
        axis(tickprefix=prefix, tickformat="~s")
    if percent_axis:
        axis = fig.update_yaxes if percent_axis == "y" else fig.update_xaxes
        axis(ticksuffix="%")
    return fig
