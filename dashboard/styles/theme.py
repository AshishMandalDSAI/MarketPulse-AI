"""MarketPulse AI - Dashboard theme (custom CSS injected into Streamlit)."""

THEME_CSS = """
<style>
:root {
    --mp-navy: #0B1220;
    --mp-navy-soft: #121B2E;
    --mp-card: #FFFFFF;
    --mp-accent: #5B6CFF;
    --mp-accent-2: #8B5CF6;
    --mp-green: #16A34A;
    --mp-red: #DC2626;
    --mp-amber: #D97706;
    --mp-text-light: #E7EAF3;
    --mp-text-muted: #9AA4C2;
}

.stApp {
    background: linear-gradient(180deg, var(--mp-navy) 0%, var(--mp-navy-soft) 100%);
}

section[data-testid="stSidebar"] {
    background-color: #0B1220;
    border-right: 1px solid #1E2A44;
}

section[data-testid="stSidebar"] * {
    color: var(--mp-text-light) !important;
}

h1, h2, h3, h4 {
    color: var(--mp-text-light) !important;
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    letter-spacing: -0.01em;
}

p, span, label, div {
    color: var(--mp-text-light);
}

.mp-kpi-card {
    background: var(--mp-card);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.25);
    border: 1px solid #E5E7EB;
}
.mp-kpi-label {
    color: #6B7280;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.mp-kpi-value {
    color: #111827;
    font-size: 26px;
    font-weight: 700;
}
.mp-kpi-delta-pos { color: var(--mp-green); font-weight: 600; font-size: 13px; }
.mp-kpi-delta-neg { color: var(--mp-red); font-weight: 600; font-size: 13px; }

.mp-header-badge {
    display: inline-block;
    background: linear-gradient(90deg, var(--mp-accent), var(--mp-accent-2));
    color: white;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.mp-section-title {
    color: var(--mp-text-light);
    font-size: 18px;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 10px;
    border-left: 4px solid var(--mp-accent);
    padding-left: 10px;
}

.mp-callout {
    background: rgba(91,108,255,0.10);
    border: 1px solid rgba(91,108,255,0.35);
    border-radius: 10px;
    padding: 14px 16px;
    color: var(--mp-text-light);
}
</style>
"""

PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORWAY = ["#5B6CFF", "#8B5CF6", "#16A34A", "#D97706", "#DC2626", "#06B6D4", "#EC4899", "#84CC16"]
