"""
MarketPulse AI — AI-Driven Marketing Mix Modeling & Budget Optimization Platform
====================================================================================
Main Streamlit entry point. Run with:

    streamlit run app.py

Requires the analytics/model pipeline to have been run at least once:

    python src/data/generate_dataset.py
    python src/data/data_cleaning.py
    python src/features/feature_engineering.py
    python src/analytics/marketing_analytics.py
    python src/models/marketing_mix_model.py
    python src/models/model_evaluation.py
    python src/models/channel_contribution.py
    python src/optimization/budget_optimizer.py
    python src/optimization/simulator.py
    python src/forecasting/forecast.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dashboard.styles.theme import THEME_CSS
from dashboard.components.data_loader import pipeline_ready
from dashboard.components import session_paths
from dashboard import navigation

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",  # collapsed on phones so it never covers the page
)
st.markdown(THEME_CSS, unsafe_allow_html=True)

# Menu order, labels and groups live in dashboard/navigation.py (single source of
# truth; upgrade PART 01). Labels are never written as literals elsewhere.
PAGES = navigation.PAGES


def _render_dataset_status() -> None:
    """Sidebar dataset card: which dataset is active, its size, and actions."""
    import html as _html
    if not session_paths.dataset_chosen():
        st.markdown(
            '<div class="mp-dataset-card"><div class="mp-ds-title">ACTIVE DATASET</div>'
            '<div class="mp-ds-name"><span class="mp-pill mp-pill-none">None</span></div>'
            '<div class="mp-ds-meta">Upload a CSV/XLSX or try the demo.</div></div>',
            unsafe_allow_html=True,
        )
        return
    active = session_paths.get_active_dataset() or {}
    if active.get("is_demo"):
        name, pill, meta = "NovaMart (demo)", '<span class="mp-pill mp-pill-demo">Demo</span>', \
            "Synthetic data for demonstration only."
    else:
        name = active.get("filename") or "Uploaded dataset"
        pill = '<span class="mp-pill mp-pill-upload">Uploaded</span>'
        dims = ""
        if active.get("n_rows") and active.get("n_cols"):
            dims = f"{active['n_rows']:,} rows · {active['n_cols']} columns · "
        meta = f"{dims}ID {str(active.get('dataset_id', ''))[:16]}"
    st.markdown(
        f'<div class="mp-dataset-card"><div class="mp-ds-title">ACTIVE DATASET &nbsp;{pill}</div>'
        f'<div class="mp-ds-name">{_html.escape(str(name))}</div>'
        f'<div class="mp-ds-meta">{_html.escape(meta)}</div></div>',
        unsafe_allow_html=True,
    )
    if active.get("is_demo"):
        if st.button("🔄 Reset", use_container_width=True, help="Clear the active dataset and start over"):
            session_paths.reset_analysis()
            st.rerun()
        return
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Demo data", use_container_width=True, help="Switch to the built-in demo dataset"):
            session_paths.reset_to_demo_dataset()
            st.rerun()
    with c2:
        if st.button("🔄 Reset", use_container_width=True,
                     help="Clear the active dataset and start over"):
            session_paths.reset_analysis()
            st.rerun()


def main():
    from dashboard.components import auth
    if not auth.require_login():
        return
    with st.sidebar:
        st.markdown(
            '<div class="mp-brand"><span class="mp-brand-mark">M</span>'
            '<span class="mp-brand-name">MarketPulse AI</span>'
            '<div class="mp-brand-tag">Marketing mix modeling, forecasting and budget decisions '
            'from your own data.</div></div>',
            unsafe_allow_html=True,
        )
        _render_dataset_status()
        page_keys = list(PAGES.keys())
        # A keyed radio whose state is set directly on a programmatic jump.
        # (Passing a different `index` on each rerun made Streamlit recreate
        # the widget, so the first click after a jump landed on page 00.)
        forced = st.session_state.pop("mp_force_page", None)
        if forced in page_keys:
            st.session_state["mp_nav_page"] = forced
        elif st.session_state.get("mp_nav_page") not in page_keys:
            st.session_state["mp_nav_page"] = page_keys[0]
        selection = st.radio("Navigate", page_keys, key="mp_nav_page", label_visibility="collapsed")
        st.markdown("---")
        st.caption("Start on **Upload Data**, confirm the column mapping, then explore. "
                   "Pages show only analyses your data supports.")
        auth.render_logout()

    # Dataset-first gate: before rendering any page except Upload Data, make
    # the person explicitly choose a dataset (uploaded, or the demo) rather
    # than silently defaulting to demo data on first load.
    if selection != navigation.UPLOAD and not session_paths.dataset_chosen():
        from dashboard.components.ui import page_header
        page_header("Welcome", "Choose a dataset to begin",
                    "MarketPulse AI works with **your own data** — any CSV or Excel (.xlsx) file "
                    "with a header row. Or explore every page first with the built-in demo dataset.")
        st.info("No dataset is currently active. Upload a CSV/XLSX file to begin, "
                "or try the demo dataset.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Go to Upload Data", type="primary", use_container_width=True):
                st.session_state["mp_force_page"] = navigation.UPLOAD
                st.rerun()
        with col2:
            if st.button("▶️ Try Demo Dataset instead", use_container_width=True):
                session_paths.reset_to_demo_dataset()
                st.rerun()
        st.markdown("&nbsp;")
        s1, s2, s3 = st.columns(3)
        for col, (num, title, body) in zip((s1, s2, s3), (
            ("1", "Upload", "CSV or .xlsx. Column names can be anything."),
            ("2", "Review", "Confirm detected columns and data-quality fixes."),
            ("3", "Decide", "Explore models, forecasts, budgets and insights."),
        )):
            with col:
                st.markdown(
                    f'<div class="mp-kpi-card"><div class="mp-kpi-label">Step {num}</div>'
                    f'<div class="mp-kpi-value" style="font-size:19px">{title}</div>'
                    f'<div class="mp-kpi-help">{body}</div></div>', unsafe_allow_html=True)
        return

    if selection == navigation.UPLOAD:
        import importlib
        module = importlib.import_module(PAGES[selection])
        module.render()
        return

    if not pipeline_ready():
        st.error("⚠️ Pipeline outputs not found — this dashboard needs the analytics/model pipeline to have "
                 "run at least once before it can display real numbers.")
        st.markdown(
            "**Option 1 — run it now (recommended for a live demo):**"
        )
        if st.button("▶️ Initialize Demo Data (run full pipeline)", type="primary"):
            with st.spinner("Generating dataset, training the MMM, and running optimization — this takes under a minute..."):
                import subprocess
                steps = [
                    "src/data/generate_dataset.py", "src/data/data_cleaning.py",
                    "src/features/feature_engineering.py", "src/analytics/marketing_analytics.py",
                    "src/models/marketing_mix_model.py", "src/models/model_evaluation.py",
                    "src/models/channel_contribution.py", "src/optimization/budget_optimizer.py",
                    "src/optimization/simulator.py", "src/forecasting/forecast.py",
                    "src/utils/database.py",
                ]
                failed_step = None
                for step in steps:
                    result = subprocess.run([sys.executable, step], cwd=str(ROOT), capture_output=True, text=True)
                    if result.returncode != 0:
                        failed_step = (step, result.stderr[-800:])
                        break
                if failed_step:
                    st.error(f"Pipeline step failed: `{failed_step[0]}`\n\n```\n{failed_step[1]}\n```")
                else:
                    st.cache_data.clear()
                    st.success("Demo data initialized — reloading...")
                    st.rerun()
        st.markdown("**Option 2 — run manually from a terminal:**")
        st.code(
            "python src/data/generate_dataset.py\n"
            "python src/data/data_cleaning.py\n"
            "python src/features/feature_engineering.py\n"
            "python src/analytics/marketing_analytics.py\n"
            "python src/models/marketing_mix_model.py\n"
            "python src/models/model_evaluation.py\n"
            "python src/models/channel_contribution.py\n"
            "python src/optimization/budget_optimizer.py\n"
            "python src/optimization/simulator.py\n"
            "python src/forecasting/forecast.py\n"
            "python src/utils/database.py",
            language="bash",
        )
        st.caption("See README.md → *Quick Start* for full details.")
        return

    import importlib
    module = importlib.import_module(PAGES[selection])
    module.render()


if __name__ == "__main__":
    main()
