"""Decision Intelligence dashboard page — Phase 7 Part 7.

Displays the output of the DecisionOrchestrator through the dashboard
service layer. All results are scoped to the currently active dataset.
No demo-data fallback, no fabricated results.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.components.graceful import safe_page
from dashboard.components import session_paths
from dashboard.components.data_loader import (
    load_dynamic_analytics,
    load_prediction_report,
    load_phase5_analysis,
)
from src.decision.dashboard_service import (
    run_decision_workflow,
    build_copilot_context,
    DashboardResult,
)


_SESSION_KEY_DECISION_RESULT = "decision_result"
_SESSION_KEY_COPILOT_CTX = "decision_copilot_context"


def _load_cached_result() -> DashboardResult | None:
    """Return the cached DashboardResult for the current dataset, or None."""
    result = st.session_state.get(_SESSION_KEY_DECISION_RESULT)
    if result is None:
        return None
    # Verify dataset match — never show stale results
    current_id = session_paths.get_dataset_id()
    if getattr(result, "dataset_id", None) != current_id:
        # Stale result from a previous dataset — discard
        st.session_state.pop(_SESSION_KEY_DECISION_RESULT, None)
        st.session_state.pop(_SESSION_KEY_COPILOT_CTX, None)
        return None
    return result


def _run_and_cache() -> DashboardResult:
    """Run the decision workflow for the active dataset, cache, and return."""
    dataset_id = session_paths.get_dataset_id()

    # Gather available analytical outputs
    analytics_result = None
    prediction_report = None
    phase5_result = None

    try:
        analytics_result = load_dynamic_analytics()
    except Exception:
        pass
    try:
        prediction_report = load_prediction_report()
    except Exception:
        pass
    try:
        phase5_result = load_phase5_analysis()
    except Exception:
        pass

    result = run_decision_workflow(
        dataset_id=dataset_id,
        analytics_result=analytics_result,
        prediction_report=prediction_report,
        phase5_result=phase5_result,
    )

    # Cache for this render cycle
    st.session_state[_SESSION_KEY_DECISION_RESULT] = result

    # Build and cache copilot context
    ctx = build_copilot_context(result, dataset_id)
    st.session_state[_SESSION_KEY_COPILOT_CTX] = ctx

    return result


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_status_header(result: DashboardResult) -> None:
    """Show workflow status and dataset identity."""
    status = result.status_display
    level = status.get("level", "info")
    label = status.get("label", "Unknown")
    desc = status.get("description", "")

    # Map level to streamlit method
    if level == "success":
        st.success(f"**Workflow Status: {label}** — {desc}")
    elif level == "warning":
        st.warning(f"**Workflow Status: {label}** — {desc}")
    elif level == "error":
        st.error(f"**Workflow Status: {label}** — {desc}")
    else:
        st.info(f"**Workflow Status: {label}** — {desc}")

    # Dataset and workflow identity
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Dataset:** `{result.dataset_id[:24]}`" if result.dataset_id else "**Dataset:** None")
    with col2:
        st.caption(f"**Workflow:** `{result.workflow_id[:24]}`" if result.workflow_id else "**Workflow:** —")


def _render_confidence(result: DashboardResult) -> None:
    """Show overall confidence level — separate from impact, urgency, priority."""
    conf = result.confidence_display
    label = conf.get("label", "Unknown")
    desc = conf.get("description", "")
    level = conf.get("level", "info")

    st.markdown("#### Overall Confidence")
    st.caption(
        "Confidence reflects how much evidence supports the findings. "
        "It does not imply high or low impact, urgency, or priority."
    )
    if level == "success":
        st.success(f"**{label}** — {desc}")
    elif level == "error":
        st.error(f"**{label}** — {desc}")
    else:
        st.warning(f"**{label}** — {desc}")


def _render_decision_cards(result: DashboardResult) -> None:
    """Render each decision card with clear separation of dimensions."""
    st.markdown("#### Decision Options")
    st.caption(
        "Each decision's priority, urgency, confidence, and impact are "
        "independent dimensions — a high-priority decision may have low "
        "confidence or low impact."
    )

    for i, card in enumerate(result.decision_cards):
        with st.expander(f"**{card.title or f'Decision {i+1}'}**", expanded=(i == 0)):
            if card.category:
                st.caption(f"Category: {card.category}")
            if card.description:
                st.markdown(card.description)

            # Separate dimension metrics
            dim_cols = st.columns(4)
            dim_cols[0].metric("Priority", card.priority or "—")
            dim_cols[1].metric("Urgency", card.urgency or "—")
            dim_cols[2].metric("Confidence", card.confidence or "—")
            dim_cols[3].metric("Reversibility", card.reversibility or "—")

            # Impact — separate section
            if card.impact_summary:
                st.markdown(f"**Impact Assessment:** {card.impact_summary}")
                st.caption("Impact is an estimate, not a guarantee. Actual outcomes depend on execution and external factors.")

            # Evidence provenance
            st.markdown(f"**Evidence:** {card.evidence_summary}")
            if card.evidence_ids:
                with st.expander("Evidence references"):
                    for eid in card.evidence_ids:
                        st.text(f"• {eid}")

            # Limitations
            if card.limitations:
                st.markdown("**Limitations:**")
                for lim in card.limitations:
                    st.caption(f"⚠ {lim}")

            # Warnings
            if card.warnings:
                for w in card.warnings:
                    st.warning(w)


def _render_empty_state(result: DashboardResult) -> None:
    """Honest empty state — explains why no decisions are available."""
    st.info(f"**No decisions available.** {result.empty_reason}")


def _render_scenarios(result: DashboardResult) -> None:
    """Show scenario results if explicitly evaluated."""
    if not result.has_scenarios:
        return
    st.markdown("#### Scenario Analysis Results")
    st.caption(
        "Scenario results are model-based projections, not observed facts or guaranteed outcomes. "
        "Actual outcomes may differ."
    )
    report = result.report
    if report and report.scenario_results:
        for sr in report.scenario_results:
            name = getattr(sr, "name", "Scenario")
            status = getattr(sr, "status", "")
            st.markdown(f"**{name}** — Status: {status}")


def _render_stages(result: DashboardResult) -> None:
    """Show pipeline stage statuses."""
    if not result.stage_displays:
        return
    with st.expander("Pipeline Stage Details"):
        for stage in result.stage_displays:
            name = stage.get("name", "Unknown")
            s_status = stage.get("status", {})
            label = s_status.get("label", "Unknown")
            level = s_status.get("level", "info")
            error = stage.get("error", "")

            icon = "✅" if level == "success" else ("⚠️" if level == "warning" else ("❌" if level == "error" else "ℹ️"))
            st.markdown(f"{icon} **{name}**: {label}")
            if error:
                st.error(f"Error: {error}")
            for w in stage.get("warnings", []):
                st.warning(w)


def _render_warnings_limitations(result: DashboardResult) -> None:
    """Render warnings and limitations."""
    if result.warnings:
        st.markdown("#### Warnings")
        for w in result.warnings:
            st.warning(w)

    if result.limitations:
        st.markdown("#### Known Limitations")
        st.caption("These limitations affect the reliability of the analysis.")
        for lim in result.limitations:
            st.caption(f"⚠ {lim}")

    if result.errors:
        st.markdown("#### Errors")
        for e in result.errors:
            st.error(e)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

@safe_page("Decision Intelligence")
def render():
    st.markdown('<div class="mp-header-badge">DECISION INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Decision Intelligence")
    st.caption(
        "AI-assisted analysis that synthesizes evidence from your data to surface "
        "decision options. Results are scoped to the currently active dataset."
    )

    # Check for cached result matching current dataset
    result = _load_cached_result()

    if result is None:
        st.info(
            "Decision Intelligence has not been run for this dataset yet. "
            "Click **Run Analysis** to generate decisions from your data."
        )
        if st.button("▶️ Run Analysis", type="primary"):
            with st.spinner("Running Decision Intelligence workflow..."):
                result = _run_and_cache()
            st.rerun()
        return

    # Show rerun button
    if st.button("🔄 Re-run Analysis"):
        with st.spinner("Re-running Decision Intelligence workflow..."):
            result = _run_and_cache()
        st.rerun()

    # Render the result
    _render_status_header(result)

    st.markdown("---")

    _render_confidence(result)

    st.markdown("---")

    if result.has_decisions:
        _render_decision_cards(result)
    else:
        _render_empty_state(result)

    _render_scenarios(result)

    st.markdown("---")

    _render_stages(result)
    _render_warnings_limitations(result)
