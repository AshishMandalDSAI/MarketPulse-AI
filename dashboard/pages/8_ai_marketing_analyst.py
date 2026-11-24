"""AI Marketing Analyst page — chat-style Q&A grounded in project analytics."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard import navigation as _nav
from dashboard.components.graceful import safe_page

EXAMPLE_QUESTIONS = [
    "Which marketing channel should receive more budget?",
    "Which channel has the best efficiency?",
    "What happens if Search spending increases by 20%?",
    "What is the recommended budget allocation?",
    "Which channel should management reduce?",
]


@safe_page("AI Marketing Analyst")
def render():
    st.markdown('<div class="mp-header-badge">AI MARKETING ANALYST</div>', unsafe_allow_html=True)
    st.title("AI Marketing Analyst")

    from src.ai.analyst import MarketingAnalyst
    from dashboard.components import session_paths

    # Pass decision context scoped to the active dataset (Phase 7 Part 7)
    decision_ctx = st.session_state.get("decision_copilot_context")
    dataset_id = session_paths.get_dataset_id()

    analyst = MarketingAnalyst(
        data_dir=session_paths.get_processed_dir(),
        decision_context=decision_ctx,
        dataset_id=dataset_id,
    )

    st.markdown(
        "<div class='mp-callout'><div class='mp-callout-label'>Rule-based analyst · offline</div>"
        "Answers come from this dataset's <b>computed results</b> (KPIs, model estimates, optimizer and "
        "Decision Intelligence outputs) using a fixed set of recognised questions. It is <b>not</b> a large "
        "language model: it won't understand arbitrary questions, and it never sends your data anywhere. "
        "Each answer says whether it is an observed fact, a model estimate, a scenario or a recommendation."
        "</div>", unsafe_allow_html=True)
    if analyst.api_key_available:
        st.caption("An ANTHROPIC_API_KEY is configured, but this version does not use an LLM — answers are "
                   "still produced by the rule-based engine.")
    if not analyst.is_demo:
        st.caption("Uploaded dataset active: answers use only this dataset's results. Demo-only spend scenarios "
                   f"are switched off; use **{_nav.label('prescriptive')}** for allocation scenarios.")

    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked_question = None
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(q, key=f"eg_{q}", use_container_width=True):
                clicked_question = q

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_question = st.chat_input("Ask about channel performance, budget allocation, or scenarios...")
    question = clicked_question or user_question

    if question:
        result = analyst.answer(question)
        st.session_state["chat_history"].append(result)

    for entry in reversed(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            kind = entry.get("evidence_type")
            if kind:
                src = ", ".join(entry.get("sources") or []) or "—"
                st.caption(f"Evidence type: **{kind}** · Source: {src}")
