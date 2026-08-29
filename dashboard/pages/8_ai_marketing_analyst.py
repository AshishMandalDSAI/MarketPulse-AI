"""AI Marketing Analyst page — chat-style Q&A grounded in project analytics."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXAMPLE_QUESTIONS = [
    "Which marketing channel should receive more budget?",
    "Which channel has the best efficiency?",
    "What happens if Search spending increases by 20%?",
    "What is the recommended budget allocation?",
    "Which channel should management reduce?",
]


def render():
    st.markdown('<div class="mp-header-badge">AI MARKETING ANALYST</div>', unsafe_allow_html=True)
    st.title("AI Marketing Analyst")

    from src.ai.analyst import MarketingAnalyst
    analyst = MarketingAnalyst()

    mode_label = "🟢 Demo AI Mode (offline, rule-based)" if not analyst.api_key_available else "🔵 LLM-Assisted Mode"
    st.markdown(f"<div class='mp-callout'><b>{mode_label}</b> — answers are always grounded in this project's "
                f"own computed analytics and model outputs, whether or not an LLM API key is configured.</div>",
                unsafe_allow_html=True)

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
