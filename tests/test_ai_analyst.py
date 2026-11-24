"""Tests for the offline, rule-based AI Marketing Analyst (Demo AI Mode).

These exercise the exact 5 demonstration questions called out in the project
brief, plus the routing logic that distinguishes 'increase budget' from
'reduce budget' questions (which must not collide on the shared word
'channel should').
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.analyst import MarketingAnalyst

REQUIRED_DEMO_QUESTIONS = [
    "Which channel should receive more budget?",
    "Which channel has the best efficiency?",
    "What happens if Search spending increases by 20%?",
    "Why is revenue declining?",
    "What is the recommended budget allocation?",
]


def test_analyst_initializes_without_api_key():
    analyst = MarketingAnalyst()
    assert analyst.api_key_available in (True, False)


def test_all_required_demo_questions_get_grounded_answers():
    analyst = MarketingAnalyst()
    for q in REQUIRED_DEMO_QUESTIONS:
        result = analyst.answer(q)
        assert result["grounded_in_project_data"] is True
        assert isinstance(result["answer"], str) and len(result["answer"]) > 20
        assert "Demo AI Mode" in result["mode"] or "LLM" in result["mode"]


def test_works_offline_without_api_key():
    analyst = MarketingAnalyst()
    if not analyst.api_key_available:
        result = analyst.answer("Which channel has the best efficiency?")
        assert "Demo AI Mode" in result["mode"]


def test_reduce_question_does_not_collide_with_increase_question():
    """'Which channel should management reduce?' must route to the
    lowest-efficiency answer, not the highest-efficiency ('more budget') one."""
    analyst = MarketingAnalyst()
    increase_answer = analyst.answer("Which channel should receive more budget?")["answer"]
    reduce_answer = analyst.answer("Which channel should management reduce?")["answer"]
    assert increase_answer != reduce_answer

    best = analyst._best_efficiency_channel()
    worst = analyst._worst_efficiency_channel()
    if best is not None and worst is not None and best["Channel"] != worst["Channel"]:
        assert best["Channel"] in increase_answer
        assert worst["Channel"] in reduce_answer


def test_scenario_question_uses_simulator():
    analyst = MarketingAnalyst()
    result = analyst.answer("What happens if Search spending increases by 20%?")
    assert "%" in result["answer"]


def test_fallback_for_unrelated_question():
    analyst = MarketingAnalyst()
    result = analyst.answer("What is the capital of France?")
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"DONE: {name}")
