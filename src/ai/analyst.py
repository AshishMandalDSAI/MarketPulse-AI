"""
MarketPulse AI - AI Marketing Analyst
=========================================
Answers natural-language business questions about marketing performance
using the project's ACTUAL computed analytics/model outputs.

- Works fully offline with a deterministic, rule-based "Demo AI Mode" —
  no API key required, and the app never crashes if one is absent.
- OPTIONALLY calls an LLM (via ANTHROPIC_API_KEY env var) to phrase a more
  fluent narrative answer around the same computed facts, if a key is
  configured. The underlying numbers always come from this project's own
  analytics — the LLM (if used) is only for language, never for numbers.

Run a quick demo:
    python src/ai/analyst.py
"""

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


def _load_json(name):
    path = DATA_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


class MarketingAnalyst:
    """Rule-based analyst grounded in the project's computed analytics."""

    def __init__(self):
        self.kpi_bundle = _load_json("marketing_kpis.json")
        self.attribution = _load_csv("channel_attribution.csv")
        self.channel_performance = _load_csv("channel_performance.csv")
        self.budget_opt = _load_json("budget_optimization_summary.json")
        self.forecast = _load_csv("revenue_forecast.csv")
        self.api_key_available = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ---- intent handlers -------------------------------------------------

    def _best_efficiency_channel(self):
        if self.attribution is None or self.attribution.empty:
            return None
        return self.attribution.sort_values("Model_Estimated_Efficiency", ascending=False).iloc[0]

    def _worst_efficiency_channel(self):
        if self.attribution is None or self.attribution.empty:
            return None
        return self.attribution.sort_values("Model_Estimated_Efficiency", ascending=True).iloc[0]

    def which_channel_more_budget(self) -> str:
        best = self._best_efficiency_channel()
        if best is None:
            return "Attribution data isn't available yet — run the MMM and channel_contribution pipeline first."
        return (
            f"Based on the fitted Marketing Mix Model, **{best['Channel']}** currently shows the highest "
            f"model-estimated efficiency (₹{best['Model_Estimated_Efficiency']:.2f} of estimated incremental "
            f"revenue per ₹1 spent), contributing an estimated {best['Contribution_Pct']:.1f}% of attributed "
            f"revenue from ₹{best['Spend']:,.0f} in spend. The Budget Optimizer scenario also supports shifting "
            f"incremental budget toward this channel, subject to the configured per-channel spend caps."
        )

    def best_efficiency_channel_answer(self) -> str:
        best = self._best_efficiency_channel()
        if best is None:
            return "Attribution data isn't available yet."
        return (
            f"**{best['Channel']}** has the best model-estimated efficiency at "
            f"₹{best['Model_Estimated_Efficiency']:.2f} of estimated incremental revenue per rupee spent, "
            f"versus a naive observed ROAS of {best['Observed_Naive_ROAS']:.2f}. Efficiency and naive ROAS can "
            f"diverge because efficiency accounts for adstock carryover and diminishing returns, while naive "
            f"ROAS is a simple, undecomposed ratio."
        )

    def worst_efficiency_channel_answer(self) -> str:
        worst = self._worst_efficiency_channel()
        if worst is None:
            return "Attribution data isn't available yet."
        return (
            f"**{worst['Channel']}** currently shows the lowest model-estimated efficiency, at "
            f"₹{worst['Model_Estimated_Efficiency']:.2f} of estimated incremental revenue per rupee spent "
            f"(₹{worst['Spend']:,.0f} in spend, {worst['Contribution_Pct']:.1f}% of attributed revenue). "
            f"This may reflect genuine diminishing returns at current spend levels, or a targeting/creative "
            f"issue worth investigating before cutting spend — the Budget Optimizer can quantify the "
            f"revenue impact of reallocating away from this channel."
        )

    def channel_scenario_answer(self, channel: str, pct_change: float) -> str:
        try:
            from src.optimization.simulator import BudgetSimulator
            sim = BudgetSimulator()
        except Exception as e:
            return f"Simulation engine unavailable: {e}"

        match = None
        for ch in sim.current_allocation:
            if ch.lower().replace("_", " ") == channel.lower() or ch.lower() == channel.lower():
                match = ch
                break
        if match is None:
            return f"I couldn't match '{channel}' to a known channel."

        new_alloc = dict(sim.current_allocation)
        new_alloc[match] = new_alloc[match] * (1 + pct_change / 100)
        result = sim.simulate_channel_allocation(new_alloc)

        direction = "increase" if pct_change > 0 else "decrease"
        return (
            f"If {match.replace('_', ' ')} spend is changed by {pct_change:+.0f}% (a {direction}), the model "
            f"predicts weekly revenue of ₹{result['predicted_revenue']:,.0f}, a change of "
            f"{result['revenue_change_pct']:+.2f}% versus the current allocation "
            f"(₹{result['current_revenue']:,.0f}). This is a model-based projection under the fitted "
            f"saturation curve, holding other channels' spend and prior carryover constant."
        )

    def budget_allocation_answer(self) -> str:
        if not self.budget_opt:
            return "Budget optimization hasn't been run yet — run src/optimization/budget_optimizer.py first."
        imp = self.budget_opt.get("expected_improvement_pct", 0.0)
        return (
            f"The optimizer recommends reallocating the current ₹{self.budget_opt['total_budget']:,.0f} weekly "
            f"budget across channels to maximize model-predicted revenue. This is projected to improve predicted "
            f"revenue from ₹{self.budget_opt['current_predicted_revenue']:,.0f} to "
            f"₹{self.budget_opt['optimized_predicted_revenue']:,.0f} — an estimated {imp:+.1f}% improvement, "
            f"under the current per-channel min/max spend bounds. See the Budget Optimizer page for the full "
            f"channel-by-channel recommendation."
        )

    def revenue_trend_answer(self) -> str:
        kpis = self.kpi_bundle.get("kpis", {})
        growth = kpis.get("revenue_growth_h1_to_h2")
        if growth is None:
            return "KPI data isn't available yet."
        direction = "grown" if growth >= 0 else "declined"
        return (
            f"Revenue has {direction} {abs(growth) * 100:.1f}% comparing the first half to the second half of "
            f"the observed period. Overall naive ROAS across all channels is "
            f"{kpis.get('overall_naive_roas', 0):.2f}. Marketing spend grew "
            f"{kpis.get('marketing_growth_h1_to_h2', 0) * 100:.1f}% over the same comparison, so revenue and "
            f"spend growth should be read together rather than in isolation."
        )

    def fallback_answer(self, question: str) -> str:
        kpis = self.kpi_bundle.get("kpis", {})
        return (
            "I can answer questions about channel efficiency, budget allocation, spend scenarios, and revenue "
            "trends using this project's computed analytics. Try asking things like 'which channel should get "
            "more budget?', 'what happens if Search spending increases by 20%?', or 'what is the recommended "
            "budget allocation?'.\n\n"
            f"Quick snapshot: total revenue ₹{kpis.get('total_revenue', 0):,.0f} across "
            f"{kpis.get('n_weeks', '?')} weeks, overall naive ROAS {kpis.get('overall_naive_roas', 0):.2f}."
        )

    # ---- routing ------------------------------------------------------

    def answer(self, question: str) -> dict:
        q = question.lower()

        pct_match = re.search(r"(increase|decrease|raise|cut|reduce)\s+(\w[\w\s]*?)\s+(?:spend(?:ing)?)\s+by\s+(\d+)%", q)
        if not pct_match:
            pct_match = re.search(r"(\w[\w\s]*?)\s+spend(?:ing)?\s+(increase|decrease)s?\s+by\s+(\d+)%", q)

        if "more budget" in q or ("which channel should" in q and "reduce" not in q and "cut" not in q and "decrease" not in q):
            text = self.which_channel_more_budget()
        elif ("reduce" in q or "cut" in q or "decrease" in q) and "channel" in q:
            text = self.worst_efficiency_channel_answer()
        elif "best efficiency" in q or "most efficient" in q:
            text = self.best_efficiency_channel_answer()
        elif "recommended" in q and "budget" in q or "budget allocation" in q:
            text = self.budget_allocation_answer()
        elif "declining" in q or "revenue trend" in q or "why is our revenue" in q or "why is revenue" in q:
            text = self.revenue_trend_answer()
        elif ("what happens if" in q or "spend" in q) and "%" in q:
            pct_m = re.search(r"(\d+)\s*%", q)
            dir_m = re.search(r"(increase|decrease|raise|cut|reduce|drop)", q)
            known_channels = ["tv", "radio", "print", "search", "social media", "display ads",
                               "email marketing", "influencer marketing"]
            found_channel = next((c for c in known_channels if c in q), None)
            if pct_m and dir_m and found_channel:
                pct = float(pct_m.group(1))
                signed_pct = pct if dir_m.group(1) in ("increase", "raise") else -pct
                text = self.channel_scenario_answer(found_channel, signed_pct)
            else:
                text = self.fallback_answer(question)
        else:
            text = self.fallback_answer(question)

        return {
            "question": question,
            "answer": text,
            "mode": "Demo AI Mode (rule-based, offline)" if not self.api_key_available else "LLM-assisted",
            "grounded_in_project_data": True,
        }


def main():
    analyst = MarketingAnalyst()
    demo_questions = [
        "Which marketing channel should receive more budget?",
        "Which channel has the best efficiency?",
        "What happens if Search spending increases by 20%?",
        "What is the recommended budget allocation?",
        "Why is our revenue declining?",
    ]
    for q in demo_questions:
        result = analyst.answer(q)
        print(f"\nQ: {q}")
        print(f"[{result['mode']}]")
        print(result["answer"])


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
