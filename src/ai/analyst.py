"""
MarketPulse AI - AI Marketing Analyst
=========================================
Answers natural-language business questions about marketing performance
using the project's ACTUAL computed analytics/model outputs.

- Deterministic, rule-based and fully offline ("Demo AI Mode"): questions
  are matched to a fixed set of intents by keywords and answered from the
  active dataset's computed results. It is NOT a large language model: it
  does not understand arbitrary questions and never sends data anywhere.
- This version does not call any LLM, even if ANTHROPIC_API_KEY is set
  (the key is detected only so the UI can say so explicitly).
- Every answer carries an ``evidence_type`` (observed fact, model estimate,
  model scenario, recommendation, guidance) and the files it was built from.
- Answers never mix datasets: demo-only engines (the fitted demo MMM
  simulator) are not used while an uploaded dataset is active.

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
MODE_LABEL = "Demo AI Mode (rule-based, offline)"
ASSOCIATION_NOTE = ("model estimate from historical data — a statistical association, "
                    "not a measured incremental (causal) effect")


def _load_json(name, data_dir=None):
    path = (data_dir or DATA_DIR) / name
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _load_csv(name, data_dir=None):
    path = (data_dir or DATA_DIR) / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


class MarketingAnalyst:
    """Rule-based analyst grounded in the project's computed analytics.

    `data_dir` scopes every load to one dataset's processed directory — by
    default the built-in demo dataset, but an uploaded dataset's directory
    can be passed to keep answers scoped to that dataset only (no
    cross-dataset leakage).

    `decision_context` accepts a dict built by
    ``dashboard_service.build_copilot_context()`` so decision-related
    questions can be answered from the active dataset's decision results.
    The context is validated against ``dataset_id`` — a mismatch is
    silently treated as absent to prevent cross-dataset leakage."""

    def __init__(self, data_dir=None, decision_context=None, dataset_id=None):
        self.data_dir = data_dir or DATA_DIR
        self.kpi_bundle = _load_json("marketing_kpis.json", self.data_dir)
        self.attribution = _load_csv("channel_attribution.csv", self.data_dir)
        self.channel_performance = _load_csv("channel_performance.csv", self.data_dir)
        self.budget_opt = _load_json("budget_optimization_summary.json", self.data_dir)
        self.forecast = _load_csv("revenue_forecast.csv", self.data_dir)
        self.generic_kpis = _load_json("generic_kpis.json", self.data_dir)
        self.api_key_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.llm_enabled = False  # no LLM integration in this version (see module docstring)
        try:
            from config.settings import DEMO_DATASET_ID
        except Exception:  # pragma: no cover
            DEMO_DATASET_ID = "demo"
        self.is_demo = dataset_id in (None, "", DEMO_DATASET_ID) and Path(self.data_dir) == Path(DATA_DIR)
        self._last_evidence = ("guidance", [])
        # Currency (upgrade PART 01): the demo is INR; an upload uses the currency stored
        # by the Upload page, or none. INR is never assumed for an upload.
        if self.is_demo:
            self.cur, self.per_unit = "₹", "₹1"
        else:
            try:
                from src.data_engine.currency import load_currency
                info = load_currency(Path(self.data_dir)) or {}
            except Exception:
                info = {}
            self.cur = info.get("symbol") or ""
            self.per_unit = f"{self.cur}1" if self.cur else "unit of currency"

        # Decision Intelligence context — validated for dataset match
        self._decision_context = None
        if decision_context and dataset_id:
            ctx_dataset = decision_context.get("dataset_id", "")
            if ctx_dataset == dataset_id:
                self._decision_context = decision_context

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
        self._last_evidence = ("recommendation", ["channel_attribution.csv"])
        best = self._best_efficiency_channel()
        if best is None:
            return "Attribution data isn't available yet — run the MMM and channel_contribution pipeline first."
        return (
            f"Based on the fitted model, **{best['Channel']}** currently shows the highest "
            f"model-estimated efficiency ({self.cur}{best['Model_Estimated_Efficiency']:.2f} of revenue per {self.per_unit} spent — "
            f"{ASSOCIATION_NOTE}), with an estimated {best['Contribution_Pct']:.1f}% of attributed "
            f"revenue from {self.cur}{best['Spend']:,.0f} in spend. Shifting budget toward it is a candidate action; "
            f"validate it with a controlled test before a large reallocation."
        )

    def best_efficiency_channel_answer(self) -> str:
        self._last_evidence = ("model estimate", ["channel_attribution.csv"])
        best = self._best_efficiency_channel()
        if best is None:
            return "Attribution data isn't available yet."
        text = (f"**{best['Channel']}** has the best model-estimated efficiency at "
                f"{self.cur}{best['Model_Estimated_Efficiency']:.2f} of revenue per {self.per_unit} spent ({ASSOCIATION_NOTE}).")
        if "Observed_Naive_ROAS" in best.index and pd.notna(best.get("Observed_Naive_ROAS")):
            text += (f" Its naive observed ROAS is {best['Observed_Naive_ROAS']:.2f}. The two can diverge because "
                     f"the model accounts for carryover and diminishing returns, while naive ROAS is a simple "
                     f"revenue ÷ spend ratio.")
        return text

    def worst_efficiency_channel_answer(self) -> str:
        self._last_evidence = ("recommendation", ["channel_attribution.csv"])
        worst = self._worst_efficiency_channel()
        if worst is None:
            return "Attribution data isn't available yet."
        return (
            f"**{worst['Channel']}** currently shows the lowest model-estimated efficiency, at "
            f"{self.cur}{worst['Model_Estimated_Efficiency']:.2f} of model-estimated revenue per {self.per_unit} spent "
            f"({self.cur}{worst['Spend']:,.0f} in spend, {worst['Contribution_Pct']:.1f}% of attributed revenue). "
            f"This may reflect genuine diminishing returns at current spend levels, or a targeting/creative "
            f"issue worth investigating before cutting spend — the Budget Optimizer can quantify the "
            f"revenue impact of reallocating away from this channel."
        )

    def channel_scenario_answer(self, channel: str, pct_change: float) -> str:
        self._last_evidence = ("model scenario", ["mmm_model.pkl", "simulator"])
        if not self.is_demo:
            self._last_evidence = ("guidance", [])
            return ("Spend scenarios here use the demo dataset's fitted response model, so they are not run while "
                    "an uploaded dataset is active (that would describe the demo data, not yours). For your "
                    "data, use the **Prescriptive Analytics** page, which fits a response model on the active "
                    "dataset and simulates allocations from it.")
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
            f"predicts weekly revenue of {self.cur}{result['predicted_revenue']:,.0f}, a change of "
            f"{result['revenue_change_pct']:+.2f}% versus the current allocation "
            f"({self.cur}{result['current_revenue']:,.0f}). This is a model-based projection under the fitted "
            f"saturation curve, holding other channels' spend and prior carryover constant."
        )

    def budget_allocation_answer(self) -> str:
        self._last_evidence = ("recommendation", ["budget_optimization_summary.json"])
        if not self.budget_opt:
            return "Budget optimization hasn't been run yet — run src/optimization/budget_optimizer.py first."
        imp = self.budget_opt.get("expected_improvement_pct", 0.0)
        return (
            f"The optimizer recommends reallocating the current {self.cur}{self.budget_opt['total_budget']:,.0f} weekly "
            f"budget across channels to maximize model-predicted revenue. This is projected to improve predicted "
            f"revenue from {self.cur}{self.budget_opt['current_predicted_revenue']:,.0f} to "
            f"{self.cur}{self.budget_opt['optimized_predicted_revenue']:,.0f} — an estimated {imp:+.1f}% improvement, "
            f"under the current per-channel min/max spend bounds. This is a model estimate, not a guaranteed "
            f"outcome. See the Budget Optimizer page for the full channel-by-channel recommendation."
        )

    def revenue_trend_answer(self) -> str:
        self._last_evidence = ("observed fact", ["marketing_kpis.json"])
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

    def decision_intelligence_answer(self, question: str) -> str:
        """Answer a decision-intelligence question from available context.

        Never fabricates decisions or impact estimates. Distinguishes
        facts from inferences, predictions, and recommendations.
        Communicates confidence and limitations accurately."""
        if self._decision_context is None:
            return (
                "Decision Intelligence has not been run for this dataset, "
                "or no results are available. Go to the Decision Intelligence "
                "page and click 'Run Analysis' first."
            )

        ctx = self._decision_context
        if not ctx.get("has_decisions"):
            reason = ctx.get("empty_reason", "No decision results available.")
            status = ctx.get("status", "Not Run")
            return (
                f"Decision Intelligence status: {status}. {reason} "
                "No recommendations can be provided without sufficient evidence."
            )

        from src.decision.dashboard_service import format_copilot_answer_context
        return format_copilot_answer_context(ctx)

    def _is_decision_question(self, q: str) -> bool:
        """Detect whether a question is about decision intelligence."""
        decision_keywords = [
            "decision", "decisions", "decision intelligence",
            "evidence strength", "confidence level",
            "what should we do", "what action",
            "priority", "urgency", "impact assessment",
            "recommendations from decision", "decision options",
            "orchestrat",
        ]
        return any(kw in q for kw in decision_keywords)

    def fallback_answer(self, question: str) -> str:
        self._last_evidence = ("guidance", [])
        text = ("I'm a rule-based analyst, not a general-purpose chatbot: I recognise a fixed set of questions "
                "about channel efficiency, budget allocation, spend scenarios, revenue trends and Decision "
                "Intelligence results, and answer them from this dataset's computed analytics. Try 'which channel "
                "should get more budget?', 'which channel has the best efficiency?', 'what is the recommended "
                "budget allocation?' or 'why is revenue declining?'.")
        kpis = self.kpi_bundle.get("kpis", {}) if self.is_demo else {}
        if kpis.get("total_revenue") is not None:
            text += (f"\n\nQuick snapshot (observed): total revenue {self.cur}{kpis['total_revenue']:,.0f} across "
                     f"{kpis.get('n_weeks', '?')} weeks, overall naive ROAS {kpis.get('overall_naive_roas', 0):.2f}.")
        elif isinstance(self.generic_kpis, list) and self.generic_kpis:
            facts = [f"{k.get('label')}: {k.get('value')}" for k in self.generic_kpis[:3]
                     if isinstance(k, dict) and k.get("label")]
            if facts:
                text += "\n\nQuick snapshot (observed): " + "; ".join(facts) + "."
        return text

    # ---- routing ------------------------------------------------------

    def answer(self, question: str) -> dict:
        q = question.lower()
        self._last_evidence = ("guidance", [])

        # Decision Intelligence questions — checked first
        if self._is_decision_question(q):
            text = self.decision_intelligence_answer(question)
            return {
                "question": question,
                "answer": text,
                "mode": MODE_LABEL,
                "llm_used": False,
                "evidence_type": "recommendation" if self._decision_context else "guidance",
                "sources": ["Decision Intelligence results"] if self._decision_context else [],
                "grounded_in_project_data": True,
            }

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

        evidence_type, sources = self._last_evidence
        return {
            "question": question,
            "answer": text,
            "mode": MODE_LABEL,
            "llm_used": False,
            "evidence_type": evidence_type,
            "sources": sources,
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
