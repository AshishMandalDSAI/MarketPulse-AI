"""
MarketPulse AI - What-If Budget Simulator
=============================================
Reusable simulation engine that lets a user hypothesize a total budget
and/or individual channel allocations and see the fitted model's predicted
revenue, revenue change vs. current, ROI, and per-channel contribution —
all computed live from the actual fitted MMM (never hard-coded).

Run standalone for a demo scenario comparison:
    python src/optimization/simulator.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.optimization.budget_optimizer import (
    CHANNELS, load_bundle, get_recent_carry_in, predict_revenue_for_allocation,
    optimize_budget,
)

MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "marketing_data_clean.csv"


class BudgetSimulator:
    """Stateful simulator wrapping the fitted MMM for interactive what-if use
    (used directly by both the Streamlit dashboard and the FastAPI /simulate-budget
    endpoint, so results are always consistent).
    """

    def __init__(self):
        self.bundle = load_bundle()
        self.df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        self.channel_params = self.bundle["channel_params"]
        self.carry_in = get_recent_carry_in(self.df, self.channel_params)
        recent = self.df.tail(8)
        self.control_values = {c: float(recent[c].mean()) for c in self.bundle["control_cols"]}
        self.current_allocation = {ch: float(recent[f"{ch}_Spend"].mean()) for ch in CHANNELS}
        self.current_total_budget = sum(self.current_allocation.values())

    def predict(self, allocation: dict) -> float:
        alloc_vec = np.array([allocation.get(ch, 0.0) for ch in CHANNELS])
        return predict_revenue_for_allocation(alloc_vec, self.bundle, self.carry_in, self.control_values)

    def simulate_total_budget(self, total_budget: float, keep_current_mix: bool = True) -> dict:
        """Scale the current channel mix proportionally to a new total budget."""
        if keep_current_mix and self.current_total_budget > 0:
            scale = total_budget / self.current_total_budget
            allocation = {ch: v * scale for ch, v in self.current_allocation.items()}
        else:
            allocation = {ch: total_budget / len(CHANNELS) for ch in CHANNELS}

        predicted_revenue = self.predict(allocation)
        current_revenue = self.predict(self.current_allocation)
        return self._package_result("custom_total_budget", allocation, predicted_revenue, current_revenue, total_budget)

    def simulate_channel_allocation(self, allocation: dict) -> dict:
        """Simulate a fully custom, user-specified channel allocation."""
        total_budget = sum(allocation.values())
        predicted_revenue = self.predict(allocation)
        current_revenue = self.predict(self.current_allocation)
        return self._package_result("custom_allocation", allocation, predicted_revenue, current_revenue, total_budget)

    def scenario_comparison(self) -> pd.DataFrame:
        """Standard 4-scenario comparison: Current, Optimized, Aggressive (+20%),
        Conservative (-20%) — each evaluated through the actual fitted model.
        """
        scenarios = []

        current_revenue = self.predict(self.current_allocation)
        scenarios.append({
            "Scenario": "Current", "Budget": self.current_total_budget,
            "Predicted_Revenue": current_revenue, "Revenue_Change_Pct": 0.0,
            "ROI": current_revenue / self.current_total_budget if self.current_total_budget else 0.0,
        })

        opt = optimize_budget(
            self.current_total_budget, self.bundle, self.carry_in,
            self.control_values, self.current_allocation,
        )
        scenarios.append({
            "Scenario": "Optimized", "Budget": self.current_total_budget,
            "Predicted_Revenue": opt["optimized_predicted_revenue"],
            "Revenue_Change_Pct": opt["expected_improvement_pct"],
            "ROI": opt["optimized_efficiency"],
        })

        for label, mult in [("Aggressive (+20%)", 1.20), ("Conservative (-20%)", 0.80)]:
            budget = self.current_total_budget * mult
            res = self.simulate_total_budget(budget)
            scenarios.append({
                "Scenario": label, "Budget": budget,
                "Predicted_Revenue": res["predicted_revenue"],
                "Revenue_Change_Pct": res["revenue_change_pct"],
                "ROI": res["roi"],
            })

        return pd.DataFrame(scenarios)

    def _package_result(self, scenario_name, allocation, predicted_revenue, current_revenue, total_budget) -> dict:
        revenue_change = predicted_revenue - current_revenue
        revenue_change_pct = (revenue_change / current_revenue * 100) if current_revenue else 0.0
        roi = predicted_revenue / total_budget if total_budget else 0.0
        return {
            "scenario": scenario_name,
            "allocation": allocation,
            "total_budget": float(total_budget),
            "predicted_revenue": float(predicted_revenue),
            "current_revenue": float(current_revenue),
            "revenue_change": float(revenue_change),
            "revenue_change_pct": float(revenue_change_pct),
            "roi": float(roi),
        }


def main():
    sim = BudgetSimulator()
    comparison = sim.scenario_comparison()

    out_dir = ROOT / "data" / "processed"
    comparison.to_csv(out_dir / "scenario_comparison.csv", index=False)

    print("What-If Scenario Comparison")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print(f"\nWritten to: {out_dir / 'scenario_comparison.csv'}")


if __name__ == "__main__":
    main()
