"""Detects which analysis features are actually usable for a given
StandardizedDataset, each with a specific, plain-English reason."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import MIN_WEEKS_FOR_FORECAST
from src.data_engine.schemas import StandardizedDataset


@dataclass
class Capability:
    key: str
    name: str
    available: bool
    reason: str


@dataclass
class CapabilityReport:
    capabilities: list = field(default_factory=list)  # list[Capability]

    def is_available(self, key: str) -> bool:
        cap = self.get(key)
        return bool(cap and cap.available)

    def get(self, key: str):
        for c in self.capabilities:
            if c.key == key:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            c.key: {"name": c.name, "available": c.available, "reason": c.reason}
            for c in self.capabilities
        }


def detect_capabilities(dataset: StandardizedDataset) -> CapabilityReport:
    caps = []
    n_rows = len(dataset.df) if dataset.df is not None else 0

    caps.append(Capability("data_profiling", "Data Profiling", True,
                            "Always available once a dataset is loaded."))

    has_rows = n_rows > 0
    caps.append(Capability("generic_kpis", "Generic KPIs", has_rows,
                            "Computed from the uploaded columns." if has_rows else "Dataset has no rows."))
    caps.append(Capability("ai_analyst", "AI Analyst", has_rows,
                            "Can answer questions about the uploaded data." if has_rows else "Dataset has no rows."))
    caps.append(Capability("business_report", "Business Report", has_rows,
                            "Can generate a narrative report from the uploaded data." if has_rows else "Dataset has no rows."))

    has_dimension = len(dataset.dimension_cols) > 0
    caps.append(Capability(
        "category_analysis", "Category Analysis", has_dimension,
        f"Found {len(dataset.dimension_cols)} categorical column(s) to break down by."
        if has_dimension else "No categorical/dimension column was identified in this dataset."
    ))

    has_date = dataset.has_date
    caps.append(Capability(
        "trend_analysis", "Trend Analysis", has_date,
        f"Date column '{dataset.date_col}' identified." if has_date
        else "No date/time column was identified or confirmed for this dataset."
    ))

    has_numeric_target = bool(dataset.revenue_col) or bool(dataset.control_cols)
    n_points = n_rows
    if has_date and dataset.df is not None and dataset.date_col in dataset.df.columns:
        # Forecasts run on one value per period, so count distinct dates, not rows.
        import pandas as _pd
        n_points = int(_pd.to_datetime(dataset.df[dataset.date_col], errors="coerce").dropna().nunique())
    forecast_ok = has_date and has_numeric_target and n_points >= MIN_WEEKS_FOR_FORECAST
    if not has_date:
        forecast_reason = "No date/time column identified — forecasting requires a time axis."
    elif not has_numeric_target:
        forecast_reason = "No numeric target column (revenue or a generic numeric measure) was identified."
    elif n_points < MIN_WEEKS_FOR_FORECAST:
        forecast_reason = (f"Only {n_points} time point(s) available; at least "
                            f"{MIN_WEEKS_FOR_FORECAST} are recommended for a reliable forecast.")
    else:
        target = dataset.revenue_col or dataset.control_cols[0]
        forecast_reason = f"Date column '{dataset.date_col}' and numeric target '{target}' identified with {n_points} time points."
    caps.append(Capability("forecasting", "Forecasting", forecast_ok, forecast_reason))

    mmm_ok, mmm_missing = dataset.mmm_readiness()
    mmm_reason = "Date, revenue, and enough channel/spend columns were identified." if mmm_ok else \
        "Marketing Mix Modeling unavailable: " + "; ".join(mmm_missing) + "."
    caps.append(Capability("marketing_mix_modeling", "Marketing Mix Modeling", mmm_ok, mmm_reason))
    caps.append(Capability("channel_attribution", "Channel Attribution", mmm_ok,
                            mmm_reason if not mmm_ok else "Derived from the trained Marketing Mix Model."))
    caps.append(Capability("model_diagnostics", "Model Diagnostics", mmm_ok,
                            mmm_reason if not mmm_ok else "Derived from the trained Marketing Mix Model."))

    roi_ok = dataset.has_revenue and len(dataset.channels) >= 1
    if roi_ok:
        roi_reason = f"Revenue column '{dataset.revenue_col}' and {len(dataset.channels)} spend column(s) identified."
    else:
        missing = []
        if not dataset.has_revenue:
            missing.append("no revenue column identified")
        if len(dataset.channels) < 1:
            missing.append("no marketing spend column identified")
        roi_reason = "ROI/ROAS unavailable: " + "; ".join(missing) + "."
    caps.append(Capability("roi_roas", "ROI / ROAS", roi_ok, roi_reason))

    caps.append(Capability("budget_optimization", "Budget Optimization", mmm_ok,
                            mmm_reason if not mmm_ok else "Requires a trained Marketing Mix Model, which is available."))
    caps.append(Capability("what_if_simulation", "What-If Simulation", mmm_ok,
                            mmm_reason if not mmm_ok else "Requires a trained Marketing Mix Model, which is available."))

    # Decision Intelligence — available when upstream analysis stages can
    # produce evidence (requires rows for at least generic KPI computation).
    # This gate is purely structural — it does not depend on the outcome
    # of any particular decision workflow.
    di_ok = has_rows
    caps.append(Capability(
        "decision_intelligence", "Decision Intelligence", di_ok,
        "Upstream analysis stages can produce evidence for decision assessment."
        if di_ok else "Dataset has no rows — upstream analysis cannot produce evidence."
    ))

    return CapabilityReport(capabilities=caps)
