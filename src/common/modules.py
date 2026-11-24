"""Registry of the Phase 8 analytical modules (14-17).

Single source of truth for each module's plain-English overview and delivery
status. Menu placement comes from ``dashboard/navigation.py``; ``nav_label``
returns the menu entry hosting the module (enforced by tests).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    number: int
    icon: str
    title: str
    tagline: str
    overview: str            # simple-English explanation
    capabilities: tuple      # what the module provides once delivered
    page_module: str         # importable dashboard page module
    delivered_in_part: int   # Phase 8 implementation part that delivers the analytics
    implemented: bool = False  # flipped to True by the part that delivers the analytics

    @property
    def nav_label(self) -> str:
        """Menu entry that hosts this module. Since upgrade PART 01 the descriptive,
        inferential and predictive modules are tabs of "01 · Statistics", and
        Prescriptive Analytics is menu entry 15; ``number`` (14-17) remains the
        module's permanent identity."""
        from dashboard.navigation import page_for_module
        return page_for_module(self.key)[0]

    @property
    def stats_tab(self):
        """Tab label on the Statistics page, or None for a stand-alone page."""
        from dashboard.navigation import page_for_module
        return page_for_module(self.key)[1]


MODULE_SPECS = (
    ModuleSpec(
        key="descriptive", number=14, icon="📊", title="Descriptive Statistics",
        tagline="Understand what your data looks like.",
        overview=("Summarizes every column of your dataset: typical values, how spread out "
                  "they are, how they are shaped, which values are missing or unusual, and "
                  "how columns move together."),
        capabilities=("Mean, median, mode, spread, quartiles and percentiles",
                      "Skewness, kurtosis, correlation and covariance",
                      "Frequency tables, value counts, missing-value and outlier summaries",
                      "Histograms, box, violin and density plots, heatmaps, scatter and bar charts",
                      "CSV and Excel summary downloads"),
        page_module="dashboard.pages.14_descriptive_statistics", delivered_in_part=2,
        implemented=True,
    ),
    ModuleSpec(
        key="inferential", number=15, icon="🔬", title="Inferential Statistics",
        tagline="Test whether what you see is likely real or just chance.",
        overview=("Runs standard hypothesis tests on your own data, checks that each test's "
                  "assumptions hold, and explains the result in plain English. A statistically "
                  "significant result is not automatically a business-important one, and a "
                  "correlation never proves cause."),
        capabilities=("One-sample, independent and paired t-tests; Z-test when valid",
                      "One-way ANOVA, chi-square, Pearson and Spearman correlation",
                      "Confidence intervals, effect sizes and assumption checks",
                      "Clear explanations when a test is unsuitable for the chosen columns"),
        page_module="dashboard.pages.15_inferential_statistics", delivered_in_part=3,
        implemented=True,
    ),
    ModuleSpec(
        key="predictive", number=16, icon="🔮", title="Predictive Analysis",
        tagline="Train and compare models on your own data.",
        overview=("Lets you choose a target column, checks that prediction is sensible for the "
                  "dataset, trains and compares regression, classification and clustering "
                  "models with leakage protection, and reports honest accuracy measured on "
                  "data the model did not see."),
        capabilities=("Linear, multiple linear, decision tree and random forest regression",
                      "Logistic, decision tree and random forest classification",
                      "K-Means clustering and customer segmentation",
                      "Train/test split, cross-validation, model comparison, feature importance",
                      "Prediction export"),
        page_module="dashboard.pages.16_predictive_analytics", delivered_in_part=4,
        implemented=True,
    ),
    ModuleSpec(
        key="prescriptive", number=17, icon="🎯", title="Prescriptive Analytics",
        tagline="Turn findings into decisions you can act on.",
        overview=("Uses the platform's existing optimizer, simulator and decision engines to "
                  "recommend how to allocate budget or resources under your constraints. "
                  "Every outcome is an estimate from a fitted model, never a guaranteed result."),
        capabilities=("Goal-based and budget allocation optimization under constraints",
                      "Profit maximization and cost minimization",
                      "What-if and scenario comparison",
                      "Risk-aware recommendations with expected impact and alternatives"),
        page_module="dashboard.pages.17_prescriptive_analytics", delivered_in_part=5,
        implemented=True,
    ),
)

_BY_KEY = {m.key: m for m in MODULE_SPECS}


def get_module_spec(key: str) -> ModuleSpec:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown Phase 8 module '{key}'. Known: {sorted(_BY_KEY)}") from None
