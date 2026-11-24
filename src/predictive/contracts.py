"""Constants, settings validation and result containers for Module 16."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SEED = 42
MIN_LABELLED = 20            # supervised: refuse below this many labelled rows (A1)
WARN_LABELLED = 100
STRONG_WARN_LABELLED = 50
MIN_CLASS_ROWS = 3           # refuse if a class has fewer labelled rows
WARN_CLASS_ROWS = 10
MAX_CLASSES = 20
ROW_CAP = 20_000             # labelled rows analysed (train + test), D4/A3
MIN_TEST_WARN = 30
MIN_TEST_FOR_CI = 20         # bootstrap omitted below this (A8)
BOOTSTRAP_N = 1000
MIN_CHRONO_BLOCK = 10        # A2
MIN_CHRONO_FOLDS = 2
MAX_CARDINALITY = 50
MAX_FEATURES = 200
ID_RATIO = 0.9
TIER2_DISPLAY = 0.95
TIER2_SINGLE_FEATURE = 0.98
TIER2_MAX_CHECKED = 30
SENSITIVITY_MAX = 10
IMPORTANCE_REPEATS = 5
MIN_ROWS_SEGMENTATION = 10
MIN_FEATURES_SEGMENTATION = 2
MAX_K = 10
SILHOUETTE_SAMPLE = 5000
STABILITY_RUNS = 5

REGRESSION_MODELS = ("simple_linear", "multiple_linear", "decision_tree", "random_forest")
CLASSIFICATION_MODELS = ("logistic", "decision_tree", "random_forest")
MODEL_LABELS = {
    "simple_linear": "Simple Linear Regression", "multiple_linear": "Multiple Linear Regression",
    "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "logistic": "Logistic Regression", "baseline": "Baseline (mean / most frequent)",
    # PART 03: the nine further classifiers (registry: src/predictive/classifiers.py)
    "svm": "Support Vector Machine", "knn": "K-Nearest Neighbors", "gradient_boosting": "Gradient Boosting",
    "extra_trees": "Extra Trees", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "naive_bayes": "Naive Bayes", "adaboost": "AdaBoost",
}
# PART 03: all 12 classifiers; CLASSIFICATION_MODELS above stays the default selection.
ALL_CLASSIFICATION_MODELS = ("logistic", "decision_tree", "random_forest", "svm", "knn", "gradient_boosting",
                             "extra_trees", "xgboost", "lightgbm", "catboost", "naive_bayes", "adaboost")

SILHOUETTE_CAVEAT = ("A higher silhouette score means the clusters are more compact and better separated in the "
                     "chosen numeric columns. It does not show that the segments are meaningful, stable over time, "
                     "or useful for business decisions; judge them using the profiles and domain knowledge.")
BOOTSTRAP_NOTE = ("This interval shows how much the held-out metric could vary because the test sample is finite. "
                  "It does not include uncertainty from training-set sampling, model selection, the random split, "
                  "or future changes in the data, and it is not a complete measure of model uncertainty.")
BOOTSTRAP_NOTE_CHRONO = "Rows in time order are usually correlated, so this interval is probably too narrow."
IMPORTANCE_NOTE = ("Importance describes the fitted model, not causation; correlated features can share or hide "
                   "importance. It was computed on cross-validation validation folds of the training portion only, "
                   "after the model was selected, and never influenced the selection.")
LEAKAGE_LIMIT = ("Leakage that is not visible in the data (for example a field recorded after the outcome) cannot be "
                 "detected automatically; judge it using business knowledge.")


class SettingsError(ValueError):
    """Invalid user setting; the UI shows the message."""


@dataclass
class Settings:
    test_share: float = 0.20
    cv_folds: int = 5
    seed: int = SEED
    chronological: bool = False
    time_col: Optional[str] = None       # None + chronological => current row order
    models: tuple = ()
    simple_predictor: Optional[str] = None   # None => automatic (chosen inside each fold)
    user_excluded: tuple = ()
    reinclude_identifiers: tuple = ()
    skip_sensitivity: bool = False
    balance_classes: bool = False            # PART 03: class weights / equal priors where the model supports them

    def validate(self) -> "Settings":
        if not (0.10 <= float(self.test_share) <= 0.40):
            raise SettingsError("Test share must be between 10% and 40%.")
        if not (3 <= int(self.cv_folds) <= 10):
            raise SettingsError("Number of cross-validation folds must be between 3 and 10.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SettingsError("Seed must be a non-negative whole number.")
        return self

    def fingerprint(self) -> tuple:
        return (round(float(self.test_share), 4), int(self.cv_folds), int(self.seed), bool(self.chronological),
                self.time_col, tuple(self.models), self.simple_predictor, tuple(sorted(self.user_excluded)),
                tuple(sorted(self.reinclude_identifiers)), bool(self.balance_classes))


@dataclass
class Refusal:
    reason: str
    recommendation: str = ""


@dataclass
class SupervisedResult:
    status: str = "refused"                  # ok | refused
    refusal: Optional[Refusal] = None
    task: str = ""
    target: str = ""
    settings: Any = None
    warnings: list = field(default_factory=list)
    checks: list = field(default_factory=list)        # [{check, status, detail}]
    data_prep: dict = field(default_factory=dict)
    conflicts: Any = None                              # DataFrame
    comparison: Any = None                             # DataFrame (CV only)
    selected: str = ""
    selection_reason: str = ""
    selection_metric: str = ""
    fold_audit: Any = None                             # DataFrame
    final_review: Any = None                           # DataFrame
    sensitivity: Any = None                            # DataFrame
    importance_folds: Any = None                       # DataFrame
    importance: Any = None                             # DataFrame
    internal_importance: Any = None                    # DataFrame
    heldout: dict = field(default_factory=dict)        # metric -> value
    heldout_ci: dict = field(default_factory=dict)     # metric -> (lo, hi)
    ci_note: str = ""
    class_metrics: Any = None
    confusion: Any = None
    classes: list = field(default_factory=list)
    predictions: Any = None                            # DataFrame
    baseline_beaten: Optional[bool] = None
    cv_folds_used: int = 0
    # PART 03: independent holdout evaluation of EVERY model that passed cross-validation (reporting only;
    # selection above uses cross-validation alone)
    holdout_all: Any = None                            # DataFrame, one row per model
    holdout_confusions: dict = field(default_factory=dict)   # model label -> DataFrame
    holdout_roc: dict = field(default_factory=dict)          # model label -> {curve name: (fpr, tpr)}
    holdout_class_tables: dict = field(default_factory=dict) # model label -> per-class DataFrame
    feature_importance_all: dict = field(default_factory=dict)  # model label -> DataFrame (model-internal)
    internals: dict = field(default_factory=dict)      # audit logs for tests

    def refuse(self, reason: str, recommendation: str = "") -> "SupervisedResult":
        self.status = "refused"
        self.refusal = Refusal(reason, recommendation)
        return self


@dataclass
class SegmentationResult:
    status: str = "refused"
    refusal: Optional[Refusal] = None
    warnings: list = field(default_factory=list)
    features: list = field(default_factory=list)
    excluded_features: dict = field(default_factory=dict)
    k: int = 0
    k_table: Any = None
    assignments: Any = None
    profiles: Any = None
    metrics: dict = field(default_factory=dict)
    data_prep: dict = field(default_factory=dict)
    entity_doc: Any = None
    excluded_records: Any = None
    pca: Any = None
    caveat: str = SILHOUETTE_CAVEAT

    def refuse(self, reason: str, recommendation: str = "") -> "SegmentationResult":
        self.status = "refused"
        self.refusal = Refusal(reason, recommendation)
        return self
