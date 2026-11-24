"""The 12 classifiers of Predictive Analysis (PART 03).

One registry: key, label, factory, whether it needs an optional package, and
how class imbalance is handled. ``procedure.make_estimator`` calls
``make_classifier`` for every classification key it does not build itself, so
all 12 models go through the same leakage screening, preprocessing
(imputation, scaling, one-hot encoding fitted on training rows only),
cross-validation and holdout evaluation as the original three.

Optional packages (``requirements-optional.txt``): xgboost-cpu, lightgbm,
catboost. They are detected at run time. A missing package makes that model
*unavailable, with the reason*; no other algorithm is ever substituted.

Also: extra holdout metrics (weighted averages, log loss) and ROC curve points.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    label: str
    package: Optional[str]          # importable module name when optional, else None
    pip_name: Optional[str]
    imbalance: str                  # how class imbalance is handled
    note: str = ""


SPECS: dict[str, ClassifierSpec] = {s.key: s for s in [
    ClassifierSpec("logistic", "Logistic Regression", None, None, "class_weight='balanced'"),
    ClassifierSpec("decision_tree", "Decision Tree", None, None, "class_weight='balanced'"),
    ClassifierSpec("random_forest", "Random Forest", None, None, "class_weight='balanced'"),
    ClassifierSpec("svm", "Support Vector Machine", None, None, "class_weight='balanced'",
                   "RBF kernel; probabilities by Platt (sigmoid) calibration on 3 internal training folds; slow on large data"),
    ClassifierSpec("knn", "K-Nearest Neighbors", None, None, "none available (distance-weighted votes)",
                   "k = 15, distance-weighted; uses the scaled features"),
    ClassifierSpec("gradient_boosting", "Gradient Boosting", None, None, "none available in scikit-learn"),
    ClassifierSpec("extra_trees", "Extra Trees", None, None, "class_weight='balanced'"),
    ClassifierSpec("xgboost", "XGBoost", "xgboost", "xgboost-cpu", "none (multiclass has no class weights)",
                   "CPU build (xgboost-cpu); labels are encoded internally"),
    ClassifierSpec("lightgbm", "LightGBM", "lightgbm", "lightgbm", "class_weight='balanced'"),
    ClassifierSpec("catboost", "CatBoost", "catboost", "catboost", "auto_class_weights='Balanced'"),
    ClassifierSpec("naive_bayes", "Naive Bayes", None, None, "equal class priors",
                   "Gaussian Naive Bayes; assumes independent, normally distributed features"),
    ClassifierSpec("adaboost", "AdaBoost", None, None, "none available in scikit-learn"),
]}
ALL_CLASSIFIERS = tuple(SPECS)
DEFAULT_CLASSIFIERS = ("logistic", "decision_tree", "random_forest")
LABELS = {k: s.label for k, s in SPECS.items()}


def availability(key: str) -> tuple[bool, str, Optional[str]]:
    """(available, reason, package version)."""
    s = SPECS[key]
    if s.package is None:
        return True, "built into scikit-learn", None
    if importlib.util.find_spec(s.package) is None:
        return False, (f"the optional package '{s.pip_name}' is not installed. Install it with "
                       f"`pip install -r requirements-optional.txt`; no other model is used in its place."), None
    try:
        mod = importlib.import_module(s.package)
    except Exception as exc:                     # installed but broken (e.g. missing system library)
        return False, f"'{s.pip_name}' is installed but cannot be loaded ({type(exc).__name__}: {exc}).", None
    return True, "installed", getattr(mod, "__version__", "?")


def availability_table() -> pd.DataFrame:
    rows = []
    for k, s in SPECS.items():
        ok, why, ver = availability(k)
        rows.append({"key": k, "model": s.label, "available": ok, "detail": why, "version": ver or "",
                     "class_imbalance_handling": s.imbalance, "note": s.note})
    return pd.DataFrame(rows)


class _EncodedXGB(ClassifierMixin, BaseEstimator):
    """XGBoost needs labels 0..k-1; this keeps the platform's original labels outside."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y):
        import xgboost as xgb
        self.le_ = LabelEncoder().fit(np.asarray(y).astype(str))
        self.classes_ = self.le_.classes_
        yy = self.le_.transform(np.asarray(y).astype(str))
        obj = "binary:logistic" if len(self.classes_) == 2 else "multi:softprob"
        self.model_ = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9,
                                        colsample_bytree=0.9, objective=obj, eval_metric="logloss",
                                        random_state=self.seed, n_jobs=1, tree_method="hist", verbosity=0)
        self.model_.fit(X, yy)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(np.asarray(self.model_.predict(X)).astype(int))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class ModelUnavailable(RuntimeError):
    """An optional package needed by the model is missing or broken (the reason is shown to the user)."""


def make_classifier(key: str, seed: int, balanced: bool = True):
    """Fresh estimator for ``key``. Raises ``RuntimeError`` (with the reason) when an optional package
    is unavailable: the caller reports the model as failed/unavailable, never swaps it."""
    if key not in SPECS:
        raise KeyError(f"unknown classifier '{key}'")
    ok, why, _ = availability(key)
    if not ok:
        raise ModelUnavailable(f"{SPECS[key].label} is unavailable: {why}")
    cw = "balanced" if balanced else None
    if key == "logistic":
        return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed, class_weight=cw)
    if key == "decision_tree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed, class_weight=cw)
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=seed, n_jobs=1,
                                      class_weight=cw)
    if key == "svm":
        # Probabilities via Platt (sigmoid) calibration, the replacement scikit-learn recommends for the deprecated
        # SVC(probability=True); 3 internal folds on the training rows only.
        return CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed, class_weight=cw),
                                      method="sigmoid", cv=3, ensemble=False)
    if key == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    if key == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=seed)
    if key == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1, class_weight=cw)
    if key == "xgboost":
        return _EncodedXGB(seed=seed)
    if key == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, n_jobs=1,
                                  class_weight=cw, verbose=-1)
    if key == "catboost":
        return _FlatCatBoost(seed=seed, balanced=balanced)
    if key == "naive_bayes":
        return _UniformPriorNB() if balanced else GaussianNB()
    if key == "adaboost":
        return AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=seed)
    raise KeyError(key)


class _FlatCatBoost(ClassifierMixin, BaseEstimator):
    """CatBoost returns predictions as a 2-D column for multiclass; this returns flat labels like scikit-learn."""

    def __init__(self, seed: int = 42, balanced: bool = True):
        self.seed, self.balanced = seed, balanced

    def fit(self, X, y):
        import catboost as cb
        self.model_ = cb.CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_seed=self.seed,
                                            verbose=False, thread_count=1, allow_writing_files=False,
                                            auto_class_weights="Balanced" if self.balanced else None)
        self.model_.fit(X, np.asarray(y).astype(str))
        self.classes_ = np.asarray(self.model_.classes_)
        self.feature_importances_ = np.asarray(self.model_.get_feature_importance(), float)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).ravel()

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class _UniformPriorNB(GaussianNB):
    """Gaussian Naive Bayes with equal class priors (its form of imbalance handling). Priors are set at fit
    time because the number of classes is only known then."""

    def fit(self, X, y, sample_weight=None):
        k = len(np.unique(np.asarray(y).astype(str)))
        self.priors = np.full(k, 1.0 / k)
        return super().fit(X, y, sample_weight=sample_weight)


# --------------------------------------------------------------------------- extra holdout metrics
def weighted_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    pred, act = cm.sum(0).astype(float), cm.sum(1).astype(float)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, act, out=np.zeros_like(tp), where=act > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    w = act / act.sum() if act.sum() else np.zeros_like(act)
    return {"weighted_precision": float((prec * w).sum()), "weighted_recall": float((rec * w).sum()),
            "weighted_f1": float((f1 * w).sum())}


def safe_log_loss(y_true, proba, classes) -> Optional[float]:
    """Log loss only when probabilities exist for every training class and every test label is a training class."""
    if proba is None:
        return None
    lut = {str(c): i for i, c in enumerate(classes)}
    t = np.array([lut.get(str(v), -1) for v in y_true])
    if (t < 0).any() or np.asarray(proba).shape[1] != len(classes):
        return None
    p = np.clip(np.asarray(proba, float), 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(log_loss(t, p, labels=list(range(len(classes)))))


def roc_points(y_true, proba, classes) -> Optional[dict]:
    """ROC curve points: binary -> {'binary': (fpr, tpr)}; multiclass (all classes present) -> one-vs-rest
    curve per class. None when not valid (no probabilities or a class absent)."""
    if proba is None:
        return None
    lut = {str(c): i for i, c in enumerate(classes)}
    t = np.array([lut.get(str(v), -1) for v in y_true])
    keep = t >= 0
    t, P = t[keep], np.asarray(proba, float)[keep]
    if len(np.unique(t)) < 2 or P.shape[1] != len(classes):
        return None
    if len(classes) == 2:
        f, r, _ = roc_curve((t == 1).astype(int), P[:, 1])
        return {f"{classes[1]} vs {classes[0]}": (f.tolist(), r.tolist())}
    if len(np.unique(t)) != len(classes):
        return None
    out = {}
    for j, c in enumerate(classes):
        f, r, _ = roc_curve((t == j).astype(int), P[:, j])
        out[f"{c} vs rest"] = (f.tolist(), r.tolist())
    return out
