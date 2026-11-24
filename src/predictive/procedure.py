"""``ScreenedModel``: screen -> eligibility -> preprocessing fit -> model fit, ALL inside ``fit``.

``fit(X, y)`` receives only the training subset it may learn from (``T_f`` in a CV fold, ``T`` for the final
refit); ``predict(X)`` takes no target. Cross-validation therefore cannot hand validation targets to any
screen, and the CV score estimates the whole screen -> preprocess -> fit procedure.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.predictive import leakage
from src.predictive.contracts import SEED


class ModelNotApplicable(ValueError):
    """The model cannot be fitted on this partition (reason is shown to the user)."""


def _num_frame(X: pd.DataFrame, cols) -> pd.DataFrame:
    return X[list(cols)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def _cat_frame(X: pd.DataFrame, cols) -> pd.DataFrame:
    sub = X[list(cols)].astype(object)
    return sub.where(sub.isna(), sub.astype(str)).where(sub.notna(), np.nan)


def make_estimator(key: str, task: str, seed: int, balanced: bool = False):
    if key == "baseline":
        return DummyRegressor(strategy="mean") if task == "regression" else DummyClassifier(strategy="most_frequent")
    if task == "regression":
        return {"simple_linear": lambda: LinearRegression(), "multiple_linear": lambda: LinearRegression(),
                "decision_tree": lambda: DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=seed),
                "random_forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2,
                                                               random_state=seed, n_jobs=1)}[key]()
    # PART 03: the nine further classifiers, and class balancing for any classifier, come from the registry.
    if key not in ("logistic", "decision_tree", "random_forest") or balanced:
        from src.predictive.classifiers import make_classifier
        return make_classifier(key, seed, balanced)
    return {"logistic": lambda: LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed),
            "decision_tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=seed),
            "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2,
                                                            random_state=seed, n_jobs=1)}[key]()


class ScreenedModel:
    def __init__(self, model_key: str, task: str, candidates: list, seed: int = SEED, classes: Optional[list] = None,
                 user_excluded=(), reinclude=(), simple_predictor: Optional[str] = None, forced_out=(),
                 partition: str = "T_f", with_tier2: bool = True, screen=None, balanced: bool = False):
        self.model_key, self.task, self.candidates, self.seed = model_key, task, list(candidates), seed
        self.classes = list(classes) if classes is not None else None
        self.user_excluded, self.reinclude = set(user_excluded), set(reinclude)
        self.simple_predictor, self.forced_out = simple_predictor, set(forced_out)
        self.partition, self.with_tier2 = partition, with_tier2
        self._given_screen = screen
        self.balanced = bool(balanced)
        self.screen_ = None
        self.cols_: list = []
        self.num_cols_: list = []
        self.cat_cols_: list = []
        self.predictor_: Optional[str] = None
        self.pre_ = None
        self.est_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScreenedModel":
        if self.model_key == "baseline":
            self.est_ = make_estimator("baseline", self.task, self.seed).fit(np.zeros((len(y), 1)), y)
            return self
        if self.task == "classification" and pd.Series(y).nunique() < 2:
            raise ModelNotApplicable(f"the training partition ({self.partition}) contains a single class")
        gs = self._given_screen
        if gs is not None and gs.partition == self.partition and gs.row_ids == tuple(X.index.tolist()):
            self.screen_ = gs      # screen of THIS partition computed once by the caller (same rows)
        else:
            self.screen_ = leakage.screen_features(X[self.candidates], y, self.task, self.partition, self.seed,
                                                   self.with_tier2)
        cols = [c for c in self.screen_.eligible]
        cols += [c for c in self.screen_.tier1b if c in self.reinclude]
        blocked = self.user_excluded | self.forced_out
        cols = [c for c in cols if c not in blocked and c not in self.screen_.tier1a]
        types = self.screen_.types
        if self.model_key == "simple_linear":
            nums = [c for c in cols if types[c]["numeric_dtype"]]
            if self.simple_predictor is not None:
                if self.simple_predictor not in nums:
                    raise ModelNotApplicable(f"the chosen predictor '{self.simple_predictor}' is not an eligible "
                                             f"numeric column on {self.partition}")
                self.predictor_ = self.simple_predictor
            else:
                if not nums:
                    raise ModelNotApplicable(f"no eligible numeric predictor on {self.partition}")
                yv = pd.to_numeric(y, errors="coerce").reset_index(drop=True)
                best, best_r = None, -1.0
                for c in nums:
                    xv = _num_frame(X, [c])[c].reset_index(drop=True)
                    ok = xv.notna() & yv.notna()
                    r = abs(float(np.corrcoef(xv[ok], yv[ok])[0, 1])) if ok.sum() > 2 and xv[ok].std() > 0 else 0.0
                    if r > best_r:
                        best, best_r = c, r
                self.predictor_ = best
            cols = [self.predictor_]
        if self.model_key == "multiple_linear" and len(cols) < 2:
            raise ModelNotApplicable(f"Multiple Linear Regression needs at least 2 eligible predictors; "
                                     f"{len(cols)} remain on {self.partition}")
        if not cols:
            raise ModelNotApplicable(f"no eligible predictor columns remain on {self.partition}")
        self.cols_ = cols
        self.num_cols_ = [c for c in cols if types[c]["numeric_dtype"]]
        self.cat_cols_ = [c for c in cols if not types[c]["numeric_dtype"]]
        steps = []
        if self.num_cols_:
            nsteps = [("imp", SimpleImputer(strategy="median"))]
            if self.model_key != "simple_linear":
                nsteps.append(("sc", StandardScaler()))
            steps.append(("num", Pipeline(nsteps), self.num_cols_))
        if self.cat_cols_:
            steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                          self.cat_cols_))
        self.pre_ = ColumnTransformer(steps, remainder="drop")
        Z = self.pre_.fit_transform(self._prep(X))
        self.est_ = make_estimator(self.model_key, self.task, self.seed, self.balanced).fit(Z, y)
        return self

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.num_cols_:
            parts.append(_num_frame(X, self.num_cols_))
        if self.cat_cols_:
            parts.append(_cat_frame(X, self.cat_cols_))
        return pd.concat(parts, axis=1)[self.cols_] if parts else pd.DataFrame(index=X.index)

    # -------------------------------------------------------------- predict
    def _Z(self, X):
        return np.zeros((len(X), 1)) if self.model_key == "baseline" else self.pre_.transform(self._prep(X))

    def predict(self, X: pd.DataFrame):
        return self.est_.predict(self._Z(X))

    def predict_all(self, X: pd.DataFrame) -> tuple:
        """(predictions, probabilities or None) from ONE transform of ``X``."""
        Z = self._Z(X)
        pred = self.est_.predict(Z)
        if self.task != "classification" or not hasattr(self.est_, "predict_proba"):
            return pred, None
        return pred, self._expand_proba(self.est_.predict_proba(Z), len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._expand_proba(self.est_.predict_proba(self._Z(X)), len(X))

    def _expand_proba(self, p, n) -> np.ndarray:
        have = [str(c) for c in self.est_.classes_]
        out = np.zeros((n, len(self.classes)))
        for j, c in enumerate(self.classes):
            if str(c) in have:
                out[:, j] = p[:, have.index(str(c))]
        return out

    # ----------------------------------------------------------- inspection
    def fitted_signature(self) -> str:
        """Hash of every fitted parameter (preprocessing + model) for invariance tests."""
        h = hashlib.sha256()
        h.update(repr((self.cols_, self.predictor_)).encode())
        if self.pre_ is not None:
            for name, tr, _ in self.pre_.transformers_:
                for step in tr.steps:
                    o = step[1]
                    for attr in ("statistics_", "mean_", "scale_", "var_"):
                        if hasattr(o, attr):
                            v = np.asarray(getattr(o, attr))
                            h.update(v.astype(float).tobytes() if v.dtype.kind in "fiub" else repr(v.tolist()).encode())
                    if hasattr(o, "categories_"):
                        h.update(repr([list(c) for c in o.categories_]).encode())
        e = self.est_
        for attr in ("coef_", "intercept_", "feature_importances_", "constant_", "class_prior_"):
            if hasattr(e, attr):
                h.update(np.asarray(getattr(e, attr)).astype(float).tobytes())
        if hasattr(e, "tree_"):
            h.update(e.tree_.threshold.tobytes()); h.update(e.tree_.value.tobytes())
        if hasattr(e, "estimators_"):
            for t in np.ravel(np.asarray(e.estimators_, dtype=object)):   # GradientBoosting stores a 2-D array
                if hasattr(t, "tree_"):
                    h.update(t.tree_.threshold.tobytes()); h.update(t.tree_.value.tobytes())
        return h.hexdigest()

    def internal_importance(self) -> pd.DataFrame:
        """Model-internal view (coefficients or impurity), aggregated per original column. Secondary, non-causal."""
        if self.model_key == "baseline" or self.est_ is None:
            return pd.DataFrame(columns=["feature", "value", "kind"])
        names = []
        for name, tr, cols in self.pre_.transformers_:
            if name == "num":
                names += list(cols)
            elif name == "cat":
                enc = tr.named_steps["oh"]
                for c, cats in zip(cols, enc.categories_):
                    names += [c] * len(cats)
        e = self.est_
        if hasattr(e, "coef_") and getattr(e, "kernel", "linear") == "linear":
            coef = np.abs(np.atleast_2d(e.coef_)).mean(0)
            kind = "mean |standardised coefficient|" if self.model_key != "simple_linear" else "|slope| (raw units)"
        elif hasattr(e, "feature_importances_"):
            coef, kind = np.asarray(e.feature_importances_, float), "model-internal importance"
            if self.model_key in ("decision_tree", "random_forest"):
                kind = "impurity importance"
        else:   # SVM (RBF), K-Nearest Neighbors, Naive Bayes have no built-in importance
            return pd.DataFrame(columns=["feature", "value", "kind"])
        if len(coef) != len(names):
            return pd.DataFrame(columns=["feature", "value", "kind"])
        s = pd.Series(coef, index=names).groupby(level=0).sum()
        return pd.DataFrame({"feature": s.index, "value": s.values, "kind": kind}).sort_values("value", ascending=False)
