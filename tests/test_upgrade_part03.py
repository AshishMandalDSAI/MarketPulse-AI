"""PART 03 -- Statistics completion: Z-test input fix, chart labelling on the Statistics tabs,
12 classifiers with cross-validated selection and an independent holdout evaluation of every model,
regression kept for continuous targets, optional boosting packages."""
from __future__ import annotations

import html
import json
import math
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictive import classifiers as CLF  # noqa: E402
from src.predictive.contracts import (ALL_CLASSIFICATION_MODELS, CLASSIFICATION_MODELS, MODEL_LABELS,  # noqa: E402
                                      REGRESSION_MODELS, Settings)
from src.predictive.evaluate import run_supervised  # noqa: E402

APP = str(ROOT / "app.py")
STATS = "01 · 🧮 Statistics"


# ----------------------------------------------------------------------------- data
def binary_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    seg = rng.choice(["a", "b", "c"], n)
    y = np.where(1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.4 * (seg == "b") + rng.normal(0, 1, n) > 0.3, "churn", "stay")
    return pd.DataFrame({"tenure": X[:, 0], "spend": X[:, 1], "visits": X[:, 2], "segment": seg, "Outcome": y})


def multi_df(n=600, seed=1):
    d = binary_df(n, seed)
    logit = 1.2 * d.tenure - 0.8 * d.spend
    d["Outcome"] = np.select([logit < -0.5, logit < 0.7], ["low", "mid"], "high")
    return d


def regression_df(n=300, seed=2):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(["p", "q"], n),
                         "Sales": 100 + 20 * x1 - 7 * x2 + rng.normal(0, 3, n)})


# ============================================================================ I. classifier registry
class TestRegistry:
    def test_twelve_classifiers_and_defaults(self):
        assert ALL_CLASSIFICATION_MODELS == CLF.ALL_CLASSIFIERS and len(ALL_CLASSIFICATION_MODELS) == 12
        assert {MODEL_LABELS[k] for k in ALL_CLASSIFICATION_MODELS} == {
            "Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors",
            "Gradient Boosting", "Extra Trees", "XGBoost", "LightGBM", "CatBoost", "Naive Bayes", "AdaBoost"}
        assert CLASSIFICATION_MODELS == ("logistic", "decision_tree", "random_forest") == CLF.DEFAULT_CLASSIFIERS
        assert REGRESSION_MODELS == ("simple_linear", "multiple_linear", "decision_tree", "random_forest")

    @pytest.mark.parametrize("key,module", [("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("catboost", "catboost")])
    def test_optional_packages_detected_when_installed(self, key, module):
        pytest.importorskip(module, reason=f"optional package for {key} not installed in this environment "
                                           "(install requirements-optional.txt)")
        t = CLF.availability_table().set_index("key")
        assert t.loc[key, "available"], t.loc[key, "detail"]
        assert t.loc[key, "version"]

    @pytest.mark.parametrize("key", ALL_CLASSIFICATION_MODELS)
    @pytest.mark.parametrize("data", ["binary", "multiclass"])
    def test_each_classifier_fits_predicts_and_gives_probabilities(self, key, data):
        if CLF.SPECS[key].package:
            pytest.importorskip(CLF.SPECS[key].package, reason="optional package not installed")
        d = binary_df(300) if data == "binary" else multi_df(300)
        X = d[["tenure", "spend", "visits"]].to_numpy()
        y = d.Outcome.astype(str).to_numpy()
        m = CLF.make_classifier(key, 7, balanced=True).fit(X[:240], y[:240])
        p = m.predict(X[240:])
        assert p.ndim == 1 and set(p) <= set(y)
        pr = m.predict_proba(X[240:])
        assert pr.shape == (60, len(set(y))) and np.allclose(pr.sum(1), 1, atol=1e-6)
        assert (p == y[240:]).mean() > 0.5

    def test_missing_optional_package_is_reported_never_substituted(self, monkeypatch):
        real = CLF.importlib.util.find_spec
        monkeypatch.setattr(CLF.importlib.util, "find_spec", lambda n, *a: None if n == "catboost" else real(n, *a))
        ok, why, _ = CLF.availability("catboost")
        assert not ok and "requirements-optional.txt" in why and "no other model is used" in why
        with pytest.raises(CLF.ModelUnavailable):
            CLF.make_classifier("catboost", 1)
        res = run_supervised(binary_df(), "Outcome", settings=Settings(models=("logistic", "catboost")))
        assert res.status == "ok"
        row = res.comparison.set_index("model").loc["CatBoost"]
        assert row.status == "unavailable" and "not installed" in row.detail
        h = res.holdout_all.set_index("model")
        assert h.loc["CatBoost", "status"] == "unavailable" and math.isnan(h.loc["CatBoost", "accuracy"]) \
            if "accuracy" in h else True

    def test_requirements_files(self):
        core = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        opt = (ROOT / "requirements-optional.txt").read_text(encoding="utf-8").lower()
        for pkg in ("xgboost", "lightgbm", "catboost"):
            assert pkg not in core and pkg in opt
        assert "xgboost-cpu" in opt


# ============================================================================ II. evaluation protocol
@pytest.fixture(scope="module")
def all12():
    return run_supervised(binary_df(), "Outcome", settings=Settings(models=ALL_CLASSIFICATION_MODELS))


class TestEvaluation:

    def test_every_selected_model_gets_its_own_holdout_row(self, all12):
        r = all12
        assert r.status == "ok"
        h = r.holdout_all
        assert set(h.model) == {MODEL_LABELS[k] for k in ALL_CLASSIFICATION_MODELS} | {MODEL_LABELS["baseline"]}
        assert (h.status == "ok").all(), h[h.status != "ok"]
        for col in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted",
                    "f1_weighted", "roc_auc", "log_loss", "train_seconds", "predict_seconds"):
            assert h[col].notna().all(), col
        assert h.selected_by_cv.sum() == 1
        assert h.loc[h.selected_by_cv, "model"].iloc[0] == r.selected
        assert len(r.holdout_confusions) == 13 and len(r.holdout_roc) == 13

    def test_selection_is_by_cross_validation_only(self, all12):
        c = all12.comparison
        ok = c[(c.status == "ok") & (c.model != MODEL_LABELS["baseline"])]
        assert all12.selected == ok.loc[ok.cv_mean.idxmax(), "model"]
        assert "Test rows played no part in the choice" in all12.selection_reason

    def test_holdout_labels_cannot_change_the_choice(self):
        """With a chronological split the test rows are fixed (the last 20%) whatever their labels, so flipping
        every test label must leave cross-validation and the choice untouched while the holdout scores change."""
        d = binary_df(400)
        stg = Settings(models=("logistic", "random_forest", "knn", "naive_bayes", "adaboost"), chronological=True)
        r1 = run_supervised(d, "Outcome", settings=stg)
        test_ids = r1.predictions.loc[r1.predictions.data_split == "test", "source_row"].astype(int).to_numpy() - 1
        assert list(test_ids) == list(range(320, 400))
        d2 = d.copy()
        d2.loc[test_ids, "Outcome"] = d2.loc[test_ids, "Outcome"].map({"churn": "stay", "stay": "churn"})
        r2 = run_supervised(d2, "Outcome", settings=stg)
        assert r2.selected == r1.selected
        assert np.allclose(r2.comparison.cv_mean.to_numpy(float), r1.comparison.cv_mean.to_numpy(float), equal_nan=True)
        a1 = r1.holdout_all.set_index("model").accuracy
        a2 = r2.holdout_all.set_index("model").accuracy
        assert np.allclose(a1 + a2, 1.0)            # every test prediction flipped from right to wrong or back

    def test_metrics_match_scikit_learn(self, all12):
        from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, roc_auc_score
        p = all12.predictions
        te = p[p.data_split == "test"]
        sel = all12.holdout_all.set_index("model").loc[all12.selected]
        assert sel.accuracy == pytest.approx(accuracy_score(te.actual, te.predicted))
        assert sel.f1_macro == pytest.approx(f1_score(te.actual, te.predicted, average="macro"))
        assert sel.f1_weighted == pytest.approx(f1_score(te.actual, te.predicted, average="weighted"))
        assert sel.precision_weighted == pytest.approx(precision_score(te.actual, te.predicted, average="weighted",
                                                                       zero_division=0))
        pcol = [c for c in te.columns if c.startswith("probability_")]
        if pcol:
            P = te[sorted(pcol)].to_numpy(float)
            assert sel.roc_auc == pytest.approx(roc_auc_score((te.actual == all12.classes[1]).astype(int), P[:, 1]))
            assert sel.log_loss == pytest.approx(log_loss(te.actual, P, labels=all12.classes), rel=1e-6)

    def test_multiclass_averaging_and_roc(self):
        r = run_supervised(multi_df(), "Outcome", settings=Settings(models=("logistic", "random_forest", "naive_bayes")))
        h = r.holdout_all.set_index("model")
        assert r.status == "ok" and len(r.classes) == 3
        assert h.loc["Logistic Regression", "roc_auc"] > 0.5
        curves = r.holdout_roc["Logistic Regression"]
        assert set(curves) == {f"{c} vs rest" for c in r.classes}

    def test_roc_and_log_loss_only_when_valid(self):
        classes = ["a", "b", "c"]
        P = np.full((4, 3), 1 / 3)
        assert CLF.roc_points(["a", "a", "a", "a"], P, classes) is None          # one class present
        assert CLF.roc_points(["a", "b", "a", "b"], P, classes) is None          # multiclass, class 'c' absent
        assert CLF.roc_points(["a", "b"], None, classes) is None                 # no probabilities
        assert CLF.safe_log_loss(["a", "z"], P[:2], classes) is None             # unseen label
        assert CLF.safe_log_loss(["a", "b"], P[:2, :2], classes) is None         # wrong width
        assert CLF.safe_log_loss(["a", "b", "c"], np.eye(3), classes) < 1e-6

    def test_weighted_metrics_from_confusion_matrix(self):
        from sklearn.metrics import f1_score, precision_score, recall_score
        yt = ["a"] * 6 + ["b"] * 3 + ["c"] * 1
        yp = ["a", "a", "b", "a", "a", "c", "b", "b", "a", "c"]
        idx = {"a": 0, "b": 1, "c": 2}
        cm = np.zeros((3, 3), int)
        for t, p in zip(yt, yp):
            cm[idx[t], idx[p]] += 1
        w = CLF.weighted_from_cm(cm)
        assert w["weighted_f1"] == pytest.approx(f1_score(yt, yp, average="weighted"))
        assert w["weighted_precision"] == pytest.approx(precision_score(yt, yp, average="weighted"))
        assert w["weighted_recall"] == pytest.approx(recall_score(yt, yp, average="weighted"))

    def test_class_imbalance_option(self):
        d = binary_df(700)
        d.loc[d.index[:560], "Outcome"] = "stay"          # 80 % / 20 %
        r0 = run_supervised(d, "Outcome", settings=Settings(models=("logistic",)))
        r1 = run_supervised(d, "Outcome", settings=Settings(models=("logistic",), balance_classes=True))
        rec0 = r0.holdout_all.set_index("model").loc["Logistic Regression", "recall_macro"]
        rec1 = r1.holdout_all.set_index("model").loc["Logistic Regression", "recall_macro"]
        assert rec1 >= rec0 - 1e-9
        assert Settings(balance_classes=True).fingerprint() != Settings().fingerprint()
        from src.predictive.procedure import make_estimator
        assert make_estimator("logistic", "classification", 1).class_weight is None        # unchanged default
        assert make_estimator("logistic", "classification", 1, balanced=True).class_weight == "balanced"

    def test_regression_targets_keep_regression_models_and_get_holdout_rows(self):
        r = run_supervised(regression_df(), "Sales")
        assert r.task == "regression" and r.status == "ok"
        h = r.holdout_all.set_index("model")
        for m in ("Simple Linear Regression", "Multiple Linear Regression", "Decision Tree", "Random Forest"):
            for col in ("r2", "mae", "mse", "rmse", "mape", "train_seconds", "predict_seconds"):
                assert pd.notna(h.loc[m, col]), (m, col)
            assert h.loc[m, "mse"] == pytest.approx(h.loc[m, "rmse"] ** 2)
        assert h.loc["Multiple Linear Regression", "r2"] > 0.9

    def test_classifiers_are_never_forced_on_continuous_targets(self):
        r = run_supervised(regression_df(), "Sales", settings=Settings(models=("svm", "xgboost")))
        assert r.status == "refused" and "No model was selected" in r.refusal.reason

    def test_mape_undefined_with_zero_actuals(self):
        d = regression_df()
        d.loc[d.index[::7], "Sales"] = 0.0
        r = run_supervised(d, "Sales", settings=Settings(models=("multiple_linear",)))
        assert pd.isna(r.holdout_all.set_index("model").loc["Multiple Linear Regression", "mape"])

    def test_original_three_models_unchanged(self):
        """The three original classifiers are built exactly as before PART 03 when balancing is off."""
        from src.predictive.procedure import make_estimator
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        assert type(make_estimator("logistic", "classification", 3)) is LogisticRegression
        assert type(make_estimator("decision_tree", "classification", 3)) is DecisionTreeClassifier
        assert type(make_estimator("random_forest", "classification", 3)) is RandomForestClassifier
        assert make_estimator("random_forest", "classification", 3).get_params()["class_weight"] is None


# ============================================================================ III. Streamlit pages
def _app(state):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=900)
    for k, v in state.items():
        at.session_state[k] = v
    return at


def _demo():
    from config.settings import DEMO_DATASET_ID
    return {"mp_active_dataset_id": DEMO_DATASET_ID, "mp_demo_explicitly_chosen": True}


def _goto_tab(at, tab):
    from dashboard import navigation
    at.run()
    at.session_state[navigation.STATS_TAB_FORCE_KEY] = tab
    at.sidebar.radio[0].set_value(STATS).run()
    return at


def _texts(at):
    out = []
    for coll in (at.markdown, at.info, at.warning, at.error, at.success, at.caption):
        out.extend(html.unescape(str(e.value)) for e in coll)
    return out


def _layouts(at):
    return [json.loads(c.proto.spec)["layout"] for c in at.get("plotly_chart")]


def _t(layout, axis=None):
    node = layout.get(axis, {}) if axis else layout
    t = node.get("title")
    return ((t.get("text") if isinstance(t, dict) else t) or "").strip()


def _assert_labelled(at, n=1):
    lays = _layouts(at)
    assert len(lays) >= n
    for l in lays:
        assert _t(l), l.get("title")
        assert _t(l, "xaxis") and _t(l, "yaxis"), _t(l)
    return lays


class TestZTestPage:
    def _z(self):
        at = _goto_tab(_app(_demo()), "🔬 Inferential Statistics")
        [s for s in at.selectbox if s.label == "Which test do you want to run?"][0].set_value("z").run()
        return at

    def test_blank_sigma_guides_and_does_not_run(self):
        at = self._z()
        assert not at.exception
        text = " ".join(_texts(at))
        assert "Enter the known population standard deviation" in text and "one-sample t-test" in text
        assert "was not run" in text                          # the engine's explicit refusal: nothing computed
        assert not at.latex and "Z-statistic" not in text
        assert "It is **not** used as σ" in text or "is not used as σ" in text.replace("**", "")

    def test_switch_to_t_test(self):
        at = self._z()
        [b for b in at.button if b.label == "Use the one-sample t-test instead"][0].click().run()
        assert [s.value for s in at.selectbox if s.label == "Which test do you want to run?"] == ["t1"]

    @pytest.mark.parametrize("bad", ["0", "-5", "abc"])
    def test_invalid_sigma_rejected(self, bad):
        at = self._z()
        [x for x in at.text_input if x.label.startswith("Known population")][0].set_value(bad).run()
        assert at.error and not at.latex and "Z-statistic" not in " ".join(_texts(at))
        assert any("greater than zero" in e.value or "not a number" in e.value for e in at.error)

    @pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
    def test_valid_sigma_formula_result_and_conclusion(self, alt):
        at = self._z()
        [s for s in at.selectbox if s.label == "Alternative hypothesis"][0].set_value(alt).run()
        [s for s in at.selectbox if s.label == "Significance level (α)"][0].set_value(0.01).run()
        col = [s for s in at.selectbox if s.label == "Numeric column"][0].value
        [x for x in at.text_input if x.label.startswith("Hypothesised mean")][0].set_value("500000").run()
        [x for x in at.text_input if x.label.startswith("Known population")][0].set_value("150000").run()
        assert not at.exception
        x = pd.read_csv(ROOT / "data/processed/marketing_data_clean.csv")[col].dropna().to_numpy(float)
        z = (x.mean() - 500000) / (150000 / math.sqrt(len(x)))
        from scipy import stats
        p = {"two-sided": 2 * stats.norm.sf(abs(z)), "greater": stats.norm.sf(z), "less": stats.norm.cdf(z)}[alt]
        lat = " ".join(l.value for l in at.latex)
        assert r"\frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}" in lat and f"{z:.4f}" in lat
        text = " ".join(_texts(at))
        assert f"**Z-statistic:** {z:.4f}" in text and "**α:** 0.01" in text
        assert ("reject H0" in text) and ("do not reject H0" in text) == (p >= 0.01)
        assert "standard errors away from" in text

    def test_sample_sd_never_substituted(self):
        at = self._z()
        col = [s for s in at.selectbox if s.label == "Numeric column"][0].value
        x = pd.read_csv(ROOT / "data/processed/marketing_data_clean.csv")[col].dropna().to_numpy(float)
        [x_ for x_ in at.text_input if x_.label.startswith("Hypothesised mean")][0].set_value("500000").run()
        [x_ for x_ in at.text_input if x_.label.startswith("Known population")][0].set_value("123456").run()
        lat = " ".join(l.value for l in at.latex)
        assert "123,456.00" in lat
        assert f"{x.std(ddof=1):,.2f}" not in lat


class TestStatisticsCharts:
    def test_every_inferential_chart_is_labelled(self):
        from src.inferential import tests as T
        at = _goto_tab(_app(_demo()), "🔬 Inferential Statistics")
        for key in T.TEST_ORDER:
            [s for s in at.selectbox if s.label == "Which test do you want to run?"][0].set_value(key).run()
            if key == "z":
                [x for x in at.text_input if x.label.startswith("Known population")][0].set_value("100000").run()
            assert not at.exception, key
            _assert_labelled(at)

    def test_every_descriptive_chart_is_labelled(self):
        at = _goto_tab(_app(_demo()), "📊 Descriptive Statistics")
        for opt in [s for s in at.selectbox if s.label == "Chart type"][0].options:
            [s for s in at.selectbox if s.label == "Chart type"][0].set_value(opt).run()
            assert not at.exception, opt
            _assert_labelled(at)

    def test_descriptive_statistics_cover_the_required_measures(self):
        from src.descriptive.engine import analyze_dataframe
        rep = analyze_dataframe(pd.read_csv(ROOT / "data/processed/marketing_data_clean.csv"))
        s = rep.numeric[0]
        for attr in ("mean", "median", "modes", "std_sample", "variance_sample", "minimum", "maximum", "q1", "q3",
                     "percentiles", "skewness", "kurtosis", "iqr"):
            assert getattr(s, attr) is not None, attr
        assert rep.correlation is not None and rep.categorical

    def test_tab_opens_on_the_t_test_not_the_z_test(self):
        at = _goto_tab(_app(_demo()), "🔬 Inferential Statistics")
        assert [s.value for s in at.selectbox if s.label == "Which test do you want to run?"] == ["t1"]
        assert not any("was not run" in t for t in _texts(at))


@pytest.fixture(scope="module")
def churn_upload():
    from config.settings import dataset_upload_dir
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.pipeline import profile_and_map
    d = binary_df()
    d.insert(0, "Date", pd.date_range("2024-01-01", periods=len(d)).astype(str))
    did = f"upload_p3{uuid.uuid4().hex[:10]}"
    _, m = profile_and_map(d)
    run_full_analysis(did, df=d, mapping=m)
    yield {"mp_active_dataset_id": did, "mp_active_dataset_filename": "churn.csv", "mp_demo_explicitly_chosen": False}
    shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)


class TestPredictivePage:
    def _open(self, state):
        at = _goto_tab(_app(state), "🔮 Predictive Analysis")
        [s for s in at.selectbox if s.label.startswith("Target")][0].set_value("Outcome").run()
        return at

    def test_defaults_select_all_and_run_with_labelled_charts(self, churn_upload):
        at = self._open(churn_upload)
        ms = [m for m in at.multiselect if m.label.startswith("Models to compare")][0]
        assert ms.value == list(CLASSIFICATION_MODELS)
        [b for b in at.button if b.label == "Select all available models"][0].click().run()
        assert len([m for m in at.multiselect if m.label.startswith("Models to compare")][0].value) == 12
        [b for b in at.button if b.label == "Run prediction"][0].click().run()
        assert not at.exception and not at.error
        titles = [_t(l) for l in _assert_labelled(at, 10)]
        for needle in ("Model performance comparison", "Accuracy by model", "Precision (macro) by model",
                       "Recall (macro) by model", "F1 (macro) by model", "ROC curves", "Confusion matrix —",
                       "Training time by model", "Prediction time by model", "Feature importance —"):
            assert any(needle in t for t in titles), needle
        text = " ".join(_texts(at))
        assert "were **not** used to choose the model" in text.replace("\n", " ") or "not used to choose" in text
        tbl = [d.value for d in at.dataframe if "selected_by_cv" in d.value.columns][0]
        assert len(tbl) == 13 and (tbl.status == "ok").all()

    def test_unavailable_model_is_shown_not_hidden(self, churn_upload, monkeypatch):
        real = CLF.availability
        monkeypatch.setattr(CLF, "availability",
                            lambda k: (False, "the optional package 'lightgbm' is not installed (test).", None)
                            if k == "lightgbm" else real(k))
        at = self._open(churn_upload)
        [m for m in at.multiselect if m.label.startswith("Models to compare")][0].set_value(
            ["logistic", "lightgbm"]).run()
        assert any("LightGBM" in w.value and "not installed" in w.value for w in at.warning)
        [b for b in at.button if b.label == "Run prediction"][0].click().run()
        assert not at.exception
        assert any("LightGBM" in e.value and "unavailable" in e.value for e in at.error)

    def test_continuous_target_explains_regression(self):
        at = _goto_tab(_app(_demo()), "🔮 Predictive Analysis")
        [s for s in at.selectbox if s.label.startswith("Target")][0].set_value("Revenue").run()
        text = " ".join(_texts(at))
        assert "regression problem" in text and "12 classifiers are not offered" in text
        ms = [m for m in at.multiselect if m.label.startswith("Models to compare")][0]
        assert set(ms.options) == {MODEL_LABELS[k] for k in REGRESSION_MODELS} or set(ms.value) == set(REGRESSION_MODELS)
