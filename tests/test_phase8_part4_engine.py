"""Phase 8 Part 4 - Predictive engine: leakage protocol, metrics, edge cases, determinism, export."""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictive import evaluate as ev  # noqa: E402
from src.predictive import export as pexp  # noqa: E402
from src.predictive import leakage, metrics as M  # noqa: E402
from src.predictive.contracts import ROW_CAP, Settings  # noqa: E402
from src.predictive.evaluate import run_supervised  # noqa: E402
from src.predictive.segment import run_segmentation  # noqa: E402

@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    from src.common import export as common_export
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)



def reg_df(n=400, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n), "cat": r.choice(list("xyz"), n)})
    d["y"] = 3 * d.a - d.b + (d.cat == "x") * 2 + r.normal(size=n) * .4
    return d


def cls_df(n=400, seed=0):
    d = reg_df(n, seed)
    d["lab"] = np.where(d.y > d.y.median(), "hi", "lo")
    return d.drop(columns="y")


# ---------------------------------------------------------------- metrics against hand-computed values
def test_regression_metrics_hand_computed():
    m = M.regression_metrics(pd.Series([1., 2., 3., 4.]), np.array([1., 2., 3., 6.]))
    assert m["rmse"] == pytest.approx(1.0) and m["mae"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 4 / 5.0)


def test_mape_undefined_when_any_actual_zero():
    assert M.regression_metrics(pd.Series([0., 1., 2.]), np.array([1., 1., 2.]))["mape"] is None


def test_classification_metrics_hand_computed():
    y = pd.Series(["a", "a", "b", "b"]); p = np.array(["a", "b", "b", "b"])
    m = M.classification_metrics(y, p, ["a", "b"], None)
    # a: P=1 R=.5 F=2/3 ; b: P=2/3 R=1 F=.8
    assert m["macro_f1"] == pytest.approx((2 / 3 + .8) / 2)
    assert m["macro_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert m["roc_auc"] is None
    assert m["_cm"].tolist() == [[1, 1], [0, 2]]


# ---------------------------------------------------------------- refusals / insufficient data
def test_too_few_rows_refused():
    r = run_supervised(reg_df(15), "y")
    assert r.status == "refused" and r.refusal.reason


def test_missing_target_column_refused():
    assert run_supervised(reg_df(), "nope").status == "refused"


def test_single_class_refused():
    d = cls_df(60); d["lab"] = "same"
    assert run_supervised(d, "lab").status == "refused"


def test_rare_class_refused():
    d = cls_df(100); d.loc[:1, "lab"] = "rare"
    assert run_supervised(d, "lab").status == "refused"


def test_missing_target_rows_become_unlabelled_not_trained_on():
    d = reg_df(); d.loc[:19, "y"] = np.nan
    r = run_supervised(d, "y")
    assert r.status == "ok" and r.data_prep["unlabelled_rows_missing_or_invalid_target"] == 20
    assert (r.predictions.data_split == "unlabelled").sum() == 20


def test_nan_in_features_handled():
    d = reg_df(); d.loc[::7, "a"] = np.nan; d.loc[::11, "cat"] = None
    assert run_supervised(d, "y").status == "ok"


# ---------------------------------------------------------------- duplicates
def test_exact_duplicates_removed_conflicts_kept_and_reported():
    d = reg_df(200)
    d = pd.concat([d, d.iloc[:10]], ignore_index=True)
    c = d.iloc[20:25].copy(); c["y"] = c["y"] + 5
    d = pd.concat([d, c], ignore_index=True)
    r = run_supervised(d, "y")
    assert r.status == "ok"
    assert r.data_prep["exact_duplicate_rows_removed"] == 10
    assert r.data_prep["conflicting_label_rows"] >= 10
    assert r.conflicts is not None and len(r.conflicts) == 5 and int(r.conflicts["rows"].sum()) == 10


# ---------------------------------------------------------------- leakage screening
def test_target_copy_and_recode_are_never_predictors():
    d = reg_df(); d["copy"] = d.y; d["recode"] = d.y * 3 + 1
    r = run_supervised(d, "y")
    fr = r.final_review.set_index("column")
    for c in ("copy", "recode"):
        assert fr.loc[c, "tier"] == "1a" and fr.loc[c, "decision"] == "excluded"
    assert r.final_review.shape[0] >= 2


def test_tier1a_cannot_be_reincluded():
    d = reg_df(); d["copy"] = d.y
    r = run_supervised(d, "y", settings=Settings(reinclude_identifiers=("copy",)))
    assert r.final_review.set_index("column").loc["copy", "decision"] == "excluded"


def test_identifier_excluded_by_default_and_reinclude_works():
    d = reg_df(); d["rid"] = np.arange(len(d))
    assert run_supervised(d, "y").final_review.set_index("column").loc["rid", "decision"] == "excluded"
    r = run_supervised(d, "y", settings=Settings(reinclude_identifiers=("rid",)))
    assert r.final_review.set_index("column").loc["rid", "decision"] == "included after your review"


def test_continuous_float_not_an_identifier():
    d = reg_df(); d["cont"] = np.random.default_rng(9).normal(size=len(d))
    r = run_supervised(d, "y")
    assert "cont" not in set(r.final_review.column)


def test_screen_functions_take_only_partition_data():
    X = reg_df(100).drop(columns="y"); y = reg_df(100).y
    s = leakage.screen_features(X, y, "regression", "T_1")
    assert s.partition == "T_1" and s.n_rows == 100 and set(s.row_ids) == set(X.index)


def test_each_fold_screen_uses_only_its_training_rows():
    r = run_supervised(reg_df(), "y")
    n_train = r.data_prep["train_rows"]
    for f, s in r.internals["fold_screens"].items():
        assert s.n_rows < n_train
    assert r.internals["final_screen"].n_rows == n_train
    assert r.fold_audit["rows_screened"].max() < n_train


def test_test_rows_never_reach_screening_or_fitting():
    r = run_supervised(reg_df(), "y")
    te_ids = set(r.predictions.loc[r.predictions.data_split == "test", "source_row"] - 1)
    for s in list(r.internals["fold_screens"].values()) + [r.internals["final_screen"]]:
        assert not (set(s.row_ids) & te_ids)


# ---------------------------------------------------------------- mandatory invariance (revised IC-T2)
def _perturb(d, ids, seed=5):
    d = d.copy(); rng = np.random.default_rng(seed); ids = list(ids)
    d.loc[ids, "a"] = rng.normal(size=len(ids)) * 10
    d.loc[ids, "b"] = rng.normal(size=len(ids)) * 10
    d.loc[ids, "cat"] = "z"
    return d


def _cmp(r):
    return r.comparison.drop(columns="fit_seconds")


def test_changing_test_features_changes_no_screen_selection_or_fit():
    d = reg_df(); d["rid"] = np.arange(len(d)); d["copy"] = d.y
    base = run_supervised(d, "y")
    te = (base.predictions.loc[base.predictions.data_split == "test", "source_row"] - 1).tolist()
    alt = run_supervised(_perturb(d, te), "y")
    assert base.selected == alt.selected
    pd.testing.assert_frame_equal(_cmp(base), _cmp(alt))           # CV is training-only
    pd.testing.assert_frame_equal(base.final_review, alt.final_review)
    pd.testing.assert_frame_equal(base.fold_audit, alt.fold_audit)
    pd.testing.assert_frame_equal(base.importance, alt.importance)
    a, b = base.internals["final_screen"], alt.internals["final_screen"]
    assert a.excluded == b.excluded and a.eligible == b.eligible and a.tier2 == b.tier2
    pd.testing.assert_frame_equal(base.internal_importance, alt.internal_importance) if base.internal_importance is not None else None
    # predictions on the test rows ARE recomputed and may differ; held-out scores may differ
    pb = base.predictions.query("data_split=='test'").predicted.to_numpy()
    pa = alt.predictions.query("data_split=='test'").predicted.to_numpy()
    assert pb.shape == pa.shape and not np.allclose(pb, pa)


def test_changing_validation_features_inside_cv_changes_no_fitted_state():
    d = reg_df(); X, y = d[["a", "b", "cat"]], d.y
    from src.predictive.procedure import ScreenedModel
    tr, va = np.arange(0, 300), np.arange(300, 400)
    def fit(Xv_changed):
        Xa = X.copy()
        if Xv_changed:
            Xa = _perturb(d, d.index[va])[["a", "b", "cat"]]
        m = ScreenedModel("random_forest", "regression", ["a", "b", "cat"], 42, [], (), (), None,
                          partition="T_1").fit(Xa.iloc[tr], y.iloc[tr])
        return m, m.predict(Xa.iloc[va])
    m1, p1 = fit(False); m2, p2 = fit(True)
    assert m1.fitted_signature() == m2.fitted_signature()
    assert m1.cols_ == m2.cols_ and m1.screen_.excluded == m2.screen_.excluded
    assert not np.allclose(p1, p2)


def test_changing_test_targets_does_not_change_model_selection():
    d = reg_df(); base = run_supervised(d, "y")
    te = (base.predictions.loc[base.predictions.data_split == "test", "source_row"] - 1).tolist()
    d2 = d.copy(); d2.loc[te, "y"] = d2.loc[te, "y"].to_numpy()[::-1] + 50
    alt = run_supervised(d2, "y")
    assert alt.selected == base.selected
    pd.testing.assert_frame_equal(_cmp(base), _cmp(alt))
    assert alt.heldout["rmse"] != base.heldout["rmse"]


def test_heldout_evaluated_once_no_test_rows_in_oof_or_selection(monkeypatch):
    calls = []
    orig = M.regression_metrics
    monkeypatch.setattr(ev.M, "regression_metrics", lambda yt, *a, **k: (calls.append(len(yt)), orig(yt, *a, **k))[1])
    r = run_supervised(reg_df(), "y")
    # PART 03: one held-out evaluation per model (the selected model's is reused for its row, not repeated);
    # nothing else sees test-sized input. Selection independence: tests/test_upgrade_part03.py.
    assert calls.count(r.data_prep["test_rows"]) == int((r.holdout_all.status == "ok").sum())
    assert r.holdout_all.model.is_unique


# ---------------------------------------------------------------- determinism, baseline, models
def test_deterministic_for_same_seed_and_changes_with_seed():
    d = reg_df()
    a, b = run_supervised(d, "y"), run_supervised(d, "y")
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    assert a.heldout == b.heldout
    c = run_supervised(d, "y", settings=Settings(seed=7))
    assert c.heldout != a.heldout


def test_all_models_reported_with_baseline():
    r = run_supervised(reg_df(), "y")
    assert set(r.comparison.model) >= {"Simple Linear Regression", "Multiple Linear Regression", "Decision Tree",
                                       "Random Forest"} and r.comparison.model.str.startswith("Baseline").any()
    r = run_supervised(cls_df(), "lab")
    assert {"Logistic Regression", "Decision Tree", "Random Forest"} <= set(r.comparison.model)


def test_good_model_beats_baseline_and_noise_target_is_flagged():
    r = run_supervised(reg_df(), "y")
    assert r.baseline_beaten is True
    d = reg_df(); d["y"] = np.random.default_rng(1).normal(size=len(d))
    r = run_supervised(d, "y")
    assert r.baseline_beaten is False and any("baseline" in w for w in r.warnings)


def test_multiple_linear_recovers_known_coefficients_fit():
    r = run_supervised(reg_df(1000), "y", settings=Settings(models=("multiple_linear",)))
    assert r.heldout["r2"] > 0.95


def test_classification_roc_auc_valid_and_confusion_sums():
    r = run_supervised(cls_df(), "lab")
    assert 0.5 < r.heldout["roc_auc"] <= 1.0
    assert int(r.confusion.to_numpy().sum()) == r.data_prep["test_rows"]


def test_imbalanced_classes_warn_and_use_macro_f1():
    d = cls_df(600); d["lab"] = np.where(np.random.default_rng(0).random(len(d)) < .08, "rare", "common")
    r = run_supervised(d, "lab")
    assert r.status == "ok" and any("imbalanced" in w for w in r.warnings) and "macro-F1" in r.selection_metric


# ---------------------------------------------------------------- chronological
def test_chronological_split_trains_on_earlier_rows():
    d = reg_df(300); d["when"] = pd.date_range("2020-01-01", periods=300)
    d = d.sample(frac=1, random_state=1).reset_index(drop=True)          # shuffled on disk
    r = run_supervised(d, "y", settings=Settings(chronological=True, time_col="when"))
    assert r.status == "ok"
    tr = d.loc[r.predictions.query("data_split=='train'").source_row - 1, "when"]
    te = d.loc[r.predictions.query("data_split=='test'").source_row - 1, "when"]
    assert tr.max() <= te.min()


def test_chronological_too_short_refused():
    r = run_supervised(reg_df(24), "y", settings=Settings(chronological=True, test_share=.4))
    assert r.status == "refused"


# ---------------------------------------------------------------- cap
def test_row_cap_applied_and_disclosed(monkeypatch):
    monkeypatch.setattr(ev, "ROW_CAP", 300)
    r = run_supervised(reg_df(500), "y")
    assert r.data_prep["labelled_rows_analysed"] == 300 and r.data_prep["rows_capped_out"] == 200
    assert (r.predictions.data_split == "capped_out").sum() == 200


# ---------------------------------------------------------------- export
def test_export_roundtrip_matches_result():
    r = run_supervised(reg_df(), "y")
    x = pd.ExcelFile(io.BytesIO(pexp.supervised_excel(r)))
    assert {"Summary", "Model comparison", "Predictions", "Held-out test"} <= set(x.sheet_names)
    back = pd.read_csv(io.BytesIO(pexp.predictions_csv(r)))
    assert len(back) == len(r.predictions)


def test_export_formula_injection_neutralised():
    d = reg_df(); d["cat"] = d["cat"].map({"x": "=cmd()", "y": "+1", "z": "ok"})
    r = run_supervised(d, "y", settings=Settings(skip_sensitivity=True))
    assert r.status == "ok"
    x = pd.read_excel(io.BytesIO(pexp.supervised_excel(r)), sheet_name="Summary")
    assert not x.astype(str).apply(lambda c: c.str.startswith("=")).any().any()


def test_export_refused_result_raises():
    with pytest.raises(Exception):
        pexp.supervised_excel(run_supervised(reg_df(10), "y"))


# ---------------------------------------------------------------- segmentation
def test_segmentation_finds_two_blobs_and_deterministic():
    r = np.random.default_rng(1)
    d = pd.DataFrame(np.vstack([r.normal(0, 1, (100, 2)), r.normal(8, 1, (100, 2))]), columns=["u", "v"])
    a, b = run_segmentation(d, ["u", "v"]), run_segmentation(d, ["u", "v"])
    assert a.k == 2 and a.metrics["silhouette"] > .7
    pd.testing.assert_frame_equal(a.assignments, b.assignments)
    assert a.profiles["size"].sum() == 200 and "description" in a.profiles


def test_segmentation_refusals_and_constant_columns():
    d = pd.DataFrame({"u": range(30), "c": 1.0})
    assert run_segmentation(d, ["u", "c"]).status == "refused"
    assert run_segmentation(d.iloc[:5], ["u"]).status == "refused"
    assert run_segmentation(pd.DataFrame(), ["u"]).status == "refused"
