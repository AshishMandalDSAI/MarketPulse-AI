"""Phase 8 Part 4 audit - Modules 14/15 cross-checked against independent scipy / numpy / pandas calculations,
and Module 16 protocol checks (test set scored once, importance never sees test rows, no test rows in any screen)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.descriptive.engine import analyze_dataframe, summarize_numeric_values  # noqa: E402
from src.inferential import tests as T  # noqa: E402

RNG = np.random.default_rng(1)
X = np.array([2.1, 3.4, 5.6, 4.4, 9.8, 1.2, 7.7, 6.1, 3.3, 4.9])


# ---------------------------------------------------------------- Module 14
def test_descriptive_numeric_matches_numpy_scipy():
    s = summarize_numeric_values(X, "v")
    assert s.mean == pytest.approx(X.mean()) and s.median == pytest.approx(np.median(X))
    assert s.variance_sample == pytest.approx(X.var(ddof=1)) and s.std_sample == pytest.approx(X.std(ddof=1))
    assert s.variance_population == pytest.approx(X.var()) and s.std_population == pytest.approx(X.std())
    assert s.q1 == pytest.approx(np.percentile(X, 25)) and s.q3 == pytest.approx(np.percentile(X, 75))
    assert s.iqr == pytest.approx(np.percentile(X, 75) - np.percentile(X, 25))
    assert s.skewness == pytest.approx(st.skew(X, bias=False)) and s.kurtosis == pytest.approx(st.kurtosis(X, bias=False))
    assert (s.minimum, s.maximum, s.total) == (X.min(), X.max(), pytest.approx(X.sum()))


def test_descriptive_missing_duplicates_and_categories():
    d = pd.DataFrame({"a": [1, 2, 2, None, 5, 5, 5], "b": ["x", "y", "y", "y", None, "x", "x"], "c": [1, 2, 2, 3, 5, 5, 5],
                      "id": [0, 1, 2, 3, 4, 5, 5]})
    rep = analyze_dataframe(d)
    assert rep.quality["duplicate_rows"] == int(d.duplicated().sum()) == 1
    m = {r["column"]: r for r in rep.missing.rows}
    assert m["a"]["missing_count"] == 1 and m["b"]["missing_count"] == 1 and m["c"]["missing_count"] == 0
    assert rep.missing.missing_cells == int(d.isna().sum().sum())
    assert rep.missing.complete_rows == int(d.dropna().shape[0])
    freq = {r["value"]: r["count"] for r in rep.categorical["b"].frequency} if isinstance(rep.categorical, dict) else None
    if freq is not None:
        assert freq == d["b"].value_counts().to_dict()
    r = np.array(rep.correlation.values)
    ok = d[["a", "c"]].dropna()
    assert r[0][1] == pytest.approx(np.corrcoef(ok.a, ok.c)[0, 1])


def test_descriptive_outlier_fences_match_tukey():
    from src.descriptive.engine import detect_outliers
    v = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9, 100])
    o = detect_outliers(v, "v")
    q1, q3 = np.percentile(v, [25, 75])
    assert o.lower_bound == pytest.approx(q1 - 1.5 * (q3 - q1)) and o.upper_bound == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert o.n_outliers == int(((v < o.lower_bound) | (v > o.upper_bound)).sum()) == 1


# ---------------------------------------------------------------- Module 15
A, B = RNG.normal(5, 2, 30), RNG.normal(6, 3, 25)
V, G = np.concatenate([A, B]), np.array(["A"] * 30 + ["B"] * 25)


def test_one_sample_t_p_ci_and_cohens_d():
    r = T.one_sample_t(X, 4.0)
    ref = st.ttest_1samp(X, 4.0)
    ci = ref.confidence_interval(0.95)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue) and r.df == 9
    assert (r.ci.lower, r.ci.upper) == (pytest.approx(ci.low), pytest.approx(ci.high))
    d = (X.mean() - 4.0) / X.std(ddof=1)
    assert r.effect_size.value == pytest.approx(d)
    assert r.effect_size.extra["hedges_g"] == pytest.approx(d * (1 - 3 / (4 * 9 - 1)), rel=1e-3)


@pytest.mark.parametrize("equal_var", [False, True])
def test_independent_t_matches_scipy(equal_var):
    r = T.independent_t(V, G, "A", "B", equal_var=equal_var)
    ref = st.ttest_ind(A, B, equal_var=equal_var)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    ci = ref.confidence_interval(0.95)
    assert (r.ci.lower, r.ci.upper) == (pytest.approx(ci.low), pytest.approx(ci.high))
    sp = np.sqrt(((len(A) - 1) * A.var(ddof=1) + (len(B) - 1) * B.var(ddof=1)) / (len(A) + len(B) - 2))
    if equal_var:
        assert abs(r.effect_size.value) == pytest.approx(abs(A.mean() - B.mean()) / sp, rel=1e-6)


def test_one_sided_p_value_is_half_or_complement():
    two = T.one_sample_t(X, 4.0).p_value
    gt = T.one_sample_t(X, 4.0, alternative="greater").p_value
    lt = T.one_sample_t(X, 4.0, alternative="less").p_value
    assert gt == pytest.approx(two / 2) and lt == pytest.approx(1 - two / 2)


def test_paired_t_matches_scipy():
    p = RNG.normal(0, 1, 20); q = p + RNG.normal(0.4, 1, 20)
    r = T.paired_t(p, q); ref = st.ttest_rel(p, q); ci = ref.confidence_interval(0.95)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    assert (r.ci.lower, r.ci.upper) == (pytest.approx(ci.low), pytest.approx(ci.high))


def test_anova_matches_scipy_and_eta_squared_definition():
    c = RNG.normal(8, 2, 20)
    vv = np.concatenate([A[:20], B[:20], c]); gg = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
    r = T.one_way_anova(vv, gg); ref = st.f_oneway(A[:20], B[:20], c)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    grand = vv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in (A[:20], B[:20], c))
    sst = ((vv - grand) ** 2).sum()
    assert r.effect_size.value == pytest.approx(ssb / sst)


def test_correlation_tests_and_fisher_ci():
    x = RNG.normal(size=40); y = 0.5 * x + RNG.normal(size=40)
    r = T.pearson_test(x, y); ref = st.pearsonr(x, y)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    z, se = np.arctanh(ref.statistic), 1 / np.sqrt(len(x) - 3); q = st.norm.ppf(.975)
    assert (r.ci.lower, r.ci.upper) == (pytest.approx(np.tanh(z - q * se)), pytest.approx(np.tanh(z + q * se)))
    s = T.spearman_test(x, y); rs = st.spearmanr(x, y)
    assert s.statistic == pytest.approx(rs.statistic) and s.p_value == pytest.approx(rs.pvalue)


def test_chi_square_independence_gof_and_cramers_v():
    a = RNG.choice(["x", "y", "z"], 200); b = np.where(RNG.random(200) < .3, a, RNG.choice(["p", "q"], 200))
    r = T.chi_square_independence(a, b); tab = pd.crosstab(a, b); ref = st.chi2_contingency(tab, correction=False)
    assert r.statistic == pytest.approx(ref[0]) and r.p_value == pytest.approx(ref[1]) and r.df == ref[2]
    assert r.effect_size.value == pytest.approx(np.sqrt(ref[0] / (200 * (min(tab.shape) - 1))))
    g = T.chi_square_gof(a); rg = st.chisquare(pd.Series(a).value_counts().sort_index().values)
    assert g.statistic == pytest.approx(rg.statistic) and g.p_value == pytest.approx(rg.pvalue)


@pytest.mark.parametrize("bad", [np.array([1.0]), np.array([]), np.array([5.0, 5.0, 5.0]), np.array([np.nan, np.nan, np.nan])])
def test_invalid_or_insufficient_input_is_refused_not_computed(bad):
    r = T.one_sample_t(bad, 1.0)
    assert r.status != "ok" and r.p_value is None


def test_interpretation_does_not_overclaim():
    r = T.one_sample_t(X, 4.0)                   # p = 0.33: must not be described as significant
    assert r.significant is False
    text = " ".join(str(v) for v in r.interpretation.values()).lower()
    assert "proves" not in text and "is significant" not in text.replace("not statistically significant", "")
    assert "not statistically significant" in text
    assert r.interpretation["p_value_meaning"].startswith("The p-value is 0.3263")


# ---------------------------------------------------------------- Module 16 protocol (audit)
def _reg(n=400, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n), "cat": r.choice(list("xyz"), n)})
    d["y"] = 3 * d.a - d.b + (d.cat == "x") * 2 + r.normal(size=n) * .4
    return d


def test_importance_is_never_computed_on_test_rows(monkeypatch):
    from src.predictive import evaluate as ev
    from src.predictive import importance as imp
    seen = []
    orig = imp.fold_importance
    monkeypatch.setattr(ev.imp, "fold_importance", lambda m, Xv, yv, *a, **k: (seen.append(set(Xv.index)), orig(m, Xv, yv, *a, **k))[1])
    r = ev.run_supervised(_reg(), "y")
    te = set(r.predictions.loc[r.predictions.data_split == "test", "source_row"] - 1)
    assert seen and all(not (s & te) for s in seen)
    assert len(seen) == r.cv_folds_used


def test_heldout_scored_once_for_classification(monkeypatch):
    from src.predictive import evaluate as ev
    d = _reg(); d["lab"] = np.where(d.y > d.y.median(), "hi", "lo"); d = d.drop(columns="y")
    sizes = []
    orig = ev.M.classification_metrics
    monkeypatch.setattr(ev.M, "classification_metrics", lambda yt, *a, **k: (sizes.append(len(yt)), orig(yt, *a, **k))[1])
    r = ev.run_supervised(d, "lab")
    # PART 03 requirement: every model gets its own independent holdout evaluation (reporting only; selection is by
    # cross-validation). Still strict: each model is scored on the test rows exactly ONCE, and nothing else is.
    assert sizes.count(r.data_prep["test_rows"]) == int((r.holdout_all.status == "ok").sum())
    assert r.holdout_all.model.is_unique


def test_test_rows_absent_from_every_fold_and_final_screen_and_cv_validation_disjoint_from_fit():
    from src.predictive import evaluate as ev
    r = ev.run_supervised(_reg(), "y")
    te = set(r.predictions.loc[r.predictions.data_split == "test", "source_row"] - 1)
    for s in list(r.internals["fold_screens"].values()) + [r.internals["final_screen"]]:
        assert not (set(s.row_ids) & te)
    oof = r.predictions[r.predictions.prediction_basis == "out_of_fold"]
    assert not (set(oof.source_row - 1) & te) and len(oof) == r.data_prep["train_rows"]


# ---------------------------------------------------------------- Module 17 (audit): outputs come from the dataset and the fitted model
def test_prescriptive_current_plan_and_marginals_come_from_data_and_model():
    from src.prescriptive import engine as E
    r0 = np.random.default_rng(0); n = 104
    d = pd.DataFrame({"week": pd.date_range("2022-01-03", periods=n, freq="W-MON"), "tv": r0.uniform(1e4, 5e4, n),
                      "search": r0.uniform(5e3, 2e4, n), "social": r0.uniform(3e3, 1.5e4, n)})
    d["rev"] = 2e5 + 2 * d.tv ** .9 + 4 * d.search ** .85 + 3 * d.social + r0.normal(0, 5e3, n)
    ch = ["tv", "search", "social"]
    m = E.fit_response_model(d, ch, "rev", "week")
    tot = float(d[ch].tail(8).mean().sum())
    res = E.optimize_allocation(d, m, tot, {c: 0.05 * tot for c in ch}, {c: 0.6 * tot for c in ch})
    for c in ch:                                                     # current plan = mean of the last 8 rows
        assert res.current[c] == pytest.approx(float(d[c].tail(8).mean()))
    ctx = E._Ctx(d, m)
    step = res.total_budget * E.MARGINAL_STEP
    base = ctx.predict(np.array([res.recommended[c] for c in ch]))
    for i, c in enumerate(ch):                                       # table marginals re-derived independently
        x = np.array([res.recommended[k] for k in ch]); x[i] += step
        got = res.table.set_index("channel").loc[c, "marginal_return_recommended"]
        assert got == pytest.approx((ctx.predict(x) - base) / step)
    # a different dataset gives different numbers (nothing is cached or invented)
    d2 = d.copy(); d2["tv"] = d2["tv"] * 0.5; d2["rev"] = 2e5 + 2 * d2.tv ** .9 + 4 * d2.search ** .85 + 3 * d2.social
    m2 = E.fit_response_model(d2, ch, "rev", "week")
    tot2 = float(d2[ch].tail(8).mean().sum())
    res2 = E.optimize_allocation(d2, m2, tot2, {c: 0.05 * tot2 for c in ch}, {c: 0.6 * tot2 for c in ch})
    assert res2.current != res.current and res2.recommended != res.recommended
    assert res.model["n_train"] == m.n_train and res.model["selected_model"] == m.selected_model
