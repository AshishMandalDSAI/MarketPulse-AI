"""Phase 8 Part 3 - FINAL CORRECTION tests (engine level).

Issue 1: Chi-square tests must never silently present time-ordered observations as satisfying independence.
Issue 2: Spearman's degrees of freedom are those of an approximating t distribution and must be labelled so.

Every expected number here is calculated independently of ``src.inferential`` (own rank, transition-table and
permutation code), so these tests cannot pass by sharing a bug with the implementation.
"""

import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import export as common_export  # noqa: E402
from src.inferential import assumptions as A  # noqa: E402
from src.inferential import export as X  # noqa: E402
from src.inferential import interpret as I  # noqa: E402
from src.inferential import tests as T  # noqa: E402

approx = pytest.approx
IND = "Independence of observations"


@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


# ---------------------------------------------------------------------------
# Deterministic data
# ---------------------------------------------------------------------------

def _sticky(n, cats, stay, seed):
    """A first-order Markov sequence: a category tends to repeat (serial dependence)."""
    rng = np.random.default_rng(seed)
    out = [cats[0]]
    for _ in range(n - 1):
        out.append(out[-1] if rng.random() < stay else cats[int(rng.integers(len(cats)))])
    return np.array(out, dtype=object)


def _iid(n, cats, seed):
    rng = np.random.default_rng(seed)
    return np.array([cats[int(i)] for i in rng.integers(len(cats), size=n)], dtype=object)


IID_A, IID_B = _iid(120, list("abc"), 7), _iid(120, list("xy"), 8)
STICKY_A = _sticky(120, list("abc"), 0.85, 11)
STICKY_B = _sticky(120, list("xy"), 0.85, 12)
BLOCKS = np.array(list("a" * 20 + "b" * 20 + "c" * 20 + "a" * 20 + "b" * 20 + "c" * 20), dtype=object)


def gof(values, **kw):
    return T.chi_square_gof(values, **kw)


def ind(a, b, **kw):
    return T.chi_square_independence(a, b, label_a="A", label_b="B", **kw)


CASES = {  # name -> (callable taking time_ordered, iid-version kwargs)
    "gof": (lambda v, to: gof(v, time_ordered=to), IID_A, STICKY_A),
    "ind": (lambda v, to: ind(v[0], v[1], time_ordered=to), (IID_A, IID_B), (STICKY_A, STICKY_B)),
}


def _ind_check(res):
    return next(a for a in res.assumptions if a.name == IND)


# ===========================================================================
# Issue 1 - independent reference for the screening diagnostic
# ===========================================================================

def reference_lag1_chi2(seq):
    """Hand-built transition table + Pearson chi-square (no SciPy contingency function)."""
    seq = [str(s) for s in seq]
    cats = sorted(set(seq))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    t = np.zeros((k, k))
    for a, b in zip(seq[:-1], seq[1:]):
        t[idx[a], idx[b]] += 1
    exp = np.outer(t.sum(1), t.sum(0)) / t.sum()
    chi2 = float(((t - exp) ** 2 / exp).sum())
    dof = (k - 1) ** 2
    return chi2, dof, float(stats.chi2.sf(chi2, dof))


class TestSerialDependenceScreen:
    @pytest.mark.parametrize("seq", [IID_A, STICKY_A, BLOCKS, IID_B])
    def test_matches_independent_reference(self, seq):
        s = A.serial_dependence_screen(seq, 0.05)
        chi2, dof, p = reference_lag1_chi2(seq)
        assert s["assessable"] and s["dof"] == dof
        assert s["chi2"] == approx(chi2, rel=1e-9) and s["p_value"] == approx(p, rel=1e-6, abs=1e-300)

    def test_detects_dependence_and_not_iid(self):
        assert A.serial_dependence_screen(STICKY_A, 0.05)["dependent"] is True
        assert A.serial_dependence_screen(BLOCKS, 0.05)["dependent"] is True
        assert A.serial_dependence_screen(IID_A, 0.05)["dependent"] is False

    def test_too_short_or_constant_or_sparse_is_not_assessable(self):
        assert not A.serial_dependence_screen(list("ab") * 5)["assessable"]
        assert "at least" in A.serial_dependence_screen(list("ab") * 5)["reason"]
        assert not A.serial_dependence_screen(["a"] * 60)["assessable"]
        sparse = list("abcdefghij") * 2 + list("abcdefghij")            # 10 categories, 29 transitions -> sparse table
        assert not A.serial_dependence_screen(sparse)["assessable"]
        assert "sparse" in A.serial_dependence_screen(sparse)["reason"]

    def test_alpha_controls_the_decision(self):
        s = A.serial_dependence_screen(STICKY_A, 1e-80)
        assert s["assessable"] and s["dependent"] is False              # same statistic, stricter alpha


# ===========================================================================
# Issue 1 - policy for both chi-square procedures
# ===========================================================================

@pytest.mark.parametrize("name", ["gof", "ind"])
class TestChiSquareTimeOrderedPolicy:
    def test_not_flagged_independence_is_not_checked_never_met(self, name):
        run, iid, _ = CASES[name]
        r = run(iid, False)
        assert r.status == "ok" and r.inputs["time_ordered"] is False
        c = _ind_check(r)
        assert c.status == "not_checked" and c.consequence == "none"
        assert "does not establish it" in c.detail and "cannot be verified" in c.detail
        assert r.independence_status == "not_verifiable"
        assert T.CHI_TIME_ORDER_WARNING not in r.warnings and "independence" not in r.interpretation

    def test_flagged_runs_but_warns_and_is_never_valid(self, name):
        run, iid, _ = CASES[name]
        r = run(iid, True)
        assert r.status == "ok" and r.p_value is not None                  # not rejected outright
        c = _ind_check(r)
        assert c.status == "caution" and c.consequence == "usable_with_caution"
        assert r.validity == "usable_with_caution" and r.validity != "valid"
        assert T.CHI_TIME_ORDER_WARNING in r.warnings
        for phrase in ("time-ordered", "does not establish", "serially dependent", "must not be read as showing that independence holds"):
            assert phrase in c.detail, phrase
        assert r.inputs["time_ordered"] is True and r.independence_status == "time_ordered_unverified"

    def test_flagged_statistic_and_p_value_are_unchanged(self, name):
        run, iid, _ = CASES[name]
        a, b = run(iid, False), run(iid, True)
        assert (a.statistic, a.p_value, a.df) == (b.statistic, b.p_value, b.df)
        assert a.effect_size.value == b.effect_size.value

    def test_serial_dependence_downgrades_to_unreliable_but_still_runs(self, name):
        run, _, sticky = CASES[name]
        r = run(sticky, True)
        assert r.status == "ok"                                              # a warning/status, not a blanket refusal
        c = _ind_check(r)
        assert c.status == "violated" and c.consequence == "unreliable" and r.validity == "unreliable"
        assert "Serial dependence WAS detected" in c.detail
        assert r.independence_status == "serial_dependence_detected"
        assert "unreliable" in r.interpretation["independence"]

    def test_same_serial_data_not_flagged_is_not_downgraded(self, name):
        """The policy keys on the user's declaration: unflagged rows are not screened (no hidden verdict either way)."""
        run, _, sticky = CASES[name]
        r = run(sticky, False)
        assert _ind_check(r).status == "not_checked" and r.validity == "valid"      # but the check never says 'met'
        assert _ind_check(r).status != "met"

    def test_not_assessable_screen_still_warns(self, name, monkeypatch):
        if name == "gof":
            r = gof(np.array(list("ab") * 6, dtype=object), time_ordered=True)       # n = 12 < 20: genuinely too short
        else:                                                                         # a 2 x 2 table needs n >= 20 anyway,
            monkeypatch.setattr(A, "SERIAL_MIN_N", 10_000)                            # so force 'too short for the screen'
            r = ind(IID_A, IID_B, time_ordered=True)
        assert r.status == "ok" and r.validity == "usable_with_caution"
        assert "not assessable" in _ind_check(r).detail and "independence is not established" in _ind_check(r).detail

    def test_interpretation_states_the_warning_explicitly(self, name):
        run, iid, _ = CASES[name]
        r = run(iid, True)
        txt = r.interpretation["independence"]
        assert "ASSUMES independent observations" in txt and "does NOT establish independence" in txt
        assert "Serial dependence may invalidate ordinary chi-square inference" in txt
        assert txt in r.interpretation["summary"]
        assert "No assumption problems were found" not in r.interpretation["summary"]

    def test_flag_is_exported(self, name):
        run, iid, sticky = CASES[name]
        for data, expected in ((iid, "Time-ordered: NOT established"), (sticky, "Time-ordered: serial dependence detected")):
            r = run(data, True)
            tables = X.build_tables([r], "d", "id")
            row = tables["Test Summary"].iloc[0]
            assert row["Independence of observations"] == expected
            assert "does NOT establish independence" in row["Warnings"] and "Time-ordered" in row["Inputs"] or "time_ordered: True" in row["Inputs"]
            a = tables["Assumption Checks"]
            assert (a["Assumption"] == IND).any() and "does not establish" in a[a["Assumption"] == IND].iloc[0]["What was found"]
        unflagged = X.build_tables([run(iid, False)], "d", "id")["Test Summary"].iloc[0]
        assert unflagged["Independence of observations"] == "Not verifiable (assumed, not tested)"

    def test_refused_result_never_claims_independence(self, name):
        if name == "gof":
            r = gof(labelled_counts({"a": 6, "b": 2, "c": 2}), time_ordered=True)
        else:
            r = ind(np.array(["a"] * 9 + ["b"] * 1, dtype=object), np.array(["x", "y"] * 5, dtype=object), time_ordered=True)
        assert r.status == "refused" and r.validity == "not_run"
        assert r.independence_status == "time_ordered_unverified"
        assert all(a.status != "met" for a in r.assumptions if a.name == IND)


def labelled_counts(counts):
    return np.array([k for k, n in counts.items() for _ in range(n)], dtype=object)


class TestChiSquareGuards:
    def test_flag_is_per_call_not_global(self):
        a = gof(STICKY_A, time_ordered=True)
        b = gof(STICKY_A)
        assert a.validity == "unreliable" and b.validity == "valid" and b.independence_status == "not_verifiable"

    def test_missing_rows_removed_before_the_screen(self):
        v = STICKY_A.copy()
        v[::7] = None
        r = gof(v, time_ordered=True)
        assert r.status == "ok" and r.missing.removed_n == int((v == None).sum())  # noqa: E711
        assert "lag-1" in _ind_check(r).detail

    def test_independence_screen_uses_bonferroni_alpha_per_column(self):
        r = ind(IID_A, IID_B, time_ordered=True)
        assert "alpha = 0.025 per column" in _ind_check(r).detail
        assert "alpha = 0.05 per column" in _ind_check(gof(IID_A, time_ordered=True)).detail

    def test_one_dependent_column_is_enough_to_downgrade(self):
        r = ind(STICKY_A, IID_B, time_ordered=True)
        assert r.validity == "unreliable" and "'A'" in _ind_check(r).detail.split("WAS detected in")[1]

    def test_other_tests_are_untouched_by_the_new_status_field(self):
        x = np.arange(40.0)
        assert T.one_sample_t(x, 0).independence_status == "not_verifiable"
        assert T.one_sample_t(np.cumsum(np.random.default_rng(0).normal(size=40)), 0, time_ordered=True).independence_status in (
            "time_ordered_unverified", "serial_dependence_detected")

    def test_methodology_documents_the_policy(self):
        text = X.METHODOLOGY["Chi-square time-ordered data"]
        assert "ASSUME independent observations" in text and "never establish" in text and "unreliable" in text

    def test_to_dict_carries_the_new_fields(self):
        d = gof(IID_A, time_ordered=True).to_dict()
        assert d["independence_status"] == "time_ordered_unverified" and d["df_is_approximate"] is False and d["df_note"] == ""


# ===========================================================================
# Issue 2 - Spearman
# ===========================================================================

def own_ranks(v):
    """Average ranks without pandas/scipy."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def own_spearman_t_approx(x, y, alternative="two-sided"):
    rx, ry = own_ranks(x), own_ranks(y)
    n = len(rx)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    p = {"two-sided": 2 * stats.t.sf(abs(t), n - 2), "greater": stats.t.sf(t, n - 2), "less": stats.t.cdf(t, n - 2)}[alternative]
    return rho, t, n - 2, float(p)


def exact_permutation_p(x, y):
    """Exact two-sided permutation p-value for rho_s (feasible for n <= 8)."""
    rx, ry = own_ranks(x), own_ranks(y)
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    hits = tot = 0
    for perm in permutations(range(len(ry))):
        tot += 1
        if abs(np.corrcoef(rx, ry[list(perm)])[0, 1]) >= obs - 1e-12:
            hits += 1
    return hits / tot


SP_X = np.array([3.1, 4.7, 1.2, 9.9, 5.5, 2.8, 7.3, 6.1, 8.4, 0.4, 5.9, 3.3])
SP_Y = np.array([2.0, 5.1, 1.1, 8.7, 4.9, 3.9, 6.6, 7.1, 9.2, 0.9, 4.2, 2.7])
SP_TIES_X = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9], float)
SP_TIES_Y = np.array([2, 1, 3, 3, 2, 4, 4, 6, 5, 5, 8, 7, 7, 9, 10], float)


class TestSpearmanApproximateDf:
    @pytest.mark.parametrize("x,y", [(SP_X, SP_Y), (SP_TIES_X, SP_TIES_Y), (np.arange(8.0), np.array([2, 1, 4, 3, 6, 5, 8, 7.0]))])
    @pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
    def test_p_value_is_the_t_approximation_with_n_minus_2_df(self, x, y, alt):
        rho, t, df, p = own_spearman_t_approx(x, y, alt)
        r = T.spearman_test(x, y, alternative=alt)
        assert r.statistic == approx(rho, abs=1e-12) and r.df == df
        assert r.p_value == approx(p, rel=1e-9, abs=1e-14)       # pins the statistical basis: if SciPy changes it, this fails

    def test_df_is_labelled_approximate_with_explanation(self):
        r = T.spearman_test(SP_X, SP_Y)
        assert r.df == len(SP_X) - 2 and r.df_is_approximate is True
        for phrase in ("Approximate", "no exact degrees of freedom", "Student t", "t = rho*sqrt((n-2)/(1-rho^2))",
                       "not an exact property of rank correlation"):
            assert phrase in r.df_note, phrase
        assert T.SPEARMAN_DF_WARNING in r.warnings
        assert "APPROXIMATE" in T.SPEARMAN_DF_WARNING and "p-value is approximate" in T.SPEARMAN_DF_WARNING

    def test_approximation_is_real_not_cosmetic(self):
        """The label is justified: for small n the t-approximation visibly differs from the exact permutation p-value."""
        x, y = np.arange(1.0, 7.0), np.array([2, 1, 4, 3, 6, 5.0])
        r = T.spearman_test(x, y)
        exact = exact_permutation_p(x, y)
        assert exact == approx(42 / 720)                                  # 0.05833...
        assert r.p_value == approx(0.04156, abs=1e-4)
        assert abs(r.p_value - exact) > 0.015                              # decision at alpha = 0.05 even flips
        assert r.p_value < 0.05 < exact
        assert r.df_is_approximate

    def test_interpretation_carries_the_approximation(self):
        r = T.spearman_test(SP_X, SP_Y)
        assert r.interpretation["approximation"] == r.df_note
        assert r.df_note in r.interpretation["summary"]

    def test_pearson_and_other_exact_df_tests_are_not_labelled_approximate(self):
        x = np.sort(SP_X)
        for r in (T.pearson_test(SP_X, SP_Y), T.one_sample_t(SP_X, 5), T.paired_t(SP_X, SP_Y),
                  T.one_way_anova(np.tile(SP_X, 2), np.repeat(["a", "b"], len(SP_X))),
                  T.chi_square_gof(IID_A), T.chi_square_independence(IID_A, IID_B)):
            assert r.df_is_approximate is False and r.df_note == "", r.test_key
            assert "approximation" not in r.interpretation
        assert x is not None

    def test_refused_spearman_has_no_df_claim(self):
        r = T.spearman_test([1, 1, 1, 1], [1, 2, 3, 4])
        assert r.status == "refused" and r.df is None and r.df_is_approximate is False

    def test_export_labels_basis_and_note(self):
        sp = X.build_tables([T.spearman_test(SP_X, SP_Y)], "d", "id")["Test Summary"].iloc[0]
        assert sp["Degrees of freedom"] == len(SP_X) - 2 and sp["Degrees of freedom basis"] == "Approximate"
        assert "Approximate" in sp["Degrees of freedom note"] and "Student t" in sp["Degrees of freedom note"]
        assert "APPROXIMATE" in sp["Warnings"]
        pe = X.build_tables([T.pearson_test(SP_X, SP_Y)], "d", "id")["Test Summary"].iloc[0]
        assert pe["Degrees of freedom basis"] == "Exact" and pe["Degrees of freedom note"] == ""
        z = X.build_tables([T.z_test(np.arange(40.0), 0, 5)], "d", "id")["Test Summary"].iloc[0]
        assert z["Degrees of freedom basis"] == ""                       # no df at all

    def test_methodology_states_the_statistical_basis(self):
        m = X.METHODOLOGY["Spearman correlation"]
        assert "APPROXIMATION" in m and "APPROXIMATE" in m and "not an exact" in m and "n − 2" in m

    def test_csv_excel_round_trip_keeps_the_label(self):
        tables = X.build_tables([T.spearman_test(SP_X, SP_Y)], "d", "id")
        csv = X.table_to_csv(tables, "Test Summary").decode("utf-8-sig")
        assert "Degrees of freedom basis" in csv and "Approximate" in csv
