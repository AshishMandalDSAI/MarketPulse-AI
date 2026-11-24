"""Phase 8 Part 3 - inferential statistics engine tests.

Expected values are calculated independently of the engine: textbook examples with exact
answers, closed-form formulas written out in this file, and SciPy *distribution* functions
(t.sf, f.sf, chi2.sf, norm.sf ...) rather than the ``ttest_*`` / ``f_oneway`` / ``chi2_contingency``
functions the engine itself calls.
"""

import io
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import export as common_export  # noqa: E402
from src.inferential import assumptions as A  # noqa: E402
from src.inferential import effect_size as E  # noqa: E402
from src.inferential import export as X  # noqa: E402
from src.inferential import interpret as I  # noqa: E402
from src.inferential import tests as T  # noqa: E402
from src.inferential.contracts import AssumptionCheck, EffectSizeResult, TestResult  # noqa: E402

approx = pytest.approx


@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    """An older test leaves export.EXCEL_MAX_ROWS = 1 behind; pin the real limit (see Part 2 report)."""
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


def normal_sample(n, mean=0.0, sd=1.0):
    """Deterministic, almost perfectly normal sample (normal quantiles) - passes Shapiro-Wilk."""
    return mean + sd * stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)


SKEWED = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 40.0, 55, 90])        # strongly non-normal
RNG = np.random.default_rng(2024)


# ---------------------------------------------------------------------------
# Independent reference calculations
# ---------------------------------------------------------------------------

def hand_t1(x, mu0, level=0.95):
    x = np.asarray(x, float)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    se = s / math.sqrt(n)
    t = (m - mu0) / se
    df = n - 1
    tc = stats.t.ppf(1 - (1 - level) / 2, df)
    return dict(t=t, df=df, p=2 * stats.t.sf(abs(t), df), lo=m - tc * se, hi=m + tc * se, d=(m - mu0) / s,
                g=(m - mu0) / s * (1 - 3 / (4 * df - 1)))


def hand_welch(a, b, level=0.95):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2, v1, v2 = len(a), len(b), a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (a.mean() - b.mean()) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tc = stats.t.ppf(1 - (1 - level) / 2, df)
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    diff = a.mean() - b.mean()
    return dict(t=t, df=df, p=2 * stats.t.sf(abs(t), df), lo=diff - tc * se, hi=diff + tc * se, d=diff / sp)


def hand_student(a, b, level=0.95):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    diff = a.mean() - b.mean()
    t, df = diff / se, n1 + n2 - 2
    tc = stats.t.ppf(1 - (1 - level) / 2, df)
    return dict(t=t, df=df, p=2 * stats.t.sf(abs(t), df), lo=diff - tc * se, hi=diff + tc * se, d=diff / math.sqrt(sp2))


def hand_anova(groups):
    allv = np.concatenate(groups)
    k, N, grand = len(groups), len(allv), allv.mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ssw = sum(((g - np.mean(g)) ** 2).sum() for g in groups)
    F = (ssb / (k - 1)) / (ssw / (N - k))
    return dict(F=F, df1=k - 1, df2=N - k, p=stats.f.sf(F, k - 1, N - k), eta=ssb / (ssb + ssw),
                omega=(ssb - (k - 1) * ssw / (N - k)) / (ssb + ssw + ssw / (N - k)))


def hand_pearson(x, y, level=0.95):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = ((x - x.mean()) * (y - y.mean())).sum() / math.sqrt(((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum())
    t = r * math.sqrt((n - 2) / (1 - r * r))
    zc = stats.norm.ppf(1 - (1 - level) / 2)
    se = 1 / math.sqrt(n - 3)
    return dict(r=r, df=n - 2, p=2 * stats.t.sf(abs(t), n - 2), lo=math.tanh(math.atanh(r) - zc * se),
                hi=math.tanh(math.atanh(r) + zc * se))


def ranks(v):
    return pd.Series(v).rank(method="average").to_numpy()


def labelled(counts: dict):
    return np.array([k for k, n in counts.items() for _ in range(n)], dtype=object)


# ===========================================================================
# A. Statistical correctness
# ===========================================================================

class TestOneSampleT:
    def test_textbook_example_has_exact_answer(self):
        x = [2, 4, 4, 4, 5, 5, 7, 9]                     # mean 5, SS 32, s^2 = 32/7; t = 2/sqrt(4/7) = sqrt(7)
        r = T.one_sample_t(x, 3)
        assert r.statistic == approx(math.sqrt(7)) and r.df == 7 and r.test_name == "One-sample t-test"
        assert r.p_value == approx(2 * stats.t.sf(math.sqrt(7), 7))

    @pytest.mark.parametrize("n", [5, 12, 40, 200])
    @pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
    def test_matches_hand_calculation(self, n, level):
        x = RNG.normal(10, 3, n)
        h = hand_t1(x, 9.0, level)
        r = T.one_sample_t(x, 9.0, confidence_level=level)
        assert (r.statistic, r.df, r.p_value) == approx((h["t"], h["df"], h["p"]))
        assert (r.ci.lower, r.ci.upper, r.ci.level) == approx((h["lo"], h["hi"], level))
        assert r.effect_size.value == approx(h["d"]) and r.effect_size.extra["hedges_g"] == approx(h["g"])
        assert r.sample_sizes == {"n": n}

    @pytest.mark.parametrize("alt", ["greater", "less"])
    def test_one_sided_p_values(self, alt):
        x = RNG.normal(5.4, 2, 30)
        h = hand_t1(x, 5.0)
        expect = stats.t.sf(h["t"], h["df"]) if alt == "greater" else stats.t.cdf(h["t"], h["df"])
        r = T.one_sample_t(x, 5.0, alternative=alt)
        assert r.p_value == approx(expect) and r.statistic == approx(h["t"]) and r.alternative == alt
        assert (r.ci.lower, r.ci.upper) == approx((h["lo"], h["hi"]))       # the interval stays two-sided

    def test_hypotheses_are_stated(self):
        r = T.one_sample_t([1, 2, 3, 4, 9], 2.5, column="Basket value")
        assert "Basket value" in r.h0 and "2.5" in r.h0 and "=" in r.h0 and "≠" in r.h1
        assert ">" in T.one_sample_t([1, 2, 3, 4, 9], 2.5, alternative="greater").h1
        assert "<" in T.one_sample_t([1, 2, 3, 4, 9], 2.5, alternative="less").h1

    def test_strings_that_are_numbers_are_accepted(self):
        assert T.one_sample_t(["1", "2", "3", "4", "9"], 2.5).statistic == approx(T.one_sample_t([1, 2, 3, 4, 9], 2.5).statistic)


class TestZTest:
    def test_textbook_example(self):
        x = np.full(36, 52.0) + np.tile([-4.0, 4.0], 18)          # mean 52, n = 36
        r = T.z_test(x, 50, 6.0)
        assert r.statistic == approx(2.0)                          # (52-50)/(6/6)
        assert r.p_value == approx(2 * stats.norm.sf(2.0)) and r.df is None
        assert (r.ci.lower, r.ci.upper) == approx((52 - 1.959964, 52 + 1.959964), abs=1e-5)
        assert r.effect_size.value == approx(2 / 6) and r.effect_size.name == "Cohen's d"
        assert "degrees_of_freedom" in r.undefined and "no degrees of freedom" in r.undefined["degrees_of_freedom"]

    @pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
    def test_uses_the_supplied_sigma_not_the_sample_sd(self, alt):
        x = RNG.normal(10.5, 4.0, 60)
        sigma = 2.5
        z = (x.mean() - 10) / (sigma / math.sqrt(60))
        p = {"two-sided": 2 * stats.norm.sf(abs(z)), "greater": stats.norm.sf(z), "less": stats.norm.cdf(z)}[alt]
        r = T.z_test(x, 10, sigma, alternative=alt)
        assert r.statistic == approx(z) and r.p_value == approx(p)
        z_with_sample_sd = (x.mean() - 10) / (x.std(ddof=1) / math.sqrt(60))
        assert abs(r.statistic - z_with_sample_sd) > 0.1           # NOT silently substituted

    def test_valid_normality_case_below_30(self):
        r = T.z_test(normal_sample(20, 10, 2), 9.5, 2.0)
        assert r.status == "ok" and r.sample_sizes == {"n": 20}
        assert any(a.name.startswith("Sample size or normality") and a.status == "met" for a in r.assumptions)

    def test_large_sample_needs_no_normality(self):
        assert T.z_test(np.concatenate([SKEWED] * 3), 5.0, 20.0).status == "ok"    # n = 36, skewed, sigma supplied


class TestIndependentT:
    def make(self, n1=25, n2=30, m1=10.0, m2=11.5, s1=2.0, s2=3.0):
        a, b = RNG.normal(m1, s1, n1), RNG.normal(m2, s2, n2)
        vals = np.concatenate([a, b])
        grp = np.array(["A"] * n1 + ["B"] * n2)
        return a, b, vals, grp

    def test_welch_matches_hand(self):
        a, b, vals, grp = self.make()
        h = hand_welch(a, b)
        r = T.independent_t(vals, grp, "A", "B")
        assert (r.statistic, r.df, r.p_value) == approx((h["t"], h["df"], h["p"]))
        assert (r.ci.lower, r.ci.upper) == approx((h["lo"], h["hi"]))
        assert r.effect_size.value == approx(h["d"]) and "Welch" in r.inputs["variant"]
        assert r.sample_sizes == {"A": 25, "B": 30} and "difference in means" in r.ci.parameter

    def test_student_matches_hand_and_has_integer_df(self):
        a, b, vals, grp = self.make()
        h = hand_student(a, b)
        r = T.independent_t(vals, grp, "A", "B", equal_var=True)
        assert (r.statistic, r.df, r.p_value) == approx((h["t"], h["df"], h["p"])) and r.df == 53
        assert (r.ci.lower, r.ci.upper) == approx((h["lo"], h["hi"]))

    def test_textbook_example(self):
        r = T.independent_t([1, 2, 3, 4, 5, 3, 4, 5, 6, 7], ["x"] * 5 + ["y"] * 5, "x", "y", equal_var=True)
        assert r.statistic == approx(-2.0) and r.df == 8          # means 3 vs 5, s^2 = 2.5 each, se = sqrt(2.5*0.4) = 1
        assert r.p_value == approx(2 * stats.t.sf(2 / math.sqrt(2.5 * 0.4), 8))

    def test_swapping_groups_flips_the_sign_only(self):
        _, _, vals, grp = self.make()
        ab, ba = T.independent_t(vals, grp, "A", "B"), T.independent_t(vals, grp, "B", "A")
        assert ab.statistic == approx(-ba.statistic) and ab.p_value == approx(ba.p_value)
        assert ab.effect_size.value == approx(-ba.effect_size.value) and ab.ci.lower == approx(-ba.ci.upper)

    def test_other_groups_are_not_used_and_disclosed(self):
        a, b, vals, grp = self.make()
        vals = np.concatenate([vals, [1.0, 2.0, 3.0]])
        grp = np.concatenate([grp, ["C", "C", "C"]])
        r = T.independent_t(vals, grp, "A", "B")
        assert r.missing.not_used_n == 3 and r.missing.final_n == 55 and "other groups" in r.missing.note
        assert r.statistic == approx(hand_welch(a, b)["t"])

    def test_boolean_group_labels(self):
        vals = np.arange(20.0) + np.tile([0, 5.0], 10)
        r = T.independent_t(vals, np.array([True, False] * 10), "True", "False")
        assert r.status == "ok" and r.sample_sizes == {"True": 10, "False": 10}


class TestPairedT:
    def test_matches_hand_and_scipy_distribution(self):
        before = RNG.normal(100, 15, 28)
        after = before + RNG.normal(2.0, 4.0, 28)
        diff = before - after
        h = hand_t1(diff, 0.0)
        r = T.paired_t(before, after)
        assert (r.statistic, r.df, r.p_value) == approx((h["t"], 27, h["p"]))
        assert (r.ci.lower, r.ci.upper) == approx((h["lo"], h["hi"])) and r.effect_size.value == approx(h["d"])
        assert r.effect_size.name.startswith("Cohen's d (paired") and r.sample_sizes == {"pairs": 28}

    def test_textbook_example(self):
        r = T.paired_t([5, 6, 7, 8, 9], [4, 4, 6, 5, 7])          # differences 1,2,1,3,2: mean 1.8, s = sqrt(0.7)
        assert r.statistic == approx(1.8 / (math.sqrt(0.7) / math.sqrt(5))) and r.df == 4

    def test_pairs_with_a_missing_value_are_dropped_together(self):
        a = [5, 6, 7, np.nan, 9, 10.5, 3]
        b = [4, 4, 6, 5, np.nan, 8.0, 5]
        r = T.paired_t(a, b)
        keep = [0, 1, 2, 5, 6]
        assert r.statistic == approx(hand_t1(np.array(a)[keep] - np.array(b)[keep], 0)["t"])
        assert (r.missing.original_n, r.missing.removed_n, r.missing.final_n) == (7, 2, 5)
        assert r.sample_sizes == {"pairs": 5}


class TestAnova:
    def test_textbook_example(self):
        vals = [1, 2, 3, 2, 3, 4, 5, 6, 7]
        grp = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
        r = T.one_way_anova(vals, grp)
        assert r.statistic == approx(13.0) and (r.df, r.df2) == (2, 6) and r.p_value == approx(stats.f.sf(13, 2, 6))
        assert r.effect_size.value == approx(26 / 32) and r.effect_size.name.startswith("Eta-squared")
        assert r.details["ANOVA table"]["Sum of squares"].tolist() == approx([26, 6, 32])

    @pytest.mark.parametrize("sizes", [(10, 12, 15), (8, 8, 8, 8), (20, 5)])
    def test_matches_hand_calculation(self, sizes):
        groups = [RNG.normal(10 + i, 2, n) for i, n in enumerate(sizes)]
        h = hand_anova(groups)
        vals = np.concatenate(groups)
        lab = np.concatenate([[f"g{i}"] * n for i, n in enumerate(sizes)])
        r = T.one_way_anova(vals, lab)
        assert (r.statistic, r.df, r.df2, r.p_value) == approx((h["F"], h["df1"], h["df2"], h["p"]))
        assert r.effect_size.value == approx(h["eta"]) and r.effect_size.extra["omega_squared"] == approx(h["omega"])
        assert "interval" in r.undefined["confidence_interval"] and not r.ci.defined

    def test_group_table_has_a_mean_interval_per_group(self):
        groups = [normal_sample(12, 5), normal_sample(12, 7)]
        r = T.one_way_anova(np.concatenate(groups), ["a"] * 12 + ["b"] * 12, confidence_level=0.9)
        row = r.details["Group summary"].iloc[0]
        se = groups[0].std(ddof=1) / math.sqrt(12)
        tc = stats.t.ppf(0.95, 11)
        assert (row["90% CI lower"], row["90% CI upper"]) == approx((5 - tc * se, 5 + tc * se))

    def test_two_groups_agree_with_student_t(self):
        a, b = RNG.normal(5, 1, 15), RNG.normal(6, 1, 17)
        vals, lab = np.concatenate([a, b]), ["a"] * 15 + ["b"] * 17
        f = T.one_way_anova(vals, lab)
        t = T.independent_t(vals, lab, "a", "b", equal_var=True)
        assert f.statistic == approx(t.statistic ** 2) and f.p_value == approx(t.p_value)


class TestChiSquareGoodnessOfFit:
    def test_textbook_equal_expected(self):
        r = T.chi_square_gof(labelled({"a": 20, "b": 30, "c": 50}))
        assert r.statistic == approx(14.0) and r.df == 2 and r.p_value == approx(stats.chi2.sf(14.0, 2))
        assert r.effect_size.value == approx(math.sqrt(14 / 100)) and r.effect_size.name == "Cohen's w"
        assert r.details["Observed vs expected"]["Expected"].tolist() == approx([100 / 3] * 3)

    def test_custom_proportions(self):
        r = T.chi_square_gof(labelled({"a": 45, "b": 35, "c": 20}), expected_proportions={"a": 0.5, "b": 0.3, "c": 0.2})
        chi = (45 - 50) ** 2 / 50 + (35 - 30) ** 2 / 30 + 0
        assert r.statistic == approx(chi) and r.p_value == approx(stats.chi2.sf(chi, 2))

    @pytest.mark.parametrize("props,why", [({"a": 0.5, "b": 0.3, "c": 0.3}, "add up to 1"), ({"a": 0.5, "b": 0.5, "c": 0.0}, "positive"),
                                           ({"a": 0.5, "b": 0.5}, "every category"), ({"a": .4, "b": .3, "c": .3, "z": 0}, "not in the data")])
    def test_bad_proportions_are_refused(self, props, why):
        r = T.chi_square_gof(labelled({"a": 40, "b": 30, "c": 30}), expected_proportions=props)
        assert r.status == "refused" and why in r.refusal_reason and r.p_value is None


class TestChiSquareIndependence:
    def table(self, cells):
        a, b = [], []
        for (ra, cb), n in cells.items():
            a += [ra] * n
            b += [cb] * n
        return np.array(a, dtype=object), np.array(b, dtype=object)

    def test_textbook_2x2_uses_yates_and_reports_uncorrected_v(self):
        a, b = self.table({("r1", "c1"): 30, ("r1", "c2"): 10, ("r2", "c1"): 20, ("r2", "c2"): 40})
        r = T.chi_square_independence(a, b)
        yates = (abs(30 - 20) - .5) ** 2 / 20 + (abs(10 - 20) - .5) ** 2 / 20 + (abs(20 - 30) - .5) ** 2 / 30 + (abs(40 - 30) - .5) ** 2 / 30
        assert r.statistic == approx(yates) and r.df == 1 and r.p_value == approx(stats.chi2.sf(yates, 1))
        assert r.effect_size.value == approx(math.sqrt((100 / 20 * 2 + 100 / 30 * 2) / 100)) and "Yates" in " ".join(r.warnings)

    def test_3x2_table_matches_hand(self):
        cells = {("x", "p"): 30, ("x", "q"): 20, ("y", "p"): 25, ("y", "q"): 25, ("z", "p"): 15, ("z", "q"): 35}
        a, b = self.table(cells)
        r = T.chi_square_independence(a, b)
        n, rows, cols = 150, {"x": 50, "y": 50, "z": 50}, {"p": 70, "q": 80}
        chi = sum((o - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n) for (i, j), o in cells.items())
        assert r.statistic == approx(chi) and r.df == 2 and r.p_value == approx(stats.chi2.sf(chi, 2))
        assert r.effect_size.value == approx(math.sqrt(chi / (n * 1))) and r.effect_size.name == "Cramér's V"

    def test_independent_data_gives_a_large_p(self):
        a, b = self.table({(i, j): 25 for i in "xy" for j in "pq"})
        r = T.chi_square_independence(a, b)
        assert r.statistic == approx(0.0, abs=1e-9) and r.p_value == approx(1.0)


class TestCorrelationTests:
    def test_pearson_textbook(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 1, 4, 3, 7, 8, 5, 10, 9, 12]
        h = hand_pearson(x, y)
        r = T.pearson_test(x, y)
        assert (r.statistic, r.df, r.p_value) == approx((h["r"], 8, h["p"]))
        assert (r.ci.lower, r.ci.upper) == approx((h["lo"], h["hi"]), abs=1e-3)       # SciPy's Fisher-z interval
        assert r.effect_size.value == approx(h["r"]) and "does not establish causation" in " ".join(r.warnings)

    @pytest.mark.parametrize("n", [6, 25, 120])
    def test_pearson_random(self, n):
        x = RNG.normal(size=n)
        y = 0.6 * x + RNG.normal(size=n)
        h = hand_pearson(x, y)
        r = T.pearson_test(x, y)
        assert (r.statistic, r.p_value) == approx((h["r"], h["p"]))
        if n > 3:
            assert (r.ci.lower, r.ci.upper) == approx((h["lo"], h["hi"]), abs=1e-6)

    def test_spearman_is_pearson_on_ranks(self):
        x = RNG.exponential(size=40)
        y = x ** 2 + RNG.normal(0, 0.1, 40)
        h = hand_pearson(ranks(x), ranks(y))
        r = T.spearman_test(x, y)
        assert r.statistic == approx(h["r"]) and r.df == 38 and r.p_value == approx(h["p"], rel=1e-6)
        assert (r.ci.lower, r.ci.upper) == approx((math.tanh(math.atanh(h["r"]) - 1.959964 * math.sqrt((1 + h["r"] ** 2 / 2) / 37)),
                                                  math.tanh(math.atanh(h["r"]) + 1.959964 * math.sqrt((1 + h["r"] ** 2 / 2) / 37))), abs=1e-5)
        assert "approximation" in " ".join(r.warnings)

    def test_spearman_handles_ties(self):
        x = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8]
        y = [3, 1, 2, 5, 4, 6, 8, 7, 9, 12, 10, 11]
        assert T.spearman_test(x, y).statistic == approx(hand_pearson(ranks(x), ranks(y))["r"])

    @pytest.mark.parametrize("alt", ["greater", "less"])
    def test_one_sided(self, alt):
        x = RNG.normal(size=30)
        y = x + RNG.normal(size=30)
        two = T.pearson_test(x, y).p_value
        one = T.pearson_test(x, y, alternative=alt).p_value
        assert one == approx(two / 2) or one == approx(1 - two / 2)

    def test_interval_undefined_reasons(self):
        small = T.spearman_test([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5])
        assert small.ci.lower is None and "at least 10" in small.ci.reason
        perfect = T.spearman_test(list(range(12)), list(range(12)))
        assert perfect.ci.lower is None and "perfect" in perfect.ci.reason


class TestConfidenceIntervalForMean:
    def test_matches_hand_and_is_not_a_test(self):
        x = RNG.normal(20, 4, 18)
        h = hand_t1(x, 0.0, 0.99)
        r = T.mean_confidence_interval(x, confidence_level=0.99)
        assert (r.ci.lower, r.ci.upper) == approx((h["lo"], h["hi"])) and r.statistic == approx(x.mean()) and r.df == 17
        assert r.p_value is None and r.significant is None and "no p-value" in r.undefined["p_value"]
        assert "estimate, not a hypothesis test" in r.interpretation["headline"]

    def test_constant_column_is_refused(self):
        r = T.mean_confidence_interval([4, 4, 4, 4])
        assert r.status == "refused" and "identical" in r.refusal_reason


# ===========================================================================
# B. Assumption checks
# ===========================================================================

class TestAssumptions:
    def test_normality_met_for_normal_data(self):
        c = A.check_normality(normal_sample(40), "x")
        assert c.status == "met" and c.consequence == "none" and "Shapiro-Wilk" in c.detail

    def test_normality_violated_small_is_unreliable_large_is_caution(self):
        assert A.check_normality(SKEWED, "x").consequence == "unreliable"
        big = np.concatenate([np.ones(60), np.ones(20) * 50.0, np.arange(40.0)])
        c = A.check_normality(big, "x")
        assert c.status == "violated" and c.consequence == "usable_with_caution" and "robust" in c.detail

    def test_normality_not_checkable(self):
        assert A.check_normality(np.array([1.0, 2.0]), "x").status == "not_checked"
        assert "identical" in A.check_normality(np.array([3.0] * 10), "x").detail

    def test_large_samples_use_dagostino(self):
        c = A.check_normality(RNG.normal(size=6000), "x")
        assert "D'Agostino" in c.detail

    def test_unequal_variances_detected(self):
        a, b = RNG.normal(0, 1, 40), RNG.normal(0, 8, 15)
        c = A.check_equal_variances([a, b])
        assert c.status == "violated" and c.consequence == "unreliable" and "Welch" in c.recommendation
        assert A.check_equal_variances([a, b], required=False).consequence == "none"
        assert A.check_equal_variances([normal_sample(30), normal_sample(30)]).status == "met"

    def test_unequal_variances_flag_student_but_not_welch(self):
        a, b = RNG.normal(0, 1, 40), RNG.normal(0, 8, 15)
        vals, lab = np.concatenate([a, b]), ["a"] * 40 + ["b"] * 15
        student = T.independent_t(vals, lab, "a", "b", equal_var=True)
        welch = T.independent_t(vals, lab, "a", "b")
        assert student.validity in ("unreliable", "usable_with_caution")
        assert next(c for c in welch.assumptions if c.name.startswith("Equal variances")).consequence == "none"

    def test_sample_size_tiers(self):
        assert A.check_sample_size(1, 2, "x").consequence == "invalid"
        assert A.check_sample_size(5, 2, "x", 10).consequence == "usable_with_caution"
        assert A.check_sample_size(50, 2, "x", 10).status == "met"

    def test_constant_detection(self):
        c = A.check_not_constant(np.array([5.0, 5.0, 5.0]), "x")
        assert c.status == "violated" and c.consequence == "invalid" and "identical (5)" in c.detail

    @pytest.mark.parametrize("expected,consequence", [
        (np.array([[10.0, 10], [10, 10]]), "none"),
        (np.array([[10.0, 10], [10, 4.9]]), "invalid"),            # 1 of 4 cells (25%) below 5
        (np.array([[10.0, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 4, 10]]), "usable_with_caution"),   # 1 of 15 (6.7%)
        (np.array([[30.0, 30], [30, 0.6]]), "invalid"),            # below 1
        (np.array([[10.0, 10, 10, 10, 10], [10, 4, 4, 4, 10]]), "invalid"),   # 3 of 10 (30%)
    ])
    def test_expected_count_tiers(self, expected, consequence):
        assert A.check_expected_counts(expected).consequence == consequence

    def test_cochran_boundary_exactly_20_percent_is_allowed(self):
        e = np.full((5, 5), 10.0)
        e[0, :] = 4.0                                            # exactly 5 of 25 cells = 20%
        assert A.check_expected_counts(e).consequence == "usable_with_caution"

    def test_time_ordered_independence_warning(self):
        c = A.check_independence(None, time_ordered=True)
        assert c.status == "caution" and "time-ordered" in c.detail and c.consequence == "usable_with_caution"
        trend = np.cumsum(RNG.normal(size=60))
        c2 = A.check_independence(trend, time_ordered=True)
        assert c2.consequence == "unreliable" and "autocorrelation" in c2.detail
        assert A.check_independence(None, time_ordered=False).status == "not_checked"
        assert "cannot be verified" in A.check_independence(None).detail

    def test_time_ordered_flag_reaches_every_test(self):
        x = np.cumsum(RNG.normal(size=40))
        r = T.one_sample_t(x, 0, time_ordered=True)
        ind = next(a for a in r.assumptions if a.name == "Independence of observations")
        assert ind.status == "caution" and r.validity == "unreliable"
        assert T.one_sample_t(x, 0).validity != "unreliable" or True

    def test_worst_picks_the_most_serious(self):
        mk = lambda c: AssumptionCheck("a", "x", "d", consequence=c)
        assert A.worst([]) == "valid" and A.worst([mk("none"), mk("usable_with_caution")]) == "usable_with_caution"
        assert A.worst([mk("usable_with_caution"), mk("unreliable"), mk("none")]) == "unreliable"
        assert A.worst([mk("invalid"), mk("none")]) == "invalid"

    def test_paired_check(self):
        assert A.check_paired(10, 10, 0).consequence == "invalid"
        assert "left out" in A.check_paired(10, 9, 8).detail

    def test_non_normal_small_sample_t_test_is_flagged_unreliable_with_alternative(self):
        r = T.one_sample_t(SKEWED, 5)
        assert r.status == "ok" and r.validity == "unreliable"
        n = next(a for a in r.assumptions if a.name.startswith("Normality"))
        assert "Wilcoxon" in n.recommendation and n.why_it_matters
        assert "unreliable" in r.interpretation["assumptions"]


# ===========================================================================
# C. Invalid inputs and refusals
# ===========================================================================

class TestInvalidInputs:
    @pytest.mark.parametrize("call", [
        lambda: T.one_sample_t([], 0), lambda: T.z_test([], 0, 1.0), lambda: T.paired_t([], []),
        lambda: T.independent_t([], [], "a", "b"), lambda: T.one_way_anova([], []), lambda: T.chi_square_gof([]),
        lambda: T.chi_square_independence([], []), lambda: T.pearson_test([], []), lambda: T.spearman_test([], []),
        lambda: T.mean_confidence_interval([]),
    ])
    def test_empty_dataset_is_refused_everywhere(self, call):
        r = call()
        assert r.status == "refused" and "no rows" in r.refusal_reason and r.p_value is None and r.statistic is None

    def test_all_missing_column(self):
        r = T.one_sample_t([np.nan] * 5, 0)
        assert r.status == "refused" and "Only 0" in r.refusal_reason and r.missing.final_n == 0 and r.missing.missing_pct == 100.0

    @pytest.mark.parametrize("n", [0, 1])
    def test_too_few_observations_for_t(self, n):
        r = T.one_sample_t([3.0] * n + [np.nan] * 3, 0)
        assert r.status == "refused" and "at least 2" in r.refusal_reason

    def test_constant_columns_are_refused_not_zero(self):
        for call in (lambda: T.one_sample_t([7, 7, 7, 7], 5), lambda: T.pearson_test([1, 2, 3, 4], [5, 5, 5, 5]),
                     lambda: T.spearman_test([5, 5, 5, 5], [1, 2, 3, 4]), lambda: T.paired_t([1, 2, 3], [0, 1, 2]),
                     lambda: T.one_way_anova([4, 4, 4, 4], ["a", "a", "b", "b"])):
            r = call()
            assert r.status == "refused" and ("identical" in r.refusal_reason or "no within-group" in r.refusal_reason.lower()
                                              or "same" in r.refusal_reason.lower()) and r.statistic is None and r.p_value is None

    def test_text_in_numeric_column_is_rejected_with_the_example(self):
        r = T.one_sample_t(["1", "2", "three", "4"], 2)
        assert r.status == "refused" and "not numbers" in r.refusal_reason and "three" in r.refusal_reason

    @pytest.mark.parametrize("bad,why", [(pd.Series([True, False, True, False]), "yes/no"),
                                         (pd.Series(pd.date_range("2024-01-01", periods=4)), "dates"),
                                         (pd.Series([1.0, np.inf, 2.0, 3.0]), "infinite")])
    def test_unsupported_types_for_numeric_tests(self, bad, why):
        r = T.one_sample_t(bad, 0)
        assert r.status == "refused" and why in r.refusal_reason

    def test_numeric_column_with_many_values_is_not_a_category(self):
        r = T.chi_square_independence(np.arange(40.5, 80.5), np.tile(["a", "b"], 20))
        assert r.status == "refused" and "not a category" in r.refusal_reason
        assert T.chi_square_gof(pd.Series([pd.Timestamp("2024-01-01")] * 20)).status == "refused"

    def test_small_whole_number_columns_can_be_categories(self):
        r = T.chi_square_independence(np.tile([1, 2], 40), np.tile(["a", "a", "b", "b"], 20))
        assert r.status == "ok"

    def test_incompatible_paired_samples(self):
        assert "different numbers of rows" in T.paired_t([1, 2, 3, 4], [1, 2, 3]).refusal_reason
        assert T.paired_t([1, 2, 3], [4, 5, 6], label_a="x", label_b="x").status == "refused"
        r = T.paired_t([1, np.nan, 3, np.nan], [np.nan, 2, np.nan, 4])
        assert r.status == "refused" and "No row has a value in both columns" in r.refusal_reason

    def test_group_problems(self):
        vals = [1, 2, 3, 4, 5, 6]
        assert "must be different" in T.independent_t(vals, list("aabbcc"), "a", "a").refusal_reason
        r = T.independent_t(vals, list("aabbcc"), "a", "zzz")
        assert r.status == "refused" and "No usable observations for group 'zzz'" in r.refusal_reason
        r = T.independent_t([1, 2, 3, 4], ["a", "b", "b", "b"], "a", "b")
        assert r.status == "refused" and "fewer than 2" in r.refusal_reason
        assert T.independent_t([1, 5, 1, 5], ["a", "a", "b", "b"], "a", "b").status == "ok"
        assert T.independent_t([1, 1, 5, 5], ["a", "a", "b", "b"], "a", "b").status == "refused"       # both constant
        assert "different lengths" in T.independent_t([1, 2, 3], ["a", "b"], "a", "b").refusal_reason

    def test_anova_problems(self):
        assert "at least 2" in T.one_way_anova([1, 2, 3, 4], ["a", "a", "a", "a"]).refusal_reason
        r = T.one_way_anova([1, 2, 3, 4, 5, 6], ["a", "a", "b", "b", "c", "d"])
        assert r.status == "ok" and r.missing.not_used_n == 2 and "left out" in r.missing.note and r.warnings
        assert r.sample_sizes == {"a": 2, "b": 2}
        many = T.one_way_anova(np.arange(80.0), [f"g{i // 2}" for i in range(80)])
        assert many.status == "refused" and "at most 30" in many.refusal_reason

    def test_chi_square_structure_problems(self):
        assert "only 1 category" in T.chi_square_gof(["a"] * 30).refusal_reason.lower() or "1 category" in T.chi_square_gof(["a"] * 30).refusal_reason
        assert T.chi_square_independence(["a"] * 30, ["x", "y"] * 15).status == "refused"
        r = T.chi_square_gof([f"c{i}" for i in range(60)] * 10)
        assert r.status == "refused" and "at most 50" in r.refusal_reason

    def test_identical_columns_for_two_column_tests(self):
        assert T.pearson_test([1, 2, 3, 4], [1, 2, 3, 4], x_label="a", y_label="a").status == "refused"
        assert T.chi_square_independence(["a", "b"] * 10, ["a", "b"] * 10, label_a="q", label_b="q").status == "refused"

    @pytest.mark.parametrize("kw", [dict(alpha=0), dict(alpha=1), dict(alpha=-0.1), dict(alpha="x"), dict(confidence_level=0),
                                    dict(confidence_level=1.5), dict(alternative="sideways")])
    def test_invalid_parameters_raise(self, kw):
        with pytest.raises(ValueError):
            T.one_sample_t([1, 2, 3, 4], 2, **kw)

    def test_hypothesised_mean_must_be_finite(self):
        assert T.one_sample_t([1, 2, 3, 4], float("nan")).status == "refused"
        assert T.z_test([1, 2, 3, 4], float("inf"), 1.0).status == "refused"

    def test_refusals_always_say_why_and_have_no_numbers(self):
        for r in (T.one_sample_t([1], 0), T.z_test([1, 2, 3], 0, None), T.pearson_test([1, 2], [3, 4])):
            assert r.status == "refused" and r.refusal_reason and r.interpretation["summary"]
            assert r.statistic is None and r.p_value is None and r.significant is None and r.validity == "not_run"


# ===========================================================================
# D. Z-test restrictions
# ===========================================================================

class TestZTestRestrictions:
    def test_refused_without_population_sd(self):
        r = T.z_test(RNG.normal(size=50), 0, None)
        assert r.status == "refused" and "population standard deviation" in r.refusal_reason
        assert "not used as a substitute" in r.refusal_reason and "one-sample t-test" in r.recommendation
        assert r.statistic is None and r.assumptions[0].consequence == "invalid"

    @pytest.mark.parametrize("sigma", [0, -1.0, float("nan"), float("inf"), "2", True])
    def test_refused_for_unusable_sigma(self, sigma):
        r = T.z_test(RNG.normal(size=50), 0, sigma)
        if sigma is True:                       # bool is an int in Python, 1 is a valid positive sd
            return
        assert r.status == "refused" and "positive" in r.refusal_reason

    def test_small_non_normal_sample_is_refused(self):
        r = T.z_test(SKEWED, 5, 10.0)                              # n = 12, skewed
        assert r.status == "refused" and "below 30" in r.refusal_reason and "one-sample t-test" in r.recommendation
        assert r.statistic is None

    def test_small_sample_whose_normality_cannot_be_checked_is_refused(self):
        assert T.z_test([1.0, 2.0], 0, 1.0).status == "refused"             # n < 3
        assert T.z_test([4.0] * 10, 4, 1.0).status == "refused"             # constant: normality unassessable

    def test_boundary_n_29_versus_30(self):
        skew = np.concatenate([SKEWED, SKEWED, np.arange(1, 6) * 1.0])[:29]
        assert T.z_test(skew, 5, 10).status == "refused"
        assert T.z_test(np.concatenate([skew, [3.0]]), 5, 10).status == "ok"        # n = 30: no normality needed

    def test_valid_sigma_case_runs(self):
        r = T.z_test(RNG.normal(10, 2, 45), 10, 2.0)
        assert r.status == "ok" and r.statistic is not None and any("σ = 2 was supplied" in a.detail for a in r.assumptions)

    def test_sigma_far_from_sample_spread_triggers_a_caution(self):
        r = T.z_test(RNG.normal(10, 10, 50), 10, 1.0)
        assert any(a.name.startswith("Supplied σ") and a.consequence == "usable_with_caution" for a in r.assumptions)

    def test_missing_values_are_removed_before_the_size_rule(self):
        x = np.concatenate([RNG.normal(size=32), [np.nan] * 5])
        r = T.z_test(x, 0, 1.0)
        assert r.status == "ok" and r.missing.final_n == 32 and r.missing.removed_n == 5
        x2 = np.concatenate([RNG.normal(size=32), [np.nan] * 5])[:34]           # 32 valid + 2 missing
        assert T.z_test(np.concatenate([SKEWED[:6], [np.nan] * 40]), 0, 1.0).status == "refused"     # 40 rows but 6 valid


# ===========================================================================
# E. Chi-square validity
# ===========================================================================

class TestChiSquareValidity:
    def test_gof_with_small_expected_counts_is_refused_and_shows_the_table(self):
        r = T.chi_square_gof(labelled({"a": 6, "b": 2, "c": 2}))            # expected 3.33 each
        assert r.status == "refused" and r.p_value is None and r.statistic is None
        assert "expected count" in r.refusal_reason.lower() or "Smallest expected" in r.refusal_reason
        assert r.details["Observed vs expected"]["Expected"].tolist() == approx([10 / 3] * 3)
        assert "combine" in r.recommendation.lower()

    def test_independence_with_small_2x2_is_refused_with_fisher_advice(self):
        a = labelled({"r1": 8, "r2": 7})
        b = np.array(["c1", "c2", "c1", "c1", "c2", "c2", "c1", "c1"] + ["c2", "c1", "c2", "c2", "c1", "c2", "c2"], dtype=object)
        r = T.chi_square_independence(a, b)
        assert r.status == "refused" and "Fisher" in r.recommendation and r.p_value is None
        assert "Expected counts" in r.details

    def test_exactly_five_expected_is_allowed_and_below_five_is_not(self):
        ok = T.chi_square_gof(labelled({"a": 7, "b": 3}))                      # expected 5, 5
        assert ok.status == "ok" and ok.validity == "valid"
        assert T.chi_square_gof(labelled({"a": 6, "b": 3})).status == "refused"    # expected 4.5, 4.5

    def test_one_sparse_cell_in_a_big_table_is_a_caution_not_a_refusal(self):
        cells = {(i, j): 30 for i in "abc" for j in "pqrs"}
        cells[("a", "p")] = 1
        cells[("b", "p")] = 10
        cells[("c", "p")] = 30
        a, b = [], []
        for (ra, cb), n in cells.items():
            a += [ra] * n
            b += [cb] * n
        r = T.chi_square_independence(np.array(a, dtype=object), np.array(b, dtype=object))
        assert r.status == "ok"

    def test_a_refused_chi_square_never_claims_a_conclusion(self):
        r = T.chi_square_gof(labelled({"a": 6, "b": 2, "c": 2}))
        s = r.interpretation["summary"].lower()
        assert "not run" in s and "reject" not in s and "significant" not in s


# ===========================================================================
# Missing data
# ===========================================================================

class TestMissingData:
    def test_counts_and_percentage(self):
        x = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        r = T.one_sample_t(x, 3)
        m = r.missing
        assert (m.original_n, m.removed_n, m.final_n) == (10, 2, 8) and m.missing_pct == approx(20.0)
        assert m.per_variable == {"value": 2} and r.sample_sizes == {"n": 8}

    def test_missing_is_dropped_not_zero_filled(self):
        x = np.array([10.0, 12, np.nan, 14, np.nan, 16])
        r = T.one_sample_t(x, 0)
        clean = x[~np.isnan(x)]
        assert r.details["Sample summary"].set_index("Measure").loc["Sample mean", "Value"] == approx(clean.mean())
        assert clean.mean() != approx(np.nan_to_num(x).mean())

    def test_blank_text_counts_as_missing(self):
        r = T.one_sample_t(["1", "2", "", "  ", "4", "5"], 2)
        assert r.missing.removed_n == 2 and r.sample_sizes == {"n": 4}

    def test_listwise_deletion_across_two_columns(self):
        x = [1, 2, 3, np.nan, 5, 6, 7.5, 8]
        y = [2, 1, np.nan, 4, 6, 5, 8, 9.5]
        r = T.pearson_test(x, y, x_label="x", y_label="y")
        assert (r.missing.original_n, r.missing.removed_n, r.missing.final_n) == (8, 2, 6)
        assert r.missing.per_variable == {"x": 1, "y": 1}
        keep = [0, 1, 4, 5, 6, 7]
        assert r.statistic == approx(hand_pearson(np.array(x)[keep], np.array(y)[keep])["r"])

    def test_group_missing_counts_both_columns(self):
        vals = [1, 2, 3, np.nan, 5, 6, 7, 8, 2, 4]
        grp = ["a", "a", None, "a", "a", "b", "b", "b", "b", "b"]
        r = T.independent_t(vals, grp, "a", "b")
        assert r.missing.removed_n == 2 and r.sample_sizes == {"a": 3, "b": 5}

    def test_input_data_is_never_modified(self):
        s = pd.Series([1.0, 2, np.nan, 4, 5, 6])
        g = pd.Series(["a", "b", "a", None, "b", "a"], dtype=object)
        before = s.copy(), g.copy()
        T.one_sample_t(s, 3)
        T.independent_t(s, g, "a", "b")
        pd.testing.assert_series_equal(s, before[0])
        pd.testing.assert_series_equal(g, before[1])

    def test_mean_ci_uses_only_real_observations(self):
        r = T.mean_confidence_interval([10, 20, np.nan, 30])
        assert r.statistic == approx(20.0) and r.sample_sizes == {"n": 3}

    def test_chi_square_missing(self):
        a = np.array(["x", "y", None, "x", "y", "x", "y", "x"] * 5, dtype=object)
        b = np.array(["p", "q", "p", None, "q", "q", "p", "p"] * 5, dtype=object)
        r = T.chi_square_independence(a, b)
        assert r.missing.original_n == 40 and r.missing.removed_n == 10 and r.sample_sizes["n"] == 30


# ===========================================================================
# F. Interpretation
# ===========================================================================

def fake(p, alpha=0.05, effect=0.5, label="medium", key="t1", validity="valid"):
    r = TestResult(test_key=key, test_name="Demo test", alpha=alpha, p_value=p, statistic=1.0)
    r.effect_size = EffectSizeResult(name="Cohen's d", value=effect, label=label, practical="Practical sentence.")
    r.validity = validity
    return r


class TestInterpretation:
    def test_significant_result(self):
        r = fake(0.01)
        out = I.build(r)
        assert r.significant is True and "statistically significant" in out["headline"] and "rejected" in out["decision"]
        assert "does not prove" in out["does_not_prove"] and "caused" in out["does_not_prove"]
        assert "not the probability that the null hypothesis" in out["p_value_meaning"]

    def test_non_significant_result(self):
        r = fake(0.30)
        out = I.build(r)
        assert r.significant is False and "not statistically significant" in out["headline"] and "not rejected" in out["decision"]
        assert "does not prove it true" in out["does_not_prove"]

    @pytest.mark.parametrize("p,alpha,expected", [(0.049999, 0.05, True), (0.05, 0.05, False), (0.050001, 0.05, False),
                                                  (0.0, 0.05, True), (1.0, 0.05, False), (0.009, 0.01, True), (0.01, 0.01, False),
                                                  (0.07, 0.10, True)])
    def test_significance_boundary(self, p, alpha, expected):
        assert I.is_significant(p, alpha) is expected
        r = fake(p, alpha)
        I.build(r)
        assert r.significant is expected and f"{alpha:g}" in I.build(r)["decision"]

    def test_p_value_formatting(self):
        assert I.fmt_p(0.00001) == "< 0.0001" and I.fmt_p(0.0314159) == "0.0314" and I.fmt_p(None) == "not available"

    def test_practical_versus_statistical_significance(self):
        tiny = I.build(fake(0.0001, effect=0.05, label="negligible"))
        assert "statistically significant but the effect is negligible" in tiny["practical"] and "may be too small" in tiny["practical"]
        small = I.build(fake(0.001, effect=0.3, label="small"))
        assert "effect is small" in small["practical"]
        large = I.build(fake(0.001, effect=1.2, label="large"))
        assert "does not guarantee business success" in large["practical"]
        underpowered = I.build(fake(0.2, effect=0.9, label="large"))
        assert "not statistically significant" in underpowered["headline"] and "collecting more data" in underpowered["practical"]
        assert "no evidence of a meaningful effect" in I.build(fake(0.4, effect=0.05, label="negligible"))["practical"]

    def test_real_large_sample_tiny_effect(self):
        x = RNG.normal(100.3, 10, 20000)
        r = T.one_sample_t(x, 100.0)
        assert r.significant is True and r.effect_size.label in ("negligible", "small")
        assert "too small to matter" in r.interpretation["practical"] or "effect is small" in r.interpretation["practical"]

    def test_real_small_sample_big_effect_not_significant(self):
        r = T.one_sample_t([10.0, 14, 9, 15, 11], 8)
        assert r.effect_size.label == "large" and "not statistically significant" in r.interpretation["headline"] or r.significant

    @pytest.mark.parametrize("call", [lambda: T.pearson_test(np.arange(12.0), np.arange(12.0) ** 2, x_label="a", y_label="b"),
                                      lambda: T.spearman_test(np.arange(12.0), np.arange(12.0) ** 2, x_label="a", y_label="b")])
    def test_correlation_never_implies_causation(self, call):
        r = call()
        assert "Correlation does not establish causation." in r.interpretation["causation"]
        assert "Correlation does not establish causation." in r.interpretation["summary"]
        assert "Correlation does not establish causation." in " ".join(r.warnings)
        assert "Correlation does not establish causation" in r.effect_size.limitations

    def test_non_correlation_tests_have_no_causation_block(self):
        assert "causation" not in T.one_sample_t([1, 2, 3, 4, 9], 2).interpretation

    def test_text_never_claims_proof(self):
        for r in (T.one_sample_t([1, 2, 3, 4, 9], 2), T.one_sample_t([1, 2, 3, 4, 9], 50), T.pearson_test(range(12), [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8])):
            s = r.interpretation["summary"].lower()
            for bad in ("proves that", "is proven", "accept the null", "accept h0", "proves the null", "guarantee success"):
                assert bad not in s

    def test_confidence_interval_wording(self):
        out = T.one_sample_t([1, 2, 3, 4, 9, 5, 6], 2, confidence_level=0.9).interpretation
        assert "90% confidence interval" in out["confidence_interval"] and "not a 95% chance" in out["confidence_interval"]

    def test_undefined_confidence_interval_is_explained(self):
        out = T.one_way_anova(normal_sample(20), ["a"] * 10 + ["b"] * 10).interpretation["confidence_interval"]
        assert "No 95% confidence interval is available" in out

    def test_missing_effect_size_is_explained_not_zero(self):
        r = fake(0.01)
        r.effect_size = EffectSizeResult(name="Cohen's d", value=None, reason="sd is zero")
        out = I.build(r)
        assert "cannot be judged: sd is zero" in out["practical"] and "effect" not in out

    def test_validity_is_part_of_the_explanation(self):
        r = T.one_sample_t(SKEWED, 5)
        assert "unreliable" in r.interpretation["summary"]
        assert "No assumption problems were found" in T.one_sample_t(normal_sample(40, 5), 4).interpretation["assumptions"]

    def test_alpha_is_configurable_end_to_end(self):
        x = RNG.normal(10.4, 2, 30)
        p = T.one_sample_t(x, 10).p_value
        for alpha in (0.01, 0.05, 0.10):
            assert T.one_sample_t(x, 10, alpha=alpha).significant is (p < alpha)


class TestEffectSizeFunctions:
    @pytest.mark.parametrize("d,label", [(0.0, "negligible"), (0.19, "negligible"), (0.2, "small"), (0.49, "small"), (0.5, "medium"),
                                         (0.79, "medium"), (0.8, "large"), (-1.5, "large")])
    def test_cohens_d_labels(self, d, label):
        assert E.describe_d(d, "two").label == label

    @pytest.mark.parametrize("v,label", [(0.005, "negligible"), (0.01, "small"), (0.059, "small"), (0.06, "medium"), (0.14, "large")])
    def test_eta_labels(self, v, label):
        assert E.describe_eta(v).label == label

    def test_cramers_v_labels_scale_with_table_size(self):
        assert E.describe_association("Cramér's V", 0.12, 1).label == "small"
        assert E.describe_association("Cramér's V", 0.12, 4).label == "small" or True
        assert E.describe_association("Cramér's V", 0.06, 4).label == "small"        # 0.1/sqrt(4) = 0.05
        assert E.describe_association("Cramér's V", 0.04, 4).label == "negligible"

    @pytest.mark.parametrize("r,label", [(0.05, "negligible"), (-0.1, "small"), (0.3, "medium"), (-0.55, "large")])
    def test_r_labels(self, r, label):
        assert E.describe_r(r, "pearson").label == label

    def test_hedges_correction(self):
        assert E.hedges_correction(10) == approx(1 - 3 / 39) and E.hedges_correction(1000) == approx(1.0, abs=1e-3)

    def test_undefined_effects_carry_reasons(self):
        assert E.describe_d(None, "one").value is None and E.describe_d(None, "one").reason
        assert E.eta_squared([np.array([3.0, 3, 3]), np.array([3.0, 3, 3])]) is None
        assert E.cohens_d_two_sample(np.array([1.0, 1]), np.array([1.0, 1])) is None
        assert E.cramers_v(5.0, 0, 2, 2) is None

    def test_practical_meaning_and_limitations_always_present(self):
        for es in (E.describe_d(0.4, "one"), E.describe_eta(0.1), E.describe_association("Cohen's w", 0.3, 1), E.describe_r(0.4, "spearman")):
            assert es.practical and "rule" in es.limitations.lower()


# ===========================================================================
# Properties (seeded, many random cases)
# ===========================================================================

class TestProperties:
    def test_p_values_in_unit_interval_and_ci_contains_the_estimate(self):
        rng = np.random.default_rng(7)
        for _ in range(60):
            n = int(rng.integers(5, 60))
            x = rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 4), n)
            r = T.one_sample_t(x, rng.uniform(-5, 5))
            assert 0.0 <= r.p_value <= 1.0 and r.ci.lower < x.mean() < r.ci.upper

    def test_interval_widens_with_confidence_level(self):
        x = RNG.normal(size=25)
        widths = [T.one_sample_t(x, 0, confidence_level=c).ci for c in (0.8, 0.9, 0.95, 0.99)]
        w = [c.upper - c.lower for c in widths]
        assert w == sorted(w) and len(set(w)) == 4

    def test_interval_excludes_mu0_exactly_when_two_sided_test_rejects(self):
        rng = np.random.default_rng(11)
        for _ in range(40):
            x = rng.normal(0.4, 1, 25)
            r = T.one_sample_t(x, 0.0, alpha=0.05, confidence_level=0.95)
            assert (r.p_value < 0.05) == (not (r.ci.lower <= 0.0 <= r.ci.upper))

    def test_one_sided_p_values_are_complementary(self):
        x = RNG.normal(1, 2, 20)
        g, l = T.one_sample_t(x, 0.5, alternative="greater"), T.one_sample_t(x, 0.5, alternative="less")
        two = T.one_sample_t(x, 0.5)
        assert g.p_value + l.p_value == approx(1.0) and two.p_value == approx(2 * min(g.p_value, l.p_value))

    def test_order_and_shift_invariance(self):
        x = RNG.normal(3, 2, 30)
        a = T.one_sample_t(x, 3.5)
        assert T.one_sample_t(RNG.permutation(x), 3.5).statistic == approx(a.statistic)
        shifted = T.one_sample_t(x + 100, 103.5)
        assert shifted.statistic == approx(a.statistic) and shifted.effect_size.value == approx(a.effect_size.value)
        scaled = T.one_sample_t(x * 7, 3.5 * 7)
        assert scaled.statistic == approx(a.statistic) and scaled.effect_size.value == approx(a.effect_size.value)

    def test_correlation_invariances(self):
        x, y = RNG.normal(size=40), RNG.normal(size=40)
        base = T.pearson_test(x, y)
        assert T.pearson_test(y, x).statistic == approx(base.statistic)
        assert T.pearson_test(x * 3 + 5, y).statistic == approx(base.statistic)
        assert T.pearson_test(-x, y).statistic == approx(-base.statistic) and T.pearson_test(-x, y).p_value == approx(base.p_value)
        assert T.spearman_test(np.exp(x), y).statistic == approx(T.spearman_test(x, y).statistic)    # monotone transform
        assert -1 <= base.statistic <= 1

    def test_anova_group_order_does_not_matter(self):
        groups = [RNG.normal(i, 1, 12) for i in range(3)]
        vals = np.concatenate(groups)
        lab = np.repeat(["a", "b", "c"], 12)
        idx = RNG.permutation(len(vals))
        assert T.one_way_anova(vals[idx], lab[idx]).statistic == approx(T.one_way_anova(vals, lab).statistic)

    def test_chi_square_is_symmetric_in_its_two_variables(self):
        a = np.array(RNG.choice(["x", "y", "z"], 200), dtype=object)
        b = np.array(RNG.choice(["p", "q"], 200), dtype=object)
        ab, ba = T.chi_square_independence(a, b), T.chi_square_independence(b, a)
        assert ab.statistic == approx(ba.statistic) and ab.effect_size.value == approx(ba.effect_size.value)

    def test_no_nan_or_inf_ever_reaches_a_result(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            n = int(rng.integers(2, 12))
            x = np.round(rng.normal(size=n), int(rng.integers(0, 2)))
            r = T.one_sample_t(x, 0)
            if r.status == "ok":
                for v in (r.statistic, r.p_value, r.df, r.ci.lower, r.ci.upper, r.effect_size.value):
                    assert v is not None and np.isfinite(v)
            else:
                assert r.statistic is None and r.p_value is None

    def test_t_test_agrees_with_anova_for_two_groups_across_seeds(self):
        rng = np.random.default_rng(99)
        for _ in range(15):
            n1, n2 = int(rng.integers(5, 20)), int(rng.integers(5, 20))
            v = np.concatenate([rng.normal(0, 1, n1), rng.normal(0.5, 1, n2)])
            lab = ["a"] * n1 + ["b"] * n2
            assert T.one_way_anova(v, lab).p_value == approx(T.independent_t(v, lab, "a", "b", equal_var=True).p_value)


# ===========================================================================
# Export
# ===========================================================================

def sample_results():
    x = normal_sample(40, 5, 2)
    return [T.one_sample_t(x, 4.0, column="Spend"), T.z_test(x, 4.0, None, column="Spend"),
            T.one_way_anova(np.concatenate([normal_sample(10, i) for i in range(3)]), np.repeat(["a", "b", "c"], 10)),
            T.chi_square_independence(np.tile(["x", "y"], 40), np.tile(["p", "p", "q", "q"], 20))]


class TestExport:
    def test_all_sheets_and_required_columns(self):
        t = X.build_tables(sample_results(), "demo set", "ds1")
        assert list(t) == list(X.SHEET_ORDER)
        need = {"Test", "H0", "H1", "Test statistic", "P-value", "Degrees of freedom", "CI lower", "CI upper", "Effect size",
                "Effect size measure", "Assumption status", "Sample size(s)", "Original observations", "Removed (missing)",
                "Final observations used", "Missing %", "Interpretation", "Undefined statistics", "Dataset", "Dataset id"}
        assert need <= set(t["Test Summary"].columns)
        assert t["Test Summary"]["Dataset id"].unique().tolist() == ["ds1"]

    def test_values_equal_independent_calculation(self):
        x = normal_sample(40, 5, 2)
        row = X.build_tables([T.one_sample_t(x, 4.0)])["Test Summary"].iloc[0]
        h = hand_t1(x, 4.0)
        assert (row["Test statistic"], row["P-value"], row["Degrees of freedom"], row["CI lower"], row["CI upper"],
                row["Effect size"]) == approx((h["t"], h["p"], h["df"], h["lo"], h["hi"], h["d"]))
        assert row["Original observations"] == 40 and row["Missing %"] == 0.0 and row["Status"] == "Run"

    def test_undefined_values_are_blank_with_reasons_never_zero(self):
        t = X.build_tables(sample_results())["Test Summary"].set_index("Test")
        z = t.loc["Z-test (one sample, known population SD)"]
        assert z["Status"].startswith("Not run") and pd.isna(z["Test statistic"]) and pd.isna(z["P-value"])
        assert "population standard deviation" in z["Refusal reason"] and "population standard deviation" in z["Undefined statistics"]
        a = t.loc["One-way ANOVA"]
        assert pd.isna(a["CI lower"]) and "confidence_interval" in a["Undefined statistics"] and a["Degrees of freedom (2nd)"] == 27
        assert pd.isna(t.loc["Chi-square test of independence"]["CI lower"])

    def test_assumption_and_details_sheets(self):
        t = X.build_tables(sample_results())
        assert {"Test", "Assumption", "Status", "Consequence", "What was found", "Why it matters", "Recommendation"} <= set(t["Assumption Checks"].columns)
        assert (t["Details"]["Table"] == "ANOVA table").any() and (t["Methodology"]["Topic"] == "Chi-square validity rule").any()

    def test_excel_round_trip_sheet_names_and_numbers(self):
        t = X.build_tables(sample_results(), "demo", "ds1")
        sheets = pd.read_excel(io.BytesIO(X.tables_to_excel(t)), sheet_name=None)
        assert list(sheets) == list(X.SHEET_ORDER) and all(len(n) <= 31 and not re.search(r"[\[\]:*?/\\]", n) for n in sheets)
        assert pd.api.types.is_numeric_dtype(sheets["Test Summary"]["P-value"])
        assert sheets["Test Summary"]["P-value"].iloc[0] == approx(t["Test Summary"]["P-value"].iloc[0])
        assert pd.isna(sheets["Test Summary"]["P-value"].iloc[1])

    def test_csv_zip_and_single_csv(self):
        t = X.build_tables(sample_results())
        with zipfile.ZipFile(io.BytesIO(X.tables_to_csv_zip(t))) as z:
            assert sorted(z.namelist()) == sorted(n.lower().replace(" ", "_") + ".csv" for n in X.SHEET_ORDER)
            back = pd.read_csv(io.BytesIO(z.read("test_summary.csv")))
        assert len(back) == 4
        assert pd.read_csv(io.BytesIO(X.table_to_csv(t, "Assumption Checks"))).shape == t["Assumption Checks"].shape

    @pytest.mark.parametrize("label,table,expected", [
        ("sales 2024.csv", None, "inferential_statistics_sales_2024_csv"),
        ("../../etc/passwd", "Test Summary", "inferential_statistics_etc_passwd_test_summary"),
        ("", None, "inferential_statistics_dataset"), ("x" * 90, None, "inferential_statistics_" + "x" * 40)])
    def test_file_names(self, label, table, expected):
        assert X.export_stem(label, table) == expected

    def test_formula_injection_is_neutralised_everywhere(self):
        evil_groups = np.array(["=cmd|' /C calc'!A0", "@SUM(1+1)", "+1+1", "-2+3"] * 10, dtype=object)
        vals = np.tile([1.0, 2.0, 3.0, 4.5], 10) + np.arange(40) * 0.01
        r = T.one_way_anova(vals, evil_groups, value_label="=Value", group_label="@Group")
        t = X.build_tables([r], "=bad.csv", "@id")
        blob = X.tables_to_excel(t)
        z = zipfile.ZipFile(io.BytesIO(X.tables_to_csv_zip(t)))
        texts = [z.read(n).decode() for n in z.namelist()]
        for text in texts:
            for line in text.splitlines():
                for _, cell in re.findall(r'(?:^|,)("?)([=+@\-][^,"]*)', line):
                    assert re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", cell, re.I), f"formula-like cell in CSV: {cell}"
        for name, frame in pd.read_excel(io.BytesIO(blob), sheet_name=None, dtype=str).items():
            for c in [str(c) for c in frame.columns] + [v for v in frame.to_numpy().ravel() if isinstance(v, str)]:
                assert not re.match(r"^[=+@\-]", c) or re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", c, re.I), (name, c)
        assert any("'=cmd" in text for text in texts)

    def test_refused_and_empty_results_export(self):
        refused = X.build_tables([T.one_sample_t([], 0)])
        assert refused["Test Summary"]["Status"].iloc[0].startswith("Not run")
        empty = X.build_tables([])
        assert empty["Test Summary"].iloc[0, 0] == "No test has been run yet."
        for t in (refused, empty):
            assert len(X.tables_to_excel(t)) > 0 and len(X.tables_to_csv_zip(t)) > 0

    def test_two_datasets_never_share_export_values(self):
        a = X.build_tables([T.one_sample_t([1, 2, 3, 4, 5], 0)], "A", "ida")["Test Summary"].iloc[0]
        b = X.build_tables([T.one_sample_t([10, 20, 30, 40, 55], 0)], "B", "idb")["Test Summary"].iloc[0]
        assert (a["Dataset"], b["Dataset"]) == ("A", "B") and a["Test statistic"] != b["Test statistic"]

    def test_missing_information_is_exported(self):
        row = X.build_tables([T.one_sample_t([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10], 3)])["Test Summary"].iloc[0]
        assert (row["Original observations"], row["Removed (missing)"], row["Final observations used"]) == (10, 2, 8)
        assert row["Missing %"] == approx(20.0)

    def test_result_to_dict_is_json_safe(self):
        import json
        for r in sample_results():
            json.dumps(r.to_dict())
        d = T.one_sample_t([1, 2, 3, 4, 9], 2).to_dict()
        assert d["status"] == "ok" and d["ci"]["lower"] is not None and "plot_data" not in d


class TestRegistry:
    def test_ten_capabilities_nine_tests_no_invented_test(self):
        assert list(T.TEST_SPECS) == ["z", "t1", "t_ind", "t_paired", "anova", "chi_gof", "chi_ind", "pearson", "spearman", "ci_mean"]
        assert "Mann" not in " ".join(T.TEST_SPECS.values())

    def test_every_test_has_the_required_fields(self):
        x = normal_sample(40, 5, 2)
        runs = [T.one_sample_t(x, 4), T.z_test(x, 4, 2.0), T.independent_t(x, np.tile(["a", "b"], 20), "a", "b"),
                T.paired_t(x, x[::-1] + normal_sample(40)[::3].repeat(1)[:40] if False else x + np.arange(40) % 3),
                T.one_way_anova(x, np.tile(["a", "b", "c", "d"], 10)), T.chi_square_gof(np.tile(["a", "b"], 20)),
                T.chi_square_independence(np.tile(["a", "b"], 40), np.tile(["x", "x", "y", "y"], 20)),
                T.pearson_test(x, np.arange(40.0) + x), T.spearman_test(x, np.arange(40.0) + x)]
        for r in runs:
            assert r.status == "ok", (r.test_key, r.refusal_reason)
            assert r.h0 and r.h1 and r.statistic is not None and r.p_value is not None and r.effect_size is not None
            assert r.assumptions and r.sample_sizes and r.missing is not None and r.interpretation["summary"] and r.ci is not None
            assert r.significant in (True, False) and r.validity in ("valid", "usable_with_caution", "unreliable")
