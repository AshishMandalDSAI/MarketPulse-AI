"""Phase 8 Part 2 - Descriptive Statistics engine tests.

Every expected value is calculated independently (Python ``statistics``/``math``,
numpy, scipy, or by hand) and compared with the engine's result. No test only
checks that "something is returned".
"""

import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.descriptive.contracts import DescriptiveSettings  # noqa: E402
from src.descriptive.engine import (  # noqa: E402
    analyze_dataframe, detect_outliers, pairwise_matrix, summarize_categorical, summarize_datetime,
    summarize_numeric_values, validate_percentiles,
)
from tests.helpers import GENERIC_FIXTURES_DIR, all_diverse_fixtures, load_fixture  # noqa: E402

approx = pytest.approx


def num(values, **kw):
    return summarize_numeric_values(np.asarray(values, dtype=float), "x", **kw)


def one_col(values, name="x"):
    return analyze_dataframe(pd.DataFrame({name: values}))


def col(report, name, kind="numeric"):
    return next(c for c in getattr(report, kind) if c.column == name)


# ---------------------------------------------------------------------------
# A. Numeric statistics
# ---------------------------------------------------------------------------

class TestNumericExactValues:
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]          # textbook example, hand-computed below

    def test_hand_computed_values(self):
        s = num(self.DATA)
        assert s.valid_count == 8 and s.total == 40 and s.mean == 5
        assert s.median == 4.5 and s.modes == [4.0] and s.mode_count == 3
        assert (s.minimum, s.maximum, s.value_range) == (2, 9, 7)
        assert s.variance_population == approx(4.0) and s.std_population == approx(2.0)
        assert s.variance_sample == approx(32 / 7) and s.std_sample == approx(math.sqrt(32 / 7))
        assert (s.q1, s.q2, s.q3, s.iqr) == (4.0, 4.5, 5.5, 1.5)   # linear interpolation, positions 1.75 / 5.25

    def test_skewness_and_kurtosis_follow_the_documented_conventions(self):
        s = num(self.DATA)
        n = 8
        mean = sum(self.DATA) / n
        m2 = sum((v - mean) ** 2 for v in self.DATA) / n
        m3 = sum((v - mean) ** 3 for v in self.DATA) / n
        m4 = sum((v - mean) ** 4 for v in self.DATA) / n
        g1 = m3 / m2 ** 1.5
        g2 = m4 / m2 ** 2 - 3
        assert s.skewness == approx(math.sqrt(n * (n - 1)) / (n - 2) * g1)           # adjusted Fisher-Pearson
        assert s.kurtosis == approx((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6))
        assert s.skewness == approx(sps.skew(self.DATA, bias=False))
        assert s.kurtosis == approx(sps.kurtosis(self.DATA, bias=False))

    @pytest.mark.parametrize("seed", range(25))
    def test_random_data_matches_independent_libraries(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(rng.uniform(-50, 50), rng.uniform(0.1, 20), rng.integers(5, 400))
        s = num(x)
        assert s.mean == approx(statistics.fmean(x), rel=1e-12)
        assert s.median == approx(statistics.median(x))
        assert s.variance_sample == approx(statistics.variance(x), rel=1e-10)
        assert s.std_sample == approx(statistics.stdev(x), rel=1e-10)
        assert s.variance_population == approx(statistics.pvariance(x), rel=1e-10)
        assert s.skewness == approx(sps.skew(x, bias=False), rel=1e-8, abs=1e-12)
        assert s.kurtosis == approx(sps.kurtosis(x, bias=False), rel=1e-8, abs=1e-10)
        assert (s.q1, s.q3) == approx(tuple(np.percentile(x, [25, 75])))
        assert s.total == approx(math.fsum(x))

    @pytest.mark.parametrize("pct,expected", [(0, 1.0), (10, 2.0), (25, 3.5), (50, 6.0), (90, 10.0), (100, 11.0)])
    def test_percentiles_use_linear_interpolation(self, pct, expected):
        # values 1..11: position = p/100 * 10 (P25 -> 2.5, halfway between 3 and 4)
        s = num(range(1, 12), settings=DescriptiveSettings(percentiles=(pct,)))
        assert s.percentiles[float(pct)] == approx(expected)

    def test_percentile_between_ranks(self):
        s = num([1, 2, 3, 4], settings=DescriptiveSettings(percentiles=(95, 33)))
        assert s.percentiles[95.0] == approx(3.85)          # 0.95 * 3 = 2.85 -> 3 + 0.85
        assert s.percentiles[33.0] == approx(1.99)          # 0.33 * 3 = 0.99 -> 1 + 0.99

    def test_percentiles_are_monotonic_and_consistent_with_quartiles(self):
        rng = np.random.default_rng(3)
        s = num(rng.exponential(3, 500), settings=DescriptiveSettings(percentiles=(1, 25, 50, 75, 99, 5, 95)))
        keys = sorted(s.percentiles)
        assert [s.percentiles[k] for k in keys] == sorted(s.percentiles[k] for k in keys)
        assert s.percentiles[25.0] == approx(s.q1) and s.percentiles[75.0] == approx(s.q3)
        assert s.percentiles[50.0] == approx(s.median)

    @pytest.mark.parametrize("values", [[3, 1, 2, 5, 4], [-3, -1, -2, -5, -4], [0, 0, 1, 0, 2], [0.1, 0.2, 0.3, 0.4, 0.7]])
    def test_positive_negative_zero_and_decimal_values(self, values):
        s = num(values)
        assert s.mean == approx(sum(values) / 5) and s.median == approx(statistics.median(values))
        assert s.minimum == min(values) and s.maximum == max(values)
        assert s.zero_count == sum(1 for v in values if v == 0)
        assert s.negative_count == sum(1 for v in values if v < 0)
        assert s.variance_sample == approx(statistics.variance(values))

    def test_decimal_sum_is_exactly_rounded(self):
        s = num([0.1] * 10)
        assert s.total == 1.0 and s.mean == 0.1            # naive summation would give 0.9999999999999999

    def test_large_values_keep_precision(self):
        s = num([1e15 + 1, 1e15 + 2, 1e15 + 3])
        assert s.mean == approx(1e15 + 2) and s.variance_sample == approx(1.0)

    def test_values_too_large_to_sum_are_undefined_not_inf(self):
        s = num([1e308, 1e308, 1e308])
        assert s.mean is None and s.total is None and s.variance_sample is None
        assert "overflow" in s.undefined["mean"] and s.skewness is None and s.kurtosis is None
        assert s.minimum == 1e308 and s.median == 1e308

    def test_zero_variance_ties_and_modes(self):
        assert num([1, 2, 2, 3, 3, 4]).modes == [2.0, 3.0]                  # tie: all modes, ascending
        assert "tie" in num([1, 2, 2, 3, 3, 4]).mode_note
        s = num([5, 3, 1])
        assert s.modes == [] and s.mode_count is None and "mode" in s.undefined and "once" in s.mode_note

    def test_mode_order_does_not_depend_on_input_order(self):
        a = num([9, 1, 9, 1, 5]).modes
        b = num([1, 5, 9, 1, 9]).modes
        assert a == b == [1.0, 9.0]


class TestConstantAndTinySamples:
    @pytest.mark.parametrize("value,n", [(7.0, 2), (7.0, 50), (0.0, 5), (-3.5, 3), (1e9, 4)])
    def test_constant_column_rules(self, value, n):
        s = num([value] * n)
        assert s.is_constant
        assert s.mean == value and s.median == value
        assert s.variance_sample == 0 and s.std_sample == 0 and s.variance_population == 0
        assert s.value_range == 0
        assert s.skewness is None and s.kurtosis is None              # undefined, NOT 0
        assert "identical" in s.undefined["skewness"] and "identical" in s.undefined["kurtosis"]

    def test_one_observation(self):
        s = num([42.0])
        assert s.mean == s.median == s.minimum == s.maximum == 42.0 and s.modes == [42.0]
        assert s.variance_sample is None and s.std_sample is None          # undefined, NOT 0
        assert s.skewness is None and s.kurtosis is None
        assert "at least 2" in s.undefined["variance"] and "at least 2" in s.undefined["standard deviation"]
        assert "at least 3" in s.undefined["skewness"] and "at least 4" in s.undefined["kurtosis"]
        assert s.variance_population == 0.0                                  # defined: a single point has no spread

    def test_two_observations(self):
        s = num([1.0, 3.0])
        assert s.variance_sample == approx(2.0) and s.std_sample == approx(math.sqrt(2))
        assert s.skewness is None and s.kurtosis is None and "at least 3" in s.undefined["skewness"]

    def test_three_observations_give_skewness_but_not_kurtosis(self):
        s = num([1.0, 2.0, 6.0])
        assert s.skewness == approx(sps.skew([1, 2, 6], bias=False))
        assert s.kurtosis is None and "at least 4" in s.undefined["kurtosis"]

    def test_four_observations_give_kurtosis(self):
        s = num([1.0, 2.0, 6.0, 7.5])
        assert s.kurtosis == approx(sps.kurtosis([1, 2, 6, 7.5], bias=False))

    def test_empty_array(self):
        s = num([])
        assert s.valid_count == 0 and s.mean is None and s.median is None and s.minimum is None
        assert all(k in s.undefined for k in ("mean", "median", "variance", "skewness", "kurtosis"))


class TestNumericThroughTheReport:
    def test_all_missing_column_is_empty_not_numeric(self):
        r = one_col([np.nan] * 6)
        assert r.empty_columns == ["x"] and not r.numeric
        assert r.missing.rows[0]["missing_pct"] == 100.0

    def test_nan_missing_and_infinite_are_counted_separately(self):
        r = one_col([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 5.0, None, 7.0, 8.0])
        s = col(r, "x")
        assert (s.sample_size, s.valid_count, s.missing_count, s.infinite_count) == (10, 6, 2, 2)
        assert s.missing_pct == approx(20.0)                       # 2 of ALL 10 rows
        assert s.mean == approx(statistics.fmean([1, 2, 4, 5, 7, 8]))
        assert s.maximum == 8 and s.minimum == 1                   # infinity never leaks into min/max
        assert any("infinite" in w for w in s.warnings)

    def test_missing_is_not_zero(self):
        with_nan = col(one_col([10.0, np.nan, 20.0, np.nan, 30.0]), "x")
        with_zero = col(one_col([10.0, 0.0, 20.0, 0.0, 30.0]), "x")
        assert with_nan.mean == approx(20.0) and with_zero.mean == approx(12.0)
        assert with_nan.zero_count == 0 and with_zero.zero_count == 2
        assert with_nan.valid_count == 3 and with_zero.valid_count == 5

    @pytest.mark.parametrize("n_missing,n_rows", [(3, 10), (1, 8), (7, 20), (0, 9), (9, 10)])
    def test_missing_percentage_uses_all_rows_as_denominator(self, n_missing, n_rows):
        vals = [float(i % 5) + (i // 5) * 10 for i in range(n_rows)]
        for i in range(n_missing):
            vals[i] = np.nan
        r = one_col(vals)
        row = r.missing.rows[0]
        assert row["missing_count"] == n_missing and row["missing_pct"] == approx(n_missing / n_rows * 100)
        assert row["valid_count"] == n_rows - n_missing

    def test_numeric_strings_are_converted_and_the_decision_is_reported(self):
        r = one_col(["1", " 2.5 ", "3", None, "4"])
        s = col(r, "x")
        assert s.converted_from_text and s.valid_count == 4 and s.mean == approx(2.625)
        assert any("converted to numbers" in c for c in r.conversions)

    def test_mixed_invalid_strings_are_not_converted_and_nothing_is_lost(self):
        r = one_col(["1", "2", "three", "4", "5"])
        assert r.column_kinds["x"] == "categorical"
        c = col(r, "x", "categorical")
        assert c.valid_count == 5 and c.unique_count == 5          # all five values still counted
        assert any("NOT converted" in x and "4 of 5" in x for x in r.conversions)

    def test_leading_zero_codes_are_kept_as_text(self):
        r = one_col(["001", "002", "010", "007"])
        assert r.column_kinds["x"] == "categorical"
        assert [t["value"] for t in col(r, "x", "categorical").frequency] == ["001", "002", "007", "010"]

    def test_integer_and_float_dtypes_and_nullable_ints(self):
        for series in (pd.Series([1, 2, 3, 4, 5], dtype="int64"), pd.Series([1.5, 2.5, 3.5, 9.0], dtype="float32"),
                       pd.Series([1, 2, None, 4, 8], dtype="Int64")):
            r = analyze_dataframe(pd.DataFrame({"x": series}))
            assert r.column_kinds["x"] == "numeric"

    def test_zero_one_numeric_column_is_a_yes_no_column(self):
        r = one_col([0, 1, 1, 0, 1, 1, 0, 1])
        assert r.column_kinds["x"] == "boolean"
        b = col(r, "x", "boolean")
        assert (b.true_count, b.false_count) == (5, 3) and b.true_pct == approx(62.5)
        assert any("0/1" in c for c in r.conversions)

    def test_constant_zero_or_one_column_stays_numeric(self):
        for v in (0, 1):
            assert one_col([v] * 6).column_kinds["x"] == "numeric"


# ---------------------------------------------------------------------------
# Property-style tests (seeded, deterministic; the `hypothesis` package is not a project dependency)
# ---------------------------------------------------------------------------

class TestNumericProperties:
    @pytest.mark.parametrize("seed", range(40))
    def test_invariants_hold_for_random_data(self, seed):
        rng = np.random.default_rng(1000 + seed)
        x = np.round(rng.normal(0, rng.uniform(0.5, 30), rng.integers(2, 120)), rng.integers(0, 4))
        s = num(x)
        assert s.minimum <= s.q1 <= s.median <= s.q3 <= s.maximum
        assert s.minimum <= s.mean <= s.maximum
        assert s.iqr == approx(s.q3 - s.q1) and s.value_range == approx(s.maximum - s.minimum)
        assert s.variance_population >= 0 and (s.variance_sample is None or s.variance_sample >= 0)
        if s.variance_sample is not None:
            assert s.std_sample == approx(math.sqrt(s.variance_sample))
        perm = rng.permutation(x)
        t = num(perm)
        assert (t.mean, t.median, t.q1, t.q3, t.modes) == approx((s.mean, s.median, s.q1, s.q3, s.modes))

    @pytest.mark.parametrize("seed", range(15))
    def test_shift_and_scale_laws(self, seed):
        rng = np.random.default_rng(2000 + seed)
        x = rng.gamma(2.0, 3.0, 60)
        a, b = rng.uniform(-20, 20), rng.uniform(0.5, 5)
        base, moved = num(x), num(a + b * x)
        assert moved.mean == approx(a + b * base.mean)
        assert moved.std_sample == approx(b * base.std_sample)
        assert moved.skewness == approx(base.skewness)                       # location/scale free
        assert moved.kurtosis == approx(base.kurtosis)
        assert num(-x).skewness == approx(-base.skewness)                    # mirror flips the sign


# ---------------------------------------------------------------------------
# B. Categorical statistics
# ---------------------------------------------------------------------------

class TestCategorical:
    S = DescriptiveSettings()

    def cat(self, values, name="c", settings=None):
        return summarize_categorical(pd.Series(values), name, settings or self.S)

    def test_low_cardinality_frequencies_and_percentages(self):
        c = self.cat(["a"] * 5 + ["b"] * 3 + ["c"] * 2)
        assert c.valid_count == 10 and c.unique_count == 3
        assert [(t["value"], t["count"]) for t in c.frequency] == [("a", 5), ("b", 3), ("c", 2)]
        assert [round(t["pct"], 6) for t in c.frequency] == [50.0, 30.0, 20.0]
        assert c.modes == ["a"] and c.mode_count == 5 and c.mode_pct == 50.0
        assert [t["value"] for t in c.least_frequent] == ["c", "b", "a"]

    def test_multiple_modes_are_all_reported_in_sorted_order(self):
        c = self.cat(["b", "a", "b", "a", "c"])
        assert c.modes == ["a", "b"] and c.mode_count == 2 and "tie" in c.mode_note
        assert [t["value"] for t in c.frequency] == ["a", "b", "c"]            # ties broken by label, deterministically

    def test_every_category_once_means_a_full_tie(self):
        c = self.cat(["x", "y", "z"])
        assert c.modes == ["x", "y", "z"] and "once" in c.mode_note

    def test_single_category(self):
        c = self.cat(["a"] * 6)
        assert c.unique_count == 1 and c.modes == ["a"] and c.mode_pct == 100.0 and c.mode_note == "Only one category."

    def test_missing_and_blank_categories(self):
        c = self.cat(["a", None, "b", "", "  ", "a", np.nan, "b", "a", "c"])
        assert c.sample_size == 10 and c.missing_count == 4 and c.blank_count == 2
        assert c.missing_pct == approx(40.0) and c.valid_count == 6
        assert [(t["value"], t["count"]) for t in c.frequency] == [("a", 3), ("b", 2), ("c", 1)]
        assert any("blank" in w for w in c.warnings)
        assert c.frequency[0]["pct"] == approx(50.0)                         # percentages are of VALID values

    def test_all_missing_categories(self):
        c = self.cat([None, "", np.nan])
        assert c.valid_count == 0 and c.modes == [] and "mode" in c.undefined and c.frequency == []

    def test_mixed_text_and_numbers_are_kept_apart_from_numeric_conversion(self):
        c = self.cat(["a", 1, "b", 1, 2.5, "a"])
        assert dict((t["value"], t["count"]) for t in c.frequency) == {"a": 2, "1": 2, "b": 1, "2.5": 1}

    def test_unicode_and_unusual_labels(self):
        vals = ["café", "日本語", "😀", "café", "naïve — x", "日本語", "café"]
        c = self.cat(vals)
        assert c.modes == ["café"] and c.mode_count == 3
        assert {t["value"] for t in c.frequency} == {"café", "日本語", "😀", "naïve — x"}

    @pytest.mark.parametrize("name", ["Weird Name!", "col.with.dots", "日本", "=cmd", "a/b\\c", "  spaced  ", "x" * 80])
    def test_unusual_column_names_work_end_to_end(self, name):
        df = pd.DataFrame({name: ["a", "b", "a", "c"], "n": [1.5, 2.5, 3.5, 9.5]})
        r = analyze_dataframe(df)
        assert name in r.column_kinds and col(r, name, "categorical").modes == ["a"]

    def test_high_cardinality_is_truncated_and_disclosed(self):
        vals = [f"id{i % 500:04d}" for i in range(2000)] + ["top"] * 50
        c = self.cat(vals)
        assert c.unique_count == 501 and c.high_cardinality and c.frequency_truncated
        assert len(c.frequency) == 50 and c.frequency[0]["value"] == "top"
        assert any("501 different categories" in w for w in c.warnings)

    def test_top_n_setting(self):
        c = self.cat([str(i % 12) for i in range(120)] + ["0"] * 5, settings=DescriptiveSettings(top_n=3))
        assert len(c.top_categories) == 3 and len(c.least_frequent) == 3 and c.top_categories[0]["value"] == "0"

    def test_unhashable_cells_do_not_crash(self):
        c = self.cat([[1, 2], {"a": 1}, [1, 2], "x"])
        assert c.valid_count == 4 and c.modes == ["[1, 2]"]

    def test_pandas_categorical_dtype(self):
        r = analyze_dataframe(pd.DataFrame({"c": pd.Categorical(["a", "b", "a", "c", "a"])}))
        assert col(r, "c", "categorical").modes == ["a"]


# ---------------------------------------------------------------------------
# C. Datetime statistics
# ---------------------------------------------------------------------------

class TestDatetime:
    def dt(self, values, name="d"):
        r = analyze_dataframe(pd.DataFrame({name: values, "pad": range(len(values))}))
        return r, col(r, name, "datetime")

    def test_valid_dates_range_and_monthly_periods(self):
        r, d = self.dt(pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10", "2024-03-01", "2024-03-31"]))
        assert d.valid_count == 5 and d.missing_count == 0
        assert d.earliest.startswith("2024-01-05") and d.latest.startswith("2024-03-31")
        assert d.range_days == approx(86.0) and d.unique_count == 5
        assert d.period == "month"
        assert d.period_counts == [{"period": "2024-01", "count": 2}, {"period": "2024-02", "count": 1},
                                   {"period": "2024-03", "count": 2}]

    def test_text_dates_with_invalid_and_missing_values(self):
        r, d = self.dt(["2024-01-01", "2024-01-02", "not a date", None, "2024-01-04", "2024-01-05", "2024-01-06",
                        "2024-01-07", "2024-01-08", "2024-01-09"])
        assert r.column_kinds["d"] == "datetime"
        assert (d.valid_count, d.missing_count, d.invalid_count) == (8, 1, 1)
        assert d.missing_pct == approx(10.0)
        assert d.period == "day" and sum(p["count"] for p in d.period_counts) == 8
        assert any("could not be read" in w for w in d.warnings)

    def test_mixed_date_formats(self):
        r, d = self.dt(["2024-01-05", "01/06/2024", "Jan 7, 2024", "2024-01-08", "2024/01/09", "10 Jan 2024"])
        assert d.valid_count == 6 and d.invalid_count == 0
        assert d.earliest.startswith("2024-01-05") and d.latest.startswith("2024-01-10")

    def test_single_date_has_no_period_breakdown(self):
        r, d = self.dt(pd.to_datetime(["2024-05-05", None, None]))
        assert d.valid_count == 1 and d.range_days == 0 and d.period is None and d.period_counts == []
        assert "period_counts" in d.undefined

    def test_long_range_uses_years(self):
        r, d = self.dt(pd.to_datetime(["2015-06-01", "2018-02-01", "2022-09-30", "2022-10-01", "2020-01-01"]))
        assert d.period == "year" and {p["period"] for p in d.period_counts} == {"2015", "2018", "2020", "2022"}

    def test_timezone_aware_dates(self):
        from datetime import timedelta, timezone
        s = pd.to_datetime(["2024-03-01 10:00", "2024-03-05 12:30", "2024-04-01 00:00"]).tz_localize(timezone(timedelta(hours=-5)))
        r, d = self.dt(s)
        assert d.timezone is not None
        assert d.earliest.startswith("2024-03-01T10:00:00-05:00") and d.valid_count == 3 and d.period == "day"      # 31-day span -> daily

    def test_mixed_timezone_text_is_converted_to_utc_and_disclosed(self):
        r, d = self.dt(["2024-03-01 10:00:00+00:00", "2024-03-02 10:00:00+05:30", "2024-03-03 10:00:00-04:00",
                        "2024-03-04 08:00:00+00:00"])
        assert d.valid_count == 4 and d.timezone is not None
        assert any("UTC" in w for w in d.warnings)

    def test_direct_function_on_all_missing(self):
        d = summarize_datetime(pd.Series([None, None], dtype="datetime64[ns]"), "d")
        assert d.valid_count == 0 and d.earliest is None and "range" in d.undefined


# ---------------------------------------------------------------------------
# D. Correlation and covariance
# ---------------------------------------------------------------------------

def pm(cols, kind="correlation", method="pearson"):
    return pairwise_matrix({k: np.asarray(v, dtype=float) for k, v in cols.items()}, kind, method)


class TestCorrelation:
    def test_perfect_positive_and_negative(self):
        x = np.arange(1, 11, dtype=float)
        m = pm({"x": x, "up": 3 * x + 2, "down": -0.5 * x + 9})
        assert m.value("x", "up") == approx(1.0) and m.value("x", "down") == approx(-1.0)
        assert m.value("up", "down") == approx(-1.0) and m.value("x", "x") == 1.0

    @pytest.mark.parametrize("seed", range(12))
    def test_pearson_and_spearman_match_scipy(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=60)
        y = 0.4 * x + rng.normal(size=60)
        y[::7] = np.round(y[::7], 0)                       # create ties
        assert pm({"x": x, "y": y}).value("x", "y") == approx(sps.pearsonr(x, y).statistic, abs=1e-12)
        assert pm({"x": x, "y": y}, method="spearman").value("x", "y") == approx(sps.spearmanr(x, y).statistic, abs=1e-12)

    def test_spearman_sees_monotone_nonlinear_relationship(self):
        x = np.arange(1, 21, dtype=float)
        y = np.exp(x / 3)
        assert pm({"x": x, "y": y}, method="spearman").value("x", "y") == approx(1.0)
        assert pm({"x": x, "y": y}).value("x", "y") < 0.99

    def test_no_correlation_is_near_zero_but_defined(self):
        rng = np.random.default_rng(11)
        v = pm({"a": rng.normal(size=4000), "b": rng.normal(size=4000)}).value("a", "b")
        assert v is not None and abs(v) < 0.06

    def test_constant_column_is_undefined_not_zero(self):
        m = pm({"x": [1, 2, 3, 4, 5], "k": [7, 7, 7, 7, 7]})
        assert m.value("x", "k") is None and m.value("k", "k") is None
        assert "identical" in m.reasons["x || k"] and "'k'" in m.reasons["x || k"]
        assert m.value("x", "x") == 1.0

    def test_constant_only_within_the_pairwise_rows_is_undefined(self):
        # b is not constant overall, but where a has values b is always 5
        a = [1, 2, 3, np.nan, np.nan]
        b = [5, 5, 5, 8, 9]
        m = pm({"a": a, "b": b})
        assert m.value("a", "b") is None and m.n_pairs[0][1] == 3

    def test_one_valid_pair_and_insufficient_pairs(self):
        m = pm({"a": [1, np.nan, np.nan, np.nan], "b": [2, 3, 4, 5]})
        assert m.n_pairs[0][1] == 1 and m.value("a", "b") is None and "only 1" in m.reasons["a || b"]
        two = pm({"a": [1, 2, np.nan], "b": [3, 5, 6]})
        assert two.n_pairs[0][1] == 2 and two.value("a", "b") is None and "at least 3" in two.reasons["a || b"]

    def test_three_pairs_is_enough(self):
        assert pm({"a": [1, 2, 4], "b": [2, 4, 9]}).value("a", "b") == approx(sps.pearsonr([1, 2, 4], [2, 4, 9]).statistic)

    def test_pairwise_complete_observations_with_missing_values(self):
        rng = np.random.default_rng(5)
        a, b, c = rng.normal(size=50), rng.normal(size=50), rng.normal(size=50)
        a[:6] = np.nan
        b[10:18] = np.nan
        b2 = b + a * 0.5
        m = pm({"a": a, "b": b2, "c": c})
        ok = ~np.isnan(a) & ~np.isnan(b2)
        assert m.value("a", "b") == approx(np.corrcoef(a[ok], b2[ok])[0, 1])
        assert m.n_pairs[0][1] == int(ok.sum())
        ok2 = ~np.isnan(b2)
        assert m.value("b", "c") == approx(np.corrcoef(b2[ok2], c[ok2])[0, 1]) and m.n_pairs[1][2] == int(ok2.sum())

    def test_infinite_values_are_ignored_pairwise(self):
        m = pm({"a": [1, 2, 3, 4, np.inf], "b": [2, 4, 6, 8, 100]})
        assert m.n_pairs[0][1] == 4 and m.value("a", "b") == approx(1.0)

    def test_matrix_is_symmetric_with_unit_diagonal(self):
        rng = np.random.default_rng(2)
        m = pm({c: rng.normal(size=30) for c in "abcd"})
        v = np.array(m.values, dtype=float)
        assert np.allclose(v, v.T) and np.allclose(np.diag(v), 1.0) and (np.abs(v) <= 1.0 + 1e-12).all()

    def test_identifiers_are_excluded_by_default_and_never_imply_causation(self):
        n = 60
        df = pd.DataFrame({"row_id": range(1, n + 1), "a": np.arange(n) * 0.5, "b": np.sin(np.arange(n))})
        r = analyze_dataframe(df)
        assert r.column_kinds["row_id"] == "identifier"
        assert r.correlation.columns == ["a", "b"]
        assert any("cause" in w.lower() for w in r.warnings)

    def test_boolean_columns_are_not_in_the_numeric_matrix(self):
        df = pd.DataFrame({"a": [1.5, 2, 3, 4], "b": [2.5, 1, 5, 8], "flag": [True, False, True, True]})
        assert analyze_dataframe(df).correlation.columns == ["a", "b"]

    def test_single_numeric_column_has_no_matrix(self):
        assert analyze_dataframe(pd.DataFrame({"a": [1.5, 2, 3, 4], "t": list("abcd")})).correlation is None

    def test_method_is_stated(self):
        r = analyze_dataframe(pd.DataFrame({"a": [1.0, 2, 3, 4, 9], "b": [3.0, 1, 4, 1, 5]}),
                              DescriptiveSettings(correlation_method="spearman"))
        assert r.correlation.method == "spearman"
        assert r.correlation.value("a", "b") == approx(sps.spearmanr([1, 2, 3, 4, 9], [3, 1, 4, 1, 5]).statistic)

    def test_column_limit_is_disclosed(self):
        df = pd.DataFrame({f"c{i}": np.random.default_rng(i).normal(size=30) for i in range(8)})
        r = analyze_dataframe(df, DescriptiveSettings(max_correlation_columns=5))
        assert len(r.correlation.columns) == 5 and len(r.correlation.excluded) == 3
        assert "left out" in r.correlation.notes[0]


class TestCovariance:
    def test_sample_and_population_differ_by_the_documented_factor(self):
        x = np.array([2.0, 4, 4, 4, 5, 5, 7, 9])
        y = np.array([1.0, 3, 2, 8, 5, 4, 9, 7])
        s = pm({"x": x, "y": y}, "covariance_sample")
        p = pm({"x": x, "y": y}, "covariance_population")
        assert s.value("x", "y") == approx(np.cov(x, y, ddof=1)[0, 1]) and p.value("x", "y") == approx(np.cov(x, y, ddof=0)[0, 1])
        assert p.value("x", "y") == approx(s.value("x", "y") * 7 / 8)
        assert s.value("x", "x") == approx(statistics.variance(x)) and p.value("x", "x") == approx(statistics.pvariance(x))
        assert s.method != "" and s.kind == "covariance_sample" and p.kind == "covariance_population"

    def test_covariance_with_a_constant_column_is_zero_because_it_is_defined(self):
        m = pm({"x": [1, 2, 3, 4], "k": [5, 5, 5, 5]}, "covariance_sample")
        assert m.value("x", "k") == 0.0

    def test_pairwise_complete_covariance(self):
        a = [1.0, 2, 3, np.nan, 5]
        b = [2.0, 1, np.nan, 7, 4]
        m = pm({"a": a, "b": b}, "covariance_sample")
        assert m.n_pairs[0][1] == 3 and m.value("a", "b") == approx(np.cov([1, 2, 5], [2, 1, 4], ddof=1)[0, 1])

    def test_insufficient_observations_are_explained(self):
        m = pm({"a": [1.0, np.nan], "b": [2.0, 3.0]}, "covariance_sample")
        assert m.value("a", "b") is None and "at least 2" in m.reasons["a || b"]
        p = pm({"a": [1.0, np.nan], "b": [2.0, 3.0]}, "covariance_population")
        assert p.value("a", "b") == 0.0                      # one pair: population covariance is defined (0)

    def test_report_contains_both_versions(self):
        r = analyze_dataframe(pd.DataFrame({"a": [1.0, 2, 3, 4, 9], "b": [3.0, 1, 4, 1, 5]}))
        assert r.covariance_sample.value("a", "b") == approx(np.cov([1, 2, 3, 4, 9], [3, 1, 4, 1, 5])[0, 1])
        assert r.covariance_population.value("a", "b") == approx(
            np.cov([1, 2, 3, 4, 9], [3, 1, 4, 1, 5], ddof=0)[0, 1])


# ---------------------------------------------------------------------------
# E. Outliers
# ---------------------------------------------------------------------------

class TestOutliers:
    def test_no_outliers(self):
        o = detect_outliers(np.arange(1, 21, dtype=float))
        assert o.available and o.n_outliers == 0 and o.outlier_pct == 0 and o.examples == []

    def test_one_outlier_with_exact_fences(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], dtype=float)
        o = detect_outliers(x)
        assert (o.q1, o.q3, o.iqr) == (3.5, 8.5, 5.0)
        assert (o.lower_bound, o.upper_bound) == (-4.0, 16.0)
        assert o.n_high == 1 and o.n_low == 0 and o.n_outliers == 1
        assert o.outlier_pct == approx(1 / 11 * 100)
        assert o.examples == [{"row": 10, "value": 100.0, "side": "high"}]

    def test_fences_are_strict(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 12.0])             # q1=2, q3=6, upper fence exactly 12
        o = detect_outliers(x)
        assert o.upper_bound == 12.0 and o.n_outliers == 0       # a value ON the fence is not flagged
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 12.001])
        assert detect_outliers(y).n_outliers == 1

    def test_multiplier_changes_the_fences(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], dtype=float)
        assert detect_outliers(x, multiplier=1.5).n_outliers == 1
        o3 = detect_outliers(x, multiplier=3.0)
        assert o3.upper_bound == approx(8.5 + 3 * 5) and o3.n_outliers == 0

    def test_multiple_outliers_low_and_high_with_row_labels_and_ordering(self):
        x = np.array([-50, 10, 11, 12, 13, 14, 15, 16, 17, 18, 90, 200], dtype=float)
        o = detect_outliers(x, row_labels=[f"r{i}" for i in range(len(x))])
        assert o.n_low == 1 and o.n_high == 2 and o.n_outliers == 3
        assert [e["row"] for e in o.examples] == ["r11", "r10", "r0"]      # furthest outside the fences first (200, 90, -50)
        assert (o.lower_bound, o.upper_bound) == (3.5, 25.5)

    def test_negative_values(self):
        o = detect_outliers(np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 30.0]))
        assert o.n_high == 1 and o.examples[0]["value"] == 30.0

    def test_constant_column_has_no_outliers(self):
        o = detect_outliers(np.full(8, 3.0))
        assert o.available and o.n_outliers == 0 and (o.lower_bound, o.upper_bound) == (3.0, 3.0)

    def test_zero_iqr_flags_any_different_value_with_a_warning(self):
        o = detect_outliers(np.array([5, 5, 5, 5, 5, 5, 9.0]))
        assert o.n_outliers == 1 and any("IQR = 0" in w for w in o.warnings)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_small_samples_are_not_assessed(self, n):
        o = detect_outliers(np.arange(n, dtype=float))
        assert not o.available and o.n_outliers == 0 and "at least 4" in o.reason and o.outlier_pct is None

    def test_missing_and_infinite_values_are_ignored_and_percentage_uses_valid_values(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, np.nan, np.nan, np.inf])
        o = detect_outliers(x)
        assert o.valid_count == 11 and o.n_outliers == 1 and o.outlier_pct == approx(100 / 11)

    def test_examples_are_capped_and_flagged_as_truncated(self):
        x = np.concatenate([np.arange(100) * 0.01, np.arange(1000, 1010, dtype=float)])    # tight cluster + 10 far values
        o = detect_outliers(x, max_examples=5)
        assert o.n_outliers == 10 and len(o.examples) == 5 and o.examples_truncated

    def test_original_data_is_never_modified(self):
        df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100], "y": range(11)})
        before = df.copy()
        analyze_dataframe(df)
        pd.testing.assert_frame_equal(df, before)

    def test_report_outliers_use_row_labels_of_the_original_index(self):
        df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]}, index=[f"row-{i}" for i in range(11)])
        o = col(analyze_dataframe(df), "x")
        out = next(x for x in analyze_dataframe(df).outliers if x.column == "x")
        assert out.examples[0]["row"] == "row-10" and o.valid_count == 11


# ---------------------------------------------------------------------------
# Settings, determinism, datasets, whole-report invariants
# ---------------------------------------------------------------------------

class TestSettingsAndReport:
    @pytest.mark.parametrize("bad", [(-1,), (101,), (float("nan"),)])
    def test_invalid_percentiles_are_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_percentiles(bad)

    def test_percentiles_are_deduplicated_and_sorted(self):
        assert validate_percentiles([90, 10, 10, 50]) == (10.0, 50.0, 90.0)

    @pytest.mark.parametrize("kw", [dict(outlier_multiplier=0), dict(outlier_multiplier=-1),
                                    dict(correlation_method="kendall"), dict(top_n=0)])
    def test_invalid_settings_are_rejected(self, kw):
        with pytest.raises(ValueError):
            DescriptiveSettings(**kw).validate()

    def test_not_a_dataframe_is_rejected(self):
        with pytest.raises(TypeError):
            analyze_dataframe([1, 2, 3])

    def test_empty_dataframes_give_a_warning_and_no_results(self):
        for df in (pd.DataFrame(), pd.DataFrame({"a": []}), pd.DataFrame(index=range(3))):
            r = analyze_dataframe(df)
            assert not r.numeric and not r.categorical and r.warnings

    def test_one_row_dataset(self):
        r = analyze_dataframe(pd.DataFrame({"n": [5.0], "t": ["a"], "d": pd.to_datetime(["2024-01-01"])}))
        n = col(r, "n")
        assert n.mean == 5 and n.variance_sample is None and n.skewness is None
        assert col(r, "t", "categorical").modes == ["a"] and r.missing.complete_rows == 1

    def test_duplicate_column_names_are_kept_apart(self):
        df = pd.DataFrame([[1.0, 2.0, "a"], [3.0, 5.0, "b"], [4.0, 9.0, "a"]], columns=["v", "v", "t"])
        r = analyze_dataframe(df)
        assert [n.column for n in r.numeric] == ["v", "v (2)"]
        assert col(r, "v").mean == approx(8 / 3) and col(r, "v (2)").mean == approx(16 / 3)

    def test_non_string_column_names(self):
        r = analyze_dataframe(pd.DataFrame({0: [1.5, 2.5, 4.5], 1: ["a", "b", "a"]}))
        assert r.column_kinds == {"0": "numeric", "1": "categorical"}

    def test_columns_are_independent_of_order(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame({"a": rng.normal(size=40), "b": rng.choice(list("xyz"), 40), "c": rng.exponential(2, 40)})
        r1 = analyze_dataframe(df)
        r2 = analyze_dataframe(df[["c", "b", "a"]])
        assert col(r1, "a").to_dict() == col(r2, "a").to_dict()
        assert col(r1, "b", "categorical").to_dict() == col(r2, "b", "categorical").to_dict()
        assert r1.correlation.value("a", "c") == approx(r2.correlation.value("a", "c"))

    def test_report_is_serializable_and_has_no_nan(self):
        import json
        df = pd.DataFrame({"a": [1.0, 2, np.nan, 4, 5], "k": [3.0] * 5, "t": list("ababc"),
                           "d": pd.to_datetime(["2024-01-01", "2024-01-02", None, "2024-01-09", "2024-02-01"])})
        text = json.dumps(analyze_dataframe(df).to_dict(), allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text

    def test_data_quality_overview(self):
        df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "k": [9, 9, 9, 9], "e": [np.nan] * 4, "t": list("aabc")})
        df.loc[len(df)] = df.loc[0]       # row 4 repeats row 0; row 1 also repeats row 0 (a=1.0, k=9, e=NaN, t="a")
        r = analyze_dataframe(df)
        q = r.quality
        assert q["duplicate_rows"] == 2 and "k" in q["constant_columns"] and "e" in q["empty_columns"]
        assert r.missing.rows_with_missing == 5 and r.missing.complete_rows == 0       # every row misses column e
        assert r.missing.missing_cells == 5 and r.missing.total_cells == 20
        assert r.missing.missing_cell_pct == approx(25.0)

    def test_missing_summary_is_sorted_worst_first_with_stable_ties(self):
        df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [np.nan, np.nan, 3, 4], "c": [np.nan, 2, 3, 4], "d": [np.nan, 1, 1, 2]})
        assert [r["column"] for r in analyze_dataframe(df).missing.rows] == ["b", "c", "d", "a"]

    def test_counts_reconcile_for_every_numeric_column(self):
        rng = np.random.default_rng(4)
        v = rng.normal(size=100)
        v[rng.choice(100, 9, replace=False)] = np.nan
        v[rng.choice(100, 3, replace=False)] = np.inf
        s = col(one_col(v), "x")
        assert s.valid_count + s.missing_count + s.infinite_count == s.sample_size == 100

    def test_analysis_is_deterministic(self):
        df = load_fixture("i_missing_values.csv")
        assert analyze_dataframe(df).to_dict() == analyze_dataframe(df).to_dict()


class TestDatasetAgnostic:
    ALL = [(n, "diverse_schemas") for n, _ in all_diverse_fixtures()] + \
          [(p.name, "generic_schemas") for p in sorted(GENERIC_FIXTURES_DIR.glob("*.csv"))]

    def test_there_are_sixteen_fixture_datasets(self):
        assert len(self.ALL) == 16

    @pytest.mark.parametrize("name,sub", ALL, ids=[n for n, _ in ALL])
    def test_every_fixture_dataset_is_analyzed_correctly(self, name, sub):
        df = load_fixture(name, sub)
        r = analyze_dataframe(df)
        assert r.n_rows == len(df) and r.n_cols == df.shape[1]
        assert set(r.column_kinds) == {str(c) for c in df.columns}
        for n in r.numeric:                                         # independent check against the file
            v = pd.to_numeric(df[n.column], errors="coerce")
            fin = v[np.isfinite(v)]
            assert n.valid_count == len(fin) and n.missing_count == int(df[n.column].isna().sum())
            assert n.mean == approx(statistics.fmean(fin)) and n.median == approx(statistics.median(fin))
            if len(fin) >= 2:
                assert n.std_sample == approx(statistics.stdev(fin))
        for c in r.categorical:
            assert c.valid_count == int(df[c.column].replace(r"^\s*$", np.nan, regex=True).notna().sum())
            assert sum(t["count"] for t in c.frequency) <= c.valid_count
        assert r.missing.missing_cells == sum(x["missing_count"] for x in r.missing.rows)
        kinds = list(r.column_kinds.values())
        assert len(kinds) == len(df.columns)

    def test_totally_different_schemas_use_no_fixed_column_names(self):
        a = analyze_dataframe(pd.DataFrame({"patient_age": [34.0, 51, 29, 62, 45], "ward": list("AABBA")}))
        b = analyze_dataframe(pd.DataFrame({"torque_nm": [10.5, 12.1, 9.8, 11.0, 10.2], "line": list("XYXYX")}))
        assert col(a, "patient_age").mean == approx(44.2) and col(b, "torque_nm").mean == approx(10.72)
        assert col(a, "ward", "categorical").modes == ["A"] and col(b, "line", "categorical").modes == ["X"]
