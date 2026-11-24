"""Phase 8 / Part 1 - Availability Correction 1 tests.

Behavioral tests for ``src/common/availability.py``: the descriptive gate no
longer depends on variation, the prescriptive gate needs two *varying*
numeric columns, unknown module keys are handled before dataset state, and a
wrong summary type raises an informative TypeError. Every test asserts the
actual verdict and message content, not merely that a value came back.
"""

import ast
import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers import all_diverse_fixtures, GENERIC_FIXTURES_DIR
from src.common import availability as availability_module
from src.common.availability import Availability, KNOWN_MODULES, check_availability
from src.common.dataset_summary import DatasetSummary, summarize_dataset

RNG = np.random.default_rng(20261117)


def S(df):
    return summarize_dataset(df)


def varying(n, k=2):
    """n rows, k independent varying float columns."""
    return pd.DataFrame({f"v{i}": RNG.normal(size=n) for i in range(k)})


# ---------------------------------------------------------------------------
# Structure / architecture preserved
# ---------------------------------------------------------------------------

class TestStructurePreserved:
    def test_availability_fields_unchanged_and_typed(self):
        assert [f for f in Availability.__dataclass_fields__] == ["available", "reasons", "missing"]
        hints = typing.get_type_hints(Availability)
        assert hints["available"] is bool
        assert hints["reasons"] == list[str] and hints["missing"] == list[str]

    def test_module_keys_unchanged(self):
        assert KNOWN_MODULES == ("descriptive", "inferential", "predictive", "prescriptive")

    def test_defaults_are_independent_lists(self):
        a, b = Availability(True), Availability(True)
        a.reasons.append("x")
        assert b.reasons == []

    def test_module_does_not_import_streamlit(self):
        tree = ast.parse((ROOT / "src/common/availability.py").read_text(encoding="utf-8"))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                imported.add((n.module or "").split(".")[0])
        assert "streamlit" not in imported

    def test_wide_dataset_is_evaluated_correctly(self):
        wide = pd.DataFrame({f"c{i}": RNG.normal(size=15) for i in range(1500)})
        wide["const"] = 4.0
        a = check_availability("predictive", S(wide))
        assert a.available and a.reasons == [] and a.missing == []


# ---------------------------------------------------------------------------
# Descriptive: depends on non-empty columns, not on variation
# ---------------------------------------------------------------------------

class TestDescriptiveGate:
    def test_constant_numeric_columns_only_is_available(self):
        df = pd.DataFrame({"x": [5.5] * 30, "y": [2.0] * 30})
        s = S(df)
        assert s.constant_columns == ["x", "y"]            # precondition: really constant
        a = check_availability("descriptive", s)
        assert a.available is True and a.reasons == [] and a.missing == []

    def test_constant_categorical_column_only_is_available(self):
        s = S(pd.DataFrame({"dept": ["CSE"] * 30}))
        assert s.constant_columns == ["dept"] and s.categorical_columns == ["dept"]
        assert check_availability("descriptive", s).available is True

    def test_mixed_constant_and_varying_is_available(self):
        df = pd.DataFrame({"k": [3.0] * 30, "v": RNG.normal(size=30)})
        s = S(df)
        assert s.constant_columns == ["k"]
        assert check_availability("descriptive", s).available is True

    def test_user_example_with_constant_marks_and_department_only(self):
        df = pd.DataFrame({"Marks": [80] * 4, "Department": ["CSE"] * 4})
        assert check_availability("descriptive", S(df)).available is True

    def test_one_valid_observation_is_available(self):
        s = S(pd.DataFrame({"a": [1.5], "b": ["x"], "c": [7]}))
        assert s.n_rows == 1 and set(s.constant_columns) == {"a", "b", "c"}
        assert check_availability("descriptive", s).available is True

    def test_many_observations_all_equal_with_some_missing_is_available(self):
        s = S(pd.DataFrame({"a": [80, np.nan, 80, 80]}))
        assert s.constant_columns == ["a"] and s.missing_cells == 1
        assert check_availability("descriptive", s).available is True

    def test_one_non_empty_column_among_empty_ones_is_available(self):
        df = pd.DataFrame({"e1": [np.nan] * 6, "e2": [None] * 6, "ok": [4.0] * 6})
        s = S(df)
        assert set(s.empty_columns) == {"e1", "e2"}
        assert check_availability("descriptive", s).available is True

    def test_entirely_empty_columns_only_is_unavailable_with_explanation(self):
        s = S(pd.DataFrame({"a": [np.nan] * 5, "b": [None] * 5}))
        assert set(s.empty_columns) == {"a", "b"}
        a = check_availability("descriptive", s)
        assert a.available is False
        assert len(a.reasons) == 1 and "entirely empty" in a.reasons[0]
        assert len(a.missing) == 1 and "contains values" in a.missing[0]

    @pytest.mark.parametrize("df", [pd.DataFrame({"a": []}), pd.DataFrame(columns=["a", "b"])],
                             ids=["zero-rows-1col", "zero-rows-2cols"])
    def test_zero_rows_is_unavailable(self, df):
        a = check_availability("descriptive", S(df))
        assert not a.available and a.reasons == ["The active dataset has no rows or no columns."]
        assert a.missing == ["A dataset with at least one row and one column."]

    def test_zero_columns_is_unavailable(self):
        a = check_availability("descriptive", S(pd.DataFrame(index=range(5))))
        assert not a.available and "no rows or no columns" in a.reasons[0]

    def test_none_summary_is_unavailable_not_an_error(self):
        a = check_availability("descriptive", None)
        assert not a.available and "no rows or no columns" in a.reasons[0]


# ---------------------------------------------------------------------------
# Inferential
# ---------------------------------------------------------------------------

class TestInferentialGate:
    def test_exactly_three_rows_is_available(self):
        s = S(pd.DataFrame({"a": [1.0, 2.5, 4.0]}))
        assert s.n_rows == 3 and check_availability("inferential", s).available is True

    def test_two_rows_is_blocked_and_names_the_numbers(self):
        a = check_availability("inferential", S(pd.DataFrame({"a": [1.0, 2.5]})))
        assert not a.available and len(a.reasons) == 1
        assert "2 row(s)" in a.reasons[0] and "at least 3" in a.reasons[0]

    def test_boolean_only_dataset_is_available(self):
        df = pd.DataFrame({"flag": RNG.integers(0, 2, 40).astype(bool),
                           "paid": RNG.choice(["yes", "no"], 40)})
        s = S(df)
        assert s.boolean_columns and not s.numeric_columns   # precondition: no numeric column
        a = check_availability("inferential", s)
        assert a.available is True and a.reasons == []

    def test_dataset_with_no_testable_column_names_all_accepted_types(self):
        df = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=20)})
        s = S(df)
        assert s.datetime_columns == ["when"] and not (
            s.numeric_columns or s.categorical_columns or s.boolean_columns)
        a = check_availability("inferential", s)
        assert not a.available
        assert a.reasons == ["No numeric, categorical or yes/no column was detected."]
        assert a.missing == ["At least one numeric, categorical or yes/no column."]

    def test_too_few_rows_and_no_testable_column_report_both_in_order(self):
        df = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=2)})
        a = check_availability("inferential", S(df))
        assert len(a.reasons) == 2 and len(a.missing) == 2
        assert "row(s)" in a.reasons[0] and "yes/no" in a.reasons[1]


# ---------------------------------------------------------------------------
# Predictive
# ---------------------------------------------------------------------------

class TestPredictiveGate:
    def test_exactly_ten_rows_is_available(self):
        s = S(varying(10))
        assert s.n_rows == 10 and check_availability("predictive", s).available is True

    def test_nine_rows_is_blocked(self):
        a = check_availability("predictive", S(varying(9)))
        assert not a.available and len(a.reasons) == 1
        assert "9 row(s)" in a.reasons[0] and "At least 10 rows" in a.missing[0]

    def test_fewer_than_two_usable_columns_is_blocked(self):
        df = varying(20, k=1)
        df["const"] = 3.0
        df["empty"] = np.nan
        s = S(df)
        assert s.constant_columns == ["const"] and s.empty_columns == ["empty"]
        a = check_availability("predictive", s)
        assert not a.available and len(a.reasons) == 1
        assert "target" in a.reasons[0] and "two columns" in a.missing[0]

    def test_two_usable_columns_among_constants_is_available(self):
        df = varying(20, k=2)
        df["const"] = 3.0
        assert check_availability("predictive", S(df)).available is True

    def test_few_rows_and_few_usable_columns_report_both_in_order(self):
        df = varying(5, k=1)
        a = check_availability("predictive", S(df))
        assert not a.available
        assert len(a.reasons) == 2 and len(a.missing) == 2
        assert "5 row(s)" in a.reasons[0] and "target" in a.reasons[1]
        assert "10 rows" in a.missing[0] and "two columns" in a.missing[1]


# ---------------------------------------------------------------------------
# Prescriptive (D1: two VARYING numeric columns, 10 rows)
# ---------------------------------------------------------------------------

class TestPrescriptiveGate:
    def test_exactly_ten_rows_with_two_varying_numeric_columns_is_available(self):
        s = S(varying(10, 2))
        assert s.n_rows == 10 and len(s.numeric_columns) == 2
        a = check_availability("prescriptive", s)
        assert a.available is True and a.reasons == [] and a.missing == []

    def test_nine_rows_is_blocked(self):
        a = check_availability("prescriptive", S(varying(9, 2)))
        assert not a.available and len(a.reasons) == 1 and "9 row(s)" in a.reasons[0]

    def test_single_numeric_column_is_blocked(self):
        df = varying(30, 1)
        df["group"] = ["a", "b"] * 15
        a = check_availability("prescriptive", S(df))
        assert not a.available and len(a.reasons) == 1 and "numeric" in a.reasons[0]

    def test_constant_numeric_column_cannot_satisfy_the_gate(self):
        df = varying(30, 1)
        df["const"] = 12.5
        s = S(df)
        assert len(s.numeric_columns) == 2 and "const" in s.constant_columns   # 2 numeric, 1 varying
        a = check_availability("prescriptive", s)
        assert not a.available
        assert "vary" in a.reasons[0] and "constant" in a.reasons[0]

    def test_only_constant_numeric_columns_is_blocked(self):
        df = pd.DataFrame({"x": [1.5] * 30, "y": [2.5] * 30, "z": [3.5] * 30})
        assert not check_availability("prescriptive", S(df)).available

    def test_three_numeric_columns_with_two_varying_is_available(self):
        df = varying(30, 2)
        df["const"] = 9.0
        assert check_availability("prescriptive", S(df)).available is True

    def test_few_rows_and_too_few_numeric_report_both_in_order(self):
        a = check_availability("prescriptive", S(varying(5, 1)))
        assert not a.available
        assert len(a.reasons) == 2 and len(a.missing) == 2
        assert "5 row(s)" in a.reasons[0] and "numeric" in a.reasons[1]
        assert a.missing[0] == "At least 10 rows."


# ---------------------------------------------------------------------------
# Unknown module keys (D3): checked before dataset state
# ---------------------------------------------------------------------------

BAD_KEYS = ["zzz", "", "Descriptive", " descriptive", None, 5, ["descriptive"]]


class TestUnknownModuleKeys:
    @pytest.mark.parametrize("key", BAD_KEYS, ids=repr)
    @pytest.mark.parametrize("summary", [None, summarize_dataset(None), summarize_dataset(varying(30, 2))],
                             ids=["none", "empty", "valid"])
    def test_unknown_key_gives_the_same_verdict_for_every_dataset_state(self, key, summary):
        a = check_availability(key, summary)
        assert a.available is False
        assert a.reasons == [f"Unknown module '{key}'."] and a.missing == []

    @pytest.mark.parametrize("bad_summary", [pd.DataFrame({"a": [1]}), "text", 7])
    def test_unknown_key_wins_even_over_a_wrongly_typed_summary(self, bad_summary):
        a = check_availability("zzz", bad_summary)
        assert not a.available and a.reasons[0].startswith("Unknown module")


# ---------------------------------------------------------------------------
# Incorrect summary types
# ---------------------------------------------------------------------------

class TestInvalidSummaryType:
    @pytest.mark.parametrize("key", KNOWN_MODULES)
    @pytest.mark.parametrize("bad,typename", [
        (pd.DataFrame({"a": [1, 2, 3]}), "DataFrame"), ("text", "str"), (5, "int"),
        ({"n_rows": 3}, "dict"), ([1, 2], "list")])
    def test_wrong_type_raises_informative_type_error(self, key, bad, typename):
        with pytest.raises(TypeError) as exc:
            check_availability(key, bad)
        msg = str(exc.value)
        assert "DatasetSummary" in msg and typename in msg

    def test_none_is_not_an_error_for_any_module(self):
        for key in KNOWN_MODULES:
            a = check_availability(key, None)
            assert a.available is False and len(a.reasons) == 1 and len(a.missing) == 1


# ---------------------------------------------------------------------------
# All 16 existing fixture datasets remain compatible
# ---------------------------------------------------------------------------

def _fixture_frames():
    frames = [(f"diverse/{n}", pd.read_csv(p)) for n, p in all_diverse_fixtures()]
    frames += [(f"generic/{p.name}", pd.read_csv(p)) for p in sorted(GENERIC_FIXTURES_DIR.glob("*.csv"))]
    return frames


class TestAllFixtureDatasets:
    def test_there_are_sixteen_fixtures(self):
        assert len(_fixture_frames()) == 16

    @pytest.mark.parametrize("name,df", _fixture_frames(), ids=lambda v: v if isinstance(v, str) else "")
    def test_verdicts_are_consistent_and_match_independent_rules(self, name, df):
        s = S(df)
        const = set(s.constant_columns)
        for key in KNOWN_MODULES:
            a = check_availability(key, s)
            assert isinstance(a, Availability)
            assert a.available == (a.reasons == [])
            assert len(a.missing) == len(a.reasons)           # every reason says what is needed
        # independent re-statement of each rule from the summary's own lists
        assert check_availability("descriptive", s).available == any(
            c["column"] not in set(s.empty_columns) and c["missing_pct"] < 100 for c in s.columns)
        assert check_availability("inferential", s).available == (
            s.n_rows >= 3 and bool(s.numeric_columns or s.categorical_columns or s.boolean_columns))
        assert check_availability("predictive", s).available == (
            s.n_rows >= 10 and len([c["column"] for c in s.columns
                                    if c["column"] not in const | set(s.empty_columns)]) >= 2)
        assert check_availability("prescriptive", s).available == (
            s.n_rows >= 10 and len([c for c in s.numeric_columns if c not in const]) >= 2)
