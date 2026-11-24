"""Ultimate release — data-quality engine and cleaning workflow."""

from __future__ import annotations

import io
import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.data_quality import (SCORE_WEIGHTS, apply_cleaning, assess_quality,  # noqa: E402
                                         quality_summary, recommend_plan)


def messy():
    return pd.DataFrame({
        "Order_Date": ["2024-01-01", "2024-01-08", "not a date", "2024-01-22", "2024-01-29", None,
                       "2024-02-12", "2024-02-19"],
        "Net_Sales": [100.0, np.nan, 120.0, 130.0, np.inf, 150.0, 160.0, 170.0],
        "Meta_Ads": ["1,000", "1,200", " ", "1,400", "N/A", "1,500", "abc", "1,700"],
        "Region": ["N", "S", None, "N", "  ", "S", "N", "N"],
        "Units": [5, 6, 7, np.nan, 9, 10, np.nan, 12],
    })


class TestDetection:
    def test_counts_every_kind_of_missing(self):
        r = assess_quality(messy())
        c = r["columns"]
        assert c["Meta_Ads"]["n_missing"] == 2 and c["Meta_Ads"]["n_blank_strings"] == 2   # ' ' and 'N/A'
        assert c["Region"]["n_missing"] == 2                                               # None and '  '
        assert c["Units"]["n_missing"] == 2
        assert c["Order_Date"]["n_missing"] == 1
        assert r["blank_string_cells"] == 3          # " ", "N/A", "  " (None is a real null)

    def test_invalid_values(self):
        c = assess_quality(messy())["columns"]
        assert c["Order_Date"]["kind"] == "date" and c["Order_Date"]["n_invalid"] == 1
        assert c["Net_Sales"]["n_inf"] == 1
        assert c["Meta_Ads"]["kind"] == "numeric_text" and c["Meta_Ads"]["n_invalid"] == 1   # 'abc'

    def test_missing_by_row_and_duplicates(self):
        df = pd.concat([messy(), messy().iloc[[0]]], ignore_index=True)
        r = assess_quality(df)
        assert r["duplicate_rows"] == 1
        assert sum(r["missing_per_row_histogram"].values()) == len(df)
        assert r["rows_with_missing"] == int((r["missing_per_row_histogram"]["0"] != len(df))) * \
            (len(df) - r["missing_per_row_histogram"]["0"])

    def test_duplicate_column_names_detected(self):
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"])
        r = assess_quality(df)
        assert r["duplicate_column_names"] == ["x"]
        assert r["dimensions"]["consistency"] < 100

    def test_mixed_type_column_flagged(self):
        df = pd.DataFrame({"v": ["1", "2", "three", "four", "5", "six", "x", "y"]})
        assert assess_quality(df)["columns"]["v"]["mixed_types"]

    def test_all_missing_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        r = assess_quality(df)
        assert r["columns"]["b"]["kind"] == "empty"
        assert r["missing_cells"] == 3


class TestScore:
    def test_clean_data_scores_100(self):
        df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=5).astype(str), "x": [1.0, 2, 3, 4, 5]})
        r = assess_quality(df)
        assert r["score"] == 100.0
        assert set(r["dimensions"]) == set(SCORE_WEIGHTS)

    def test_messy_data_scores_below_100_and_is_explained(self):
        r = assess_quality(messy())
        assert r["score"] < 95
        assert "completeness 30%" in r["score_method"]

    def test_imputed_cells_are_not_counted_as_complete(self):
        df = messy()
        plan = recommend_plan(df, date_column="Order_Date")
        clean, audit = apply_cleaning(df, plan)
        qs = quality_summary(df, clean, audit)
        assert audit["total_imputed_cells"] > 0
        assert qs["cleaned"]["imputed_cells"] == audit["total_imputed_cells"]
        assert qs["cleaned"]["dimensions"]["completeness"] < 100.0
        assert qs["cleaned"]["filled_pct"] > qs["cleaned"]["dimensions"]["completeness"]


class TestRecommendations:
    def test_role_and_type_rules(self):
        df = messy()
        plan = recommend_plan(df, date_column="Order_Date",
                              roles={"Net_Sales": "revenue", "Meta_Ads": "channel", "Order_Date": "date"})["columns"]
        assert plan["Order_Date"]["strategy"] == "leave"            # dates never invented
        assert plan["Net_Sales"]["strategy"] == "leave"             # revenue never invented
        assert plan["Meta_Ads"]["strategy"] == "zero" and plan["Meta_Ads"]["convert_numeric"]
        assert plan["Units"]["strategy"] == "past_median"           # time-ordered: past only
        assert plan["Region"]["strategy"] in ("mode", "unknown")
        assert all(p["reason"] for p in plan.values())

    def test_heavily_missing_numeric_left_alone(self):
        df = pd.DataFrame({"x": [1.0] + [np.nan] * 9, "y": range(10)})
        assert recommend_plan(df)["columns"]["x"]["strategy"] == "leave"

    def test_skewed_numeric_uses_median(self):
        vals = [1, 1, 1, 2, 2, 3, 1000, 5000, np.nan, 2, 1, 1]
        assert recommend_plan(pd.DataFrame({"x": vals}))["columns"]["x"]["strategy"] == "median"

    def test_duplicates_kept_by_default(self):
        df = pd.concat([messy(), messy().iloc[[0]]], ignore_index=True)
        plan = recommend_plan(df)
        assert plan["drop_duplicate_rows"] is False and plan["duplicate_rows_found"] == 1


class TestApply:
    def test_original_never_modified(self):
        df = messy()
        before = df.copy()
        apply_cleaning(df, recommend_plan(df, date_column="Order_Date"))
        pd.testing.assert_frame_equal(df, before)

    def test_audit_trail_complete(self):
        df = messy()
        _, audit = apply_cleaning(df, recommend_plan(df, date_column="Order_Date"))
        for key in ("processed_at", "original_rows", "original_cols", "final_rows", "imputed_cells",
                    "rows_removed", "strategies", "transformed_columns", "blank_strings_normalised",
                    "infinite_values_to_missing", "plan"):
            assert key in audit
        assert audit["original_rows"] == 8 and audit["blank_strings_normalised"] == 3
        assert audit["infinite_values_to_missing"] == 1

    def test_past_median_uses_only_earlier_rows(self):
        df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=6).astype(str),
                           "x": [10.0, 20.0, np.nan, 1000.0, np.nan, 5.0]})
        plan = {"columns": {"x": {"strategy": "past_median"}}, "date_column": "d"}
        clean, _ = apply_cleaning(df, plan)
        assert clean["x"].iloc[2] == 15.0                 # median(10, 20) — 1000 is in the future
        assert clean["x"].iloc[4] == 20.0                 # median(10, 20, 1000)

    def test_past_median_respects_date_order_not_row_order(self):
        df = pd.DataFrame({"d": ["2024-01-03", "2024-01-01", "2024-01-02"], "x": [np.nan, 4.0, 8.0]})
        clean, _ = apply_cleaning(df, {"columns": {"x": {"strategy": "past_median"}}, "date_column": "d"})
        assert clean["x"].iloc[0] == 6.0

    def test_ffill_never_backfills_leading_gap(self):
        df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=4).astype(str), "x": [np.nan, 1.0, np.nan, 3.0]})
        clean, _ = apply_cleaning(df, {"columns": {"x": {"strategy": "ffill"}}, "date_column": "d"})
        assert np.isnan(clean["x"].iloc[0]) and clean["x"].iloc[2] == 1.0

    def test_interpolate_only_between_known_values(self):
        df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=5).astype(str),
                           "x": [np.nan, 2.0, np.nan, 4.0, np.nan]})
        clean, _ = apply_cleaning(df, {"columns": {"x": {"strategy": "interpolate"}}, "date_column": "d"})
        assert clean["x"].iloc[2] == 3.0
        assert np.isnan(clean["x"].iloc[0]) and np.isnan(clean["x"].iloc[4])

    def test_drop_rows_and_dedupe_recorded(self):
        df = pd.concat([messy(), messy().iloc[[0]]], ignore_index=True)
        plan = recommend_plan(df)
        plan["drop_duplicate_rows"] = True
        plan["columns"]["Units"]["strategy"] = "drop_rows"
        clean, audit = apply_cleaning(df, plan)
        assert audit["rows_removed"]["duplicates"] == 1
        assert audit["rows_removed"]["missing_required"] == 2
        assert len(clean) == len(df) - 3 == audit["final_rows"]

    def test_unknown_category_keeps_missingness_visible(self):
        df = pd.DataFrame({"c": ["a", None, "b", "  "]})
        clean, audit = apply_cleaning(df, {"columns": {"c": {"strategy": "unknown"}}})
        assert clean["c"].tolist() == ["a", "Unknown", "b", "Unknown"]
        assert audit["imputed_cells"]["c"] == 2

    def test_numbers_as_text_converted_and_invalid_counted(self):
        df = pd.DataFrame({"v": ["1,000", "₹2,500", "3%", "oops"]})
        clean, audit = apply_cleaning(df, {"columns": {"v": {"strategy": "none", "convert_numeric": True}}})
        assert clean["v"].tolist()[:3] == [1000.0, 2500.0, 3.0] and np.isnan(clean["v"].iloc[3])
        assert audit["invalid_numbers_to_missing"] == 1


class TestPersistenceAndUpload:
    def test_quality_report_persisted_per_dataset(self):
        from config.settings import dataset_processed_dir, dataset_upload_dir
        from src.data_engine.analysis_runner import run_full_analysis
        df = messy().rename(columns={"Order_Date": "Date"})
        ds = f"test_dq_{uuid.uuid4().hex[:10]}"
        try:
            run_full_analysis(ds, df=df)
            rep = json.loads((dataset_processed_dir(ds) / "data_quality_report.json").read_text(encoding="utf-8"))
            assert rep["original"]["n_rows"] == 8 and rep["cleaned"] is None
        finally:
            shutil.rmtree(dataset_upload_dir(ds), ignore_errors=True)

    @pytest.mark.parametrize("name,data,needle", [
        ("e.csv", b"", "empty"),
        ("e.csv", b"   \n  ", "empty"),
        ("bad.xlsx", b"not an excel file", "Excel"),
        ("x.xls", b"abc", ".xlsx"),
    ])
    def test_friendly_read_errors(self, name, data, needle):
        import importlib
        page = importlib.import_module("dashboard.pages.0_upload_data")

        class F:
            def __init__(self, n, d):
                self.name, self._d = n, d

            def getvalue(self):
                return self._d
        with pytest.raises(page.UploadReadError, match=needle):
            page._load_dataframe(F(name, data))

    def test_semicolon_and_latin1_csv_read(self):
        import importlib
        page = importlib.import_module("dashboard.pages.0_upload_data")

        class F:
            name = "s.csv"

            def getvalue(self):
                return "Date;Café;Revenue\n2024-01-01;a;10\n2024-01-02;b;20\n".encode("latin-1")
        df = page._load_dataframe(F())
        assert list(df.columns) == ["Date", "Café", "Revenue"] and len(df) == 2


def test_fast_past_median_matches_reference_definition():
    from src.data_engine.data_quality import _past_median_fill, _past_median_fill_reference
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(1, 50))
        v = pd.Series(rng.normal(size=n))
        v[rng.random(n) < 0.4] = np.nan
        np.testing.assert_allclose(_past_median_fill(v).to_numpy(), _past_median_fill_reference(v).to_numpy())
