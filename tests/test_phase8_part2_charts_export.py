"""Phase 8 Part 2 - chart builders and CSV / Excel export tests."""

import io
import re
import statistics
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import export as common_export  # noqa: E402
from src.descriptive import charts as ch  # noqa: E402
from src.descriptive import export as dx  # noqa: E402
from src.descriptive.contracts import DescriptiveSettings  # noqa: E402
from src.descriptive.engine import analyze_dataframe  # noqa: E402
from tests.helpers import load_fixture  # noqa: E402

approx = pytest.approx

@pytest.fixture(autouse=True)
def _real_excel_row_limit(monkeypatch):
    """An older test (test_phase8_part1_foundation.py::test_oversized_excel_...) assigns
    ``export.EXCEL_MAX_ROWS = 1`` without restoring it, which leaks into later tests in the
    same process. Pin the real limit here so these tests do not depend on test order."""
    monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 1_048_575)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame({
        "price": rng.normal(50, 10, n).round(2),
        "qty": rng.integers(1, 30, n).astype(float),
        "weight": rng.gamma(2, 3, n),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "tier": rng.choice(["a", "b"], n),
    })


def trace_values(fig, attr="y"):
    return np.concatenate([np.asarray(getattr(t, attr), dtype=float) for t in fig.data])


# ---------------------------------------------------------------------------
# F. Charts
# ---------------------------------------------------------------------------

class TestEveryChartType:
    def test_chart_type_list_matches_the_documented_set(self):
        assert set(ch.CHART_TYPES) == {"Histogram", "Box plot", "Violin plot", "Density plot", "Correlation heatmap",
                                       "Scatter plot", "Bar chart", "Frequency distribution"}

    def test_histogram_counts_add_up_and_bins_are_respected(self, df):
        r = ch.histogram(df, "price", bins=12)
        assert r.available and len(r.figure.data[0].x) == 12
        counts = np.asarray(r.figure.data[0].y)
        expected, _ = np.histogram(df["price"], bins=12)
        assert counts.sum() == len(df) and (counts == expected).all()
        assert r.figure.layout.title.text == "Histogram of price" and r.figure.layout.xaxis.title.text == "price"
        assert not r.sampled

    @pytest.mark.parametrize("fn,cls", [(ch.box_plot, "box"), (ch.violin_plot, "violin")])
    def test_box_and_violin_plain_and_grouped(self, df, fn, cls):
        plain = fn(df, "price")
        assert plain.available and plain.figure.data[0].type == cls
        assert np.allclose(sorted(plain.figure.data[0].y), sorted(df["price"]))
        grouped = fn(df, "price", "region")
        assert grouped.available and sorted(t.name for t in grouped.figure.data) == ["East", "North", "South", "West"]
        for t in grouped.figure.data:
            assert len(t.y) == int((df["region"] == t.name).sum())

    def test_density_plot_integrates_to_about_one(self, df):
        r = ch.density_plot(df, "weight")
        x, y = np.asarray(r.figure.data[0].x), np.asarray(r.figure.data[0].y)
        assert r.available and np.trapezoid(y, x) == approx(1.0, abs=0.05) and (y >= 0).all()

    def test_density_plot_by_group(self, df):
        r = ch.density_plot(df, "price", "tier")
        assert r.available and sorted(t.name for t in r.figure.data) == ["a", "b"]

    def test_correlation_heatmap_shows_the_matrix_values(self, df):
        rep = analyze_dataframe(df)
        r = ch.correlation_heatmap(rep.correlation)
        assert r.available
        z = np.asarray(r.figure.data[0].z, dtype=float)
        assert z.shape == (3, 3) and np.allclose(np.diag(z), 1.0)
        assert z[0, 1] == approx(np.corrcoef(df["price"], df["qty"])[0, 1])
        assert list(r.figure.data[0].x) == ["price", "qty", "weight"]

    def test_scatter_plot_points_match_the_data(self, df):
        r = ch.scatter_plot(df, "price", "qty")
        assert r.available and r.n_total == len(df) and not r.sampled
        assert np.allclose(r.figure.data[0].x, df["price"]) and np.allclose(r.figure.data[0].y, df["qty"])
        coloured = ch.scatter_plot(df, "price", "qty", color="tier")
        assert sorted(t.name for t in coloured.figure.data) == ["a", "b"]

    def test_bar_chart_aggregates_match_pandas(self, df):
        for agg in ("mean", "median", "sum"):
            r = ch.bar_chart(df, "region", "price", agg)
            expected = df.groupby("region")["price"].agg(agg)
            got = dict(zip(r.figure.data[0].x, r.figure.data[0].y))
            assert r.available and got == approx(expected.to_dict())
            assert list(r.figure.data[0].y) == sorted(r.figure.data[0].y, reverse=True)

    def test_bar_chart_count_equals_frequency_chart(self, df):
        r = ch.bar_chart(df, "region", None, "count")
        f = ch.frequency_chart(df, "region")
        assert list(r.figure.data[0].x) == list(f.figure.data[0].x) and list(r.figure.data[0].y) == list(f.figure.data[0].y)
        assert dict(zip(f.figure.data[0].x, f.figure.data[0].y)) == df["region"].value_counts().to_dict()

    def test_frequency_chart_titles_and_hover(self, df):
        r = ch.frequency_chart(df, "tier")
        assert r.figure.layout.title.text == "Frequency of tier" and "Count" in r.figure.data[0].hovertemplate


class TestUnavailableCharts:
    CONST = pd.DataFrame({"k": [5.0] * 10, "v": np.arange(10.0), "c": ["x"] * 10})

    @pytest.mark.parametrize("call", [
        lambda d: ch.histogram(d, "k"), lambda d: ch.box_plot(d, "k"), lambda d: ch.violin_plot(d, "k"),
        lambda d: ch.density_plot(d, "k"), lambda d: ch.scatter_plot(d, "k", "v"), lambda d: ch.scatter_plot(d, "v", "k"),
    ])
    def test_constant_columns_are_refused_with_an_explanation(self, call):
        r = call(self.CONST)
        assert not r.available and r.figure is None and "vary" in r.message

    @pytest.mark.parametrize("call", [
        lambda d: ch.histogram(d, "v"), lambda d: ch.box_plot(d, "v"), lambda d: ch.violin_plot(d, "v"),
        lambda d: ch.density_plot(d, "v"), lambda d: ch.scatter_plot(d, "v", "w"), lambda d: ch.frequency_chart(d, "v"),
        lambda d: ch.bar_chart(d, "v", "w", "mean"),
    ])
    def test_empty_dataframe(self, call):
        r = call(pd.DataFrame({"v": [], "w": []}))
        assert not r.available and r.message

    @pytest.mark.parametrize("call", [
        lambda d: ch.histogram(d, "nope"), lambda d: ch.box_plot(d, "nope"), lambda d: ch.violin_plot(d, "v", "nope"),
        lambda d: ch.density_plot(d, "nope"), lambda d: ch.scatter_plot(d, "v", "nope"), lambda d: ch.frequency_chart(d, "nope"),
        lambda d: ch.bar_chart(d, "nope", "v", "mean"), lambda d: ch.histogram(d, None),
    ])
    def test_invalid_column_selection(self, call):
        r = call(pd.DataFrame({"v": np.arange(10.0), "w": np.arange(10.0) ** 2}))
        assert not r.available and "not in the active dataset" in r.message

    @pytest.mark.parametrize("bins", [0, 1, 201, -5, 2.5, "10", None, True])
    def test_invalid_bin_counts(self, bins):
        r = ch.histogram(pd.DataFrame({"v": np.arange(10.0)}), "v", bins)
        assert not r.available and "bins" in r.message

    @pytest.mark.parametrize("bins", [2, 50, 200])
    def test_valid_bin_boundaries(self, bins):
        assert ch.histogram(pd.DataFrame({"v": np.arange(500.0)}), "v", bins).available

    def test_all_missing_and_all_infinite_columns(self):
        d = pd.DataFrame({"a": [np.nan] * 6, "b": [np.inf, -np.inf] * 3, "v": np.arange(6.0)})
        for c in ("a", "b"):
            r = ch.histogram(d, c)
            assert not r.available and "no valid numbers" in r.message

    def test_too_few_values(self):
        d = pd.DataFrame({"v": [1.0, np.nan, np.nan], "w": [1.0, 2.0, 3.0]})
        assert "only 1 valid value" in ch.histogram(d, "v").message
        assert not ch.scatter_plot(d, "v", "w").available
        assert "at least 3" in ch.density_plot(pd.DataFrame({"v": [1.0, 2.0]}), "v").message

    def test_text_column_is_not_a_numeric_chart(self, df):
        r = ch.histogram(df, "region")
        assert not r.available

    def test_numeric_text_is_charted_only_when_every_value_is_a_number(self):
        ok = pd.DataFrame({"v": ["1", "2", "3", "4"]})
        bad = pd.DataFrame({"v": ["1", "2", "x", "4"]})
        assert ch.histogram(ok, "v", 2).available and not ch.histogram(bad, "v", 2).available

    def test_same_column_for_both_scatter_axes(self, df):
        assert "different columns" in ch.scatter_plot(df, "price", "price").message

    def test_scatter_with_one_flat_axis_or_too_few_pairs(self):
        d = pd.DataFrame({"x": [1.0, 2, 3], "y": [4.0, 4, 4], "z": [1.0, np.nan, np.nan]})
        assert "flat line" in ch.scatter_plot(d, "x", "y").message
        assert "at least 2" in ch.scatter_plot(d, "x", "z").message

    def test_heatmap_needs_two_columns_and_some_defined_values(self):
        assert not ch.correlation_heatmap(None).available
        only_const = analyze_dataframe(pd.DataFrame({"a": [1.0] * 6, "b": [2.0] * 6}))
        assert "undefined" in ch.correlation_heatmap(only_const.correlation).message
        one = analyze_dataframe(pd.DataFrame({"a": [1.5, 2, 3, 4], "t": list("abcd")}))
        assert not ch.correlation_heatmap(one.correlation).available

    def test_heatmap_leaves_undefined_cells_blank_not_zero(self):
        d = pd.DataFrame({"a": np.arange(6.0), "b": np.arange(6.0) ** 2, "k": [1.0] * 6})
        z = np.asarray(ch.correlation_heatmap(analyze_dataframe(d).correlation).figure.data[0].z, dtype=float)
        assert np.isnan(z[2, 0]) and np.isnan(z[0, 2]) and np.isnan(z[2, 2]) and not np.isnan(z[0, 1])

    def test_unknown_aggregation(self, df):
        assert "must be one of" in ch.bar_chart(df, "region", "price", "mode").message

    def test_bar_chart_without_any_valid_pair(self):
        d = pd.DataFrame({"c": ["a", None, "b"], "v": [np.nan, 1.0, np.nan]})
        assert "No row has both" in ch.bar_chart(d, "c", "v", "mean").message


class TestHighCardinalityAndLargeInputs:
    def test_grouped_charts_refuse_too_many_categories(self):
        d = pd.DataFrame({"v": np.arange(500.0), "g": [f"g{i}" for i in range(500)]})
        for fn in (ch.box_plot, ch.violin_plot, ch.density_plot):
            r = fn(d, "v", "g")
            assert not r.available and "500 different values" in r.message
        assert not ch.scatter_plot(d.assign(w=d["v"] * 2), "v", "w", color="g").available

    def test_frequency_chart_caps_the_number_of_bars_and_discloses_it(self):
        d = pd.DataFrame({"c": [f"cat{i % 3000}" for i in range(6000)]})
        r = ch.frequency_chart(d, "c")
        assert r.available and len(r.figure.data[0].x) == ch.MAX_BARS
        assert "30 most frequent of 3,000" in r.notes[0]

    def test_bar_chart_refuses_a_thousand_categories(self):
        d = pd.DataFrame({"c": [f"k{i}" for i in range(1000)], "v": np.arange(1000.0)})
        assert "too many" in ch.bar_chart(d, "c", "v", "sum").message

    def test_large_scatter_is_sampled_deterministically_and_disclosed(self):
        rng = np.random.default_rng(1)
        d = pd.DataFrame({"x": rng.normal(size=60_000), "y": rng.normal(size=60_000)})
        a, b = ch.scatter_plot(d, "x", "y"), ch.scatter_plot(d, "x", "y")
        assert a.n_total == 60_000 and a.n_drawn == ch.MAX_POINTS and a.sampled
        assert "20,000 of 60,000" in a.notes[0]
        assert np.array_equal(a.figure.data[0].x, b.figure.data[0].x)             # same sample every time

    def test_sampling_never_changes_the_statistics(self):
        rng = np.random.default_rng(2)
        d = pd.DataFrame({"v": rng.normal(7, 2, 50_000)})
        assert ch.box_plot(d, "v").sampled
        assert analyze_dataframe(d).numeric[0].valid_count == 50_000
        assert analyze_dataframe(d).numeric[0].mean == approx(statistics.fmean(d["v"]))

    def test_histogram_uses_all_values_even_for_large_data(self):
        d = pd.DataFrame({"v": np.random.default_rng(3).normal(size=80_000)})
        r = ch.histogram(d, "v", 40)
        assert np.asarray(r.figure.data[0].y).sum() == 80_000 and not r.sampled


class TestDatasetSwitching:
    def test_charts_reflect_the_dataframe_they_are_given(self):
        a, b = load_fixture("f_employee.csv"), load_fixture("h_unusual_columns.csv")
        num_a = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and a[c].nunique() > 5][0]
        num_b = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c]) and b[c].nunique() > 5][0]
        ra, rb = ch.histogram(a, num_a, 10), ch.histogram(b, num_b, 10)
        assert np.asarray(ra.figure.data[0].y).sum() == a[num_a].notna().sum()
        assert np.asarray(rb.figure.data[0].y).sum() == b[num_b].notna().sum()
        assert ra.figure.layout.title.text != rb.figure.layout.title.text
        assert not ch.histogram(b, num_a, 10).available or num_a in b.columns     # a column of A is not found in B


# ---------------------------------------------------------------------------
# G. Export
# ---------------------------------------------------------------------------

@pytest.fixture
def report_and_df():
    rng = np.random.default_rng(10)
    n = 80
    d = pd.DataFrame({
        "sales": rng.gamma(3, 100, n).round(1), "visits": rng.integers(50, 500, n).astype(float),
        "flat": [4.0] * n, "city": rng.choice(["Oslo", "Rome", "Kyiv"], n), "member": rng.choice([True, False], n),
        "joined": pd.date_range("2024-01-01", periods=n, freq="3D"),
    })
    d.loc[3, "sales"] = np.nan
    d.loc[[5, 6], "city"] = None
    return analyze_dataframe(d), d


class TestTables:
    def test_all_documented_sheets_exist_in_order(self, report_and_df):
        rep, _ = report_and_df
        t = dx.build_tables(rep, "my data", "ds1")
        assert list(t) == list(dx.SHEET_ORDER)
        for needed in ("Overview", "Numeric Statistics", "Categorical Statistics", "Datetime Statistics",
                       "Missing Values", "Outliers", "Correlation", "Covariance", "Frequency Tables", "Methodology"):
            assert needed in t

    def test_numeric_sheet_values_equal_independent_calculations(self, report_and_df):
        rep, d = report_and_df
        t = dx.build_tables(rep)["Numeric Statistics"].set_index("Column")
        v = d["sales"].dropna()
        assert t.loc["sales", "Mean"] == approx(statistics.fmean(v)) and t.loc["sales", "Median"] == approx(statistics.median(v))
        assert t.loc["sales", "Sample std dev"] == approx(statistics.stdev(v))
        assert t.loc["sales", "Valid count"] == 79 and t.loc["sales", "Missing count"] == 1
        assert t.loc["sales", "Missing %"] == approx(1 / 80 * 100)
        assert t.loc["sales", "P95"] == approx(np.percentile(v, 95)) and t.loc["sales", "Q1"] == approx(np.percentile(v, 25))

    def test_undefined_statistics_are_blank_with_a_reason_never_zero(self, report_and_df):
        rep, _ = report_and_df
        t = dx.build_tables(rep)["Numeric Statistics"].set_index("Column")
        assert pd.isna(t.loc["flat", "Skewness (G1)"]) and pd.isna(t.loc["flat", "Excess kurtosis (G2)"])
        assert t.loc["flat", "Sample variance"] == 0 and "identical" in t.loc["flat", "Undefined statistics"]

    def test_correlation_sheet_matches_the_report_and_has_blank_undefined_cells(self, report_and_df):
        rep, d = report_and_df
        t = dx.build_tables(rep)
        c = t["Correlation"].set_index("Column")
        assert c.loc["sales", "visits"] == approx(d["sales"].corr(d["visits"]))
        assert pd.isna(c.loc["flat", "sales"]) and pd.isna(c.loc["sales", "flat"])
        assert t["Undefined Pairs"]["Why undefined"].str.contains("identical").any()

    def test_covariance_sheet_follows_the_selected_kind(self, report_and_df):
        rep, d = report_and_df
        s = dx.build_tables(rep, covariance="sample")["Covariance"].set_index("Column")
        p = dx.build_tables(rep, covariance="population")["Covariance"].set_index("Column")
        ok = d[["visits", "sales"]].dropna()
        assert s.loc["sales", "visits"] == approx(np.cov(ok["sales"], ok["visits"], ddof=1)[0, 1])
        assert p.loc["sales", "visits"] == approx(np.cov(ok["sales"], ok["visits"], ddof=0)[0, 1])
        assert dx.build_tables(rep, covariance="population")["Overview"].set_index("Item").loc["Covariance in this file", "Value"] == "population"

    def test_missing_outlier_frequency_and_overview_sheets(self, report_and_df):
        rep, d = report_and_df
        t = dx.build_tables(rep, "my data", "ds1")
        miss = t["Missing Values"].set_index("Column")
        assert miss.loc["city", "Missing count"] == 2 and miss.loc["city", "Missing %"] == approx(2.5)
        assert miss.index[0] == "city" or miss.loc[miss.index[0], "Missing %"] >= miss.loc["sales", "Missing %"]
        freq = t["Frequency Tables"]
        assert freq[freq["Column"] == "city"]["Count"].sum() == 78
        assert dict(zip(freq[freq["Column"] == "member"]["Value"], freq[freq["Column"] == "member"]["Count"])) == {
            "True": int(d["member"].sum()), "False": int((~d["member"]).sum())}
        o = t["Overview"].set_index("Item")["Value"]
        assert o["Dataset"] == "my data" and o["Dataset id"] == "ds1" and o["Rows"] == "80"
        assert "Outlier flags" in set(t["Methodology"]["Topic"])
        assert "Disclaimer" in t["Outliers"].columns

    def test_conversions_and_warnings_are_exported(self):
        rep = analyze_dataframe(pd.DataFrame({"n": ["1", "2", "3", "4"], "c": ["x", "y", "x", "y"]}))
        m = dx.build_tables(rep)["Methodology"]
        assert (m["Topic"] == "Type decision").any() and (m["Topic"] == "Warning").any()


class TestFiles:
    def test_excel_has_one_valid_sheet_per_table_with_correct_values(self, report_and_df):
        rep, d = report_and_df
        tables = dx.build_tables(rep, "x", "id")
        sheets = pd.read_excel(io.BytesIO(dx.tables_to_excel(tables)), sheet_name=None)
        assert list(sheets) == list(dx.SHEET_ORDER)
        assert all(len(name) <= 31 and not re.search(r"[\[\]:*?/\\]", name) for name in sheets)
        num = sheets["Numeric Statistics"].set_index("Column")
        assert num.loc["visits", "Mean"] == approx(d["visits"].mean())
        assert pd.isna(num.loc["flat", "Skewness (G1)"])                       # still blank after the round trip
        assert sheets["Correlation"].shape == tables["Correlation"].shape

    def test_excel_keeps_numbers_as_numbers(self, report_and_df):
        rep, _ = report_and_df
        sheets = pd.read_excel(io.BytesIO(dx.tables_to_excel(dx.build_tables(rep))), sheet_name="Numeric Statistics")
        assert pd.api.types.is_numeric_dtype(sheets["Mean"]) and pd.api.types.is_numeric_dtype(sheets["Valid count"])

    def test_csv_zip_contains_every_table_with_the_same_values(self, report_and_df):
        rep, _ = report_and_df
        tables = dx.build_tables(rep)
        with zipfile.ZipFile(io.BytesIO(dx.tables_to_csv_zip(tables))) as zf:
            names = zf.namelist()
            assert len(names) == len(tables) and all(n.endswith(".csv") for n in names)
            back = pd.read_csv(io.BytesIO(zf.read("numeric_statistics.csv")))
        assert back["Mean"].round(6).tolist() == tables["Numeric Statistics"]["Mean"].round(6).tolist()

    def test_single_csv_download(self, report_and_df):
        rep, _ = report_and_df
        tables = dx.build_tables(rep)
        csv = pd.read_csv(io.BytesIO(dx.table_to_csv(tables, "Missing Values")))
        assert list(csv["Column"]) == list(tables["Missing Values"]["Column"])

    @pytest.mark.parametrize("label,table,expected", [
        ("sales 2024.csv", None, "descriptive_statistics_sales_2024_csv"),
        ("../../etc/passwd", "Numeric Statistics", "descriptive_statistics_etc_passwd_numeric_statistics"),
        ("NovaMart_demo", "Missing Values", "descriptive_statistics_NovaMart_demo_missing_values"),
        ("", None, "descriptive_statistics_dataset"), ("日本語", "Outliers", "descriptive_statistics_dataset_outliers"),
        ("x" * 200, None, "descriptive_statistics_" + "x" * 40),
    ])
    def test_filenames_are_safe_and_name_the_dataset(self, label, table, expected):
        assert dx.export_stem(label, table) == expected

    def test_invalid_sheet_names_are_made_valid(self):
        data = pd.DataFrame({"a": [1]})
        xl = pd.read_excel(io.BytesIO(common_export.sheets_to_excel_bytes(
            {"Bad:Name/1": data, "x" * 50: data, "Bad:Name/1 ": data, "[x]": data})), sheet_name=None)
        assert all(len(n) <= 31 and not re.search(r"[\[\]:*?/\\]", n) for n in xl) and len(xl) == 4

    def test_formula_injection_is_neutralized_in_values_and_column_names(self):
        evil = pd.DataFrame({"=HYPERLINK(\"http://x\")": [1.5, 2.5, 3.5, 4.5], "@cmd": ["=1+1", "+SUM(A1)", "-2+3", "ok"],
                             "plain": [10.0, 20.0, 5.0, 1.0]})
        rep = analyze_dataframe(evil)
        tables = dx.build_tables(rep, "=bad", "@id")
        blob = dx.tables_to_excel(tables)
        csvs = zipfile.ZipFile(io.BytesIO(dx.tables_to_csv_zip(tables)))
        for text in [csvs.read(n).decode() for n in csvs.namelist()]:
            for line in text.splitlines():
                for _, cell in re.findall(r'(?:^|,)("?)([=+@\-][^,"]*)', line):
                    assert re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", cell, re.I), f"unneutralized formula-like cell in CSV: {cell}"
        for name, frame in pd.read_excel(io.BytesIO(blob), sheet_name=None, dtype=str).items():
            cells = [str(c) for c in frame.columns] + [str(v) for v in frame.to_numpy().ravel() if isinstance(v, str)]
            for c in cells:
                assert not re.match(r"^[=+@\-]", c) or re.match(r"^-?\d+(\.\d+)?(e[+-]?\d+)?$", c, re.I), (name, c)
        freq_csv = csvs.read("frequency_tables.csv").decode()
        assert "'=1+1" in freq_csv and "\n=1+1" not in freq_csv and ",=1+1" not in freq_csv
        assert tables["Frequency Tables"]["Value"].tolist().count("=1+1") == 1      # tables stay raw; files are sanitized

    def test_empty_results_still_export_with_explanatory_sheets(self):
        rep = analyze_dataframe(pd.DataFrame({"only": ["a", "b", "a"]}))
        tables = dx.build_tables(rep)
        assert tables["Numeric Statistics"].iloc[0, 0] == "This dataset has no numeric columns."
        assert tables["Correlation"].iloc[0, 0].startswith("Correlation needs")
        sheets = pd.read_excel(io.BytesIO(dx.tables_to_excel(tables)), sheet_name=None)
        assert len(sheets) == len(dx.SHEET_ORDER) and all(len(s) >= 1 for s in sheets.values())

    def test_report_for_an_empty_dataframe_exports(self):
        tables = dx.build_tables(analyze_dataframe(pd.DataFrame()))
        assert len(dx.tables_to_excel(tables)) > 0 and len(dx.tables_to_csv_zip(tables)) > 0

    def test_a_table_over_the_excel_row_limit_is_refused_with_advice(self, monkeypatch, report_and_df):
        rep, _ = report_and_df
        tables = dx.build_tables(rep)
        monkeypatch.setattr(common_export, "EXCEL_MAX_ROWS", 3)
        with pytest.raises(common_export.ExportError, match="CSV download"):
            dx.tables_to_excel(tables)
        assert len(dx.tables_to_csv_zip(tables)) > 0                          # CSV still works

    def test_large_dataset_export_is_fast_and_bounded(self):
        rng = np.random.default_rng(6)
        n = 150_000
        d = pd.DataFrame({"v": rng.normal(size=n), "w": rng.normal(size=n), "c": [f"k{i % 90000}" for i in range(n)]})
        rep = analyze_dataframe(d)
        tables = dx.build_tables(rep, "big", "big")
        assert len(tables["Frequency Tables"]) <= 50 + 3                       # tables stay small whatever the row count
        assert rep.numeric[0].valid_count == n
        cat = tables["Categorical Statistics"].set_index("Column").loc["c"]
        assert len(cat["Mode"]) < 1000 and "+59,975 more tied" in cat["Mode"]       # 60,000 tied; cell stays small, loss disclosed
        assert rep.categorical[0].modes.__len__() == 60000                          # the report itself keeps every mode
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(dx.tables_to_excel(tables)) > 0                              # no Excel cell truncation

    def test_export_reflects_the_current_settings(self):
        d = pd.DataFrame({"x": np.arange(1.0, 21.0), "y": np.arange(1.0, 21.0) ** 2})
        a = dx.build_tables(analyze_dataframe(d, DescriptiveSettings(percentiles=(10, 90), correlation_method="spearman")))
        assert {"P10", "P90"} <= set(a["Numeric Statistics"].columns) and "P5" not in a["Numeric Statistics"].columns
        o = a["Overview"].set_index("Item")["Value"]
        assert o["Correlation method"] == "spearman" and o["Percentiles shown"] == "10; 90"

    def test_two_datasets_never_share_export_values(self):
        a = analyze_dataframe(pd.DataFrame({"v": [1.0, 2, 3, 4]}))
        b = analyze_dataframe(pd.DataFrame({"v": [10.0, 20, 30, 40]}))
        ta, tb = dx.build_tables(a, "A", "ida"), dx.build_tables(b, "B", "idb")
        assert ta["Numeric Statistics"]["Mean"].iloc[0] == 2.5 and tb["Numeric Statistics"]["Mean"].iloc[0] == 25.0
        assert ta["Overview"].set_index("Item").loc["Dataset id", "Value"] == "ida"

    def test_no_fabricated_values_for_missing_inputs(self):
        rep = analyze_dataframe(pd.DataFrame({"v": [np.nan, np.nan, np.nan, 5.0]}))
        row = dx.build_tables(rep)["Numeric Statistics"].iloc[0]
        assert row["Valid count"] == 1 and pd.isna(row["Sample variance"]) and pd.isna(row["Skewness (G1)"])
        assert row["Mean"] == 5.0 and row["Missing %"] == 75.0
