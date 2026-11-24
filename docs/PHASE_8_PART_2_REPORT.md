# Phase 8 – Part 2: Descriptive Statistics (module 14)

This document describes what is implemented and was tested. Nothing here is planned-only.

## 1. Objective

Deliver the Descriptive Statistics module (`descriptive`, number 14) for any dataset: statistics, a data-quality
view, charts, and CSV / Excel exports, using the Phase 8 Part 1 scaffold. Inferential, predictive and prescriptive
modules (15–17) are untouched and remain "not yet delivered".

## 2. Baseline hash

Baseline: `MarketPulse_AI_Legacy_Cache_Fix_Part1_FINAL.zip`
SHA-256 `004d362b2f641b510654ff172efd6364742b44ce6b9825ef2bd90edcb194ecf1` – verified before any work (match).
Baseline test result: 1,942 passed, 16 warnings (12 scikit-learn `InconsistentVersionWarning`, 2 numpy
"invalid value … divide", 2 numpy "… reduce"). The original ZIP was kept read-only and never modified.
The hash of the final ZIP is not stored here (a ZIP cannot contain its own hash); it is given in
`PHASE_8_PART_2_FINAL_REPORT.md` and in the delivery message.

## 3. Architecture study

* `src/common/modules.py` – module registry (`ModuleSpec`, `implemented` flag).
* `src/common/availability.py` – the descriptive module is available when the dataset has at least one non-empty column.
* `src/common/dataset_summary.py` – shared column typing (`summarize_dataset`). Reused for the first typing pass.
* `src/common/export.py` – `sanitize_for_spreadsheet`, CSV/Excel writers, Excel row limit. Reused unchanged.
* `dashboard/components/stats_ui.py` – `render_module_page(spec, body_fn)` scaffold: dataset gate, KPI row, column
  details, scoped session state. Reused unchanged.
* Part 1 cache fix (`dashboard/components/data_loader.py`) – accepted and **not modified**.

Findings that shaped the design: the shared classifier calls numeric strings and mixed-time-zone text dates
"categorical", and treats a column with at most two values drawn from {True, False, 0, 1, yes/no…} as boolean. The
engine therefore re-checks these cases itself and records each decision (see §13).

## 4. Implementation summary

| Layer | File | Role |
|---|---|---|
| Contracts | `src/descriptive/contracts.py` | typed dataclasses, `to_dict()` (NaN/inf → null), `DescriptiveSettings.validate()` |
| Engine | `src/descriptive/engine.py` | all statistics; no Streamlit; never modifies its input |
| Charts | `src/descriptive/charts.py` | Plotly figure builders returning `ChartResult(available, figure, message, notes, …)` |
| Export | `src/descriptive/export.py` | report → 12 named tables → CSV (single / zipped) and Excel |
| UI | `dashboard/components/descriptive_ui.py` | 13 page sections, controls, downloads |
| Page | `dashboard/pages/14_descriptive_statistics.py` | passes `render_descriptive_body` to the scaffold |
| Registry | `src/common/modules.py` | `implemented=True` on the descriptive spec only (one added line) |

The report is calculated by `analyze_dataframe(df, settings)` and cached by `st.cache_data(max_entries=8)` on a
worker that receives only the DataFrame and the settings. The cache key is the table's **content**, so a changed
file or another dataset can never receive an earlier result, and the worker never reads session state.

## 5. Statistical methods and formulas

| Quantity | Definition |
|---|---|
| Mean | `math.fsum(x) / n` |
| Median, quartiles, percentiles | linear interpolation between closest ranks (numpy default, Hyndman–Fan type 7); same method for every column |
| Mode | every value tied for most frequent, ascending; none if every value occurs once; ties are stated |
| Sample variance / sd | `Σ(x−x̄)² / (n−1)`, needs n ≥ 2 |
| Population variance / sd | `Σ(x−x̄)² / n` |
| Range, IQR | `max−min`; `Q3−Q1` |
| Skewness | adjusted Fisher–Pearson `G1 = √(n(n−1))/(n−2) · m3/m2^1.5` (equals `scipy.stats.skew(bias=False)`), needs n ≥ 3 and variance > 0 |
| Kurtosis | **excess** kurtosis `G2 = (n−1)/((n−2)(n−3)) · ((n+1)·g2 + 6)`, `g2 = m4/m2² − 3` (equals `scipy.stats.kurtosis(bias=False)`; normal = 0), needs n ≥ 4 and variance > 0 |

Sums use `math.fsum`; results that overflow are reported as undefined with the reason, not as a number.
Categorical columns get counts, unique count, mode(s), top/least frequent values and percentages; boolean columns
get true/false counts and percentages; datetime columns get earliest, latest, span, typical spacing, rows per
period and invalid-date counts.

## 6. Handling of undefined statistics

An undefined value is `None` in the report, blank in CSV/Excel, "—" on the page, and always has a plain-English
reason in the `undefined` dictionary (shown on the page and in the "Undefined statistics" column). It is **never 0**.

* Constant numeric column (n ≥ 2): mean = median = the value; variance = sd = range = IQR = 0; skewness and kurtosis undefined ("all values are identical").
* One valid value: sample variance, sample sd, skewness, kurtosis undefined ("needs at least 2/3/4 values"); population variance 0.
* Fewer than 3 (skewness) or 4 (kurtosis) values: undefined with the count stated.
* Correlation with a constant column: undefined, naming the constant column.
* Empty or all-missing column: reported as empty, with no statistics.

## 7. Missing-value methodology

`Missing % = missing / all rows in the dataset × 100` (the denominator includes the missing rows). Blank text
counts as missing. ±infinity is counted separately and is **not** missing; it is excluded from statistics.
Missing values are never converted to 0 and never imputed. Valid % = (rows − missing − infinite − invalid dates) / rows.
The missing-value section also gives per-column counts, rows with at least one missing cell, complete-row
percentage, and duplicate-row count (duplicates are reported, never removed).

## 8. Outlier methodology

Tukey fences with multiplier k (1.5 default; 2.0 and 3.0 selectable): flagged if `x < Q1 − k·IQR` or `x > Q3 + k·IQR`
(strict inequality, so a value exactly on a fence is not flagged). Needs at least 4 valid values. Reported per column:
Q1, Q3, IQR, both bounds, count, percentage of valid values, and up to 20 flagged values with row labels, ordered by
distance outside the fence. If IQR = 0 any different value is flagged and a warning says so. Every outlier table carries
the statement that a flag is a prompt to look closer, **not proof of an error**. The original data is never changed.

## 9. Correlation and covariance methodology

* Pearson and Spearman (Spearman = Pearson on average ranks, ties handled). The method used is shown on the page and in every export.
* Pairwise-complete observations for every pair; the number of pairs is reported per pair.
* A pair needs at least 3 complete pairs and variation in both columns; otherwise it is undefined with the reason (never 0).
* Identifier columns are excluded by default (listed with the reason). At most 60 columns are used; the rest are listed as excluded.
* Covariance: sample (÷ n−1, needs 2 pairs) and population (÷ n, needs 1 pair), both computed; the page and export choose which to show.
* The page states that correlation does not imply causation.

## 10. Visualization features

Eight chart types, selectable on the page: Histogram, Box plot, Violin plot, Density plot, Correlation heatmap,
Scatter plot, Bar chart, Frequency distribution. Interactive choices: chart type, column(s), optional grouping /
colour, number of bins (2–200; the page offers 2–100), bar aggregation (count / mean / median / sum), and the
correlation method (from the settings panel).

* A chart that cannot be drawn returns `available=False` with a plain-English reason (constant, empty, all-missing or non-numeric column; fewer than 2 / 3 valid values; same column on both scatter axes; invalid bins; unknown column; too many groups).
* Grouped charts allow at most 20 groups; bar charts at most 30 bars (the "most frequent N of M" is disclosed); a bar chart of a numeric value is refused above 120 categories (4 × the bar limit), with advice to pick a column with fewer.
* Scatter, box, violin and density plots draw at most 20,000 points. Larger inputs are sampled with a fixed seed (same sample every time) and the chart states "20,000 of N rows drawn". The histogram always uses every value. **Statistics are never calculated from the sample.**
* Undefined heatmap cells are blank, never 0.
* Charts are built from the current DataFrame on every run, so they update on dataset switch.

## 11. Dashboard integration

Page `dashboard/pages/14_descriptive_statistics.py` uses the shared scaffold (dataset gate, KPI row, column details).
Sections in order: Analysis Settings, Data Quality Overview, Numeric Column Statistics, Categorical and Yes/No Column
Statistics, Date Column Summaries, Frequency Distributions, Missing-Value Analysis, Outlier Analysis, Correlation
Analysis, Covariance Analysis, Visualizations, Download Reports, Methodology and Statistical Warnings. Each section
opens with a plain-English explanation. Widget keys include the dataset id, so one dataset's selections never apply to
another. The reset button clears all module state (scoped `mp_page_` keys) as before.

The page keeps a status info box ("Descriptive Statistics is delivered in Part 2 of Phase 8 and is implemented…").
This is deliberate: the existing test `test_phase8_part1_foundation.py::TestDashboardIntegration::
test_new_page_renders_on_demo_without_exception` asserts an info box containing "Part" and the module title for all
four modules; that test was **not edited**. Its inline comment ("not delivered yet") is now outdated for module 14 but
the assertion still holds.

## 12. Export functionality

Tables (one sheet / one CSV each, in this order): Overview, Numeric Statistics, Categorical Statistics, Boolean
Statistics, Datetime Statistics, Missing Values, Outliers, Correlation, Covariance, Undefined Pairs, Frequency Tables,
Methodology. Downloads on the page: one table as CSV, all tables as a zipped set of CSVs, and one Excel workbook.

* File names: `descriptive_statistics_<dataset>[_<table>]` using a sanitised dataset label (letters, digits, `_`, max 40 characters).
* Sheet names are valid Excel names (≤ 31 characters, no `[]:*?/\`, unique).
* Formula injection: cell values and column headers starting with `=`, `+`, `-` (non-numeric), `@` are neutralised in CSV and Excel through the shared `sanitize_for_spreadsheet`; column names that look like formulas are neutralised in the tables. Real negative numbers are untouched.
* Undefined statistics are blank cells with the reason beside them; nothing is invented. Empty tables carry a one-row note.
* Long tie lists are shortened to 25 items with "(+N more tied values not listed)" because an Excel cell holds 32,767 characters (found by a test with 60,000 tied categories, which previously produced a 468,888-character cell). The full list stays in the report object.
* A sheet over Excel's row limit raises a clear message advising the CSV download; CSV remains available.
* Exports are built from the report of the active dataset and current settings (covariance type follows the page selection).

## 13. Dataset compatibility

No column name is hard-coded. Columns are typed from their content, and each decision that changes the shared
typing is recorded in the report ("conversions") and exported:

* Numeric text becomes numeric only if **every** non-missing value parses and none has a leading zero (so codes such as `001` stay text).
* 0/1 numeric columns are described as yes/no; ≤ 2-value columns drawn from the shared boolean vocabulary likewise.
* Digit-only text is never read as a date; date-like text the shared classifier called categorical is read as dates; mixed UTC offsets are converted to UTC with a note.
* Identifiers (high-cardinality or integer-like ID columns, using the shared ratio constant) are described but excluded from correlation.
* High-cardinality categoricals keep counts for the top values and state the number of distinct values.
* Mixed types, constant, empty, one-row, duplicated-row, all-missing and infinite-value columns are all handled and covered by tests (the 16 files of the project's diverse-schema and generic-schema fixture sets).

## 14. Performance considerations

Computation is vectorised (numpy/pandas); percentiles are one call per column; correlation uses array operations per
pair. Measured on this environment (single run, Python 3.13, numpy 2.5, pandas 3.0; 50% numeric, 50% text columns):

| Rows × columns | `analyze_dataframe` | tables + Excel | scatter + box |
|---|---|---|---|
| 10,000 × 20 | 0.26 s | 0.13 s | 0.09 s |
| 200,000 × 20 | 3.5 s | 0.05 s | 0.04 s |
| 1,000,000 × 12 | 11.9 s | 0.05 s | 0.16 s |

The result is cached by content (max 8 entries), so changing a chart control does not recompute the statistics.
Hashing a very large frame for the cache key is itself a cost; no further caching was added. Chart sampling is disclosed (§10).

## 15. Exact changed-file list

Created (10 files: 6 source, 3 test, 1 document):
`src/descriptive/__init__.py`, `src/descriptive/contracts.py`, `src/descriptive/engine.py`, `src/descriptive/charts.py`,
`src/descriptive/export.py`, `dashboard/components/descriptive_ui.py`, `tests/test_phase8_part2_engine.py`,
`tests/test_phase8_part2_charts_export.py`, `tests/test_phase8_part2_dashboard.py`, `docs/PHASE_8_PART_2_REPORT.md`.
Modified (2): `src/common/modules.py` (one line), `dashboard/pages/14_descriptive_statistics.py`.
No other file differs from the baseline (verified by a recursive comparison of the extracted baseline and the final tree;
SHA-256 per file is in `PHASE_8_PART_2_CHANGED_FILES.txt`, delivered next to the ZIP).

## 16. Test coverage

367 new tests: 235 engine, 86 chart/export, 46 dashboard (Streamlit `AppTest`). Expected values come from the Python
`statistics` module, scipy, numpy, or hand calculation – never from the engine itself. Areas: exact numeric values;
constant, one-observation and tiny samples; seeded randomised property checks (`hypothesis` is not installed, so
these are loops over fixed seeds, not shrinking property tests); categorical, boolean, datetime; Pearson/Spearman vs
scipy; sample/population covariance; outlier fences and ordering; settings validation; 16 dataset shapes; every chart
type with empty/constant/missing/invalid/large input; CSV and Excel round trips, sheet names, formula injection,
filenames, Excel row limit; page rendering on the demo and on 7 uploaded fixtures; every chart type selectable; dataset
switching A → B → A → demo in one session; back-to-back and interleaved sessions; a changed file; reset; registry.

Two environment notes: the dashboard test simulates several sessions in **one process** (this is not a live
multi-browser test), and a pre-existing test (`test_phase8_part1_foundation.py::…test_oversized_excel_shows_reason_instead_of_crashing`)
assigns `export.EXCEL_MAX_ROWS = 1` and never restores it. The new test files pin the real limit before each test so
they do not depend on test order. The old test was not changed.

## 17. Mutation testing results

Run on scratch copies only (never in the project). The unmutated copy passed 367/367. Each fault was confirmed to be
injected (the target text had to occur exactly once). All were detected:

| # | Fault | Result | First failing test |
|---|---|---|---|
| 1 | constant-column skewness returned as 0 | killed (7 failed) | `engine::TestConstantAndTinySamples::test_constant_column_rules` |
| 2 | constant-column kurtosis returned as 0 | killed (6) | same |
| 3 | one-observation sample variance returned as 0 | killed (3) | `engine::TestConstantAndTinySamples::test_one_observation` |
| 4 | constant-column correlation returned as 0 | killed (5) | `engine::TestCorrelation::test_constant_column_is_undefined_not_zero` |
| 5a | wrong missing denominator (numeric summary) | killed (3) | `engine::…::test_nan_missing_and_infinite_are_counted_separately` |
| 5b | wrong missing denominator (quality table) | killed (6) | `engine::…::test_missing_percentage_uses_all_rows_as_denominator` |
| 6 | wrong percentile (`lower` instead of linear) | killed (4) | `engine::…::test_percentiles_use_linear_interpolation` |
| 7 | wrong outlier boundary (upper fence) | killed (4) | `engine::TestOutliers::test_one_outlier_with_exact_fences` |
| 8 | stale dataset results | killed (3) | `dashboard::TestIsolation::test_statistics_tables_belong_to_the_active_dataset_after_every_switch` |
| 9 | wrong export values (Mean column = median) | killed (2) | `charts_export::TestTables::test_numeric_sheet_values_equal_independent_calculations` |
| X1 | extra: outlier `>=` instead of `>` | killed (3) | `engine::TestOutliers::test_fences_are_strict` |
| X2 | extra: sample covariance ÷ n | killed (4) | `engine::TestCovariance::…` |
| X3 | extra: Spearman without ranking | killed (14) | `engine::TestCorrelation::test_pearson_and_spearman_match_scipy` |

Honest note on fault 8: the first version of my dashboard tests **did not** detect it (46/46 passed against the stale
mutant), because they only compared row counts and the shared column-details table, which is always correct. I
strengthened them to read the report-driven numeric table after each switch; the same mutant is now killed by three tests.

## 18. Full regression results

Full suite on the final tree: **2,309 passed, 0 failed, 0 skipped, 0 xfail, 16 warnings** (1,942 baseline + 367 new;
the same 16 warnings as the baseline). No existing test was edited, weakened, skipped or marked xfail.

## 19. Known limitations

* No live multi-browser test; session isolation is shown with simulated `AppTest` sessions in one process.
* `hypothesis` is unavailable; randomised checks are seeded loops.
* `tzdata` is not installed here, so named-time-zone parsing was tested with fixed offsets only.
* Time-zone-aware dates with mixed offsets are converted to UTC (disclosed); named-zone behaviour is untested.
* A column of numbers stored as text with one non-numeric entry stays text (all-or-nothing rule); it is not partially converted.
* Correlation uses at most 60 columns; the others are listed as excluded.
* Very large frames (≥ 1 M rows) take ~12 s for the first calculation; later interactions use the cache.
* Charts of more than 20,000 points are sampled (disclosed); statistics always use all valid rows.
* The Part 1 scaffold and the page still call `use_container_width`, which Streamlit 1.67 logs as deprecated; kept for the project's declared `streamlit>=1.30`.
* Excel tie lists are shortened (§12); the CSV/Excel mode cell is therefore not complete when more than 25 values tie.
* The pre-existing leak of `EXCEL_MAX_ROWS` in an older test remains (not in scope to edit).

## 20. Final ZIP hash

Not stored inside the ZIP (impossible). See `PHASE_8_PART_2_FINAL_REPORT.md` and the delivery message.

## 21. Final recommendation

All acceptance criteria in the authorization were checked and passed (see §2, §16–18 and the fresh-extraction results
in `PHASE_8_PART_2_TEST_REPORT.md`). Recommendation: **READY FOR INDEPENDENT REVIEW**. Part 3 was not started; nothing
was deployed or pushed.
