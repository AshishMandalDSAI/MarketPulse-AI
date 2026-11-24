# MarketPulse AI Upgrade: Test Report

## PART 01 (navigation, Statistics workspace, chart labelling, currency)

**Labels used in this report:**

- **[EXECUTED]**: run here, with the output recorded;
- **[BROWSER]**: Chrome via Playwright against `streamlit run app.py`;
- **[NOT RUN]**: not executed.

### 1. Full regression suite on the final PART 01 code [EXECUTED]

| Item | Value |
|---|---|
| Command | `python -I -m pytest tests/ -v -p no:cacheprovider -rfEs` |
| Environment | Python 3.13.18, pytest 9.1.1, virtualenv installed from `requirements.txt` (60 packages), Linux x86-64 |
| **Collected** | **3,150** |
| **Passed** | **3,150** |
| **Failed** | **0** |
| **Skipped** | **0** |
| Errors | 0 |
| Not run | none |
| Warnings | 1: third-party `StarletteDeprecationWarning` from `fastapi.testclient` (also present in the baseline) |
| Duration | 548.8 s |
| Log | `PART01_TEST_LOG.txt` (delivered with the ZIP) |
| Baseline (PART 00) | 3,030 collected, 3,030 passed |

The suite was run again from a fresh extract of the PART 01 backup ZIP, in a new virtualenv. That run's counts and the ZIP's SHA-256 are in `PART01_SHA256.txt`, delivered next to the ZIP, because a ZIP cannot contain its own checksum.

**Where the 120 additional tests come from:**

| Source | Tests |
|---|---|
| New `tests/test_upgrade_part01.py` | 107 |
| More menu and kept-module cases in `test_phase8_part1_foundation.py` | +8 |
| More page files covered by `test_final_dashboard_pages.py` | +5 |

### 2. New tests: `tests/test_upgrade_part01.py` (107) [EXECUTED]

The letters refer to the master specification's test list.

| Area | What is verified |
|---|---|
| **W: navigation registry** | The menu equals the approved 16 entries in order. `app.py` uses the registry and contains no literal labels. No dashboard file (or the analyst) hard-codes a menu label. Group headings sit on the right entries, including the CSS rules. Every menu and kept module imports and has `render()`. The four old models and the three statistics pages are out of the menu but still on disk. Module-to-page mapping, including unknown keys. |
| **W: in the app** | Every menu entry renders on the demo with no exception and no failure fallback (any exception name inside a "not available" message counts as a failure). Every entry renders on an uploaded dataset. A stale old label falls back to page 00. A forced jump to the Executive Overview works. The no-dataset gate still blocks pages. The Prescriptive page badge shows "PAGE 15". |
| **Statistics workspace** | The three tab names are exact. The default tab is Descriptive and only that tab runs. Each tab renders its own module and none of the others. The chosen tab survives a rerun. An unknown requested tab falls back safely. All tabs work on uploaded data. |
| **Transition pages 04–07** | Each shows "*title*: not implemented yet … PART 02". The previous module does **not** run until switched on. When switched on, it renders unchanged and is labelled "Previous module · X · not the *new module*". |
| **V: chart-label helpers** | Readable names. Total, Average and Number-of-records wording. Percentage units. Time axis named from the real frequency (Date, Week, Month, Quarter, Year). Chart titles state the aggregation. Currency not repeated when already in the name. One point per date when aggregating. Expected axis titles for **all 9** recommender chart types. |
| **V: charts in the app** | Executive Overview demo: every chart has a title and x/y titles, including "Revenue (INR)". Upload without a currency: every chart labelled, with no INR or ₹ anywhere. Diverse fixtures (`a_marketing`, `b_sales`, `e_education`, `g_no_date`): every rendered chart labelled. Marketing Performance on demo and upload: labelled, with one point per date and "Total …" on trends. A USD upload shows "(USD)" on monetary axes only. |
| **K: currency detection** | 9 identified cases, 3 shared-symbol cases (symbol kept, no code), 2 conflict cases (no currency), 4 no-evidence cases (including plain numbers, so INR is never assumed). XLSX header detection. The user override (automatic, none, a code, an invalid code). Save and load round trip. |
| **K: currency display** | Demo is INR and its formatting is identical to the previous `format_inr` for every value tested. Demo money columns are explicit. An upload with no currency shows no symbol. An upload with USD shows `$` and the "USD" unit. **F:** switching datasets switches the currency. Chart tick prefixes follow the active currency. KPI cards get a currency only for sums and averages of monetary columns. |
| **Analyst** | An upload is never described in "₹" or "per rupee". An upload with no currency gets no symbol. The demo stays in INR. |
| **D/E: upload page** | Monetary-column selection. The currency step offers automatic, none and all codes, and saves the choice. (XLSX is covered by the detection test.) |
| **X: Forecast** | The page renders on the demo (with a chart) and on an upload (no "not available"). |

### 3. Existing tests adapted to the approved menu (no assertion weakened)

| File | Change |
|---|---|
| `test_phase8_part1_foundation.py` | `test_original_modules_00_to_13_are_unchanged_and_in_order` and `test_new_pages_registered_after_13_matching_specs` asserted the **superseded** layout (00–13 unchanged; statistics pages at 14–17). They were replaced by stricter tests: `test_menu_is_exactly_the_approved_structure_in_order`, `test_every_module_is_reachable_from_the_menu` and `test_modules_removed_from_menu_are_kept`. The import test now covers all 16 menu modules plus the 7 kept modules (previously 18). Module navigation goes through the Statistics tab, and the title check accepts the tab heading. The 4-module label test asserts the new hosting labels, the tab labels, and the unchanged module identities 14–17. |
| `test_phase8_part2_dashboard.py`, `test_phase8_part3_dashboard.py`, `test_phase8_part4_dashboard.py`, `test_phase8_part4_workflow.py`, `test_phase8_part6_integration.py` | The navigation line is now `goto_module(at, key)`, which selects the page and the tab. The title check also looks at the tab heading. Each file's test count is unchanged. |
| `test_final_dashboard_pages.py` | Page labels come from the registry. Parametrised IDs changed with the labels (`[01 …]` is now `[02 …]`, `[11 …]` is now `[12 …]`). |
| `test_ultimate_ui.py` | The upload callback is expected to jump to `navigation.EXECUTIVE_OVERVIEW` ("02 · 📊 Executive Overview"). |
| `test_phase7_part7_dashboard_copilot.py` | The Decision Intelligence registration is checked in the registry that `app.py` renders, instead of by searching `app.py`'s text. |

Comparing test IDs with the baseline: apart from the parametrised IDs renamed with the labels, the only baseline tests no longer present are the two superseded structure tests listed above.

### 4. Browser verification (Chrome, Playwright) [BROWSER]

| Pass | Result |
|---|---|
| Demo, 1440 px: all 16 menu pages | 0 exceptions, 0 error banners, 0 horizontal overflow, 0 console errors; page switch 1.9–3.9 s |
| Demo, 820 px and 390 px: Upload, Executive Overview, Forecast | same, 0 issues |
| Statistics tabs clicked one by one | each tab shows only its module; 0 exceptions; 0 console errors |
| Transition page 06 | previous module hidden until switched on; when on, it renders inside its labelled frame with 0 exceptions |
| USD upload (2,000 rows), all 16 pages | Currency step detected USD; KPIs and axes in USD; 0 exceptions. **One error banner on 15 · Prescriptive Analytics:** "The channel maximums add up to only 91.15, less than the total budget 165.72" (91.15 / 165.72 = 0.55). This is the **known pre-existing optimizer default-limit bug**, scheduled for PART 02 under your decision. It is not a PART 01 regression. |
| USD upload re-checked after the duplicate-currency fix | axes read "Total Net Sales (USD)" and so on; 0 issues |

### 5. Protected files [EXECUTED]

84 of 85 protected files are byte-identical to the baseline. `dashboard/pages/0_upload_data.py` changed in two places only: the two menu labels now use registry constants, and the new "2b · Currency" step plus saving the chosen currency. The Forecast module, model artifacts, security files and `requirements.txt` are unchanged.

### 6. Not run / platform limitations

- Windows and macOS were not run (Linux only).
- Browsers other than Chrome were not run.
- Docker was not rebuilt in PART 01: no Dockerfile, entrypoint or requirement changed.

---

# PART 02 test report

## Final run: on the exact packaged snapshot

| Item | Value |
|---|---|
| Where | Fresh extraction of the PART 02 package into an empty directory; new virtual environment; `pip install -r requirements.txt pytest` (requirements unchanged) |
| Python / pytest | 3.13.18 / 9.1.1 |
| Command | `python -m pytest -p no:cacheprovider -q -rfEs` |
| **Collected** | **3,298** (3,150 from PART 01 + 148 new) |
| **Passed** | **3,295** |
| **Failed** | **0** |
| **Skipped** | **3**: `TestRealKaggleFile`, because the user's real Kaggle file was not provided (reason printed in the log) |
| Not run | 0 |
| Duration | 650.98 s |
| Warning | 1, the same FastAPI/Starlette TestClient deprecation as the baseline |
| Log | `PART02_TEST_LOG.txt` (verbatim) |

**Code identity:** the snapshot's code is identical to the delivered ZIP. The final ZIP adds only this report, the log and the status / manifest documents, and its non-documentation files were verified byte-identical to the tested snapshot.

## Also verified on the packaged snapshot

- **Dashboard and API start:**
  - `streamlit run app.py`: HTTP 200, `/_stcore/health` 200;
  - `uvicorn api.main:app`: `/health` reports `healthy`, `pipeline_ready: true`.
- **Browser pass, real Chrome against that server:**
  - **Datasets covered:** the demo (spend-based), an upload of the Kaggle-structure replica (exposure-based) and an upload of `general_sales.csv` (general).
  - **Upload flow:** the files went through the real Upload page. "Confirm & Analyze" was disabled until the target was chosen.
  - **Pages and buttons:** pages 04–07 were visited, and the optimizer buttons were pressed.
  - **Result:** 12 page visits, 0 exceptions, 0 error boxes, 0 browser console errors. Both optimizer runs reported "Solver status: Solved · converged: yes".

## New tests: `tests/test_upgrade_part02.py` (148)

| Class | Tests | Covers |
|---|---|---|
| TestRoleEngine | 31 | Roles and mode for all 11 fixtures; confidence / evidence / alternatives on every column; alternative names (Revenue / Sales_Value / Net_Sales / Total_Revenue, TV_Investment / Television_Cost, FB_Reach / Meta_Views, Units_Sold / Quantity); the first numeric column never assumed to be revenue; multi-target confirmation; ambiguity warnings; aggregate and derived columns (demo); the units × price relationship; tokenizer; Excel input |
| TestTradeSpendRule | 11 | Trade_Spend as a business variable; the name rule; manual correction to a channel; both legacy detectors skip it; a confirmed override still reaches the pipeline |
| TestMappingPersistence | 5 | Save / load / version / history; a stale structure or schema is rejected; dtype-independent fingerprint; targets must be outcomes; legacy bridge (a unit target is never revenue) |
| TestCapabilities | 4 | Exposure mode never offers ROI / money; spend mode with a money target; spend mode with a unit target (no ROI); general mode explains what's missing |
| TestStats | 3 | HC3 against the textbook sandwich formula; VIF; deterministic permutation importance |
| TestContextAndGrain | 3 | Exposure grain detected and counted once; no leakage from other outcomes or derived columns; a non-outcome target is rejected |
| TestEffectiveness | 6 | Kaggle-structure replica, both targets independently (six exposure drivers, shares = 100%, no ROI); spend-mode ROI only for money targets; general-mode business drivers; promotions, segments, time; an unusable model is reported |
| TestChannelPerformance | 3 | Shares within the same unit, no money in exposure mode; ROAS ranking in spend mode; no channels explained |
| TestExposureOptimizer | 4 | Only same-unit groups of 2+; total and limits respected and the result not worse than current; infeasible limits refused with no fake solution; a weak model blocks optimisation |
| TestScenarios | 5 | Actual vs model baseline kept separate; extrapolation and reliability flags; promotion-share scenarios; MMM cross-check; weak model flagged |
| TestDefaultLimits | 31 | Feasibility for 1–60 channels × 3 totals; the reported one-channel case; no fixed 55%; a dominant channel isn't forced down; refresh vs manual vs reset; feasibility messages; no 0.55 left in the UI |
| TestModelPages | 17 | Pages 04–07 in spend (demo), exposure (replica upload) and general modes; every chart has a title and both axis titles; no ROI/ROAS value, column or chart in exposure mode, and every mention is a negation; target selector; unconfirmed target blocks the models until it's saved (v2); previous-module switch; currency follows the dataset (EUR, none); daily data |
| TestOptimizerPages | 7 | 1 / 2 / 4 channels: feasible defaults, Solved, objective shown; budget change refreshes defaults but keeps typed limits; infeasible typed limits explained, not changed, button disabled, reset works; page 15 one-channel regression; exposure optimizer page |
| TestScenarioPage | 1 | Add and remove a scenario in the real page |
| TestLegacyIntegration | 3 | The full pipeline on the replica gives no channel ROI; the AI analyst gives no channel ROI figures; API upload / confirm gives no Trade_Spend channel |
| TestRobustness | 11 | Missing values; duplicate rows; 5 date formats; Benjamini-Hochberg matches scipy; minimum-sample gate; click-through rate only when both measures exist; no target leakage |
| TestRealKaggleFile | 3 | **Skipped**: needs the real file |

## Changes to existing tests

- **`tests/test_upgrade_part01.py::TestInterimPages` (4 parametrized cases).**
  - **Removed:** the assertion that pages 04–07 show "*X*: not implemented yet … PART 02", because PART 02 implements them.
  - **Replaced by:** the opposite assertion (no "not implemented yet" text) plus a check that the model frame is shown.
  - **Kept unchanged:** the previous-module switch assertions (off by default; when switched on, the old module renders labelled "Previous module · *X* · not the *new model*").
- **No other existing test was changed, removed or weakened.**

## Earlier full runs on the working copy (for transparency)

- **First run (before the new test file was written): 3,145 passed, 5 failed.**
  - **4 failures:** the interim-page cases above, expected.
  - **1 failure:** a PART 01 rule that flags a menu label written as a literal. My new page docstrings began with `"""04 · …`, so I reworded them.
- **A second run: 3 failures**, because the files were being edited while it ran (a page used a field the already-imported engine didn't yet have). The same tests passed when rerun on their own, and the final run above passed.

---

# PART 03 test report

## Final run: on the exact packaged snapshot

| Item | Value |
|---|---|
| Where | Fresh extraction of the PART 03 package into an empty directory; new virtual environment; `pip install -r requirements.txt -r requirements-optional.txt pytest` (xgboost-cpu 3.4.1, lightgbm 4.7.0, catboost 1.2.10 installed and imported) |
| **Collected** | **3,356** |
| **Passed** | **3,353** |
| **Failed** | **0** |
| **Skipped** | **3**: the real Kaggle file is not provided (as in PART 02) |
| Not run | 0 |
| Warnings | 1: FastAPI/Starlette TestClient deprecation (as in the baseline) |
| Duration | 695 s |
| Log | `PART03_TEST_LOG.txt` |

## New tests: `tests/test_upgrade_part03.py`

**TestRegistry**
- the 12 models and the defaults;
- optional packages detected when installed;
- **each of the 12 fits, predicts and gives valid probabilities on binary and multiclass data (24 cases);**
- a missing package is reported and never substituted (engine);
- the requirements files.

**TestEvaluation**
- every model gets its own holdout row, with all metrics and times;
- selection is by CV;
- **flipping every test label changes no CV score and not the choice, and every holdout accuracy becomes 1 − original;**
- metrics match `sklearn.metrics`;
- multiclass averaging and one-vs-rest ROC;
- ROC and log loss appear only when valid;
- weighted metrics from a confusion matrix;
- the imbalance option;
- regression keeps its models and gets per-model R² / MAE / MSE / RMSE / MAPE;
- classifiers are never forced onto continuous targets;
- MAPE is undefined with zero actuals;
- the original three models are unchanged.

**TestZTestPage**
- blank σ gives guidance plus the engine's explicit refusal, with no statistic;
- the switch to the t-test;
- σ of 0, −5 or "abc" is rejected;
- for all three alternatives: the formula and calculation, Z equal to an independent computation, the conclusion at α = 0.01, and the plain-language text;
- the sample SD is never substituted.

**TestStatisticsCharts**
- every inferential chart (10 tests) and every descriptive chart (8 types) has a title and both axis titles;
- the descriptive measures are present;
- the tab opens on the t-test.

**TestPredictivePage**
- the defaults, Select all (12), Run;
- the 10 required charts present and labelled;
- 13 holdout rows, all ok;
- an unavailable model is shown as a warning and an error, not hidden;
- a continuous target gets the regression explanation.

## Existing tests changed in PART 03 (requirement-driven)

| Test | Before | Now | Why |
|---|---|---|---|
| `test_phase8_part4_audit.py::test_heldout_scored_once_for_classification` | the test rows are scored by exactly one model | each model is scored on the test rows exactly once; nothing else sees them; model rows are unique | Your PART 03 requirement: "Every selected model must receive its own independent holdout evaluation." Selection independence is proven by the new label-flip test. |
| `test_phase8_part4_engine.py::test_heldout_evaluated_once_no_test_rows_in_oof_or_selection` | same, for regression | same | same |

**No other existing test was changed, and none was removed or skipped.** The first PART 03 full run found 23 failures:

- **21 were caused by my changes**, and I fixed the code rather than the tests:
  - I restored the "delivered in Part *n* of Phase 8" banners, which existing tests assert.
  - A blank σ again produces the engine's structured refusal, alongside the new guidance.
- **The other 2 are the protocol tests above.**

---

# PART 04 test report

## Final run: on the exact packaged snapshot

| Item | Value |
|---|---|
| Where | Fresh extraction of the PART 04 package; new virtual environment; `pip install -r requirements.txt -r requirements-optional.txt pytest` |
| **Collected** | **3,403** |
| **Passed** | **3,400** |
| **Failed** | **0** |
| **Skipped** | **3**: the real Kaggle file is not provided |
| Not run | 0 |
| Warnings | 1: FastAPI/Starlette TestClient deprecation |
| Duration | 776 s |
| Log | `PART04_TEST_LOG.txt` |

## New tests: `tests/test_upgrade_part04.py` (47)

**TestDiagnose (13)**
- feasible case;
- maximums too low: the exact message, four remedies, exact amounts;
- remedy D not applicable without extra channels;
- minimums too high: A and B applicable, C and D not;
- a channel's maximum below its minimum;
- the reserve makes the maximum side feasible and says how much stays unallocated;
- inputs never modified;
- the 5-status classification (7 cases).

**TestEngine (18)**
- 1, 2 and 4 channels: Optimal, budget used = total, utilisation;
- infeasible maximums and infeasible minimums: no fabricated allocation;
- reserve: unallocated = total − used, never more than the total, can't fix minimums that are too high;
- totals at 70%, 100% and 130%;
- profit objective;
- solver exception gives "Feasible but not proven optimal"; solver returning nothing gives "Failed"; non-convergence is never "Optimal";
- an unfitted model gives "Not supported";
- missing spend values;
- no ROAS when the channels aren't money;
- binding flags.

**TestExposureOptimizerStatus (1)**
- optimal or not proven / not supported / infeasible.

**TestScenarios (3)**
- budget change and multiplicative combination;
- one-at-a-time sensitivity: sorted, flags as shares, price sign;
- no model, so no sensitivity.

**TestOptimizerPage (6)**
- **Result panel:** status, solver, objective, total / used / unallocated / feasibility, utilisation chart and table, every chart labelled.
- **Remedies:**
  - infeasible limits show A–C without changing the inputs;
  - Apply B sets the total;
  - Apply A raises the maximums and marks them "set by you";
  - Apply C plus Run reports the exact unallocated amount;
  - Apply D adds back a deselected channel.
- **General data:** reported as not supported.

**TestPrescriptivePageModes (2)**
- **Exposure columns:**
  - mixed units are refused as "Not supported";
  - in a single unit they're labelled "not money", get no profit objective and no ROAS, and the totals are in impressions.
- **Spend data:** keeps the profit objective.

**TestScenarioPage (2)**
- **Spend mode:** a total-budget input, a labelled sensitivity chart, and a budget scenario added.
- **Exposure mode:** no budget input, but the sensitivity chart is there.

## Existing tests changed in PART 04

My own PART 02 tests, updated for the more specific feasibility wording:

- **Feasible messages:** `startswith("Feasible")` became `"Feasible:" in …`.
- **Infeasible messages:** "maximums add up to only 4.00" became "add up to only 4.00" or "is below the minimum". With every maximum set to 1.0, each one is below its minimum, which is now diagnosed first.
- **The page-15 total label** now carries the currency unit.

No test from the baseline or from PART 01 was changed.

---

# PART 05: final regression

The full details are in `FINAL_TEST_REPORT.md`, and the complete verbose output is in
`FINAL_TEST_EXECUTION_LOG.txt`.

**Run.** A fresh extraction of the PART 05 package, a brand-new virtual environment (core and
optional requirements), Python 3.13.18, pytest 9.1.1, `python -I -m pytest tests -v -rs`.

| Collected | Passed | Failed | Errors | Skipped | Not run | Warnings |
|---|---|---|---|---|---|---|
| 3,449 | 3,446 | 0 | 0 | 3 (real Kaggle file not provided) | 0 | 1 (FastAPI/Starlette TestClient deprecation) |

## New tests: `tests/test_upgrade_part05.py` (46)

- 18 tests on the demo (INR): every page and tab has labelled charts, no exception, no fallback.
- 18 tests on an unknown-currency upload: the same checks, plus no INR, USD, ₹ or $ anywhere.
- 3 explicit label tests: the Forecast chart on the demo and on an unknown-currency upload, and
  Driver Importance.
- 7 security tests: HTML in a column name is escaped on 7 pages.

No existing test was changed in PART 05.

---

# PART 06: final verified regression

The full details are in `FINAL_TEST_REPORT.md`; the complete output is in `FINAL_TEST_EXECUTION_LOG.txt`.

| Collected | Passed | Failed | Errors | Skipped | Not run | Warnings |
|---|---|---|---|---|---|---|
| 3,522 | 3,522 | 0 | 0 | 0 | 0 | 0 |

**New file:** `tests/test_upgrade_part06.py` (73 tests):

- real-file shape, quality, 22 role checks, exposures, mode, targets, legacy channels;
- capabilities and models for both targets, with optimiser bounds;
- page target switching and ROI explanations on pages 04–07;
- quality-score roles, and existing roles unchanged;
- Z-test textbook values, refusal without σ, no sample-SD substitution, and the UI CI line;
- API security headers, docs off in production, truncation reported, upload cleanup, CSV-injection escaping;
- the optimiser default-limit matrix (1/2/4 channels × 0.5/1/2 budgets) and infeasible cases.

**Changed existing tests** (Windows compatibility only; same assertions):

- explicit UTF-8 reads in 18 files;
- the `os.geteuid` skip guard;
- the bash-only entrypoint tests are skipped on Windows.
