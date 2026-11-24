# MarketPulse AI Upgrade: Changelog

This changelog covers the four-part upgrade defined in `IMPLEMENTATION_PLAN.md`. Earlier releases are described in `FINAL_CHANGELOG.md`.

**Baseline:** `MarketPulse_AI_ULTIMATE_FINAL_RELEASE.zip`, SHA-256 `fd46e3c6…8515c`. The PART 00 backup is `MarketPulse_AI_PART00_BASELINE_BACKUP.zip`, SHA-256 `659ab2a0…6898`.

---

## PART 01: Navigation, Statistics workspace, chart labelling, currency detection

### Navigation (approved structure)

- **New `dashboard/navigation.py`.** A single registry of page keys, labels, order, menu numbers, group headings, Statistics tabs and kept-but-unlisted modules.
  - `app.py` renders `navigation.PAGES`. No page label is written as a literal anywhere else, and a test enforces this.
  - The sidebar group headings are derived from the registry, so renumbering can't misplace them. The baseline used fixed list positions.
- **Approved menu:**

  | # | Page | # | Page |
  |---|---|---|---|
  | 00 | Upload Data | 08 | Forecast |
  | 01 | Statistics | 09 | AI Marketing Analyst |
  | 02 | Executive Overview | 10 | Recommendations |
  | 03 | Marketing Performance | 11 | Reports |
  | 04 | Marketing Effectiveness Model | 12 | Predictions |
  | 05 | Channel Performance & Contribution | 13 | Anomalies & Drivers |
  | 06 | Marketing Resource Optimizer | 14 | Decision Intelligence |
  | 07 | Marketing Scenario Simulator | 15 | Prescriptive Analytics |

- **Removed from the menu but kept unchanged:**
  - **modules:** the Marketing Mix Model, Channel ROI, Budget Optimizer and What-If Simulator pages;
  - **standalone statistics pages:** 14 Descriptive, 15 Inferential and 16 Predictive.

  Their engines, the API (`/optimize-budget`, `/simulate-budget`) and the AI analyst are untouched.
- **Pages 04–07 are transition pages.** Each one:
  - states "*<Module>*: not implemented yet. This module is built in PART 02 … No result on this page comes from it.";
  - lists what the module will provide;
  - keeps the previous module reachable behind a switch that is off by default, labelled "Previous module · *X* · not the *new module*". The previous module only runs after the user switches it on.
- **Other navigation details:**
  - A stale menu label left in a live session falls back safely to page 00.
  - Links in the app text to removed pages ("see **Channel ROI**", "(module 17)") now point to the new pages.
  - The page badge shows the menu position ("PAGE 15"), not the internal module identity ("MODULE 17").

### Statistics workspace

- **New `01 · Statistics` page** with three separated tabs:
  - 📊 **Descriptive Statistics**
  - 🔬 **Inferential Statistics**
  - 🔮 **Predictive Analysis**
- **The tabs reuse the existing modules unchanged**, so all statistical functionality is preserved. Tabs are *lazy* (`st.tabs(..., on_change="rerun")`): only the open tab runs, and a one-shot request can open a specific tab.
- `render_module_page(..., embedded=True)` draws each module inside its tab with its own heading, overview, detected-data summary and availability gate.
- The module title "Predictive Analytics" is now "Predictive Analysis", as you requested. It also no longer clashes with page 12 "Predictions".

### Chart standard and Executive Overview

- **`apply_chart_style` gains `x_title` and `y_title`.** Its currency tick prefix is now the active dataset's currency; the baseline hard-coded `₹`. `stats_ui.apply_dark_layout` passes axis titles through.
- **New `dashboard/components/chart_labels.py`.** Axis titles reflect what is actually plotted:
  - the readable variable name;
  - the aggregation really applied (Total, Average, or Number of records);
  - units only when known: `(%)` for percentage or rate columns, and the currency only for monetary columns of a dataset whose currency is known (never repeated when already in the name);
  - time axes named after their real frequency (Date, Week, Month, Quarter or Year);
  - category axes named after the category;
  - chart titles that state the aggregation ("Average Net Sales by Region").
- **Executive Overview:** every chart is labelled:
  - **demo (5):** revenue and spend trends, the contribution donut (legend title), budget distribution, and current vs optimized;
  - **uploads:** all 9 recommender chart types (line, area, bar, horizontal bar, scatter, histogram, donut, box, heatmap).
- **Pages that reuse these components** are labelled the same way: Marketing Performance (14 figures, none labelled before), Predictions, the Predictive and Prescriptive statistics charts, and the fallback trend chart.
- **Trend charts of row-level uploads** now draw one point per date: values are summed per date, and the label says "Total …". Before, a line was drawn through every row. This fix also covers the Marketing Performance page and the fallback chart.
- **Monetary KPI cards** for uploads show the dataset currency only for sums and averages of monetary columns. Counts, rates and percentages never get a currency.

### Currency detection (new)

- **New `src/data_engine/currency.py`** detects the currency from column names (`Revenue (INR)`, `amount_usd`) and raw cell text (`₹1,200`, `$12`, `EUR 300`). It covers 37 ISO currencies.
  - A shared symbol (`$`, `¥`, `Rs`) keeps the symbol but not the code.
  - Conflicting evidence results in no currency.
  - No evidence results in no currency; INR is never assumed for uploads.
- **Upload Data** has a new step, **"2b · Currency"**. It shows the detected result with its evidence. The user can keep the automatic result, choose any of the 37 currencies, or choose "None / not monetary". The choice is saved as `currency.json` in the dataset's processed folder, together with the monetary columns.
- **Display** is handled by the new `dashboard/components/currency.py`:
  - KPI cards, tables, chart ticks, Reports and the AI analyst use the active currency;
  - the demo keeps its INR configuration, with output identical to before (`₹2.22Cr` style);
  - when the currency is unknown, no symbol is shown.
- **AI analyst:** its channel answers for uploaded data no longer use "₹" or "per rupee". This was a baseline behaviour that assumed INR for every upload. The demo still uses INR.

### Pre-existing defects found and fixed

| Defect | Fix |
|---|---|
| **Recommendations** never rendered on the demo: `pd` was used without being imported. `safe_page` hid the `NameError` as "not available for this dataset", and the existing tests only checked for exceptions. | Imported pandas. New tests now fail on any exception name inside a fallback message. |
| Executive Overview demo insight said "incremental revenue per ₹1", a causal claim. | Now "model-estimated revenue per ₹1 spent (an association, not proof of cause)". |

### Tests

- **New:** `tests/test_upgrade_part01.py`, covering navigation, Statistics tabs, transition pages, axis labels (helpers, every chart type, demo and uploads), currency detection, display and switching, the analyst's currency, the Upload page currency step, and the Forecast regression. It also adds `tests/helpers.goto_page()` and `goto_module()`.
- **Existing tests adapted** to the approved menu; their assertions are unchanged unless stated:
  - `test_phase8_part1_foundation.py`: the structure tests that pinned the old 00–13 and 14–17 layout now assert the approved menu exactly (stronger), check that modules removed from the menu are kept, and navigate to modules through their tab.
  - `test_phase8_part2/3/4_dashboard.py`, `test_phase8_part4_workflow.py`, `test_phase8_part6_integration.py`: navigation lines now use `goto_module`. The title assertions also accept the tab heading.
  - `test_final_dashboard_pages.py`, `test_ultimate_ui.py`, `test_phase7_part7_dashboard_copilot.py`: labels now come from the registry instead of literals.

### Not changed

- Forecast module (`7_forecast.py`, `src/forecasting`, `src/prediction/forecasting_engine.py`, `ts_evaluation.py`).
- Model artifacts (`models/`, `data/processed/`).
- Security (`api/main.py`, `auth.py`, `config/settings.py`, Docker and `deploy/`).
- Calculation engines.
- `requirements.txt`.

Of the 85 protected files, only `dashboard/pages/0_upload_data.py` changed: its two label constants and the currency step.

---

## PART 02: Universal dataset engine, three analytics modes, four new marketing models, optimizer default-limit fix

### Universal dataset engine (new: `src/data_engine/role_engine.py`, `role_lexicon.py`)

- **Roles:** a business role for every column, from names, types, values, statistics and relationships between columns. The role groups are outcomes, channel ad spend, total spend, exposure, business variables (including **trade / promotion spend**), dimensions, time, identifiers and derived columns.
- **Per column:** a confidence score, evidence, alternatives and ambiguity / missing-role warnings.
- **Relationships checked:**
  - value ≈ units × price;
  - aggregate spend columns, which are never treated as channels;
  - outcome totals;
  - duplicate columns.
- **Target confirmation:** when several columns could be the target, none is chosen. Upload and the model pages wait for the user. More than one target can be confirmed and analysed independently.
- **Corrections and storage:** users can correct any role in an editable table. Mappings are saved as versioned `role_mapping.json` files (schema version, fingerprint, history).
- **Automatic modes:** Spend-Based, Exposure-Based or General Business Analytics, with the evidence shown.

### Upload page (`dashboard/pages/0_upload_data.py`, protected; documented change)

- **Step 2 is now "Column roles and target"** (the universal mapping). It replaces the earlier role widgets and covers the same roles (date, revenue/target, channels) plus every other role.
- **The existing pipeline is fed through the same overrides as before:**
  - only advertising spend becomes channel columns;
  - a unit target is never passed on as revenue.
- **"Confirm & Analyze"** is disabled until an ambiguous target is confirmed.
- **Currency:** the confirmed mapping is saved, and the money columns from the mapping now feed the currency record.

### Trade / promotion spend rule (protected: `src/data_engine/column_classifier.py`, `src/marketing/field_discovery.py`; documented change)

- **Both legacy channel detectors** now skip trade / promotion spend names (Trade_Spend, Promo_Budget, Coupon_Cost…) when *guessing* channels.
- **A channel the user confirms** still passes through.
- **Effect:**
  - a Trade_Spend column no longer receives channel ROI anywhere: pipeline, API, AI analyst or legacy pages;
  - real advertising channels are detected exactly as before.

### Four new models (pages 04–07)

The pages replace the PART 01 transition pages. The engines are in `src/marketing_intel/`.

- **04 · Marketing Effectiveness Model**
- **05 · Channel Performance & Contribution**
- **06 · Marketing Resource Optimizer**
- **07 · Marketing Scenario Simulator**

Methods, outputs and limits are in `PART02_MODEL_IMPLEMENTATION_REPORT.md`.

**Every page shows:**

- dataset name, analytics mode with its evidence;
- column roles and target;
- available and unavailable capabilities, with reasons and missing information;
- results, methodology, assumptions and warnings.

The previous modules stay reachable behind their labelled switches.

### Optimizer default-limit fix (`src/prescriptive/limits.py`, `dashboard/components/prescriptive_ui.py`)

- **No fixed 55% maximum.** Defaults are always feasible for the number of channels and the total.
- **Budget changes:** defaults follow the total when it changes; typed limits are never overwritten (marked "set by you"). Resetting is an explicit button.
- **Feasibility:** a `sum(min) ≤ total ≤ sum(max)` summary with plain explanations, and the button is disabled when infeasible.
- **Results:** solver status and objective are always shown.
- **Where:** applies to 15 · Prescriptive Analytics and to page 06 (spend and exposure modes). See `PART02_OPTIMIZER_FIX_REPORT.md`.

### Tests and fixtures

- **`tests/test_upgrade_part02.py`:** 148 tests (145 run, 3 skipped because the real Kaggle file isn't present).
- **`tests/fixtures/part02/`:** 11 seeded datasets and their generator.
- **Changed test files:**
  - `tests/test_upgrade_part01.py`: the interim-page test now asserts the opposite (no "not implemented yet"; the model frame is shown) and keeps its previous-module switch assertions;
  - no other existing test changed.

### Tools and docs

- **Tool:** `tools/part02_compat_report.py` (compatibility report for any CSV / XLSX).
- **Docs:**
  - `KAGGLE_DATASET_TEST_REPORT.md`
  - `PART02_MODEL_IMPLEMENTATION_REPORT.md`
  - `PART02_OPTIMIZER_FIX_REPORT.md`
  - `docs/part02/*.md`
- **Updated:** `DATASET_COMPATIBILITY_REPORT.md`, `PART_STATUS.md`, `IMPLEMENTATION_PLAN.md`, `TEST_REPORT.md`.

---

## PART 03: Statistics completion

### Z-test (`dashboard/components/inferential_ui.py`)

- **σ input:** the known population standard deviation is now a labelled **required** input.
  - **Blank σ:** guidance explains why σ must come from outside the data, with a one-click **Use the one-sample t-test instead**. The engine still records its explicit "was not run" report, so nothing is computed and nothing is substituted.
  - **σ ≤ 0 or not a number:** a validation error, and no statistic is computed.
- **Sample SD:** shown "for reference only — not used as σ". It is never substituted.
- **Inputs:** μ0, the alternative hypothesis and α are user inputs (as before).
- **Result:** the formula is shown, then the calculation with the actual numbers, the p-value formula for the chosen alternative, the Z-statistic, p-value, α and the conclusion, plus a plain-language explanation.
- **Default test:** the Inferential tab now opens on the one-sample t-test instead of the Z-test, so users no longer land on a refusal before they can enter σ.
- **Unchanged:** the engine (`src/inferential/tests.py`, protected).

### Chart labelling on the Statistics tabs

- **Inferential:** every chart has a descriptive title and named axes, for all 10 tests (before PART 03 they had no titles and generic "value"/"count" axes).
- **Descriptive:** the box, violin and correlation-heatmap charts now name both axes.

### Predictive Analysis: 12 classifiers

- **New `src/predictive/classifiers.py`:**
  - a registry of 12 classifiers, with factories, run-time availability checks for the optional packages, and imbalance handling;
  - an XGBoost label-encoding wrapper, a CatBoost 1-D prediction wrapper, and calibrated SVM probabilities;
  - weighted metrics, safe log loss and ROC curve points.
- **`src/predictive/procedure.py` (protected; additive):**
  - `make_estimator` routes the nine new keys, and the `balanced` option, to the registry;
  - `ScreenedModel` accepts `balanced`;
  - `fitted_signature` handles 2-D GradientBoosting estimators;
  - `internal_importance` returns empty for models with no built-in importance instead of failing.
- **`src/predictive/contracts.py` (protected; additive):**
  - `ALL_CLASSIFICATION_MODELS`, the new labels, and `Settings.balance_classes` (default False, so the original behaviour is unchanged);
  - new result fields for per-model holdout results, confusion matrices, ROC curves, class tables and importances.
- **`src/predictive/evaluate.py` (protected; additive):**
  - all 12 keys are allowed, and the defaults stay the original three;
  - an "unavailable" status is reported for missing optional packages;
  - the `balanced` setting is passed through;
  - **P5:** an independent holdout evaluation of every model that passed CV, with training and prediction times. The selected model's P4 evaluation is reused. Selection is still by CV only.
- **`dashboard/components/predictive_ui.py`:**
  - 12-model multiselect (defaults LR / DT / RF), **Select all available models** and **Back to the defaults**;
  - warnings for unavailable models, the imbalance option with its explanation, and an availability table;
  - the regression explanation for continuous targets;
  - a new **Independent Holdout Evaluation of Every Model** section with 10 labelled charts, with failures shown as errors.
- **New `requirements-optional.txt`:** xgboost-cpu, lightgbm, catboost.

### Tests

- **New: `tests/test_upgrade_part03.py`.**
- **Existing tests changed because PART 03 requirements changed** (details in `TEST_REPORT.md`):
  - **`test_phase8_part4_audit.py::test_heldout_scored_once_for_classification` and `test_phase8_part4_engine.py::test_heldout_evaluated_once_no_test_rows_in_oof_or_selection`:** they asserted that exactly one model is scored on the test rows. Your requirement is that every model gets its own holdout evaluation, so they now assert that each model is scored on the test rows exactly once and that nothing else sees them.

### Docs

`MODEL_VALIDATION_REPORT.md` (new).

---

## PART 04: Prescriptive analytics, optimizer feasibility and Scenario Simulator completion

### Feasibility engine (new `src/prescriptive/feasibility.py`)

- **Checks:** `sum(min) ≤ total ≤ sum(max)` and each channel's own `min ≤ max`, with a plain explanation that states the exact gap. When the maximums are too low, the message reads "The entered channel maximums cannot accommodate the full budget: …".
- **Remedies, each with the exact amounts:**
  - **A:** increase the maximums (or lower the minimums);
  - **B:** change the total budget;
  - **C:** explicitly allow an unallocated reserve (only when the maximums are too low);
  - **D:** add eligible channels (lists the deselected channels, or says there are none).
- **Applying a remedy:** only on an explicit **Apply A/B/C/D** click. Values set this way are marked "set by you". The inputs are never changed automatically.
- **Statuses:** **Optimal**, **Feasible but not proven optimal**, **Infeasible**, **Failed** and **Not supported**.
  - "Optimal" requires the solver to report convergence and every converged starting point to reach the same value (a confirmed local optimum; global optimality is not claimed for non-concave responses).
  - When the solver fails and the best feasible starting allocation is kept, the status is "Feasible but not proven optimal", never "Optimal".

### Engine (`src/prescriptive/engine.py`, protected; additive)

- **New parameters:** `allow_reserve` (the total becomes a ceiling; the unspent amount is reported) and `monetary` (False suppresses the ROAS-style column when the channels are not money).
- **Per-start log:** `_solve` records each starting point (an optional `log` argument), and supports the outcome objective within a total range, which the reserve uses.
- **New result fields:** `requested_total`, `budget_used`, `unallocated`, `allow_reserve`, `monetary`, `solution_status` (+ text), `utilization` (per-channel min / recommended / max, share of max, position in range, binding) and `feasibility`.
- **Unchanged:** existing behaviour with default arguments.

### UI

- **`prescriptive_ui.constraint_panel` (used by page 15 and page 06):**
  - objective (profit only when the channels are money), total, reserve option, limits (PART 02 defaults);
  - a feasibility status with the remedies;
  - a result panel with solution status, solver status and convergence agreement, the objective, and total budget / budget used / unallocated / feasibility;
  - a constraint-utilisation chart and table.
- **Page 15 · Prescriptive Analytics:** reads the dataset's role mapping.
  - Exposure columns are labelled in their own unit, never as money (no profit objective, no ROAS).
  - Columns in different units are refused as **Not supported**, with the reason.
  - Generic numeric resources behave as before.
- **Page 06, spend mode:** a channel selection (enables remedy D) and the full panel.
- **Page 06, exposure mode:** the same diagnostics in exposure units (no reserve, which is a budget concept) and the same solution statuses.
- **Page 07 · Scenario Simulator:**
  - a **Total marketing budget % change** input (spend mode, keeps the mix, combined multiplicatively with per-channel changes);
  - **one-at-a-time sensitivity analysis** (±5 / 10 / 20%; promotion flags move by points of records) with a labelled tornado chart and table.

### Tests

**New: `tests/test_upgrade_part04.py`** (47).

**Wording assertions updated in `tests/test_upgrade_part02.py`** (my own PART 02 tests; the new feasibility messages are more specific):

- `Feasible…` → `Feasible:`
- "add up to only" or "is below the minimum"
- total-budget labels now carry the currency unit

**No other existing test changed.**

---

## PART 05: Final integration, audits and release

### Code changes (all small, all tested)

| File | Change | Reason |
|---|---|---|
| `dashboard/pages/7_forecast.py` (**protected**) | Forecast chart: title "‹target›: history and forecast", a time-axis title from the date spacing, a y-axis title with the active currency unit (none when unknown), and the legend moved below the chart | PART 05 §5.3 chart audit. The chart had no title and no x-axis title, and its y-axis had no unit (a PART 01 limitation scheduled for this pass). Only the `st.plotly_chart` call changed; the data, model and evaluation code are untouched. |
| `dashboard/pages/12_anomalies_drivers.py` | "Driver Importance" chart: axis titles "Importance score" / "Driver" | The chart audit found both axis titles missing |
| `dashboard/components/model_page.py`, `src/data_engine/role_engine.py` | `hashlib.sha1(..., usedforsecurity=False)` | bandit B324 (high). These hashes are cache and duplicate keys, not security uses. |

### New tests: `tests/test_upgrade_part05.py` (46)

- Every page and Statistics tab is rendered for the demo (INR) and for an unknown-currency
  upload. Each run checks for no exception, no safe-page fallback, a title on every chart, both
  axis titles (pie, indicator, table and sunburst charts excepted), and no foreign currency.
- Forecast and Driver Importance labels are checked explicitly.
- **Security:** HTML in a column name (`<img src=x onerror=…>`) is escaped on 7 pages.

### Tools (not used at runtime): `tools/part05/`

`audit_pages.py` (AppTest audit of all pages on 4 datasets), `browser_all.py` (Chrome pass),
`perf.py`, `perf_rfa.py` (performance and profiling).

### Audits and documentation

- `SECURITY_AUDIT_REPORT.md`: bandit, pip-audit (0 known vulnerabilities), a review of the controls.
- `PERFORMANCE_REPORT.md`: start-up, page times, 10k and 100k rows, profiling against the baseline.
- `DEPLOYMENT_READINESS_REPORT.md`: Docker checks and what is not verified.
- Final reports: `FINAL_TEST_REPORT.md`, `FINAL_TEST_EXECUTION_LOG.txt`, `FINAL_PROJECT_STATUS.md`,
  `FINAL_FILE_MANIFEST.txt`, `SHA256_CHECKSUM.txt`.
- New docs: `docs/INSTALLATION.md`, `docs/USER_GUIDE_UPGRADE.md`, `docs/ARCHITECTURE_UPGRADE.md`.
- Updated: `README.md` (upgrade section, optional requirements, limitations),
  `DATASET_COMPATIBILITY_REPORT.md`, `MODEL_VALIDATION_REPORT.md`, `PART_STATUS.md`,
  `FINAL_CHANGELOG.md`, `TEST_REPORT.md`.

### Not changed

- No model, metric, optimiser or selection logic.
- No API code (`api/main.py` is byte-identical to the baseline).
- No Dockerfile or compose file.
- No security control.

---

## PART 06: final completion and verification

### Defect fixed by testing the user's real Kaggle file
| File | Change | Reason |
|---|---|---|
| `src/data_engine/role_engine.py` | New role `quality_score` (business group). Names with score/rating/quality/sentiment/NPS/CSAT/stars map to it and can never be read as channel spend or exposure. | `FB_Banner_Content_Score` and `IG_Banner_Content_Score` in the real file were read as Facebook/Instagram **ad spend** (confidence 0.45), which flipped the file to Spend mode, where ROI would have been fabricated. The 3 real-file tests failed before the fix. |
| `src/marketing_intel/context.py`, `response.py`, `capabilities.py` | `quality_score` added to the business-driver role lists | Content scores enter the models as business drivers |

### Security (local penetration-style tests: 6 findings → 0; see `SECURITY_AUDIT_REPORT.md`)
| File | Change |
|---|---|
| `api/main.py` (**protected**) | (a) Security headers on every response (`nosniff`, `DENY`, `no-referrer`, `no-store`). (b) `/docs`, `/redoc` and `/openapi.json` are disabled in production. (c) Uploads over `MARKETPULSE_MAX_ROWS` are still limited, but the response now reports `rows_received`, `rows_used`, `rows_truncated` and `warnings` instead of truncating silently. (d) Unparseable uploads are deleted from disk. Additive only: no endpoint, parameter or existing response field changed. |
| `dashboard/pages/marketing_effectiveness.py`, `channel_performance.py`, `scenario_simulator.py` | CSV downloads also escape formula-like **cells** (`dataframe_to_csv_bytes`), not only headers. Fixes CSV formula injection through column names (demonstrated). |

### Performance (identical results, verified on all output files)
- `src/prediction/classification_engine.py`, `regression_engine.py`: random forests fit in parallel and predict
  sequentially, so predictions are bit-identical.
- `src/prediction/segmentation_engine.py`: the exact silhouette uses parallel distance chunks with a 256 MB working
  chunk.
- `src/anomaly/time_series.py`: the spike/drop loop uses positional arrays (about 730k `.loc` calls removed at
  100k rows).

Result for *Confirm & Analyze*:

| Rows | Before | After | Peak memory before → after |
|---|---|---|---|
| 10k | 26.7 s | 21.5 s | 978 → 732 MB |
| 50k | 123.1 s | 104.4 s | 1,294 → 786 MB |
| 100k | 291.0 s | 255.4 s | 1,331 → 904 MB |

All 17 output files per size were compared after the change, with 0 differences.

### Docker and optional ML
| File | Change |
|---|---|
| `Dockerfile` (**protected**) | Opt-in build argument `INSTALL_OPTIONAL_ML` (default `false`, so the default image is unchanged). When `true`, it installs `libgomp1` and `requirements-optional.txt`. |
| `docker-compose.yml` (**protected**) | Passes `INSTALL_OPTIONAL_ML` (default `false`) to the dashboard build |
| `.env.example` | Documents `INSTALL_OPTIONAL_ML` |
| `.gitattributes` (new) | Keeps `*.sh` and Dockerfiles LF on Windows Git checkouts (a CRLF entrypoint breaks the containers) |

### Windows
| File | Change |
|---|---|
| `scripts/windows/*.ps1` (new, 6 files) | Setup, start dashboard/API, run tests, and the one-command `validate_windows.ps1`. ASCII-only, so they work in PowerShell 5.1 and 7. |
| `tools/platform_smoke.py` (new) | Cross-platform smoke checks used by the validator |
| 18 test files (see below) | Explicit `encoding="utf-8"` when tests read project files, plus POSIX-only guards |

### Dependencies
`requirements.txt`: added `httpx2>=2.16.2`. Starlette 1.x's TestClient prefers it, and it removes the single
deprecation warning seen in every earlier test run. `httpx` is kept.

### Tests
- **New:** `tests/test_upgrade_part06.py` (73 tests) covers the real file (roles, mode, targets, no money, models,
  optimiser bounds, page target switching), quality-score roles, Z-test textbook values and UI, the API security
  fixes, CSV injection, and the optimiser default-limit matrix (1/2/4 channels × 3 budgets, plus infeasible cases).
- **The real file is now included** at `tests/fixtures/part02/real/`, so `TestRealKaggleFile` (3 tests) **runs
  instead of being skipped**.
- **Existing tests changed** for Windows compatibility only; no assertion was weakened:
  - `.read_text()` → `.read_text(encoding="utf-8")` in 18 test files (65 call sites).
  - `test_ultimate_security.py`: the root-only skip condition no longer calls `os.geteuid` on Windows, where it
    does not exist and would crash collection. `text=True` subprocess output decodes as UTF-8.
  - `test_docker_entrypoint.py::TestSentinelLogic`: skipped on Windows or when `bash` is absent. The entrypoint is a
    bash script that runs only in Linux containers.

  On Linux the same tests run and pass, so the counts are unchanged.

### Documentation
- **New:** `INSTALLATION.md`, `USER_GUIDE.md`, `WINDOWS_SETUP_GUIDE.md`, `DOCKER_SETUP_GUIDE.md`,
  `KNOWN_LIMITATIONS.md`. The `docs/` copies now point to these.
- **Updated:** `README.md`, the final reports, `DATASET_COMPATIBILITY_REPORT.md` (real-file section),
  `KAGGLE_DATASET_TEST_REPORT.md`, `MODEL_VALIDATION_REPORT.md` §7, `SECURITY_AUDIT_REPORT.md`,
  `PERFORMANCE_REPORT.md`, `DEPLOYMENT_READINESS_REPORT.md`, `PART_STATUS.md`.

### Protected baseline files changed in total (PARTs 01–06): 11 of 85
- PARTs 01–05 (8): `0_upload_data.py`, `7_forecast.py`, `column_classifier.py`, `field_discovery.py`,
  `predictive/contracts.py`, `predictive/evaluate.py`, `predictive/procedure.py`, `prescriptive/engine.py`.
- PART 06 (3): `api/main.py`, `Dockerfile`, `docker-compose.yml`.

The other 74 are byte-identical to the baseline.
