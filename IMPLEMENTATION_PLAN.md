# MarketPulse AI Upgrade: Revised Implementation Roadmap (v2: four parts)

> **Status:** approved (four parts), with your decisions recorded: the optimizer default limits are fixed in PART 02, and the PART 01 working-copy edits are kept. Progress is in `PART_STATUS.md`.

This roadmap supersedes the 11-part plan in `IMPLEMENTATION_PLAN.md`.

- **The scope is unchanged.** Every requirement from the master prompt and the approval message is kept; tasks are only regrouped.
- **Baseline:** PART 00 (complete). The backup is `MarketPulse_AI_PART00_BASELINE_BACKUP.zip`, SHA-256 `659ab2a0b6584e40913b4a6965e888830130efa055c70f6a77fbd031ae696898`, with 3,030/3,030 tests passing.

## Rules that apply to every part

1. Inspect before changing. Fix root causes, not symptoms.
2. Work in the isolated working copy. The baseline stays read-only.
3. Preserve all features, APIs, integrations, model artifacts and the Forecast module. Changes are additive and backward compatible. Nothing existing is deleted.
4. No fabricated analytics, metrics, ROI, units or predictions. Unavailable functionality is explained, not hidden.
5. **End of each part:**
   - run that part's tests, then the full regression suite, and report passed, failed, skipped, not run and platform limits;
   - review the changed files;
   - compare protected-file checksums against `PROTECTED_FILES_SHA256.txt`, justifying every change;
   - build a backup ZIP, integrity-test it, record its SHA-256, and attach it to the conversation;
   - update the documents listed for that part;
   - **stop and wait for authorisation.**
6. A part is complete only when implementation, tests, documentation and backup are all done.

---

## PART 01: Navigation, unified Statistics page, Executive Overview charts, currency detection

### A. Navigation restructuring (master prompt §4; approval §1, §3)

- Add a page registry (`dashboard/navigation.py`) as the single source of labels, order, numbering and group headings. Page labels are no longer duplicated as literals in code. Group headings are key-based, so renumbering can't misplace them.
- **Approved menu:**

  | # | Page | | # | Page |
  |---|---|---|---|---|
  | 00 | Upload Data | | 08 | Forecast |
  | 01 | Statistics | | 09 | AI Marketing Analyst |
  | 02 | Executive Overview | | 10 | Recommendations |
  | 03 | Marketing Performance | | 11 | Reports |
  | 04 | Marketing Effectiveness Model | | 12 | Predictions |
  | 05 | Channel Performance & Contribution | | 13 | Anomalies & Drivers |
  | 06 | Marketing Resource Optimizer | | 14 | Decision Intelligence |
  | 07 | Marketing Scenario Simulator | | 15 | Prescriptive Analytics |

- Remove the four old models from the menu, but keep their source files, engines, APIs and AI-analyst integration.
- Until PART 02 delivers them, pages 04–07 show the module's purpose, which part delivers it, and the previous module's existing view, clearly labelled. Nothing working disappears, and nothing is presented as a new result.
- Update in-app text that points to removed pages: "see **Channel ROI**", "module 17" and similar.
- An old menu label kept in a live session falls back safely to page 00.

### B. Unified Statistics page with three tabs (approval §2)

- **01 · Statistics** has three clearly separated tabs: **Descriptive Statistics**, **Inferential Statistics** and **Predictive Analysis**.
- The tabs reuse the existing modules unchanged, so all existing statistical functionality is preserved.
- The tabs are lazy (Streamlit 1.67): only the open tab runs, so one tool's widgets never interfere with another's and no work is done for hidden tabs.
- Programmatic deep links can open a specific tab.
- Each tab keeps its overview, detected-data summary, availability gate and body. These are the "what it does, what it needs, what was detected" explanations.
- Content upgrades come in PART 03.

### C. Executive Overview chart corrections and the shared chart standard (master prompt §6, §8)

- Extend `apply_chart_style` with `x_title`, `y_title` and unit/currency arguments. Add one helper that builds axis labels from the variable actually plotted, its aggregation ("Total Net_Sales" vs "Average Net_Sales"), its unit and the detected currency.
- **Audit every Executive Overview chart, demo and upload, and give each:**
  - a descriptive title;
  - x and y labels;
  - units, with currency only when known (never assumed INR or USD);
  - meaningful date labels in chronological order at the correct frequency;
  - category names on category axes and sensible sorting;
  - `(%)` on percentages;
  - clear legends and tooltips;
  - no truncated labels;
  - responsive layout.

  Affected charts: Weekly Revenue and Spend trends, the contribution donut, budget distribution, Current vs Optimized, and all 9 chart types from the generic recommender (line, area, bar, horizontal bar, scatter, histogram, donut, box, heatmap).
- **Shared components reused on other pages** get the same fixes: Marketing Performance (14 figures, 0 axis titles today), the Predictions page, and the shared statistics chart helpers.
- **Charts on pages rebuilt later** are labelled in the part that builds them:
  - PART 02: pages 04–07;
  - PART 03: statistics charts;
  - PART 04: optimizer and simulator charts.

  PART 04 then runs an app-wide guard test over everything.

### D. Automatic currency detection (approval §7)

- **Detection** from column names (`Revenue (INR)`, `amount_usd`, `sales_eur`), currency symbols and ISO codes inside values (`$1,200`, `₹`, `€`, `£`, `¥`, `INR 500`), and the consistency of what is found.
- **Reliability rule:** a currency is reported only when the evidence is unambiguous. Conflicting evidence means no currency.
- **Manual override** on the Upload Data page: auto-detected, a chosen currency, or "none / not monetary".
- **Display:** when no currency can be identified, no symbol is shown.
- **Persistence:** the detected or chosen currency is stored with the dataset's metadata.
- **Formatting:** KPI cards, tables and chart axes use it. The hard-coded `₹` in 8 files is replaced by the detected currency. The demo keeps its INR configuration.
- Every uploaded dataset is handled without assuming INR.

### Tests (test-list letters from the master prompt)

- **A:** full regression.
- **W:** navigation:
  - exact approved menu;
  - every entry renders on demo and upload;
  - legacy modules importable;
  - forced jumps after upload;
  - group headings;
  - stale-label fallback;
  - tab deep-links;
  - lazy-tab isolation.
- **V:** axis labels. Every Executive Overview figure (demo, and uploads covering all 9 chart types) has a title and x/y titles; pies and donuts are exempt from axes. Aggregation wording; percentage axes.
- **K:** currency:
  - INR, USD, EUR, GBP and JPY detection from names and values;
  - mixed or conflicting evidence gives none;
  - non-monetary columns give none;
  - the override is persisted;
  - the demo stays INR;
  - no INR is assumed for uploads.
- **D / E:** CSV and XLSX uploads with the currency override.
- **F:** dataset switching keeps each dataset's currency.
- **X:** Forecast regression.

**Existing-test adaptations** (labels only, assertions kept): the 3 label-pinning files and the 14 navigation call sites in 6 statistics dashboard test files. The foundation file's structure tests are rewritten to assert the approved menu exactly.

### Documents

`PART_STATUS.md`, `CHANGELOG.md`, `TEST_REPORT.md`, and the PART 01 section of `DATASET_COMPATIBILITY_REPORT.md` (currency).

---

## PART 02: Four universal marketing models with Spend / Exposure / General modes

> **Status: implemented** (see `PART_STATUS.md`). As delivered:
> - **Scope:** this section plus the optimizer default-limit fix you approved and the PART 02 brief's additions, namely trade / promotion spend as a business variable, the real-Kaggle-file checks and independent multi-target analysis.
> - **Deviation 1:** the real Kaggle file wasn't provided, so its checks ran on a replica with the same structure.
> - **Deviation 2:** long-format channel columns are recognised as channel dimensions, not pivoted.
> - **Deviation 3:** fixtures 4 and 5 (e-commerce, FMCG) are covered by `general_sales.csv` and `kaggle_structure_replica.csv`.

### A. Universal dataset compatibility engine (master prompt §3)

- **Semantic mapping** reuses and extends the existing classifier. It considers column names, data types, actual values, statistical characteristics, unique values, date patterns, relationships between columns and business context.
- **It identifies:**
  - date, sales, revenue and target variables;
  - marketing channels and marketing spend;
  - impressions, reach, clicks and conversions, **per channel** (wide layout, e.g. `TV_Impressions`) and as long-format channel columns;
  - product, geography/region, brand, promotion and price columns;
  - other useful business variables.
- **Alternative target names** (Sales, Revenue, Turnover, Total_Revenue, Sales_Value, Net_Sales) are checked semantically and never assumed interchangeable.
- **Multiple plausible targets** produce an explicit "confirm the intended target" prompt.
- **Three analytical modes** are selected automatically from the detected data:
  - **A. Spend-Based**
  - **B. Exposure-Based**
  - **C. General Business Analytics**
- **Every model explains:**
  - available functionality;
  - missing requirements;
  - the method selected;
  - limitations;
  - why anything is unavailable (useful explanations instead of generic errors).
- **Aliases:** additive only. Existing upload, cleaning and mapping behaviour is preserved.

### B. 04 · Marketing Effectiveness Model (master prompt §2, module 1)

- **Inputs supported:**
  - marketing spend, impressions, reach, clicks, conversions, website traffic and campaign performance;
  - sales and revenue;
  - product, promotion, price, distribution and geographic variables;
  - other numeric business variables.
- **Analyses:**
  - marketing variable identification;
  - correlation analysis (Pearson and Spearman, with p-values and a multiple-testing correction);
  - statistical relationship analysis;
  - channel effectiveness ranking;
  - feature importance (permutation importance on held-out data);
  - regression-based effect estimates where appropriate (OLS with robust HC3 standard errors, standardised coefficients and a VIF multicollinearity check, using numpy and scipy only);
  - time-based effectiveness;
  - brand-wise, geo-wise and promotion-related effectiveness, with minimum-sample gates;
  - confidence intervals only where statistically justified.
- **Causation:** correlation is never presented as causation. Causal effects are reported only when an experimental design or defensible causal assumptions are present, and those assumptions are stated.

### C. 05 · Channel Performance & Contribution (master prompt §2, module 2)

- **Channels:** TV, Radio, YouTube, Facebook, Instagram, Search, Display, Print, Email, Affiliate, and any others detected automatically. Works even when channel spend is unavailable.
- **Analyses:**
  - performance ranking and exposure comparison;
  - impressions and reach analysis;
  - click-through and conversion analysis where available;
  - spend-based analysis where available;
  - revenue association;
  - contribution estimates with statistical qualifications;
  - trend comparison;
  - brand-wise and geographic channel comparison.
- **Without spend:** **no ROI or ROAS is ever invented**; exposure-based metrics are shown instead.

### D. 06 · Marketing Resource Optimizer: universal module (master prompt §2, module 3)

- **Allocation types supported:**
  - channel budget, marketing spend, campaign, promotion, product, geographic and inventory/resource allocation where relevant;
  - user-defined total budget, minimum and maximum allocations, and channel-specific constraints.
- **Objectives:** the user selects an objective, offered only when the data supports it; otherwise the module explains why.
- **Outputs:** allocation comparison, scenario comparison and expected outcome estimates.
- **Without channel spend:** no fake monetary recommendations. Exposure/resource allocation is offered only when meaningful inputs and constraints exist.
- **Engine:** built on the existing dataset-agnostic `src/prescriptive/engine.py`, which already validates constraints and refuses infeasible input, so PART 02 delivers a working, honest optimizer.

### E. 07 · Marketing Scenario Simulator: universal module (master prompt §2, module 4)

- **Inputs varied:** spend, impressions, channel exposure, price, discount, promotion, product, geography, distribution and sales, using percentage-change controls.
- **Scenarios:** a baseline and a custom scenario, each with a predicted outcome and an impact summary.
- **Model:** uses a validated predictive model or a statistically defensible method only; predictions are never arbitrary. Model confidence and limitations are shown.

### F. Preservation

- The old pages 3–6 and the engines in `src/models`, `src/optimization` and `src/marketing` are kept.
- `/optimize-budget` and `/simulate-budget` in the API are unchanged.
- The AI analyst keeps working.
- New charts follow the PART 01 standard.

### Tests

- **G:** arbitrary column names.
- **H:** missing values.
- **I:** duplicate records.
- **J:** different date formats.
- **L:** different target variables, plus the multi-target confirmation.
- **M / N / O:** spend, exposure and general modes, including the reason for each mode.
- **Y:** leakage, for importance and effect estimates.
- **V:** axis labels for pages 04–07.
- **Z:** error handling.
- **A:** regression, including the old engines, the API and the analyst.
- **Hand-verifiable maths:** regression coefficients and robust SEs against a closed-form fixture; correlation p-values; ROI and ROAS formulas when spend exists, and their absence when it doesn't.
- **Fixtures,** mathematically controlled and labelled as test fixtures:
  1. marketing spend;
  2. marketing impressions;
  3. general retail;
  4. e-commerce;
  5. FMCG;
  6. arbitrary names;
  7. missing values.

### Documents

`DATASET_COMPATIBILITY_REPORT.md` (main content), `PART_STATUS.md`, `CHANGELOG.md`, `TEST_REPORT.md`.

---

## PART 03: Complete Statistics implementation

### A. Descriptive Statistics tab

All existing functionality is preserved. Improvements cover the interface, usability, calculations, validation and visualisations, and every chart is labelled to the PART 01 standard.

### B. Inferential Statistics tab (master prompt §5A)

- **Tests:**
  - descriptive statistics;
  - one-sample Z-test and one-sample t-test;
  - independent-samples t-test and paired t-test;
  - one-way ANOVA;
  - chi-square goodness of fit and chi-square independence;
  - Pearson and Spearman correlation significance;
  - confidence intervals;
  - all other procedures already supported.
- **Every test shows:**
  - the test name;
  - null and alternative hypotheses;
  - test statistic and p-value;
  - significance level;
  - degrees of freedom where applicable;
  - CI where applicable;
  - interpretation;
  - statistical limitations.
- **Z-test fix** (the root cause is the UI running before σ is entered):
  1. A clear input for the known population SD (σ), validated as numeric and greater than 0.
  2. Input for the hypothesised mean μ0.
  3. Alternatives: two-sided, greater than, less than.
  4. Selection of the significance level.
  5. The correct Z = (x̄ − μ0) / (σ/√n).
  6. The p-value and the conclusion.
  7. The formula shown with the user's values substituted, plus an explanation.
  8. **No silent substitution of the sample SD, ever.**
  9. Without σ, it explains why and offers the one-sample t-test when its assumptions are reasonable.
  10. No fabricated population parameters; the user always sees why the test is or isn't available.

### C. Predictive Analysis tab (master prompt §5B; approval §4, §5, §6)

- **All 12 classifiers, user-selectable:**
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. SVM
  5. KNN
  6. Gradient Boosting
  7. Extra Trees
  8. XGBoost
  9. LightGBM
  10. CatBoost
  11. Naive Bayes
  12. AdaBoost
- **Selection UI:**
  - select one, several or all compatible models, and deselect;
  - only the selected models run; the user controls how many run;
  - **the default is Logistic Regression, Decision Tree and Random Forest**;
  - nothing is selected silently;
  - each model's status is shown: trained, or failed with the reason.
- **Pre-training checks:**
  - identify the target; categorical vs continuous; binary vs multiclass;
  - class counts and sample size;
  - leakage columns;
  - missing values;
  - preprocessing requirements.
- **Continuous targets:** never converted to classes. The page explains that the 12 classifiers need a categorical target, and the **existing regression models stay available** for continuous targets.
- **Model-specific requirements:**

  | Model | Requirement |
  |---|---|
  | Logistic Regression | scaling and regularisation |
  | Decision Tree | depth and overfitting controls |
  | Random Forest | reproducibility and feature importance |
  | SVM | scaling and kernel handling |
  | KNN | scaling and neighbour-count validation |
  | Gradient Boosting | reproducible training |
  | Extra Trees | ensemble configuration |
  | XGBoost, LightGBM, CatBoost | optional-dependency detection |
  | Naive Bayes | variant chosen from feature characteristics |
  | AdaBoost | compatible estimator configuration |

- **Optional dependencies:**
  - `requirements-optional.txt` lists `xgboost-cpu`, `lightgbm` and `catboost`; none of them is added to mandatory dependencies;
  - an unavailable package shows a clear explanation and the install command; nothing crashes, nothing pretends a model was trained, and nothing is silently substituted;
  - **Linux/Docker** compatibility is verified by installing and training each in a test environment;
  - **Windows** compatibility is verified at the level possible here: `win_amd64` wheel availability for Python 3.13, and code paths without POSIX-only calls. Actual execution on Windows is reported as a platform limitation, because no Windows machine is available.
  - the LightGBM `libgomp1` requirement for slim Docker images is documented.
- **Evaluation:**
  - one stratified train/holdout split, shared by every model;
  - preprocessing and feature-selection pipelines fitted on training data only;
  - reproducible seeds;
  - **cross-validation results are used for selection and comparison and are shown for every successfully trained model**;
  - **holdout metrics are shown for all selected models, are clearly separated from CV, and are never used to select or tune the winner**;
  - models are never compared on different samples.
- **Metrics:**
  - accuracy, precision, recall and F1;
  - ROC-AUC and log loss where appropriate;
  - confusion matrix;
  - training and prediction time;
  - class-wise metrics with macro and weighted averages for imbalanced data.
- **Charts (below the model selection and results section):**
  1. accuracy comparison;
  2. precision comparison;
  3. recall comparison;
  4. F1 comparison;
  5. ROC-AUC comparison, where valid;
  6. training time comparison;
  7. prediction time comparison;
  8. confusion matrices for the selected models;
  9. feature importance for compatible models;
  10. ROC curves, binary targets only.

  Each chart has a title, named x and y axes (for example *Machine Learning Algorithm* / *Accuracy (%)*, or *Training Time (seconds)*), units, legends, readable labels and a responsive layout. Charts show exactly the models the user selected (3 → 3, 8 → 8). No results are shown for failed models, and no ROC is shown where the class structure doesn't support it.

### Tests

- **P:** all 12 algorithms where compatible.
- **Q:** optional-dependency failure handling (simulated missing packages) and real training when installed.
- **R:** statistical calculations against hand-computed values.
- **S:** the Z-test input-validation matrix, plus the no-substitution guarantee.
- **Y:** leakage: preprocessing and feature selection fitted inside folds; the holdout is untouched by selection.
- **V:** labels on every statistics chart.
- **A:** regression of all existing statistics tests.
- **Fixtures:**
  8. categorical target;
  9. continuous target;
  10. imbalanced classification target.

### Documents

`STATISTICAL_VALIDATION_REPORT.md`, `MODEL_COMPARISON_REPORT.md`, `PART_STATUS.md`, `CHANGELOG.md`, `TEST_REPORT.md`.

---

## PART 04: Prescriptive Analytics, optimizer, simulator, constraints, and full integration

### A. Budget-constraint and feasibility engine (master prompt §7)

This applies to 06 · Marketing Resource Optimizer, 15 · Prescriptive Analytics and the legacy optimizer.

- **Feasibility checks** compute the sum of minimums, the sum of maximums and the total budget, and test two conditions:
  1. Σmin ≤ budget.
  2. Σmax ≥ budget, when the full budget must be allocated.
- **When the maximums can't cover the budget,** the module shows *"The entered channel maximums cannot accommodate the full budget."* and offers:
  - **A.** increase channel maximums;
  - **B.** reduce the total budget;
  - **C.** allow a configurable unallocated reserve;
  - **D.** add eligible channels, if supported.
- **No silent changes** to user constraints, and no fake optimal solution when the problem is infeasible.
- **Fixing the root cause:**
  - feasible defaults that depend on the number of channels and are re-derived when the budget or channel set changes;
  - Σmin, Σmax and the budget shown next to the inputs.
- **Optimisation:**
  - constraints validated before optimising;
  - a mathematically appropriate method;
  - all minimums, maximums and the total budget respected;
  - no negative allocations and no unsupported channels;
  - the objective, solver status, feasibility, unused budget and reasons for infeasibility all reported;
  - "optimal" claimed only when the solver returns a valid optimal or appropriately qualified solution.
- **Objectives,** offered only when the data supports them: maximise predicted sales, revenue, expected conversions or the modelled outcome.
- **Sensitivity analysis:**
  - allocations compared under different constraints;
  - original vs optimised allocation, the difference, the expected outcome and constraint utilisation;
  - charts with labelled axes.

### B. Scenario Simulator completion (master prompt §2, module 4, plus §7 integration)

- Multiple-scenario comparison and scenario comparison graphs.
- An impact summary.
- Model confidence and limitations, including extrapolation warnings outside the training range and prediction intervals.
- Shared engine behaviour with the optimizer and with 15 · Prescriptive Analytics.

### C. Prescriptive Analytics improvements

- The fixes and reporting above, applied to page 15.
- The existing behaviour and tests are preserved.

### D. Complete project integration and final consolidated testing (approval: strict requirement 9)

- **Integration:**
  - all updated modules together;
  - dataset switching;
  - CSV and XLSX;
  - the existing Forecast;
  - old and new navigation;
  - the entire application.
- **App-wide chart guard:** every Plotly figure on every page has a title and x/y axis titles.
- **Browser pass (Chrome):** all pages on demo and on uploads, at desktop and phone widths.
- **Final independent audit:**
  - fresh extraction into a clean venv;
  - full suite;
  - security audit, including that auth, rate limits and validation are not weakened;
  - dependency audit, including the optional packages;
  - application start-up (dashboard and API).
- **Final release:** documentation, then the release ZIP, its integrity test and SHA-256.

### Tests

- **T:** budget feasibility, including the exact reported case (one channel with a 55% default maximum).
- **U:** optimisation constraints:
  - bounds, total and reserve accounting;
  - a hand-solvable optimum with a known KKT solution;
  - no output when infeasible;
  - solver-status reporting.
- **Scenario simulator:** baseline reproduction, extrapolation flags, and interval presence.
- **F:** dataset switching.
- **X:** Forecast regression.
- **W:** old and new navigation.
- **Z:** security and error handling.
- **A:** full regression.
- **The complete A–Z list re-run** as final consolidated testing.

### Documents

- `OPTIMIZATION_VALIDATION_REPORT.md`
- `FINAL_RELEASE_NOTES.md`
- final versions of all nine documents: `PART_STATUS.md`, `IMPLEMENTATION_PLAN.md`, `CHANGELOG.md`, `TEST_REPORT.md`, `DATASET_COMPATIBILITY_REPORT.md`, `MODEL_COMPARISON_REPORT.md`, `STATISTICAL_VALIDATION_REPORT.md`, `OPTIMIZATION_VALIDATION_REPORT.md`, `FINAL_RELEASE_NOTES.md`

---

## Traceability: every original requirement and where it is implemented

| Original requirement (master prompt / approval) | Part |
|---|---|
| §1 protection rules 1–15 (preserve features, Forecast, uploads, detection, navigation, APIs, tests, validation, no fabrication, no hard-coded columns or schema, artifacts, dependencies, unrelated modules) | All parts (checked at each part's end) |
| §1 inspection, backup, manifest, checksums, baseline tests, protected files, dependency versions | 00 (done) |
| §2 removal of the old four from navigation; no simple renaming | 01 (menu), 02 (real modules) |
| §2 Module 1: Marketing Effectiveness Model (all features) | 02 |
| §2 Module 2: Channel Performance & Contribution (all features; no invented ROI/ROAS) | 02 |
| §2 Module 3: Resource Optimizer (all allocation types, user constraints, objectives, comparisons, estimates) | 02 (universal module), 04 (constraint/feasibility engine, reserve, solver reporting, sensitivity) |
| §2 Module 3: no fake monetary recommendations without spend | 02, re-verified in 04 |
| §2 Module 4: Scenario Simulator (baseline, custom, % controls, predicted outcome, validated model) | 02 |
| §2 Module 4: multiple-scenario comparison, comparison graphs, impact summary, confidence and limitations | 02 (initial), 04 (complete) |
| §3 universal dataset compatibility engine (all signals and roles) | 02 |
| §3 alternative target names; multi-target confirmation | 02 |
| §3 Spend / Exposure / General modes; explanations per model | 02 |
| §4 navigation restructure (approved 00–15) | 01 |
| §4 Statistics as the second menu entry and a complete workspace | 01 (page), 03 (content) |
| Approval §2: three tabs, Descriptive / Inferential / Predictive | 01 |
| §5A all inferential tests and the required output fields | 03 |
| §5A Z-test fix (items 1–8, no substitution, t-test alternative) | 03 |
| §5B the 12 classifiers, selection UI, defaults, pre-training validation | 03 |
| §5B model-specific requirements; optional dependencies | 03 |
| Approval §4: regression models kept; no target binning | 03 |
| Approval §5: CV for selection, CV for all models, untouched holdout for all, clear separation, no holdout tuning, leakage prevention | 03 |
| Approval §6: optional requirements file; Windows and Docker/Linux verification | 03 |
| §5B metrics and imbalanced-data handling | 03 |
| §5B the 10 charts below the results, with label standards | 03 |
| §6 Executive Overview: full chart audit (items 1–13) | 01 |
| §6 labels in other pages reusing the same components | 01 |
| §7 budget-constraint bug: feasibility, message, actions A–D, no silent changes | 04 |
| §7 optimisation requirements, objectives, sensitivity, "optimal" claims | 04 |
| §8 universal graph standard | 01 (shared utility and existing pages), 02/03/04 (new charts), 04 (app-wide guard) |
| Approval §7: currency detection, override, no symbol when unknown, demo INR | 01 |
| §9 UX: navigation, headings, hierarchy, model controls, errors, loading and empty states, summaries, spacing, responsiveness | All parts, each for the pages it touches; 04 browser pass |
| §9 the six explanations on every analytical page | 01 (Statistics tabs), 02 (pages 04–07), 03 (statistics content), 04 (optimizer and simulator) |
| §11 tests A–Z | Mapped in each part's test list; all re-run in 04 |
| §11 the 10 test-dataset families | 02 (1–7), 03 (8–10) |
| §12 backup, integrity, SHA-256, attachment, change summary, test results, limitations, protected-file confirmation after each part | All parts |
| §12 the nine maintained documents | Started in 01, extended per part, finalised in 04 |
| §13 one part at a time; stop for authorisation | All parts |
| §14 acceptance criteria 1–20 | Verified in 04 (final consolidated testing) |

## Decisions for you

1. **Split of the optimizer and simulator between PART 02 and PART 04.** Your structure puts the four universal models in PART 02 and "Budget Optimizer, Scenario Simulator, budget constraints, feasibility validation" in PART 04. As mapped above:
   - **PART 02** delivers working universal versions of pages 06 and 07 on the existing validated engine (which already refuses infeasible constraints).
   - **PART 04** delivers the full feasibility and constraint engine (reserve, remedies, solver reporting, sensitivity) and completes the multi-scenario simulator.

   Until PART 04, page 06 uses the current constraint inputs. The known one-channel default problem (maximums fixed at 55% each) would therefore still be visible on 06 and 15 between PART 02 and PART 04. If you prefer, PART 02 can include the minimal default-limits fix now, with the full engine still in PART 04.
2. **Edits already made in the working copy.** Before your new instruction I had started the old PART 01 in the isolated working copy (not the baseline):
   - the navigation registry, the Statistics page with lazy tabs, the four interim pages 04–07, the scaffold and badge changes, and the reference-text updates;
   - adapting the existing tests' navigation calls.

   These fall inside the new PART 01 scope, but **they are untested** (the test run was interrupted). No backup contains them. My plan is to keep them as the starting point of the new PART 01 and test them fully; otherwise I can reset the working copy to the baseline before starting.
