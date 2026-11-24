# MarketPulse AI: Final Changelog

# Release: MarketPulse_AI_FINAL_RELEASE_VERIFIED, PART 06 (2026-11-23)

The base is `MarketPulse_AI_FINAL_RELEASE.zip` (PART 05, SHA-256 `c1c26df4…f87c3`), preserved unchanged.
The details are in `CHANGELOG.md` → *PART 06*.

| Area | Delivered |
|---|---|
| Real Kaggle file | Validated end to end (A–I). **Fixed:** `*_Content_Score` columns were read as channel ad spend (new `quality_score` role). The 3 real-file tests now run and pass. |
| Z-test | Verified against hand-computed textbook values (no scipy); UI shows the statistic, p-value, CI and conclusion |
| Docker / optional ML | `INSTALL_OPTIONAL_ML` build argument (default off); 10 required container checks pass on the stand-in base, and 12/12 models run in a container. The official base image is **blocked**. |
| Windows | `scripts/windows/` (setup, start, tests, one-command validator); explicit UTF-8 in tests; Windows-safe test guards; `.gitattributes`. **Not tested on Windows.** |
| Performance | Confirm & Analyze 12–19% faster and up to 0.5 GB lighter at 10k–100k rows, with **identical outputs** |
| Security | 65-probe local penetration-style set: 6 findings → 0. CSV formula injection in exports fixed. API security headers; docs off in production; truncation reported; parse-failure cleanup. |
| Tests | 3,522 / 3,522 passed, 0 skipped, **0 warnings** (`httpx2`); 76 tests added |
| Protected files | 3 more changed (`api/main.py`, `Dockerfile`, `docker-compose.yml`), documented; 11 of 85 changed in total |

---

# Release: MarketPulse_AI_FINAL_RELEASE, upgrade PARTs 01–05 (2026-11-23)

**Base:** `MarketPulse_AI_ULTIMATE_FINAL_RELEASE.zip`, SHA-256 `fd46e3c6…8515c` (preserved unchanged).
The full per-part details are in `CHANGELOG.md`, and the test evidence is in `FINAL_TEST_REPORT.md`.

| Part | Delivered |
|---|---|
| 01 | Approved 00–15 navigation registry; a Statistics page with three tabs; Executive Overview chart corrections; automatic currency detection that never assumes INR or USD for uploads |
| 02 | Universal role engine (roles, evidence, confidence, multiple targets); Spend, Exposure and General modes; four universal marketing models (pages 04–07); trade spend handled as a business variable; no fabricated ROI or ROAS; optimiser 55%-default fix |
| 03 | Z-test that requires a known σ and never substitutes the sample SD; formulas and labelled charts for all 10 inferential tests; 12 classifiers with optional XGBoost, LightGBM and CatBoost; class-imbalance option; selection by CV only, then each model scored once on the holdout; 10 comparison charts |
| 04 | Feasibility engine with exact messages and remedies A–D (applied only on click); optional unallocated reserve; five solution statuses; constraint utilisation; Scenario Simulator total-budget % and sensitivity |
| 05 | App-wide chart and UI audit and fixes (Forecast, Driver Importance); browser pass of all 16 pages on 4 datasets; security audit (bandit, pip-audit); performance report; Docker readiness; final docs and regression |

**Protected baseline files changed (8, each documented in `CHANGELOG.md`):**

- `dashboard/pages/0_upload_data.py`
- `dashboard/pages/7_forecast.py`
- `src/data_engine/column_classifier.py`
- `src/marketing/field_discovery.py`
- `src/predictive/contracts.py`
- `src/predictive/evaluate.py`
- `src/predictive/procedure.py`
- `src/prescriptive/engine.py`

The other 77 protected files are byte-identical.

**Tests.** No test was removed or skipped to make the suite pass. Requirement-driven edits to
existing tests are listed in `CHANGELOG.md` (PART 03: the holdout-once tests now count one holdout
scoring per evaluated model; PART 02 and 04: wording). The real Kaggle file was never provided, so
its 3 tests are skipped.

---

# Release: MarketPulse_AI_ULTIMATE_FINAL_RELEASE (2026-11-22)

**Base:** `MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip`, SHA-256 `e6c4d86251218ca3976138d1983e283712b486ad719754c3eb4c817ef8990431`, preserved unchanged.

**Baseline suite (re-run before any change):** 2,888 passed, 0 failed.

- All 18 pages, the API contracts, the branding and the formulas are preserved.
- No test was removed, skipped or weakened.
- One test helper's ground truth was corrected because it encoded a bug; see "Tests" below.

## User interface

The baseline had a light theme with forced light text, which made text invisible and charts white. It is now a professional dark design system:

- `.streamlit/config.toml` (dark theme);
- tokens in `dashboard/styles/theme.py`;
- reusable components in `dashboard/components/ui.py` (callout, empty state, section, page header, number formatters, `show_table`).

Other changes:

- **Sidebar:** brand block, an "Active dataset" card (demo, uploaded or none) with Demo/Reset buttons, grouped navigation (Data · Overview · Marketing models · AI & insights · Advanced analytics · Statistics), an active-page highlight, and Sign out when authentication is on. It is collapsed by default on phones (`initial_sidebar_state="auto"`).
- **Executive Overview:**
  - KPI cards show half-over-half comparisons only when the data supports them;
  - a forecast overview reads the walk-forward evaluation;
  - a data-quality panel shows original → cleaned scores, using the same engine as the Upload page;
  - insights are rendered safely (`md_inline`);
  - time-series charts aggregate per date;
  - dates are shown readably;
  - donut charts use inside labels without collisions.
- **Upload page:** rewritten as four steps (quality, roles, cleaning, analyze), with friendly read errors (encoding and delimiter sniffing, `.xls` guidance), an editable cleaning plan, a before/after preview, and downloads of the cleaned data and the audit.
- **Charts:** `apply_chart_style` is applied to all Plotly charts (40 call sites). Currency axes appear only where the unit is known (₹ on the demo).
- **Tables:** readable headers, number formats, and dates shown as `YYYY-MM-DD`.

### Pre-existing bugs found in the browser and fixed

| Bug | Fix | Test |
|---|---|---|
| The first navigation click after a forced page change landed on page 00. | Keyed radio. | `test_ultimate_ui.py` |
| The "Go to Executive Overview →" button after an upload never worked; it was nested inside another button's branch. | `on_click` callback, plus an AST check that no page nests buttons. | `test_ultimate_ui.py` |
| The Forecast page said "not available" for every upload whose date column was not literally named `Date`. | The loader parses the date column that is actually present. | `test_ultimate_upload_flow.py` |
| The Forecast page had no history line for uploads. | Uses the evaluated target column. | `test_ultimate_upload_flow.py` (mutation-checked) |
| Literal `**markdown**` appeared inside HTML cards. | `md_inline`. | `test_ultimate_ui.py` |
| The Z-test "not run, needs population SD" message was a red error on first load. | Shown as a warning (expected behaviour). | existing Phase 8 tests |

## Data quality and missing values (new)

- **New module `src/data_engine/data_quality.py`:** detects NaN, blanks and placeholders, invalid numbers and dates, ±inf, duplicates, duplicate column names and type issues. It produces an explainable five-dimension score (original and after cleaning) and offers user-controlled cleaning with an audit trail. The original file is preserved; dates and revenue are never invented; duplicates are kept by default.
- **New module `src/data_engine/date_parsing.py`:** decides the day/month order once per column. Before this, `dd/mm/yyyy` data lost 62% of its dates and silently misread the rest.
- Infinite values are converted to missing at the start of analysis and reported.

## Analytics accuracy

- **MMM leakage removed:** channel parameters are selected on training rows; missing targets are excluded instead of median-filled; control medians come from training rows only. The demo artifacts were regenerated: adstock test R² 0.362 → 0.348, optimizer uplift 21.7% → 16.6%, same selected model.
- **Forecasting:**
  - zero-safe metrics (MAE, RMSE, MAPE, sMAPE, WAPE, R²; undefined metrics are `None`);
  - walk-forward backtests against naive and seasonal-naive baselines;
  - baseline-first selection with a 2% rule;
  - out-of-sample intervals;
  - per-period aggregation of transactional data;
  - moving-average validation made honest;
  - most-recent contiguous window instead of random sampling.
- **Demo forecast evaluation:** Ridge beats naive by 33.2% MAE (4 folds × 12 weeks). The published forecast is unchanged.
- **Budget optimizer:** input validation (`ValueError`, or HTTP 400 from the API), bounds clipping and re-checking, `success` false when constraints are violated, and like-for-like comparison at the same budget.
- **Forecast capability:** counts distinct dates instead of rows.
- **AI Analyst made honest:** labelled "rule-based, offline"; every answer shows its evidence type and source; no fabricated ₹0; no causal wording; the demo simulator is refused for uploads.

## Dataset compatibility

- Channel "cost" and "budget" aliases (`facebook_cost`, `TV Budget`, …). Previously, `facebook_cost` matched the business-cost (COGS) role.
- Compatibility sweep over 5 dataset shapes (`tools/compat_sweep.py` results in `FINAL_ANALYTICS_ACCURACY_REPORT.md`).

## Performance

- Segmentation chooses k on a 5,000-row sample: 20k-row analysis went from 122 s to 48 s.
- Past-only median: O(n²) → expanding median, with identical results.
- Benchmarks at 10k, 100k, 500k and 1M rows (`tools/benchmark.py`, `FINAL_PERFORMANCE_REPORT.md`).

## Security and deployment

- **API rate limiting:** HTTP 429 with `Retry-After`; on in production, off in development; configurable.
- **Dashboard authentication:** PBKDF2-SHA256 (600k iterations), constant-time compare, lockout, no plaintext setting, required in production.
- **Non-root containers:** an entrypoint privilege drop that handles bind-mount ownership.
- **Single-worker enforcement:** `--workers 1`; production refuses `WEB_CONCURRENCY > 1`.
- **HTTPS:** `deploy/` contains a Caddyfile, `nginx.conf`, a Compose proxy overlay and a README with Streamlit Cloud steps.
- **Non-root volumes:** a `PermissionError` writing `docs/data_dictionary.csv` on a fresh non-root volume is handled.
- **Dependencies:** the unused `anthropic` dependency was removed (no code imported it).
- **Documentation:** `.env.example` and README updated (the AI analyst is rule-based; new environment variables).

## Tests

- **New test files:**
  - `test_ultimate_analytics_accuracy.py`
  - `test_ultimate_data_quality.py`
  - `test_ultimate_ui.py`
  - `test_ultimate_security.py`
  - `test_ultimate_ai_analyst.py`
  - `test_ultimate_upload_flow.py`
  - `test_ultimate_dates_aliases.py`

  The counts are in `FINAL_TESTING_REPORT.md`.
- **Corrected helper:** `tests/test_legacy_cache_isolation.py::truth()` for `load_forecast` mirrored the old loader's failure mode ("a date column that is not in the file makes the read fail"). Its ground truth is now the file's real shape. The comparison between the loader and the file is unchanged (it is no weaker).

---

# Previous release: MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip

**Release:** MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip
**Base:** MarketPulse_AI_FINAL_RELEASE.zip (SHA-256 `cde0db6a62fcacc3136eebd32f9ba74ab753518b2ad9cf276d5f34b23e8cb2dc`, preserved unchanged)
**Date:** 2026-11-22

Every change below was made because a problem was reproduced against the base
release, not for style. Each fix has a regression test that fails on the base
code and passes on this release, except where noted.

## Issue list and resolution

Severity scale: Critical, High, Medium, Low.

| ID | Severity | Problem found | Fix | Files |
|---|---|---|---|---|
| C-1 | Critical | `POST /upload` wrote the file to `raw_dir / file.filename` with the client-supplied name. `../../x.csv` or `/abs/path.csv` wrote outside the dataset folder (arbitrary file write; the containers run as root). | Filename reduced to a safe basename (`_safe_upload_filename`), plus a resolved-path containment check. | `api/main.py` |
| H-1 | High | No production security mode. With `API_KEY` unset, every protected endpoint was open with only a log warning, while `audit_reports/FINAL_SECURITY_AUDIT_REPORT.md` reported an overall PASS. | Added `MARKETPULSE_ENV` (`development` default, `production`). Production refuses to start without an API key; an unknown value refuses to start; protected endpoints fail closed (503) if the key disappears. The startup log now lists exactly which endpoints are open. | `config/settings.py`, `api/main.py` |
| H-2 | High | The upload preview ran `standardize()` on the same mapping object stored for `/confirm`. `standardize()` clears unconfirmed MEDIUM/LOW roles in place, so the upload response hid the suggestions the user had to confirm, and `accept_suggested` on `/confirm` had nothing left to accept. `/confirm` also mutated the stored mapping, so a second confirm started from a damaged mapping. The existing test passed only because its asserts were weak (`confidence < 0.80` is true for a wiped guess). | Preview and confirm both work on `copy.deepcopy(mapping)`. Existing test left unchanged; a strict test was added. | `api/main.py` |
| H-3 | High | Pages 01 (Executive Overview), 11 (Predictions) and 12 (Anomalies & Drivers) called `render()` at module import, and `app.py` then called `render()` again, so the first visit in each server process drew the page twice. For uploaded marketing datasets this crashed page 12 (`StreamlitDuplicateElementId`) and page 11. | Module-level call guarded with `if __name__ == "__main__":`. Standalone `streamlit run <page>` still works. | `dashboard/pages/1_executive_overview.py`, `11_predictions.py`, `12_anomalies_drivers.py` |
| H-4 | High | The Predictions page expected forecast predictions as records, but the forecasting engine returns parallel flat lists (`predictions`, `lower_bound`, `upper_bound`, `future_dates`). Result: integer column names, then `AttributeError: 'int' object has no attribute 'lower'`. | Builds the chart frame from the engine's flat lists (dates, forecast, bounds); column-name checks made type-safe. | `dashboard/pages/11_predictions.py` |
| M-1 | Medium | `GET /kpis?dataset_id=<uploaded>` returned an uploaded dataset's KPIs without authentication. | Requires the API key for uploaded datasets; demo KPIs stay public. | `api/main.py` |
| M-2 | Medium | API key compared with `!=` (not constant-time). | `hmac.compare_digest`. | `api/main.py` |
| M-3 | Medium | Upload body fully buffered before the size check. | Reads at most limit + 1 bytes. Empty files rejected with a clear message. | `api/main.py` |
| M-4 | Medium | `.xls` advertised in the API and dashboard, but `xlrd` isn't a dependency, so every `.xls` upload failed. | `.xls` removed from accepted types, with an actionable "save as .xlsx" message. No dependency added. | `api/main.py`, `dashboard/pages/0_upload_data.py` |
| M-5 | Medium | Parse errors echoed raw exception text to API clients. | Generic, actionable message; exception type logged server-side. | `api/main.py` |
| M-6 | Medium | Column names named in the release spec weren't recognised: `Meta_Ads`, `Google_Advertising`, `Marketing_Investment` (spend) and `Net_Sales` (revenue; it fell into the units-sold role). | Alias vocabulary extended (`meta ads`, `google advertising`, `marketing investment`, `net sales`, `gross sales`, …). No dataset column names were hardcoded in logic. | `config/column_aliases.py` |
| M-7 | Medium | Integer columns (IDs, scores) were offered as the date guess (`pd.to_datetime` reads integers as epoch nanoseconds), and identifier columns as the revenue guess. These were LOW-confidence and confirmation-gated, but misleading. | Numeric/bool columns are never date candidates; identifier-like columns (the profiler's existing rule) are skipped in the revenue fallback. | `src/data_engine/column_classifier.py` |
| M-8 | Medium | `GET /channels` returns display names (`Social Media`) that `POST /simulate-budget` rejected (it required `Social_Media`). | Simulate accepts canonical and display names case-insensitively and rejects duplicates and unknown names. `/channels` output unchanged (no client break). | `api/main.py` |
| M-9 | Medium | `.dockerignore` didn't exclude `data/uploads/*`, so a build on a host with real uploads baked user data into the images. | Excluded (`.gitkeep` kept), plus `data/.init.lock`. | `.dockerignore` |
| M-10 | Medium | Outdated docs: `docs/DECISION_INTELLIGENCE.md` said "Part 8: Not Started" although Part 8 is implemented and tested; the README had no install/run/test/deploy/limitations sections; the security report overclaimed. | Part 8 section rewritten from the actual code; README sections added; superseded audit reports marked. | `docs/DECISION_INTELLIGENCE.md`, `README.md`, `audit_reports/*` |
| L-1 | Low | 12 × scikit-learn `InconsistentVersionWarning`: `models/mmm_model.pkl` was pickled with scikit-learn 1.8.0. | Re-serialised with the verified scikit-learn 1.9.1 by running the project's own `src/models/marketing_mix_model.py`. Verified **numerically identical**: coefficients, intercept, scaler mean/scale (max abs diff 0.0), selected model, feature names, channel params, `model_metrics.json` and `actual_vs_predicted.csv` (max abs diff 0.0). | `models/mmm_model.pkl` |
| L-2 | Low | 2 × NumPy `RuntimeWarning: invalid value encountered in reduce` from `test_infinite_values_handled`: three places computed mean/std on a column holding +inf and −inf. | (a) Profiler: display stats over finite values, plus `non_finite_count`; finite columns unchanged. (b) `_looks_like_currency_measure`: returns False for non-finite columns explicitly (it already returned False via NaN; same result, no warning). (c) Canonical column metadata: values kept bit-identical, because the NaN deliberately blocks a variance bonus in target scoring; `np.errstate(invalid="ignore")` is scoped to those two reductions, with a comment. | `src/data_engine/profiler.py`, `src/data_engine/column_classifier.py`, `src/data_engine/canonical.py` |
| L-3 | Low | Obsolete `version:` key in `docker-compose.yml` (Compose v2 warning). | Removed. | `docker-compose.yml` |

`.env.example` was updated to document `MARKETPULSE_ENV`, `MARKETPULSE_API_KEY`
and the exact public/protected endpoint split.

## Model file

| | SHA-256 |
|---|---|
| Base `models/mmm_model.pkl` (scikit-learn 1.8.0) | `8a86df923206e15e184279ace7537a0256e6152655705acfdde1da102425a50c` |
| This release (scikit-learn 1.9.1, numerically identical) | `83e4fd49c83561b5dd4650360f7118b16be830f8948402e46c4e85304932680a` |

## Tests added (no existing test removed, skipped or weakened)

| File | Covers |
|---|---|
| `tests/test_final_security_hardening.py` | Dev/prod security modes, fail-fast start, protected/public endpoint split, constant-time key, upload filename sanitising, size/empty/xls/corrupt-file handling, real CSV + XLSX uploads with non-demo schemas, profiler infinities, upload→confirm suggestion preservation, `/channels`→`/simulate-budget` names |
| `tests/test_final_column_mapping.py` | Every date/revenue/spend column name in the release spec; numeric IDs never guessed as date/revenue; statistical guesses still need confirmation |
| `tests/test_final_dashboard_pages.py` | No page renders at import time (AST check over all 18 pages); pages 01/11/12 render for an uploaded marketing dataset; the forecast chart is drawn |

Evidence against the base release:
- H-2 and H-3/H-4: the new regression tests were run on the base code and
  **fail** there (2 and 6 failures respectively); they pass on this release.
- C-1: demonstrated directly. On the base code, `POST /upload` with filename
  `../../../C1_PROBE_escape.csv` returned 200 and wrote the file to `data/`,
  outside `data/uploads/<id>/raw/`. On this release the same request stores
  `escape.csv` inside the dataset's `raw/` folder (test
  `test_traversal_filename_is_written_inside_raw_dir`).
- H-3: the page crashes were reproduced on the base code by rendering all 18
  pages for an uploaded marketing dataset (34/36 renders passed on the base;
  72/72 on this release across demo + 3 uploaded datasets).

## Not changed

Architecture, branding, page set, API routes, response shapes, dependency
ranges in `requirements.txt`, analytical formulas, the demo dataset, and all
pre-existing tests.
