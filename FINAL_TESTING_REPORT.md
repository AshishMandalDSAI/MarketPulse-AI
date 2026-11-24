# MarketPulse AI: Testing Report

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22

## Evidence labels

- **[EXECUTED]**: run in this environment, with the output recorded.
- **[TEST]**: automated test.
- **[STATIC]**: code reading.
- **[NOT VERIFIED]**: not checked.

## 1. Final full suite [EXECUTED]

| Item | Value |
|---|---|
| Where | Fresh extract of the release ZIP, in a **new virtualenv** installed only from `requirements.txt` |
| Command | `python -I -m pytest tests/ -v -p no:cacheprovider -rfEs` |
| Environment | Python 3.13.18, pytest 9.1.1, Linux, 2 vCPU |
| **Collected** | **3,030** |
| **Passed** | **3,030** |
| **Failed** | **0** |
| **Skipped** | **0** |
| Errors | 0 |
| Warnings | 1: a third-party `StarletteDeprecationWarning` from `fastapi.testclient`, not from project code |
| Duration | 579.9 s (9 min 40 s) |
| Full log | `FINAL_TEST_LOG.txt` (verbose, one line per test) |

The log above came from the candidate build. The released ZIP contains the same code and tests; only report files differ. The suite was then **run again on a fresh extract of the released ZIP**, in another new virtualenv. A ZIP cannot contain its own checksum or its own post-build verification, so that run's counts and the SHA-256 are recorded in `FINAL_SHA256.txt`, delivered next to the ZIP.

## 2. Baseline reproduction [EXECUTED]

Before any change, the untouched baseline (`MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip`) gave **2,888 passed, 0 failed** in 555 s. This matches its published figure.

## 3. Tests added in this release: 142

| File | Tests | Covers |
|---|---|---|
| `test_ultimate_analytics_accuracy.py` | 30 | Zero-safe metrics computed by hand. Walk-forward folds use only the past. Moving-average validation does not use actuals. Adstock and Hill transforms computed by hand. MMM leakage (parameters, controls and target). Missing revenue is excluded. MAPE is never a fabricated 0. Transactional rows are summed per period. Baseline-first selection. Too little history. Optimizer bounds, sum and non-negativity, the zero-spend case, infeasible input, like-for-like gain, and API 400. |
| `test_ultimate_data_quality.py` | 29 | Missing, blank and placeholder values; invalid numbers; numbers stored as text; ±inf; invalid dates; duplicate rows and column names. Score dimensions and weights. Imputed cells are not counted as complete. Each recommendation rule. Every strategy. Copy semantics and the audit. Dates and revenue are never filled. Duplicates are kept by default. The past-only median, including its equivalence to the reference. |
| `test_ultimate_security.py` | 31 | Rate limiter: window, bound, 429 and `Retry-After`, the upload class, exemptions, and the development and production defaults. Single-worker enforcement. PBKDF2: round trip, salt, malformed hash, no plaintext setting. Dashboard gate: open in development, refusal in production, sign-in, lockout. Real privilege-drop runs of the entrypoint. Secret hygiene. Proxy configuration files present. |
| `test_ultimate_ui.py` | 23 | Theme and config, the CSS selectors that match the Streamlit DOM, `md_inline` escaping, formatters, KPI cards, `apply_chart_style`, keyed navigation, the nested-button guard (an AST check across all pages), the upload "Go to overview" callback, the auto sidebar on phones, and chart rows stacking on phones. |
| `test_ultimate_upload_flow.py` | 11 | The forecast loader with a non-`Date` date column. An end-to-end upload whose forecast is loadable. A Forecast page AppTest showing the history and forecast traces; this test was mutation-checked and fails without the fix. Per-date aggregation. Date-range formatting. The forecast capability counts distinct dates. Table date format. |
| `test_ultimate_dates_aliases.py` | 12 | Exact day-first and month-first parsing. ISO text, blanks and garbage values. Ambiguous and inconsistent dates are reported. Numbers are never treated as dates. The quality engine on day-first data. Channel-cost aliases versus COGS. End-to-end MMM on weekly `dd/mm/yyyy` data. |
| `test_ultimate_ai_analyst.py` | 6 | Honest mode label. No claim of using an LLM. Evidence type and sources. No fabricated ₹0. No causal wording. The demo simulator is refused for uploads. |

## 4. Changes to existing tests

**No test was removed, skipped, or had an assertion weakened.**

There is one correction to a ground-truth helper:

| File | Change | Why it is not a weakening |
|---|---|---|
| `tests/test_legacy_cache_isolation.py::truth()` (the `load_forecast` branch) | The expected value for an uploaded dataset's forecast is now the forecast file's real shape. Previously it was an empty frame. | The helper's docstring said it *mirrors* the old loader: "a date column that is not in the file makes the read fail, which the loaders … report as empty". That failure was the bug fixed in this release, where uploads never showed their forecast. The test still compares every loader against an independent reading of the files. 7 test cases (5 fixture datasets plus 2 session walk-throughs) changed from expecting empty to expecting the real data; all 133 tests in the file pass. |

## 5. Other executed verification

| Check | Result | Report |
|---|---|---|
| Browser (Chrome via Playwright) | Demo 18/18 and upload 18/18 at 1440 px; 4 key pages at 820 and 390 px; 18/18 again against the clean extract. 0 exceptions, 0 error banners, 0 overflow, 0 console errors. | `FINAL_UI_UX_AUDIT_REPORT.md` |
| Live API (development and production) | Health, docs, upload, auth 401/200, rate limit 429, production refusal without a key | `FINAL_DEPLOYMENT_VERIFICATION.md` |
| Docker Compose (substitute base) | Build, healthy, non-root, sign-in in Chrome, restart, down | `FINAL_DEPLOYMENT_VERIFICATION.md` |
| Benchmarks: 10k, 100k, 500k and 1M rows | All completed; peak 1.8 GB | `FINAL_PERFORMANCE_REPORT.md` |
| Dataset compatibility sweep (5 shapes) | All ran; capabilities as expected | `FINAL_ANALYTICS_ACCURACY_REPORT.md` |
| pip-audit | 0 known vulnerabilities, 60 packages | `FINAL_DEPENDENCY_SECURITY_REPORT.md` |

## 6. Not verified

- Python versions other than 3.13.18. The ranges in `requirements.txt` are broader than what was tested.
- Windows and macOS.
- Browsers other than Chrome.
- The official Docker base image.
- Streamlit Community Cloud.
