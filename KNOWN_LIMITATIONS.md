# Known limitations: MarketPulse AI final verified release

Status labels used everywhere in this release:

| Label | Meaning |
|---|---|
| **Implemented** | The code exists. Says nothing about testing. |
| **Tested: passed** | Executed here, and the recorded result passed. |
| **Tested with limitations** | Executed, but in a substitute environment or in part (the limitation is stated). |
| **Not tested** | Not executed. |
| **Blocked** | Could not be executed because of an external restriction (named). |
| **Not applicable** | Does not apply to this project. |

## 1. Verification gaps

| # | Item | Status | Detail / what is needed |
|---|---|---|---|
| 1 | Docker build from the **official** `python:3.13-slim` | **Blocked** | Docker Hub returns HTTP 403 from the build environment's proxy, and Debian mirrors do not resolve. The build, run, restart, persistence and security checks passed with a **local stand-in base image** (*tested with limitations*). Run `DOCKER_SETUP_GUIDE.md` §5 on a networked machine. |
| 2 | Optional ML packages **inside Docker with the official image** | **Blocked** (same cause) | With the stand-in base plus the PyPI wheels: 12/12 models ran. Without `libgomp1`: LightGBM was reported *unavailable* and 11 ran (*tested with limitations*). |
| 3 | **Windows** | **Not tested** on Windows | The scripts were run end to end with PowerShell 7.5.4 **on Linux**, and the suite was run under a strict non-UTF-8 locale (*tested with limitations*). Run `scripts\windows\validate_windows.ps1` (`WINDOWS_SETUP_GUIDE.md` §4). |
| 4 | macOS | **Not tested** | `xgboost-cpu` has no macOS wheels; use `xgboost` there. |
| 5 | Browsers other than Chrome | **Not tested** | |
| 6 | Load testing beyond 3 concurrent dashboard users and 30 concurrent API requests | **Not tested** | `PERFORMANCE_REPORT.md` §PART 06. |
| 7 | External penetration test, DAST tools, TLS configuration, container image CVE scan | **Not tested** | A controlled local probe set (65 probes), bandit and pip-audit were run instead. |

## 2. Analytical limitations

| # | Limitation | Status |
|---|---|---|
| 8 | The shared driver response model has no adstock. It is graded *weak* on the demo (and says so), and *good* on the real Kaggle file (holdout R² 0.91 for Sales_Value, 0.94 for Sales_Units). | Implemented as designed; tested: passed |
| 9 | Exposure datasets: no monetary ROI, ROAS or budget allocation. This is by design: without channel-wise cost, money cannot be derived from impressions. | Tested: passed (real file) |
| 10 | Exposure reallocation needs at least two channels in the **same unit**, each with a positive, statistically clear effect. On the real file: Sales_Value allows reallocation between TV and Instagram impressions; Sales_Units has fewer than two eligible channels, so it is *Not supported*, with the reason shown. | Tested: passed |
| 11 | Strong collinearity among the real file's exposure, price and TPR columns (VIF > 10). Individual channel effects are uncertain, and the page says so. | Disclosed in the UI |
| 12 | Scenario ranges reflect holdout error only; they are not full prediction intervals. | Disclosed |
| 13 | The optimiser can return *feasible, not proven* when its starts disagree. This is reported honestly, never upgraded. | Tested: passed |
| 14 | Long-format channel data (channel name in a column) is not pivoted automatically. | Implemented limitation |
| 15 | Legacy `.xls` files are not supported (`.csv`, `.xlsx` only). | Tested: passed (rejected with a message) |
| 16 | The bundled demo has no Phase 5 anomaly/driver output (baseline), so page 13 shows "not yet available" for the demo. | Baseline behaviour |

## 3. Operational limitations

| # | Limitation | Status |
|---|---|---|
| 17 | *Confirm & Analyze* takes about 255 s at 100,000 rows on 2 vCPUs (291 s before PART 06), and about 22 s for the real 11,232-row file. With 3 simultaneous users it took about 48 s each on 2 vCPUs. | Tested: passed (measured) |
| 18 | The API accepts at most `MARKETPULSE_MAX_ROWS` (100,000) rows per upload. Extra rows are dropped **and reported** in the response (`rows_truncated`, `warnings`). | Tested: passed |
| 19 | Single shared API key and dashboard password; no per-user accounts or roles; any key holder can read any dataset ID (single-tenant design). | Baseline design |
| 20 | The API runs exactly one worker (in-memory upload registry and rate limiter). | Baseline design |
| 21 | Model files (`models/*.pkl`) are pickles; only files produced by this project's pipeline may be placed there. | Documented |
| 22 | A client that wrongly puts the API key in the URL will have it recorded in the uvicorn access log. The API rejects it with 401; keys belong in the `X-API-Key` header. | Informational |
| 23 | The *Download cleaned data* CSV re-exports the user's own values unchanged (no formula escaping), so data such as `+91…` phone numbers round-trip intact. All analysis and result exports escape formula-like cells. | By design, documented |
| 24 | The dashboard's own HTTP responses get security headers (HSTS, nosniff, frame denial) only through the reverse proxy in `deploy/`. The API sets them itself. | Documented |
| 25 | Backup ZIPs from PARTs 01–04 store folders as mode 0600 (non-root `unzip` problem). The PART 05 and final ZIPs are fixed. Workaround: `chmod -R u+rwX,go+rX`. | Fixed going forward |

## 4. Not applicable

| Item | Why |
|---|---|
| SQL injection in the app | No user input reaches SQL. The only SQL code, `src/utils/database.py`, runs once as a fixed demo-pipeline step (started by `app.py` with a fixed argument list). It loads the project's own demo CSVs into SQLite; its only query is a fixed string, and neither the dashboard nor the API exposes it. |
| Database migrations | No database schema; all state is files under `data/` and `models/`. |
