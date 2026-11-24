# PART 06: deeper security validation

**Scope:** the final working copy, with emphasis on the API, uploads and exports.

**What this is, and is not.** This is a **controlled, penetration-style probe set against a local test instance only**,
plus static analysis and a dependency audit. It is **not** a third-party penetration test. No DAST scanner, fuzzing
campaign, TLS review or container-image CVE scan was performed (no such tools were reachable).

## 1. Tools that were run (PART 06)

| Tool | Result |
|---|---|
| `tools/part06/pentest_local.py`: 65 probes against the API in **production mode** (`MARKETPULSE_ENV=production`, random 43-character key, rate limiting on), on `127.0.0.1` | Before the fixes: **6 findings**. After the fixes: **0 findings** (65/65 pass). |
| bandit 1.9.4 over `src api dashboard config app.py tools scripts` (35,988 LOC) | 0 high, 3 medium (the accepted pickle loads from §2 of the PART 05 section), 45 low (try/except-pass fallbacks, asserts in test tools, fixed-argument `subprocess`) |
| pip-audit, 71 installed packages (core + optional + test) | **No known vulnerabilities** |
| AppTest probes: HTML in an uploaded **file name**, and in column names on 7 pages | Escaped everywhere: no live HTML |
| Strict non-UTF-8 locale run of the whole suite | Encoding is not a security issue; recorded in `FINAL_TEST_REPORT.md` |

## 2. Findings and fixes

| # | Finding | Severity | Status |
|---|---|---|---|
| S1 | **CSV formula injection** in the page 04/05/07 downloads (effect estimates, channel table, scenarios). A column named `=HYPERLINK("http://evil","x") TV_Spend` was exported as a live formula cell (demonstrated). Headers were neutralised; cells were not. | **Medium** | **Fixed:** the pages now use the shared cell sanitiser (`dataframe_to_csv_bytes`) as well as the header neutraliser. Test: `test_page_csv_exports_neutralise_formula_cells`. |
| S2 | **Silent data truncation:** API uploads over `MARKETPULSE_MAX_ROWS` (100,000) were cut to 100,000 rows with no notice in the response | Medium (integrity) | **Fixed:** the response now carries `rows_received`, `rows_used`, `rows_truncated` and a plain-language `warnings` entry. Test: `test_api_row_limit_is_reported_not_silent`. |
| S3 | Interactive API docs and the OpenAPI schema (`/docs`, `/redoc`, `/openapi.json`) were served in production | Low (information exposure) | **Fixed:** disabled when `MARKETPULSE_ENV=production`; still served in development. Test: `test_api_docs_disabled_in_production_only`. |
| S4–S6 | No `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` or `Cache-Control` on **direct** API responses. The proxy configurations in `deploy/` add them. | Low | **Fixed:** an API middleware sets `nosniff`, `DENY`, `no-referrer` and `no-store` on every response. Test: `test_api_security_headers_on_every_response`. |
| S7 | Files that failed to parse were left on disk under `data/uploads/<id>/raw/` | Low (storage) | **Fixed:** the upload directory is removed on a parse failure or when the file has no columns. Test: `test_api_unparseable_upload_leaves_nothing_on_disk`. |
| S8 | A client that puts the API key in the **URL** (`?api_key=`) gets 401, but uvicorn's access log records the query string, including the key | Informational | Documented (`KNOWN_LIMITATIONS.md` #22). Keys belong in the `X-API-Key` header, which is never logged. |
| S9 | *Download cleaned data* re-exports user values without formula escaping | Informational | Kept by design, so the user's own data round-trips unchanged (e.g. `+91…`). Documented. |
| S10 | Single shared key and password; any key holder can read any dataset ID (no per-user authorisation) | Medium (design) | Baseline single-tenant design, unchanged and documented. Per-user accounts would be a new feature. |
| S11 | Pickle model files | Medium (bandit B301) | Accepted as in PART 05: they load only the project's own `models/*.pkl` and never uploads. |

## 3. Probe results after the fixes (all PASS)

| Area | Probes | What was verified |
|---|---|---|
| Authentication / API key | 17 | All 7 protected endpoints return **401** with no key and with a key one character off. A key in the query string, or an empty key header, gives 401. KPIs of a non-demo dataset without a key give 401/404. |
| Upload validation | 10 | A valid CSV is accepted. `.exe`, `.csv.exe`, `.xls`, `.txt` and `.html` are rejected (400). Binary data renamed `.csv` and CSV text renamed `.xlsx` give 400 with a generic message. An empty file gives 400. A 51 MB upload gives **413**. |
| Path traversal | 2 | `../../../../etc/passwd.csv` is stored as `passwd.csv`; `..\..\windows\win.csv` is stored as `win.csv`. |
| Dataset ID handling | 18 | `../../etc`, URL-encoded traversal, 65 characters, a null byte, shell metacharacters, `<script>`, backslashes and non-ASCII on two endpoints all return 400/404, never 500 and never a stack trace. |
| XML / XLSX | 3 | Billion-laughs entity expansion rejected in 0.2 s. **XXE** (`file:///etc/passwd`) gives 400 and nothing leaked. A 3 MB workbook that expands to 400,000 rows is handled in 4.9 s and then limited to 100,000 rows **with a warning**. The XML parser is expat 2.9.0, which has built-in amplification limits; no external entities are resolved. |
| Errors / information exposure | 6 | Malformed JSON, absurd numbers, a 200 KB question and unknown routes give no stack traces. `TRACE` gives 405. `/docs` gives **404** in production. `/health` reveals no paths. |
| Headers / CORS / rate limiting | 7 | All four headers are present. The `Server` banner has no version. A preflight from `https://evil.example` is refused (no `Access-Control-Allow-Origin`). 14 rapid uploads produce **429** with `Retry-After`. |
| Dashboard | — | HTML in the file name or a column name is escaped. Streamlit XSRF protection is on (default; not disabled in `.streamlit/config.toml`). The password gate holds: a wrong password stays locked (Chrome, in Docker). |
| Docker | — | uid/gid 10001 in both containers. The app user can write `data/` and `models/` but **not** the code. The key is never written to container logs. Security controls are still active after a restart (`DEPLOYMENT_READINESS_REPORT.md`). |
| SQL injection | — | **Not applicable.** No user input reaches SQL. The only SQL is a fixed demo-pipeline step (`src/utils/database.py`). |
| Secrets | — | `.env`, `secrets.toml`, keys and uploads are excluded from the ZIP (`FINAL_FILE_MANIFEST.txt`). `.env.example` has no secret values (tested). |

---

# Security audit report (upgrade PART 05)

**Scope.** The full project at the PART 05 working copy, with emphasis on the code the upgrade
added in PARTs 01–04. It builds on the baseline's `FINAL_SECURITY_AUDIT_REPORT.md`, which is
kept unchanged.

**Method.** Static analysis, a dependency vulnerability scan, a code review of the controls, and
the existing and new security tests. Live API checks against the clean-environment server are in
`DEPLOYMENT_READINESS_REPORT.md` §3.

**Conclusion.** No security control was weakened, and no known-vulnerable dependency was found.
Two findings in upgrade code were corrected. The remaining items are accepted risks or design
limits, listed in §5. This is **not** a penetration test.

## 1. Tools that were actually run

| Tool | Version | Scope | Result |
|---|---|---|---|
| bandit | 1.9.4 | `src api dashboard config app.py` (34,880 LOC) | 39 findings: 2 high, 3 medium, 34 low (details in §2) |
| pip-audit | latest from PyPI, run 2026-11-22 | all 71 packages installed in the verified environment, including the optional ones (`pip freeze`) | **No known vulnerabilities found** (0 of 71 packages affected) |
| pytest security suites | — | `test_ultimate_security.py`, `test_final_security_hardening.py`, `test_api.py`, `test_phase1_isolation.py`, `test_legacy_cache_isolation.py`, `test_ultimate_upload_flow.py` + PART 05 escaping tests | Pass; counts in `FINAL_TEST_REPORT.md` |

pip-audit checks the PyPI/OSV advisory databases as of the run date. New advisories may appear
later, so re-run it before any deployment:
`pip install pip-audit && pip-audit -r requirements.txt`.

## 2. bandit findings and how they were handled

| Severity | ID | Location | Assessment | Action |
|---|---|---|---|---|
| High | B324 SHA-1 | `dashboard/components/model_page.py:71` (PART 02) | Cache key for the session memo, not a security use | **Fixed:** `usedforsecurity=False` |
| High | B324 SHA-1 | `src/data_engine/role_engine.py:638` (PART 02) | Duplicate-column detection, not a security use | **Fixed:** `usedforsecurity=False` |
| Medium | B301 pickle | `src/models/channel_contribution.py:43`, `src/models/model_evaluation.py:31`, `src/optimization/budget_optimizer.py:50` (baseline) | They load `models/mmm_model.pkl`, which the project's own pipeline writes. User uploads are never unpickled. | Accepted. Documented below: never put untrusted files in `models/`. |
| Low | B110/B112 try-except-pass/continue (27) | various | Deliberate fallbacks in rendering and optional features | Accepted |
| Low | B403/B404/B603 (6) | pickle import; `subprocess` in `app.py` | `subprocess.run([sys.executable, step])` uses a fixed list of project scripts, with no shell and no user input | Accepted |
| Low | B101 assert (1) | — | Not used for security decisions | Accepted |

## 3. Controls reviewed

| Area | Control (file) | Status after the upgrade |
|---|---|---|
| Authentication (API) | `X-API-Key` with constant-time `hmac.compare_digest`; production refuses to start without a key (`api/main.py`) | `api/main.py` is **byte-identical** to the baseline |
| Authentication (dashboard) | PBKDF2-SHA256 shared password; required in production (`dashboard/components/auth.py`) | Unchanged |
| CORS | Explicit origins; warning on `*` | Unchanged |
| Rate limiting | Per-IP limits for upload, compute and other requests, returning 429 with `Retry-After` | Unchanged |
| Upload size | API: `MARKETPULSE_MAX_UPLOAD_MB` (50), returning 413. Dashboard: Streamlit `maxUploadSize` (200 MB). | Unchanged (the difference is documented) |
| Row limit | API: `MARKETPULSE_MAX_ROWS` (100,000) | Unchanged |
| Path safety | Dataset IDs must match `^[a-zA-Z0-9_-]{1,64}$` (`validate_dataset_id`). The new `role_mapping.json` and currency files are written only inside the validated dataset directory. | New writes use the same validated paths |
| Data isolation | Per-dataset directories; page caches cleared on dataset switch. The new session memo (`model_page.py`) is keyed by dataset ID and signature. | Covered by the isolation suites, which pass |
| Output encoding (XSS) | The new UI writes HTML only through `html.escape` / `md_inline`. **PART 05 test:** a column named `TV_Spend<img src=x onerror=alert(1)>` was rendered on 7 pages, and no HTML-enabled element contained the raw tag. | **New test** (`tests/test_upgrade_part05.py`) |
| Error handling | `safe_page` wrapper. The API returns structured error bodies with no stack traces. | No exceptions or fallback pages found on 16 pages × 4 datasets (AppTest), nor in Chrome (`FINAL_TEST_REPORT.md` §4) |
| Logging | Startup security logs give the mode, never key values. No new logging of data values was added. | Reviewed |
| Secrets in the release | `.env`, `secrets.toml`, caches, uploads and logs are excluded by the zip builder and `.dockerignore` | Verified in `FINAL_FILE_MANIFEST.txt` |
| Container user | Non-root (uid 10001, or the owner of the data bind mount) via the entrypoint | Unchanged; see `DEPLOYMENT_READINESS_REPORT.md` |
| Optional ML libraries | XGBoost, LightGBM and CatBoost are imported lazily and fail closed ("unavailable") | New; pip-audit clean |

## 4. Upgrade-specific review points

- **No new network calls.** The upgrade adds no outbound HTTP, no LLM calls and no telemetry.
- **No `eval` or `exec`, and no dynamic imports from user data.** Classifier keys come from a
  fixed registry (`src/predictive/classifiers.py`).
- **Remedies A–D change only the user's own limit inputs, and only on an explicit click.**
  The feasibility module does not modify its inputs. A test checks this:
  `test_inputs_never_modified`.
- **Numbers from uploads are never used as money unless a currency exists.** This prevents
  misleading financial output (an integrity concern rather than confidentiality).

## 5. Residual risks and accepted limitations

1. **Pickle model files.** `models/*.pkl` must come only from this project's pipeline.
   Never mount untrusted files there.
2. **Shared dashboard password.** There are no per-user accounts, roles or audit trail
   (baseline design).
3. **In-memory API state.** The API is limited to a single worker. A restart clears uploads that
   were not yet confirmed (baseline design).
4. **Dashboard upload limit.** It is 200 MB, larger than the API's 50 MB. A very large file can
   take minutes in *Confirm & Analyze* (see `PERFORMANCE_REPORT.md`). For a shared deployment,
   lower `server.maxUploadSize` to match your needs.
5. **Plotly labels** take column names as text. Plotly renders only a small whitelist of tags
   (such as `<b>` and `<br>`) and does not run scripts. This was not separately fuzzed.
6. **Not performed:** a penetration test, DAST scanning, TLS configuration testing, or a
   container image CVE scan. The image could not be built from the official base image
   (Docker Hub is unreachable here).
