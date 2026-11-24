# Final test report: MarketPulse AI final verified release (PARTs 01–06)

**Date:** 2026-11-23 (UTC).
**Platform:** Linux x86-64, Python 3.13.18, pytest 9.1.1.
**Environment:** a **brand-new virtual environment** built from `requirements.txt` and `requirements-optional.txt`
(74 packages). Key versions: streamlit 1.67.0, pandas 3.1.1, scikit-learn 1.9.1, fastapi 0.165.0, starlette 1.10.0,
httpx2 2.16.2, xgboost-cpu 3.4.1, lightgbm 4.7.0, catboost 1.2.10. The full list is at the end of
`FINAL_TEST_EXECUTION_LOG.txt`.

**Code under test.** A **fresh extraction** of the PART 06 stage package (SHA-256 `3f4ecdbc…7196`), run with
`python -I -m pytest tests -v -rs`. The final `MarketPulse_AI_FINAL_RELEASE_VERIFIED.zip` contains byte-identical
code, tests, data and configuration. Only documentation files differ, and each one is marked `[D]` in
`FINAL_FILE_MANIFEST.txt`.

## 1. Totals: full suite, clean environment

| Collected | Passed | Failed | Errors | Skipped | Not run | Warnings | Duration |
|---|---|---|---|---|---|---|---|
| **3,522** | **3,522** | **0** | **0** | **0** | **0** | **0** | 1,132.6 s |

- **Skipped: 0.** The 3 real-Kaggle tests that were skipped in PART 05 now **run and pass**, because the user's
  real file is included. They skip with an explanation only if that file is removed.
- **Warnings: 0.** The single warning in every earlier run (Starlette TestClient: "install `httpx2`") was resolved
  by adding `httpx2` to `requirements.txt`.
- The complete verbose output is in `FINAL_TEST_EXECUTION_LOG.txt`.

## 2. Breakdown

| Test files | Passed |
|---|---|
| 57 files from the baseline, plus earlier upgrade additions to them | 3,043 |
| `test_upgrade_part01.py` (navigation, Statistics, currency, labels) | 107 |
| `test_upgrade_part02.py` (role engine, modes, 4 models, optimiser fix), incl. **`TestRealKaggleFile` 3/3** | 148 |
| `test_upgrade_part03.py` (Z-test, 12 classifiers, CV/holdout) | 58 |
| `test_upgrade_part04.py` (feasibility, remedies, reserve, statuses, scenarios) | 47 |
| `test_upgrade_part05.py` (app-wide chart guard, HTML escaping) | 46 |
| `test_upgrade_part06.py` (real file, quality scores, Z-test, API security, CSV injection, optimiser matrix) | 73 |
| **Total** | **3,522** |

**By area.** This is grouped by test-file name, and each test is counted once:

| Area | Tests |
|---|---|
| Models and analytics | 2,248 |
| Dashboard / UI | 401 |
| Dataset compatibility | 328 |
| Other regression guards | 262 |
| Security | 240 |
| Docker / entrypoint | 30 |
| API | 13 |

Further API tests sit inside the security, upgrade and integration files, for example `test_api_*` and
`TestLegacyIntegration`.

**Progression** (each part tested on a fresh extraction of its own package):

| Part | Passed | Failed | Skipped | Warnings |
|---|---|---|---|---|
| Baseline | 3,030 | 0 | 0 | 1 |
| PART 01 | 3,150 | 0 | 0 | 1 |
| PART 02 | 3,295 | 0 | 3 | 1 |
| PART 03 | 3,353 | 0 | 3 | 1 |
| PART 04 | 3,400 | 0 | 3 | 1 |
| PART 05 | 3,446 | 0 | 3 | 1 |
| **PART 06** | **3,522** | **0** | **0** | **0** |

No test was deleted or weakened. Existing tests changed in PART 06 for Windows compatibility only:

- explicit UTF-8 reads (65 call sites in 18 files);
- the `os.geteuid` guard;
- skipping the bash-only entrypoint tests on Windows.

The tests run identically on Linux. Details are in `CHANGELOG.md`.

## 3. Other test runs

| Run | Environment | Result |
|---|---|---|
| **Windows-encoding emulation:** the whole suite under a strict non-UTF-8 locale (`LC_ALL=C`, `-X utf8=0`; ASCII, stricter than the Windows cp1252 default) | Same fresh extraction and venv | **3,522 passed, 0 failed, 0 skipped** (1,085 s) |
| Before the encoding fixes, same emulation | Working copy | 3,471 passed, **34 failed**. All were tests reading project files without an encoding (`app.py` cannot even be decoded as cp1252), plus `os.geteuid` at collection. The **application code** passed. Fixed as listed above. |
| **Windows scripts:** `scripts/windows/validate_windows.ps1 -Quick`, PowerShell 7.5.4 **on Linux**, from-scratch setup on a fresh extraction | `.venv` created by `setup.ps1` from PyPI | **All 20 steps PASS:** setup (venv, core and optional install from PyPI), 8 smoke checks, quick pytest (164 passed), `data` writable, API start/health, upload without a key returns 401, CSV and XLSX uploads accepted, an invalid ID returns 400, the dashboard starts with a home page 200, and both ports are released after shutdown. **This is not a Windows run.** |
| **Real Kaggle file, end to end** (`tools/part06/real_dataset_validation.py`) | Working copy | 12/12 steps ok: detection, mapping (22/22 columns correct), targets, mode, capabilities, models for both targets, legacy pipeline, 12 classifiers, inferential |
| **Real file, all pages (AppTest)**: 2 configurations × 18 renders | Working copy | 36 renders, 86 charts, 0 label/currency problems, 0 exceptions |
| **Real file in Chrome**: CSV and XLSX uploads, 16 pages each, optimiser click, 6 downloads, target filter | Working copy, real server | 0 exceptions, 0 error boxes, 0 console errors; every download parsed; optimiser *Solved* |
| **Concurrency**: 3 simultaneous Chrome users analysing the real file; 30 simultaneous API requests | Real servers | All succeeded (`PERFORMANCE_REPORT.md`) |
| **Penetration-style probes** (65) against the production-mode API | Local only | After the fixes, 65/65 pass, 0 findings (`SECURITY_AUDIT_REPORT.md`) |
| **Docker**: two complete compose verifications (default and optional-ML images), 10 required checks + logs, and a Chrome sign-in pass of 16 pages | Fresh extraction of the stage ZIP, **local stand-in base image** | All checks pass. One check was mis-run as root and re-run correctly. The official base image is **blocked** (`DEPLOYMENT_READINESS_REPORT.md`) |
| **Performance before and after**: Confirm & Analyze at 10k/50k/100k rows, with all 17 output files compared | Working copy | 12–19% faster, 0.4–0.5 GB less peak memory, **0 output differences** |

## 4. Defects found and fixed in PART 06

1. **Real file:** content-score columns were read as channel ad spend, which flipped the dataset to Spend mode
   (the replica had no such columns). Fixed with the `quality_score` role.
2. **CSV formula injection** in the page 04/05/07 downloads (medium).
3. **Silent row truncation** in API uploads (medium, integrity).
4. API docs exposed in production; missing security headers; leftover unparseable uploads (low).
5. **Windows:** 34 test-side encoding failures and `os.geteuid` at collection.
6. **Windows scripts:** hard-coded `\` paths and a Windows-only `-WindowStyle`, found by actually running the
   scripts.
7. The Starlette TestClient deprecation warning, resolved with `httpx2`.

## 5. Limits of this testing

- Windows, macOS and browsers other than Chrome were not tested.
- The official Docker base image is blocked.
- Load testing went no further than 3 dashboard users and 30 API requests.
- No external penetration test was performed.

See `KNOWN_LIMITATIONS.md`.
