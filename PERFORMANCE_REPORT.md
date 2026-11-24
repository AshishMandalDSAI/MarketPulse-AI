# PART 06: performance investigation and justified optimisation

**Machine:** Linux, 2 vCPU, 8 GB RAM, Python 3.13.18.
**Data:** synthetic spend datasets (208 weeks × N stores, 9 columns) and the user's real Kaggle file.
**Scripts:** the benchmark and output comparison scripts were run from the build workspace (`part06/bench_rfa.py`,
`part06/compare_outputs.py`); the profiler and session timing scripts were the same; the logic is described below.

## 1. Where the time went (profile at 99,840 rows, before)

| Component (baseline code) | Share | Notes |
|---|---|---|
| Automatic classification (Logistic, Random Forest, Gradient Boosting) | ~40% | Gradient boosting dominates; it has no parallel mode |
| Segmentation: exact silhouette score on all rows (O(n²)) | ~32% | 8 calls: K selection on a disclosed 10k sample, then the final score on all rows |
| Automatic regression | ~15% | Random forest and gradient boosting |
| Time-series anomaly spike/drop detection | ~8% | Python loop with about 730,000 single-row pandas lookups |

## 2. Changes (no sampling, no shortcuts, no accuracy change)

| Change | Why it is safe |
|---|---|
| Random forests in the auto engines: parallel **fit**, sequential **predict** | Identical trees for a fixed seed; sequential prediction keeps averaged outputs bit-identical (verified). Parallel prediction was rejected because it changes the last floating-point digits. |
| Exact silhouette: parallel distance chunks with a 256 MB working chunk | Same value to the last bit (verified); peak memory falls from 1.2 GB to 0.7 GB at 100k rows. |
| Spike/drop loop: positional arrays in place of `.loc` per row | Same values, same order, same messages. |

Rejected:

- Sampling the final silhouette, which would be hidden sampling.
- Switching to histogram gradient boosting, which gives different models.

## 3. Before and after: *Confirm & Analyze* (`run_full_analysis`)

| Rows | Before | After | Change | Peak memory before → after |
|---|---|---|---|---|
| 9,984 | 26.7 s | 21.5 s | −19% | 978 MB → 732 MB |
| 49,920 | 123.1 s | 104.4 s | −15% | 1,294 MB → 786 MB |
| 99,840 | 291.0 s | 255.4 s | −12% | 1,331 MB → 904 MB |

**Output equality.** All 17 output files per size were compared after normalising the random run ID, the IDs
derived from it, and the "completed in N s" text. Result: **0 differences** at all three sizes.

On machines with more cores, the parallel parts scale further. Gradient boosting stays single-threaded, so it
remains the floor.

## 4. Other measurements (after)

| Measurement | Result |
|---|---|
| Upload profiling + role mapping (`profile_and_map` + `suggest_roles`) | 0.02–0.08 s for 10k–100k rows |
| Real Kaggle file (11,232 rows), single user in Chrome | first page 2.5 s; upload, validation and mapping 1.7 s; *Confirm & Analyze* 21.8 s; page 04 (10 charts) 3.8 s |
| **3 simultaneous users**, real file, Chrome | All succeeded: *Confirm & Analyze* 47–48 s each (2 vCPUs shared), page 04 9.5–10.5 s, 0 errors |
| API: 30 simultaneous requests (`/kpis`, `/channels`, `/health`) | all 200; median 103 ms, p95 128 ms |
| Real file models | driver model about 3 s per target; legacy pipeline 23 s; 12 classifiers on 11k rows about 7 min (including the regression run) |
| Page render (AppTest, demo) | unchanged from PART 05 (0.1–1.3 s) |

## 5. Not measured

- Load beyond 3 dashboard users or 30 API requests.
- Windows or macOS timings.

---

# Performance report (upgrade PART 05)

All figures below were measured on 2026-11-22 on the build machine: Linux, 2 vCPU, 7 GB RAM,
Python 3.13.18, the core requirements plus the optional XGBoost, LightGBM and CatBoost. They are
single runs on synthetic data, so treat them as order-of-magnitude values, not benchmarks. The
baseline's `FINAL_PERFORMANCE_REPORT.md` (made with `tools/benchmark.py`) is kept unchanged.

Scripts are in `tools/part05/`:

- `perf.py ROOT OUT.json ROWS` measures the pipeline stages.
- `perf_rfa.py ROOT ROWS` profiles *Confirm & Analyze*.
- `audit_pages.py ROOT OUT.json` runs the AppTest page and chart audit.
- `browser_all.py BASE OUT TAG` runs the Chrome pass with Playwright.

## 1. Start-up and idle resources

| Item | Result |
|---|---|
| Dashboard (`streamlit run app.py`): launch until `/_stcore/health` returns 200 | 0.9 s |
| API (`uvicorn api.main:app`): launch until `/health` returns 200 | 0.9 s |
| Dashboard process RSS after start | ≈ 79 MB |
| API process RSS after start | ≈ 161 MB |
| API `GET /kpis` (demo), 3 sequential requests | 1–2 ms each |

## 2. Page render times, demo dataset (Streamlit AppTest, server side, warm)

| Page | Seconds |
|---|---|
| 00 Upload | 0.21 |
| 01 Statistics: Descriptive / Inferential / Predictive | 1.33 / 0.23 / 0.18 |
| 02 Executive Overview / 03 Marketing Performance | 0.36 / 0.41 |
| 04 Effectiveness / 05 Channel Performance | 0.81 / 0.78 |
| 06 Resource Optimizer / 07 Scenario Simulator | 0.64 / 1.03 |
| 08–15 | 0.11–0.15 each |

These times do not include browser drawing. In the Chrome pass every page settled within the
script's idle wait, and the optimizer click returned *Solved* (`FINAL_TEST_REPORT.md` §4).

## 3. Large datasets

The synthetic spend dataset has 9 columns: week, store, region, 4 spend channels, price and
revenue. Its rows are 208 weeks × N stores.

| Stage | 9,984 rows | 99,840 rows |
|---|---|---|
| Upload profiling (`profile_and_map`) | 0.02 s | 0.07 s |
| Role engine (`suggest_roles`) | 0.04 s | 0.24 s |
| **Confirm & Analyze** (`run_full_analysis`, baseline Phase 5/6 analysis) | **23.6 s** | **271.5 s** |
| Model context | <0.01 s | <0.01 s |
| Driver response model (OLS, HC3, fixed effects) | 0.27 s | 0.36 s |
| Spend MMM | 0.06 s | 0.15 s |
| Optimiser, multi-start (status) | 3.3 s (*feasible, not proven*) | 2.9 s (*feasible, not proven*) |
| Scenario sensitivity | 0.23 s | 0.13 s |
| Predictive regression, default models, 20,000-row sample | 45.4 s | 91.4 s |
| Peak RSS of the process | 979 MB | 1,288 MB |

The model pages (04–07) aggregate to the time grain, so their cost barely grows with the row
count. The optimiser reported *feasible, not proven* because its starts did not all agree on
this noisy synthetic data. The status is shown honestly, not upgraded to "optimal".

**12 classifiers (Predictive tab, all models selected), 8 features plus 1 categorical, binary target:**

| Rows | Time | Holdout scored |
|---|---|---|
| 2,000 | 29.3 s | 12 models + baseline |
| 10,000 | 84.9 s | 12 models + baseline |

## 4. Where the time goes: *Confirm & Analyze* compared with the baseline

The function was profiled at 29,952 rows, with the same data and the same environment, on the
**untouched baseline** and on this release:

| Code | `run_full_analysis` |
|---|---|
| Baseline (ULTIMATE FINAL) | 74.9 s |
| This release | 75.3 s |

This is **no regression.** The upgrade did not change this code path. About 60% of the time (44 s of 75 s) is
spent in the baseline's automatic classification and regression engines (gradient boosting and
random forest in `src/prediction/`). Anomaly detection and segmentation (silhouette scoring)
take most of the rest.

**Recommendation, not implemented because it would change baseline behaviour:** sample or
cap rows in the automatic Phase 5/6 engines (anomaly detection already samples to 50,000 rows),
or run them in the background. Until then, for uploads above roughly 50,000 rows, expect a
wait of several minutes after *Confirm & Analyze*.

## 5. Workers and concurrency

- The API is pinned to one worker by design: the upload registry and rate limiter are
  per-process, and production refuses `WEB_CONCURRENCY > 1`.
- Streamlit runs each session in its own script thread. Long computations (Confirm & Analyze,
  12-classifier runs) hold that session but not others. A CPU-bound run slows other sessions on a
  2-vCPU host. Concurrent-user load testing was **not performed**.

## 6. Not measured

- Load or concurrency tests.
- Windows or macOS.
- Browsers other than Chrome.
- Container resource limits, because the image was not built from the official base image.
