# MarketPulse AI: Performance Report

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22

**Machine:** Linux container, 2 vCPU, 7 GB RAM, Python 3.13.18.

**Method [EXECUTED]:** `python tools/benchmark.py <rows>`. It generates a synthetic retail CSV with about 1% missing cells and times the same stages the Upload page runs:

1. read
2. quality assessment
3. role mapping
4. cleaning
5. the full analysis (`run_full_analysis`: MMM, forecasting, KPIs, insights, predictive tasks, anomaly and driver analysis, decision intelligence)

Peak memory is the process's maximum RSS. Each size ran once (no repeats). Other jobs, such as browser tests, were sometimes running at the same time, so treat timings as ±20% indications.

## 1. Results

| Rows | CSV size | Read | Quality | Mapping | Cleaning | Full analysis | Total | Peak RSS before analysis | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.5 MB | 0.00 s | 0.15 s | 0.04 s | 0.04 s | 16.8 s | **17.0 s** | 128 MB | **984 MB** |
| 100,000 | 5.1 MB | 0.03 s | 1.37 s | 0.29 s | 0.36 s | 234.0 s | **236.1 s** | 195 MB | **1,316 MB** |
| 500,000 | 25.6 MB | 0.10 s | 5.96 s | 1.09 s | 1.63 s | 189.6 s | **198.4 s** | 394 MB | **1,542 MB** |
| 1,000,000 | 51.1 MB | 0.28 s | 12.94 s | 2.77 s | 3.92 s | 178.9 s | **198.8 s** | 691 MB | **1,781 MB** |

All four sizes completed. Raw output: `bench.jsonl`, shipped alongside the release reports.

### Why 100k rows is slower than 500k and 1M

Several stages deliberately sample large inputs. Above their caps, analysis time stops growing with row count:

- anomaly detection, driver analysis and the MMM engine work on at most 50,000 rows (a random sample with seed 42, where they sample);
- the regression, classification and segmentation engines sample down to 100,000 rows;
- the forecasting engine keeps the most recent 100,000 periods;
- segmentation chooses k on a 5,000-row sample (see §2).

At 100k rows some stages still run on the full data, while at 500k and 1M every capped stage is already sampling. Each sampling stage states in its output that it sampled. Concurrent load during the 100k run may also have contributed; it was not isolated.

The **front stages** (read, quality, mapping, cleaning) scale roughly linearly: about 20 s at 1M rows.

## 2. Optimisations made

| Change | Before | After | Evidence |
|---|---|---|---|
| Segmentation chooses k with silhouette scores on a 5,000-row sample. The final model is still fitted on all rows. | 122 s for the analysis of 20k rows (O(n²) silhouette) | 48 s | [EXECUTED]; 33 segmentation tests pass |
| Past-only median fill uses an expanding median. | O(n²) Python loop | vectorised O(n log n) | [TEST] identical results to the reference |
| Upload analysis (`_analyse_file`) cached by file hash with `st.cache_data`. | re-analysed on every widget interaction | computed once per file | [STATIC] |
| Forecast engine keeps the most recent contiguous `MAX_ROWS` rows. | random rows, which broke time order | contiguous recent window | [TEST] |

Dashboard data loaders are keyed by dataset ID and a file fingerprint (absolute path, mtime_ns, size), so a new upload or re-analysis invalidates the caches [STATIC; existing tests].

## 3. Browser timings [EXECUTED]

Page switch times, measured by Playwright with Chrome on the same machine (time until Streamlit is idle), are 1.5–2.6 s across all 18 pages for both the demo and an uploaded 2,000-row dataset. Full analysis of that 2,000-row upload took about 30 s server-side.

## 4. Limits: what is NOT claimed

- Only the sizes above were tested. No claim is made beyond 1,000,000 rows or for files with many more columns (the benchmark has 8).
- Above the sampling caps, some results describe a sample, not every row.
- Measured peak memory is about 1.0 GB at 10k rows and 1.8 GB at 1M rows. Hosts with less memory, such as small Streamlit Community Cloud apps (check their current limits), may fail on large files. Not tested on Streamlit Cloud.
- The API limits uploads to 50 MB and analyses the first 100,000 rows by default (`MARKETPULSE_MAX_UPLOAD_MB`, `MARKETPULSE_MAX_ROWS`). The dashboard path has no row cap, only Streamlit's 200 MB upload limit.
- No concurrent-user load testing was done.
