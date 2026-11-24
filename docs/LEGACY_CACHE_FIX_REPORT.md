# Legacy Cache Fix, Part 1: Implementation Report

MarketPulse AI, Phase 8. Legacy cache isolation fix, as authorized after the Part 0 investigation. Phase 8 Part 2 was not started.

## 1. Result

The 19 legacy `@st.cache_data` loaders in `dashboard/components/data_loader.py` no longer leak one user's dataset to another. Each public loader keeps its name, zero-argument signature, return type, empty-value behaviour and `.clear()` method. Callers (pages, `graceful.py`, `app.py`) are untouched.

| Check | Result |
|---|---|
| New tests (`tests/test_legacy_cache_isolation.py`) | **111 passed** |
| Same new tests against the **baseline** `data_loader.py` | **45 failed, 66 passed** (the tests detect the bug) |
| Phase 1 isolation, dataset switching, dataset-first platform tests | 54 passed |
| Phase 8 Part 1 foundation tests | 154 passed, 4 warnings |
| Phase 8 Part 1 availability tests | 96 passed |
| Complete suite | **1,920 passed, 0 failed, 0 skipped, 0 errors**, 16 warnings |
| Baseline suite | 1,809 passed. The difference is exactly the 111 new tests: 1,809 + 111 = 1,920. |

## 2. Baseline verification (before any change)

| Item | Value |
|---|---|
| Baseline ZIP | `MarketPulse_AI_Phase8_Part1_Correction1_Backup.zip` |
| ZIP SHA-256 | `b52e51cfb625bf6e2730c8507d5d464d4ea9ce0a5cf2f19baa22636235f24d07` (matches expected) |
| `dashboard/components/data_loader.py` SHA-256 | `4ac09b12eddaeb9dbaa383d68c44ece28909dc035fd9fb9479252a2b53660026` (matches expected) |
| Original manifest | 209 files, recorded before implementation |

## 3. The problem (recap)

Streamlit's cache is shared by all sessions of a server process and is keyed by the function and its arguments only. The loaders took no arguments and read the active dataset from `st.session_state` inside the function, so the dataset was not part of the key. After user B loaded a dataset, user A's next rerun received B's data.

## 4. What changed

### `dashboard/components/data_loader.py` (modified, the only source change)

- **Public loaders** (19): now uncached wrappers. Each resolves the caller's context (dataset id, processed directory, models directory), the date column where the loader uses one, and the fingerprints of the files it reads. It then calls a private cached worker with all of these as explicit arguments.
- **Private workers** (19, named `_*_worker`): `@st.cache_data`, one per loader. They never read `st.session_state` or `session_paths`, and never call a public loader.
- **Cache key** contains, as applicable: dataset id, processed/models directory, date column, and for every input file a fingerprint `(absolute resolved path, mtime_ns, size)`.
- **Missing file:** the deterministic fingerprint `(absolute path, -1, -1)`, with no exception. A file that appears later changes the key.
- **Two-file loaders** include both candidate files: `load_marketing_data` (clean and standardized csv), `load_channel_attribution` (processed and models copy), `load_forecast` (processed and models copy), and `load_dynamic_analytics` (metadata plus both data files).
- **`load_dynamic_analytics`:** its worker takes one explicit context and reads the metadata and the data table from the same processed directory through an uncached helper, `_read_marketing_df`, shared with `load_marketing_data`. It cannot combine two datasets' inputs.
- **`.clear()`:** each public loader has `.clear`, delegating to its private worker's `clear`. Verified callable for all 19.
- **File contents are never hashed.** Only `os.stat`.
- **Unchanged, byte for byte:** `_safe_read_csv`, `_safe_read_json`, `has_capability`, `capability_reason`, `load_analysis_context`, `pipeline_ready`, `DATA_DIR`, `MODELS_DIR`.
- The module docstring gained one paragraph describing the fix, and the imports gained `os` and `typing.NamedTuple`/`Optional`.

The decorator, return types and empty values are unchanged. For example, `load_generic_kpis` still returns `[]` when the file is missing, and `load_dynamic_analytics` still returns `{}`.

### Added files

- `tests/test_legacy_cache_isolation.py` (671 lines, 111 tests; see section 12 for the review correction)
- `docs/LEGACY_CACHE_FIX_REPORT.md` (this report)

### Authorized decisions applied

- **D1:** file modification information is part of every key (`mtime_ns` and size).
- **D2:** the existing `st.cache_data.clear()` calls (`session_paths.py`, `app.py`) are unchanged.
- **D3:** every public loader keeps a callable `.clear()`.

## 5. Changed-file manifest

| Path | Change | SHA-256 |
|---|---|---|
| `dashboard/components/data_loader.py` | modified (baseline 289 lines, now 541) | `ba4df412c49e9a59bec9722eb2a74e9bf89058d3a0bcd627a7de066bb0666747` |
| `tests/test_legacy_cache_isolation.py` | added (671 lines, after the section 12 correction; 587 lines and `50f26dcc…31c1` in the first delivery) | `a5425bdfe3c30e77e09ce1c9a3a5064f5e27be381b82596daf651fbeda53ea20` |
| `docs/LEGACY_CACHE_FIX_REPORT.md` | added | not listed: a file cannot contain its own hash. The standalone copy's hash is given in the delivery message. |

Baseline `data_loader.py` was `4ac09b12…0026`. Against it, 40 lines were removed and 292 added.

## 6. Test strategy and coverage

`tests/test_legacy_cache_isolation.py` simulates users with Streamlit `AppTest`: each `AppTest` is one session, and the cache is process-wide, as for two browser tabs on one server. Expected values are read directly from the dataset files, independently of the loaders. Switching is tested two ways: through the supported `set_active_dataset()` API (which clears the cache), and by assigning the session's dataset directly (so **nothing** clears the cache and only the key can keep datasets apart).

| Section | Class | Tests | What it proves |
|---|---|---|---|
| A | `TestCrossSessionIsolation` | 6 | User A loads A, user B loads B, user A loads again and still gets A. Row counts, column names and 18 loaders are compared. Both orders, both modes, plus demo vs upload. |
| B | `TestDatasetSwitching` | 10 | A to B, B to A, demo to upload to demo, upload to demo to upload, and a five-step sequence, in both modes. |
| C | `TestFileModification` | 7 | A changed file is re-read with no global clear (csv and json, and a missing file that appears). Cache **hits** while path, mtime and size are unchanged. A change of mtime alone, or of size alone, changes the key. `loader.clear()` forces a re-read. Fingerprint properties. |
| D | `TestUploadReplacement` | 2 | Re-processing the same file under a new dataset id never returns the old id's cached data. The upload page still creates a fresh id for every analysis. |
| E | `TestSessionReset` | 2 | `reset_analysis()` in one session cannot expose another session's data or remove it. After a reset the old dataset cannot reappear. |
| F | `TestPublicApiCompatibility` | 59 | For all 19 loaders: name, zero parameters, return annotation, callable `.clear()` that does not raise, the same empty value when the dataset has no files, plain wrapper (not itself cached). |
| G | `TestNestedLoaderConsistency` | 5 | Workers and helpers contain no `session_paths` or `session_state` reference and call no public loader (checked from the source). `load_dynamic_analytics` reads all inputs from one dataset directory. Its result belongs to the active dataset in both orders, and follows an input-file change. |
| H | `TestAllFixtureDatasets` | 20 | All 16 existing fixtures run through the real pipeline. Every loader matches the files per fixture. Walking all 16 datasets in one session, forward and reverse, never mixes them. Two sessions interleaved over all 16 stay isolated. |

**Detection check:** run against the baseline `data_loader.py`, the same file gives 45 failures. They are the leak scenarios (A, B in direct mode, D, E, G, H sequences), the stale-file scenarios (C), and the structural checks of the new design. The 66 tests that pass on both versions cover unchanged behaviour: return types, empty values, per-dataset file contents, and the supported switch API. That is evidence that behaviour was preserved.

**Expected-value note:** for `load_forecast` and `load_actual_vs_predicted`, a dataset whose date column is not in the forecast file makes pandas fail the date parse, and the loader returns an empty table (for `load_forecast`, after falling back to the models copy). This is existing baseline behaviour, unchanged by the fix. The expected values in the tests model it, and the tests pass on both the baseline and the fixed code.

## 7. Regression testing

Environment: Python 3.13.18, Ubuntu 24.04, pytest 9.1.1, streamlit 1.67.0, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1. No `anthropic` package (optional, as in the baseline). No dependency changed. Commands ran from the project root with `-p no:cacheprovider`:

```
python3 -m pytest -q tests/test_legacy_cache_isolation.py
python3 -m pytest -q tests/test_phase1_isolation.py tests/test_dataset_switching.py tests/test_dataset_first_platform.py
python3 -m pytest -q tests/test_phase8_part1_foundation.py
python3 -m pytest -q tests/test_phase8_part1_availability.py
python3 -m pytest -q -rsxX
python3 -m pytest --collect-only -q          # 1920 tests collected
```

| Step | Result |
|---|---|
| 1. New legacy cache isolation tests | 111 passed in 52.25 s |
| 2. Phase 1 isolation, dataset switching, dataset-first platform | 54 passed in 18.20 s |
| 3. Phase 8 Part 1 foundation | 154 passed, 4 warnings in 35.29 s |
| 4. Phase 8 Part 1 availability | 96 passed in 1.33 s |
| 5. Complete suite | **1,920 passed, 0 failed, 0 skipped, 0 errors, 0 xfail**, 16 warnings, 124.87 s |

There is no pytest configuration file and no filter. All 1,920 collected tests ran. **Difference from the 1,809 baseline:** +111 passed, the new tests. No failure and no skip.

### Warnings (16, all pre-existing)

The 16 warnings are the same as in the accepted baseline run:
- 12 scikit-learn `InconsistentVersionWarning` (the shipped model was saved with 1.8.0, 1.9.1 is installed).
- 2 numpy `RuntimeWarning: invalid value encountered in divide` from the existing `test_constant_target`.
- 2 numpy `RuntimeWarning: invalid value encountered in reduce` from the existing `test_infinite_values_handled`.

The new tests produce no warnings of their own.

## 8. Performance

The cache still produces hits while the dataset id, path, mtime and size are unchanged (tested in section C), and changing the file changes the key. Overhead on a **cache hit**, measured with 300 calls per loader, median of 3 runs, same machine, before and after:

| | Baseline | Fixed |
|---|---|---|
| Mean per call, 19 loaders, demo dataset | 61.8 µs | 193.4 µs |
| Mean per call, 19 loaders, uploaded dataset | 64.5 µs | 202.7 µs |
| `load_marketing_data` per call (demo) | 222.9 µs | 547.4 µs |
| `load_kpis` per call (demo) | 7.1 µs | 136.0 µs |
| `_fingerprint` (path resolve plus `os.stat`), per call | not applicable | 18.7 µs |
| Bare `os.stat`, per call | not applicable | 0.8 µs |

The fix adds roughly **130 µs per loader call on a cache hit** (for one page rerun that calls about 5–10 loaders, around 1 ms). The `os.stat` itself is 0.8 µs. Most of the added cost is the context lookup, resolving paths, and Streamlit hashing the larger key. Cache misses cost the same file read as before. I did not optimize this further because the authorization limits the change to correctness. A cache hit is still orders of magnitude cheaper than reading a CSV.

## 9. Integrity

| Check | Result |
|---|---|
| Files in the baseline manifest | 209 |
| Files in the fixed project | 211 (209 baseline + 2 added) |
| Modified | 1: `dashboard/components/data_loader.py` |
| Added | 2: `tests/test_legacy_cache_isolation.py`, `docs/LEGACY_CACHE_FIX_REPORT.md` |
| Deleted | 0 |
| Byte-identical to baseline | 208 |
| Unauthorized files changed | none: `app.py`, `session_paths.py`, `stats_ui.py`, `graceful.py`, `dashboard/pages/*`, `src/*`, `api/*`, `config/*`, `requirements.txt` and every existing test are byte-identical |
| Dependencies | no change (`requirements.txt` untouched, nothing installed) |
| Formatting | no unrelated formatting changes: every function outside the 19 loaders is byte-identical (verified per function) |
| Generated files | `data/` and `__pycache__` created by test runs are excluded from the ZIP, as in earlier deliveries |
| `docker-entrypoint.sh` | execute bit preserved |

## 10. Limits

- Sessions are simulated in one process with `AppTest`. I did not run a live server with two real browsers. The behaviour matches the cache-key logic in the installed Streamlit 1.67.0.
- A file edited in place with the same size and the same `mtime_ns` is not noticed. Detecting that would require hashing contents, which the authorization excludes.
- The global `st.cache_data.clear()` calls are unchanged (decision D2), so one session's dataset switch still empties the shared cache for others. This now costs only a re-read, not correctness.
- Pages 14–17 on the `safe_page` error-fallback path now receive correctly keyed data, because the fallback calls the same loaders.

## 11. Rollback

Restore `dashboard/components/data_loader.py` to the baseline version (SHA-256 `4ac09b12eddaeb9dbaa383d68c44ece28909dc035fd9fb9479252a2b53660026`), and delete `tests/test_legacy_cache_isolation.py` and `docs/LEGACY_CACHE_FIX_REPORT.md`. The accepted baseline ZIP (`b52e51cf…cc6b`) is the full restore point.

## 12. Review correction (test file only)

Independent review of the first delivery found two regression tests that were too permissive. Only `tests/test_legacy_cache_isolation.py` was changed. `dashboard/components/data_loader.py` is byte-identical to the first delivery (SHA-256 `ba4df412c49e9a59bec9722eb2a74e9bf89058d3a0bcd627a7de066bb0666747`), and no existing test, dependency or other file was touched. The test count is unchanged (111): both tests were strengthened in place, and no existing assertion was removed or weakened.

### 12.1 `test_dynamic_analytics_cache_follows_input_file_changes`

- **Weakness:** the old final assertion, `before != after or before == {}`, passed whenever the first result was empty, so an empty result proved nothing.
- **Fixture:** the existing scratch dataset (`h_unusual_columns.csv`, 60 rows). I probed all 16 fixtures through the real pipeline: every one gives a non-empty dynamic analytics result, both before and after a file change.
- **Now asserts:**
  1. the initial result is non-empty, has the keys `dataset_id`, `kpis`, `charts`, `insights`, and its `dataset_id` matches the dataset's canonical metadata;
  2. with unchanged files a second load returns the identical result (a true cache hit);
  3. after changing the input table (same dataset id, **no global cache clear**) the result is non-empty, is not the stale cached object, and its content really differs from the old one;
  4. that no-clear result equals an independent recomputation (whole cache emptied, new session, same files);
  5. after restoring the original file content (again with no clear) the content is back to the original.
- **Finding:** dynamic analytics stamps every KPI, chart, insight and relationship with a freshly generated random `id` (`src/analytics/contracts.py`, `uuid4`). Two computations of identical data therefore differ in those ids. The content comparisons (3 to 5) drop the `id` keys with a small helper `_without_ids`. This is how the code behaves already; it is not a change.

### 12.2 `test_workers_and_helpers_never_touch_session_context`

- **Weakness:** the check only inspected each worker's own body, not the private helpers the workers call, so session access added to a helper would not have been noticed.
- **Kept:** every original assertion, including the count of 20 inspected functions.
- **Added:**
  1. the call graph from all 19 `*_worker` functions is followed transitively through module-level functions. It must reach the shared helpers `_read_marketing_df`, `_safe_read_csv` and `_safe_read_json` (so the walk is not vacuous), and must never reach a public wrapper, `_context`, `has_capability`, `capability_reason`, `load_analysis_context` or `pipeline_ready`;
  2. every reachable function body is inspected for any way of reaching the caller's session: `session_paths`, `_context`, `st`, `session_state`, the session-resolving functions (`get_dataset_id`, `get_processed_dir`, `get_models_dir`, `get_date_column`, `is_user_dataset_mode`, `get_active_dataset`), a public loader, or an import whose name contains "session". Function bodies are inspected, not the `@st.cache_data` decorator;
  3. a negative control: the same inspection must flag the intentional session access in the public wrappers `load_marketing_data`, `load_kpis` and `load_dynamic_analytics`.
- **Limit:** the inspection is static and covers this module's own functions. Code in other modules that a function imports (for example `src.analytics`) is not followed.

### 12.3 Proof that the corrected tests detect real faults

Each fault was introduced in a scratch copy of `data_loader.py` (never in the project), and the two tests were run:

| Scratch fault | Result |
|---|---|
| Dynamic worker ignores the input-file fingerprints (stale cache after a file change) | `test_dynamic_analytics_cache_follows_input_file_changes` fails |
| Dynamic analytics always returns an empty result (the old assertion would have passed this) | same test fails |
| The shared helper `_read_marketing_df` resolves the session dataset itself | `test_workers_and_helpers_never_touch_session_context` fails |
| `_safe_read_json`, reached only through the call chain, reads `st.session_state` | same test fails |
| Unmodified copy (control) | both pass |

### 12.4 Re-run after the correction

| Run | Result |
|---|---|
| Complete new test file | **111 passed** in 55.92 s |
| Complete suite (`pytest -q -rsxX`) | **1,920 passed**, 0 failed, 0 skipped, 0 errors, 16 warnings, 146.48 s |
| `pytest --collect-only` | 1,920 tests collected |

The 16 warnings are the same pre-existing ones as in section 7. Counts match the expected 1,920 (1,809 baseline + 111).

### 12.5 Integrity of the correction

- `dashboard/components/data_loader.py`: byte-identical to the first delivery.
- Differences from the first delivery: `tests/test_legacy_cache_isolation.py` (corrected) and this report (section 12 and the updated manifest rows). Nothing else.
- Differences from the accepted baseline are still exactly three paths: 1 modified (`data_loader.py`) and 2 added (the test file and this report).

## 13. Final independent audit (test file only)

A full audit of the delivered implementation was done against the code itself, not the earlier reports.

### 13.1 Implementation audit - no defect found

| Area | Finding |
|---|---|
| Cache isolation | All 19 public loaders resolve the caller's context and pass it to a keyed worker. No worker or helper reads `st.session_state` or `session_paths` (AST call-graph test). |
| Cache keys | Dataset id, directories, date column and (absolute path, `mtime_ns`, size) of every input file are in the key. Two-file loaders (marketing data, channel attribution, forecast, dynamic analytics) fingerprint both files. |
| Dynamic analytics | One context for metadata and data; no call back into a public wrapper. |
| Public API | Names, zero-argument signatures, return types and `.clear()` unchanged. Global `st.cache_data.clear()` calls unchanged. |
| Other `st.cache_data` users | Only `stats_ui._read_analysis_table`, which is already keyed by dataset id, path and mtime. |
| Hidden state | None added. `sys.path.insert` at import is from the baseline. |

`data_loader.py` was therefore NOT changed (SHA-256 `ba4df412c49e9a59bec9722eb2a74e9bf89058d3a0bcd627a7de066bb0666747`).

### 13.2 Test gap found and fixed

Mutation runs (a fingerprint in one loader replaced by a constant, in a scratch copy) showed that the existing 111 tests did NOT detect a broken fingerprint in `load_kpis`, `load_forecast` (models copy), `load_channel_attribution`, `load_model_metrics` or the dynamic-analytics metadata file: all five mutants passed (111 passed each). The file-change and cache-hit tests only covered `load_marketing_data` and `load_prediction_report`.

Added to `tests/test_legacy_cache_isolation.py` (22 tests, class `TestEveryInputFileInvalidatesItsLoader`; nothing existing was changed or weakened):

- One parametrized test for each of the 21 (loader, input file) pairs: an unchanged file is a cache hit (independent count of real file reads), a changed file is re-read without any global clear and equals the value read directly from the file, a missing file gives the empty value, a restored file gives the original value.
- One dynamic-analytics test: unchanged inputs are a cache hit; changing or restoring the metadata file, adding or removing the second table file is followed without a clear and agrees with an independent recompute.

### 13.3 Mutation results (scratch copies, never the project)

Twelve single-fault mutants, each replacing one fingerprint by a constant, were run against the new tests. Every one is killed by the targeted test (M9, the clean-table fingerprint shared by two loaders, by two tests). The unmodified code passes all 22.

### 13.4 Results

| Run | Result |
|---|---|
| New tests only (`-k TestEveryInput`) | 22 passed |
| Complete new test file | 133 passed |
| Complete suite | **1,942 passed**, 0 failed, 0 skipped, 0 errors, 16 pre-existing warnings (1,809 baseline + 133) |

Testing method: Streamlit `AppTest` sessions inside one process sharing the process-wide cache. This is not a live server with two real browsers.

### 13.5 Test file

`tests/test_legacy_cache_isolation.py`: 772 lines, 133 tests, SHA-256 `01d1ac85bbe606022ac77c53cee966e887d0837e5c2515c6141921f09ddc9b87`.

### 13.6 Known limitations

- A file edited in place with identical size and identical `mtime_ns` is not detected (authorized design D1); coarse-mtime file systems make this slightly more likely.
- Global `st.cache_data.clear()` calls remain (decision D2), so one session's dataset switch still empties everyone's cache (a performance effect, not a correctness one).
- Cache entries for old file versions stay until the next global clear (no `max_entries`).
- Within one call, `load_analysis_context` calls two public loaders, each of which resolves the session context; they cannot differ inside one script run.
- The dynamic-analytics ids are random by existing design.
