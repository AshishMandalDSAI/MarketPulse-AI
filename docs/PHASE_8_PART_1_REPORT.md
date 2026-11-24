# Phase 8 – Part 1: Shared Foundation & Dashboard Integration

Status: **complete, tested, awaiting approval to start Part 2.**
Scope: foundation and navigation only. **No statistical, predictive or prescriptive result is calculated in
this part**; the four new pages show the module overview, real dataset information, availability checks and an
explicit status notice.

## 1. Baseline (before any feature work)

| | Result |
|---|---|
| Original ZIP, as received | 1,556 passed, **3 failed** |
| After fixes below | 1,559 passed, 0 failed (fresh extraction, no `data/` folder) |

Root causes of the three failures (none was a code defect):

| Failing test | Cause | Resolution |
|---|---|---|
| `test_ai_analyst::test_scenario_question_uses_simulator` | Reads demo pipeline outputs in `data/processed/`, which are git-ignored and so absent from a ZIP | New `tests/conftest.py` session fixture generates them once if missing |
| `test_regression_guards::test_channel_attribution_data_exists` | Same | Same |
| `test_docker_entrypoint::test_entrypoint_is_executable` | The ZIP did not preserve the execute bit of `docker-entrypoint.sh` | Execute bit restored and preserved when the backup ZIP is built (verified) |

The conftest backs up `models/` before regenerating demo data and restores it afterwards, so shipped model
artifacts are not changed (verified byte-for-byte). Note: `models/mmm_model.pkl` was saved with scikit-learn
1.8.0; newer versions load it with an `InconsistentVersionWarning`. No behavior change was observed.

## 2. What was added

New: `src/common/` (formatting, dataset_summary, export, session_scope, modules, availability),
`dashboard/components/stats_ui.py`, `dashboard/pages/14_…17_*.py`, `tests/conftest.py`,
`tests/test_phase8_part1_foundation.py`, this report.
Modified: `app.py` (**+4 lines**: the four new navigation entries), `README.md`, `docs/architecture.md`.
Modules 00–13, their pages, engines, `session_paths` and `data_loader` were **not modified**.

## 3. Findings

1. **Existing cache is not dataset-keyed (not changed, reported).** `load_marketing_data()` and the other
   `@st.cache_data` loaders take no arguments, so their cache is correct only if every dataset switch clears
   it. The app's own buttons do (`session_paths._clear_page_caches`), but Streamlit's cache is shared by all
   browser sessions of a server process, so two simultaneous users on different datasets could be served each
   other's data. A test caught this: a page showed 80 rows for a 60-row dataset. Phase 8 pages therefore use a
   loader keyed by dataset id and file modification time. **Recommendation:** make the legacy loaders
   dataset-keyed in a separate, approved change.
2. **`build_column_metadata` crashes on list/dict cell values** (unhashable). `summarize_dataset` works around
   it on a copy; the shared function was not modified.
3. **Excel export bug found and fixed in my own code during testing:** rejecting an oversize sheet inside the
   writer produced a misleading openpyxl error; validation now happens before the writer opens.

## 4. Test results (actually executed)

| Run | Result |
|---|---|
| New Part 1 tests | **154 passed** |
| Complete suite (existing + new), from a clean state | **1,713 passed, 0 failed** (1,559 existing + 154 new) |
| Coverage of new code (`pytest-cov`) | 92% overall (`src/common` 91–100% per file; `stats_ui.py` 83%) |

Part 1 tests cover: formatting; dataset typing on all 16 fixture datasets plus unusual names, numeric-only,
missing, duplicate, constant, empty and unhashable data; CSV/Excel export (round trips, formula neutralization,
sheet-name rules, row limit); dataset-scoped storage; registry and navigation integrity (00–13 unchanged and
in order); availability gate; real rendering of pages 00–17 on the demo dataset through Streamlit `AppTest`;
the four new pages on six diverse uploaded datasets; dataset switching, reset, stale-state and back-to-back
dataset isolation; download buttons.

`stats_ui.py` coverage is lower because some branches only run inside Streamlit's script thread.

## 5. Dependencies and limitations

No new dependencies. (`pytest-cov` was used to measure coverage and is a development tool only; it is not
required to run the app.)
Limitations at this stage: the four pages do not analyze anything yet; the availability gate is a coarse
first check that each module will tighten in its own part; legacy cache issue above remains until approved.
