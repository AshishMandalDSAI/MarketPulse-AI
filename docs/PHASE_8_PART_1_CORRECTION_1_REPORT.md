# Phase 8 – Part 1 – Availability Correction 1

Status: **implemented and tested; awaiting independent review.**
Scope: `src/common/availability.py` only (plus tests and this report). Part 2 and the legacy cache fix are **not** started.
The original `docs/PHASE_8_PART_1_REPORT.md` is unchanged.

## 1. Problem

The descriptive availability gate rejected any dataset whose columns were all constant, including every
one-row dataset (each column of a single row has one unique value). That confused "has variation" with
"can be described". A constant column still has a valid count, mean, median, mode, minimum, maximum and
frequency table.

Behaviour on the original build (executed before the change):

| Dataset | Before | After |
|---|---|---|
| Constant numeric columns only | rejected | **available** |
| Constant categorical column only | rejected | **available** |
| Mixed constant + varying | available | available |
| One row | rejected | **available** |
| All values equal, some missing | rejected | **available** |
| Entirely empty columns only | rejected | rejected (new wording) |
| Zero rows / zero columns / `None` | rejected | rejected |

## 2. Approved decisions and what was implemented

| Item | Decision | Implementation |
|---|---|---|
| Descriptive gate | depends on non-empty columns, not variation | available when at least one column is not entirely empty (`_non_empty_columns`) |
| D1 prescriptive | two **varying** numeric columns, ≥ 10 rows | `_varying_numeric_columns`; a constant numeric column no longer counts |
| D2 variation helper | **deferred** | **not implemented**; no analysis-specific rules added |
| D3 unknown key | checked before dataset state | unknown/non-text key returns the "unknown module" result for every dataset state |
| Invalid summary type | clear error | `TypeError` naming `DatasetSummary` and the received type; `None` still means "unavailable" |
| Annotations | `list[str]` | applied to `Availability.reasons` / `.missing` (module already uses `from __future__ import annotations`; project targets Python 3.13) |
| Performance | build sets once | `_usable_columns` builds one excluded-set instead of two sets per column |
| Inferential wording | include boolean | "No numeric, categorical or yes/no column was detected." |
| Structure | preserved | `Availability` fields and the four module keys unchanged; new public constant `KNOWN_MODULES` lists them |

Unchanged rules: inferential needs ≥ 3 rows; predictive needs ≥ 10 rows and ≥ 2 usable (non-empty, non-constant) columns; prescriptive row minimum stays 10.

## 3. Requirements recorded for Part 2 (NOT implemented here)

When the Descriptive Statistics module is built it must distinguish:

- Standard deviation and variance of a constant column: **0** when the sample size permits.
- Sample standard deviation / variance with **one** observation: **undefined** ("needs at least 2 observations"), not 0.
- Skewness and kurtosis of a zero-variance column: **not defined**. (Observed while analysing this issue: pandas returns `0.0` for these on a constant column, which would wrongly suggest a symmetric distribution; scipy returns `nan`. Part 2 must not display pandas' `0.0`.)
- Correlation involving a constant column: **not defined** (`nan`).

The requirements for correlation, covariance, distribution comparison, skewness and kurtosis gating will be finalised with the Descriptive module itself.

## 4. Files changed (verified against the delivered Part 1 build with `diff -rq`)

| File | Change |
|---|---|
| `src/common/availability.py` | modified |
| `tests/test_phase8_part1_foundation.py` | one obsolete test replaced (`test_descriptive_needs_a_varying_column` → `test_descriptive_needs_a_non_empty_column_not_a_varying_one`) |
| `tests/test_phase8_part1_availability.py` | **new**, 96 tests |
| `docs/PHASE_8_PART_1_CORRECTION_1_REPORT.md` | **new** (this file) |

No other file differs: `app.py`, pages 0–17, `stats_ui.py`, the other `src/common/` files, analytics engines,
API, `requirements.txt`, `models/` and the original Part 1 report are byte-identical to the delivered Part 1 build.
No Streamlit import in `src/common/` (asserted by a test). No new dependencies.

## 5. Tests (all actually executed)

| Run | Result |
|---|---|
| New focused file `test_phase8_part1_availability.py` | 96 passed |
| All availability-related tests (new file + updated foundation tests) | 149 passed |
| All Part 1 tests (foundation 154 + availability 96) | 250 passed |
| **Complete regression suite**, clean state (no `data/`) | **1,809 passed, 0 failed, 0 skipped** (previous build: 1,713; +96) |

One of my new assertions initially failed because of a letter-case mismatch in the expected text
("at least 10" vs the message "At least 10 rows"); this was a test error, the code was correct, and the
assertion was fixed before the runs above.

What the new tests verify (with exact verdicts and message text, not just "returned a value"):
descriptive on constant numeric only, constant categorical only, mixed, one row, all-equal with missing
values, empty columns mixed with a non-empty one, entirely empty columns, zero rows, zero columns, `None`;
inferential at exactly 3 rows (and 2), boolean-only, no testable column type, two failures at once;
predictive at exactly 10 rows (and 9), fewer than 2 usable columns, constants not counted, two failures
in order, a 1,500-column dataset; prescriptive at exactly 10 rows (and 9), one numeric column, a constant
numeric column not satisfying the gate, only-constant columns, three numeric with two varying, two failures
in order; seven unknown-key values × three dataset states, and unknown key beating a wrongly typed summary;
`TypeError` for DataFrame / str / int / dict / list on every module, with the type name in the message;
structure preserved (fields, `list[str]`, module keys, independent default lists, no Streamlit import);
and all 16 fixture datasets (verdicts checked against independently restated rules).

## 6. Impact and risks

- The gate has one caller (`render_module_page` in `stats_ui.py`), which always passes a `DatasetSummary`
  from `summarize_dataset`, so the new `TypeError` cannot be reached by current code.
- Visible effect: all-constant and one-row datasets now show module 14's normal "delivered in Part 2" status
  notice instead of a data-availability warning. No result is calculated either way.
- Stricter prescriptive gate (D1): a dataset whose only second numeric column is constant is now blocked.
  This is intended.
- Limitations unchanged: the gate is a coarse first check; each module applies stricter checks when delivered.
  Columns classified as identifier or text still count as "usable" for the predictive gate.
