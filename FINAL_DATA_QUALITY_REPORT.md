# MarketPulse AI — Data Quality & Missing-Value Report

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22
Evidence labels: **[EXECUTED]** run here · **[TEST]** automated test · **[STATIC]** code reading · **[NOT VERIFIED]**.

## 1. What was added

New module `src/data_engine/data_quality.py` (used by the Upload page, the analysis
runner and the Executive Overview) plus `src/data_engine/date_parsing.py`.

### Detection — `assess_quality(df)` [TEST]

| Problem | How it is detected |
|---|---|
| Missing | `NaN` / `None` / `NaT` |
| Blank & placeholders | empty or whitespace-only strings and tokens `na`, `n/a`, `nan`, `null`, `none`, `nil`, `-`, `--`, `?`, `#n/a`, `missing` (case-insensitive) — counted separately and treated as missing |
| Invalid numbers | text in a column whose present values are ≥ 75% numeric (e.g. `"abc"` in a spend column); thousands separators and currency symbols are understood (`"1,234.50"` → 1234.5) |
| Infinite values | `±inf` counted per column; converted to missing at the start of analysis and reported (they previously produced NumPy RuntimeWarnings and NaN statistics) |
| Invalid dates | values in a date column that cannot be read with the column's day/month order |
| Duplicates | exact duplicate rows; duplicate column names (renamed `Name`, `Name_1`, `Name_2` on upload, reported) |
| Type issues | numbers stored as text, mixed content in one column |
| Row profile | histogram of rows by number of missing cells; rows entirely empty |

### Explainable score [TEST]

Weighted average of five 0–100 dimensions, shown on the Upload page with its formula:

| Dimension | Weight | Definition |
|---|---|---|
| Completeness | 30% | non-missing **and non-imputed** cells |
| Validity | 25% | present cells that are valid (no infinities, unreadable numbers or invalid dates) |
| Uniqueness | 15% | rows that are not exact duplicates |
| Consistency | 15% | columns without mixed content or duplicate names |
| Type correctness | 15% | columns stored in a type matching their content |

The score is reported for the **original file** and **after cleaning**. Imputed cells
do not count as complete, so cleaning cannot inflate completeness — the after-score
only rises when real problems are resolved (e.g. numbers-as-text converted).

### Cleaning — `recommend_plan` + `apply_cleaning` [TEST]

Recommendations (every one can be changed by the user in an editable table before
anything is applied; each row shows *why*):

| Column situation | Default | Reason |
|---|---|---|
| Date column | leave | dates are never invented |
| Revenue / target | leave (rows excluded where a value is needed) | inventing revenue would bias every model |
| > 40% missing | leave | too sparse to fill credibly |
| Marketing spend column | fill with 0 | a blank usually means nothing was spent — stated as an assumption |
| Time-ordered numeric | median of **earlier rows only** | no future information |
| Other numeric | median (skewed) / mean | |
| Categorical ≤ 5% missing | most frequent value | |
| Categorical > 5% missing | `"Unknown"` | |
| Exact duplicate rows | **kept** unless the user ticks "remove" | they may be legitimate repeated transactions |

Also available: zero, forward-fill, interpolate, mode, drop rows, leave.
`apply_cleaning` works on a copy; the original upload is saved byte-for-byte in
`raw/` with its SHA-256. The audit records: timestamp, rows/columns before and after,
blank strings normalised, infinities converted, columns converted to numeric,
invalid numbers set to missing, cells imputed per column, rows removed, the plan
applied. It is written to `data_quality_report.json` and offered as a download.
The Upload page shows a preview (score, cells imputed, rows removed, conversions) and
a before/after table of changed rows before the user confirms.

## 2. Defects fixed in this area

| Defect | Fix | Evidence |
|---|---|---|
| Past-median fill was O(n²) | expanding median; result proven identical to the reference implementation | [TEST] equivalence test |
| Numeric-text spend column (e.g. `"1,234"`) misclassified as text | threshold 0.75 parse rate + separator handling | [TEST] |
| Day-first date columns: values parsed per-value (`03/01` → March, `17/01` → January in the same column) → wrong time order for the past-only median | one day/month order per column (`date_parsing.py`) | [TEST] `test_quality_engine_counts_no_invalid_dates_for_day_first` |
| Executive Overview showed a different quality score (KPI-engine estimate on cleaned data, 99.9) from the Upload page (99.7) | Overview now shows the engine's original → cleaned scores, missing %, cells imputed, rows removed | [EXECUTED] browser screenshot; [TEST] |

## 3. Measured examples [EXECUTED]

**Browser upload file** (2,000 rows × 8 columns, synthetic retail orders):
score **99.7 / 100**; 157 missing cells (0.98%), of which 11 were blank/placeholder
text; 0 invalid; 0 duplicates. Plan: Region → most frequent; Net_Sales → leave;
Meta_Ads → 0; Units / Discount_Pct → past-only median. Result: 126 cells imputed,
0 rows removed, after-score 99.7 (imputed cells not counted as complete).

**Messy file** (300 rows × 4: `N/A`, whitespace, `"1,234.50"`, `"not a date"`, blank
regions, 10% missing spend):

| | Score | Completeness | Validity | Type correctness |
|---|---|---|---|---|
| Original | **92.9** | 88.8 | 99.9 | 75.0 |
| After cleaning | **96.6** | 88.8 | 99.9 | 100.0 |

Revenue converted from text to numbers; spend gaps → 0 (assumption shown);
blank regions → `"Unknown"`; 132 cells imputed; 0 rows removed; the unreadable date
stays invalid and its row is excluded from time-based analysis (reported in the
processing log), never given an invented date.

## 4. Guarantees and their evidence

| Guarantee | Evidence |
|---|---|
| Original upload preserved unchanged | [STATIC] raw bytes written before cleaning; [EXECUTED] browser run produced `raw/upload_sample.csv` |
| Dates never invented | [TEST] date strategy is always `leave`; invalid dates → excluded, counted |
| Revenue never invented | [TEST] `test_missing_revenue_excluded_not_zero` |
| No silent deletion | [TEST] duplicates kept by default; `rows_removed` always reported |
| Demo data never mixed with uploads | [TEST] existing Phase 1–3 isolation tests (unchanged, passing) |

## 5. Not verified / limitations

- Placeholder vocabulary is English-centric (11 tokens listed in §1).
- Outliers are reported by the anomaly module, not "cleaned" — by design.
- Ambiguous date order (no day > 12 anywhere) is read month-first and **flagged**;
  the user must check the date range if their data is day-first.
