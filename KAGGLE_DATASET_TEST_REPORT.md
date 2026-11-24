> **PART 06 update:** the real `synthetic_mmm_weekly_india_SAT.csv` has now been provided and tested. It exposed one
> defect: the content-score columns were read as channel spend. That is now fixed. Full results are in
> `DATASET_COMPATIBILITY_REPORT.md` → "PART 06: the user's REAL Kaggle file". The text below is the original PART 02
> report about the structure replica.

# Real Kaggle dataset test report (PART 02)

## Status: the real file was NOT tested

`synthetic_mmm_weekly_india_SAT.csv` was not attached to any message in this conversation and isn't anywhere in the working environment. Following your instruction ("If the CSV is not accessible, do not pretend it was tested"):

- **No result in any PART 02 document comes from your file.**
- **The 3 tests that need it are skipped** (`TestRealKaggleFile` in `tests/test_upgrade_part02.py`). They appear as *skipped*, with the reason, in the test log.

## What was tested instead

`tests/fixtures/part02/kaggle_structure_replica.csv` is a seeded synthetic file. It has the **same 26 columns, in the same order,** as your spec's column list:

- **Keys:** Week, Geo, Brand, SKU
- **Outcomes:** Sales_Units, Sales_Value
- **Prices:** MRP, Net_Price
- **Promotion flags:** Feature_Flag, Display_Flag, TPR_Flag
- **Trade spend:** Trade_Spend
- **Exposure:** TV_Impressions, YouTube_Impressions, Facebook_Impressions, Instagram_Impressions, Print_Readership, Radio_Listenership
- **Distribution:** Weighted_Distribution, Numeric_Distribution, TDP, NOS
- **Macro and seasonal:** CPI, GDP_Growth, Festival_Index, Rainfall_Index

**Shape:** weekly data, 104 weeks × 5 geographies × 3 brands × 2 SKUs per brand = 3,120 rows.

**How the exposure columns are recorded:** once per week × geography × brand, then repeated on each SKU row, as is common in real panel files.

The full engine output is in `docs/part02/kaggle_structure_replica_report.md`.

### Pre-analysis inspection (replica)

| Item | Result |
|---|---|
| Rows × columns | 3,120 × 26 |
| Date | `Week`, 2022-01-03 to 2023-12-25, 104 periods, weekly |
| Missing values / duplicate rows | 0 / 0 |
| Brand / geography / SKU | 3 brands, 5 geographies, 6 SKUs (each SKU belongs to one brand) |
| Analytics mode | **Exposure-Based**: six exposure columns and no channel-wise advertising spend |

### Your five checks, on the replica

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Identified as exposure-based when there's no channel media spend | **Pass** | `TestRoleEngine::test_roles_and_mode_per_dataset[kaggle…]`; pages show "Exposure-Based Marketing Analytics" with the evidence |
| 2 | `Trade_Spend` not treated as channel advertising spend | **Pass** | It's mapped as *Trade / promotion spend (not advertising)*, a business variable. Both legacy detectors (`column_classifier`, Phase-6 `field_discovery`) and the API upload/confirm path no longer treat it as a channel. You can still reclassify it by hand. Tests: `TestTradeSpendRule` (6 tests), `TestLegacyIntegration::test_api_upload_confirm_kaggle_structure` |
| 3 | No channel ROI / ROAS fabricated | **Pass** | No ROI/ROAS columns, charts or values on pages 04–07. Every visible mention of ROI/ROAS is a "not available" statement (asserted sentence by sentence). The pipeline's `phase6_marketing.json` says "ROI unavailable", `capabilities.json` has `roi_roas: false`, and the AI analyst gives no channel ROI figures. Tests: `TestModelPages::test_kaggle_structure_exposure_mode[*]`, `TestLegacyIntegration` |
| 4 | Exposure analysis uses the six real exposure columns | **Pass** | All six are model drivers, each in its own unit: impressions ×4, readers, listeners. Association shares add up to 100%. Totals count each brand × geo value once, not once per SKU. Tests: `TestEffectiveness::test_kaggle_structure_both_targets_independently[*]`, `TestContextAndGrain` |
| 5 | All four models report supported and unsupported capabilities | **Pass** | Each page lists "Available for this dataset" and "Not available, and why", with what's needed. Tests: `TestCapabilities::test_exposure_mode_never_offers_roi_or_money` and the page tests |

**Target selection:** `Sales_Units` and `Sales_Value` are both offered and **neither is preselected**.

- **Upload page:** "Confirm & Analyze" stays disabled until you choose; this was verified in Chrome.
- **Analysing both:** each can be analysed independently. Every model page has a "Target analysed on this page" selector (test `test_kaggle_targets_selectable_independently`).
- **Leakage:** `Sales_Units` is never used to explain `Sales_Value`, or the other way round.

### Results on the replica (model estimates on synthetic data; not findings about your business)

| | Target Sales_Value | Target Sales_Units |
|---|---|---|
| Driver model quality (holdout R² on weekly totals, last 21 weeks held out) | good (0.81) | good (0.83) |
| Exposure optimizer (impressions, same unit only) | solved; +2.50% (approx. range ±168,991 per week) | solved; +3.33% |
| Exposure optimizer (readers, listeners) | not available: only one channel per unit | same |
| ROI / ROAS / monetary contribution | not computed: no channel cost | same |

## How to run the checks on the real file

1. Copy the file to `tests/fixtures/part02/real/synthetic_mmm_weekly_india_SAT.csv`.
2. Run `python -m pytest tests/test_upgrade_part02.py -k RealKaggleFile`. These are the 3 tests that are currently skipped.
3. Run `python tools/part02_compat_report.py tests/fixtures/part02/real/synthetic_mmm_weekly_india_SAT.csv --out kaggle_real_report.md`. This produces the full inspection and model report for your file.
4. Or upload it on **00 · Upload Data** and open pages 04–07.

If you attach the file, I'll run all of this and report the actual results.
