# PART 06: the user's REAL Kaggle file (validated)

> **Status: tested, passed.** The actual file `synthetic_mmm_weekly_india_SAT.csv` was supplied in PART 06 (SHA-256
> `5991fd4ba51818ced61d87f9eb9f802b6aff905ce6544b4658e6aad9e41c0d7b`, 4.6 MB) and is included in the release at
> `tests/fixtures/part02/real/`. Earlier sections of this report, written before the file was available, described
> only the structure replica. Where they say "not tested", this section supersedes them.
>
> Evidence: `tools/part06/real_dataset_validation.py` (raw JSON results), `tests/test_upgrade_part06.py`,
> `tests/test_upgrade_part02.py::TestRealKaggleFile` (now run, no longer skipped), an AppTest audit of all pages and
> two Chrome passes (CSV and XLSX uploads).

## A. Defect found by the real file, and fixed

The real file has two columns the replica did not: `FB_Banner_Content_Score` and `IG_Banner_Content_Score`
(creative content scores, 0–100).

- **Before the fix:** the role engine read them as *advertising spend for Facebook / Instagram*
  (confidence 0.45, from the channel word alone). That put the whole file in **Spend-Based** mode, which would have
  produced ROI and ROAS on non-monetary scores, and the PART 02 real-file tests failed (3 of 3).
- **Fix:** a new business role, `quality_score` ("Creative / content / quality score (not spend, not exposure)"),
  for names with *score, rating, quality, sentiment, NPS, CSAT, stars*. Such a column is never channel spend, and it
  enters the models as a business driver.
- **Not changed:** existing roles such as `TV_Spend`, `Facebook_Impressions`, `Trade_Spend` and
  `Facebook_Engagements` (tested).

## B. Results on the real file

| Check | Result |
|---|---|
| A. Automatic detection | 11,232 rows × 28 columns. 21 float, 3 integer and 4 text columns. Weekly, 2022-07-02 to 2025-06-21: 156 weeks, all 7 days apart, 72 rows per week (8 Geo × 9 SKU; 3 brands). |
| B/C. Column mapping (22 columns checked; all correct) | Week → date (0.90); Sales_Units → sales units (0.72); Sales_Value → revenue (0.97); Trade_Spend → trade/promotion spend (0.87, *business variable*); TV/YouTube/Facebook/Instagram_Impressions → impressions; Print_Readership → readership; Radio_Listenership → listenership; Brand → brand; SKU → SKU; Geo → geography; MRP, Net_Price → price; Feature/Display/TPR_Flag → promotion; Weighted/Numeric_Distribution, TDP, NOS → distribution; CPI, GDP_Growth, Festival_Index, Rainfall_Index → external; FB/IG_Banner_Content_Score → **quality score**. No column is read as channel spend. |
| D. Target selection | Candidates: Sales_Value and Sales_Units. `needs_target_confirmation = true`; **no target pre-selected**. In Chrome, *Confirm & Analyze* stayed disabled until a target was chosen. Both targets were run, separately and together; the per-page target selector switches them (tested on pages 04–07). |
| E. Mode | **Exposure-Based Marketing Analytics** (6 exposure channels; no channel-wise spend). |
| F. No fabricated money | Trade_Spend is a business driver, not six channel budgets. Pages 04–07 show *"No channel-wise cost exists, so ROI, ROAS and monetary contribution are not calculated"*, and every ROI, ROAS, monetary-contribution and budget capability reports *unavailable: needs channel-wise advertising spend*. The channel table has no money columns. The legacy Phase 6 module reports "ROI unavailable: no marketing channels detected". No ₹ unit appears on any exposure chart. |
| G. Models (both targets) | Driver response model: holdout R² on weekly totals **0.91 (Sales_Value) / 0.94 (Sales_Units)**, graded *good*, n = 11,232. All 6 exposure channels and 16 business drivers (including Trade_Spend and the two content scores) are in the model. Exposure optimiser: Sales_Value reallocates impressions between TV and Instagram (*optimal*, +5.3% estimated, all bounds respected). Readers and listeners are *Not supported* (one channel per unit), with the reason shown. For Sales_Units, fewer than two impression channels have a positive, significant effect, so it is *Not supported* with the reason. Scenarios: 4 presets, reliability *good*. Sensitivity: 22 drivers. Legacy pipeline (*Confirm & Analyze*): 23 s, 13 output files, forecast written, no spend channels. Predictive: regression on Sales_Value ran (selected Random Forest by CV); classification on TPR_Flag with **all 12 classifiers** gave 12 + baseline holdout rows, all *ok*. Inferential: t-test ran; the Z-test refused without σ and ran with σ. |
| H. Data quality | Missing cells **0**, duplicate rows **0**, duplicate Week×Geo×Brand×SKU keys **0**, date parse 100%, no negative values, no constant columns, quality score 100/100. Extreme values beyond 3×IQR (informational, kept): Sales_Units 116, Sales_Value 189, Trade_Spend 153, TV 322, YouTube 194, Facebook 168, Instagram 43, Print 304, Radio 311. The 0/1 promotion flags show as "outliers" only because their IQR is 0; that is expected for binary columns. |
| I. Charts, reports, filters, downloads, interactions | AppTest: 2 configurations × 18 renders = **36 renders, 86 charts, 0 label/currency problems, 0 exceptions**. Chrome: CSV upload and XLSX upload of the real data, all 16 pages, **0 exceptions, 0 error boxes, 0 console errors**, and the optimiser click returned *Solved*. Downloads clicked in Chrome and parsed: effect estimates (22×15), channel table (6×17), scenario comparison (6×8), revenue forecast (12×4), generic KPIs (valid JSON), executive summary (Markdown). Filters: the target selector on 07 switched every chart to Sales_Units in Chrome, and on 04–07 in AppTest. |
| Currency | The file has no currency symbol or code, so detection returns *unknown* and the user chooses (INR). Both INR and "none" were tested. |

---

# MarketPulse AI Upgrade: Dataset Compatibility Report

| Part | Content |
|---|---|
| PART 01 (this edition) | Currency detection and display; chart labelling across dataset shapes; navigation and Statistics on demo and uploaded data |
| PART 02 | The universal semantic engine and the Spend, Exposure and General modes, tested on fixtures 1–7 of the master specification |
| PART 03 | Categorical, continuous and imbalanced targets (fixtures 8–10) |

Evidence labels: **[TEST]** automated test in the PART 01 suite; **[BROWSER]** verified in Chrome; **[NOT VERIFIED]** not checked.

## 1. Currency detection

### Rules

- A currency is reported only when the evidence is unambiguous.
- **Name evidence:** a currency code or word in a column name, such as `Revenue (INR)`, `amount_usd` or `Turnover GBP`. Codes that are also ordinary words (`try`, `cad`, `pln`, …) only count inside brackets.
- **Value evidence:** symbols or codes in the raw text of cells, such as `₹1,200`, `$12.50` or `EUR 300`. Detection runs on the raw upload, before cleaning strips the symbols.
- **Shared symbols** (`$`, `¥`, `Rs`, `kr`) identify the symbol but not the country: the symbol is shown and the ISO code is left empty. `$` is never assumed to mean USD.
- **Conflicts** (different currencies in different columns, or in one column) result in no currency; the conflict is listed.
- **No evidence** results in no currency and no symbol. INR is never assumed for an upload.
- **The user can override the result** on the Upload Data page: keep the automatic result, choose any of the 37 supported currencies, or choose "None / not monetary".

### Verified cases [TEST]

| Input | Result |
|---|---|
| `Revenue (INR)` | INR ₹ (column names) |
| `amount_usd` | USD $ |
| values `€1.200`, `€30` | EUR € (cell values) |
| `Turnover GBP` | GBP £ |
| `Revenue (JPY)` | JPY ¥ |
| `Revenue USD` with values `$1,200` | USD $ (names and values) |
| values `₹1,200`, `Rs. 300` | INR ₹ |
| values `EUR 300` | EUR € |
| `Revenue (CAD)` | CAD C$ |
| values `$1,200` only | symbol `$`, code not identified |
| values `¥300` only | symbol `¥`, code not identified |
| `Cost (Rs)` | symbol `Rs`, code not identified |
| `rev_usd` and `cost_eur` | **no currency** (conflict) |
| values `€5` and `£6` in one column | **no currency** (conflict) |
| `Price`, `Qty` | no currency |
| `try_count`, `cad_hours` | no currency (ordinary words) |
| `users`, `hrs` | no currency |
| `Revenue` with plain numbers | **no currency** (INR is not assumed) |
| XLSX header `Revenue (EUR)` | EUR |

### Display rules [TEST]

| Situation | What is shown |
|---|---|
| Demo dataset | INR ₹. Output is identical to the previous release for every tested value (₹12, ₹999, ₹2.50L, ₹1.50Cr, negatives, "—" for NaN). |
| Upload with no currency | No symbol anywhere; plain numbers (`1.23M`); axis titles without a unit |
| Upload with USD | `$1.50K`, `$12`; axis titles `… (USD)` only on monetary columns. "Units" stays without a unit. |
| Switching between datasets | Each dataset shows its own currency (EUR → GBP → demo ₹) |
| Executive Overview KPI cards | Currency only on sums and averages of monetary columns; never on counts, rates or percentages |
| AI analyst on an upload | Uses the dataset's symbol, or "unit of currency" when none; never "₹" or "per rupee" |

### Browser check [BROWSER]

A 2,000-row CSV with `Net_Sales_USD` and `Meta_Ads_USD` columns was uploaded in Chrome:

- step "2b · Currency" showed "Detected USD ($) from column names", with the evidence and the override list;
- after the default ("Auto-detected: USD") was confirmed, the Executive Overview showed `$2.01M`, `$1.0K` and `$407.4K` on the monetary KPIs;
- the axes read "Total Net Sales (USD)" and "Average Meta Ads (USD)";
- the non-monetary KPIs and axes had no currency.

## 2. Chart labelling across dataset shapes [TEST]

Every chart on the Executive Overview has a title and named x and y axes (pie and donut charts have legends instead of axes). This was checked on:

- **the demo:** 4+ charts, including "Revenue (INR)" and "Total Marketing Spend (INR)";
- **a synthetic order-level upload** (400 rows over 60 dates; one point per date on trend charts, labelled "Total …");
- **the fixtures** `a_marketing.csv`, `b_sales.csv`, `e_education.csv` and `g_no_date.csv`;
- **all 9 chart types** of the chart recommender, checked directly. A bar chart with no stated aggregation is labelled "Average …", because the renderer averages in that case.

Marketing Performance was checked on the demo (5+ figures) and on an upload, including one point per date on trends.

## 3. Navigation and Statistics on different datasets

| Check | Evidence |
|---|---|
| Every menu entry renders on the demo | [TEST] [BROWSER] |
| Every menu entry renders on an uploaded dataset | [TEST] [BROWSER] |
| Statistics: all three tabs render on demo and on upload, and only the open tab runs | [TEST] [BROWSER] |

## 4. Known limitations (PART 01)

- The currency vocabulary covers 37 ISO codes and their common symbols. Rarer currencies can still be chosen manually only if they are in the list. **[NOT VERIFIED]** for others.
- **The Forecast page is protected and was not changed in PART 01.** For uploads, its table and axes still show no currency, as before. It is included in the app-wide chart pass in PART 04.
- Spend columns whose name or values carry no currency evidence count as monetary when they are confirmed as channels or revenue. The currency attached to them is the dataset-level choice.

---

# PART 02: universal dataset engine and three analytics modes

**The real Kaggle file was not available.** `synthetic_mmm_weekly_india_SAT.csv` was not provided, so no result below comes from it. `kaggle_structure_replica.csv` is a seeded synthetic file with its exact 26-column structure. See `KAGGLE_DATASET_TEST_REPORT.md`.

**Files:** every fixture is in `tests/fixtures/part02/` (seeded generator `make_datasets.py`).

**Generated details:** per-column roles, confidence, evidence, alternatives, warnings, capabilities and model results are in:

- `docs/part02/compatibility_details.md` (demo and fixtures);
- `docs/part02/kaggle_structure_replica_report.md`.

Both are generated by `tools/part02_compat_report.py`, which can be run on any CSV/XLSX.

| Dataset | Rows×cols | Mode | Ad spend / exposure / trade-spend columns | Target candidates (never auto-chosen when >1) | Driver-model quality (holdout) |
|---|---|---|---|---|---|
| `marketing_data_clean.csv` (demo) | 208×45 | Spend-Based | 8 / 1 / 1 | Revenue, Orders, New_Customers, Returning_Customers, Total_Customers | Revenue: weak (the pages say so; MMM shown alongside) |
| `kaggle_structure_replica.csv` | 3,120×26 | Exposure-Based | 0 / 6 / 1 | Sales_Value, Sales_Units | Sales_Value: good; Sales_Units: good |
| `spend_weekly.csv` | 120×6 | Spend-Based | 4 / 0 / 0 | Revenue | Revenue: good |
| `exposure_daily.csv` (daily, dd/mm/yyyy) | 240×6 | Exposure-Based | 0 / 4 / 0 | Conversions | Conversions: good |
| `general_sales.csv` (order-level) | 1,500×9 | General | 0 / 0 / 0 | Net_Sales, Quantity | Net_Sales: good; Quantity: good |
| `renamed_columns.csv` | 100×5 | Spend-Based | 3 / 0 / 0 | Sales_Turnover | Sales_Turnover: good |
| `multi_revenue.csv` | 90×6 | Spend-Based | 2 / 0 / 0 | Gross_Revenue, Net_Revenue, Total_Sales | all good |
| `missing_marketing.csv` (monthly) | 144×5 | General | 0 / 0 / 0 | Revenue, Customers | n/a: no driver columns (explained on the pages) |
| `one_channel.csv` | 80×3 | Spend-Based | 1 / 0 / 0 | Revenue | Revenue: good |
| `two_channel.csv` | 80×4 | Spend-Based | 2 / 0 / 0 | Revenue | Revenue: good |
| `brand_geo_spend.csv` | 468×9 | Spend-Based | 2 / 0 / 0 | Revenue, Units_Sold | both good |
| `mixed_types.csv` | 180×7 | Exposure-Based | 0 / 1 / 0 | Leads | Leads: good |

The 10 required dataset categories map to the fixtures as follows:

1. **Spend-based:** `spend_weekly`.
2. **Exposure-based:** `kaggle_structure_replica`, `exposure_daily`.
3. **General sales:** `general_sales`.
4. **Different column names:** `renamed_columns` (Television_Cost, Online_Ads_Investment, Meta_Ad_Cost, Sales_Turnover); `exposure_daily` (FB_Reach, Meta_Views, YouTube_Impr).
5. **Multiple revenue columns:** `multi_revenue`.
6. **Missing marketing columns:** `missing_marketing`.
7. **One channel:** `one_channel`, plus `two_channel` for the optimizer.
8. **Brands and geographies:** `kaggle_structure_replica`, `brand_geo_spend`.
9. **Daily and weekly dates:** `exposure_daily` (daily), the weekly fixtures and `missing_marketing` (monthly). Five date formats are tested.
10. **Categorical and numerical:** `mixed_types`, `general_sales`.

**Other inputs:** Excel input is tested (`test_excel_input_gives_the_same_roles`), and so are missing values and duplicate rows (`TestRobustness`).

**Demo:** the engine marks `Total_Marketing_Spend`, `Digital_Spend` and `Traditional_Spend` as **aggregates**, because each equals a sum of channel columns. It marks the `*_Share`, `Naive_ROAS`, `AOV` and `*_Growth_WoW` columns as **derived**. Neither group is ever used as a channel or a driver. `Promotion_Spend` is trade/promotion spend, matching how the original MMM treated it (a control, not a channel).

# PART 05: final dataset audit

## Kaggle dataset
*(Superseded by the PART 06 section above: the real file was provided and tested in PART 06.)*
At the time of PART 05 the real file `synthetic_mmm_weekly_india_SAT.csv` had **not been provided**, so it was **not
tested** then. Every Kaggle-related claim rests on `tests/fixtures/part02/kaggle_structure_replica.csv`,
a replica with the same structure (`KAGGLE_DATASET_TEST_REPORT.md`). The three
`TestRealKaggleFile` tests are **skipped**, not passed. They run automatically when the file
is placed in `tests/fixtures/part02/real/`.

## Final page-by-page audit (PART 05)

| Dataset | Mode | Currency | Pages and tabs rendered (AppTest) | Charts checked | Exceptions / safe-page fallbacks | Chart label or currency problems |
|---|---|---|---|---|---|---|
| Demo NovaMart | Spend | INR | 18 (16 pages + 3 Statistics tabs) | 137 across all 4 datasets | 0 | 0 |
| `spend_weekly.csv` | Spend | USD (chosen) | 18 | (see above) | 0 | 0 |
| `kaggle_structure_replica.csv` (targets Sales_Value + Sales_Units) | Exposure | INR (chosen) | 18 | (see above) | 0 | 0 |
| `general_sales.csv` | General | none | 18 | (see above) | 0 | 0 |

The only error boxes were the Inferential tab's *"This result is unreliable…"* verdict, shown when
a test's assumptions fail. That is a statistical result, not a malfunction.

**Chrome pass** (real server, Playwright and Chrome, all 16 pages): demo, spend upload, exposure
upload with both targets, and general upload. Result: 0 exceptions, 0 error boxes, 0 browser
console errors. *Confirm & Analyze* stayed disabled until a target was chosen when the target
was ambiguous (exposure and general uploads). The Resource Optimizer solved in spend mode
("3 of 3 converged starts agree") and in exposure mode.

## Fixes from the PART 05 audit

| Finding | Fix | Test |
|---|---|---|
| Forecast chart had no title or x-axis title, and its y-axis had no unit | Title "‹target›: history and forecast", time-axis title from the date spacing, y-axis title with the currency unit, or none when unknown | `test_forecast_chart_title_and_axes_demo`, `test_forecast_chart_no_currency_when_unknown` |
| "Driver Importance" chart (Anomalies & Drivers) had no axis titles | "Importance score" / "Driver" | `test_driver_importance_chart_axes` |

`tests/test_upgrade_part05.py` repeats the whole audit on every test run, covering every page
and tab for the demo (INR) and for an unknown-currency upload.

## Known dataset limitations (final)

1. ~~Real Kaggle file not tested~~: tested in PART 06 (see the top of this report).
2. Long-format channel data (a channel-name column plus a value column) is not pivoted.
3. The bundled demo has no Phase 5 anomaly/driver output (baseline), so page 13 shows its
   "not yet available" message for the demo.
4. *Confirm & Analyze* takes about 4.5 minutes at 100,000 rows (`PERFORMANCE_REPORT.md`).
5. Legacy `.xls` files are not supported.
