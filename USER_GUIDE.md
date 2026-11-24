# MarketPulse AI user guide

This guide covers every page that the upgrade (PARTs 01–06) added or changed. Pages 09–15 (AI Marketing Analyst,
Recommendations, Reports, Predictions, Anomalies & Drivers, Decision Intelligence, Prescriptive Analytics) work as
described in `FINAL_USER_GUIDE.md`. Installation: `INSTALLATION.md` (Windows: `WINDOWS_SETUP_GUIDE.md`).

## Example: the Kaggle MMM file (`synthetic_mmm_weekly_india_SAT.csv`)

1. Upload the file. The quality score is 100/100 (no missing cells, no duplicate rows), and the mode is
   **Exposure-Based**.
2. Six exposure channels are recognised: TV, YouTube, Facebook and Instagram impressions, Print readership and
   Radio listenership.
3. `Trade_Spend` is a **business variable**, not advertising. `FB_Banner_Content_Score` and
   `IG_Banner_Content_Score` are **creative quality scores**, not spend.
4. Choose **Sales_Value**, **Sales_Units** or both as targets. Nothing is chosen for you.
5. Under currency, choose **INR**; the file itself has no currency symbol.
6. Pages 04–07 show effects per unit of exposure, and say why ROI, ROAS and monetary budget allocation are not
   available: the file has no channel-wise cost.

## 00 · Upload Data

1. Choose a `.csv` or `.xlsx` file. Legacy `.xls` files are not supported.
2. **What we found** shows the row and column counts, a quality score, missing cells and duplicates.
3. **Column roles and target**:
   - Each column has a suggested business role, a confidence level, the evidence behind it, and alternatives.
   - Names with *score / rating / quality / sentiment / NPS* (e.g. `FB_Banner_Content_Score`) are read as a
     **creative / quality score**, never as channel spend (PART 06).
   - To correct a role, edit its **Role** cell. Uncertain roles are never applied without your confirmation.
   - Choose one or more **analysis targets**. When several plausible targets exist (for example *Sales_Units* and *Sales_Value*), none is chosen for you. *Confirm & Analyze* stays disabled until you pick at least one.
   - The **Detected analytics mode** box says whether the data is Spend-Based, Exposure-Based or General, and why.
4. **Currency**: the app detects it from symbols or codes when it can. You can override it, or choose *none*. With an unknown currency, no symbol is shown anywhere.
5. Review the cleaning step, then click **Confirm & Analyze**.

## 01 · Statistics

- **Descriptive**: summaries, distributions and group comparisons with labelled axes.
- **Inferential**: ten tests.
  - The **one-sample t-test** is the default.
  - The **Z-test** needs the *known population σ*. The sample SD is shown for reference only and is never substituted. If σ is blank or ≤ 0, the test is refused, and a button switches you to the t-test.
  - Results show the formula with your numbers, the p-value, a conclusion, and a plain-language reading.
  - An "unreliable" box means the test's assumptions are not met. It is a statistical verdict, not an error.
- **Predictive**:
  - Choose up to **12 classifiers**: *Select all available models* or *Back to the defaults*. For imbalanced targets, you can tick *Handle class imbalance*; it is pre-ticked when the class ratio is 3:1 or more.
  - Models are selected by cross-validation only. Each model is then scored once on its own chronological holdout.
  - Ten comparison charts show the metrics, confusion matrices, ROC curves, per-class tables and feature importance.
  - Unavailable optional libraries are listed with the reason.

## 04 · Marketing Effectiveness Model

The page fits a driver response model to your data:

- channels are entered as `log1p`, with optional controls and fixed effects;
- standard errors are HC3, and the model is checked on a chronological holdout;
- it gets a quality grade (strong / moderate / weak), and a weak grade is stated, not hidden.

In spend mode, the adstock/saturation MMM is shown alongside.

## 05 · Channel Performance & Contribution

This page shows contributions, response curves and relationship evidence. ROI and ROAS
appear only when channel-wise cost exists. In exposure mode the page explains why no monetary
ROI or ROAS is shown.

## 06 · Marketing Resource Optimizer

- **Spend mode**:
  1. Choose the channels, the total budget, and the minimum and maximum for each channel. The defaults follow recent levels; *Reset all limits to the defaults* restores them.
  2. The **feasibility check** runs before solving. If the limits cannot hold the budget, you get the exact shortfall and up to four remedies (A–D). Each remedy is applied only when you click it.
  3. Tick **Allow an unallocated reserve** if spending less than the total is acceptable.
- **Exposure mode**: the page reallocates an exposure total across channels with the same unit. Mixed units are reported as *Not supported*. No reserve and no money are involved.
- **Solution status**: *Optimal*, *Feasible, optimality not proven*, *Infeasible*, *Failed*, or *Not supported*. The status comes with the total, the budget used, the unallocated amount and a **Constraint Utilisation** chart.

## 07 · Marketing Scenario Simulator

- **Channel scenarios** change one or more channels by a percentage.
- **Total-budget %** (spend mode) scales every channel and can be combined multiplicatively with channel scenarios.
- **Sensitivity** plots a tornado chart of the outcome change for ±step on each driver.
- Ranges reflect holdout error only, so they are approximate.

## Downloads

CSV downloads escape any cell that a spreadsheet could run as a formula: text beginning with `=`, `+`, `-` or `@`
gets a leading `'`. The *Download cleaned data* file is the exception: it re-exports your own values unchanged.

## 08 · Forecast and the other charts

Every chart now has a title and axis titles. Money axes show the active currency, or no unit
when the currency is unknown.
