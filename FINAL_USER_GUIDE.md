# MarketPulse AI: User Guide

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE

MarketPulse AI turns a CSV or Excel file into marketing analytics: KPIs, trends, a marketing mix model (MMM), channel ROI, budget optimisation, forecasts, anomaly detection and statistics. It shows only the analyses your data supports. Anything it cannot support is shown as unavailable, with the reason.

## 1. Start the app

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                 # opens http://localhost:8501
```

If the administrator set a dashboard password, sign in first. Five wrong attempts lock sign-in for 60 seconds. **Sign out** is at the bottom of the sidebar.

## 2. Choose data

On first load there is no active dataset, and the app opens on **00 · Upload Data**. Either:

- upload your own file there, or
- choose **▶️ Or just try the built-in demo dataset**. NovaMart is synthetic: 209 weeks and 8 channels.

Other pages show **📤 Go to Upload Data** and **▶️ Try Demo Dataset instead** until a dataset is active.

The sidebar's **Active dataset** card always shows what you are looking at (Demo, Uploaded, or None). **Reset** clears it. Demo data is never mixed with your upload.

## 3. Upload your own file (page 00)

**Format.** Use `.csv` or `.xlsx` with one header row and one record per row. Column names can be anything. Legacy `.xls` files must be saved as `.xlsx` first.

1. **What we found.** You see the rows, columns, the data-quality score (0–100) and the missing-cell percentage. A table lists missing and invalid values per column. Open "How the quality score is calculated" for the formula: completeness 30%, validity 25%, uniqueness 15%, consistency 15%, type correctness 15%.
2. **Column roles.** HIGH-confidence roles (for example `Order_Date` → date) are pre-selected. MEDIUM and LOW guesses must be confirmed in a dropdown; nothing uncertain is applied silently. Tick the marketing spend columns.
3. **Data cleaning.** Every column with problems gets a recommended treatment and the reason for it. Change any treatment in the table. Defaults:
   - dates are **never** filled;
   - revenue is left missing (those rows are excluded where a value is needed);
   - spend gaps become 0;
   - other gaps use the median of earlier rows only.

   Exact duplicate rows are kept unless you tick "remove". The preview shows the score after cleaning, the number of cells imputed and the rows removed, plus a before/after table. **Download cleaned data (CSV)** if you need it.
4. **Analyze.** **✅ Confirm & Analyze** saves your original file unchanged, applies the plan you approved, and runs every supported analysis. This takes about 30 seconds for a few thousand rows. You can then download the cleaning audit (JSON).

**Dates.** If your dates are written as `dd/mm/yyyy`, MarketPulse detects that from values above 12. If every value is ambiguous (for example `03/01/2022`), it reads them month-first and says so in the processing log. Check the date range on the Executive Overview.

## 4. Pages

| Group | Page | What you get |
|---|---|---|
| Overview | 01 Executive Overview | KPIs, trends, data quality, recommended charts, insights; for the demo, also forecast and budget summaries |
| | 02 Marketing Performance | KPIs and breakdowns by channel and dimension |
| Marketing models | 03 Marketing Mix Model | Model comparison on held-out weeks, baseline check, channel contributions |
| | 04 Channel ROI | Revenue per unit of spend by channel (a statistical association) |
| | 05 Budget Optimizer | Recommended reallocation within bounds; constraint check; predicted change at the same budget |
| | 06 What-If Simulator | Scenario changes using the demo model |
| | 07 Forecast | Forecast with a 90% interval; "How accurate is this forecast?" (walk-forward MAE, WAPE, comparison with a naive baseline) |
| AI & insights | 08 AI Marketing Analyst | **Rule-based** answers to common questions, each labelled with its evidence type (observed fact, model estimate, model scenario, recommendation, guidance) and the files it was built from. It is not a chatbot or LLM. |
| | 09 Recommendations | Management recommendations derived from the results |
| | 10 Reports | Downloadable exports |
| Advanced analytics | 11 Predictions · 12 Anomalies & Drivers · 13 Decision Intelligence | Automated predictive tasks, anomaly and driver analysis, decision scenarios |
| Statistics | 14 Descriptive · 15 Inferential · 16 Predictive · 17 Prescriptive | Statistical modules with assumption checks; tests whose assumptions fail are refused with an explanation |

## 5. Reading the results correctly

- **Association is not causation.** MMM and ROI figures describe relationships in your data, not proof that spend caused revenue.
- **Estimates are not guarantees.** Forecasts and optimised budgets are model estimates. The interval shows typical error, not a bound.
- **Significance is not importance.** On the Inferential page, read the effect size and its plain-language sentence, not only the p-value.
- **"—" means a metric is undefined.** For example, MAPE when every actual value is 0. It never means zero error.
- **Model quality.** Check the "baseline" comparisons. If a model does not beat a naive baseline, the app says so and uses the baseline.

## 6. API (optional)

```bash
uvicorn api.main:app --reload          # http://127.0.0.1:8000/docs
```

In production (`MARKETPULSE_ENV=production`):

- protected endpoints require the `X-API-Key` header;
- clients get HTTP 429 with `Retry-After` if they exceed the rate limits.

See README.md for the endpoint list.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| An error when reading the file (empty file, no header row, inconsistent rows, unsupported encoding, invalid workbook) | The message names the cause. Re-save the file as UTF-8 CSV or `.xlsx` with one header row. |
| A capability is "not available" | The reason is shown. Typically there is no date column, no revenue column, too few time periods, or no spend columns. Confirm roles on the Upload page. |
| Date range looks wrong | Ambiguous `dd/mm` vs `mm/dd` dates. See the processing log, and re-save the dates as `YYYY-MM-DD`. |
| "Dashboard sign-in is not configured…" | Set `MARKETPULSE_DASHBOARD_PASSWORD_HASH` (generate it with `python -m dashboard.components.auth`). |
| Large file is slow | About 3–4 minutes for 100k–1M rows on 2 vCPUs. Some stages sample large data and say so. |
