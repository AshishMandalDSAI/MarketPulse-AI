# MarketPulse AI — UI/UX Audit Report

**Release:** MarketPulse_AI_ULTIMATE_FINAL_RELEASE · **Date:** 2026-11-22
Evidence labels: **[BROWSER]** real Chrome (Playwright, headless) against `streamlit run app.py` ·
**[TEST]** automated test · **[STATIC]** code reading · **[NOT VERIFIED]**.

## 1. Method

`tools/browser_check.py` (kept outside the release, used for this audit) opens the real
app in Google Chrome, chooses the demo dataset **or uploads a CSV through the real file
input and clicks ✅ Confirm & Analyze**, then clicks every sidebar entry. For each page it
records load time, Streamlit exceptions (`stException`), error banners
(`stAlertContentError`), horizontal overflow, browser console errors, and a screenshot.
Screenshots were reviewed by eye; every visible problem below was fixed and re-checked.

## 2. Final browser results (code as released)

| Pass | Width | Pages | Exceptions | Error banners | Horizontal overflow | Console errors | Page switch |
|---|---|---|---|---|---|---|---|
| Demo dataset | 1440 px | 18 / 18 | 0 | 0 | 0 | 0 | 1.5–2.6 s |
| Uploaded CSV (2,000 rows, missing values, placeholders) | 1440 px | 18 / 18 | 0 | 0 | 0 | 0 | 1.5–1.8 s |
| Demo, tablet | 820 px | Upload, Executive Overview, Budget Optimizer, Forecast | 0 | 0 | 0 | 0 | ≤ 4.0 s |
| Demo, phone | 390 px | same 4 pages | 0 | 0 | 0 | 0 | ≤ 4.0 s |

Sidebar state measured in Chrome: expanded at 1440 / 1024 / 820 px, collapsed at 390 px.
Dashboard sign-in screen was also verified in Chrome against the Docker container (see
`FINAL_DEPLOYMENT_VERIFICATION.md`). Representative screenshots: `docs/screenshots/`.
Browsers other than Chrome: **[NOT VERIFIED]**.

## 3. Design system

| Element | Implementation |
|---|---|
| Theme | `.streamlit/config.toml` — dark base (#0B1220 / #131C2E, text #E6EAF2, primary #3B82F6). Root cause of the baseline's unreadable UI was a light base theme combined with CSS that forced light text: invisible text, white Plotly charts. |
| Tokens & CSS | `dashboard/styles/theme.py` (`TOKENS`, `THEME_CSS`, `CHART_COLORWAY`); legacy classes kept so no page broke |
| Components | `dashboard/components/ui.py`: `callout`, `empty_state`, `section`, `page_header`, `fmt_currency/number/pct` ("—" for undefined), `show_table` (readable headers, number and date formats) ; `kpi_card.py` (value, optional delta only when a baseline exists, help text) |
| Charts | `apply_chart_style` on all 40 Plotly call sites — transparent background, consistent fonts/grid/hover, titles never clipped; ₹ only where the unit is known (demo) |
| Navigation | grouped sidebar (Data · Overview · Marketing models · AI & insights · Advanced analytics · Statistics), active-page highlight, Active-dataset card (Demo / Uploaded / None) with Demo-data and Reset buttons, Sign out when auth is on |
| Responsive | ≤ 1100 px tighter padding; ≤ 640 px KPI rows 2-up, rows containing charts / tables / inputs stack full-width; sidebar collapsed on phones |

## 4. Problems found in the browser and fixed

| # | Problem (seen in screenshots or by the harness) | Pre-existing in baseline? | Fix | Evidence |
|---|---|---|---|---|
| U1 | Unreadable text and white charts | yes | dark theme config; removed global text override | [BROWSER] before/after |
| U2 | Literal `**bold**` markers inside HTML cards | yes | `md_inline` (escapes, then renders bold/italic/code) | [TEST] |
| U3 | First sidebar click after a forced page change landed on page 00 | yes | keyed `st.radio` | [TEST] [BROWSER] |
| U4 | "Go to Executive Overview →" after an upload did nothing (button nested in another button's branch) | yes | `on_click` callback + AST guard against nested buttons in all pages | [TEST] |
| U5 | Forecast page: "not available" for every upload whose date column isn't named `Date` | yes | loader parses the column that exists | [TEST] [BROWSER] |
| U6 | Forecast page (uploads): forecast without the history line | yes | uses the evaluated target column | [TEST, mutation-checked] [BROWSER] |
| U7 | Overview time-series of row-level data drawn as a zig-zag through every row | yes | one point per date (sum; mean when hinted) | [TEST] [BROWSER] |
| U8 | Date Range KPI showed `2021-01-01T00:00:00 → …` and wrapped | yes | dates shown as `YYYY-MM-DD` | [TEST] [BROWSER] |
| U9 | Overview quality score (99.9) disagreed with Upload page (99.7) | introduced by mixing engines | Overview shows the data-quality engine's original → cleaned scores | [BROWSER] |
| U10 | Tables showed `2021-01-31 00:00:00` | yes | date column format | [TEST] |
| U11 | Donut labels overlapped the legend | yes | inside labels, small slices hover-only | [BROWSER] |
| U12 | Z-test "not run" shown as a red error on first visit | yes | warning (expected behaviour, not an error) | [BROWSER] existing tests |
| U13 | Phone: sidebar open at first load, covering the page (demo button unclickable) | yes | `initial_sidebar_state="auto"` | [BROWSER] [TEST] |
| U14 | Phone: two charts side by side, legend over plot, truncated titles | introduced by this release's 2-up rule | rows with charts stack full-width | [BROWSER] [TEST] |

## 5. Page-by-page notes (desktop, demo unless stated)

| Page | Verified content |
|---|---|
| 00 Upload Data | four steps; quality KPIs; per-column missing/invalid table; role confirmation; editable cleaning plan; preview; before/after; downloads; analysis completes and activates the dataset |
| 01 Executive Overview | 4 KPI cards with half-over-half deltas (only where computable); predicted weekly revenue (current vs recommended mix, labelled estimates); trends; contribution donut; budget distribution; current vs optimized; top channels; forecast overview (walk-forward accuracy vs naive); insights. Upload: quality panel, dynamic KPIs, per-date charts, insights |
| 02 Marketing Performance | KPIs and breakdowns |
| 03 Marketing Mix Model | holdout R²/RMSE/MAE/MAPE, model comparison incl. baseline, actual vs predicted, residuals, adstock & saturation curves |
| 04 Channel ROI · 06 What-If · 09 Recommendations · 10 Reports | render without errors; captions state association/estimate |
| 05 Budget Optimizer | live re-optimisation; like-for-like uplift; recommendation table; bounds caption |
| 07 Forecast | forecast + 90% interval; "How accurate is this forecast?" (model, OOS MAE, WAPE, vs baseline); candidate table |
| 08 AI Marketing Analyst | "Rule-based analyst · offline" banner; suggested questions; answers carry evidence type & sources |
| 11–17 | Predictions, Anomalies & Drivers, Decision Intelligence, Descriptive / Inferential / Predictive / Prescriptive Statistics render on demo and upload |

## 6. Not verified / limitations

- Firefox, Safari, real mobile devices; accessibility audit (screen reader, contrast
  tool) — **[NOT VERIFIED]** (colours chosen for dark-theme contrast but not measured).
- Phone/tablet passes covered 4 pages; the other 14 were checked at desktop width only.
- Wide tables scroll horizontally inside their own frame on phones (by design).
