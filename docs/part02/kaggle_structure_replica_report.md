# Kaggle-structure replica: full engine report

**This is NOT the user's file.** `kaggle_structure_replica.csv` is a seeded synthetic file with the same 26 columns as `synthetic_mmm_weekly_india_SAT.csv`. Results describe the replica only.

## kaggle_structure_replica.csv

- Rows × columns: **3,120 × 26**
- Date column: `Week` — 2022-01-03 to 2023-12-25, 104 distinct periods, frequency **weekly**
- Missing values: 0 cells
- Exact duplicate rows: 0
- Brand: `Brand` — 3 levels (Aurel, Brivo, Cendra)
- Geography / region / store: `Geo` — 5 levels (Central, East, North, South, West)
- SKU / item code: `SKU` — 6 levels (AUR-200G, AUR-500G, BRV-1KG, BRV-250G, CEN-100G, CEN-400G)

**Analytics mode: Exposure-Based Marketing Analytics**

- 6 marketing exposure column(s): TV_Impressions, YouTube_Impressions, Facebook_Impressions, Instagram_Impressions, Print_Readership, Radio_Listenership
- no channel-wise advertising spend was found, so monetary ROI / ROAS and budget allocation are not available
- trade / promotion spend (Trade_Spend) is a business variable, not channel advertising spend
- outcome column(s): Sales_Value, Sales_Units

| Column | Type | Suggested role | Channel | Confidence | Evidence | Alternatives | Warnings |
|---|---|---|---|---|---|---|---|
| `Week` | str | Date / period |  | 0.90 | the name contains a date word (week); 100% of values parse as dates | Other category (0.50) |  |
| `Geo` | str | Geography / region / store |  | 0.89 | the name contains a geography word (geo); 5 distinct text values | Other category (0.50) |  |
| `Brand` | str | Brand |  | 0.91 | the name contains a brand word; 3 distinct text values | Other category (0.50) |  |
| `SKU` | str | SKU / item code |  | 0.91 | the name contains an SKU / item-code word; 6 distinct text values | Other category (0.50) |  |
| `Sales_Units` | int64 | Sales units / quantity |  | 0.95 | 'sales' together with a quantity word; values are whole numbers, consistent with counts; Sales_Value ≈ Sales_Units × Net_Price (median difference 0.0%), so it is a quantity | Other numeric measure (0.38) |  |
| `Sales_Value` | float64 | Revenue / sales value |  | 0.97 | the name contains a revenue word; values have decimals, consistent with money; Sales_Value ≈ Sales_Units × Net_Price (median difference 0.0%), so it is a sales value | Other numeric measure (0.38) |  |
| `MRP` | float64 | Price |  | 0.83 | the name contains a price word (mrp); values have decimals, consistent with money | Other numeric measure (0.38) |  |
| `Net_Price` | float64 | Price |  | 0.83 | the name contains a price word (price); values have decimals, consistent with money; Sales_Value ≈ Sales_Units × Net_Price (median difference 0.0%) | Other numeric measure (0.38) |  |
| `Feature_Flag` | int64 | Promotion (flag or intensity) |  | 0.83 | the name contains a promotion word (feature); values are 0/1 flags | Other numeric measure (0.40) |  |
| `Display_Flag` | int64 | Promotion (flag or intensity) |  | 0.83 | the name contains a promotion word (display); values are 0/1 flags | Advertising spend for one channel (0.40); Impressions (0.40) |  |
| `TPR_Flag` | int64 | Promotion (flag or intensity) |  | 0.83 | the name contains a promotion word (tpr); values are 0/1 flags | Other numeric measure (0.40) |  |
| `Trade_Spend` | float64 | Trade / promotion spend (not advertising) |  | 0.87 | a spend/cost word together with a trade / promotion word (trade) -- trade or promotion expenditure, not advertising for one channel; values have decimals, consistent with money | Other numeric measure (0.38) |  |
| `TV_Impressions` | int64 | Impressions | TV | 0.88 | the name contains the exposure word 'impressions' and the channel 'TV'; values are whole numbers, consistent with counts | Other numeric measure (0.38) |  |
| `YouTube_Impressions` | int64 | Impressions | YouTube | 0.88 | the name contains the exposure word 'impressions' and the channel 'YouTube'; values are whole numbers, consistent with counts | Other numeric measure (0.38) |  |
| `Facebook_Impressions` | int64 | Impressions | Facebook | 0.88 | the name contains the exposure word 'impressions' and the channel 'Facebook'; values are whole numbers, consistent with counts | Other numeric measure (0.38) |  |
| `Instagram_Impressions` | int64 | Impressions | Instagram | 0.88 | the name contains the exposure word 'impressions' and the channel 'Instagram'; values are whole numbers, consistent with counts | Other numeric measure (0.38) |  |
| `Print_Readership` | int64 | Readership | Print | 0.88 | the name contains the exposure word 'readership' and the channel 'Print'; values are whole numbers, consistent with counts | Other numeric measure (0.38) |  |
| `Radio_Listenership` | int64 | Listenership | Radio | 0.88 | the name contains the exposure word 'listenership' and the channel 'Radio'; values are whole numbers, consistent with counts | Other numeric measure (0.38) |  |
| `Weighted_Distribution` | float64 | Distribution / availability |  | 0.78 | the name contains a distribution word (distribution); values lie between 0 and 100 | Other numeric measure (0.38) |  |
| `Numeric_Distribution` | float64 | Distribution / availability |  | 0.78 | the name contains a distribution word (distribution); values lie between 0 and 100 | Other numeric measure (0.38) |  |
| `TDP` | int64 | Distribution / availability |  | 0.72 | the name contains a distribution word (tdp) | Other numeric measure (0.38) |  |
| `NOS` | int64 | Distribution / availability |  | 0.72 | the name contains a distribution word (nos) | Other numeric measure (0.38) |  |
| `CPI` | float64 | External / seasonal factor |  | 0.70 | the name contains an external / seasonal word (cpi) | Other numeric measure (0.38) |  |
| `GDP_Growth` | float64 | External / seasonal factor |  | 0.70 | the name contains an external / seasonal word (gdp) | Derived ratio / share / growth (kept out of models) (0.40); Other numeric measure (0.38) |  |
| `Festival_Index` | float64 | External / seasonal factor |  | 0.70 | the name contains an external / seasonal word (festival) | Other numeric measure (0.38) |  |
| `Rainfall_Index` | float64 | External / seasonal factor |  | 0.70 | the name contains an external / seasonal word (rainfall) | Other numeric measure (0.38) |  |

Dataset-level notes:
- Several columns could be the analysis target (Sales_Value, Sales_Units). Confirm which one(s) to analyse; none is chosen for you.
- Trade / promotion spend (Trade_Spend) is treated as a business variable, not as advertising for a channel. Change its role if that is wrong.
- Sales_Value ≈ Sales_Units × Net_Price (median difference 0.0%)

### Target `Sales_Value`

**Marketing Effectiveness Model**
- ⛔ Channel spend effectiveness (response model) — no channel-wise advertising spend (needs: channel-wise advertising spend columns)
- ⛔ Marketing-mix model (adstock and saturation) — needs channel-wise advertising spend (needs: channel-wise advertising spend and a date column)
- ⛔ ROI / ROAS by channel — ROI / ROAS need channel-wise advertising spend (needs: channel-wise advertising spend)
- ✅ Exposure versus outcome relationships — exposure channels are available
- ✅ Statistical association, elasticities and variable importance — drivers are available
- ✅ Effectiveness over time — date and channels available
- ✅ Effectiveness by brand / geography / product — segment columns available
- ✅ Promotion effectiveness — promotion columns available
- ✅ Business driver analysis (price, promotion, distribution...) — business drivers available

**Channel Performance & Contribution**
- ✅ Channel comparison and ranking — 6 channel(s)
- ✅ Channel trends — date available
- ✅ Statistical contribution share (%) — from the driver model
- ⛔ Model-estimated monetary contribution — needs channel-wise advertising spend (needs: channel-wise advertising spend)
- ⛔ ROI / ROAS by channel — needs channel-wise advertising spend (needs: channel-wise advertising spend and a monetary target)
- ✅ Brand / geography breakdowns — segment columns available

**Marketing Resource Optimizer**
- ⛔ Monetary budget allocation — needs channel-wise advertising spend (monetary) (needs: channel-wise advertising spend and a date column)
- ✅ Minimum / maximum limits and feasibility checks — available with the allocation
- ✅ Exposure mix reallocation (same unit only) — at least two channels share an exposure unit [needs a validated model (holdout R² ≥ 0.3) and positive, significant channel effects]

**Marketing Scenario Simulator**
- ⛔ Spend changes — no channel-wise advertising spend (needs: channel-wise advertising spend columns)
- ✅ Exposure changes — exposure channels
- ✅ Price changes — price column
- ✅ Promotion changes — promotion column
- ✅ Scenario comparison with approximate ranges — drivers available [estimates are flagged as unreliable when the model is weak]

Results (computed in 1.1s):
- Driver model: 19 drivers, 3,120 rows, quality **good** (holdout R² on per-period totals = 0.81 (good ≥ 0.5, usable ≥ 0.3)); validation: chronological: trained on periods before 2023-08-07, tested on the last 21 period(s)
- Channel effects: TV (impressions): elasticity 0.0028, p=2.57e-38, association share 4.6%; YouTube (impressions): elasticity 0.0263, p=1.87e-23, association share 39.7%; Facebook (impressions): elasticity 0.0209, p=8.71e-15, association share 32.0%; Instagram (impressions): elasticity 0.0137, p=2.49e-08, association share 20.2%; Print (readers): elasticity 0.0008, p=5.3e-05, association share 1.0%; Radio (listeners): elasticity 0.0019, p=6e-16, association share 2.5%
- Marketing-mix model / ROI / ROAS: not computed (no channel-wise advertising spend).
- Channel performance: ok; ranking basis: statistical association share (driver model); units: impressions, readers, listeners
- Exposure optimizer (impressions): ok, solver converged=True, estimated change 2.50% (range ±168,991)
- Exposure optimizer (readers): not available — Only one channel is measured in readers; reallocation needs at least two channels in the same unit (different units such as impressions and readers cannot be traded).
- Exposure optimizer (listeners): not available — Only one channel is measured in listeners; reallocation needs at least two channels in the same unit (different units such as impressions and readers cannot be traded).
- Scenario simulator: reliability good; baseline actual 3,895,180.40 vs model 3,894,371.81 per period; scenarios: All channels +10% exposure: 0.59% [extrapolation]; All channels -10% exposure: -0.66% [extrapolation]; Price +5%: 1.96% [extrapolation]; No promotions: 1.45%

### Target `Sales_Units`

**Marketing Effectiveness Model**
- ⛔ Channel spend effectiveness (response model) — no channel-wise advertising spend (needs: channel-wise advertising spend columns)
- ⛔ Marketing-mix model (adstock and saturation) — needs channel-wise advertising spend (needs: channel-wise advertising spend and a date column)
- ⛔ ROI / ROAS by channel — ROI / ROAS need channel-wise advertising spend (needs: channel-wise advertising spend)
- ✅ Exposure versus outcome relationships — exposure channels are available
- ✅ Statistical association, elasticities and variable importance — drivers are available
- ✅ Effectiveness over time — date and channels available
- ✅ Effectiveness by brand / geography / product — segment columns available
- ✅ Promotion effectiveness — promotion columns available
- ✅ Business driver analysis (price, promotion, distribution...) — business drivers available

**Channel Performance & Contribution**
- ✅ Channel comparison and ranking — 6 channel(s)
- ✅ Channel trends — date available
- ✅ Statistical contribution share (%) — from the driver model
- ⛔ Model-estimated monetary contribution — needs channel-wise advertising spend (needs: channel-wise advertising spend)
- ⛔ ROI / ROAS by channel — needs channel-wise advertising spend (needs: channel-wise advertising spend and a monetary target)
- ✅ Brand / geography breakdowns — segment columns available

**Marketing Resource Optimizer**
- ⛔ Monetary budget allocation — needs channel-wise advertising spend (monetary) (needs: channel-wise advertising spend and a date column)
- ✅ Minimum / maximum limits and feasibility checks — available with the allocation
- ✅ Exposure mix reallocation (same unit only) — at least two channels share an exposure unit [needs a validated model (holdout R² ≥ 0.3) and positive, significant channel effects]

**Marketing Scenario Simulator**
- ⛔ Spend changes — no channel-wise advertising spend (needs: channel-wise advertising spend columns)
- ✅ Exposure changes — exposure channels
- ✅ Price changes — price column
- ✅ Promotion changes — promotion column
- ✅ Scenario comparison with approximate ranges — drivers available [estimates are flagged as unreliable when the model is weak]

Results (computed in 1.1s):
- Driver model: 19 drivers, 3,120 rows, quality **good** (holdout R² on per-period totals = 0.83 (good ≥ 0.5, usable ≥ 0.3)); validation: chronological: trained on periods before 2023-08-07, tested on the last 21 period(s)
- Channel effects: TV (impressions): elasticity 0.0035, p=9.74e-23, association share 4.4%; YouTube (impressions): elasticity 0.0305, p=2.84e-13, association share 35.5%; Facebook (impressions): elasticity 0.0314, p=1.58e-12, association share 37.2%; Instagram (impressions): elasticity 0.0172, p=5.81e-05, association share 19.7%; Print (readers): elasticity 0.0006, p=0.0535, association share 0.6%; Radio (listeners): elasticity 0.0026, p=5.18e-12, association share 2.6%
- Marketing-mix model / ROI / ROAS: not computed (no channel-wise advertising spend).
- Channel performance: ok; ranking basis: statistical association share (driver model); units: impressions, readers, listeners
- Exposure optimizer (impressions): ok, solver converged=True, estimated change 3.33% (range ±1,990)
- Exposure optimizer (readers): not available — Only one channel is measured in readers; reallocation needs at least two channels in the same unit (different units such as impressions and readers cannot be traded).
- Exposure optimizer (listeners): not available — Only one channel is measured in listeners; reallocation needs at least two channels in the same unit (different units such as impressions and readers cannot be traded).
- Scenario simulator: reliability good; baseline actual 52,426.12 vs model 52,619.94 per period; scenarios: All channels +10% exposure: 0.81% [extrapolation]; All channels -10% exposure: -0.90% [extrapolation]; Price +5%: 2.54% [extrapolation]; No promotions: -8.55%
