"""
MarketPulse AI - Synthetic Dataset Generator
==============================================
Generates a realistic ~4-year (208-week) synthetic weekly marketing dataset
for the fictional e-commerce company "NovaMart".

IMPORTANT: This dataset is entirely SYNTHETIC and was generated for academic /
final-year project demonstration purposes. It does NOT represent real company
data. The data-generating process deliberately encodes marketing carryover
(adstock), diminishing returns (saturation), seasonality, promotions,
holidays, price effects, competitor effects, economic effects, and noise so
that downstream Marketing Mix Modeling has genuine, recoverable structure to
find (rather than being pure random noise).

Run:
    python src/data/generate_dataset.py
Output:
    data/raw/marketing_data_raw.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_WEEKS = 208  # ~4 years of weekly data
START_DATE = "2022-01-03"  # a Monday

CHANNELS = [
    "TV", "Radio", "Print", "Search",
    "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing",
]

# Base weekly spend level (INR) and volatility per channel — traditional
# channels get larger, lumpier budgets; digital channels are smaller/steadier.
CHANNEL_PARAMS = {
    "TV":                   {"base": 450000, "vol": 0.35, "adstock_decay": 0.55, "sat_k": 500000, "sat_slope": 1.3, "beta": 1.35},
    "Radio":                {"base": 120000, "vol": 0.30, "adstock_decay": 0.40, "sat_k": 150000, "sat_slope": 1.2, "beta": 0.55},
    "Print":                {"base": 80000,  "vol": 0.25, "adstock_decay": 0.25, "sat_k": 100000, "sat_slope": 1.1, "beta": 0.30},
    "Search":               {"base": 220000, "vol": 0.20, "adstock_decay": 0.20, "sat_k": 260000, "sat_slope": 1.4, "beta": 1.55},
    "Social_Media":         {"base": 180000, "vol": 0.28, "adstock_decay": 0.30, "sat_k": 220000, "sat_slope": 1.3, "beta": 1.10},
    "Display_Ads":          {"base": 110000, "vol": 0.30, "adstock_decay": 0.35, "sat_k": 140000, "sat_slope": 1.2, "beta": 0.65},
    "Email_Marketing":      {"base": 25000,  "vol": 0.15, "adstock_decay": 0.15, "sat_k": 35000,  "sat_slope": 1.1, "beta": 0.90},
    "Influencer_Marketing": {"base": 95000,  "vol": 0.40, "adstock_decay": 0.45, "sat_k": 120000, "sat_slope": 1.25, "beta": 0.80},
}

BASE_REVENUE = 2_800_000  # baseline weekly revenue with zero marketing effect


def geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock (carryover) transform: effect_t = spend_t + decay * effect_{t-1}."""
    out = np.zeros_like(spend, dtype=float)
    carry = 0.0
    for t in range(len(spend)):
        carry = spend[t] + decay * carry
        out[t] = carry
    return out


def hill_saturation(x: np.ndarray, k: float, s: float) -> np.ndarray:
    """Hill-function saturation transform producing diminishing returns in [0, 1)."""
    x = np.clip(x, 0, None)
    return x ** s / (k ** s + x ** s)


def generate_dataset(seed: int = RANDOM_SEED, n_weeks: int = N_WEEKS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=START_DATE, periods=n_weeks, freq="W-MON")

    week_idx = np.arange(n_weeks)
    year = dates.year.values
    week_of_year = dates.isocalendar().week.values.astype(int)

    # ---------------------------------------------------------------
    # 1. Seasonality: annual cycle + a gentle upward business trend
    # ---------------------------------------------------------------
    annual_cycle = np.sin(2 * np.pi * week_of_year / 52.0)
    seasonality_index = 1.0 + 0.14 * annual_cycle + 0.05 * np.sin(4 * np.pi * week_of_year / 52.0)
    trend = 1.0 + 0.0022 * week_idx  # slow organic growth over 4 years

    # ---------------------------------------------------------------
    # 2. Holidays: festive/sale weeks (roughly Black Friday / New Year /
    #    Diwali-style patterns) recurring near specific weeks each year
    # ---------------------------------------------------------------
    holiday_weeks = {47, 48, 51, 52, 1, 26, 39, 40}
    holiday = np.array([1 if w in holiday_weeks else 0 for w in week_of_year])

    # ---------------------------------------------------------------
    # 3. Season label
    # ---------------------------------------------------------------
    def season_from_week(w):
        if w in (12, 1, 2) or w >= 50 or w <= 2:
            return "Winter"
        if 3 <= w <= 21:
            return "Spring"
        if 22 <= w <= 35:
            return "Summer"
        return "Autumn"
    season = [season_from_week(dates[i].month) for i in range(n_weeks)]

    # ---------------------------------------------------------------
    # 4. Promotion spend: more likely during holiday weeks
    # ---------------------------------------------------------------
    promo_prob = np.where(holiday == 1, 0.85, 0.18)
    promo_flag = rng.binomial(1, promo_prob)
    promotion_spend = promo_flag * rng.normal(60000, 15000, n_weeks).clip(min=0)
    promotion_spend += holiday * rng.normal(40000, 10000, n_weeks).clip(min=0)
    promotion_spend = promotion_spend.clip(min=0)

    # ---------------------------------------------------------------
    # 5. Marketing channel spend (base * seasonal wobble * noise),
    #    with a few deliberate budget-shift "campaigns" over time.
    # ---------------------------------------------------------------
    spend = {}
    for ch, p in CHANNEL_PARAMS.items():
        base_series = p["base"] * (1 + 0.10 * annual_cycle) * trend
        noise = rng.normal(1.0, p["vol"], n_weeks)
        series = base_series * noise
        # Boost digital spend gradually over time (secular shift to digital)
        if ch in ("Search", "Social_Media", "Display_Ads", "Email_Marketing", "Influencer_Marketing"):
            series *= (1 + 0.0015 * week_idx)
        else:
            series *= (1 - 0.0006 * week_idx)
        # Extra push during holiday weeks
        series *= (1 + 0.35 * holiday)
        series = series.clip(min=0)
        spend[ch] = series

    # ---------------------------------------------------------------
    # 6. Adstock + saturation transforms feeding the TRUE revenue model
    #    (the "ground truth" data-generating process — MMM should
    #    approximately recover this structure).
    # ---------------------------------------------------------------
    channel_effect = np.zeros(n_weeks)
    adstocked = {}
    saturated = {}
    for ch, p in CHANNEL_PARAMS.items():
        ad = geometric_adstock(spend[ch], p["adstock_decay"])
        sat = hill_saturation(ad, p["sat_k"], p["sat_slope"])
        adstocked[ch] = ad
        saturated[ch] = sat
        channel_effect += p["beta"] * sat * BASE_REVENUE * 0.12

    # ---------------------------------------------------------------
    # 7. Price, competitor, and economic effects
    # ---------------------------------------------------------------
    average_price = 1350 + 60 * np.sin(2 * np.pi * week_idx / 52 + 1.0) + rng.normal(0, 20, n_weeks)
    price_index = average_price / average_price.mean()
    price_effect = -0.55 * (price_index - 1.0)  # higher price -> lower demand

    competitor_index = 100 + 8 * np.sin(2 * np.pi * week_idx / 30 + 0.5) + rng.normal(0, 3, n_weeks)
    competitor_effect = -0.20 * (competitor_index - 100) / 100

    economic_index = 100 + 0.03 * week_idx + 5 * np.sin(2 * np.pi * week_idx / 104) + rng.normal(0, 2, n_weeks)
    economic_effect = 0.30 * (economic_index - 100) / 100

    promo_effect = 0.9 * (promotion_spend / (promotion_spend.mean() + 1e-6)) * 0.05 * BASE_REVENUE / 10
    holiday_effect = holiday * BASE_REVENUE * 0.10

    # ---------------------------------------------------------------
    # 8. Assemble revenue
    # ---------------------------------------------------------------
    revenue = (
        BASE_REVENUE * seasonality_index * trend
        + channel_effect
        + holiday_effect
        + promo_effect
        + BASE_REVENUE * (price_effect + competitor_effect + economic_effect)
    )
    revenue *= rng.normal(1.0, 0.035, n_weeks)  # residual noise
    revenue = revenue.clip(min=BASE_REVENUE * 0.4)

    # A handful of controlled anomalies (e.g., stockout week, viral moment)
    anomaly_weeks = rng.choice(n_weeks, size=4, replace=False)
    anomaly_multiplier = np.ones(n_weeks)
    anomaly_multiplier[anomaly_weeks[:2]] = 0.70   # negative shocks (e.g. supply issue)
    anomaly_multiplier[anomaly_weeks[2:]] = 1.35   # positive shocks (e.g. viral campaign)
    revenue *= anomaly_multiplier

    # ---------------------------------------------------------------
    # 9. Orders, traffic, customers derived from revenue + price
    # ---------------------------------------------------------------
    orders = (revenue / average_price) * rng.normal(1.0, 0.02, n_weeks)
    website_traffic = orders * rng.uniform(14, 22, n_weeks) * (1 + 0.4 * saturated["Search"] + 0.3 * saturated["Social_Media"])
    new_customer_share = np.clip(0.28 + 0.10 * saturated["Influencer_Marketing"] + rng.normal(0, 0.03, n_weeks), 0.1, 0.6)
    new_customers = (orders * new_customer_share).round()
    returning_customers = (orders - new_customers).clip(min=0).round()

    # ---------------------------------------------------------------
    # 10. Assemble DataFrame
    # ---------------------------------------------------------------
    df = pd.DataFrame({"Date": dates})
    for ch in CHANNELS:
        df[f"{ch}_Spend"] = spend[ch].round(2)
    df["Promotion_Spend"] = promotion_spend.round(2)
    df["Average_Price"] = average_price.round(2)
    df["Website_Traffic"] = website_traffic.round(0)
    df["New_Customers"] = new_customers
    df["Returning_Customers"] = returning_customers
    df["Orders"] = orders.round(0)
    df["Holiday"] = holiday
    df["Season"] = season
    df["Competitor_Index"] = competitor_index.round(2)
    df["Economic_Index"] = economic_index.round(2)
    df["Revenue"] = revenue.round(2)

    # Inject a small, realistic amount of missingness for the data-cleaning
    # phase to legitimately have something to do (bounded, non-critical fields only)
    miss_idx = rng.choice(n_weeks, size=3, replace=False)
    df.loc[miss_idx, "Competitor_Index"] = np.nan

    return df


def main():
    df = generate_dataset()
    out_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "marketing_data_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} weekly rows -> {out_path}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
