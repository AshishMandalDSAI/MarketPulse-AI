"""Deterministic generator for the PART 02 dataset-compatibility fixtures.

Run ``python tests/fixtures/part02/make_datasets.py`` to (re)create the CSV
files next to this script. Every dataset is synthetic and seeded, so the files
are reproducible byte for byte.

``kaggle_structure_replica.csv`` copies the *column structure* of the user's
Kaggle file ``synthetic_mmm_weekly_india_SAT.csv`` (Week, Geo, Brand, SKU,
Sales_Units, Sales_Value, MRP, Net_Price, promotion flags, Trade_Spend, six
exposure columns, distribution and macro variables). Its values are generated
here; it is NOT the user's data and results on it are not results on the real
file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _weeks(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="W-MON")


def kaggle_structure_replica(seed: int = 7) -> pd.DataFrame:
    """Weekly Geo x Brand x SKU panel, exposure-based (no channel spend)."""
    rng = np.random.default_rng(seed)
    weeks = _weeks(104)
    geos = ["North", "South", "East", "West", "Central"]
    brands = {"Aurel": ["AUR-200G", "AUR-500G"], "Brivo": ["BRV-250G", "BRV-1KG"], "Cendra": ["CEN-100G", "CEN-400G"]}
    mrp = {"AUR-200G": 45.0, "AUR-500G": 99.0, "BRV-250G": 60.0, "BRV-1KG": 210.0, "CEN-100G": 30.0, "CEN-400G": 110.0}
    t = np.arange(len(weeks))
    festival = np.clip(0.2 + 0.8 * np.exp(-((t % 52 - 42) ** 2) / 18.0) + 0.5 * np.exp(-((t % 52 - 10) ** 2) / 8.0), 0, None)
    rainfall = np.clip(np.sin((t % 52 - 22) / 52 * 2 * np.pi) * 50 + 50 + rng.normal(0, 6, len(t)), 0, None)
    cpi = 100 * (1.0 + 0.0011 * t) + rng.normal(0, 0.15, len(t))
    gdp = 6.5 + 0.6 * np.sin(t / 26) + rng.normal(0, 0.1, len(t))
    rows = []
    for b_i, (brand, skus) in enumerate(brands.items()):
        for g_i, geo in enumerate(geos):
            scale = (1.0 + 0.25 * b_i) * (0.8 + 0.12 * g_i)
            tv = np.clip(rng.gamma(2.0, 1.6e6, len(t)) * scale * (rng.random(len(t)) > 0.25), 0, None)
            yt = np.clip(rng.gamma(2.5, 4.0e5, len(t)) * scale, 0, None)
            fb = np.clip(rng.gamma(2.5, 5.0e5, len(t)) * scale, 0, None)
            ig = np.clip(rng.gamma(2.2, 3.5e5, len(t)) * scale, 0, None)
            prn = np.clip(rng.gamma(3.0, 3.0e4, len(t)) * scale * (rng.random(len(t)) > 0.4), 0, None)
            rad = np.clip(rng.gamma(3.0, 6.0e4, len(t)) * scale * (rng.random(len(t)) > 0.3), 0, None)
            for sku in skus:
                feat = (rng.random(len(t)) < 0.18).astype(int)
                disp = (rng.random(len(t)) < 0.22).astype(int)
                tpr = (rng.random(len(t)) < 0.25).astype(int)
                price_m = mrp[sku] * (1 + 0.0009 * t)
                net = price_m * (1 - tpr * rng.uniform(0.08, 0.2, len(t))) * rng.uniform(0.97, 1.0, len(t))
                wd = np.clip(55 + 10 * b_i + rng.normal(0, 4, len(t)) + 0.05 * t, 5, 99)
                nd = np.clip(wd - 8 + rng.normal(0, 3, len(t)), 3, 98)
                tdp = np.round(wd * 2.1 + rng.normal(0, 4, len(t)))
                nos = np.round(1200 * nd / 100 * (1 + 0.2 * g_i) + rng.normal(0, 25, len(t)))
                base = 900 * scale * (mrp["AUR-200G"] / mrp[sku]) ** 0.6
                log_units = (np.log(base) + 0.012 * (wd - 60) - 1.6 * np.log(net / price_m)
                             + 0.18 * feat + 0.12 * disp + 0.10 * festival
                             + 0.045 * np.log1p(tv / 1e6) + 0.035 * np.log1p(yt / 1e5) + 0.03 * np.log1p(fb / 1e5)
                             + 0.02 * np.log1p(ig / 1e5) + 0.01 * np.log1p(prn / 1e4) + 0.012 * np.log1p(rad / 1e4)
                             - 0.002 * (rainfall - 50) + rng.normal(0, 0.08, len(t)))
                units = np.round(np.exp(log_units))
                trade = np.round(units * net * (0.04 + 0.10 * tpr + 0.03 * disp) + rng.normal(0, 50, len(t)), 2).clip(0)
                for i, w in enumerate(weeks):
                    rows.append({
                        "Week": w.strftime("%Y-%m-%d"), "Geo": geo, "Brand": brand, "SKU": sku,
                        "Sales_Units": int(units[i]), "Sales_Value": round(float(units[i] * net[i]), 2),
                        "MRP": round(float(price_m[i]), 2), "Net_Price": round(float(net[i]), 2),
                        "Feature_Flag": int(feat[i]), "Display_Flag": int(disp[i]), "TPR_Flag": int(tpr[i]),
                        "Trade_Spend": float(trade[i]),
                        "TV_Impressions": int(tv[i]), "YouTube_Impressions": int(yt[i]),
                        "Facebook_Impressions": int(fb[i]), "Instagram_Impressions": int(ig[i]),
                        "Print_Readership": int(prn[i]), "Radio_Listenership": int(rad[i]),
                        "Weighted_Distribution": round(float(wd[i]), 2), "Numeric_Distribution": round(float(nd[i]), 2),
                        "TDP": int(tdp[i]), "NOS": int(nos[i]),
                        "CPI": round(float(cpi[i]), 3), "GDP_Growth": round(float(gdp[i]), 3),
                        "Festival_Index": round(float(festival[i]), 4), "Rainfall_Index": round(float(rainfall[i]), 2),
                    })
    return pd.DataFrame(rows)


def _spend_response(rng, spends: dict, base: float, coefs: dict, noise: float) -> np.ndarray:
    y = base + sum(coefs[k] * np.sqrt(v) for k, v in spends.items())
    return np.round(y * rng.normal(1.0, noise, len(next(iter(spends.values())))), 2)


def spend_weekly(seed: int = 11) -> pd.DataFrame:
    """Mode A: weekly channel-wise advertising spend + revenue."""
    rng = np.random.default_rng(seed)
    w = _weeks(120)
    sp = {"TV_Spend": rng.uniform(20e3, 90e3, len(w)), "Search_Spend": rng.uniform(10e3, 50e3, len(w)),
          "Social_Spend": rng.uniform(5e3, 40e3, len(w)), "Radio_Spend": rng.uniform(2e3, 15e3, len(w))}
    rev = _spend_response(rng, sp, 150e3, {"TV_Spend": 900, "Search_Spend": 1200, "Social_Spend": 700, "Radio_Spend": 300}, 0.04)
    df = pd.DataFrame({"Week": w.strftime("%Y-%m-%d"), **{k: np.round(v, 2) for k, v in sp.items()}, "Revenue": rev})
    return df


def exposure_daily(seed: int = 12) -> pd.DataFrame:
    """Mode B: daily exposure metrics (impressions, reach, clicks, views) + conversions."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2024-01-01", periods=240, freq="D")
    fb_reach = rng.integers(20_000, 90_000, len(d))
    meta_views = rng.integers(40_000, 160_000, len(d))
    g_clicks = rng.integers(800, 4_000, len(d))
    yt_imp = rng.integers(100_000, 500_000, len(d))
    conv = np.round(50 + 0.0006 * fb_reach + 0.0002 * meta_views + 0.02 * g_clicks + 0.00005 * yt_imp + rng.normal(0, 8, len(d)))
    return pd.DataFrame({"Date": d.strftime("%d/%m/%Y"), "FB_Reach": fb_reach, "Meta_Views": meta_views,
                         "Google_Clicks": g_clicks, "YouTube_Impr": yt_imp, "Conversions": conv.astype(int)})


def general_sales(seed: int = 13) -> pd.DataFrame:
    """Mode C: order-level sales with product, region, price and discount; no marketing."""
    rng = np.random.default_rng(seed)
    n = 1500
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    products = rng.choice(["Kettle", "Toaster", "Blender", "Mixer", "Iron"], n)
    cats = {"Kettle": "Kitchen", "Toaster": "Kitchen", "Blender": "Kitchen", "Mixer": "Kitchen", "Iron": "Laundry"}
    price = {"Kettle": 30.0, "Toaster": 45.0, "Blender": 70.0, "Mixer": 120.0, "Iron": 35.0}
    qty = rng.integers(1, 8, n)
    disc = np.round(rng.choice([0, 0, 0, 0.05, 0.1, 0.15], n), 2)
    up = np.array([price[p] for p in products])
    return pd.DataFrame({"Order_ID": [f"ORD-{100000 + i}" for i in range(n)], "Order_Date": dates.strftime("%Y-%m-%d"),
                         "Region": rng.choice(["North", "South", "East", "West"], n), "Product": products,
                         "Category": [cats[p] for p in products], "Quantity": qty, "Unit_Price": up,
                         "Discount": disc, "Net_Sales": np.round(qty * up * (1 - disc), 2)})


def renamed_columns(seed: int = 14) -> pd.DataFrame:
    """Mode A with unusual names: Television_Cost, Online_Ads_Investment, Meta_Ad_Cost, Sales_Turnover."""
    rng = np.random.default_rng(seed)
    w = _weeks(100, "2023-02-06")
    sp = {"Television_Cost": rng.uniform(30e3, 80e3, len(w)), "Online_Ads_Investment": rng.uniform(10e3, 40e3, len(w)),
          "Meta_Ad_Cost": rng.uniform(5e3, 25e3, len(w))}
    rev = _spend_response(rng, sp, 100e3, {"Television_Cost": 700, "Online_Ads_Investment": 900, "Meta_Ad_Cost": 600}, 0.05)
    return pd.DataFrame({"Period": w.strftime("%Y-%m-%d"), **{k: np.round(v, 2) for k, v in sp.items()},
                         "Sales_Turnover": rev})


def multi_revenue(seed: int = 15) -> pd.DataFrame:
    """Several plausible targets: Gross_Revenue, Net_Revenue, Total_Sales, plus channel spend."""
    rng = np.random.default_rng(seed)
    w = _weeks(90)
    sp = {"TV_Spend": rng.uniform(20e3, 60e3, len(w)), "Digital_Spend": rng.uniform(10e3, 40e3, len(w))}
    gross = _spend_response(rng, sp, 200e3, {"TV_Spend": 800, "Digital_Spend": 1000}, 0.04)
    net = np.round(gross * rng.uniform(0.85, 0.9, len(w)), 2)
    tot = np.round(gross * rng.uniform(0.98, 1.02, len(w)), 2)
    return pd.DataFrame({"Week": w.strftime("%Y-%m-%d"), **{k: np.round(v, 2) for k, v in sp.items()},
                         "Gross_Revenue": gross, "Net_Revenue": net, "Total_Sales": tot})


def missing_marketing(seed: int = 16) -> pd.DataFrame:
    """Outcome data but no marketing spend or exposure columns (Mode C)."""
    rng = np.random.default_rng(seed)
    m = pd.date_range("2021-01-01", periods=36, freq="MS")
    rows = []
    for store in ["S01", "S02", "S03", "S04"]:
        cust = rng.integers(800, 1600, len(m))
        rev = np.round(cust * rng.uniform(22, 30, len(m)), 2)
        for i, d in enumerate(m):
            rows.append({"Month": d.strftime("%Y-%m"), "Store": store, "Customers": int(cust[i]),
                         "Revenue": float(rev[i]), "Avg_Basket": round(float(rev[i] / cust[i]), 2)})
    return pd.DataFrame(rows)


def one_channel(seed: int = 17) -> pd.DataFrame:
    """Mode A with a single advertising channel."""
    rng = np.random.default_rng(seed)
    w = _weeks(80)
    sp = {"Digital_Spend": rng.uniform(10e3, 60e3, len(w))}
    rev = _spend_response(rng, sp, 80e3, {"Digital_Spend": 1100}, 0.04)
    return pd.DataFrame({"Week": w.strftime("%Y-%m-%d"), "Digital_Spend": np.round(sp["Digital_Spend"], 2), "Revenue": rev})


def two_channel(seed: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = _weeks(80)
    sp = {"TV_Spend": rng.uniform(20e3, 70e3, len(w)), "Search_Spend": rng.uniform(5e3, 30e3, len(w))}
    rev = _spend_response(rng, sp, 90e3, {"TV_Spend": 700, "Search_Spend": 1000}, 0.04)
    return pd.DataFrame({"Week": w.strftime("%Y-%m-%d"), **{k: np.round(v, 2) for k, v in sp.items()}, "Revenue": rev})


def brand_geo_spend(seed: int = 19) -> pd.DataFrame:
    """Mode A panel with brands and regions, plus a promotion flag and price."""
    rng = np.random.default_rng(seed)
    w = _weeks(78)
    rows = []
    for brand in ["Zeno", "Orla"]:
        for region in ["Delhi NCR", "Mumbai", "Bengaluru"]:
            tv = rng.uniform(5e3, 30e3, len(w))
            dig = rng.uniform(2e3, 15e3, len(w))
            promo = (rng.random(len(w)) < 0.3).astype(int)
            price = np.round(rng.uniform(95, 110, len(w)), 2)
            units = np.round(400 + 1.2 * np.sqrt(tv) + 1.8 * np.sqrt(dig) + 60 * promo - 2.5 * (price - 100) + rng.normal(0, 15, len(w)))
            for i, d in enumerate(w):
                rows.append({"Week_Start": d.strftime("%Y-%m-%d"), "Brand": brand, "Region": region,
                             "TV_Spend": round(float(tv[i]), 2), "Digital_Spend": round(float(dig[i]), 2),
                             "Promo_Flag": int(promo[i]), "Avg_Price": float(price[i]),
                             "Units_Sold": int(units[i]), "Revenue": round(float(units[i] * price[i]), 2)})
    return pd.DataFrame(rows)


def mixed_types(seed: int = 20) -> pd.DataFrame:
    """Mixed categorical/numerical with leads as the outcome and a channel-name dimension."""
    rng = np.random.default_rng(seed)
    w = _weeks(60)
    rows = []
    for ch in ["Email", "Webinar", "Paid Social"]:
        visits = rng.integers(500, 4000, len(w))
        for i, d in enumerate(w):
            rows.append({"Week": d.strftime("%Y-%m-%d"), "Campaign_Type": ch, "Segment": rng.choice(["SMB", "Enterprise"]),
                         "Website_Visits": int(visits[i]), "Is_Holiday": int(rng.random() < 0.08),
                         "Leads": int(visits[i] * rng.uniform(0.02, 0.05)), "Notes": rng.choice(["", "launch", "refresh", "n/a"])})
    return pd.DataFrame(rows)


DATASETS = {
    "kaggle_structure_replica.csv": kaggle_structure_replica,
    "spend_weekly.csv": spend_weekly,
    "exposure_daily.csv": exposure_daily,
    "general_sales.csv": general_sales,
    "renamed_columns.csv": renamed_columns,
    "multi_revenue.csv": multi_revenue,
    "missing_marketing.csv": missing_marketing,
    "one_channel.csv": one_channel,
    "two_channel.csv": two_channel,
    "brand_geo_spend.csv": brand_geo_spend,
    "mixed_types.csv": mixed_types,
}


def main() -> None:
    for name, fn in DATASETS.items():
        fn().to_csv(HERE / name, index=False)
        print(name)


if __name__ == "__main__":
    main()
