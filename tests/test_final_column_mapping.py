"""Final stabilization — dataset-agnostic column-role detection.

Exercises the date / revenue / spend column names named in the final release
specification, plus guards against misleading guesses (numeric IDs offered as
dates or revenue). Guesses below HIGH confidence must still require
confirmation; nothing here relaxes that rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.pipeline import profile_and_map, confirm_and_standardize  # noqa: E402

N = 60
RNG = np.random.default_rng(11)


def _dates():
    return pd.date_range("2024-01-01", periods=N, freq="W").strftime("%Y-%m-%d")


def _money(lo=1_000, hi=9_000):
    return RNG.uniform(lo, hi, N).round(2)


@pytest.mark.parametrize("date_name", ["Date", "Order_Date", "Transaction_Time", "Sales_Date"])
def test_date_column_names_detected(date_name):
    df = pd.DataFrame({date_name: _dates(), "Revenue": _money(), "Meta_Ads": _money()})
    _, mapping = profile_and_map(df)
    assert mapping.date.column == date_name


@pytest.mark.parametrize("revenue_name", ["Revenue", "Net_Sales", "Total_Amount", "Turnover"])
def test_revenue_column_names_detected(revenue_name):
    df = pd.DataFrame({"Date": _dates(), revenue_name: _money(10_000, 90_000), "Meta_Ads": _money()})
    _, mapping = profile_and_map(df)
    assert mapping.revenue.column == revenue_name


@pytest.mark.parametrize("spend_name", [
    "Facebook_Spend", "Meta_Ads", "Google_Advertising", "Search_Ad_Cost", "Marketing_Investment",
])
def test_spend_column_names_detected_as_channels(spend_name):
    df = pd.DataFrame({"Date": _dates(), "Revenue": _money(10_000, 90_000), spend_name: _money()})
    _, mapping = profile_and_map(df)
    assert spend_name in [g.column for g in mapping.channels]


def test_full_spec_schema_enables_mmm_inputs():
    df = pd.DataFrame({
        "Order_Date": _dates(), "Net_Sales": _money(20_000, 90_000),
        "Facebook_Spend": _money(), "Google_Advertising": _money(), "Marketing_Investment": _money(),
    })
    _, mapping = profile_and_map(df)
    dataset, _ = confirm_and_standardize(df, mapping, confirmed_roles={"date", "revenue", "channels"})
    assert dataset.date_col == "Order_Date"
    assert dataset.revenue_col == "Net_Sales"
    assert set(dataset.channels) == {"Facebook_Spend", "Google_Advertising", "Marketing_Investment"}


def test_numeric_id_column_never_guessed_as_date():
    df = pd.DataFrame({"student": range(50), "score": RNG.normal(70, 10, 50)})
    _, mapping = profile_and_map(df)
    assert mapping.date.column is None


def test_identifier_column_not_guessed_as_revenue():
    df = pd.DataFrame({"customer_id": range(50), "score": RNG.normal(70, 10, 50)})
    _, mapping = profile_and_map(df)
    assert mapping.revenue.column != "customer_id"


def test_statistical_guesses_still_need_confirmation():
    df = pd.DataFrame({"when": _dates(), "x": _money()})
    _, mapping = profile_and_map(df)
    assert mapping.date.column == "when"
    assert mapping.date.needs_confirmation
    dataset, _ = confirm_and_standardize(df, mapping, confirmed_roles=set())
    assert dataset.date_col is None  # not silently applied
