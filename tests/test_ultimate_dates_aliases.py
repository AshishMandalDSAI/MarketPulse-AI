"""Day/month order and channel-cost aliases (found by the dataset-compatibility sweep).

Before the fix, 104 weekly dd/mm/yyyy dates parsed with pandas' default produced
64 NaT and 40 wrong dates (03/01/2022 -> 1 March), silently.
"""
from __future__ import annotations

import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from src.data_engine.date_parsing import detect_date_order, parse_date_column


def _weekly(n=104, fmt="%d/%m/%Y"):
    d = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    return d, pd.Series(d.strftime(fmt))


def test_day_first_column_is_parsed_exactly():
    truth, s = _weekly()
    out, info = parse_date_column(s)
    assert info.order == "dayfirst" and info.unparseable == 0
    assert (out.reset_index(drop=True) == pd.Series(truth)).all()


def test_month_first_column_is_parsed_exactly():
    truth, s = _weekly(fmt="%m/%d/%Y")
    out, info = parse_date_column(s)
    assert info.order == "monthfirst" and (out.reset_index(drop=True) == pd.Series(truth)).all()


def test_iso_text_and_blanks():
    out, info = parse_date_column(pd.Series(["2021-01-02", "", None, "garbage"]))
    assert info.order == "iso_or_text"
    assert out.iloc[0] == pd.Timestamp("2021-01-02")
    assert out.iloc[1:].isna().all()
    assert info.unparseable == 1 and info.examples_unparseable == ["garbage"]   # blanks are missing, not invalid


def test_ambiguous_order_is_reported_not_silent():
    out, info = parse_date_column(pd.Series(["03/01/2022", "04/01/2022"]))
    assert info.order == "ambiguous" and "ambiguous" in info.note.lower()


@pytest.mark.parametrize("values,order", [
    (["13/01/2022", "01/02/2022"], "dayfirst"),
    (["01/13/2022", "02/01/2022"], "monthfirst"),
    (["13/01/2022", "01/13/2022"], "inconsistent"),
    (["2022-01-13"], "iso_or_text"),
])
def test_detect_date_order(values, order):
    assert detect_date_order(pd.Series(values)) == order


def test_numbers_are_never_dates():
    out, info = parse_date_column(pd.Series([20220101, 20220102]))
    assert out.isna().all()


def test_quality_engine_counts_no_invalid_dates_for_day_first():
    from src.data_engine.data_quality import assess_quality
    _, s = _weekly()
    rep = assess_quality(pd.DataFrame({"Week": s, "Revenue": np.arange(104.0)}))
    assert rep["columns"]["Week"]["kind"] == "date"
    assert rep["columns"]["Week"]["n_invalid"] == 0


def test_channel_cost_columns_are_spend_not_business_cost():
    from src.data_engine.column_classifier import classify_columns
    df = pd.DataFrame({"facebook_cost": [1.0, 2, 3], "TV Budget": [3.0, 4, 5], "cost_of_goods": [1.0, 2, 4],
                       "revenue": [5.0, 6, 7]})
    m = classify_columns(df)
    assert {g.column for g in m.channels} == {"facebook_cost", "TV Budget"}
    assert m.cost.column == "cost_of_goods"


def test_weekly_day_first_mmm_dataset_end_to_end():
    from config.settings import dataset_processed_dir, dataset_upload_dir
    from src.data_engine.analysis_runner import run_full_analysis
    from src.data_engine.pipeline import profile_and_map

    rng = np.random.default_rng(7)
    truth, s = _weekly()
    tv, fb, gg = rng.uniform(1e4, 5e4, 104), rng.uniform(5e3, 3e4, 104), rng.uniform(5e3, 2e4, 104)
    df = pd.DataFrame({"week_start": s, "tv_spend": tv, "facebook_cost": fb, "google_ads_spend": gg,
                       "sales_revenue": 2e5 + 1.8 * tv + 2.5 * fb + 3 * gg + rng.normal(0, 2e4, 104)})
    _, mapping = profile_and_map(df)
    did = f"upload_dd{uuid.uuid4().hex[:8]}"
    try:
        res = run_full_analysis(did, df=df, mapping=mapping, confirmed_roles={"date", "revenue", "channels"})
        caps = {c.name: c.available for c in res["capabilities"].capabilities}
        assert caps["Marketing Mix Modeling"] and caps["Forecasting"]
        log = " ".join(res["report"])
        assert "day/month/year" in log
        std = pd.read_csv(dataset_processed_dir(did) / "standardized_data.csv")
        parsed = pd.to_datetime(std["week_start"])
        assert parsed.min() == truth.min() and parsed.max() == truth.max() and parsed.notna().all()
    finally:
        shutil.rmtree(dataset_upload_dir(did), ignore_errors=True)
        shutil.rmtree(dataset_processed_dir(did), ignore_errors=True)
