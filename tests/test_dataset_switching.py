"""Dataset switching integration tests — Phase 0.

Verifies that switching between different dataset types produces
correct, isolated results with no cross-dataset leakage.

Scenarios:
- Dataset A -> Dataset B (marketing -> sales)
- Marketing -> Non-Marketing
- Non-Marketing -> Marketing
- Forecastable -> Non-Forecastable
- Multiple sequential switches (A -> B -> C)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers import (
    load_fixture,
    run_pipeline_on_fixture,
)


def _run_and_extract_caps(fixture_name, subdir="diverse_schemas", confirmed=None):
    """Run pipeline on a fixture and return (dataset, capabilities) tuple."""
    df, profile, mapping, dataset, caps = run_pipeline_on_fixture(
        fixture_name, subdir, confirmed_roles=confirmed
    )
    return dataset, caps


# ---- A -> B: marketing -> sales ----

def test_switch_marketing_to_sales():
    """After processing a marketing dataset, processing a sales dataset
    must produce capabilities appropriate for sales, not marketing."""
    _, caps_mkt = _run_and_extract_caps("a_marketing.csv")
    _, caps_sales = _run_and_extract_caps("b_sales.csv")

    # Marketing dataset should have MMM capability (has date + revenue + channels)
    assert caps_mkt.is_available("marketing_mix_modeling"), \
        "Marketing fixture should support MMM"

    # Sales dataset has no channel spend columns -> no MMM
    assert not caps_sales.is_available("marketing_mix_modeling"), \
        "Sales fixture should NOT support MMM (no spend channels)"

    # Both should have generic KPIs
    assert caps_mkt.is_available("generic_kpis")
    assert caps_sales.is_available("generic_kpis")


# ---- Non-Marketing -> Marketing ----

def test_switch_nonmarketing_to_marketing():
    """A non-marketing dataset followed by a marketing dataset must
    correctly enable MMM capabilities for the marketing one."""
    _, caps_ops = _run_and_extract_caps("d_operations.csv")
    _, caps_mkt = _run_and_extract_caps("a_marketing.csv")

    assert not caps_ops.is_available("marketing_mix_modeling"), \
        "Operations fixture should NOT support MMM"
    assert caps_mkt.is_available("marketing_mix_modeling"), \
        "Marketing fixture should support MMM after non-marketing"


# ---- Forecastable -> Non-Forecastable ----

def test_switch_forecastable_to_nonforecastable():
    """A dataset with date + numeric target -> forecasting available.
    Then a no-date dataset -> forecasting unavailable."""
    _, caps_ts = _run_and_extract_caps("a_marketing.csv")
    _, caps_nodate = _run_and_extract_caps("g_no_date.csv")

    assert caps_ts.is_available("forecasting"), \
        "Time-series fixture should support forecasting"
    assert not caps_nodate.is_available("forecasting"), \
        "No-date fixture should NOT support forecasting"


# ---- Sequential: A -> B -> C ----

def test_triple_switch_no_leakage():
    """Three sequential datasets produce independent capability reports."""
    _, caps_a = _run_and_extract_caps("a_marketing.csv")
    _, caps_b = _run_and_extract_caps("e_education.csv")
    _, caps_c = _run_and_extract_caps("c_finance.csv")

    # Marketing has MMM; education doesn't; finance doesn't
    assert caps_a.is_available("marketing_mix_modeling")
    assert not caps_b.is_available("marketing_mix_modeling")
    assert not caps_c.is_available("marketing_mix_modeling")

    # Education has no date -> no trend analysis
    assert not caps_b.is_available("trend_analysis")

    # Finance has date -> trend analysis
    assert caps_c.is_available("trend_analysis")


# ---- CSV vs different structure ----

def test_unusual_columns_do_not_crash_pipeline():
    """Dataset H has completely non-standard column names.
    The pipeline must not crash — it should gracefully produce
    a capability report even if most features are unavailable."""
    _, caps = _run_and_extract_caps("h_unusual_columns.csv")

    # Should always have data profiling and generic KPIs
    assert caps.is_available("data_profiling")
    assert caps.is_available("generic_kpis")


# ---- Missing values dataset ----

def test_missing_values_dataset_does_not_crash():
    """Dataset I has 10-30% missing values per column.
    Pipeline must handle gracefully."""
    _, caps = _run_and_extract_caps("i_missing_values.csv")
    assert caps.is_available("data_profiling")
    assert caps.is_available("generic_kpis")


# ---- Minimal dataset ----

def test_minimal_dataset_handles_edge_case():
    """Dataset K has only 5 rows. Pipeline must not crash,
    and forecasting should be unavailable (too few points)."""
    _, caps = _run_and_extract_caps("k_minimal.csv")
    assert caps.is_available("data_profiling")
    assert not caps.is_available("forecasting"), \
        "5-row dataset should NOT support forecasting"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}")
    print("\nAll dataset switching tests passed.")
