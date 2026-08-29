"""Tests for adstock, saturation, MMM, and channel contribution."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.models.adstock import geometric_adstock, normalize
from src.models.saturation import hill_saturation, fit_half_saturation_point


def test_geometric_adstock_zero_decay_is_identity():
    spend = np.array([100.0, 200.0, 0.0, 50.0])
    result = geometric_adstock(spend, decay=0.0)
    np.testing.assert_allclose(result, spend)


def test_geometric_adstock_carries_forward():
    spend = np.array([100.0, 0.0, 0.0])
    result = geometric_adstock(spend, decay=0.5)
    assert result[1] > 0  # carryover from week 0 should still show up in week 1
    assert result[1] == 50.0
    assert result[2] == 25.0


def test_geometric_adstock_never_negative():
    spend = np.random.default_rng(0).uniform(0, 1000, 50)
    result = geometric_adstock(spend, decay=0.4)
    assert (result >= 0).all()


def test_hill_saturation_bounds():
    x = np.array([0, 100, 1000, 1_000_000])
    sat = hill_saturation(x, k=500, s=1.5)
    assert (sat >= 0).all()
    assert (sat < 1).all()
    assert sat[0] == 0.0


def test_hill_saturation_monotonic_increasing():
    x = np.linspace(0, 10000, 50)
    sat = hill_saturation(x, k=2000, s=1.3)
    assert np.all(np.diff(sat) >= -1e-9)  # monotonic non-decreasing


def test_fit_half_saturation_point_positive():
    x = np.array([100, 200, 300, 400, 500])
    k = fit_half_saturation_point(x, quantile=0.5)
    assert k > 0


def test_normalize_range():
    x = np.array([10, 20, 30, 40, 50])
    norm = normalize(x)
    assert norm.min() == 0.0
    assert norm.max() == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}")
