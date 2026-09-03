"""Discrimination and calibration, including the tie handling a scorecard actually produces."""

import numpy as np
import pytest

from src.metrics import (auc_score, bootstrap_gini_interval, calibration_in_the_large,
                         calibration_table, discrimination, ks_statistic, wilson_interval)


def test_perfect_and_inverted_separation():
    y = [0, 0, 1, 1]
    assert auc_score(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc_score(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_ties_are_averaged_not_broken_arbitrarily():
    """A scorecard bins, so tied scores are the normal case. Every loan tied is AUC 0.5."""
    assert auc_score([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # One tied pair out of four: three pairs ordered correctly plus a half, over four pairs.
    assert auc_score([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)


def test_auc_is_undefined_with_one_class():
    assert np.isnan(auc_score([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(ks_statistic([1, 1, 1], [0.1, 0.2, 0.3]))


def test_gini_is_twice_auc_minus_one():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    result = discrimination(y, p)
    assert result.gini == pytest.approx(2 * result.auc - 1)


def test_bootstrap_interval_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 800)
    p = np.clip(y * 0.3 + rng.random(800) * 0.7, 0, 1)
    point = 2 * auc_score(y, p) - 1
    low, high = bootstrap_gini_interval(y, p, samples=200, confidence=0.90, seed=1)
    assert low < point < high
    assert (low, high) == bootstrap_gini_interval(y, p, samples=200, confidence=0.90, seed=1)


def test_wilson_stays_inside_zero_and_one_on_small_samples():
    """The normal approximation goes negative here. That is why Wilson is used."""
    low, high = wilson_interval(1, 20, 0.95)
    assert 0.0 <= low < high <= 1.0
    assert low > 0.0


def test_wilson_on_zero_successes_has_a_zero_floor_and_a_real_ceiling():
    low, high = wilson_interval(0, 100, 0.95)
    assert low == 0.0
    assert 0.0 < high < 0.1


def test_unsupported_confidence_is_refused_rather_than_extrapolated():
    with pytest.raises(ValueError, match="outside the supported range"):
        wilson_interval(5, 100, 0.999)


def test_calibration_table_marks_thin_bands_insufficient():
    y = np.array([0] * 30 + [1] * 5)
    p = np.full(35, 0.1)
    bands = np.zeros(35, dtype=int)
    table = calibration_table(y, p, bands, n_bands=2, min_band_count=50)
    assert table.loc[0, "sufficient"] == False  # noqa: E712
    assert table.loc[1, "n"] == 0


def test_calibration_in_the_large_reports_the_ratio_and_its_interval():
    y = np.array([1] * 20 + [0] * 80)
    p = np.full(100, 0.10)
    result = calibration_in_the_large(y, p, confidence=0.95)
    assert result["realised_pd"] == pytest.approx(0.20)
    assert result["ratio"] == pytest.approx(2.0)
    assert result["ratio_low"] < result["ratio"] < result["ratio_high"]
    # The whole interval sits above 1, so this is a breach and not a small sample.
    assert result["ratio_low"] > 1.0
