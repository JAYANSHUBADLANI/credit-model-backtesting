"""Stability indices and the timing arithmetic that is the point of having them here."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.stability import (bridge, classify, first_breach_summary, signal_timing,
                           stability_index)

CONFIG = load_config()


def test_an_unmoved_population_has_a_zero_index():
    reference = np.array([0.2, 0.3, 0.5])
    assert stability_index(reference, reference) == pytest.approx(0.0)


def test_an_empty_bin_gives_a_large_finite_number_not_an_infinity():
    value = stability_index(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    assert np.isfinite(value)
    assert value > 1.0


def test_mismatched_bin_counts_are_refused():
    with pytest.raises(ValueError, match="same number of bins"):
        stability_index(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


def test_thresholds_classify_at_the_boundary():
    assert classify(0.09, 0.10, 0.25) == "ok"
    assert classify(0.10, 0.10, 0.25) == "warn"
    assert classify(0.25, 0.10, 0.25) == "alert"


def test_the_outcome_is_knowable_exactly_one_window_after_the_vintage_closes():
    timing = signal_timing(2007, window_months=24)
    assert timing["stability_known_by"] == 200712
    assert timing["outcome_known_by"] == 200912
    assert timing["lead_months"] == 24


def test_a_reporting_lag_is_added_to_the_outcome_side_only():
    timing = signal_timing(2007, window_months=24, reporting_lag_months=3)
    assert timing["stability_known_by"] == 200712
    assert timing["outcome_known_by"] == 201003
    assert timing["lead_months"] == 27


def _frames(psi_values, ratios, ratio_lows, ratio_highs):
    vintages = list(range(2005, 2005 + len(psi_values)))
    psi = pd.DataFrame({
        "vintage": vintages, "n": 1000, "psi_score": psi_values,
        "status": [classify(v, 0.10, 0.25) for v in psi_values],
        "mean_score": 600.0, "reference_mean_score": 600.0, "approval_rate": 0.7,
    })
    disc = pd.DataFrame({
        "vintage": vintages, "gini": 0.45, "gini_low": 0.40, "gini_high": 0.50,
        "degraded": False, "verdict": "no degradation",
    })
    calib = pd.DataFrame({
        "vintage": vintages, "predicted_pd": 0.02, "realised_pd": 0.05,
        "ratio": ratios, "ratio_low": ratio_lows, "ratio_high": ratio_highs,
    })
    return psi, disc, calib


def test_under_prediction_is_a_breach_only_once_its_interval_clears_tolerance():
    psi, disc, calib = _frames([0.01, 0.01], [1.30, 2.00], [0.90, 1.80], [1.70, 2.30])
    frame = bridge(psi, disc, calib, CONFIG)
    # A point ratio of 1.30 is above the 1.25 tolerance but its interval reaches below it.
    assert not bool(frame.loc[0, "calibration_breach"])
    assert bool(frame.loc[1, "calibration_breach"])
    assert frame.loc[1, "calibration_direction"] == "under predicting"


def test_over_prediction_is_a_breach_too():
    """Predicting double the risk that materialises declines good business every day."""
    psi, disc, calib = _frames([0.01], [0.45], [0.35], [0.58])
    frame = bridge(psi, disc, calib, CONFIG)
    assert bool(frame.loc[0, "calibration_breach"])
    assert frame.loc[0, "calibration_direction"] == "over predicting"


def test_the_summary_says_so_when_the_outcome_moved_before_the_early_signal():
    """The finding this project actually produced, pinned so it cannot be quietly reworded."""
    psi, disc, calib = _frames([0.01, 0.01, 0.30], [1.0, 2.0, 2.0], [0.9, 1.8, 1.8], [1.1, 2.3, 2.3])
    frame = bridge(psi, disc, calib, CONFIG)
    summary = first_breach_summary(frame)
    assert summary["first_calibration_breach_vintage"] == 2006
    assert summary["first_stability_breach_vintage"] == 2007
    assert "outcome breached first" in summary["reading"]


def test_the_summary_credits_the_lead_when_the_early_signal_does_move_first():
    psi, disc, calib = _frames([0.30, 0.01, 0.01], [1.0, 1.0, 2.0], [0.9, 0.9, 1.8], [1.1, 1.1, 2.3])
    frame = bridge(psi, disc, calib, CONFIG)
    summary = first_breach_summary(frame)
    assert summary["first_stability_breach_vintage"] == 2005
    assert summary["lead_months_stability_over_calibration"] == 48
    assert "months before the outcome" in summary["reading"]
