"""The trigger layer. Restraint is the thing being tested, not detection."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.triggers import (ACTION_NONE, ACTION_RECALIBRATE, ACTION_REFIT, apply_recalibration,
                          evaluate, recalibration_evidence, recalibration_offset, swap_set)

CONFIG = load_config()


def bridge(rows):
    return pd.DataFrame(rows)


def test_a_single_breaching_vintage_fires_nothing():
    audit = evaluate(bridge([
        {"vintage": 2005, "calibration_breach": False, "degraded": False},
        {"vintage": 2006, "calibration_breach": True, "degraded": False,
         "calibration_direction": "under predicting"},
        {"vintage": 2007, "calibration_breach": False, "degraded": False},
    ]), CONFIG)
    assert audit["action"].tolist() == [ACTION_NONE] * 3
    assert "persistence" in audit.loc[1, "suppressed"]


def test_a_persistent_calibration_breach_recommends_recalibration_not_refit():
    audit = evaluate(bridge([
        {"vintage": 2006, "calibration_breach": True, "degraded": False,
         "calibration_direction": "under predicting"},
        {"vintage": 2007, "calibration_breach": True, "degraded": False,
         "calibration_direction": "under predicting"},
    ]), CONFIG)
    assert audit.loc[1, "action"] == ACTION_RECALIBRATE
    assert "level is wrong, not the order" in audit.loc[1, "reason"]


def test_a_persistent_discrimination_breach_recommends_a_refit():
    audit = evaluate(bridge([
        {"vintage": 2006, "calibration_breach": False, "degraded": True},
        {"vintage": 2007, "calibration_breach": False, "degraded": True},
    ]), CONFIG)
    assert audit.loc[1, "action"] == ACTION_REFIT


def test_both_breaching_produces_one_recommendation_not_two():
    """One failure, one recommendation. A refit recalibrates by construction."""
    audit = evaluate(bridge([
        {"vintage": 2006, "calibration_breach": True, "degraded": True,
         "calibration_direction": "under predicting"},
        {"vintage": 2007, "calibration_breach": True, "degraded": True,
         "calibration_direction": "under predicting"},
    ]), CONFIG)
    assert audit.loc[1, "action"] == ACTION_REFIT
    assert "folded into this recommendation" in audit.loc[1, "folded"]


def test_cooldown_suppresses_the_same_recommendation_and_says_so():
    rows = [
        {"vintage": v, "calibration_breach": True, "degraded": False,
         "calibration_direction": "under predicting"}
        for v in range(2006, 2012)
    ]
    audit = evaluate(bridge(rows), CONFIG)
    fired = audit.loc[audit["action"] == ACTION_RECALIBRATE, "vintage"].tolist()
    assert fired == [2007, 2010]
    suppressed = audit.loc[(audit["vintage"] == 2008)]
    assert "cooldown" in suppressed.iloc[0]["suppressed"]


def test_every_suppressed_breach_is_still_recorded():
    """A monitoring system that silently drops signals looks identical to a blind one."""
    audit = evaluate(bridge([
        {"vintage": 2006, "calibration_breach": True, "degraded": False,
         "calibration_direction": "under predicting"},
    ]), CONFIG)
    assert audit.loc[0, "calibration_breach"]
    assert audit.loc[0, "suppressed"] != ""


def test_recalibration_moves_the_level_onto_the_realised_rate():
    rng = np.random.default_rng(5)
    p = np.clip(rng.beta(2, 60, 4000), 1e-6, 1 - 1e-6)
    offset = recalibration_offset(p, 0.12)
    assert apply_recalibration(p, offset).mean() == pytest.approx(0.12, abs=1e-6)


def test_recalibration_cannot_change_the_ranking():
    """The justification for treating recalibration as the cheap action.

    An intercept shift is monotone in the log odds, so no pair of loans can swap order and the
    Gini is unchanged to every decimal place. That is why it needs no revalidation of the
    characteristic set and no new approval of the points table.
    """
    rng = np.random.default_rng(11)
    n = 3000
    p = np.clip(rng.beta(2, 40, n), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < p * 3).astype(int)
    scored = pd.DataFrame({"vintage": 2007, "default": y, "probability": p})
    evidence = recalibration_evidence(scored, 2007)
    assert evidence["gini_change"] == 0.0
    assert evidence["ratio_after"] == pytest.approx(1.0, abs=1e-3)


def test_an_unreachable_level_is_refused_rather_than_clipped():
    p = np.full(100, 0.5)
    with pytest.raises(ValueError, match="not a calibration problem"):
        recalibration_offset(p, 0.0)


def test_swap_set_reports_both_directions_with_their_realised_performance():
    champion = pd.DataFrame({
        "loan_id": ["a", "b", "c", "d"],
        "band": ["approve", "approve", "decline", "refer"],
        "default": [0, 1, 1, 0],
    })
    challenger = pd.DataFrame({
        "loan_id": ["a", "b", "c", "d"],
        "band": ["approve", "decline", "approve", "refer"],
    })
    swap = swap_set(champion, challenger)
    moved = swap.loc[swap["moved"]]
    assert len(moved) == 2
    newly_declined = moved.loc[
        (moved["band_champion"] == "approve") & (moved["band_challenger"] == "decline")
    ]
    assert newly_declined.iloc[0]["default_rate"] == 1.0
