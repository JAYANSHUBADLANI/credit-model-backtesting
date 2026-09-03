"""Verdict logic, vintage curves, and one run of the whole pipeline on its own fixture."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.backtest import degradation_verdict, vintage_curves

ROOT = Path(__file__).resolve().parents[1]


def frame(rows):
    return pd.DataFrame(rows)


def test_degradation_needs_the_intervals_to_separate_not_the_points_to_differ():
    """A lower Gini inside the baseline interval is "not proven", never "degraded".

    At the sample sizes a vintage actually has, the bootstrap interval is wide enough to
    swallow a large move. Reporting the point estimate as a finding would manufacture a
    certainty the data cannot support. See scripts/gini_noise.py for the measurement.
    """
    verdicts = degradation_verdict(
        frame([
            {"vintage": 2006, "gini": 0.42, "gini_low": 0.36, "gini_high": 0.48},
            {"vintage": 2007, "gini": 0.20, "gini_low": 0.16, "gini_high": 0.24},
            {"vintage": 2008, "gini": 0.55, "gini_low": 0.50, "gini_high": 0.60},
        ]),
        baseline_gini=0.49, baseline_low=0.38,
    )
    assert verdicts.loc[0, "degraded"] == False  # noqa: E712
    assert "not proven" in verdicts.loc[0, "verdict"]
    assert verdicts.loc[1, "degraded"] == True   # noqa: E712
    assert verdicts.loc[2, "verdict"] == "no degradation"


def test_vintage_curves_are_cumulative_incidence_with_a_fixed_denominator():
    """The denominator is the whole cohort and does not shrink as loans leave.

    Dividing by survivors instead would produce a hazard, a different and steeper curve, and
    silently swapping one for the other is how vintage charts overstate the tail.
    """
    target = frame([
        {"vintage": 2005, "complete": True, "default": 1, "default_age": 6},
        {"vintage": 2005, "complete": True, "default": 1, "default_age": 18},
        {"vintage": 2005, "complete": True, "default": 0, "default_age": None},
        {"vintage": 2005, "complete": True, "default": 0, "default_age": None},
        {"vintage": 2005, "complete": False, "default": 1, "default_age": 3},
    ])
    curves = vintage_curves(target, window=24)
    at = {int(r["month_on_book"]): r for _, r in curves.iterrows()}
    assert at[5]["cumulative_default_rate"] == 0.0
    assert at[6]["cumulative_default_rate"] == 0.25
    assert at[24]["cumulative_default_rate"] == 0.50
    # The incomplete loan is excluded from both sides, not counted as a default.
    assert at[24]["cohort"] == 4


@pytest.mark.slow
def test_the_whole_pipeline_runs_end_to_end_on_a_generated_extract(tmp_path):
    """Fixture in, champion out, backtest and triggers on top. No mocks anywhere.

    The point of this test is that no part of this project is code that has only ever been
    read. It generates an extract, fits a card on it, backtests the card across every vintage
    and evaluates the triggers, in a throwaway tree.
    """
    (tmp_path / "config").mkdir()
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["backtest"]["bootstrap_samples"] = 40
    raw["backtest"]["min_band_count"] = 10
    (tmp_path / "config" / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    generated = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_fixture.py"),
         "--loans-per-vintage", "1500", "--out", str(tmp_path / "data" / "raw")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert generated.returncode == 0, generated.stderr

    from src.config import load_config
    from src.backtest import backtest as run_backtest
    from src.stability import bridge, score_psi_by_vintage
    from src.train import fit_champion, load_panel
    from src.triggers import evaluate

    config = load_config(tmp_path / "config" / "config.yaml", root=tmp_path)
    panel, target, as_of = load_panel(config)
    assert as_of == 201906
    assert target["complete"].sum() > 0

    artifact, summary = fit_champion(config, panel)
    assert summary["characteristics_retained"] >= 3
    assert 0.0 < summary["day_one"]["gini"] < 1.0

    results = run_backtest(artifact, panel, config)
    calibration = results["calibration"].set_index("vintage")
    # The fixture plants a much higher baseline risk in the stress vintages than the
    # characteristics account for, so the frozen card must under predict there and not in the
    # fit period. This asserts the planted signal is recovered, nothing about mortgages.
    assert calibration.loc[2007, "ratio"] > 2.0
    assert calibration.loc[2005, "ratio"] < 2.0

    psi = score_psi_by_vintage(artifact, results["scored"], config)
    from src.backtest import degradation_verdict as verdict
    discrimination = verdict(
        results["discrimination"], summary["day_one"]["gini"], summary["day_one"]["gini_low"]
    )
    timeline = bridge(psi, discrimination, results["calibration"], config)
    audit = evaluate(timeline, config)
    assert (audit["action"] == "recalibrate").any()
