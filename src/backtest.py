"""Running the frozen champion across every vintage and measuring two different things.

Discrimination and calibration are separated everywhere in this module, deliberately and
consistently, because they fail independently and they need different responses.

- **Discrimination** is whether the card still ranks. A card with intact discrimination puts
  the loans that went bad below the loans that did not, whatever the absolute level.
- **Calibration** is whether the card is still right about the level. A card can rank
  perfectly and predict two percent where eleven percent happens.

A monitoring setup that watches only the Gini can miss a total calibration failure for years,
because the ordering is the more robust of the two properties by some distance. That claim is
the reason this project exists, and the per vintage tables below are what it is tested on.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .features import build_features
from .metrics import calibration_in_the_large, calibration_table, discrimination
from .scorecard import ScorecardArtifact


def score_vintage(artifact: ScorecardArtifact, panel: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Score a slice of the panel with the frozen champion."""
    features = build_features(panel, config.features)
    scored = artifact.score_frame(features)
    out = panel[["loan_id", "vintage", "default", "prepaid"]].reset_index(drop=True)
    out = pd.concat([out, scored.reset_index(drop=True)], axis=1)
    out["score_band"] = np.searchsorted(
        artifact.reference_score_edges, out["score"].to_numpy(), side="right"
    )
    return out


def backtest(
    artifact: ScorecardArtifact,
    panel: pd.DataFrame,
    config: Config,
    vintages: Optional[List[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """Per vintage discrimination, calibration in the large, and calibration by band."""
    vintages = vintages or config.vintages.backtest
    n_bands = config.stability.reference_bins

    discrimination_rows: List[dict] = []
    calibration_rows: List[dict] = []
    band_frames: List[pd.DataFrame] = []
    scored_frames: List[pd.DataFrame] = []

    for vintage in vintages:
        slice_ = panel.loc[panel["vintage"] == vintage]
        if slice_.empty:
            continue
        scored = score_vintage(artifact, slice_, config)
        scored_frames.append(scored)

        y = scored["default"].to_numpy(dtype="int64")
        p = scored["probability"].to_numpy(dtype="float64")

        d = discrimination(
            y, p,
            bootstrap_samples=config.backtest.bootstrap_samples,
            confidence=config.backtest.confidence,
            # Seeded per vintage so a rerun reproduces exactly and two vintages do not share
            # the same resampling pattern.
            seed=config.backtest.bootstrap_seed + int(vintage),
        )
        row = {"vintage": int(vintage), "in_fit_period": vintage in config.vintages.fit}
        row.update(d.as_row())
        discrimination_rows.append(row)

        large = calibration_in_the_large(y, p, config.triggers.calibration_confidence)
        large.update({"vintage": int(vintage), "in_fit_period": vintage in config.vintages.fit})
        calibration_rows.append(large)

        bands = calibration_table(
            y, p, scored["score_band"].to_numpy(), n_bands,
            confidence=config.triggers.calibration_confidence,
            min_band_count=config.backtest.min_band_count,
        )
        bands.insert(0, "vintage", int(vintage))
        band_frames.append(bands)

    return {
        "discrimination": pd.DataFrame(discrimination_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "calibration_bands": pd.concat(band_frames, ignore_index=True) if band_frames else pd.DataFrame(),
        "scored": pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame(),
    }


def vintage_curves(target: pd.DataFrame, window: int) -> pd.DataFrame:
    """Cumulative default rate by months on book, one series per vintage.

    The denominator is every usable loan in the vintage and it does not shrink as loans
    terminate. That makes this a cumulative incidence, which is the quantity a vintage chart is
    normally read as: what share of the cohort had defaulted by month N. Dividing by the
    survivors at each month instead would produce a hazard, a different and steeper curve, and
    mixing the two up is how vintage charts end up overstating the tail.
    """
    usable = target.loc[target["complete"]]
    rows = []
    for vintage, group in usable.groupby("vintage", sort=True):
        denominator = len(group)
        ages = group.loc[group["default"] == 1, "default_age"].dropna().to_numpy()
        for month in range(1, window + 1):
            rows.append({
                "vintage": int(vintage),
                "month_on_book": month,
                "cumulative_defaults": int((ages <= month).sum()),
                "cumulative_default_rate": round(float((ages <= month).sum() / denominator), 5),
                "cohort": denominator,
            })
    return pd.DataFrame(rows)


def degradation_verdict(
    discrimination_frame: pd.DataFrame, baseline_gini: float, baseline_low: float
) -> pd.DataFrame:
    """Has discrimination measurably degraded, against the interval rather than the point.

    A vintage counts as degraded only when the top of its Gini interval sits below the bottom
    of the baseline's. That is a deliberately conservative test and it will call some real
    degradation "not proven" at these sample sizes. Which is the honest answer: with fifty odd
    defaults in a vintage the interval is wide enough to swallow a large move, and reporting
    the point estimate alone would manufacture a certainty the data cannot support.
    """
    frame = discrimination_frame.copy()
    base_point, base_low = float(baseline_gini), float(baseline_low)
    frame["baseline_gini"] = round(base_point, 4)
    frame["gini_change"] = (frame["gini"] - base_point).round(4)
    frame["degraded"] = frame["gini_high"] < base_low
    frame["verdict"] = np.where(
        frame["degraded"],
        "degraded, intervals do not overlap",
        np.where(
            frame["gini"] < base_point,
            "lower point estimate, inside the interval, not proven",
            "no degradation",
        ),
    )
    return frame
