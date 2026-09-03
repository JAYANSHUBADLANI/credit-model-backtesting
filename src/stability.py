"""Input and output stability, and the bridge to the outcome measurements.

This is the same population stability index the drift monitoring project computes, run here
against origination vintages instead of scoring windows, and against the same kind of frozen
fit period reference.

The reason it is in this repository at all is the arithmetic in `signal_timing`.

A stability index is computable the moment a vintage originates, because it needs nothing but
the applications themselves. A realised default rate is not computable until the performance
window has run, which here is two years later. So the two signals are not competing answers to
the same question, they are the same question asked twenty four months apart:

    stability index on the 2007 book   ->  knowable during 2007
    realised 24 month default rate     ->  knowable during 2009

That gap is the entire operational case for input drift monitoring, and it is also its
limitation. The stability index is available early and cannot tell you whether anything got
worse. The outcome is definitive and arrives after the book is already written. Neither
replaces the other, and this module is what lets the README say so with a number attached
rather than as an assertion.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .features import build_features
from .loans import period_add
from .scorecard import ScorecardArtifact

EPSILON = 1e-4

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ALERT = "alert"


def stability_index(actual: np.ndarray, reference: np.ndarray, epsilon: float = EPSILON) -> float:
    """Population stability index between two discrete distributions.

    Both sides are floored before the log. A bin empty on one side is a real and common case,
    and without the floor it returns an infinity that then propagates into every comparison
    downstream. Flooring caps an empty bin's contribution at a large but finite number.
    """
    actual = np.asarray(actual, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    if actual.shape != reference.shape:
        raise ValueError(
            f"distributions must have the same number of bins, got {actual.shape} and "
            f"{reference.shape}"
        )
    if actual.size == 0:
        return 0.0
    actual = np.maximum(actual, epsilon)
    reference = np.maximum(reference, epsilon)
    return float(np.sum((actual - reference) * np.log(actual / reference)))


def classify(value: float, warn: float, alert: float) -> str:
    if value >= alert:
        return STATUS_ALERT
    if value >= warn:
        return STATUS_WARN
    return STATUS_OK


def _proportions(indices: np.ndarray, n_bins: int) -> np.ndarray:
    counts = np.bincount(np.asarray(indices, dtype="int64"), minlength=n_bins).astype("float64")
    counts = counts[:n_bins]
    total = counts.sum()
    return counts / total if total > 0 else counts


def score_psi_by_vintage(
    artifact: ScorecardArtifact, scored: pd.DataFrame, config: Config
) -> pd.DataFrame:
    """Score distribution PSI per vintage against the frozen fit period deciles."""
    n_bins = config.stability.reference_bins
    reference = np.asarray(artifact.reference_score_proportions, dtype="float64")
    rows = []
    for vintage, group in scored.groupby("vintage", sort=True):
        actual = _proportions(group["score_band"].to_numpy(), n_bins)
        value = stability_index(actual, reference)
        rows.append({
            "vintage": int(vintage),
            "n": len(group),
            "psi_score": round(value, 4),
            "status": classify(value, config.stability.psi_warn, config.stability.psi_alert),
            "mean_score": round(float(group["score"].mean()), 2),
            "reference_mean_score": round(artifact.reference_mean_score, 2),
            "approval_rate": round(float((group["band"] == "approve").mean()), 4),
        })
    return pd.DataFrame(rows)


def csi_by_vintage(
    artifact: ScorecardArtifact, panel: pd.DataFrame, config: Config
) -> pd.DataFrame:
    """Characteristic stability index per retained characteristic per vintage.

    Only the characteristics the card actually retained are reported. A dropped characteristic
    can move as much as it likes without affecting a single score, and including it would put
    breaches in the attribution table that no reviewer can act on.
    """
    retained = set(artifact.scorecard.features)
    rows = []
    for vintage, group in panel.groupby("vintage", sort=True):
        features = build_features(group, config.features)
        indices = artifact.transformer.bin_indices(features)
        for name, binning in artifact.transformer.bins.items():
            if name not in retained:
                continue
            actual = _proportions(indices[name].to_numpy(), binning.n_bins)
            value = stability_index(actual, np.asarray(binning.reference_proportions))
            rows.append({
                "vintage": int(vintage),
                "characteristic": name,
                "csi": round(value, 4),
                "status": classify(value, config.stability.csi_warn, config.stability.csi_alert),
            })
    return pd.DataFrame(rows)


def signal_timing(vintage: int, window_months: int, reporting_lag_months: int = 0) -> Dict[str, int]:
    """When each signal about a vintage could first have been known.

    The stability index is dated to the end of the origination year, which is when the whole
    vintage has been seen. The outcome is dated to that plus the performance window plus any
    reporting lag. The difference is what input monitoring buys, expressed in months rather
    than in adjectives.
    """
    vintage_complete = vintage * 100 + 12
    outcome_known = period_add(vintage_complete, window_months + reporting_lag_months)
    return {
        "vintage": int(vintage),
        "stability_known_by": int(vintage_complete),
        "outcome_known_by": int(outcome_known),
        "lead_months": int(window_months + reporting_lag_months),
    }


def bridge(
    psi_frame: pd.DataFrame,
    discrimination_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    config: Config,
    reporting_lag_months: int = 0,
) -> pd.DataFrame:
    """One row per vintage carrying every signal, on one timeline.

    This is the table the whole pair of projects exists to produce. It puts the early,
    outcome free signal next to the late, definitive ones and dates all of them.
    """
    window = config.target.performance_window_months
    tolerance = config.triggers.calibration_ratio_tolerance

    merged = psi_frame.merge(
        discrimination_frame[["vintage", "gini", "gini_low", "gini_high", "degraded", "verdict"]],
        on="vintage", how="outer",
    ).merge(
        calibration_frame[[
            "vintage", "predicted_pd", "realised_pd", "ratio", "ratio_low", "ratio_high"
        ]],
        on="vintage", how="outer",
    ).sort_values("vintage").reset_index(drop=True)

    timing = pd.DataFrame([
        signal_timing(int(v), window, reporting_lag_months) for v in merged["vintage"]
    ])
    merged = merged.merge(timing, on="vintage", how="left")

    # A calibration breach has to clear its own confidence interval, not just the point ratio,
    # and it is two sided. Under prediction is the dangerous direction, because it means losses
    # arriving that were not provisioned for. Over prediction is the expensive one: a card that
    # predicts double the risk that materialises declines profitable business every day it is
    # left alone, and calling that "conservative" rather than "wrong" is how it survives for
    # years. Both are the level being wrong, both are fixed by the same cheap action.
    merged["calibration_breach_under"] = (
        merged["ratio_low"].notna() & (merged["ratio_low"] > tolerance)
    )
    merged["calibration_breach_over"] = (
        merged["ratio_high"].notna() & (merged["ratio_high"] < 1.0 / tolerance)
    )
    merged["calibration_breach"] = (
        merged["calibration_breach_under"] | merged["calibration_breach_over"]
    )
    merged["calibration_direction"] = np.where(
        merged["calibration_breach_under"], "under predicting",
        np.where(merged["calibration_breach_over"], "over predicting", ""),
    )
    merged["stability_breach"] = merged["status"].isin([STATUS_WARN, STATUS_ALERT])
    return merged


def first_breach_summary(bridge_frame: pd.DataFrame) -> Dict[str, object]:
    """The headline: which signal moved first, and how much earlier it was knowable."""
    def _first(column: str) -> pd.Series | None:
        hits = bridge_frame.loc[bridge_frame[column].fillna(False).astype(bool)]
        return hits.iloc[0] if len(hits) else None

    stability = _first("stability_breach")
    degraded = _first("degraded")
    calibration = _first("calibration_breach")

    summary: Dict[str, object] = {
        "first_stability_breach_vintage": int(stability["vintage"]) if stability is not None else None,
        "first_stability_known_by": int(stability["stability_known_by"]) if stability is not None else None,
        "first_calibration_breach_vintage": int(calibration["vintage"]) if calibration is not None else None,
        "first_calibration_known_by": int(calibration["outcome_known_by"]) if calibration is not None else None,
        "first_discrimination_breach_vintage": int(degraded["vintage"]) if degraded is not None else None,
        "first_discrimination_known_by": int(degraded["outcome_known_by"]) if degraded is not None else None,
    }

    if stability is not None and calibration is not None:
        early = int(stability["stability_known_by"])
        late = int(calibration["outcome_known_by"])
        lead = _months_between(early, late)
        summary["lead_months_stability_over_calibration"] = lead
        summary["same_vintage"] = int(stability["vintage"]) == int(calibration["vintage"])
        # The arithmetic guarantees a positive lead only if the stability index breaches on the
        # same vintage or earlier. It is not guaranteed to breach at all, and on a failure of
        # level rather than of composition it will not. Say which happened rather than
        # reporting a lead time that assumes the answer.
        if int(stability["vintage"]) < int(calibration["vintage"]):
            summary["reading"] = (
                f"the stability index breached on an earlier vintage and was knowable {lead} "
                "months before the outcome confirmed it"
            )
        elif int(stability["vintage"]) == int(calibration["vintage"]):
            summary["reading"] = (
                f"both signals point at the same vintage, but the stability index was knowable "
                f"{lead} months earlier"
            )
        else:
            summary["reading"] = (
                "the outcome breached first. The stability index did not move until a later "
                "vintage, so on this book it bought no warning at all"
            )
    elif calibration is not None and stability is None:
        summary["reading"] = (
            "the outcome breached and the stability index never did. Input monitoring was "
            "silent through the entire failure"
        )
    return summary


def _months_between(earlier: int, later: int) -> int:
    ey, em = divmod(int(earlier), 100)
    ly, lm = divmod(int(later), 100)
    return (ly * 12 + lm) - (ey * 12 + em)
