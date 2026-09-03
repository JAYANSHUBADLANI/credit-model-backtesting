"""Discrimination and calibration, both with intervals.

The intervals are the point of this module, not a decoration on it.

A Gini that falls from 0.42 to 0.39 on eight thousand loans has not necessarily fallen at all.
The drift monitoring project this accompanies made its window size defensible by measuring how
much the stability index moves on a population that has not moved, and then setting the
threshold above that floor. The same discipline applies here, and it is the difference between
"discrimination degraded" as a measurement and as an impression. Every discrimination number
reported by this project carries a bootstrap interval, and every claim that discrimination
degraded is a claim that two intervals do not overlap.

Calibration gets a Wilson interval per band for the same reason. A band holding 80 loans and
four defaults has a realised rate of 5 percent with a confidence interval running from about
2 to 12, and calling that a breach of a predicted 3 percent is reading noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Discrimination:
    auc: float
    gini: float
    ks: float
    n: int
    bad_rate: float
    gini_low: Optional[float] = None
    gini_high: Optional[float] = None

    @property
    def has_interval(self) -> bool:
        return self.gini_low is not None and self.gini_high is not None

    def as_row(self) -> dict:
        return {
            "n": self.n,
            "bad_rate": round(self.bad_rate, 5),
            "auc": round(self.auc, 4),
            "gini": round(self.gini, 4),
            "ks": round(self.ks, 4),
            "gini_low": round(self.gini_low, 4) if self.gini_low is not None else None,
            "gini_high": round(self.gini_high, 4) if self.gini_high is not None else None,
        }


def auc_score(target: Sequence[int], prediction: Sequence[float]) -> float:
    """Area under the ROC curve by the rank identity, ties averaged.

    Written out rather than imported so the tie handling is visible: a scorecard puts every
    loan in a bin, so tied scores are the normal case here, not an edge case.
    """
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    if y.size == 0:
        return float("nan")
    positives = y.sum()
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def ks_statistic(target: Sequence[int], prediction: Sequence[float]) -> float:
    y = np.asarray(target, dtype="int64")
    p = np.asarray(prediction, dtype="float64")
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    order = np.argsort(p)
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (len(y_sorted) - y_sorted.sum())
    return float(np.max(np.abs(cum_bad - cum_good)))


def discrimination(
    target: Sequence[int],
    prediction: Sequence[float],
    bootstrap_samples: int = 0,
    confidence: float = 0.90,
    seed: int = 0,
) -> Discrimination:
    """AUC, Gini and KS, with an optional bootstrap interval on the Gini.

    `prediction` is a probability of default, so a higher value means worse. Score is a
    decreasing transform of it, and passing a score here would invert every number, which is
    why callers pass probability and never score.
    """
    y = np.asarray(target, dtype="int64")
    p = np.asarray(prediction, dtype="float64")
    auc = auc_score(y, p)
    result = Discrimination(
        auc=auc,
        gini=2.0 * auc - 1.0 if np.isfinite(auc) else float("nan"),
        ks=ks_statistic(y, p),
        n=int(y.size),
        bad_rate=float(y.mean()) if y.size else float("nan"),
    )
    if bootstrap_samples > 0 and y.size > 0 and 0 < y.sum() < y.size:
        low, high = bootstrap_gini_interval(y, p, bootstrap_samples, confidence, seed)
        result.gini_low, result.gini_high = low, high
    return result


def bootstrap_gini_interval(
    target: Sequence[int],
    prediction: Sequence[float],
    samples: int,
    confidence: float = 0.90,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for the Gini, resampling loans with replacement."""
    y = np.asarray(target, dtype="int64")
    p = np.asarray(prediction, dtype="float64")
    rng = np.random.default_rng(seed)
    n = y.size
    ginis = np.empty(samples, dtype="float64")
    drawn = 0
    for i in range(samples):
        index = rng.integers(0, n, n)
        y_s, p_s = y[index], p[index]
        if y_s.sum() == 0 or y_s.sum() == y_s.size:
            continue
        ginis[drawn] = 2.0 * auc_score(y_s, p_s) - 1.0
        drawn += 1
    if drawn == 0:
        return float("nan"), float("nan")
    ginis = ginis[:drawn]
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(ginis, tail)), float(np.quantile(ginis, 1.0 - tail))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because default rates in the good bands are
    small and the sample in a band can be a few hundred loans, which is exactly where the
    normal interval goes below zero and stops being usable.
    """
    if trials <= 0:
        return float("nan"), float("nan")
    z = _z_for(confidence)
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    margin = (z / denominator) * np.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def _z_for(confidence: float) -> float:
    """Normal quantile for the common confidence levels, without a scipy dependency."""
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}
    if confidence in table:
        return table[confidence]
    # Acklam's rational approximation is overkill here; interpolate the table instead and
    # refuse anything outside it rather than returning a quietly wrong quantile.
    levels = sorted(table)
    if confidence < levels[0] or confidence > levels[-1]:
        raise ValueError(
            f"confidence {confidence} is outside the supported range "
            f"[{levels[0]}, {levels[-1]}]. Add the quantile to the table rather than "
            "letting this extrapolate."
        )
    return float(np.interp(confidence, levels, [table[level] for level in levels]))


def calibration_table(
    target: Sequence[int],
    predicted_pd: Sequence[float],
    band_index: Sequence[int],
    n_bands: int,
    confidence: float = 0.95,
    min_band_count: int = 50,
) -> pd.DataFrame:
    """Predicted against realised default rate, per fixed score band.

    Bands are the fit period score deciles, passed in as indices so that a band means the same
    range of the card in every vintage. Recomputing deciles per vintage would make the bands
    move with the population and hide exactly the shift being looked for.
    """
    y = np.asarray(target, dtype="int64")
    p = np.asarray(predicted_pd, dtype="float64")
    b = np.asarray(band_index, dtype="int64")

    rows = []
    for band in range(n_bands):
        mask = b == band
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "band": band, "n": 0, "predicted_pd": None, "realised_pd": None,
                "realised_low": None, "realised_high": None, "ratio": None,
                "sufficient": False,
            })
            continue
        defaults = int(y[mask].sum())
        predicted = float(p[mask].mean())
        realised = defaults / count
        low, high = wilson_interval(defaults, count, confidence)
        rows.append({
            "band": band,
            "n": count,
            "defaults": defaults,
            "predicted_pd": round(predicted, 5),
            "realised_pd": round(realised, 5),
            "realised_low": round(low, 5),
            "realised_high": round(high, 5),
            "ratio": round(realised / predicted, 4) if predicted > 0 else None,
            # A band below the minimum count is reported but never allowed to drive a trigger.
            "sufficient": count >= min_band_count,
        })
    return pd.DataFrame(rows)


def calibration_in_the_large(
    target: Sequence[int], predicted_pd: Sequence[float], confidence: float = 0.95
) -> dict:
    """The overall level check: realised over predicted, with an interval on the realised side.

    Separate from discrimination on purpose. A card can rank borrowers perfectly and still
    predict two percent where eleven percent happens, and the two failures need different
    responses. That distinction is the whole trigger design in src/triggers.py.
    """
    y = np.asarray(target, dtype="int64")
    p = np.asarray(predicted_pd, dtype="float64")
    n = int(y.size)
    if n == 0:
        return {"n": 0, "predicted_pd": None, "realised_pd": None, "ratio": None}
    defaults = int(y.sum())
    predicted = float(p.mean())
    realised = defaults / n
    low, high = wilson_interval(defaults, n, confidence)
    return {
        "n": n,
        "defaults": defaults,
        "predicted_pd": round(predicted, 5),
        "realised_pd": round(realised, 5),
        "realised_low": round(low, 5),
        "realised_high": round(high, 5),
        "ratio": round(realised / predicted, 4) if predicted > 0 else None,
        "ratio_low": round(low / predicted, 4) if predicted > 0 else None,
        "ratio_high": round(high / predicted, 4) if predicted > 0 else None,
    }
