"""Deciding what to actually do about a model that has moved, and mostly deciding not to.

The drift monitoring project this accompanies asked to be judged on what its alerting declines
to send. This is the same argument one layer up, and it rests on a single distinction:

**Recalibration and refitting are different actions with different costs, and they answer
different failures.**

- The level is wrong but the ranking is intact -> **recalibrate**. An intercept adjustment.
  The characteristic set is untouched, nothing needs revalidating, the points table a credit
  committee approved still stands, and every cutoff sitting on top of the card keeps its
  meaning.
- The ranking itself has broken -> **refit**. Expensive. It invalidates the cutoffs, the
  policy rules layered on them, and the documentation, and it needs the whole approval cycle
  again.

Reporting "the model is broken" when the honest finding is "the model needs its level moved"
is the same failure as sending sixteen alerts for one drift event. It is more expensive here,
because the response is not a page at 3am, it is a quarter of model risk work and a policy
freeze.

So the rules are:

1. Nothing fires on a single vintage. A breach must persist across consecutive vintages.
2. A calibration breach must clear its own confidence interval before it counts at all. A
   ratio of 1.4 on a band holding eighty loans is not a breach, it is a small sample.
3. Discrimination degradation must clear the measured bootstrap interval, not just be a lower
   number than last year.
4. When both fire together the recommendation is refit alone, never both. A card whose ranking
   has gone will be recalibrated as part of being refitted, and issuing two recommendations for
   one failure is how a model risk log fills up with work nobody does.
5. After an action is recommended, a cooldown suppresses the same recommendation until there
   has been a chance to act on it.

Every suppressed breach is written to the audit trail with the reason it was suppressed. A
monitoring system that silently drops signals is indistinguishable from one that never saw
them.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config

ACTION_NONE = "none"
ACTION_RECALIBRATE = "recalibrate"
ACTION_REFIT = "refit"


def evaluate(bridge_frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Walk the vintages in order and decide what, if anything, to recommend at each."""
    persistence = config.triggers.persistence_vintages
    cooldown = config.triggers.cooldown_vintages

    frame = bridge_frame.sort_values("vintage").reset_index(drop=True)
    calibration_run = 0
    discrimination_run = 0
    cooldown_until: dict[str, Optional[int]] = {ACTION_RECALIBRATE: None, ACTION_REFIT: None}

    rows: List[dict] = []
    for position, record in frame.iterrows():
        vintage = int(record["vintage"])
        calibration_breach = bool(record.get("calibration_breach", False))
        discrimination_breach = bool(record.get("degraded", False))

        calibration_run = calibration_run + 1 if calibration_breach else 0
        discrimination_run = discrimination_run + 1 if discrimination_breach else 0

        action = ACTION_NONE
        reason = "no breach"
        suppressed = ""

        if discrimination_run >= persistence:
            action, reason = ACTION_REFIT, (
                f"discrimination outside the fit period interval for {discrimination_run} "
                "consecutive vintages; the ranking itself has moved"
            )
        elif calibration_run >= persistence:
            direction = str(record.get("calibration_direction") or "mis-predicting")
            action, reason = ACTION_RECALIBRATE, (
                f"{direction} for {calibration_run} consecutive vintages with the ranking "
                "intact; the level is wrong, not the order"
            )
        elif discrimination_breach or calibration_breach:
            breaching = "discrimination" if discrimination_breach else "calibration"
            run = discrimination_run if discrimination_breach else calibration_run
            reason = (
                f"{breaching} breach at this vintage, run of {run} of the {persistence} "
                "consecutive vintages required"
            )
            suppressed = "below the persistence requirement"

        # Rule 4, stated in the docstring: one failure, one recommendation.
        folded = ""
        if action == ACTION_REFIT and calibration_breach:
            folded = (
                "calibration is also breached and is folded into this recommendation rather "
                "than raised separately; a refit recalibrates by construction"
            )

        if action != ACTION_NONE:
            active_until = cooldown_until[action]
            if active_until is not None and vintage <= active_until:
                suppressed = f"cooldown active until vintage {active_until}"
                reason = f"{reason}, suppressed"
                action = ACTION_NONE
            else:
                cooldown_until[action] = vintage + cooldown

        rows.append({
            "vintage": vintage,
            "calibration_breach": calibration_breach,
            "calibration_run": calibration_run,
            "discrimination_breach": discrimination_breach,
            "discrimination_run": discrimination_run,
            "direction": record.get("calibration_direction", ""),
            "ratio": record.get("ratio"),
            "gini": record.get("gini"),
            "psi_score": record.get("psi_score"),
            "action": action,
            "reason": reason,
            "folded": folded,
            "suppressed": suppressed,
            "outcome_known_by": record.get("outcome_known_by"),
        })
    return pd.DataFrame(rows)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def recalibration_offset(predicted: np.ndarray, target_rate: float) -> float:
    """The intercept shift that moves the mean predicted probability onto the realised rate.

    Bisection rather than a closed form because the mean of a logistic is not the logistic of
    the mean, and matching the mean probability is what a level correction is asked to do.
    """
    logits = _logit(predicted)

    def mean_at(offset: float) -> float:
        return float((1.0 / (1.0 + np.exp(-(logits + offset)))).mean())

    low, high = -10.0, 10.0
    if not mean_at(low) <= target_rate <= mean_at(high):
        raise ValueError(
            f"a realised rate of {target_rate:.4f} is not reachable by an intercept shift "
            "within +/-10 log odds. That is not a calibration problem."
        )
    for _ in range(80):
        middle = (low + high) / 2.0
        if mean_at(middle) < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def apply_recalibration(predicted: np.ndarray, offset: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(_logit(predicted) + offset)))


def recalibration_evidence(
    scored: pd.DataFrame, vintage: int
) -> dict:
    """Show what recalibration does and, just as importantly, what it does not do.

    The ordering is untouched by an intercept shift, so the Gini after recalibration is the
    same number to every decimal place. That is not a curiosity, it is the justification for
    treating the two actions separately: recalibration is free of everything that makes a
    refit expensive precisely because it cannot change any decision's ordering.
    """
    from .metrics import auc_score

    group = scored.loc[scored["vintage"] == vintage]
    if group.empty:
        raise ValueError(f"no scored rows for vintage {vintage}")
    y = group["default"].to_numpy(dtype="int64")
    p = group["probability"].to_numpy(dtype="float64")
    realised = float(y.mean())

    offset = recalibration_offset(p, realised)
    p_after = apply_recalibration(p, offset)

    gini_before = 2.0 * auc_score(y, p) - 1.0
    gini_after = 2.0 * auc_score(y, p_after) - 1.0
    return {
        "vintage": int(vintage),
        "n": int(len(group)),
        "realised_pd": round(realised, 5),
        "predicted_pd_before": round(float(p.mean()), 5),
        "predicted_pd_after": round(float(p_after.mean()), 5),
        "ratio_before": round(realised / float(p.mean()), 4),
        "ratio_after": round(realised / float(p_after.mean()), 4),
        "intercept_offset": round(offset, 4),
        "gini_before": round(gini_before, 6),
        "gini_after": round(gini_after, 6),
        "gini_change": round(gini_after - gini_before, 9),
    }


def swap_set(
    champion_scored: pd.DataFrame, challenger_scored: pd.DataFrame
) -> pd.DataFrame:
    """Who changes decision if the recommendation is acted on, and how they actually performed.

    The question a credit committee asks within a minute of being told a model should change,
    and the one a backtest usually cannot answer. Both sides matter: the loans newly declined
    are the benefit, the loans newly approved are the risk taken on, and a swap that improves
    nothing while moving thousands of decisions is not worth the approval cycle.
    """
    merged = champion_scored[["loan_id", "band", "default"]].merge(
        challenger_scored[["loan_id", "band"]],
        on="loan_id", how="inner", suffixes=("_champion", "_challenger"),
    )
    rows = []
    for (before, after), group in merged.groupby(["band_champion", "band_challenger"]):
        rows.append({
            "band_champion": before,
            "band_challenger": after,
            "moved": before != after,
            "loans": len(group),
            "share": round(len(group) / len(merged), 4),
            "default_rate": round(float(group["default"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values(["band_champion", "band_challenger"]).reset_index(drop=True)
