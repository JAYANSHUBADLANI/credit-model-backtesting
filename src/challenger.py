"""Would retraining have helped, and could it have been done at the time.

The second half of that question is the one usually skipped, and it changes the answer.

A challenger available when vintage v originates can only have been fitted on vintages whose
outcomes were already known by then. With a 24 month performance window, the 2006 book's
outcome is not known until the end of 2008. So a challenger sitting in front of the 2007 book
can be fitted on 2004 and nothing later. Fitting it on 2006, which is what "retrain on recent
data" means in practice, gives the backtest two years of information the decision maker did
not have, and turns an honest comparison into a demonstration that hindsight beats foresight.

Two challengers are therefore fitted at every vintage:

- **feasible**, trained only on vintages whose outcomes were knowable at origination. This is
  the one that answers "should we have retrained".
- **oracle**, trained on the immediately preceding vintages regardless of whether their
  outcomes existed yet. Infeasible by construction and labelled as such everywhere it appears.
  It is here to size the gap: the distance between the feasible and the oracle challenger is
  the value of information nobody could have had, and it is not a target anyone can be held to.

The comparison people expect to see is champion against oracle. The comparison that means
anything is champion against feasible.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import score_vintage
from .config import Config
from .metrics import calibration_in_the_large, discrimination
from .scorecard import ScorecardArtifact
from .train import fit_card

FEASIBLE = "feasible"
ORACLE = "oracle"


def outcome_known_by(vintage: int, window_months: int) -> int:
    """Period at which a vintage's full window outcome becomes knowable."""
    end_of_vintage = vintage * 100 + 12
    year, month = divmod(end_of_vintage, 100)
    index = year * 12 + (month - 1) + window_months
    return (index // 12) * 100 + (index % 12) + 1


def available_training_vintages(
    evaluation_vintage: int, all_vintages: List[int], config: Config
) -> List[int]:
    """The most recent vintages whose outcomes were already known at origination time.

    Origination time is taken as January of the evaluation vintage, which is the earliest a
    decision about that book has to be made. Taking December instead would hand the challenger
    up to a year of extra information.
    """
    window = config.target.performance_window_months
    decision_period = evaluation_vintage * 100 + 1
    eligible = [
        v for v in sorted(all_vintages)
        if v < evaluation_vintage and outcome_known_by(v, window) <= decision_period
    ]
    return eligible[-config.challenger.training_window_years:]


def oracle_training_vintages(
    evaluation_vintage: int, all_vintages: List[int], config: Config
) -> List[int]:
    eligible = [v for v in sorted(all_vintages) if v < evaluation_vintage]
    return eligible[-config.challenger.training_window_years:]


def _evaluate(
    artifact: ScorecardArtifact, panel: pd.DataFrame, vintage: int, config: Config
) -> Dict[str, object]:
    scored = score_vintage(artifact, panel.loc[panel["vintage"] == vintage], config)
    y = scored["default"].to_numpy(dtype="int64")
    p = scored["probability"].to_numpy(dtype="float64")
    d = discrimination(
        y, p,
        bootstrap_samples=config.backtest.bootstrap_samples,
        confidence=config.backtest.confidence,
        seed=config.backtest.bootstrap_seed + vintage,
    )
    c = calibration_in_the_large(y, p, config.triggers.calibration_confidence)
    return {"gini": d.gini, "gini_low": d.gini_low, "gini_high": d.gini_high,
            "ratio": c["ratio"], "predicted_pd": c["predicted_pd"],
            "realised_pd": c["realised_pd"], "n": d.n, "scored": scored}


def run(
    champion: ScorecardArtifact,
    panel: pd.DataFrame,
    config: Config,
    training_panel: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Champion against a feasible and an oracle challenger at every backtest vintage.

    `panel` is what everything is evaluated on, with the champion's own training rows already
    removed. `training_panel` is what challengers are fitted from and defaults to the same
    frame. They differ for a reason: a challenger fitted on 2004 has every right to the whole
    of 2004, including the rows the champion happened to be fitted on, because withholding
    them would handicap the challenger for no methodological reason and make the comparison
    a statement about sample size rather than about retraining. No challenger is ever fitted
    on the vintage it is then evaluated on, which is the constraint that actually matters.
    """
    training_panel = panel if training_panel is None else training_panel
    all_vintages = sorted(set(config.vintages.backtest) & set(panel["vintage"].dropna().astype(int)))
    rows: List[dict] = []
    swap_inputs: Dict[int, Dict[str, pd.DataFrame]] = {}

    for vintage in all_vintages:
        evaluation = panel.loc[panel["vintage"] == vintage]
        if evaluation.empty:
            continue
        champion_result = _evaluate(champion, panel, vintage, config)
        rows.append({
            "vintage": vintage, "model": "champion", "feasible": True,
            "trained_on": champion.fit_vintages,
            **{k: v for k, v in champion_result.items() if k != "scored"},
        })
        swap_inputs.setdefault(vintage, {})["champion"] = champion_result["scored"]

        for kind, chooser in ((FEASIBLE, available_training_vintages),
                              (ORACLE, oracle_training_vintages)):
            training = chooser(vintage, all_vintages, config)
            training_rows = training_panel.loc[training_panel["vintage"].isin(training)]
            if not training or len(training_rows) < config.challenger.min_training_rows:
                rows.append({
                    "vintage": vintage, "model": f"challenger_{kind}",
                    "feasible": kind == FEASIBLE, "trained_on": training,
                    "gini": None, "ratio": None, "predicted_pd": None, "realised_pd": None,
                    "n": 0,
                    "note": (
                        "no training vintage had a known outcome by this origination date"
                        if kind == FEASIBLE and not training
                        else "insufficient training rows"
                    ),
                })
                continue
            try:
                artifact, _ = fit_card(
                    config, training_rows, training,
                    f"challenger-{kind}-{'-'.join(str(v) for v in training)}",
                )
            except ValueError as error:
                rows.append({
                    "vintage": vintage, "model": f"challenger_{kind}",
                    "feasible": kind == FEASIBLE, "trained_on": training,
                    "gini": None, "ratio": None, "n": 0, "note": str(error),
                })
                continue
            result = _evaluate(artifact, panel, vintage, config)
            rows.append({
                "vintage": vintage, "model": f"challenger_{kind}",
                "feasible": kind == FEASIBLE, "trained_on": training,
                **{k: v for k, v in result.items() if k != "scored"},
            })
            swap_inputs[vintage][f"challenger_{kind}"] = result["scored"]

    frame = pd.DataFrame(rows)
    for column in ["gini", "gini_low", "gini_high", "ratio", "predicted_pd", "realised_pd"]:
        if column in frame:
            frame[column] = frame[column].astype("float64").round(4)
    return {"comparison": frame, "scored": swap_inputs}


def verdict(comparison: pd.DataFrame) -> pd.DataFrame:
    """What retraining fixed, split into the two failures it could have fixed.

    Reported as two separate deltas because a challenger that halves the calibration error and
    leaves the Gini exactly where it was has not "beaten" the champion in any sense that
    justifies a refit. It has done the job of a recalibration, expensively.
    """
    rows = []
    for vintage, group in comparison.groupby("vintage", sort=True):
        champion = group.loc[group["model"] == "champion"]
        if champion.empty:
            continue
        champion = champion.iloc[0]
        for kind in (FEASIBLE, ORACLE):
            candidate = group.loc[group["model"] == f"challenger_{kind}"]
            if candidate.empty or pd.isna(candidate.iloc[0].get("gini")):
                rows.append({
                    "vintage": int(vintage), "challenger": kind, "available": False,
                    "gini_delta": None, "ratio_champion": champion["ratio"],
                    "ratio_challenger": None, "reading": candidate.iloc[0].get("note")
                    if not candidate.empty else "not fitted",
                })
                continue
            candidate = candidate.iloc[0]
            gini_delta = float(candidate["gini"]) - float(champion["gini"])
            ratio_champion = float(champion["ratio"])
            ratio_candidate = float(candidate["ratio"])
            closer = abs(ratio_candidate - 1.0) < abs(ratio_champion - 1.0)
            rows.append({
                "vintage": int(vintage),
                "challenger": kind,
                "available": True,
                "trained_on": candidate["trained_on"],
                "gini_champion": round(float(champion["gini"]), 4),
                "gini_challenger": round(float(candidate["gini"]), 4),
                "gini_delta": round(gini_delta, 4),
                "ratio_champion": round(ratio_champion, 3),
                "ratio_challenger": round(ratio_candidate, 3),
                "reading": _reading(gini_delta, ratio_champion, ratio_candidate, closer),
            })
    return pd.DataFrame(rows)


def _reading(gini_delta: float, ratio_champion: float, ratio_candidate: float, closer: bool) -> str:
    ranking = (
        "ranking improved" if gini_delta > 0.02
        else "ranking worse" if gini_delta < -0.02
        else "ranking essentially unchanged"
    )
    level = (
        "level closer to right" if closer and abs(ratio_candidate - 1.0) < 0.25
        else "level closer but still off" if closer
        else "level no better"
    )
    return f"{ranking}, {level}"
