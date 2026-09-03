"""Fit the champion card on the fit vintages, then freeze it.

The champion is never refitted. Everything downstream measures what happens to a frozen model
as the world moves underneath it, which is the only way to observe degradation at all: a model
that is quietly retrained every quarter never visibly degrades, it just silently becomes a
different model, and nobody can say when the old one stopped working.

The fit period is deliberately benign. That is hindsight and the README says so. What it buys
is a clean day one number to measure every later vintage against.

What is frozen alongside the coefficients, and why each is needed later:

- the weight of evidence bins, so that a characteristic stability index in 2007 counts the
  same bins the card was fitted with
- the band cutoffs, resolved from fit period percentiles into absolute scores, so a band means
  the same range of the card in every vintage
- the score deciles, which are both the drift reference and the fixed calibration bands
- the fit period mean predicted probability, which is the level every later realised rate is
  compared against
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from .binning import WOETransformer, BinningConfig as BinnerConfig
from .config import Config
from .features import build_features
from .loans import load_origination, load_performance, data_as_of
from .metrics import discrimination
from .scorecard import BandCutoffs, Scorecard, ScorecardArtifact, ScalingConfig, select_features
from .target import assert_window_consistent, build_target, completeness_report, usable


def load_panel(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Origination joined to its constructed target, plus the data as of period."""
    origination = load_origination(config.root, config.data.origination_glob, config.data.delimiter)
    performance = load_performance(config.root, config.data.performance_glob, config.data.delimiter)
    as_of = data_as_of(performance)
    target = build_target(performance, origination, config.target, as_of=as_of)
    assert_window_consistent(target, config.target.performance_window_months)
    panel = origination.merge(
        target.drop(columns=["vintage", "orig_date"]), on="loan_id", how="inner"
    )
    return panel, target, as_of


def split_fit_period(panel: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random holdout inside the fit period.

    Random rather than time ordered on purpose. The out of time question is what every later
    vintage answers; this split exists only to give the fit period itself an honest number, and
    ordering it by month inside a two year window would just be a smaller version of the same
    out of time test with a worse sample size.
    """
    fit_rows = panel.loc[panel["vintage"].isin(config.vintages.fit)].reset_index(drop=True)
    if fit_rows.empty:
        raise ValueError(f"no usable loans in the fit vintages {config.vintages.fit}")
    rng = np.random.default_rng(config.vintages.seed)
    holdout_mask = rng.random(len(fit_rows)) < config.vintages.holdout_fraction
    return fit_rows.loc[~holdout_mask].reset_index(drop=True), fit_rows.loc[holdout_mask].reset_index(drop=True)


def fit_card(
    config: Config, train: pd.DataFrame, fit_vintages: list, version: str
) -> Tuple[ScorecardArtifact, list]:
    """Fit one frozen card on the rows given. Used for the champion and for every challenger.

    Shared rather than duplicated on purpose: a challenger fitted by a slightly different
    procedure than the champion is not a challenger, it is a confound, and any difference
    between the two would be unattributable.
    """
    train_features = build_features(train, config.features)
    y_train = train["default"].to_numpy(dtype="int64")

    transformer = WOETransformer(BinnerConfig(**vars(config.binning)))
    transformer.fit(
        train_features, y_train,
        numeric=config.features.numeric, categorical=config.features.categorical,
    )
    woe_train = transformer.transform(train_features)
    selected = select_features(
        transformer, woe_train,
        min_iv=config.selection.min_iv, max_correlation=config.selection.max_correlation,
    )
    if not selected:
        raise ValueError(
            "no characteristic cleared the information value floor. Either the floor is set "
            "too high for this extract or the target is not what it is supposed to be."
        )

    card = Scorecard(ScalingConfig(**vars(config.scaling))).fit(woe_train, y_train, selected)

    train_scores = card.score(woe_train)
    decline = float(np.percentile(train_scores, config.bands.decline_below_percentile))
    refer = float(np.percentile(train_scores, config.bands.refer_below_percentile))
    bands = BandCutoffs(decline_below=decline, refer_below=refer)

    # Deciles of the fit period score distribution. These are both the drift reference and the
    # fixed calibration bands, and they are frozen here so no later vintage can recompute them
    # against its own population and hide the shift being looked for.
    quantiles = np.linspace(0, 100, config.stability.reference_bins + 1)[1:-1]
    score_edges = np.percentile(train_scores, quantiles)
    score_bins = np.searchsorted(score_edges, train_scores, side="right")
    score_proportions = np.bincount(
        score_bins, minlength=config.stability.reference_bins
    ).astype("float64")
    score_proportions /= score_proportions.sum()

    band_assignment = bands.assign_many(train_scores)
    band_proportions = np.array([
        float((band_assignment == name).mean()) for name in ["decline", "refer", "approve"]
    ])

    artifact = ScorecardArtifact(
        transformer=transformer,
        scorecard=card,
        bands=bands,
        numeric_features=[f for f in config.features.numeric if f in selected],
        categorical_features=[f for f in config.features.categorical if f in selected],
        categorical_levels={
            name: list(binning.level_to_index)
            for name, binning in transformer.bins.items()
            if binning.kind == "categorical"
        },
        reference_score_edges=score_edges,
        reference_score_proportions=score_proportions,
        reference_band_proportions=band_proportions,
        reference_mean_score=float(train_scores.mean()),
        reference_mean_probability=float(card.predict_proba(woe_train).mean()),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_rows=int(len(train)),
        training_bad_rate=float(y_train.mean()),
        model_version=version,
        fit_vintages=list(fit_vintages),
        performance_window=config.target.performance_window_months,
    )
    return artifact, selected


def fit_champion(config: Config, panel: pd.DataFrame) -> Tuple[ScorecardArtifact, dict]:
    usable_panel = usable(panel, config.target)
    train, holdout = split_fit_period(usable_panel, config)
    y_train = train["default"].to_numpy(dtype="int64")
    y_holdout = holdout["default"].to_numpy(dtype="int64")

    artifact, selected = fit_card(
        config, train, config.vintages.fit,
        f"champion-{'-'.join(str(v) for v in config.vintages.fit)}",
    )
    transformer, card = artifact.transformer, artifact.scorecard
    decline, refer = artifact.bands.decline_below, artifact.bands.refer_below

    holdout_features = build_features(holdout, config.features)
    woe_holdout = transformer.transform(holdout_features)
    holdout_probability = card.predict_proba(woe_holdout)
    day_one = discrimination(
        y_holdout, holdout_probability,
        bootstrap_samples=config.backtest.bootstrap_samples,
        confidence=config.backtest.confidence,
        seed=config.backtest.bootstrap_seed,
    )

    summary = {
        "fit_vintages": list(config.vintages.fit),
        "performance_window_months": config.target.performance_window_months,
        "train_rows": int(len(train)),
        "train_bad_rate": round(float(y_train.mean()), 5),
        "holdout_rows": int(len(holdout)),
        "holdout_bad_rate": round(float(y_holdout.mean()), 5),
        "characteristics_considered": len(config.features.all_features),
        "characteristics_retained": len(selected),
        "retained": selected,
        "day_one": day_one.as_row(),
        "band_cutoffs": {"decline_below": round(decline, 2), "refer_below": round(refer, 2)},
    }
    return artifact, summary


def save(artifact: ScorecardArtifact, summary: dict, config: Config) -> None:
    model_path = config.path(config.artifacts.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    reference = {
        "model_version": artifact.model_version,
        "trained_at": artifact.trained_at,
        "fit_vintages": artifact.fit_vintages,
        "performance_window": artifact.performance_window,
        "score_edges": [round(float(e), 4) for e in artifact.reference_score_edges],
        "score_proportions": [round(float(p), 6) for p in artifact.reference_score_proportions],
        "band_proportions": [round(float(p), 6) for p in artifact.reference_band_proportions],
        "mean_score": round(artifact.reference_mean_score, 4),
        "mean_probability": round(artifact.reference_mean_probability, 6),
        "training_bad_rate": round(artifact.training_bad_rate, 6),
        "bin_reference": {
            name: [round(float(p), 6) for p in binning.reference_proportions]
            for name, binning in artifact.transformer.bins.items()
        },
        "summary": summary,
    }
    reference_path = config.path(config.artifacts.reference_path)
    reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")


def load_champion(config: Config) -> ScorecardArtifact:
    path = config.path(config.artifacts.model_path)
    if not path.exists():
        raise FileNotFoundError(f"no champion at {path}. Run `make train` first.")
    return joblib.load(path)


def main() -> None:
    from .config import load_config

    config = load_config()
    panel, target, as_of = load_panel(config)
    reports = config.path(config.artifacts.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    completeness = completeness_report(target, config.target.performance_window_months)
    completeness.to_csv(reports / "completeness.csv", index=False)
    print(f"data as of period {as_of}")
    print(completeness.to_string(index=False))

    artifact, summary = fit_champion(config, panel)
    save(artifact, summary, config)

    print(f"\nchampion fitted on vintages {summary['fit_vintages']}")
    print(f"  {summary['characteristics_retained']} of {summary['characteristics_considered']} "
          f"characteristics retained: {', '.join(summary['retained'])}")
    d = summary["day_one"]
    print(f"  day one holdout, {d['n']:,} loans at {d['bad_rate']:.2%} bad rate")
    print(f"  Gini {d['gini']:.4f} [{d['gini_low']:.4f}, {d['gini_high']:.4f}]  "
          f"KS {d['ks']:.4f}  AUC {d['auc']:.4f}")
    (reports / "champion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
