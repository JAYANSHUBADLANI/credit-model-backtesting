"""The documented entrypoint. Every number in the README comes from this script.

Order matters here and is worth reading once.

The champion's own fit vintages are represented in the backtest by their **holdout rows only**.
Scoring the loans the card was fitted on and putting that number on the same axis as the out of
time vintages would make the fit period look better than it was, and every later vintage would
then be measured against an inflated baseline. The split is reproduced from the same seed the
training run used, so the holdout is the same holdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import challenger as challenger_module
from src import stability, triggers
from src.backtest import backtest, degradation_verdict, vintage_curves
from src.config import load_config
from src.target import completeness_report, usable
from src.train import load_champion, load_panel, split_fit_period

pd.set_option("display.width", 200)


def build_backtest_panel(panel: pd.DataFrame, config) -> pd.DataFrame:
    """Out of time vintages in full, fit vintages reduced to their holdout rows."""
    usable_panel = usable(panel, config.target)
    train, _ = split_fit_period(usable_panel, config)
    training_ids = set(train["loan_id"])
    keep = ~usable_panel["loan_id"].isin(training_ids)
    return usable_panel.loc[keep].reset_index(drop=True)


def main() -> None:
    config = load_config()
    reports = config.path(config.artifacts.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    champion = load_champion(config)
    panel, target, as_of = load_panel(config)
    summary_path = reports / "champion_summary.json"
    champion_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    day_one = champion_summary["day_one"]

    backtest_panel = build_backtest_panel(panel, config)

    print("=" * 96)
    print(f"Champion {champion.model_version}, frozen on vintages {champion.fit_vintages}, "
          f"{champion.performance_window} month window")
    print(f"Data as of period {as_of}. Day one holdout Gini {day_one['gini']} "
          f"[{day_one['gini_low']}, {day_one['gini_high']}]")
    print("=" * 96)

    # --- completeness -----------------------------------------------------------------
    completeness = completeness_report(target, config.target.performance_window_months)
    completeness.to_csv(reports / "completeness.csv", index=False)
    print("\n## Vintage completeness and realised outcomes\n")
    print(completeness.to_string(index=False))

    # --- discrimination and calibration ------------------------------------------------
    results = backtest(champion, backtest_panel, config)
    discrimination_frame = degradation_verdict(
        results["discrimination"], day_one["gini"], day_one["gini_low"]
    )
    discrimination_frame.to_csv(reports / "discrimination_by_vintage.csv", index=False)
    results["calibration"].to_csv(reports / "calibration_by_vintage.csv", index=False)
    results["calibration_bands"].to_csv(reports / "calibration_by_band.csv", index=False)

    print("\n## Discrimination, against the fit period holdout interval\n")
    print(discrimination_frame[[
        "vintage", "n", "bad_rate", "gini", "gini_low", "gini_high", "gini_change", "verdict"
    ]].to_string(index=False))

    print("\n## Calibration in the large\n")
    calibration = results["calibration"]
    print(calibration[[
        "vintage", "n", "defaults", "predicted_pd", "realised_pd",
        "realised_low", "realised_high", "ratio", "ratio_low"
    ]].to_string(index=False))

    # --- vintage curves ---------------------------------------------------------------
    curves = vintage_curves(target, config.target.performance_window_months)
    curves.to_csv(reports / "vintage_curves.csv", index=False)

    # --- stability, the early signal ---------------------------------------------------
    psi = stability.score_psi_by_vintage(champion, results["scored"], config)
    csi = stability.csi_by_vintage(champion, backtest_panel, config)
    psi.to_csv(reports / "psi_by_vintage.csv", index=False)
    csi.to_csv(reports / "csi_by_vintage.csv", index=False)

    print("\n## Score stability against the frozen fit period reference\n")
    print(psi.to_string(index=False))

    breached = csi.loc[csi["status"] != "ok"].sort_values(["vintage", "csi"], ascending=[True, False])
    print("\n## Characteristics that moved, attribution for the score PSI\n")
    print(breached.to_string(index=False) if len(breached) else "  no characteristic breached")

    # --- the bridge -------------------------------------------------------------------
    bridge = stability.bridge(psi, discrimination_frame, calibration, config)
    bridge.to_csv(reports / "bridge_timeline.csv", index=False)
    first = stability.first_breach_summary(bridge)

    print("\n## The bridge: one timeline, every signal, dated\n")
    print(bridge[[
        "vintage", "psi_score", "status", "stability_known_by", "gini", "degraded",
        "ratio", "calibration_breach", "calibration_direction", "outcome_known_by"
    ]].to_string(index=False))

    print("\n## Which signal moved first\n")
    for key, value in first.items():
        print(f"  {key}: {value}")

    # --- triggers ----------------------------------------------------------------------
    audit = triggers.evaluate(bridge, config)
    audit.to_csv(reports / "trigger_audit.csv", index=False)
    print("\n## Trigger audit, including everything that was suppressed\n")
    print(audit[[
        "vintage", "calibration_breach", "direction", "calibration_run",
        "discrimination_breach", "discrimination_run", "action", "reason", "suppressed"
    ]].to_string(index=False))
    folded = audit.loc[audit["folded"] != ""]
    if len(folded):
        print("\n  folded recommendations:")
        for _, row in folded.iterrows():
            print(f"    {row['vintage']}: {row['folded']}")

    # --- what recalibration does, and does not, do -------------------------------------
    worst = calibration.loc[calibration["ratio"].idxmax()]
    evidence = triggers.recalibration_evidence(results["scored"], int(worst["vintage"]))
    (reports / "recalibration_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(f"\n## Recalibrating the worst vintage ({evidence['vintage']})\n")
    print(f"  realised {evidence['realised_pd']:.4f} against predicted "
          f"{evidence['predicted_pd_before']:.4f}, ratio {evidence['ratio_before']}")
    print(f"  intercept offset {evidence['intercept_offset']} moves predicted to "
          f"{evidence['predicted_pd_after']:.4f}, ratio {evidence['ratio_after']}")
    print(f"  Gini before {evidence['gini_before']}, after {evidence['gini_after']}, "
          f"change {evidence['gini_change']}")
    print("  the ranking is untouched, which is exactly why the two actions are separated")

    # --- challenger --------------------------------------------------------------------
    challenge = challenger_module.run(
        champion, backtest_panel, config, training_panel=usable(panel, config.target)
    )
    challenge["comparison"].to_csv(reports / "challenger_comparison.csv", index=False)
    verdicts = challenger_module.verdict(challenge["comparison"])
    verdicts.to_csv(reports / "challenger_verdict.csv", index=False)

    print("\n## Champion against a feasible and an oracle challenger\n")
    print(verdicts.to_string(index=False))

    # --- swap set on the worst vintage, if a challenger exists there --------------------
    target_vintage = int(worst["vintage"])
    scored_pair = challenge["scored"].get(target_vintage, {})
    if "champion" in scored_pair and "challenger_oracle" in scored_pair:
        swap = triggers.swap_set(scored_pair["champion"], scored_pair["challenger_oracle"])
        swap.to_csv(reports / "swap_set.csv", index=False)
        moved = swap.loc[swap["moved"]]
        print(f"\n## Swap set, champion to oracle challenger, vintage {target_vintage}\n")
        print(swap.to_string(index=False))
        print(f"\n  {moved['loans'].sum():,} of {swap['loans'].sum():,} loans change band "
              f"({moved['share'].sum():.1%})")

    headline = {
        "as_of_period": as_of,
        "champion": champion.model_version,
        "fit_vintages": champion.fit_vintages,
        "day_one": day_one,
        "first_breach": first,
        "actions": audit.loc[audit["action"] != "none", ["vintage", "action", "reason"]]
        .to_dict(orient="records"),
        "recalibration": evidence,
    }
    (reports / "headline.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    print(f"\nreports written to {reports}")


if __name__ == "__main__":
    main()
