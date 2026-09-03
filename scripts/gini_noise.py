"""How much does the Gini move when nothing has changed.

The direct analogue of `window_size_noise.py` in the drift monitoring project, and it exists
for the same reason. That script measured the stability index on a population that had not
moved and found that at a 100 request window, 34 percent of windows crossed the alert
threshold on sampling noise alone. The threshold was then set above the measured floor rather
than at the conventional value.

The same question here is: at a given vintage size and default rate, how far does the Gini
wander on resamples of one unchanged population? Whatever that spread is, a year on year
change smaller than it is not evidence of anything, and a backtest that reports "Gini fell
from 0.49 to 0.45" without it is reporting noise as a finding.

The answer is uncomfortable and it is supposed to be. Discrimination is estimated from the
defaults, not from the loans, so a vintage of four thousand loans at a two percent default
rate is really a sample of eighty events, and eighty events do not pin a Gini down to two
decimal places. This is the measurement that decides how much of this project's output is
allowed to be stated as fact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import score_vintage
from src.config import load_config
from src.metrics import auc_score, bootstrap_gini_interval
from src.train import load_champion, load_panel, split_fit_period
from src.target import usable

SAMPLE_SIZES = [500, 1000, 2000, 4000, 8000]
REPLICATES = 300


def main() -> None:
    config = load_config()
    champion = load_champion(config)
    panel, _, _ = load_panel(config)
    _, holdout = split_fit_period(usable(panel, config.target), config)
    scored = score_vintage(champion, holdout, config)

    y = scored["default"].to_numpy(dtype="int64")
    p = scored["probability"].to_numpy(dtype="float64")
    full_gini = 2.0 * auc_score(y, p) - 1.0
    rng = np.random.default_rng(config.backtest.bootstrap_seed)

    print(f"fit period holdout: {len(y):,} loans, {int(y.sum())} defaults, "
          f"Gini {full_gini:.4f}\n")
    print(f"{'n':>7} {'defaults':>9} {'p05':>8} {'p50':>8} {'p95':>8} {'spread':>8}")

    rows = []
    for size in SAMPLE_SIZES:
        ginis, event_counts = [], []
        for _ in range(REPLICATES):
            # Drawn with replacement so a sample size above the holdout is still meaningful:
            # the question is what a vintage of that size looks like, not what this particular
            # holdout looks like subsetted.
            index = rng.integers(0, len(y), size)
            y_s, p_s = y[index], p[index]
            if y_s.sum() < 2 or y_s.sum() == y_s.size:
                continue
            ginis.append(2.0 * auc_score(y_s, p_s) - 1.0)
            event_counts.append(int(y_s.sum()))
        if not ginis:
            continue
        ginis = np.array(ginis)
        low, median, high = np.quantile(ginis, [0.05, 0.50, 0.95])
        rows.append({
            "n": size,
            "mean_defaults": round(float(np.mean(event_counts)), 1),
            "gini_p05": round(float(low), 4),
            "gini_p50": round(float(median), 4),
            "gini_p95": round(float(high), 4),
            "spread": round(float(high - low), 4),
        })
        print(f"{size:>7,} {np.mean(event_counts):>9.1f} {low:>8.4f} {median:>8.4f} "
              f"{high:>8.4f} {high - low:>8.4f}")

    frame = pd.DataFrame(rows)
    reports = config.path(config.artifacts.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports / "gini_noise.csv", index=False)

    at_vintage_size = frame.loc[frame["n"] == 4000]
    floor = float(at_vintage_size["spread"].iloc[0]) if len(at_vintage_size) else None
    summary = {
        "holdout_loans": int(len(y)),
        "holdout_defaults": int(y.sum()),
        "holdout_gini": round(float(full_gini), 4),
        "replicates": REPLICATES,
        "noise_floor_at_4000_loans": floor,
        "reading": (
            "a year on year Gini change smaller than the spread at the relevant vintage size "
            "is not evidence of degradation. Every degradation verdict in this project is a "
            "non overlap of bootstrap intervals for this reason, never a comparison of point "
            "estimates."
        ),
    }
    (reports / "gini_noise.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nnoise floor at a 4,000 loan vintage: {floor} Gini points of spread")
    print("that width is the reason verdicts here are interval based, not point based")


if __name__ == "__main__":
    main()
