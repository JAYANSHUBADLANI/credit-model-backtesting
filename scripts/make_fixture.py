"""Generate a synthetic loan level extract in the GSE file layout.

Why this exists, stated as plainly as possible so nothing downstream is read as more than it
is: the real Freddie Mac and Fannie Mae loan level datasets are free but sit behind a
registration, and cannot be fetched by an automated build. Without data, the whole pipeline
would be code that had never once been run, which is the exact failure this author has already
made and written up in another repository. So the pipeline is exercised end to end against a
generator that writes the same pipe delimited layout the real extracts use.

**Every finding produced from this fixture is a finding about the pipeline, not about
mortgages.** The degradation below was put there deliberately, by the parameters in this file,
and the backtest recovering it is evidence that the measurement works and the triggers fire.
It is not evidence about the 2007 vintage. Swap in the real extract and the same commands
produce numbers that mean something about the world.

What is planted, and why each piece:

1. **A stable ranking relationship.** The coefficient vector below is the same in every
   vintage, apart from one term. This is what makes discrimination survive.
2. **A moving intercept.** The stress vintages carry a much higher baseline default rate than
   the characteristics account for. This is what breaks calibration while the ranking holds,
   and it is the finding the whole project is built to catch.
3. **A moving population.** Credit scores fall, loan to values rise, piggyback seconds
   appear, cash out and investor shares rise, broker origination rises, and the concentrated
   states take a larger share. This is what the stability index can see at origination, two
   years before any outcome is known.
4. **One coefficient that does move.** The concentrated states carry extra risk in the stress
   vintages that they did not carry in the fit period. A frozen card cannot know this, and it
   is what puts a modest, real dent in discrimination on top of the large calibration break.
5. **Right truncation.** Performance stops at a data as of date, so the newest vintage is
   partly unfinished, exactly as a real extract is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Data as of. Performance rows after this period do not exist yet, which leaves the newest
# vintage partly right truncated the way a real extract always is.
AS_OF_PERIOD = 201906

STATES_HOT = ["CA", "FL", "AZ", "NV"]
STATES_COLD = ["TX", "NY", "IL", "OH", "PA", "MI", "GA", "NC", "WA", "CO"]

# vintage -> (intercept, mean fico, mean ltv, piggyback rate, cashout share, investor share,
#             broker share, hot state share, prepay rate, hot state extra risk)
VINTAGE_PARAMS = {
    2004: dict(intercept=-4.45, fico=724, ltv=73.0, piggyback=0.08, cashout=0.22,
               investor=0.06, broker=0.18, hot_share=0.26, prepay=0.30, hot_extra=0.00),
    2005: dict(intercept=-4.35, fico=722, ltv=74.5, piggyback=0.13, cashout=0.26,
               investor=0.08, broker=0.22, hot_share=0.30, prepay=0.28, hot_extra=0.00),
    2006: dict(intercept=-4.00, fico=716, ltv=76.5, piggyback=0.20, cashout=0.31,
               investor=0.11, broker=0.28, hot_share=0.34, prepay=0.20, hot_extra=0.30),
    2007: dict(intercept=-3.30, fico=712, ltv=78.0, piggyback=0.23, cashout=0.33,
               investor=0.12, broker=0.31, hot_share=0.35, prepay=0.15, hot_extra=0.60),
    2008: dict(intercept=-3.40, fico=725, ltv=76.0, piggyback=0.10, cashout=0.24,
               investor=0.08, broker=0.20, hot_share=0.31, prepay=0.18, hot_extra=0.55),
    2015: dict(intercept=-4.60, fico=744, ltv=74.0, piggyback=0.05, cashout=0.18,
               investor=0.07, broker=0.14, hot_share=0.27, prepay=0.28, hot_extra=0.00),
    2016: dict(intercept=-4.70, fico=748, ltv=73.0, piggyback=0.04, cashout=0.17,
               investor=0.07, broker=0.13, hot_share=0.26, prepay=0.30, hot_extra=0.00),
    2017: dict(intercept=-4.55, fico=745, ltv=74.0, piggyback=0.05, cashout=0.19,
               investor=0.08, broker=0.15, hot_share=0.27, prepay=0.25, hot_extra=0.00),
}

# The ranking relationship. Identical in every vintage except `hot_extra`, which is what
# makes discrimination decay a real and separate phenomenon from the calibration break.
COEFFICIENTS = dict(
    fico=0.90, ltv=0.35, gap=0.30, dti=0.25,
    purpose_C=0.35, purpose_N=0.10,
    occupancy_I=0.45, occupancy_S=0.15,
    channel_B=0.25, channel_C=0.12,
    property_MH=0.50, property_CO=0.15, property_PU=0.05,
    ftb=0.12,
)

MISSING_FICO_RATE = 0.015
MISSING_DTI_RATE = 0.010
CREDIT_LOSS_TERMINATION_RATE = 0.40   # share of defaults that also get a disposition code
SILENT_LOSS_RATE = 0.06               # defaults visible only as a termination code, never D180
UNREPORTED_STATUS_RATE = 0.004        # servicer reported "XX" for that month


def _period_add(period: int, months: int) -> int:
    year, month = divmod(int(period), 100)
    index = year * 12 + (month - 1) + int(months)
    return (index // 12) * 100 + (index % 12) + 1


def _originations(vintage: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    p = VINTAGE_PARAMS[vintage]

    fico = np.clip(rng.normal(p["fico"], 46, n), 500, 830).round(0)
    ltv = np.clip(rng.normal(p["ltv"], 12, n), 20, 97).round(0)
    piggyback = rng.random(n) < p["piggyback"]
    gap = np.where(piggyback, np.clip(rng.normal(15, 5, n), 1, 25).round(0), 0.0)
    cltv = np.clip(ltv + gap, 20, 120)
    dti = np.clip(rng.normal(35, 10, n), 5, 64).round(0)
    upb = np.clip(rng.lognormal(12.05, 0.45, n), 20000, 900000).round(-3)
    rate = np.clip(rng.normal(6.1 if vintage < 2010 else 4.0, 0.55, n), 2.0, 11.0).round(3)
    term = np.full(n, 360)
    mi = np.where(ltv > 80, np.clip(rng.normal(25, 4, n), 6, 40).round(0), 0.0)
    borrowers = rng.choice([1, 2], n, p=[0.42, 0.58])
    units = rng.choice([1, 2, 3, 4], n, p=[0.94, 0.04, 0.012, 0.008])

    purpose = rng.choice(
        ["P", "C", "N"], n,
        p=[1 - p["cashout"] - 0.30, p["cashout"], 0.30],
    )
    occupancy = rng.choice(
        ["P", "S", "I"], n, p=[1 - p["investor"] - 0.05, 0.05, p["investor"]]
    )
    channel = rng.choice(
        ["R", "B", "C"], n, p=[1 - p["broker"] - 0.25, p["broker"], 0.25]
    )
    property_type = rng.choice(["SF", "CO", "PU", "MH"], n, p=[0.72, 0.11, 0.155, 0.015])
    ftb = rng.choice(["Y", "N"], n, p=[0.16, 0.84])

    hot = rng.random(n) < p["hot_share"]
    state = np.where(
        hot,
        rng.choice(STATES_HOT, n),
        rng.choice(STATES_COLD, n),
    )

    month = rng.integers(1, 13, n)
    orig_date = vintage * 100 + month

    loan_id = np.array([f"F{vintage}{i:07d}" for i in range(n)])

    return pd.DataFrame({
        "loan_id": loan_id, "orig_date": orig_date, "credit_score": fico,
        "orig_ltv": ltv, "orig_cltv": cltv, "dti": dti, "orig_upb": upb,
        "orig_rate": rate, "orig_term": term, "purpose": purpose,
        "occupancy": occupancy, "channel": channel, "property_type": property_type,
        "num_units": units, "num_borrowers": borrowers, "first_time_buyer": ftb,
        "mi_pct": mi, "state": state, "_hot": hot, "_gap": gap,
    })


def _default_probability(frame: pd.DataFrame, vintage: int) -> np.ndarray:
    p = VINTAGE_PARAMS[vintage]
    c = COEFFICIENTS
    lo = np.full(len(frame), p["intercept"], dtype="float64")
    lo += c["fico"] * (720.0 - frame["credit_score"].to_numpy()) / 50.0
    lo += c["ltv"] * (frame["orig_ltv"].to_numpy() - 75.0) / 10.0
    lo += c["gap"] * frame["_gap"].to_numpy() / 10.0
    lo += c["dti"] * (frame["dti"].to_numpy() - 35.0) / 10.0
    lo += c["purpose_C"] * (frame["purpose"] == "C").to_numpy()
    lo += c["purpose_N"] * (frame["purpose"] == "N").to_numpy()
    lo += c["occupancy_I"] * (frame["occupancy"] == "I").to_numpy()
    lo += c["occupancy_S"] * (frame["occupancy"] == "S").to_numpy()
    lo += c["channel_B"] * (frame["channel"] == "B").to_numpy()
    lo += c["channel_C"] * (frame["channel"] == "C").to_numpy()
    lo += c["property_MH"] * (frame["property_type"] == "MH").to_numpy()
    lo += c["property_CO"] * (frame["property_type"] == "CO").to_numpy()
    lo += c["property_PU"] * (frame["property_type"] == "PU").to_numpy()
    lo += c["ftb"] * (frame["first_time_buyer"] == "Y").to_numpy()
    # The one term that moves between vintages.
    lo += p["hot_extra"] * frame["_hot"].to_numpy()
    return 1.0 / (1.0 + np.exp(-lo))


def _performance(
    origination: pd.DataFrame, vintage: int, window: int, rng: np.random.Generator
) -> pd.DataFrame:
    p = VINTAGE_PARAMS[vintage]
    probability = _default_probability(origination, vintage)
    will_default = rng.random(len(origination)) < probability
    will_prepay = (~will_default) & (rng.random(len(origination)) < p["prepay"])

    default_month = rng.integers(8, window + 1, len(origination))
    prepay_month = rng.integers(4, window + 1, len(origination))
    silent = rng.random(len(origination)) < SILENT_LOSS_RATE
    disposed = rng.random(len(origination)) < CREDIT_LOSS_TERMINATION_RATE

    rows = []
    ids = origination["loan_id"].to_numpy()
    orig_dates = origination["orig_date"].to_numpy()

    for i in range(len(origination)):
        loan_id, orig_date = ids[i], int(orig_dates[i])
        if will_default[i]:
            end = int(default_month[i])
            terminal = None
            if silent[i]:
                # Terminated through a credit loss without ever being reported at 180 days.
                # A target built only on delinquency status would miss this loan entirely.
                terminal = rng.choice(["02", "15"])
            elif disposed[i]:
                terminal = rng.choice(["03", "09"])
        elif will_prepay[i]:
            end = int(prepay_month[i])
            terminal = "01"
        else:
            end = window
            terminal = None

        for age in range(1, end + 1):
            period = _period_add(orig_date, age)
            if period > AS_OF_PERIOD:
                break
            if will_default[i] and not silent[i]:
                # Delinquency escalates for six months into the default month.
                months_late = max(0, age - (end - 6))
                status = str(min(months_late, 6))
            elif will_default[i] and silent[i]:
                months_late = max(0, age - (end - 3))
                status = str(min(months_late, 3))
            else:
                status = "0"
            if rng.random() < UNREPORTED_STATUS_RATE:
                status = "XX"
            code = terminal if (age == end and terminal) else ""
            rows.append((loan_id, period, age, status, code))

    return pd.DataFrame(rows, columns=["loan_id", "period", "loan_age", "dlq_status", "zb_code"])


def _apply_missing(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    fico_missing = rng.random(len(out)) < MISSING_FICO_RATE
    dti_missing = rng.random(len(out)) < MISSING_DTI_RATE
    out.loc[fico_missing, "credit_score"] = 9999
    out.loc[dti_missing, "dti"] = 999
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loans-per-vintage", type=int, default=4000)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("origination_*.txt")) + list(out.glob("performance_*.txt")):
        stale.unlink()

    rng = np.random.default_rng(args.seed)
    total_loans = total_rows = 0

    for vintage in sorted(VINTAGE_PARAMS):
        origination = _originations(vintage, args.loans_per_vintage, rng)
        performance = _performance(origination, vintage, args.window, rng)

        emitted = _apply_missing(origination, rng)
        columns = [
            "loan_id", "orig_date", "credit_score", "orig_ltv", "orig_cltv", "dti",
            "orig_upb", "orig_rate", "orig_term", "purpose", "occupancy", "channel",
            "property_type", "num_units", "num_borrowers", "first_time_buyer",
            "mi_pct", "state",
        ]
        emitted[columns].to_csv(
            out / f"origination_{vintage}.txt", sep="|", header=False, index=False
        )
        performance.to_csv(
            out / f"performance_{vintage}.txt", sep="|", header=False, index=False
        )
        total_loans += len(origination)
        total_rows += len(performance)
        print(f"  {vintage}: {len(origination):>6,} loans, {len(performance):>8,} monthly rows")

    print(f"\nwrote {total_loans:,} loans and {total_rows:,} performance rows to {out}")
    print(f"data as of period {AS_OF_PERIOD}, so the newest vintage is partly truncated")


if __name__ == "__main__":
    main()
