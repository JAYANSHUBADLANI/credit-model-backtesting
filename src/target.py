"""Building the loan level target from the monthly performance panel.

Three decisions live here and all three are the kind that produce a beautiful and completely
wrong result if they are got wrong quietly.

**The performance window is fixed for every vintage.** A 2008 origination observed at 12
months and a 2005 origination observed at 24 months are not comparable, and a chart that puts
them side by side is measuring seasoning rather than credit risk. The window is read from
config, applied identically everywhere, and asserted after filtering.

**Right truncated loans are dropped, not counted as good.** A loan that neither terminated nor
reached the end of the window has an unknown outcome. Counting it as good is the single most
effective way to make the newest vintage look excellent, because the newest vintage is the one
with the most unfinished loans in it.

**Completeness is a property of the vintage, not of the loan.** This one is subtle and it
produced a wrong answer here before it was caught. The obvious definition of a usable loan is
"it terminated, or it reached the end of the window". Applied to a vintage the extract only
partly covers, that definition keeps every loan that defaulted early, because defaulting is a
termination, and drops every loan that was still quietly performing, because that one has not
finished yet. The survivors are discarded and the failures are kept, and the youngest vintage
comes out looking far worse than it is. On the fixture that inflated the newest vintage's
default rate from about 1 percent to 2.1 percent, above vintages that were genuinely better.

So a loan is usable only if its **whole window was observable** at the data as of date, which
is a question about its origination date and nothing else. Loans from a vintage the extract
cannot cover to term are dropped as a block, defaulters and survivors together, and the
completeness report says how many and why.

**Prepayment is a competing risk and is kept as its own column.** A loan that prepaid at month
eight never had the opportunity to default. Mortgage prepayment is not a rounding error, the
refinance waves are enormous, and a default rate computed over all loans and one computed over
loans that survived the window are different quantities. Both are reported. The naive rate is
biased downward and this module does not hide which one a caller is holding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import TargetConfig
from .loans import data_as_of, period_add


@dataclass
class TargetSummary:
    """What the target construction did, per vintage, so it can be printed and checked."""

    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame


def build_target(
    performance: pd.DataFrame,
    origination: pd.DataFrame,
    config: TargetConfig,
    as_of: int | None = None,
) -> pd.DataFrame:
    """Collapse the monthly panel to one row per loan.

    Returns loan_id, default, prepaid, last_age, complete and the reason a loan is incomplete.
    `as_of` defaults to the latest reporting period present in the performance files.
    """
    window = config.performance_window_months
    if as_of is None:
        as_of = data_as_of(performance)
    if window <= 0:
        raise ValueError("performance_window_months must be positive")

    inside = performance.loc[
        performance["loan_age"].notna()
        & (performance["loan_age"] >= 1)
        & (performance["loan_age"] <= window)
    ].copy()

    if not inside.empty and inside["loan_age"].max() > window:
        raise AssertionError(
            "performance rows survived the window filter with loan_age beyond the window"
        )

    codes_loss = set(config.credit_loss_zero_balance_codes)
    codes_prepaid = set(config.prepaid_zero_balance_codes)

    inside["is_serious_dlq"] = (
        inside["dlq_months"].notna() & (inside["dlq_months"] >= config.default_dlq_threshold)
    )
    inside["is_credit_loss"] = inside["zero_balance_code"].isin(codes_loss)
    inside["is_prepaid"] = inside["zero_balance_code"].isin(codes_prepaid)

    grouped = inside.groupby("loan_id", sort=False)
    summary = grouped.agg(
        last_age=("loan_age", "max"),
        months_observed=("loan_age", "size"),
        ever_serious_dlq=("is_serious_dlq", "any"),
        ever_credit_loss=("is_credit_loss", "any"),
        ever_prepaid=("is_prepaid", "any"),
    ).reset_index()

    # The month the loan first showed as defaulted, which the vintage curves need. Kept
    # separate from the aggregation above because it is a min over a filtered subset, not an
    # any over all rows, and folding it in would silently give surviving loans an age of 1.
    events = inside.loc[inside["is_serious_dlq"] | inside["is_credit_loss"]]
    first_event = events.groupby("loan_id", sort=False)["loan_age"].min().rename("default_age")
    summary = summary.merge(first_event, on="loan_id", how="left")

    summary["default"] = (summary["ever_serious_dlq"] | summary["ever_credit_loss"]).astype(int)
    # A loan cannot be both. Credit loss termination wins over a prepayment code appearing in
    # the same window, because a loan that defaulted and was then disposed of is a default.
    summary["prepaid"] = (summary["ever_prepaid"] & (summary["default"] == 0)).astype(int)

    summary["terminated"] = (summary["default"] == 1) | (summary["prepaid"] == 1)
    summary["reached_window"] = summary["last_age"] >= window

    loans = origination[["loan_id", "vintage", "orig_date"]].copy()
    # Observability is decided by the origination date alone, before anything about the loan's
    # own history is looked at. See the module docstring for why that ordering matters.
    loans["observable_through"] = loans["orig_date"].apply(
        lambda period: period_add(int(period), window) if pd.notna(period) else None
    )
    loans["fully_observable"] = loans["observable_through"].apply(
        lambda period: bool(period is not None and period <= as_of)
    )
    merged = loans.merge(summary, on="loan_id", how="left")

    # `.eq(True)` rather than `.fillna(False).astype(bool)`: a loan absent from the summary
    # is NaN here, and eq maps that to False without the object dtype downcast pandas warns on.
    merged["terminated"] = merged["terminated"].eq(True)
    merged["reached_window"] = merged["reached_window"].eq(True)
    merged["complete"] = merged["fully_observable"] & (
        merged["terminated"] | merged["reached_window"]
    )
    merged["incomplete_reason"] = np.where(
        merged["complete"],
        "",
        np.where(
            ~merged["fully_observable"],
            f"vintage not observable to month {window} at as of period {as_of}",
            "gap in the performance rows for this loan",
        ),
    )

    # A loan in the origination file with no performance rows at all is not a performing loan,
    # it is a loan whose performance file has not been loaded. Treat it as incomplete rather
    # than as a good, and let the completeness report make it visible.
    never_observed = merged["last_age"].isna()
    merged.loc[never_observed, ["default", "prepaid"]] = 0
    merged.loc[never_observed, "last_age"] = 0
    merged.loc[never_observed, "months_observed"] = 0
    merged.loc[never_observed, "complete"] = False
    merged.loc[never_observed, "incomplete_reason"] = "no performance rows found for this loan"

    for column in ["default", "prepaid", "last_age", "months_observed"]:
        merged[column] = merged[column].fillna(0).astype(int)
    merged["complete"] = merged["complete"].eq(True)

    return merged[[
        "loan_id", "vintage", "orig_date", "default", "prepaid", "last_age",
        "default_age", "months_observed", "fully_observable", "complete", "incomplete_reason",
    ]]


def completeness_report(target: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per vintage: how many loans there are, how many are usable, and why the rest are not.

    This table is the honest counterpart to every default rate quoted downstream. A vintage
    that is 40 percent truncated does not get to be compared with one that is 2 percent
    truncated without the reader being told.
    """
    rows = []
    for vintage, group in target.groupby("vintage", sort=True):
        total = len(group)
        complete = int(group["complete"].sum())
        usable = group.loc[group["complete"]]
        defaults = int(usable["default"].sum())
        prepaid = int(usable["prepaid"].sum())
        survived = usable.loc[usable["prepaid"] == 0]
        rows.append({
            "vintage": int(vintage),
            "loans": total,
            "complete": complete,
            "unobservable": int((~group["fully_observable"]).sum()),
            "truncated": total - complete,
            "truncated_share": round((total - complete) / total, 4) if total else 0.0,
            "defaults": defaults,
            "prepaid": prepaid,
            # The naive rate counts prepaid loans in the denominator even though they left
            # before they could default. It is biased downward and it is the one most
            # published default rates are.
            "default_rate_naive": round(defaults / complete, 5) if complete else 0.0,
            # The competing risk adjusted rate drops loans that prepaid inside the window.
            "default_rate_survivors": (
                round(defaults / len(survived), 5) if len(survived) else 0.0
            ),
            "window_months": window,
        })
    return pd.DataFrame(rows)


def usable(target: pd.DataFrame, config: TargetConfig) -> pd.DataFrame:
    """The loans a model may be fitted on or measured against."""
    if not config.drop_incomplete:
        return target
    return target.loc[target["complete"]].reset_index(drop=True)


def assert_window_consistent(target: pd.DataFrame, window: int) -> None:
    """Every usable loan either terminated inside the window or ran the full window.

    Called before fitting and before every backtest pass. It is cheap and it is the check that
    catches a config edit to the window that was not followed by a rebuild of the target.
    """
    complete = target.loc[target["complete"]]
    bad = complete.loc[
        (complete["last_age"] < window)
        & (complete["default"] == 0)
        & (complete["prepaid"] == 0)
    ]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} loans are marked complete but neither terminated nor reached "
            f"month {window}. The target was probably built with a different window than the "
            "one now in config. Rebuild it rather than comparing across two windows."
        )
    unobservable = complete.loc[~complete["fully_observable"]]
    if len(unobservable):
        raise AssertionError(
            f"{len(unobservable)} loans are marked complete despite their window running past "
            "the data as of date. Keeping those would keep the early defaulters and drop the "
            "survivors. See the module docstring."
        )
