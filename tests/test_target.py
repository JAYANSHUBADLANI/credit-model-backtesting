"""Target construction. Four of these exist because they caught something.

`test_truncation_drops_the_whole_vintage_not_just_the_survivors` is the important one. The
first version of `build_target` defined a usable loan as one that terminated or reached the end
of the window. On a vintage the extract covers only halfway, that keeps every early defaulter,
because defaulting is a termination, and drops every loan still quietly performing. The newest
vintage came out at a 2.1 percent default rate against 1.1 percent for the vintages either side
of it, and the ranking of vintages by credit quality was wrong.
"""

import pandas as pd
import pytest

from src.config import TargetConfig
from src.target import assert_window_consistent, build_target, completeness_report, usable

CONFIG = TargetConfig(
    performance_window_months=24,
    default_dlq_threshold=6,
    credit_loss_zero_balance_codes=["02", "03", "09", "15"],
    prepaid_zero_balance_codes=["01"],
)


def origination(rows):
    frame = pd.DataFrame(rows, columns=["loan_id", "orig_date"])
    frame["vintage"] = frame["orig_date"] // 100
    return frame


def performance(rows):
    frame = pd.DataFrame(rows, columns=["loan_id", "period", "loan_age", "dlq_status", "zero_balance_code"])
    frame["dlq_months"] = pd.to_numeric(frame["dlq_status"], errors="coerce")
    return frame


def months(loan_id, start_period, count, dlq=None, terminal=None, terminal_at=None):
    rows = []
    year, month = divmod(start_period, 100)
    for age in range(1, count + 1):
        index = year * 12 + (month - 1) + age
        period = (index // 12) * 100 + (index % 12) + 1
        status = "0" if dlq is None else str(dlq(age))
        code = terminal if (terminal and age == (terminal_at or count)) else ""
        rows.append((loan_id, period, age, status, code))
    return rows


def test_serious_delinquency_inside_the_window_is_a_default():
    orig = origination([("A", 200501)])
    perf = performance(months("A", 200501, 24, dlq=lambda age: min(max(0, age - 18), 6)))
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert out.loc[0, "default"] == 1
    assert out.loc[0, "default_age"] == 24


def test_credit_loss_termination_counts_even_without_reaching_180_days():
    """A loan disposed of through a short sale while only three months down is a default.

    A target built on delinquency status alone misses it entirely, and the loss is real.
    """
    orig = origination([("A", 200501)])
    perf = performance(months("A", 200501, 14, dlq=lambda age: min(max(0, age - 11), 3), terminal="03"))
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert out.loc[0, "default"] == 1
    assert out.loc[0, "prepaid"] == 0


def test_prepayment_is_not_a_default_and_is_kept_separately():
    orig = origination([("A", 200501)])
    perf = performance(months("A", 200501, 9, terminal="01"))
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert out.loc[0, "default"] == 0
    assert out.loc[0, "prepaid"] == 1
    assert out.loc[0, "complete"]


def test_default_wins_over_a_prepayment_code_in_the_same_window():
    """One loan cannot be both, and the answer must not depend on evaluation order."""
    orig = origination([("A", 200501)])
    rows = months("A", 200501, 20, dlq=lambda age: min(max(0, age - 14), 6))
    rows[-1] = (rows[-1][0], rows[-1][1], rows[-1][2], rows[-1][3], "01")
    perf = performance(rows)
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert out.loc[0, "default"] == 1
    assert out.loc[0, "prepaid"] == 0


def test_delinquency_after_the_window_does_not_count():
    orig = origination([("A", 200501)])
    perf = performance(months("A", 200501, 36, dlq=lambda age: min(max(0, age - 29), 6)))
    out = build_target(perf, orig, CONFIG, as_of=200912)
    assert out.loc[0, "default"] == 0
    assert out.loc[0, "last_age"] == 24


def test_truncation_drops_the_whole_vintage_not_just_the_survivors():
    """Regression test for a real defect. See the module docstring.

    Two loans originate in the same month, too late for the extract to cover to term. One
    defaults early, so it terminates and looks finished. The other is still performing. Keeping
    the first and dropping the second reports a 100 percent default rate for the vintage.
    """
    orig = origination([("EARLY_DEFAULT", 201801), ("STILL_PAYING", 201801)])
    perf = performance(
        months("EARLY_DEFAULT", 201801, 12, dlq=lambda age: min(max(0, age - 6), 6))
        + months("STILL_PAYING", 201801, 12)
    )
    out = build_target(perf, orig, CONFIG, as_of=201901)

    assert not out["complete"].any(), "neither loan's window was observable"
    assert not out["fully_observable"].any()
    assert len(usable(out, CONFIG)) == 0

    report = completeness_report(out, 24)
    assert report.loc[0, "unobservable"] == 2
    assert report.loc[0, "default_rate_naive"] == 0.0


def test_an_observable_vintage_keeps_its_early_terminations():
    orig = origination([("EARLY_DEFAULT", 200501), ("STILL_PAYING", 200501)])
    perf = performance(
        months("EARLY_DEFAULT", 200501, 12, dlq=lambda age: min(max(0, age - 6), 6))
        + months("STILL_PAYING", 200501, 24)
    )
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert out["complete"].all()
    assert usable(out, CONFIG)["default"].tolist() == [1, 0]


def test_a_loan_with_no_performance_rows_is_incomplete_not_good():
    orig = origination([("GHOST", 200501)])
    perf = performance(months("OTHER", 200501, 24))
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert out.loc[0, "complete"] == False  # noqa: E712
    assert "no performance rows" in out.loc[0, "incomplete_reason"]


def test_unreported_status_is_not_read_as_performing():
    """"XX" means the servicer did not report. Coercing it to zero invents a performing month."""
    orig = origination([("A", 200501)])
    rows = months("A", 200501, 24, dlq=lambda age: min(max(0, age - 18), 6))
    rows = [(r[0], r[1], r[2], "XX", r[4]) for r in rows]
    perf = performance(rows)
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert out.loc[0, "default"] == 0


def test_window_consistency_check_catches_a_changed_window():
    orig = origination([("A", 200501)])
    perf = performance(months("A", 200501, 24))
    out = build_target(perf, orig, CONFIG, as_of=200812)
    assert_window_consistent(out, 24)
    with pytest.raises(AssertionError):
        assert_window_consistent(out, 36)
