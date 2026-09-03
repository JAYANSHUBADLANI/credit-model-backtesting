"""Reading GSE loan level files.

The Freddie Mac Single Family Loan-Level Dataset and the Fannie Mae Single-Family Loan
Performance data are both pipe delimited, headerless, and split into an origination file and a
monthly performance file per period. Both publishers have changed their layout at least once,
so the column positions live in one dictionary here rather than being spread through the
parsing code. Pointing this project at a real extract is an edit to `ORIGINATION_LAYOUT` and
`PERFORMANCE_LAYOUT` after reading the current layout document, not a rewrite.

The fixture generator writes this same layout, which is the only reason the pipeline can be
run end to end without the registration the real data needs.

Sentinels matter more here than they look. Both publishers encode "not collected" as a
sentinel value rather than a blank: a credit score of 9999, a debt to income of 999. Read
naively those become the highest value in the column, they survive quantile binning as a real
bin, and the card then allocates points to a missing data code. They are mapped to NaN here,
once, so that the missing bin the binner already maintains is what picks them up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# name -> zero based column position in the raw file.
ORIGINATION_LAYOUT: Dict[str, int] = {
    "loan_id": 0,
    "orig_date": 1,          # YYYYMM
    "credit_score": 2,
    "orig_ltv": 3,
    "orig_cltv": 4,
    "dti": 5,
    "orig_upb": 6,
    "orig_rate": 7,
    "orig_term": 8,
    "purpose": 9,            # P purchase, C cash out refi, N no cash out refi
    "occupancy": 10,         # P primary, S second home, I investor
    "channel": 11,           # R retail, B broker, C correspondent
    "property_type": 12,     # SF single family, CO condo, PU planned unit, MH manufactured
    "num_units": 13,
    "num_borrowers": 14,
    "first_time_buyer": 15,  # Y / N
    "mi_pct": 16,
    "state": 17,
}

PERFORMANCE_LAYOUT: Dict[str, int] = {
    "loan_id": 0,
    "period": 1,             # YYYYMM
    "loan_age": 2,
    "dlq_status": 3,         # months delinquent as text, "XX" where unknown
    "zero_balance_code": 4,  # blank while the loan is alive
}

# Values that mean "not collected" rather than a measurement.
SENTINELS: Dict[str, Iterable[float]] = {
    "credit_score": (9999,),
    "dti": (999,),
    "orig_ltv": (999,),
    "orig_cltv": (999,),
    "mi_pct": (999,),
    "num_borrowers": (99,),
}

NUMERIC_ORIGINATION = [
    "credit_score", "orig_ltv", "orig_cltv", "dti", "orig_upb",
    "orig_rate", "orig_term", "num_units", "num_borrowers", "mi_pct",
]


def _read_layout(path: Path, layout: Dict[str, int], delimiter: str) -> pd.DataFrame:
    order = sorted(layout.items(), key=lambda item: item[1])
    frame = pd.read_csv(
        path,
        sep=delimiter,
        header=None,
        usecols=[position for _, position in order],
        names=[name for name, _ in order],
        dtype=str,
        engine="c",
        keep_default_na=False,
        na_values=[""],
    )
    return frame


def _resolve(root: Path, pattern: str) -> List[Path]:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no files matched {pattern} under {root}. Either the extract has not been "
            "downloaded yet, or `make fixture` has not been run."
        )
    return matches


def load_origination(root: Path, pattern: str, delimiter: str = "|") -> pd.DataFrame:
    """Read every origination file matching the pattern into one frame."""
    frames = [_read_layout(path, ORIGINATION_LAYOUT, delimiter) for path in _resolve(root, pattern)]
    frame = pd.concat(frames, ignore_index=True)

    for column in NUMERIC_ORIGINATION:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for sentinel in SENTINELS.get(column, ()):
            frame.loc[frame[column] == sentinel, column] = np.nan

    frame["orig_date"] = pd.to_numeric(frame["orig_date"], errors="coerce").astype("Int64")
    frame["vintage"] = (frame["orig_date"] // 100).astype("Int64")

    duplicated = frame["loan_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} duplicate loan ids in the origination files. Every "
            "downstream join is one row per loan, so a duplicate would silently double count "
            "those loans in both the fit and the realised default rate."
        )
    return frame


def load_performance(root: Path, pattern: str, delimiter: str = "|") -> pd.DataFrame:
    """Read every monthly performance file matching the pattern into one frame."""
    frames = [_read_layout(path, PERFORMANCE_LAYOUT, delimiter) for path in _resolve(root, pattern)]
    frame = pd.concat(frames, ignore_index=True)

    frame["loan_age"] = pd.to_numeric(frame["loan_age"], errors="coerce")
    frame["period"] = pd.to_numeric(frame["period"], errors="coerce").astype("Int64")
    # "XX" and anything else non numeric means the servicer did not report a status. It is not
    # a zero. Coercing it to zero would quietly turn unreported months into performing months.
    frame["dlq_months"] = pd.to_numeric(frame["dlq_status"], errors="coerce")
    frame["zero_balance_code"] = frame["zero_balance_code"].fillna("").str.strip()
    return frame


def period_add(period: int, months: int) -> int:
    """Add whole months to a YYYYMM period."""
    year, month = divmod(int(period), 100)
    index = year * 12 + (month - 1) + int(months)
    return (index // 12) * 100 + (index % 12) + 1


def data_as_of(performance: pd.DataFrame) -> int:
    """The latest reporting period present. Everything after it has not happened yet."""
    latest = performance["period"].max()
    if pd.isna(latest):
        raise ValueError("no usable reporting periods in the performance files")
    return int(latest)


def vintage_counts(origination: pd.DataFrame) -> pd.DataFrame:
    counts = origination.groupby("vintage").size().reset_index(name="loans")
    return counts.sort_values("vintage").reset_index(drop=True)
