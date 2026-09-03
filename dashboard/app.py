"""Streamlit view over the reports `make backtest` writes.

Reads the CSVs rather than recomputing anything, so the dashboard and the README cannot
disagree: both are views of the same run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config

st.set_page_config(page_title="Credit model backtesting", layout="wide")


@st.cache_data
def _read(path_str: str) -> pd.DataFrame:
    """Keyed on the path, not on no arguments.

    A cached loader keyed on nothing returns the first frame it ever built for every later
    call, whatever was asked for. That defect shipped once in the dashboard of the project
    this accompanies and was caught by a test rather than by looking at the page.
    """
    path = Path(path_str)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    config = load_config()
    reports = config.path(config.artifacts.reports_dir)

    headline_path = reports / "headline.json"
    if not headline_path.exists():
        st.warning("No reports yet. Run `make demo` first.")
        return
    headline = json.loads(headline_path.read_text(encoding="utf-8"))

    st.title("Credit model backtesting")
    st.caption(
        f"Champion {headline['champion']}, frozen on vintages {headline['fit_vintages']}. "
        f"Data as of period {headline['as_of_period']}. "
        "Every figure below is a view of the same run, not a recomputation."
    )

    day_one = headline["day_one"]
    first = headline["first_breach"]
    columns = st.columns(4)
    columns[0].metric("Day one Gini", day_one["gini"],
                      help=f"fit period holdout, {day_one['n']:,} loans")
    columns[1].metric("First calibration breach", first.get("first_calibration_breach_vintage") or "none",
                      help=f"knowable {first.get('first_calibration_known_by')}")
    columns[2].metric("First stability breach", first.get("first_stability_breach_vintage") or "none",
                      help=f"knowable {first.get('first_stability_known_by')}")
    columns[3].metric("First proven discrimination loss",
                      first.get("first_discrimination_breach_vintage") or "none")

    if first.get("reading"):
        st.info(first["reading"])

    bridge = _read(str(reports / "bridge_timeline.csv"))
    calibration = _read(str(reports / "calibration_by_vintage.csv"))
    discrimination = _read(str(reports / "discrimination_by_vintage.csv"))

    st.subheader("Calibration and discrimination, side by side")
    left, right = st.columns(2)
    with left:
        st.caption("Realised over predicted default rate. 1.0 is a correct level.")
        if not calibration.empty:
            st.line_chart(calibration.set_index("vintage")[["ratio"]])
    with right:
        st.caption("Gini with its bootstrap interval. The interval is why most moves here "
                   "are not findings.")
        if not discrimination.empty:
            st.line_chart(discrimination.set_index("vintage")[["gini", "gini_low", "gini_high"]])

    st.subheader("Predicted against realised default rate")
    if not calibration.empty:
        st.bar_chart(calibration.set_index("vintage")[["predicted_pd", "realised_pd"]])

    st.subheader("Vintage curves, cumulative default rate by months on book")
    curves = _read(str(reports / "vintage_curves.csv"))
    if not curves.empty:
        pivot = curves.pivot(index="month_on_book", columns="vintage",
                             values="cumulative_default_rate")
        st.line_chart(pivot)

    st.subheader("The bridge: every signal on one timeline, with the date it was knowable")
    if not bridge.empty:
        st.dataframe(bridge[[
            "vintage", "psi_score", "status", "stability_known_by", "gini", "degraded",
            "ratio", "calibration_breach", "calibration_direction", "outcome_known_by",
        ]], width="stretch", hide_index=True)

    st.subheader("Trigger audit, including what was suppressed and why")
    audit = _read(str(reports / "trigger_audit.csv"))
    if not audit.empty:
        st.dataframe(audit[[
            "vintage", "calibration_breach", "direction", "calibration_run",
            "discrimination_breach", "action", "reason", "suppressed",
        ]], width="stretch", hide_index=True)

    st.subheader("What recalibration does, and what it cannot do")
    evidence_path = reports / "recalibration_evidence.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        one, two, three = st.columns(3)
        one.metric("Ratio before", evidence["ratio_before"])
        two.metric("Ratio after", evidence["ratio_after"])
        three.metric("Gini change", evidence["gini_change"],
                     help="An intercept shift is monotone, so the ordering cannot move.")

    st.subheader("Champion against a feasible and an oracle challenger")
    st.caption(
        "Feasible uses only vintages whose outcomes were known at origination. Oracle ignores "
        "that and is infeasible by construction; it is here to size the value of information "
        "nobody could have had."
    )
    verdicts = _read(str(reports / "challenger_verdict.csv"))
    if not verdicts.empty:
        st.dataframe(verdicts, width="stretch", hide_index=True)

    st.subheader("Characteristics that moved")
    csi = _read(str(reports / "csi_by_vintage.csv"))
    if not csi.empty:
        breached = csi.loc[csi["status"] != "ok"]
        st.dataframe(breached if len(breached) else csi, width="stretch", hide_index=True)

    st.subheader("Vintage completeness")
    st.caption("Every default rate above rests on this table. A vintage the extract cannot "
               "cover to term is dropped whole, defaulters and survivors together.")
    completeness = _read(str(reports / "completeness.csv"))
    if not completeness.empty:
        st.dataframe(completeness, width="stretch", hide_index=True)

    st.subheader("The measured noise floor")
    noise = _read(str(reports / "gini_noise.csv"))
    if not noise.empty:
        st.caption("How far the Gini wanders on resamples of one population that has not "
                   "moved. Any year on year change smaller than this is not evidence.")
        st.dataframe(noise, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
