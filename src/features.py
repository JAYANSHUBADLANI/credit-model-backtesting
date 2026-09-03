"""Origination fields to model features. The one shared transformation path.

Deliberately thin. Almost every characteristic is taken as reported, because the binner
already handles missing values by giving them their own bin and the whole point of the
weight of evidence approach is that raw shape does not need pre-treatment.

One derived characteristic is added. `second_lien_gap` is combined loan to value minus loan to
value, which is the size of the silent second lien sitting behind the first. A borrower at 80
LTV with a 20 point gap put nothing down; a borrower at 80 LTV with no gap put twenty percent
down. The two are the same loan to the first lien and completely different credit, and the
piggyback structure is one of the things that actually distinguishes the mid decade vintages.
Leaving it to the model to infer from two correlated columns does not work, because the
selection step drops one of a correlated pair.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import FeatureConfig

DERIVED = ["second_lien_gap"]


def build_features(origination: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    frame = origination.copy()

    ltv = pd.to_numeric(frame.get("orig_ltv"), errors="coerce")
    cltv = pd.to_numeric(frame.get("orig_cltv"), errors="coerce")
    gap = cltv - ltv
    # A negative gap is not a negative second lien, it is a data error. Left as missing rather
    # than clipped to zero, so it lands in the missing bin and is visible in the bin table.
    frame["second_lien_gap"] = np.where(gap >= 0, gap, np.nan)

    for column in config.categorical:
        if column in frame:
            frame[column] = frame[column].astype("object")
            frame.loc[frame[column].isna(), column] = np.nan

    missing = [c for c in config.all_features if c not in frame.columns]
    if missing:
        raise KeyError(
            f"features named in config are absent from the origination extract: {missing}. "
            "Either the layout map in src/loans.py does not match this extract, or the "
            "feature list refers to a field this publisher does not provide."
        )
    return frame[config.all_features].copy()


def feature_columns(config: FeatureConfig) -> List[str]:
    return config.all_features
