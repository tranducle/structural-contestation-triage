from __future__ import annotations

import pandas as pd

from lmd_shift_study import CORE_COLUMNS
from salad_detector_study import engineer_salad

LMD_PRIMARY_FEATURES = tuple(CORE_COLUMNS)


def lmd_label_name(value: int) -> str:
    mapping = {0: "Normal", 1: "EoRS", 2: "EoHT"}
    try:
        return mapping[int(value)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("unsupported LMD label") from error


def build_salad_features(raw: pd.DataFrame) -> pd.DataFrame:
    prepared = raw.copy()
    if "difficulty_level" not in prepared.columns:
        prepared["difficulty_level"] = "unknown"
    if "attack_category" not in prepared.columns:
        prepared["attack_category"] = "unknown"
    engineered = engineer_salad(prepared)
    engineered["text_proxy"] = engineered["neutral_text"].astype(str)
    return engineered
