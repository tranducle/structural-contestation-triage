from __future__ import annotations

from itertools import combinations
from typing import Mapping

import pandas as pd


def assert_disjoint_ids(splits: Mapping[str, pd.DataFrame], id_column: str) -> None:
    identifiers = {
        name: set(frame[id_column].dropna().astype(str))
        for name, frame in splits.items()
    }
    for left, right in combinations(sorted(identifiers), 2):
        overlap = identifiers[left] & identifiers[right]
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(
                "identifier overlap between "
                + left
                + " and "
                + right
                + ": "
                + ", ".join(examples)
            )


def chronological_three_way_split(
    frame: pd.DataFrame,
    time_column: str,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, pd.DataFrame]:
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")
    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce")
    if ordered[time_column].isna().any():
        raise ValueError("all timestamps must be parseable")
    ordered = ordered.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = train_end + int(len(ordered) * validation_fraction)
    if train_end <= 0 or validation_end <= train_end or validation_end >= len(ordered):
        raise ValueError("split fractions produce an empty partition")
    return {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:].copy(),
    }
