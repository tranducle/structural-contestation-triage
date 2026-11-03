from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from splits import assert_disjoint_ids, chronological_three_way_split


def test_assert_disjoint_ids_accepts_separate_identifiers() -> None:
    train = pd.DataFrame({"id": ["a", "b"]})
    validation = pd.DataFrame({"id": ["c"]})
    test = pd.DataFrame({"id": ["d"]})
    assert_disjoint_ids({"train": train, "validation": validation, "test": test}, "id")


def test_assert_disjoint_ids_rejects_overlap() -> None:
    train = pd.DataFrame({"id": ["a", "b"]})
    test = pd.DataFrame({"id": ["b", "c"]})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_ids({"train": train, "test": test}, "id")


def test_chronological_split_never_trains_on_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
            "y": [0, 1] * 5,
        }
    )
    splits = chronological_three_way_split(
        frame,
        "timestamp",
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    assert splits["train"]["timestamp"].max() <= splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() <= splits["test"]["timestamp"].min()
    assert len(splits["train"]) == 6
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2
