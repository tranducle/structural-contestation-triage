"""Reported rate intervals stay in [0, 1].

A stratified bootstrap of a weighted rate divides two sums taken in different orders, so a resample in
which every weighted decision is 1 can return a rate one ulp above 1. The reported bounds of a rate are
clipped to [0, 1]; the bootstrap samples, and every interval that is not a rate, are left as they are.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import analyze_triage_study as analysis  # noqa: E402
from llm_stats import StratifiedDesign, percentile_interval, rate_interval  # noqa: E402

SALAD_STIMULI = V3 / "data/stimuli/salad_llm_stimuli.jsonl"
SALAD_MANIFEST = V3 / "data/stimuli/salad_llm_stimuli_manifest.json"
RESULTS = REPO_ROOT / "results"
OFFICIAL_ANALYSES = [
    RESULTS / "triage/triage_study_analysis.json",
    RESULTS / "triage_added/triage_study_analysis.json",
    RESULTS / "triage_extension/extension_study_analysis.json",
]
RATE_INTERVAL_KEYS = {"recall95", "fpr95"}


@pytest.fixture(scope="module")
def all_ones_rates() -> dict[str, np.ndarray]:
    """Bootstrap rates of a policy that escalates every alert, on the official design and seed."""
    stimuli = analysis.load_jsonl(SALAD_STIMULI)
    population = json.loads(SALAD_MANIFEST.read_text())["population_strata"]
    design = StratifiedDesign([s["stratum"] for s in stimuli], population)
    counts = design.resample_counts(10000, analysis.BOOTSTRAP_SEED)
    escalate = np.ones(len(stimuli))
    return {
        "recall": design.bootstrap_rates(counts, escalate, analysis.HOSTILE),
        "fpr": design.bootstrap_rates(counts, escalate, analysis.BENIGN),
    }


def test_unclipped_bound_exceeds_one_on_the_official_design(all_ones_rates: dict[str, np.ndarray]) -> None:
    # Guards the premise of the clip: if this stops failing upward, the test below proves nothing.
    assert max(percentile_interval(all_ones_rates["recall"])[1],
               percentile_interval(all_ones_rates["fpr"])[1]) > 1.0


@pytest.mark.parametrize("name", ["recall", "fpr"])
def test_rate_interval_stays_in_unit_range(all_ones_rates: dict[str, np.ndarray], name: str) -> None:
    low, high = rate_interval(all_ones_rates[name])
    assert 0.0 <= low <= high <= 1.0
    assert high == 1.0


def test_rate_interval_leaves_samples_unchanged(all_ones_rates: dict[str, np.ndarray]) -> None:
    samples = all_ones_rates["recall"]
    before = samples.copy()
    rate_interval(samples)
    np.testing.assert_array_equal(samples, before)


def test_rate_interval_equals_percentile_interval_inside_unit_range() -> None:
    samples = np.random.default_rng(7).uniform(0.05, 0.95, size=5000)
    assert rate_interval(samples) == percentile_interval(samples)


def test_percentile_interval_is_not_clipped() -> None:
    # Losses and differences are not rates: a loss above 1 review per alert, or a negative difference, must survive.
    assert percentile_interval(np.linspace(2.0, 3.0, 101))[0] > 1.0
    assert percentile_interval(np.linspace(-0.5, -0.1, 101))[1] < 0.0


def _rate_bounds(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], float]]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in RATE_INTERVAL_KEYS and isinstance(value, list):
                for bound in value:
                    yield path + (key,), bound
            else:
                yield from _rate_bounds(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _rate_bounds(value, path + (str(index),))


@pytest.mark.parametrize("path", OFFICIAL_ANALYSES, ids=lambda p: p.parent.name)
def test_official_rate_bounds_in_unit_range(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"official analysis not present: {path}")
    bounds = list(_rate_bounds(json.loads(path.read_text())))
    assert bounds, f"no rate interval found in {path}"
    outside = [(where, bound) for where, bound in bounds if bound is not None and not 0.0 <= bound <= 1.0]
    assert not outside, outside[:5]
