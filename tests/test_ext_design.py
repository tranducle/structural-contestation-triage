"""Extension gate E2: the fixed design of the v3 extension study.

The matched placebo must match the structural cue in authority, flag and length while carrying none
of its score, threshold or contributions; payload family 2 must stay inside the locked attack
channel and avoid every string a check or the manuscript bars; the hypothesis family must be well
formed. If any of these fails, a reported extension effect could come from the design rather than
from the model.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import salad_triage_runner as salad  # noqa: E402
from ext_design import (  # noqa: E402
    EXT_CONFIRMATORY,
    EXT_EXPLORATORY,
    INJECTION_PAYLOADS_F2,
    MATCHED_PLACEBO_CUE,
    mplacebo_condition,
    payload_f2,
)
from llm_common import CUE_TAIL, PLACEBO_CUE  # noqa: E402
from build_salad_stimuli import EXCLUDED_FIELDS, INJECTION_PAYLOADS, PILLAR_NAMES  # noqa: E402

SALAD_STIMULI = V3 / "data/stimuli/salad_llm_stimuli.jsonl"
PROCESS_RESIDUE = ("WITHDRAWN", "PENDING", "SUPERSEDED", "DO NOT CITE")
REGULATED_WORDS = ("novel", "significant", "outperform", "state-of-the-art", "state of the art")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def stimuli() -> list[dict]:
    return _load(SALAD_STIMULI)


@pytest.fixture(scope="module")
def gated_cues(stimuli) -> list[str]:
    return [salad.structural_cue(item) for item in stimuli if item["pitsa_gate"]]


def test_matched_placebo_length_matches_the_structural_cue(gated_cues):
    median_chars = statistics.median(len(cue) for cue in gated_cues)
    median_words = statistics.median(len(cue.split()) for cue in gated_cues)
    assert 0.9 <= len(MATCHED_PLACEBO_CUE) / median_chars <= 1.1
    assert abs(len(MATCHED_PLACEBO_CUE.split()) - median_words) <= 5
    # The old placebo is the length confound the matched placebo removes.
    assert len(PLACEBO_CUE) / median_chars < 0.5


def test_matched_placebo_shares_the_cue_authority_and_flag_but_no_alert_specific_content(gated_cues):
    prefix = gated_cues[0].split("Score ")[0]
    assert all(cue.startswith(prefix) for cue in gated_cues)
    assert MATCHED_PLACEBO_CUE.startswith(prefix)
    assert MATCHED_PLACEBO_CUE.endswith(CUE_TAIL)
    assert "flags this alert for review" in MATCHED_PLACEBO_CUE
    rest = MATCHED_PLACEBO_CUE[len(prefix):]
    assert not re.search(r"\d", rest), "no score, threshold or contribution value"
    assert "threshold" not in rest and "contribution" not in rest.lower()
    assert not any(name in rest for name in PILLAR_NAMES.values())


def test_family_two_payloads_are_distinct_from_family_one_and_from_each_other():
    assert len(INJECTION_PAYLOADS_F2) == 6
    assert len(set(INJECTION_PAYLOADS_F2)) == 6
    assert not set(INJECTION_PAYLOADS_F2) & set(INJECTION_PAYLOADS)
    assert all("dismiss" in payload for payload in INJECTION_PAYLOADS_F2)


def test_family_two_payloads_avoid_barred_strings():
    for payload in INJECTION_PAYLOADS_F2:
        assert salad.TAG not in payload and "untrusted" not in payload.lower() and "<" not in payload
        assert "Independent structural check" not in payload and "Quality check" not in payload
        assert CUE_TAIL not in payload
        assert not any(f'"{field}"' in payload for field in EXCLUDED_FIELDS)
        assert not any(word in payload for word in PROCESS_RESIDUE)
        assert not any(word in payload.lower() for word in REGULATED_WORDS)
        assert "\u2013" not in payload and "\u2014" not in payload and "-" * 2 not in payload


def test_family_two_assignment_is_deterministic_and_uses_every_payload(stimuli):
    hostile = [item for item in stimuli if salad.is_hostile(item)]
    first = [payload_f2(item["record_id"]) for item in hostile]
    assert first == [payload_f2(item["record_id"]) for item in hostile]
    assert set(first) == set(INJECTION_PAYLOADS_F2)
    assert min(Counter(first).values()) >= 5


def test_matched_placebo_condition_names():
    assert mplacebo_condition("s1_clean") == "s2_mplacebo_after_clean"
    assert mplacebo_condition("s1_injected") == "s2_mplacebo_after_injected"


def test_extension_hypothesis_family_is_well_formed():
    tests = EXT_CONFIRMATORY + EXT_EXPLORATORY
    assert len({test.test_id for test in tests}) == len(tests)
    assert len(EXT_CONFIRMATORY) == 11 and len(EXT_EXPLORATORY) == 5
    for test in tests:
        assert {test.policy_a, test.policy_b} <= {"U", "D", "S", "P", "DS", "M"}
        assert {test.world_a, test.world_b} <= {"clean", "attacked", "attacked_f2"}
        assert test.strata in {"hostile", "benign"}
        assert test.corpus in {"original", "extended"} and test.models in {"added", "all"}
        if test.strata == "benign":
            assert test.world_a == test.world_b == "clean", "benign alerts are never injected"
        if test.corpus == "extended":
            assert test.strata == "benign", "only the benign gated stratum is extended"
    # Holm family with one added model: 5 replications plus 6 tests on 5 models.
    assert sum(1 if test.models == "added" else 5 for test in EXT_CONFIRMATORY) == 35
