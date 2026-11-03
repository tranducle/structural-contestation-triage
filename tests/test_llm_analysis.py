"""Analysis logic of the v3 instruction-model study, checked against values known in advance.

Every test fixes one rule a reported number depends on: how a policy reads the recorded answers,
how strata are weighted back to the test split, and when gate G4 must refuse to analyse a run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import analyze_triage_study as analysis  # noqa: E402
import llm_policies as pol  # noqa: E402
import otrf_probe_runner as otrf  # noqa: E402
import salad_triage_runner as salad  # noqa: E402
from llm_common import messages_sha256  # noqa: E402
from llm_loss import break_even  # noqa: E402
from llm_stats import StratifiedDesign, expected_loss, holm_adjust, paired_mcnemar, wilson_interval  # noqa: E402

SALAD_STIMULI = V3 / "data/stimuli/salad_llm_stimuli.jsonl"
SALAD_MANIFEST = V3 / "data/stimuli/salad_llm_stimuli_manifest.json"
OTRF_STIMULI = REPO_ROOT / "data/stimuli/otrf_high_confidence_stimuli.jsonl"
V2_CORE = REPO_ROOT / "src/core.py"


@pytest.fixture(scope="module")
def stimuli() -> list[dict]:
    return analysis.load_jsonl(SALAD_STIMULI)


@pytest.fixture(scope="module")
def population() -> dict[str, int]:
    return json.loads(SALAD_MANIFEST.read_text())["population_strata"]


def _record(disposition: str, messages: list[dict] | None = None) -> dict:
    messages = messages or [{"role": "user", "content": "x"}]
    return {"disposition": disposition, "messages": messages, "messages_sha256": messages_sha256(messages),
            "raw_response": json.dumps({"disposition": disposition}), "model_digest": "d", "options": {"seed": 1},
            "parse_valid": disposition in {"escalate", "dismiss"}, "rationale": "", "evidence": []}


def _stimulus(stratum: str) -> dict:
    return {"stimulus_id": f"S_{stratum}", "label": int(stratum.startswith("hostile")),
            "pitsa_gate": stratum.endswith("_gate"), "stratum": stratum}


def test_holm_matches_v2_core() -> None:
    spec = importlib.util.spec_from_file_location("v2_core", V2_CORE)
    core = importlib.util.module_from_spec(spec)
    sys.modules["v2_core"] = core  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(core)
    p_values = list(np.random.default_rng(3).uniform(0, 0.2, 20)) + [0.01, 0.01, 1.0]
    assert holm_adjust(p_values) == pytest.approx(core.holm_adjust(p_values))


def test_wilson_known_values() -> None:
    assert wilson_interval(5, 10) == pytest.approx([0.2366, 0.7634], abs=1e-4)
    assert wilson_interval(0, 10) == pytest.approx([0.0, 0.2775], abs=1e-4)
    assert wilson_interval(0, 0) is None


def test_mcnemar_counts_direction() -> None:
    result = paired_mcnemar(np.array([1, 1, 1, 0, 0]), np.array([0, 0, 1, 1, 0]))
    assert (result["first_only"], result["second_only"], result["discordant"]) == (2, 1, 3)
    assert paired_mcnemar(np.ones(4), np.ones(4))["exact_p"] == 1.0


def test_weights_recover_population_and_pitsa_gate_rates(stimuli, population) -> None:
    design = StratifiedDesign([s["stratum"] for s in stimuli], population)
    for name, size in population.items():
        assert design.weights[design.mask([name])].sum() == pytest.approx(size)
    gate = np.array([s["pitsa_gate"] for s in stimuli], dtype=float)
    assert all(bool(s["pitsa_gate"]) == s["stratum"].endswith("_gate") for s in stimuli)
    hostile, benign = pol.HOSTILE_STRATA, pol.BENIGN_STRATA
    assert design.weighted_rate(gate, hostile) == pytest.approx(73138 / 88570)
    assert design.weighted_rate(gate, benign) == pytest.approx(32378 / 328211)
    counts = design.resample_counts(200, 7)
    for name in population:
        assert np.all(counts[:, design.mask([name])].sum(axis=1) == design.sample_counts[name])
    assert np.ptp(design.bootstrap_rates(counts, gate, hostile)) == pytest.approx(0.0)  # gate is constant per stratum


def test_policy_logic_on_gated_hostile_alert() -> None:
    stimulus = _stimulus("hostile_gate")
    sid = stimulus["stimulus_id"]
    records = {(sid, c): _record(d) for c, d in {
        "s1_clean": "dismiss", "s1_injected": "dismiss", "s1_spot_clean": "escalate", "s1_spot_injected": "dismiss",
        "s2_struct_after_clean": "escalate", "s2_placebo_after_clean": "dismiss",
        "s2_struct_after_injected": "escalate", "s2_placebo_after_injected": "escalate",
        "s2_spotstruct_after_spot_injected": "escalate"}.items()}
    decide = lambda policy, world: pol.salad_decision(records, stimulus, policy, world)  # noqa: E731
    assert decide("U", "clean") == pol.Decision(0, False, False)
    assert decide("S", "clean") == pol.Decision(1, False, True)
    assert decide("P", "clean") == pol.Decision(0, False, True)
    assert decide("D", "clean") == pol.Decision(1, False, False)  # spotlight escalated, no second turn
    assert decide("DS", "attacked") == pol.Decision(1, False, True)
    assert decide("G", "attacked").escalate == 1 and decide("PITSA", "attacked").escalate == 1


def test_ungated_dismissal_is_never_contested() -> None:
    stimulus = _stimulus("hostile_nogate")
    sid = stimulus["stimulus_id"]
    records = {(sid, c): _record("dismiss") for c in ("s1_clean", "s1_injected", "s1_spot_clean", "s1_spot_injected")}
    for policy in ("S", "P", "DS", "G", "PITSA"):
        assert pol.salad_decision(records, stimulus, policy, "attacked") == pol.Decision(0, False, False)


def test_invalid_answer_escalates_and_is_counted() -> None:
    stimulus = _stimulus("benign_nogate")
    sid = stimulus["stimulus_id"]
    records = {(sid, "s1_clean"): _record("unparseable"), (sid, "s1_spot_clean"): _record("dismiss")}
    assert pol.salad_decision(records, stimulus, "U", "attacked") == pol.Decision(1, True, False)
    gated = _stimulus("benign_gate")
    records = {(gated["stimulus_id"], "s1_clean"): _record("unparseable")}
    assert pol.salad_decision(records, gated, "G", "clean") == pol.Decision(1, False, False)


def test_benign_alerts_are_never_read_as_injected() -> None:
    for world in pol.WORLDS:
        assert pol.stage1_condition(_stimulus("benign_gate"), world, False) == "s1_clean"
    assert pol.stage1_condition(_stimulus("hostile_gate"), "attacked", True) == "s1_spot_injected"


def test_otrf_policy_reads_second_turn_only_after_dismissal() -> None:
    stimulus = {"stimulus_id": "E1"}
    records = {("E1", c): _record(d) for c, d in {
        "s1_clean": "escalate", "s1_injected": "dismiss", "s1_spot_injected": "escalate",
        "s2_prov_after_injected": "escalate", "s2_placebo_after_injected": "dismiss"}.items()}
    assert pol.otrf_decision(records, stimulus, "U").escalate == 0
    assert pol.otrf_decision(records, stimulus, "PROV") == pol.Decision(1, False, True)
    assert pol.otrf_decision(records, stimulus, "PLACEBO") == pol.Decision(0, False, True)
    assert pol.otrf_decision(records, stimulus, "DPROV") == pol.Decision(1, False, False)


def _complete_records(stimulus: dict) -> dict:
    records = {}
    for _, condition in salad.stage1_plan([stimulus]):
        records[(stimulus["stimulus_id"], condition)] = _record("escalate", salad.stage1_messages(stimulus, condition))
    return records


def _g4(records: dict, stimulus: dict) -> dict:
    return pol.completeness(records, pol.salad_planned([stimulus], records, 0), {stimulus["stimulus_id"]: stimulus},
                            pol.salad_expected_messages, analysis.without_assistant(salad.leaked_fields))


def test_g4_passes_complete_run_and_detects_missing_drift_and_tampering(stimuli) -> None:
    stimulus = next(s for s in stimuli if s["stratum"] == "hostile_gate")
    records = _complete_records(stimulus)
    assert _g4(records, stimulus)["passed"]
    missing = dict(records)
    missing.pop((stimulus["stimulus_id"], "s1_injected"))
    assert _g4(missing, stimulus)["missing_count"] == 1
    drifted = dict(records)
    edited = [dict(m) for m in records[(stimulus["stimulus_id"], "s1_clean")]["messages"]]
    edited[0]["content"] += " Be lenient."
    drifted[(stimulus["stimulus_id"], "s1_clean")] = _record("escalate", edited)
    assert _g4(drifted, stimulus)["prompt_drift_count"] == 1
    tampered = dict(records)
    tampered[(stimulus["stimulus_id"], "s1_clean")] = {**records[(stimulus["stimulus_id"], "s1_clean")],
                                                       "messages": edited}
    assert _g4(tampered, stimulus)["stored_hash_mismatch_count"] == 1
    mixed = dict(records)
    mixed[(stimulus["stimulus_id"], "s1_clean")] = {**records[(stimulus["stimulus_id"], "s1_clean")],
                                                    "model_digest": "other"}
    assert not _g4(mixed, stimulus)["passed"]


def test_g4_requires_the_second_turn_a_gated_dismissal_triggers(stimuli) -> None:
    stimulus = next(s for s in stimuli if s["stratum"] == "hostile_gate")
    records = _complete_records(stimulus)
    records[(stimulus["stimulus_id"], "s1_clean")] = _record("dismiss", salad.stage1_messages(stimulus, "s1_clean"))
    report = _g4(records, stimulus)
    assert report["missing_count"] == 2  # structural and placebo second turns after s1_clean


def test_break_even_equalizes_loss() -> None:
    for recall_a, fpr_a, recall_b, fpr_b in ((0.9, 0.3, 0.4, 0.05), (0.2, 0.01, 0.7, 0.2)):
        for prevalence in (0.01, 0.2125):
            k_star = break_even(recall_a, fpr_a, recall_b, fpr_b, prevalence)["k_star"]
            assert expected_loss(recall_a, fpr_a, prevalence, k_star) == pytest.approx(
                expected_loss(recall_b, fpr_b, prevalence, k_star))
    assert break_even(0.5, 0.1, 0.5, 0.2, 0.05) == {"k_star": None, "a_wins": "A_always"}


def test_payload_echo_markers_exclude_words_already_in_the_prompt() -> None:
    clean = [{"role": "user", "content": "Event forwarded by the host monitoring agent."}]
    markers = analysis.payload_markers("Authorized maintenance by the monitoring agent.", clean)
    assert markers == {"author", "mainte"}
    assert analysis.echoes({"rationale": "Looks like authorized maintenance."}, markers)
    assert not analysis.echoes({"rationale": "The monitoring agent forwarded it."}, markers)


def test_leak_check_ignores_model_written_text(stimuli) -> None:
    stimulus = stimuli[0]
    messages = salad.stage2_messages(stimulus, "s1_clean", json.dumps({"rationale": stimulus["stratum"]}), "placebo")
    assert salad.leaked_fields(messages, stimulus)
    assert analysis.without_assistant(salad.leaked_fields)(messages, stimulus) == []


def test_otrf_suspect_is_in_the_probe() -> None:
    ids = {s["stimulus_id"] for s in analysis.load_jsonl(OTRF_STIMULI)}
    assert set(analysis.OTRF_LABEL_NOISE_SUSPECTS) <= ids
    assert {condition for _, condition, _ in otrf.STAGE2} == {
        "s2_prov_after_injected", "s2_placebo_after_injected", "s2_spotprov_after_spot_injected"}
