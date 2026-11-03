"""Extension gate E5: the extension analysis and its value file.

The central guarantee is that the extension cannot move a locked number. It is tested on the real
locked response store: with every extension condition filled by a copy of its locked counterpart,
the matched placebo must decide exactly as the placebo, payload family 2 exactly as family 1, and
the locked part of the recomputed rates must equal the locked analysis JSON exactly. That is the
same check the official analysis runs as a guard, so it cannot first fail at the end of the batch.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import analyze_extension_study as ax  # noqa: E402
import analyze_triage_study as base  # noqa: E402
import ext_metrics as xm  # noqa: E402
import ext_policies as xp  # noqa: E402
import llm_policies as pol  # noqa: E402
import make_extension_macros as mx  # noqa: E402
import salad_triage_runner as salad  # noqa: E402
from ext_design import EXTENSION_SEED, mplacebo_condition  # noqa: E402
from llm_common import messages_sha256  # noqa: E402
from llm_stats import StratifiedDesign, proportion  # noqa: E402

PAPER = REPO_ROOT
LOCKED_STIMULI = V3 / "data/stimuli/salad_llm_stimuli.jsonl"
BENIGN_STIMULI = V3 / "data/stimuli/salad_llm_ext_benign_stimuli.jsonl"
POPULATION = json.loads((V3 / "data/stimuli/salad_llm_stimuli_manifest.json").read_text())["population_strata"]
BASE_STORE = V3 / "data/responses/salad_responses.jsonl"
LOCKED_ANALYSIS = PAPER / "results/triage/triage_study_analysis.json"
EXT_STORE = REPO_ROOT / "data/responses_extension/salad_ext_responses.jsonl"
EXT_ANALYSIS = REPO_ROOT / "results/triage_extension/extension_study_analysis.json"
LOCKED_MODELS = ("qwen2.5:3b", "llama3.2:3b", "phi3:latest", "qwen2.5:7b")

# Extension condition -> the locked condition whose record stands in for it.
STAND_INS = {
    "s1_injected_f2": "s1_injected", "s1_spot_injected_f2": "s1_spot_injected",
    "s2_mplacebo_after_clean": "s2_placebo_after_clean", "s2_mplacebo_after_injected": "s2_placebo_after_injected",
    "s2_struct_after_injected_f2": "s2_struct_after_injected",
    "s2_placebo_after_injected_f2": "s2_placebo_after_injected",
    "s2_mplacebo_after_injected_f2": "s2_placebo_after_injected",
    "s2_spotstruct_after_spot_injected_f2": "s2_spotstruct_after_spot_injected",
}


@pytest.fixture(scope="module")
def stimuli() -> list[dict]:
    return base.load_jsonl(LOCKED_STIMULI)


@pytest.fixture(scope="module")
def base_records() -> dict:
    grouped, duplicates = base.group_by_model(base.load_jsonl(BASE_STORE))
    assert not any(duplicates.values())
    return grouped


@pytest.fixture(scope="module")
def locked_analysis() -> dict:
    return json.loads(LOCKED_ANALYSIS.read_text(encoding="utf-8"))


def with_stand_ins(records: dict, stimuli: list[dict]) -> dict:
    out = dict(records)
    for stimulus in stimuli:
        for condition, source in STAND_INS.items():
            key = (stimulus["stimulus_id"], source)
            if key in records:
                out[(stimulus["stimulus_id"], condition)] = records[key]
    return out


# ---------------------------------------------------------------- decisions on the real locked store

@pytest.mark.parametrize("model", LOCKED_MODELS)
def test_generic_rule_equals_locked_rule_on_every_locked_pair(model, stimuli, base_records):
    records = base_records[model]
    for stimulus in stimuli:
        for world in pol.WORLDS:
            for policy in pol.SALAD_POLICIES:
                assert xp.generic_decision(records, stimulus, policy, world) == \
                    pol.salad_decision(records, stimulus, policy, world), (model, stimulus["stimulus_id"], policy)


@pytest.mark.parametrize("model", LOCKED_MODELS)
def test_new_pairs_read_the_conditions_they_name(model, stimuli, base_records):
    arrays = xp.ext_policy_arrays(with_stand_ins(base_records[model], stimuli), stimuli)
    for field in ("escalate", "invalid", "contested"):
        for world in pol.WORLDS:
            assert np.array_equal(arrays[world]["M"][field], arrays[world]["P"][field]), (world, field)
        for policy in xp.EXT_POLICIES:
            assert np.array_equal(arrays["attacked_f2"][policy][field], arrays["attacked"][policy][field]), policy
    ax.guard_locked_arrays(model, arrays, base_records[model], stimuli)


@pytest.mark.parametrize("model", LOCKED_MODELS)
def test_locked_rates_reproduced_exactly(model, stimuli, base_records, locked_analysis):
    """The official guard, run here on the real 160 alerts with the locked 10,000 resamples."""
    arrays = xp.ext_policy_arrays(with_stand_ins(base_records[model], stimuli), stimuli)
    design = StratifiedDesign([s["stratum"] for s in stimuli], POPULATION)
    counts = design.resample_counts(base.BOOTSTRAP_RESAMPLES, base.BOOTSTRAP_SEED)
    rates, _ = xm.ext_rates(design, counts, arrays)
    ax.guard_locked_rates(model, rates, locked_analysis)
    for world in xp.EXT_WORLDS:
        for policy in xp.EXT_POLICIES:
            row = rates[world][policy]
            assert {k: [r["successes"], r["trials"]] for k, r in row["per_stratum"].items()} == \
                row["escalated_per_stratum"]


def test_rate_guard_rejects_a_moved_number(stimuli, base_records, locked_analysis):
    model = LOCKED_MODELS[0]
    design = StratifiedDesign([s["stratum"] for s in stimuli], POPULATION)
    arrays = xp.ext_policy_arrays(with_stand_ins(base_records[model], stimuli), stimuli)
    rates, _ = xm.ext_rates(design, design.resample_counts(200, base.BOOTSTRAP_SEED), arrays)
    with pytest.raises(RuntimeError, match="differ from the locked analysis"):
        ax.guard_locked_rates(model, rates, locked_analysis)  # 200 resamples move every interval


def test_array_guard_rejects_a_changed_decision(stimuli, base_records):
    model = LOCKED_MODELS[0]
    arrays = xp.ext_policy_arrays(with_stand_ins(base_records[model], stimuli), stimuli)
    arrays["clean"]["U"]["escalate"] = 1 - arrays["clean"]["U"]["escalate"]
    with pytest.raises(RuntimeError, match="differs from the locked decision"):
        ax.guard_locked_arrays(model, arrays, base_records[model], stimuli)


def test_benign_alerts_are_never_injected_in_any_world(stimuli):
    for stimulus in stimuli:
        for spotlight in (False, True):
            condition = xp.ext_stage1_condition(stimulus, "attacked_f2", spotlight)
            assert ("injected" in condition) == salad.is_hostile(stimulus)
            assert condition.endswith("_f2") == salad.is_hostile(stimulus)


def test_matched_placebo_follows_the_opening_answer_it_reads(stimuli):
    hostile = next(s for s in stimuli if salad.is_hostile(s) and s["pitsa_gate"])
    sid = hostile["stimulus_id"]
    dismiss, escalate = {"disposition": "dismiss"}, {"disposition": "escalate"}
    records = {(sid, "s1_injected_f2"): dismiss, (sid, mplacebo_condition("s1_injected_f2")): escalate}
    assert xp.generic_decision(records, hostile, "M", "attacked_f2") == pol.Decision(1, False, True)
    records[(sid, "s1_injected_f2")] = escalate  # no second turn when the opening answer escalates
    assert xp.generic_decision(records, hostile, "M", "attacked_f2") == pol.Decision(1, False, False)


def test_family2_reversal_and_injection_blocks_mirror_family1(stimuli, base_records):
    records = with_stand_ins(base_records[LOCKED_MODELS[0]], stimuli)
    reversal = xm.stage2_reversal(records, stimuli)
    assert reversal["s2_mplacebo_after_clean"] == reversal["s2_placebo_after_clean"]
    family2 = xm.f2_injection(records, stimuli)
    family1 = base.injection_and_contamination(records, stimuli)
    for block in ("plain", "spotlight"):
        for subset in ("hostile_gate", "hostile_nogate", "hostile_all"):
            row = family1[block][subset]
            for field in ("escalated_clean", "escalated_injected", "flip_to_dismiss", "flip_to_escalate"):
                assert family2[block][subset][field]["successes"] == row[field]["successes"], (block, subset, field)
                assert "exact95" in family2[block][subset][field]


# ---------------------------------------------------------------- gate G4-ext on the smoke store

@pytest.fixture(scope="module")
def smoke() -> tuple[dict, dict, dict, dict]:
    ext, duplicates = base.group_by_model(base.load_jsonl(EXT_STORE))
    assert not any(duplicates.values())
    model = LOCKED_MODELS[0]
    original = base.load_jsonl(LOCKED_STIMULI)
    benign = base.load_jsonl(BENIGN_STIMULI)
    corpora = {"mplacebo": original, "f2": original, "benign": benign}
    by_id = {s["stimulus_id"]: s for s in original + benign}
    grouped, _ = base.group_by_model(base.load_jsonl(BASE_STORE))
    return ext[model], grouped[model], corpora, by_id


def g4(ext_records: dict, base_records: dict, corpora: dict, by_id: dict) -> dict:
    return xp.ext_completeness(ext_records, base_records, corpora, by_id, base.without_assistant(salad.leaked_fields))


def test_g4_ext_passes_on_the_smoke_store(smoke):
    check = g4(*smoke)
    assert check["passed"], check
    assert check["missing_count"] == check["unexpected_count"] == check["prompt_drift_count"] == 0


def test_g4_ext_catches_each_failure(smoke):
    ext, base_records, corpora, by_id = smoke
    key = next(k for k in sorted(ext) if k[0].startswith("SALAD_"))  # a locked alert, so it has base records

    missing = {k: v for k, v in ext.items() if k != key}
    assert g4(missing, base_records, corpora, by_id)["missing_count"] == 1

    unexpected = {**ext, (key[0], "s2_mplacebo_after_spot_clean"): ext[key]}
    assert g4(unexpected, base_records, corpora, by_id)["unexpected_count"] == 1

    drifted = copy.deepcopy(ext)
    drifted[key]["messages"][-1]["content"] += " "
    drifted[key]["messages_sha256"] = messages_sha256(drifted[key]["messages"])
    assert g4(drifted, base_records, corpora, by_id)["prompt_drift_count"] == 1

    other_model = copy.deepcopy(ext)
    for record in other_model.values():
        record["model_digest"] = "000000000000"
    result = g4(other_model, base_records, corpora, by_id)
    assert not result["passed"] and not result["digest_matches_base"]

    other_decoding = copy.deepcopy(ext)
    for record in other_decoding.values():
        record["options"] = {**record["options"], "temperature": 0.7}
    result = g4(other_decoding, base_records, corpora, by_id)
    assert not result["passed"] and not result["decoding_matches_base"]

    base_key = next(k for k in base_records if k[0] == key[0])
    overlap = {**ext, base_key: base_records[base_key]}
    result = g4(overlap, base_records, corpora, by_id)
    assert not result["passed"] and result["base_overlap_count"] == 1


# ---------------------------------------------------------------- exact bounds

@pytest.mark.parametrize("n, upper", [(40, 0.0880973), (80, 0.0450640), (120, 0.0302730)])
def test_exact_upper_bound_of_a_zero_count(n, upper):
    low, high = xm.exact_interval(0, n)
    assert low == 0.0 and high == pytest.approx(upper, abs=1e-6)
    assert high == pytest.approx(1 - 0.025 ** (1 / n), abs=1e-9)


def test_exact_against_wilson_direction_is_mixed():
    """For a zero count the exact bound is wider than Wilson below n = 46 and narrower from 46 up."""
    for n in range(1, 401):
        exact, wilson = xm.exact_interval(0, n)[1], proportion(0, n)["wilson95"][1]
        assert (exact > wilson) == (n < 46), n


def test_exact_mirror_keeps_locked_paths(locked_analysis):
    mirror = xm.exact_mirror({"salad": locked_analysis["salad"], "otrf": locked_analysis.get("otrf", {})})
    record = mirror["salad"]["qwen2.5:3b"]["injection"]["plain"]["hostile_gate"]["escalated_injected"]
    assert record["successes"] == 0 and record["trials"] == 40
    assert record["exact95"][1] == pytest.approx(0.0880973, abs=1e-6)
    per_stratum = mirror["salad"]["qwen2.5:3b"]["rates"]["clean"]["U"]["escalated_per_stratum"]["benign_nogate"]
    assert set(per_stratum) == {"successes", "trials", "exact95"}
    assert xm.exact_mirror({"a": {"b": 1.0}}) == {}


# ---------------------------------------------------------------- tests, family size, item 6

def synthetic_model(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for corpus, sizes in (("original", 40), ("extended", 40)):
        strata = [name for name in xm.HOSTILE + xm.BENIGN for _ in range(sizes + (80 if corpus == "extended"
                                                                                   and name == "benign_gate" else 0))]
        arrays = {w: {p: {"escalate": rng.integers(0, 2, len(strata)).astype(np.int8)} for p in xp.EXT_POLICIES}
                  for w in xp.EXT_WORLDS}
        out[corpus] = (StratifiedDesign(strata, POPULATION), arrays)
    return out


@pytest.mark.parametrize("added, size", [("mistral:7b", 35), (None, 24)])
def test_extension_family_size(added, size):
    models = list(LOCKED_MODELS) + ([added] if added else [])
    by_model = {model: synthetic_model(i) for i, model in enumerate(models)}
    tests, family = xm.extension_tests(by_model, added)
    assert family["size"] == size
    assert all(set(row) == ({added} if test_id.startswith("R") else set(models)) for test_id, row in tests.items())
    for row in tests.values():
        for result in row.values():
            assert (result["family"] == "confirmatory") == ("holm_p" in result)
            if "holm_p" in result:
                assert result["holm_p"] >= result["exact_p"]


def discrimination_case(escalated: int, recall_b: np.ndarray, fpr_b: np.ndarray) -> dict:
    strata = [name for name in xm.HOSTILE + xm.BENIGN for _ in range(40)]
    design = StratifiedDesign(strata, POPULATION)
    escalate = np.zeros(len(strata), dtype=np.int8)
    escalate[:escalated] = 1
    arrays = {"clean": {"U": {"escalate": escalate}}}
    return xm.discrimination(design, arrays, {("clean", "U"): (0.2, 0.0, recall_b, fpr_b)}, applies=True)


def test_item6_requires_both_count_and_interval():
    above, zero = np.full(500, 0.2), np.zeros(500)
    assert discrimination_case(10, above, zero)["meets_criterion"]
    assert not discrimination_case(9, above, zero)["meets_criterion"]
    assert not discrimination_case(80, above, above)["meets_criterion"]


# ---------------------------------------------------------------- value file

def test_value_formatting():
    assert mx.ext_format(("exact", "x", "exact95", "hi"), 0.0880973) == "8.8"
    assert mx.ext_format(("original", "m", "discrimination", "youden_j"), 0.25) == "25.0"
    assert mx.ext_format(("tests", "X1", "m", "holm_p"), 0.0004) == "\\ensuremath{<}0.001"
    assert mx.ext_format(("tests", "X1", "m", "holm_p"), 0.0123) == "0.012"
    assert mx.ext_format(("loss_f2", "models", "m", "break_even", "attacked", "S-U", "pi0.05", "k_star"), 3.27) == "3.3"
    assert mx.ext_format(("design", "bootstrap_seed"), EXTENSION_SEED) == str(EXTENSION_SEED)
    assert mx.ext_format(("design", "extended_alerts"), 240) == "240"
    assert mx.ext_format(("tests", "X1", "m", "family"), "confirmatory") == "confirmatory"
    assert mx.ext_format(("tests", "X1", "m", "first"), "S@attacked_f2") == "S@attacked\\_f2"


def test_value_file_from_the_smoke_analysis(tmp_path):
    if not EXT_ANALYSIS.is_file():
        pytest.skip("smoke analysis absent")
    analysis = json.loads(EXT_ANALYSIS.read_text(encoding="utf-8"))
    values = mx.ext_values(analysis)
    key = ("ext", "exact", "locked", "salad", "qwen2.5:3b", "injection", "plain", "hostile_gate",
           "escalated_injected", "exact95", "hi")
    assert values[key] == "8.8"
    assert not any(path[1] in mx.EXT_SKIPPED_TOP_LEVEL for path in values)
    text = mx.render(values, False, analysis["label"], [EXT_ANALYSIS])
    assert "\\def\\llmextvaluesource{SMOKE}" in text and "-" * 2 not in text.split("\\def\\llmextvaluesource")[1]
    out = tmp_path / "SMOKE_llm_ext_values.tex"
    out.write_text(text, encoding="utf-8")
    tex = tmp_path / "use.tex"
    tex.write_text("Bound \\val{" + "/".join(key) + "}.\n", encoding="utf-8")
    assert mx.check([tex], [out], None) == 0
    tex.write_text("Bound \\val{ext/no/such/key}.\n", encoding="utf-8")
    assert mx.check([tex], [out], None) == 1


def test_added_values_drop_the_single_model_tests(locked_analysis):
    one_model = {**locked_analysis, "models": ["qwen2.5:3b"]}
    values = mx.added_values(one_model)
    assert values and all(path[0] == "add" for path in values)
    assert not any(len(path) > 3 and path[1] == "salad" and path[3] == "tests" for path in values)


def test_only_a_passed_official_analysis_is_official():
    assert mx.is_official({"label": "OFFICIAL", "g4_passed": True})
    assert not mx.is_official({"label": "OFFICIAL", "g4_passed": False})
    assert not mx.is_official({"label": "SMOKE (validates code only; never evidence)", "g4_passed": True})
