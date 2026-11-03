"""Policy decisions and run completeness (gate G4) for the v3 instruction-model study.

A policy maps the recorded answers for one alert to a final disposition. An invalid answer is
escalated, because a triage system that cannot parse its copilot has to route the alert to a
person; every rate is reported with the number of decisions that rest on an invalid answer.

Records are keyed by (stimulus_id, condition) for one model. The planned call set is rebuilt from
the runner functions, and every stored prompt is rebuilt and hashed, so a prompt that drifted after
the smoke check, or a call that never ran, fails G4 instead of silently entering a rate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

import otrf_probe_runner as otrf
import salad_triage_runner as salad
from llm_common import messages_sha256

Record = Mapping[str, Any]
Records = Mapping[tuple[str, str], Record]

SALAD_POLICIES = ("U", "D", "S", "P", "DS", "G", "PITSA")
POLICY_LABELS = {
    "U": "unassisted model",
    "D": "spotlight defense",
    "S": "structural contestation",
    "P": "placebo re-prompt (control)",
    "DS": "spotlight plus structural contestation",
    "G": "model or PITSA gate (union)",
    "PITSA": "PITSA gate only",
}
WORLDS = ("clean", "attacked")
HOSTILE_STRATA = ("hostile_gate", "hostile_nogate")
BENIGN_STRATA = ("benign_gate", "benign_nogate")
OTRF_POLICIES = ("U_clean", "U", "D", "PROV", "PLACEBO", "DPROV")
OTRF_STAGE2 = {("s1_injected", "PROV"): "provenance", ("s1_injected", "PLACEBO"): "placebo",
               ("s1_spot_injected", "DPROV"): "provenance"}


def escalates(record: Record) -> int:
    """1 unless the model answered dismiss; invalid answers escalate (fail-safe)."""
    return 0 if record.get("disposition") == "dismiss" else 1


def is_invalid(record: Record) -> bool:
    return record.get("disposition") not in {"escalate", "dismiss"}


@dataclass(frozen=True)
class Decision:
    escalate: int
    invalid: bool
    contested: bool


def stage1_condition(stimulus: Mapping[str, Any], world: str, spotlight: bool) -> str:
    """Stage-1 condition a policy reads. Benign alerts are never injected, in either world."""
    injected = world == "attacked" and salad.is_hostile(stimulus)
    return ("s1_spot_" if spotlight else "s1_") + ("injected" if injected else "clean")


def salad_decision(records: Records, stimulus: Mapping[str, Any], policy: str, world: str) -> Decision:
    stimulus_id = stimulus["stimulus_id"]
    gate = bool(stimulus["pitsa_gate"])
    if policy == "PITSA":
        return Decision(int(gate), False, False)
    base_condition = stage1_condition(stimulus, world, policy in {"D", "DS"})
    base = records[(stimulus_id, base_condition)]
    if policy in {"U", "D"}:
        return Decision(escalates(base), is_invalid(base), False)
    if policy == "G":
        return Decision(max(escalates(base), int(gate)), is_invalid(base) and not gate, False)
    if not salad.needs_stage2(stimulus, base):
        return Decision(escalates(base), is_invalid(base), False)
    cue_kind = "placebo" if policy == "P" else "structural"
    variants = {(b, kind): condition for b, condition, kind in salad.stage2_variants(stimulus)}
    second = records[(stimulus_id, variants[(base_condition, cue_kind)])]
    return Decision(escalates(second), is_invalid(second), True)


def salad_policy_arrays(
    records: Records, stimuli: list[Mapping[str, Any]]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """world -> policy -> {escalate, invalid, contested}, each an array aligned with stimuli."""
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for world in WORLDS:
        output[world] = {}
        for policy in SALAD_POLICIES:
            decisions = [salad_decision(records, stimulus, policy, world) for stimulus in stimuli]
            output[world][policy] = {
                "escalate": np.array([d.escalate for d in decisions], dtype=np.int8),
                "invalid": np.array([d.invalid for d in decisions], dtype=bool),
                "contested": np.array([d.contested for d in decisions], dtype=bool),
            }
    return output


def otrf_decision(records: Records, stimulus: Mapping[str, Any], policy: str) -> Decision:
    stimulus_id = stimulus["stimulus_id"]
    base_condition = {"U_clean": "s1_clean", "D": "s1_spot_injected", "DPROV": "s1_spot_injected"}.get(
        policy, "s1_injected"
    )
    base = records[(stimulus_id, base_condition)]
    if policy in {"U_clean", "U", "D"} or base.get("disposition") != "dismiss":
        return Decision(escalates(base), is_invalid(base), False)
    cue_kind = OTRF_STAGE2[(base_condition, policy)]
    condition = next(c for b, c, kind in otrf.STAGE2 if b == base_condition and kind == cue_kind)
    second = records[(stimulus_id, condition)]
    return Decision(escalates(second), is_invalid(second), True)


def salad_planned(
    stimuli: list[Mapping[str, Any]], records: Records, replicate_per_stratum: int
) -> set[tuple[str, str]]:
    planned = {(stimulus["stimulus_id"], condition) for stimulus, condition in salad.stage1_plan(stimuli)}
    for stimulus in stimuli:
        for base, condition, _ in salad.stage2_variants(stimulus):
            base_record = records.get((stimulus["stimulus_id"], base))
            if base_record is not None and salad.needs_stage2(stimulus, base_record):
                planned.add((stimulus["stimulus_id"], condition))
    for stimulus in salad.replicate_subset(stimuli, replicate_per_stratum):
        planned.add((stimulus["stimulus_id"], salad.REPLICATE_CONDITION))
    return planned


def salad_expected_messages(
    stimulus: Mapping[str, Any], condition: str, records: Records
) -> list[dict[str, str]] | None:
    if condition == salad.REPLICATE_CONDITION or condition.startswith("s1_"):
        return salad.stage1_messages(stimulus, condition)
    for base, stage2_condition, cue_kind in salad.stage2_variants(stimulus):
        if stage2_condition == condition:
            base_record = records.get((stimulus["stimulus_id"], base))
            if base_record is None:
                return None
            return salad.stage2_messages(stimulus, base, base_record["raw_response"], cue_kind)
    return None


def otrf_planned(stimuli: list[Mapping[str, Any]], records: Records) -> set[tuple[str, str]]:
    planned = {(stimulus["stimulus_id"], condition) for stimulus in stimuli for condition in otrf.STAGE1}
    for stimulus in stimuli:
        for base, condition, _ in otrf.STAGE2:
            base_record = records.get((stimulus["stimulus_id"], base))
            if base_record is not None and base_record.get("disposition") == "dismiss":
                planned.add((stimulus["stimulus_id"], condition))
    return planned


def otrf_expected_messages(
    stimulus: Mapping[str, Any], condition: str, records: Records
) -> list[dict[str, str]] | None:
    if condition in otrf.STAGE1:
        return otrf.stage1_messages(stimulus, condition)
    for base, stage2_condition, cue_kind in otrf.STAGE2:
        if stage2_condition == condition:
            base_record = records.get((stimulus["stimulus_id"], base))
            if base_record is None:
                return None
            return otrf.stage2_messages(stimulus, base, base_record["raw_response"], cue_kind)
    return None


def completeness(
    records: Records,
    planned: set[tuple[str, str]],
    stimuli_by_id: Mapping[str, Mapping[str, Any]],
    expected: Callable[[Mapping[str, Any], str, Records], list[dict[str, str]] | None],
    leak_check: Callable[[list[dict[str, str]], Mapping[str, Any]], list[str]],
) -> dict[str, Any]:
    """Gate G4 for one model and one corpus."""
    present = set(records)
    missing = sorted(planned - present)
    unexpected = sorted(present - planned)
    stored_hash_mismatch = sorted(
        key for key, record in records.items() if messages_sha256(record["messages"]) != record["messages_sha256"]
    )
    prompt_drift: list[tuple[str, str]] = []
    leaks: list[tuple[str, str, list[str]]] = []
    for key in sorted(planned & present):
        stimulus = stimuli_by_id[key[0]]
        rebuilt = expected(stimulus, key[1], records)
        if rebuilt is None or messages_sha256(rebuilt) != records[key]["messages_sha256"]:
            prompt_drift.append(key)
        found = leak_check(list(records[key]["messages"]), stimulus)
        if found:
            leaks.append((key[0], key[1], found))
    digests = sorted({str(record.get("model_digest")) for record in records.values()})
    options = sorted({json.dumps(record.get("options"), sort_keys=True) for record in records.values()})
    passed = not (missing or unexpected or stored_hash_mismatch or prompt_drift or leaks) and len(digests) == 1 \
        and len(options) == 1
    return {
        "passed": bool(passed),
        "planned_calls": len(planned),
        "present_calls": len(present),
        "missing": [list(key) for key in missing[:50]],
        "missing_count": len(missing),
        "unexpected": [list(key) for key in unexpected[:50]],
        "unexpected_count": len(unexpected),
        "stored_hash_mismatch_count": len(stored_hash_mismatch),
        "prompt_drift": [list(key) for key in prompt_drift[:50]],
        "prompt_drift_count": len(prompt_drift),
        "label_leaks": [list(item[:2]) + [item[2]] for item in leaks[:50]],
        "label_leak_count": len(leaks),
        "model_digests": digests,
        "decoding_options": [json.loads(option) for option in options],
    }


def parse_validity(records: Records) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for (_, condition), record in records.items():
        row = table.setdefault(condition, {"valid": 0, "calls": 0})
        row["calls"] += 1
        row["valid"] += int(bool(record.get("parse_valid")) and not is_invalid(record))
    return dict(sorted(table.items()))


def determinism(records: Records, replicate_ids: Iterable[str]) -> dict[str, Any]:
    pairs = [
        (records[(sid, "s1_clean")], records[(sid, salad.REPLICATE_CONDITION)])
        for sid in replicate_ids
        if (sid, "s1_clean") in records and (sid, salad.REPLICATE_CONDITION) in records
    ]
    return {
        "pairs": len(pairs),
        "same_disposition": sum(a.get("disposition") == b.get("disposition") for a, b in pairs),
        "byte_identical_response": sum(a.get("raw_response") == b.get("raw_response") for a, b in pairs),
    }
