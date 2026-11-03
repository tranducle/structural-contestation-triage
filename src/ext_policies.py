"""Policy decisions and run completeness (gate G4-ext) for the v3 extension study.

The extension adds one policy and one world to the locked design:

  M            matched placebo: the plain opening turn, then, under the locked rule (PITSA gate
               fired, opening answer dismiss), the matched-placebo second turn
  attacked_f2  hostile alerts carry a payload-family-2 injection; benign alerts stay clean, as in
               the locked attacked world

Every locked (world, policy) pair is decided by the locked llm_policies.salad_decision itself, so
the extension cannot change a locked rate. The new pairs follow the same rule through a second-turn
map that the tests check against the locked decision on every locked pair.

G4-ext rebuilds the planned extension call set from the runner's own step functions and every
stored prompt from the runner's own builders, so a call that never ran or a prompt that drifted
fails the gate instead of entering a rate.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

import extension_triage_runner as er
import llm_policies as pol
import salad_triage_runner as salad
from ext_design import f2_stage2_variants, mplacebo_condition

EXT_POLICIES = pol.SALAD_POLICIES + ("M",)
EXT_WORLDS = pol.WORLDS + ("attacked_f2",)
POLICY_LABELS = {**pol.POLICY_LABELS, "M": "matched placebo re-prompt (control)"}
CUE_KIND = {"S": "structural", "DS": "structural", "P": "placebo", "M": "matched_placebo"}
SPOTLIGHT_POLICIES = {"D", "DS"}

Stimulus = Mapping[str, Any]


def ext_stage1_condition(stimulus: Stimulus, world: str, spotlight: bool) -> str:
    """Opening turn a policy reads. Benign alerts are never injected, in any world."""
    if world == "attacked_f2":
        if salad.is_hostile(stimulus):
            return "s1_spot_injected_f2" if spotlight else "s1_injected_f2"
        return pol.stage1_condition(stimulus, "clean", spotlight)
    return pol.stage1_condition(stimulus, world, spotlight)


def second_turn_map(stimulus: Stimulus) -> dict[tuple[str, str], str]:
    """(opening condition, cue kind) -> second-turn condition, for every second turn of one alert."""
    turns = {(base, kind): condition for base, condition, kind in salad.stage2_variants(stimulus)}
    bases = ["s1_clean"]
    if salad.is_hostile(stimulus):
        turns.update({(base, kind): condition for base, condition, kind in f2_stage2_variants()})
        bases += ["s1_injected", "s1_injected_f2"]
    for base in bases:
        turns[(base, "matched_placebo")] = mplacebo_condition(base)
    return turns


def second_turn_sources(stimulus: Stimulus) -> dict[str, tuple[str, str]]:
    """Second-turn condition -> (opening condition, cue kind); the inverse of second_turn_map."""
    return {condition: key for key, condition in second_turn_map(stimulus).items()}


def generic_decision(records: pol.Records, stimulus: Stimulus, policy: str, world: str) -> pol.Decision:
    """The locked decision rule, written over the extended second-turn map."""
    stimulus_id = stimulus["stimulus_id"]
    gate = bool(stimulus["pitsa_gate"])
    if policy == "PITSA":
        return pol.Decision(int(gate), False, False)
    base_condition = ext_stage1_condition(stimulus, world, policy in SPOTLIGHT_POLICIES)
    base = records[(stimulus_id, base_condition)]
    if policy in {"U", "D"}:
        return pol.Decision(pol.escalates(base), pol.is_invalid(base), False)
    if policy == "G":
        return pol.Decision(max(pol.escalates(base), int(gate)), pol.is_invalid(base) and not gate, False)
    if not salad.needs_stage2(stimulus, base):
        return pol.Decision(pol.escalates(base), pol.is_invalid(base), False)
    second = records[(stimulus_id, second_turn_map(stimulus)[(base_condition, CUE_KIND[policy])])]
    return pol.Decision(pol.escalates(second), pol.is_invalid(second), True)


def ext_decision(records: pol.Records, stimulus: Stimulus, policy: str, world: str) -> pol.Decision:
    if world in pol.WORLDS and policy in pol.SALAD_POLICIES:
        return pol.salad_decision(records, stimulus, policy, world)
    return generic_decision(records, stimulus, policy, world)


def ext_policy_arrays(
    records: pol.Records, stimuli: list[Stimulus]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """world -> policy -> {escalate, invalid, contested}, each aligned with stimuli."""
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for world in EXT_WORLDS:
        output[world] = {}
        for policy in EXT_POLICIES:
            decisions = [ext_decision(records, stimulus, policy, world) for stimulus in stimuli]
            output[world][policy] = {
                "escalate": np.array([d.escalate for d in decisions], dtype=np.int8),
                "invalid": np.array([d.invalid for d in decisions], dtype=bool),
                "contested": np.array([d.contested for d in decisions], dtype=bool),
            }
    return output


def ext_planned(corpora: Mapping[str, list[Stimulus]], merged: pol.Records) -> set[tuple[str, str]]:
    """Every extension call the runner was due to make, given the opening answers on record."""
    planned: set[tuple[str, str]] = set()
    for part, stimuli in corpora.items():
        for stimulus, condition, _, _ in er.stage1_steps(part, stimuli):
            planned.add((stimulus["stimulus_id"], condition))
        for stimulus, condition, base, _ in er.stage2_steps(part, stimuli):
            base_record = merged.get((stimulus["stimulus_id"], str(base)))
            if base_record is not None and salad.needs_stage2(stimulus, base_record):
                planned.add((stimulus["stimulus_id"], condition))
    return planned


def ext_expected_messages(
    stimulus: Stimulus, condition: str, merged: pol.Records
) -> list[dict[str, str]] | None:
    if condition.startswith("s1_"):
        return er.ext_stage1_messages(dict(stimulus), condition)
    source = second_turn_sources(stimulus).get(condition)
    if source is None:
        return None
    base_record = merged.get((stimulus["stimulus_id"], source[0]))
    if base_record is None:
        return None
    return er.ext_stage2_messages(dict(stimulus), source[0], base_record["raw_response"], source[1])


def ext_completeness(
    ext_records: pol.Records,
    base_records: pol.Records,
    corpora: Mapping[str, list[Stimulus]],
    stimuli_by_id: Mapping[str, Stimulus],
    leak_check: Callable[[list[dict[str, str]], Mapping[str, Any]], list[str]],
) -> dict[str, Any]:
    """G4-ext for one model: the locked completeness check plus the extension's own conditions."""
    overlap = sorted(set(ext_records) & set(base_records))
    merged: dict[tuple[str, str], Any] = {**base_records, **ext_records}
    planned = ext_planned(corpora, merged)
    check = pol.completeness(ext_records, planned, stimuli_by_id,
                             lambda stimulus, condition, _: ext_expected_messages(stimulus, condition, merged),
                             leak_check)
    base_digests = sorted({str(record.get("model_digest")) for record in base_records.values()})
    base_options = sorted({repr(sorted((record.get("options") or {}).items())) for record in base_records.values()})
    ext_options = sorted({repr(sorted((record.get("options") or {}).items())) for record in ext_records.values()})
    same_model = check["model_digests"] == base_digests and len(base_digests) == 1
    same_decoding = ext_options == base_options and len(base_options) == 1
    check.update(
        base_overlap_count=len(overlap), base_overlap=[list(key) for key in overlap[:50]],
        base_model_digests=base_digests, digest_matches_base=bool(same_model),
        decoding_matches_base=bool(same_decoding),
    )
    check["passed"] = bool(check["passed"] and not overlap and same_model and same_decoding)
    return check
