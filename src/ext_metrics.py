"""Rates, tests and bounds for the v3 extension study.

Population rates and their intervals follow the locked definitions in analyze_llm_v3 exactly
(population-weighted per-stratum rates, a stratified bootstrap with one resample matrix per alert
set); the extension only widens the set of worlds and policies they run over.

Clopper-Pearson (exact) intervals are added beside every Wilson interval. The reporting convention,
fixed in LOCK_ADDENDA.md after the original data were seen, is that a zero count is reported with
its exact upper bound. For a zero count the exact bound is wider than the Wilson bound below
n = 46 and narrower from n = 46 upward, so the change is not in a uniformly conservative direction.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import binomtest

import analyze_triage_study as base
import extension_triage_runner as er
import llm_policies as pol
import salad_triage_runner as salad
from ext_design import DISCRIMINATION_MIN_HOSTILE_ESCALATIONS, EXT_CONFIRMATORY, EXT_EXPLORATORY, Test
from ext_policies import EXT_POLICIES, EXT_WORLDS, second_turn_map
from llm_stats import StratifiedDesign, holm_adjust, paired_mcnemar, percentile_interval, proportion, rate_interval

HOSTILE, BENIGN = pol.HOSTILE_STRATA, pol.BENIGN_STRATA
STRATA = {"hostile": HOSTILE, "benign": BENIGN}
ORIGINAL_DIFFERENCES = (("S", "M"), ("M", "P"), ("S", "P"), ("M", "U"), ("S", "U"), ("D", "U"), ("DS", "D"))
EXTENDED_DIFFERENCES = (("S", "M"), ("S", "P"), ("M", "P"), ("S", "U"), ("M", "U"), ("G", "U"), ("DS", "D"),
                        ("D", "U"))
F2_REMAP = {"s1_injected": "s1_injected_f2", "s1_spot_injected": "s1_spot_injected_f2"}

Arrays = dict[str, dict[str, dict[str, np.ndarray]]]
Boot = dict[tuple[str, str], tuple[float, float, np.ndarray, np.ndarray]]


def exact_interval(successes: int, trials: int) -> list[float] | None:
    """Two-sided 95 percent Clopper-Pearson interval; None when there are no trials."""
    if trials <= 0:
        return None
    interval = binomtest(int(successes), int(trials)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(interval.low), float(interval.high)]


def ext_proportion(successes: int, trials: int) -> dict[str, Any]:
    return {**proportion(successes, trials), "exact95": exact_interval(successes, trials)}


def add_exact(node: Any) -> Any:
    """Copy of node with exact95 beside every proportion record (successes, trials, wilson95)."""
    if isinstance(node, Mapping):
        out = {key: add_exact(value) for key, value in node.items()}
        if {"successes", "trials", "wilson95"} <= set(node) and "exact95" not in node:
            out["exact95"] = exact_interval(int(node["successes"]), int(node["trials"]))
        return out
    if isinstance(node, list):
        return [add_exact(item) for item in node]
    return node


def exact_mirror(node: Any) -> Any:
    """The proportion records of a locked analysis, re-reported with their exact intervals.

    The tree keeps the locked paths, so a locked key .../wilson95/hi has a mirror .../exact95/hi.
    escalated_per_stratum pairs [k, n] become records under the same names.
    """
    if isinstance(node, Mapping):
        if {"successes", "trials", "wilson95"} <= set(node):
            k, n = int(node["successes"]), int(node["trials"])
            return {"successes": k, "trials": n, "exact95": exact_interval(k, n)}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "escalated_per_stratum":
                out[key] = {name: {"successes": k, "trials": n, "exact95": exact_interval(k, n)}
                            for name, (k, n) in value.items()}
                continue
            mirrored = exact_mirror(value)
            if mirrored not in (None, {}):
                out[key] = mirrored
        return out
    return None


def ext_rates(design: StratifiedDesign, counts: np.ndarray, arrays: Arrays) -> tuple[dict, Boot]:
    """Locked salad_rates over every extension world and policy, plus exact per-stratum records."""
    rates: dict[str, dict[str, Any]] = {}
    boot: Boot = {}
    for world in EXT_WORLDS:
        rates[world] = {}
        for policy in EXT_POLICIES:
            escalate = arrays[world][policy]["escalate"]
            recall_b = design.bootstrap_rates(counts, escalate, HOSTILE)
            fpr_b = design.bootstrap_rates(counts, escalate, BENIGN)
            recall, fpr = design.weighted_rate(escalate, HOSTILE), design.weighted_rate(escalate, BENIGN)
            boot[(world, policy)] = (recall, fpr, recall_b, fpr_b)
            per_stratum = {name: [int(escalate[design.mask([name])].sum()), int(design.mask([name]).sum())]
                           for name in HOSTILE + BENIGN}
            rates[world][policy] = {
                "recall": recall, "recall95": rate_interval(recall_b),
                "fpr": fpr, "fpr95": rate_interval(fpr_b),
                "invalid_decisions": int(arrays[world][policy]["invalid"].sum()),
                "contested_alerts": int(arrays[world][policy]["contested"].sum()),
                "escalated_per_stratum": per_stratum,
                "per_stratum": {name: ext_proportion(k, n) for name, (k, n) in per_stratum.items()},
            }
    return rates, boot


def locked_view(rates: Mapping[str, Any]) -> dict[str, Any]:
    """The part of ext_rates that the locked salad_rates also produces, in its exact shape."""
    return {world: {policy: {k: v for k, v in rates[world][policy].items() if k != "per_stratum"}
                    for policy in pol.SALAD_POLICIES} for world in pol.WORLDS}


def differences(boot: Boot, pairs: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for world in EXT_WORLDS:
        for first, second in pairs:
            a, b = boot[(world, first)], boot[(world, second)]
            output[f"{world}:{first}-{second}"] = {
                "recall_diff": a[0] - b[0], "recall_diff95": percentile_interval(a[2] - b[2]),
                "fpr_diff": a[1] - b[1], "fpr_diff95": percentile_interval(a[3] - b[3]),
            }
    return output


def run_test(test: Test, design: StratifiedDesign, arrays: Arrays) -> dict[str, Any]:
    mask = design.mask(STRATA[test.strata])
    result = paired_mcnemar(arrays[test.world_a][test.policy_a]["escalate"][mask],
                            arrays[test.world_b][test.policy_b]["escalate"][mask])
    return {"first": f"{test.policy_a}@{test.world_a}", "second": f"{test.policy_b}@{test.world_b}",
            "corpus": test.corpus, **result}


def extension_tests(
    by_model: Mapping[str, Mapping[str, tuple[StratifiedDesign, Arrays]]], added: str | None
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """test_id -> model -> result; Holm over every confirmatory (test, model) pair."""
    tests: dict[str, dict[str, Any]] = {}
    for family, specs in (("confirmatory", EXT_CONFIRMATORY), ("exploratory", EXT_EXPLORATORY)):
        for test in specs:
            models = [added] if test.models == "added" else list(by_model)
            for model in models:
                if model is None or model not in by_model:
                    continue
                design, arrays = by_model[model][test.corpus]
                tests.setdefault(test.test_id, {})[model] = {"family": family, **run_test(test, design, arrays)}
    pairs = [(test_id, model) for test_id, row in tests.items() for model, result in row.items()
             if result["family"] == "confirmatory"]
    adjusted = holm_adjust([tests[t][m]["exact_p"] for t, m in pairs]) if pairs else []
    for (test_id, model), value in zip(pairs, adjusted):
        tests[test_id][model]["holm_p"] = value
    family = {"size": len(pairs), "method": "Holm over (test, model) pairs R1 to R5 (added model) and X1 to X6"
              " (every model); separate from the H1 to H5 family"}
    return tests, family


def stage2_reversal(records: pol.Records, stimuli: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Escalation rate of each second-turn condition among the alerts where it was issued."""
    tally: dict[str, dict[str, list[int]]] = {}
    for stimulus in stimuli:
        for condition in sorted(set(second_turn_map(stimulus).values())):
            record = records.get((stimulus["stimulus_id"], condition))
            if record is None:
                continue
            row = tally.setdefault(condition, {}).setdefault(stimulus["stratum"], [0, 0, 0])
            row[0] += pol.escalates(record)
            row[1] += 1
            row[2] += int(pol.is_invalid(record))
    return {condition: {name: {**ext_proportion(row[0], row[1]), "invalid": row[2]}
                        for name, row in sorted(by_stratum.items())}
            for condition, by_stratum in sorted(tally.items())}


def f2_injection(records: pol.Records, stimuli: list[dict[str, Any]]) -> dict[str, Any]:
    """The locked injection and contamination block, read on the payload-family-2 opening turns."""
    remapped = dict(records)
    copies: list[dict[str, Any]] = []
    for stimulus in stimuli:
        if not salad.is_hostile(stimulus):
            copies.append(stimulus)
            continue
        for locked_condition, f2_condition in F2_REMAP.items():
            remapped[(stimulus["stimulus_id"], locked_condition)] = records[(stimulus["stimulus_id"], f2_condition)]
        copies.append(er.f2_stimulus(stimulus))
    return add_exact(base.injection_and_contamination(remapped, copies))


def discrimination(design: StratifiedDesign, arrays: Arrays, boot: Boot, applies: bool) -> dict[str, Any]:
    """Item 6: does the unassisted clean opening turn discriminate on the confirmatory sample?"""
    hostile = design.mask(HOSTILE)
    escalated = int(arrays["clean"]["U"]["escalate"][hostile].sum())
    recall, fpr, recall_b, fpr_b = boot[("clean", "U")]
    interval = percentile_interval(recall_b - fpr_b)
    return {
        "hostile_escalations": escalated, "hostile_alerts": int(hostile.sum()),
        "min_hostile_escalations": DISCRIMINATION_MIN_HOSTILE_ESCALATIONS,
        "youden_j": recall - fpr, "youden_j95": interval,
        "meets_criterion": bool(escalated >= DISCRIMINATION_MIN_HOSTILE_ESCALATIONS and interval[0] > 0.0),
        "applies": bool(applies),
    }
