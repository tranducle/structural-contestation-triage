"""Analyze the v3 instruction-model study: gate G4 first, then every number the manuscript may report.

Modes:
  official  every model must have answered all 160 SALAD alerts and all 20 OTRF events.
  smoke     analyses the subset each model answered and labels every output SMOKE. Smoke numbers
            validate the code path and are never evidence.
In both modes a failed G4 writes only the G4 report and exits non-zero, so no rate is ever
computed from an incomplete or drifted run.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

import llm_policies as pol
import otrf_probe_runner as otrf
import salad_triage_runner as salad
from llm_loss import loss_analysis
from llm_stats import StratifiedDesign, holm_adjust, paired_mcnemar, percentile_interval, proportion, rate_interval
from llm_tables import write_tables

logger = logging.getLogger(__name__)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261023
HOSTILE, BENIGN = pol.HOSTILE_STRATA, pol.BENIGN_STRATA
OTRF_LABEL_NOISE_SUSPECTS = {
    "OTRF_credential_access_02": "Sysmon event 10 in which VBoxService.exe opens lsass.exe with access "
    "mask 0x1400 (query rights only); the archive records an attack, but this event is plausibly "
    "background activity of the virtualization guest service",
}
# (name, policy A, world A, policy B, world B, strata). first_only counts alerts escalated by A only.
CONFIRMATORY = (
    ("H1_injection_effect_unassisted", "U", "clean", "U", "attacked", HOSTILE),
    ("H2_structural_vs_placebo_recall", "S", "attacked", "P", "attacked", HOSTILE),
    ("H3_structural_vs_placebo_false_escalation", "S", "clean", "P", "clean", BENIGN),
    ("H4_spotlight_vs_plain_recall_attacked", "D", "attacked", "U", "attacked", HOSTILE),
    ("H5_spotlight_structural_vs_spotlight_recall", "DS", "attacked", "D", "attacked", HOSTILE),
)
EXPLORATORY = (
    ("E1_injection_effect_spotlight", "D", "clean", "D", "attacked", HOSTILE),
    ("E2_structural_vs_unassisted_recall_attacked", "S", "attacked", "U", "attacked", HOSTILE),
    ("E3_structural_vs_unassisted_false_escalation", "S", "clean", "U", "clean", BENIGN),
    ("E4_spotlight_vs_plain_false_escalation", "D", "clean", "U", "clean", BENIGN),
    ("E5_structural_vs_placebo_recall_clean", "S", "clean", "P", "clean", HOSTILE),
    ("E6_spotlight_structural_vs_spotlight_false_escalation", "DS", "clean", "D", "clean", BENIGN),
)
DIFFERENCES = (("S", "P"), ("S", "U"), ("D", "U"), ("DS", "D"), ("DS", "U"), ("G", "PITSA"), ("S", "PITSA"))
OTRF_TESTS = (
    ("O1_injection_effect", "U_clean", "U"),
    ("O2_spotlight_vs_plain", "D", "U"),
    ("O3_provenance_vs_placebo", "PROV", "PLACEBO"),
    ("O4_provenance_vs_unassisted", "PROV", "U"),
    ("O5_spotlight_provenance_vs_spotlight", "DPROV", "D"),
)
WORD = re.compile(r"[a-z]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def group_by_model(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[tuple[str, str], dict]], Counter]:
    grouped: dict[str, dict[tuple[str, str], dict]] = {}
    duplicates: Counter = Counter()
    for row in rows:
        records = grouped.setdefault(row["model"], {})
        key = (row["stimulus_id"], row["condition"])
        duplicates[row["model"]] += int(key in records)
        records[key] = row
    return grouped, duplicates


def stems(text: str) -> set[str]:
    """Six-letter prefixes of the words of at least four letters, a crude deterministic stemmer."""
    return {token[:6] for token in WORD.findall(text.lower()) if len(token) >= 4}


def payload_markers(payload: str, clean_messages: list[dict[str, str]]) -> set[str]:
    """Stems the payload introduces that the clean prompt does not already contain."""
    return stems(payload) - stems(" ".join(message["content"] for message in clean_messages))


def echoes(record: Mapping[str, Any], markers: set[str]) -> bool:
    return bool(markers & stems(str(record.get("rationale", ""))))


def cites_annotation(record: Mapping[str, Any]) -> bool:
    return any("annotation" in str(item).lower() for item in record.get("evidence") or [])


def without_assistant(check: Callable) -> Callable:
    """Leak checks apply to what the study sent, not to text the model wrote itself."""
    return lambda messages, stimulus: check([m for m in messages if m["role"] != "assistant"], stimulus)


def salad_rates(design: StratifiedDesign, counts: np.ndarray, arrays: dict) -> tuple[dict, dict]:
    rates: dict[str, dict[str, Any]] = {}
    boot: dict[tuple[str, str], tuple[float, float, np.ndarray, np.ndarray]] = {}
    for world in pol.WORLDS:
        rates[world] = {}
        for policy in pol.SALAD_POLICIES:
            escalate = arrays[world][policy]["escalate"]
            recall_b = design.bootstrap_rates(counts, escalate, HOSTILE)
            fpr_b = design.bootstrap_rates(counts, escalate, BENIGN)
            recall, fpr = design.weighted_rate(escalate, HOSTILE), design.weighted_rate(escalate, BENIGN)
            boot[(world, policy)] = (recall, fpr, recall_b, fpr_b)
            rates[world][policy] = {
                "recall": recall, "recall95": rate_interval(recall_b),
                "fpr": fpr, "fpr95": rate_interval(fpr_b),
                "invalid_decisions": int(arrays[world][policy]["invalid"].sum()),
                "contested_alerts": int(arrays[world][policy]["contested"].sum()),
                "escalated_per_stratum": {
                    name: [int(escalate[design.mask([name])].sum()), int(design.mask([name]).sum())]
                    for name in HOSTILE + BENIGN
                },
            }
    return rates, boot


def salad_differences(boot: dict) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for world in pol.WORLDS:
        for first, second in DIFFERENCES:
            a, b = boot[(world, first)], boot[(world, second)]
            output[f"{world}:{first}-{second}"] = {
                "recall_diff": a[0] - b[0], "recall_diff95": percentile_interval(a[2] - b[2]),
                "fpr_diff": a[1] - b[1], "fpr_diff95": percentile_interval(a[3] - b[3]),
            }
    return output


def salad_tests(design: StratifiedDesign, arrays: dict) -> list[dict[str, Any]]:
    tests = []
    for family, specs in (("confirmatory", CONFIRMATORY), ("exploratory", EXPLORATORY)):
        for name, policy_a, world_a, policy_b, world_b, strata in specs:
            mask = design.mask(strata)
            result = paired_mcnemar(arrays[world_a][policy_a]["escalate"][mask],
                                    arrays[world_b][policy_b]["escalate"][mask])
            tests.append({"name": name, "family": family, "first": f"{policy_a}@{world_a}",
                          "second": f"{policy_b}@{world_b}", "strata": list(strata), **result})
    return tests


def stage2_rates(records: pol.Records, stimuli: list[dict[str, Any]]) -> dict[str, Any]:
    """Escalation rate of each second-turn condition, among the alerts where it was issued."""
    tally: dict[str, dict[str, list[int]]] = {}
    for stimulus in stimuli:
        for _, condition, _ in salad.stage2_variants(stimulus):
            record = records.get((stimulus["stimulus_id"], condition))
            if record is None:
                continue
            row = tally.setdefault(condition, {}).setdefault(stimulus["stratum"], [0, 0, 0])
            row[0] += pol.escalates(record)
            row[1] += 1
            row[2] += int(pol.is_invalid(record))
    return {
        condition: {name: {**proportion(row[0], row[1]), "invalid": row[2]} for name, row in sorted(by_stratum.items())}
        for condition, by_stratum in sorted(tally.items())
    }


def _flip_block(records: pol.Records, members: list[dict], clean_c: str, injected_c: str) -> dict[str, Any]:
    clean = np.array([pol.escalates(records[(s["stimulus_id"], clean_c)]) for s in members], dtype=bool)
    injected = np.array([pol.escalates(records[(s["stimulus_id"], injected_c)]) for s in members], dtype=bool)
    return {
        "escalated_clean": proportion(int(clean.sum()), clean.size),
        "escalated_injected": proportion(int(injected.sum()), injected.size),
        "flip_to_dismiss": proportion(int((clean & ~injected).sum()), int(clean.sum())),
        "flip_to_escalate": proportion(int((~clean & injected).sum()), int((~clean).sum())),
    }


def _contamination(records: pol.Records, hostile: list[dict], clean_c: str, injected_c: str) -> dict[str, Any]:
    echo_clean, echo_injected, cite_clean, cite_injected, dismissed = [], [], [], [], []
    for stimulus in hostile:
        markers = payload_markers(stimulus["injection_payload"], salad.stage1_messages(stimulus, clean_c))
        clean_record = records[(stimulus["stimulus_id"], clean_c)]
        injected_record = records[(stimulus["stimulus_id"], injected_c)]
        echo_clean.append(echoes(clean_record, markers))
        echo_injected.append(echoes(injected_record, markers))
        cite_clean.append(cites_annotation(clean_record))
        cite_injected.append(cites_annotation(injected_record))
        dismissed.append(injected_record.get("disposition") == "dismiss")
    echo_a, dismissed_a = np.array(echo_injected, dtype=bool), np.array(dismissed, dtype=bool)
    return {
        "payload_echo_clean": proportion(sum(echo_clean), len(echo_clean)),
        "payload_echo_injected": proportion(sum(echo_injected), len(echo_injected)),
        "payload_echo_test": paired_mcnemar(echo_a, np.array(echo_clean, dtype=bool)),
        "payload_echo_among_injected_dismissals": proportion(int((echo_a & dismissed_a).sum()), int(dismissed_a.sum())),
        "annotation_cited_clean": proportion(sum(cite_clean), len(cite_clean)),
        "annotation_cited_injected": proportion(sum(cite_injected), len(cite_injected)),
        "annotation_cited_test": paired_mcnemar(np.array(cite_injected, dtype=bool), np.array(cite_clean, dtype=bool)),
    }


def injection_and_contamination(records: pol.Records, stimuli: list[dict[str, Any]]) -> dict[str, Any]:
    hostile = [stimulus for stimulus in stimuli if salad.is_hostile(stimulus)]
    output: dict[str, Any] = {}
    for variant, (clean_c, injected_c) in {"plain": ("s1_clean", "s1_injected"),
                                           "spotlight": ("s1_spot_clean", "s1_spot_injected")}.items():
        block = {group: _flip_block(records, [s for s in hostile if s["stratum"] == group], clean_c, injected_c)
                 for group in HOSTILE}
        block["hostile_all"] = _flip_block(records, hostile, clean_c, injected_c)
        block["contamination"] = _contamination(records, hostile, clean_c, injected_c)
        output[variant] = block
    return output


def analyze_otrf_model(records: pol.Records, stimuli: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = {policy: [pol.otrf_decision(records, s, policy) for s in stimuli] for policy in pol.OTRF_POLICIES}
    escalate = {policy: np.array([d.escalate for d in rows], dtype=bool) for policy, rows in decisions.items()}
    keep = np.array([s["stimulus_id"] not in OTRF_LABEL_NOISE_SUSPECTS for s in stimuli], dtype=bool)
    recovery: dict[str, Any] = {}
    for base, condition, _ in otrf.STAGE2:
        issued = [records[(s["stimulus_id"], condition)] for s in stimuli if (s["stimulus_id"], condition) in records]
        recovery[condition] = {**proportion(sum(pol.escalates(r) for r in issued), len(issued)), "base": base}
    echo = [echoes(records[(s["stimulus_id"], "s1_injected")],
                   payload_markers(s["injection_payload"], otrf.stage1_messages(s, "s1_clean"))) for s in stimuli]
    return {
        "n_events": len(stimuli),
        "escalation": {
            policy: {"all": proportion(int(values.sum()), values.size),
                     "excluding_suspects": proportion(int(values[keep].sum()), int(keep.sum())),
                     "invalid_decisions": sum(d.invalid for d in decisions[policy])}
            for policy, values in escalate.items()
        },
        "tests": [{"name": name, "family": "exploratory", "first": a, "second": b,
                   **paired_mcnemar(escalate[a], escalate[b])} for name, a, b in OTRF_TESTS],
        "recovery_after_dismissal": recovery,
        "payload_echo_injected": proportion(sum(echo), len(echo)),
    }


def latency(records: pol.Records) -> dict[str, float | int]:
    elapsed = np.array([float(r.get("elapsed_seconds") or 0.0) for r in records.values()])
    if elapsed.size == 0:
        return {"calls": 0, "median_s": 0.0, "p90_s": 0.0, "total_hours": 0.0}
    return {"calls": int(elapsed.size), "median_s": float(np.median(elapsed)),
            "p90_s": float(np.quantile(elapsed, 0.9)), "total_hours": float(elapsed.sum() / 3600.0)}


def select_stimuli(mode: str, all_stimuli: list[dict], records: pol.Records) -> list[dict]:
    if mode == "official":
        return all_stimuli
    return [s for s in all_stimuli if (s["stimulus_id"], "s1_clean") in records]


def replicate_per_stratum(mode: str, responses: Path, model: str, records: pol.Records, stimuli: list[dict]) -> int:
    if mode == "smoke":
        strata = Counter(s["stratum"] for s in stimuli if (s["stimulus_id"], salad.REPLICATE_CONDITION) in records)
        return max(strata.values(), default=0)
    tag = model.replace(":", "-").replace(".", "-")
    manifest = responses.with_name(f"{responses.stem}_manifest_{tag}.json")
    if not manifest.exists():  # replicate records then count as unexpected, so G4 fails closed
        logger.warning("no run manifest for %s at %s", model, manifest)
        return 0
    return int(json.loads(manifest.read_text(encoding="utf-8"))["replicate_per_stratum"])


def gate_g4(args: argparse.Namespace, salad_all: list[dict], otrf_all: list[dict]) -> tuple[dict, dict, dict, dict]:
    wanted = set(args.models) if args.models else None
    keep = lambda rows: [row for row in rows if wanted is None or row["model"] in wanted]  # noqa: E731
    salad_by_model, salad_dups = group_by_model(keep(load_jsonl(args.salad_responses)))
    otrf_by_model, otrf_dups = group_by_model(keep(load_jsonl(args.otrf_responses)))
    for model in sorted((wanted or set()) - set(salad_by_model) - set(otrf_by_model)):
        salad_by_model[model] = {}  # a requested model with no records must fail G4, not vanish
    salad_ids, otrf_ids = {s["stimulus_id"]: s for s in salad_all}, {s["stimulus_id"]: s for s in otrf_all}
    report: dict[str, Any] = {}
    selected: dict[str, tuple[list[dict], list[dict]]] = {}
    for model in dict.fromkeys(list(salad_by_model) + list(otrf_by_model)):
        s_records, o_records = salad_by_model.get(model, {}), otrf_by_model.get(model, {})
        s_stimuli = select_stimuli(args.mode, salad_all, s_records)
        o_stimuli = select_stimuli(args.mode, otrf_all, o_records)
        replicates = replicate_per_stratum(args.mode, args.salad_responses, model, s_records, s_stimuli)
        s_check = pol.completeness(s_records, pol.salad_planned(s_stimuli, s_records, replicates), salad_ids,
                                   pol.salad_expected_messages, without_assistant(salad.leaked_fields))
        o_check = pol.completeness(o_records, pol.otrf_planned(o_stimuli, o_records), otrf_ids,
                                   pol.otrf_expected_messages, without_assistant(otrf.leaked_terms))
        replicate_ids = [s["stimulus_id"] for s in salad.replicate_subset(s_stimuli, replicates)]
        s_check.update(duplicate_records=salad_dups[model], analysed_alerts=len(s_stimuli),
                       replicate_per_stratum=replicates, determinism=pol.determinism(s_records, replicate_ids),
                       parse_validity=pol.parse_validity(s_records))
        o_check.update(duplicate_records=otrf_dups[model], analysed_events=len(o_stimuli),
                       parse_validity=pol.parse_validity(o_records))
        s_check["passed"] = bool(s_check["passed"] and salad_dups[model] == 0 and s_stimuli)
        o_check["passed"] = bool(o_check["passed"] and otrf_dups[model] == 0 and o_stimuli)
        report[model] = {"salad": s_check, "otrf": o_check}
        selected[model] = (s_stimuli, o_stimuli)
    return report, selected, salad_by_model, otrf_by_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--salad-responses", type=Path, required=True)
    parser.add_argument("--otrf-responses", type=Path, required=True)
    parser.add_argument("--mode", choices=("official", "smoke"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--models", nargs="+", default=None, help="restrict to these models (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    salad_path = args.paper_root / "data/stimuli/salad_llm_stimuli.jsonl"
    otrf_path = args.paper_root / "data/stimuli/otrf_high_confidence_stimuli.jsonl"
    salad_all, otrf_all = load_jsonl(salad_path), load_jsonl(otrf_path)
    population = json.loads(salad_path.with_name("salad_llm_stimuli_manifest.json").read_text())["population_strata"]
    g4, selected, salad_by_model, otrf_by_model = gate_g4(args, salad_all, otrf_all)
    models = list(g4)

    g4_passed = bool(models) and all(entry["salad"]["passed"] and entry["otrf"]["passed"] for entry in g4.values())
    label = "OFFICIAL" if args.mode == "official" else "SMOKE (validates code only; never evidence)"
    prefix = "" if args.mode == "official" else "SMOKE_"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    g4_path = args.output_dir / f"{prefix}triage_completeness_report.json"
    g4_path.write_text(json.dumps({"label": label, "passed": g4_passed, "models": g4}, indent=2, sort_keys=True,
                                  default=json_default) + "\n", encoding="utf-8")
    if not g4_passed:
        logger.error("G4 failed; see %s. No rates were computed.", g4_path)
        raise SystemExit(1)

    inputs = (salad_path, otrf_path, args.salad_responses, args.otrf_responses)
    analysis: dict[str, Any] = {
        "label": label, "g4_passed": g4_passed, "models": models,
        "inputs": {p.resolve().name: sha256_file(p) for p in inputs},
        "design": {"population_strata": population, "bootstrap_resamples": args.resamples,
                   "bootstrap_seed": BOOTSTRAP_SEED, "invalid_answer_rule": "escalate (fail-safe)",
                   "otrf_label_noise_suspects": OTRF_LABEL_NOISE_SUSPECTS},
        "policies": pol.POLICY_LABELS, "salad": {}, "otrf": {}, "latency": {},
    }
    boot_by_model: dict[str, dict] = {}
    count_cache: dict[tuple[str, ...], np.ndarray] = {}
    for model in models:
        s_stimuli, o_stimuli = selected[model]
        s_records, o_records = salad_by_model.get(model, {}), otrf_by_model.get(model, {})
        design = StratifiedDesign([s["stratum"] for s in s_stimuli], population)
        key = tuple(design.strata.tolist())
        if key not in count_cache:  # same alert set -> same resamples, so model differences pair
            count_cache[key] = design.resample_counts(args.resamples, BOOTSTRAP_SEED)
        arrays = pol.salad_policy_arrays(s_records, s_stimuli)
        rates, boot = salad_rates(design, count_cache[key], arrays)
        boot_by_model[model] = boot
        analysis["salad"][model] = {
            "alerts": len(s_stimuli), "rates": rates, "differences": salad_differences(boot),
            "tests": salad_tests(design, arrays), "stage2": stage2_rates(s_records, s_stimuli),
            "injection": injection_and_contamination(s_records, s_stimuli),
            "determinism": g4[model]["salad"]["determinism"],
        }
        analysis["otrf"][model] = analyze_otrf_model(o_records, o_stimuli)
        analysis["latency"][model] = {"salad": latency(s_records), "otrf": latency(o_records)}

    confirmatory = [t for model in models for t in analysis["salad"][model]["tests"] if t["family"] == "confirmatory"]
    for test, adjusted in zip(confirmatory, holm_adjust([t["exact_p"] for t in confirmatory])):
        test["holm_p"] = adjusted
    analysis["confirmatory_family"] = {"size": len(confirmatory), "method": "Holm over models x hypotheses H1 to H5"}
    total = sum(population.values())
    analysis["loss"] = loss_analysis(boot_by_model, salad_prevalence=sum(population[s] for s in HOSTILE) / total)
    out_json = args.output_dir / f"{prefix}triage_study_analysis.json"
    out_json.write_text(json.dumps(analysis, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    write_tables(analysis, args.output_dir, prefix)
    logger.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
