"""Analysis of the v3 extension study: gate G4-ext, then items 1 to 6.

Reads the extension store and, read-only, the stores the extension continued from: the locked SALAD
store for the four locked models and the added model's own locked-design store. Writes
ext_g4_report.json and llm_ext_analysis.json. No rate is computed unless G4-ext passes for every
model; official mode also requires that the locked analyses it extends passed G4 and that the added
model is the one the screening rule selected.

Two guards tie the extension to the locked numbers. Every locked (world, policy) decision is
recomputed by the locked salad_policy_arrays from the base store alone and must equal the
extension's decision; in official mode the locked part of the recomputed rates must also equal the
locked analysis JSON exactly. A failure of either stops the run.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import analyze_triage_study as base
import llm_policies as pol
import salad_triage_runner as salad
from ext_design import EXT_CONFIRMATORY, EXT_EXPLORATORY, EXTENSION_SEED
from ext_metrics import (
    EXTENDED_DIFFERENCES,
    ORIGINAL_DIFFERENCES,
    differences,
    discrimination,
    exact_mirror,
    ext_rates,
    extension_tests,
    f2_injection,
    locked_view,
    stage2_reversal,
)
from ext_policies import POLICY_LABELS, ext_completeness, ext_policy_arrays
from llm_loss import loss_analysis
from llm_stats import StratifiedDesign

logger = logging.getLogger(__name__)

V3 = Path(".")
LOCKED_STIMULI = Path("data/stimuli/salad_llm_stimuli.jsonl")
BENIGN_STIMULI = Path("data/stimuli/salad_llm_ext_benign_stimuli.jsonl")
EXT_MANIFEST = Path("data/stimuli/salad_llm_ext_stimuli_manifest.json")
SELECTION = Path("data/screening/screen_selection.json")
DEFAULTS = {
    "ext_responses": Path("data/responses_extension/salad_ext_responses.jsonl"),
    "base_responses": Path("data/responses/salad_responses.jsonl"),
    "added_responses": Path("data/responses_extension/salad_responses_added.jsonl"),
    "locked_analysis": Path("results/triage/triage_study_analysis.json"),
    "added_analysis": Path("results/triage_added/triage_study_analysis.json"),
    "output_dir": Path("results/triage_extension"),
}
Records = dict[tuple[str, str], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("official", "smoke"), required=True)
    parser.add_argument("--added-model", required=True, help="model the screening rule selected, or none")
    parser.add_argument("--limit-per-stratum", type=int, default=None, help="smoke checks only")
    parser.add_argument("--resamples", type=int, default=base.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--models", nargs="+", default=None, help="smoke only (official: the locked models)")
    for name in DEFAULTS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=None)
    args = parser.parse_args()
    for name, default in DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, args.paper_root / default)
    args.added_model = None if args.added_model.lower() == "none" else args.added_model
    if args.mode == "official" and (args.limit_per_stratum is not None or args.models):
        parser.error("official mode analyses every alert of every locked model")
    return args


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def official_preconditions(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The locked analyses this run extends, after checking they are the official, gated ones."""
    selected = read_json(args.paper_root / SELECTION)["selected"] or None
    if selected != args.added_model:
        raise SystemExit(f"--added-model {args.added_model!r} is not the screening selection {selected!r}")
    analyses: list[dict[str, Any] | None] = []
    for path, wanted in ((args.locked_analysis, None), (args.added_analysis, args.added_model)):
        if path == args.added_analysis and args.added_model is None:
            analyses.append(None)
            continue
        analysis = read_json(path)
        g4 = read_json(path.with_name("triage_completeness_report.json"))
        if analysis.get("label") != "OFFICIAL" or not analysis.get("g4_passed") or not g4.get("passed"):
            raise SystemExit(f"{path} is not an OFFICIAL analysis that passed G4")
        if wanted is not None and analysis.get("models") != [wanted]:
            raise SystemExit(f"{path} covers {analysis.get('models')}, not [{wanted!r}]")
        analyses.append(analysis)
    return analyses[0] or {}, analyses[1]


def load_store(path: Path, models: set[str]) -> tuple[dict[str, Records], Counter]:
    rows = base.load_jsonl(path) if path.exists() else []
    grouped, duplicates = base.group_by_model([row for row in rows if row["model"] in models])
    return grouped, duplicates


def gate_g4_ext(
    models: list[str], ext_by_model: Mapping[str, Records], ext_dups: Counter,
    base_by_model: Mapping[str, Records], corpora: Mapping[str, list[dict]], stimuli_by_id: Mapping[str, dict],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for model in models:
        ext_records, base_records = ext_by_model.get(model, {}), base_by_model.get(model, {})
        check = ext_completeness(ext_records, base_records, corpora, stimuli_by_id,
                                 base.without_assistant(salad.leaked_fields))
        check.update(duplicate_records=ext_dups[model], base_records=len(base_records),
                     parse_validity=pol.parse_validity(ext_records))
        check["passed"] = bool(check["passed"] and ext_dups[model] == 0 and ext_records and base_records)
        report[model] = check
    return report


def guard_locked_arrays(model: str, arrays: Mapping[str, Any], base_records: Records, stimuli: list[dict]) -> None:
    """Every locked decision, recomputed from the base store alone by the locked code, is unchanged."""
    locked = pol.salad_policy_arrays(base_records, stimuli)
    for world in pol.WORLDS:
        for policy in pol.SALAD_POLICIES:
            for field in ("escalate", "invalid", "contested"):
                if not np.array_equal(arrays[world][policy][field], locked[world][policy][field]):
                    raise RuntimeError(f"{model}: extension decision {policy}@{world} {field} differs from the "
                                       "locked decision; no extension number can be reported")


def normalized(node: Any) -> Any:
    return json.loads(json.dumps(node, sort_keys=True, default=base.json_default))


def differing_leaves(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [leaf for key in sorted(set(a) | set(b)) for leaf in differing_leaves(a.get(key), b.get(key),
                                                                                       f"{path}/{key}")]
    return [] if a == b else [path]


def guard_locked_rates(model: str, rates: Mapping[str, Any], locked_analysis: Mapping[str, Any]) -> None:
    """Official mode: the locked part of the recomputed rates equals the locked analysis exactly."""
    expected = locked_analysis["salad"][model]["rates"]
    diff = differing_leaves(normalized(locked_view(rates)), expected)
    if diff:
        raise RuntimeError(f"{model}: recomputed locked rates differ from the locked analysis at {diff[:5]}")


class Resamples:
    """One resample matrix per (seed, alert set), so models analysed on the same alerts pair."""

    def __init__(self, resamples: int) -> None:
        self.resamples = resamples
        self.cache: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}

    def counts(self, design: StratifiedDesign, seed: int) -> np.ndarray:
        key = (seed, tuple(design.strata.tolist()))
        if key not in self.cache:
            self.cache[key] = design.resample_counts(self.resamples, seed)
        return self.cache[key]


def analyse_model(
    model: str, ext_records: Records, base_records: Records, original: list[dict], extended: list[dict],
    population: Mapping[str, int], resamples: Resamples, applies: bool, locked_analysis: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple], dict[str, tuple], dict[str, Any]]:
    merged: Records = {**base_records, **ext_records}
    design_o = StratifiedDesign([s["stratum"] for s in original], population)
    arrays_o = ext_policy_arrays(merged, original)
    guard_locked_arrays(model, arrays_o, base_records, original)
    rates_o, boot_o = ext_rates(design_o, resamples.counts(design_o, base.BOOTSTRAP_SEED), arrays_o)
    if locked_analysis is not None:
        guard_locked_rates(model, rates_o, locked_analysis)

    design_e = StratifiedDesign([s["stratum"] for s in extended], population)
    arrays_e = ext_policy_arrays(merged, extended)
    rates_e, boot_e = ext_rates(design_e, resamples.counts(design_e, EXTENSION_SEED), arrays_e)

    original_block = {
        "alerts": len(original), "rates": rates_o, "differences": differences(boot_o, ORIGINAL_DIFFERENCES),
        "f2_injection": f2_injection(merged, original),
        "discrimination": discrimination(design_o, arrays_o, boot_o, applies),
    }
    extended_block = {
        "alerts": len(extended), "rates": rates_e, "differences": differences(boot_e, EXTENDED_DIFFERENCES),
        "stage2": stage2_reversal(merged, extended),
    }
    designs = {"original": (design_o, arrays_o), "extended": (design_e, arrays_e)}
    return original_block, extended_block, boot_o, boot_e, designs


def f2_loss_boot(boot: Mapping[tuple[str, str], tuple]) -> dict[tuple[str, str], tuple]:
    """The locked loss inputs with the attacked world read on payload family 2.

    Called on the extended sample, as loss_extended is, so the two loss blocks differ in the
    payload family alone: benign alerts are never injected, so both read the same false escalation.
    """
    return {(world, policy): boot[("attacked_f2" if world == "attacked" else world, policy)]
            for world in pol.WORLDS for policy in pol.SALAD_POLICIES}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.paper_root
    locked_all, benign_all = base.load_jsonl(root / LOCKED_STIMULI), base.load_jsonl(root / BENIGN_STIMULI)
    population = read_json((root / LOCKED_STIMULI).with_name("salad_llm_stimuli_manifest.json"))["population_strata"]
    if read_json(root / EXT_MANIFEST)["population_strata"] != population:
        raise SystemExit("the extension stimuli were weighted against a different population")
    stimuli_by_id = {s["stimulus_id"]: s for s in locked_all + benign_all}
    if len(stimuli_by_id) != len(locked_all) + len(benign_all):
        raise SystemExit("extension stimulus ids collide with locked stimulus ids")

    if args.mode == "official":
        locked_analysis, added_analysis = official_preconditions(args)
        models = list(locked_analysis["models"]) + ([args.added_model] if args.added_model else [])
        limit = None
    else:
        locked_analysis = read_json(args.locked_analysis) if args.locked_analysis.exists() else {}
        added_analysis = read_json(args.added_analysis) if args.added_model and args.added_analysis.exists() else None
        models = list(args.models or sorted(set(row["model"] for row in base.load_jsonl(args.ext_responses))))
        limit = args.limit_per_stratum
    if args.added_model and args.added_model not in models:
        models.append(args.added_model)

    original = salad.select_stimuli(locked_all, limit)
    benign = salad.select_stimuli(benign_all, limit)
    extended = original + benign
    corpora = {"mplacebo": original, "f2": original, "benign": benign}

    wanted = set(models)
    ext_by_model, ext_dups = load_store(args.ext_responses, wanted)
    locked_by_model, _ = load_store(args.base_responses, wanted)
    added_by_model, _ = load_store(args.added_responses, {args.added_model} if args.added_model else set())
    if args.added_model and args.added_model in locked_by_model:
        raise SystemExit(f"{args.added_model} has records in the locked store; it cannot be the added model")
    base_by_model = {m: (added_by_model if m == args.added_model else locked_by_model).get(m, {}) for m in models}

    g4 = gate_g4_ext(models, ext_by_model, ext_dups, base_by_model, corpora, stimuli_by_id)
    g4_passed = bool(models) and all(entry["passed"] for entry in g4.values())
    official = args.mode == "official"
    label = "OFFICIAL" if official else "SMOKE (validates code only; never evidence)"
    prefix = "" if official else "SMOKE_"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    g4_path = args.output_dir / f"{prefix}extension_completeness_report.json"
    g4_path.write_text(json.dumps({"label": label, "passed": g4_passed, "added_model": args.added_model,
                                   "models": g4}, indent=2, sort_keys=True, default=base.json_default) + "\n",
                       encoding="utf-8")
    if not g4_passed:
        logger.error("G4-ext failed; see %s. No rates were computed.", g4_path)
        raise SystemExit(1)

    resamples = Resamples(args.resamples)
    analysis: dict[str, Any] = {"original": {}, "extended": {}, "latency": {}}
    tests_input: dict[str, dict[str, tuple]] = {}
    boot_e_by_model: dict[str, dict] = {}
    for model in models:
        reference = added_analysis if model == args.added_model else locked_analysis
        original_block, extended_block, _, boot_e, designs = analyse_model(
            model, ext_by_model[model], base_by_model[model], original, extended, population, resamples,
            applies=model == args.added_model, locked_analysis=reference if official else None)
        analysis["original"][model], analysis["extended"][model] = original_block, extended_block
        analysis["latency"][model] = base.latency(ext_by_model[model])
        tests_input[model] = designs
        boot_e_by_model[model] = boot_e  # both loss blocks read the extended sample

    tests, family = extension_tests(tests_input, args.added_model)
    expected_size = 35 if args.added_model else 24
    if official and family["size"] != expected_size:
        raise RuntimeError(f"extension family has {family['size']} pairs, the lock fixes {expected_size}")
    prevalence = sum(population[s] for s in pol.HOSTILE_STRATA) / sum(population.values())
    inputs = [root / LOCKED_STIMULI, root / BENIGN_STIMULI, args.ext_responses, args.base_responses]
    inputs += [p for p in (args.locked_analysis, root / SELECTION) if p.exists()]
    if args.added_model:
        inputs += [args.added_responses, args.added_analysis]
    analysis.update({
        "label": label, "g4_passed": g4_passed, "models": models, "added_model": args.added_model,
        "inputs": {str(p.resolve().relative_to(root.resolve())): base.sha256_file(p) for p in inputs},
        "design": {
            "population_strata": population, "bootstrap_resamples": args.resamples,
            "bootstrap_seed": base.BOOTSTRAP_SEED, "extension_seed": EXTENSION_SEED,
            "original_alerts": len(original), "benign_extension_alerts": len(benign),
            "extended_alerts": len(extended), "invalid_answer_rule": "escalate (fail-safe)",
            "zero_count_bound": "Clopper-Pearson exact two-sided 95 percent interval",
        },
        "policies": POLICY_LABELS,
        "hypotheses": {test.test_id: test.label for test in EXT_CONFIRMATORY + EXT_EXPLORATORY},
        "tests": tests, "extension_family": family,
        "exact": {name: exact_mirror({"salad": a.get("salad", {}), "otrf": a.get("otrf", {})})
                  for name, a in (("locked", locked_analysis), ("added", added_analysis)) if a},
        "loss_extended": loss_analysis(boot_e_by_model, salad_prevalence=prevalence),
        "loss_f2": loss_analysis({m: f2_loss_boot(b) for m, b in boot_e_by_model.items()},
                                 salad_prevalence=prevalence),
    })
    out_json = args.output_dir / f"{prefix}extension_study_analysis.json"
    out_json.write_text(json.dumps(analysis, indent=2, sort_keys=True, default=base.json_default) + "\n",
                        encoding="utf-8")
    logger.info("wrote %s (%d models, family of %d)", out_json, len(models), family["size"])


if __name__ == "__main__":
    main()
