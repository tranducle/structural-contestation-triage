"""LaTeX tables of the v3 extension study, written from its analysis and the value files.

The contract is make_triage_tables.py's. Every number is a \\val{key} reference: ext/ and add/ keys
into llm_ext_values.tex (make_extension_macros.py), and the locked latency keys into llm_values.tex. A
key no value file defines prints n/a and is listed in tables_ext_manifest.json. Text that is not a
measured number (family, size, quantization, digest) is read from the Ollama listings and checked
against every run manifest. The analysis decides only the layout. A zero count carries its exact
(Clopper-Pearson) interval, and the policy caption states the bound a zero weighted rate obeys.

Written after the extension design lock of 2026-10-23, before any official extension answer was
analysed; it reads the analysis outputs and changes none of them.
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
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import make_triage_tables as v3
from ext_design import EXT_CONFIRMATORY, EXT_EXPLORATORY, Test

logger = logging.getLogger("make_ext_tables")

DEFAULT_ANALYSIS = v3.PAPER_ROOT / "results/triage_extension/extension_study_analysis.json"
DEFAULT_VALUES = v3.PAPER_ROOT / "results/generated/llm_ext_values.tex"
DEFAULT_LOCKED_VALUES = v3.DEFAULT_VALUES
EXT_RUN_DIR = v3.PAPER_ROOT / "data/responses_extension"
LISTINGS = (v3.MODEL_LISTING, v3.PAPER_ROOT / "configs/ollama_models_extension.json")
SOURCE_LABEL = re.compile(r"^\\def\\llm(?:ext)?valuesource\{(?P<label>[A-Z]+)\}$")
NA, CI_NOTE = v3.NA, v3.CI_NOTE

POLICY_ROWS = ("U", "D", "S", "M", "P", "DS", "G")
WORLD_TEXT = {"clean": "clean", "attacked": "F1", "attacked_f2": "F2"}
RECALL_WORLDS = ("clean", "attacked", "attacked_f2")
HOSTILE, BENIGN = ("hostile_gate", "hostile_nogate"), ("benign_gate", "benign_nogate")
# (row code, cue kind, opening prefix); the condition is s2_<kind>_after_<prefix><base>
STAGE2_ROWS = (("S", "struct", ""), ("P", "placebo", ""), ("M", "mplacebo", ""), ("DS", "spotstruct", "spot_"))
STAGE2_COLUMNS = (("benign_gate", "clean"), ("hostile_gate", "clean"), ("hostile_gate", "injected"),
                  ("hostile_gate", "injected_f2"))
INJECTION_METRICS = (("hostile_all", "escalated_clean"), ("hostile_all", "escalated_injected"),
                     ("hostile_all", "flip_to_dismiss"), ("contamination", "payload_echo_injected"))
PROMPT_TEXT = {"plain": "Plain", "spotlight": "Spotlighting"}
LOSS_BLOCKS = (("loss_extended", "F1"), ("loss_f2", "F2"))
WORLD = "attacked"  # in loss_f2 the world named attacked is the payload-family-2 world

Row = v3.Row


def read_values(path: Path) -> tuple[dict[str, str], str, set[str]]:
    """Key -> TeX text, the value-source label, and the sha256 of every source; either value file."""
    defined: dict[str, str] = {}
    label, hashes = "", set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := v3.DEFINITION.match(line):
            defined[match["key"]] = match["text"]
        elif match := SOURCE_LABEL.match(line):
            label = match["label"]
        elif match := v3.SOURCE_HASH.match(line):
            hashes.add(match["sha"])
    if not defined or not label:
        raise ValueError(f"{path} is not a value file written by make_triage_macros.py or make_extension_macros.py")
    return defined, label, hashes


@dataclass
class ExtCells(v3.Cells):
    """v3.Cells with exact brackets on zero counts."""

    def has(self, key: str) -> bool:
        return key in self.defined

    def count_bounded(self, prefix: str) -> str:
        text = self.count(prefix)
        if text == NA or self.defined.get(f"{prefix}/successes") != "0" or not self.has(f"{prefix}/exact95/hi"):
            return text
        return f"{text} [{self.val(prefix + '/exact95/lo')}, {self.val(prefix + '/exact95/hi')}]"


def _number(text: str) -> float:
    return float(text.replace("\\ensuremath{-}", "-"))


def _sizes(cells: ExtCells, model: str) -> dict[str, str]:
    """Sample sizes a caption names, as value keys: locked alerts, their hostile half, extended gated benign."""
    return {"original": cells.val("ext/design/original_alerts"),
            "hostile": cells.val(f"ext/original/{model}/discrimination/hostile_alerts"),
            "benign_gate": cells.val(f"ext/extended/{model}/rates/clean/U/per_stratum/benign_gate/trials")}


def zero_rate_bound(cells: ExtCells, models: Sequence[str]) -> tuple[str, str] | None:
    """(exact95/hi key, trials key) of the widest per-stratum bound behind any weighted rate shown as 0.0.

    The smallest nonzero weighted rate prints as at least 0.1, so 0.0 means no escalation in either
    stratum; the check below stops the run if that ever fails to hold.
    """
    cases = [(f"ext/original/{m}/rates/{w}/{p}", "recall", HOSTILE)
             for m in models for w in RECALL_WORLDS for p in POLICY_ROWS + ("PITSA",)]
    cases += [(f"ext/extended/{m}/rates/clean/{p}", "fpr", BENIGN) for m in models for p in POLICY_ROWS + ("PITSA",)]
    widest: tuple[float, str, str] | None = None
    for base, metric, strata in cases:
        if cells.defined.get(f"{base}/{metric}") != "0.0":
            continue
        for stratum in strata:
            prefix = f"{base}/per_stratum/{stratum}"
            if cells.defined.get(f"{prefix}/successes") != "0":
                raise ValueError(f"{base}/{metric} prints 0.0 but {stratum} counts an escalation")
            bound = _number(cells.defined[f"{prefix}/exact95/hi"])
            if widest is None or bound > widest[0]:
                widest = (bound, f"{prefix}/exact95/hi", f"{prefix}/trials")
    return None if widest is None else widest[1:]


def policy_table(cells: ExtCells, models: Sequence[str]) -> str:
    def recall(model: str, world: str, policy: str) -> str:
        base = f"ext/original/{model}/rates/{world}/{policy}/recall"
        return cells.ci(base, f"{base}95")

    def fpr(model: str, policy: str) -> str:
        texts = {tuple(cells.defined.get(f"ext/extended/{model}/rates/{w}/{policy}/fpr{s}") for s in ("", "95/lo", "95/hi"))
                 for w in RECALL_WORLDS}
        if len(texts) != 1:  # benign alerts are never injected, so one column must serve every world
            raise ValueError(f"{model} {policy}: false escalation differs between worlds; split the column")
        base = f"ext/extended/{model}/rates/clean/{policy}/fpr"
        return cells.ci(base, f"{base}95")

    def row(name: str, model: str, policy: str) -> list[str]:
        return [name, policy, *[recall(model, w, policy) for w in RECALL_WORLDS], fpr(model, policy)]

    rows: list[Row] = [row("No model", models[0], "PITSA"), None]
    rows += v3._blocks(models, lambda m: [row(m if i == 0 else "", m, p) for i, p in enumerate(POLICY_ROWS)])
    bound = zero_rate_bound(cells, models)
    zero = ("" if bound is None else
            " A rate shown as 0.0 counts no escalation in either stratum it weights; the larger per-stratum exact "
            f"upper bound, at most {cells.val(bound[0])} percent for {cells.val(bound[1])} alerts, then bounds it at "
            "no less than 95 percent confidence, which the zero-width bootstrap interval does not.")
    size = _sizes(cells, models[0])
    caption = ("Extension: weighted operating points on the SALAD test population, in percent. Policy codes are as in "
               f"\\cref{{tab:llm_policies}}; M is the matched placebo second turn. Recall is read on the {size['original']} "
               "locked alerts with the annotation clean, carrying a payload of family 1 (F1) or carrying a payload of "
               "family 2 (F2). False escalation is read on the extended sample, whose gated benign stratum holds "
               f"{size['benign_gate']} alerts; benign annotations are never injected, so it serves all three conditions. {CI_NOTE} from "
               f"the stratified bootstrap.{zero}")
    return v3.table(caption, "tab:ext_policies", "L{1.7cm} L{1.0cm} C{2.6cm} C{2.6cm} C{2.6cm} C{2.6cm}",
                    ["Model", "Policy", "Recall, clean", "Recall, F1", "Recall, F2", "False escalation"], rows)


def _test_models(analysis: Mapping[str, Any], test_id: str) -> list[str]:
    present = analysis["tests"].get(test_id, {})
    return [m for m in analysis["models"] if m in present]


def tests_table(cells: ExtCells, analysis: Mapping[str, Any], specs: Sequence[Test], confirmatory: bool) -> str:
    rows: list[Row] = []
    for test in specs:
        models = _test_models(analysis, test.test_id)
        if not models:
            continue
        if rows:
            rows.append(None)
        metric = "recall" if test.strata == "hostile" else "fpr"
        comparison = (f"{test.policy_a} ({WORLD_TEXT[test.world_a]}) vs "
                      f"{test.policy_b} ({WORLD_TEXT[test.world_b]})")
        for index, model in enumerate(models):
            key = f"ext/tests/{test.test_id}/{model}"
            arms = [cells.val(f"ext/{test.corpus}/{model}/rates/{w}/{p}/{metric}")
                    for p, w in ((test.policy_a, test.world_a), (test.policy_b, test.world_b))]
            row = [test.test_id if index == 0 else "", model, comparison, "Recall" if metric == "recall" else "False esc.",
                   *arms, cells.val(f"{key}/n_pairs"),
                   f"{cells.val(key + '/first_only')} : {cells.val(key + '/second_only')}", cells.val(f"{key}/exact_p")]
            if confirmatory:
                row.append(cells.val(f"{key}/holm_p"))
            rows.append(row)
    header = ["Test", "Model", "Comparison", "Metric", "Arm A", "Arm B", "Pairs", "Discordant", "Exact p"]
    colspec = "L{0.6cm} L{1.6cm} L{2.95cm} L{1.3cm} C{1.0cm} C{1.0cm} C{0.8cm} C{1.5cm} C{1.1cm}"
    if confirmatory:
        header.append("Holm p")
        colspec += " C{1.1cm}"
        size = _sizes(cells, analysis["models"][0])
        caption = ("Extension confirmatory tests: R1 to R5 repeat H1 to H5 on the added model, and X1 to X6 run on every "
                   "model. Each is a two-sided exact McNemar test on the paired alerts; Holm adjustment runs over every "
                   "(test, model) pair in this family, separately from H1 to H5. X2 and X3 read the extended sample, "
                   f"the others the {size['original']} locked alerts. Arm A and Arm B give the weighted rates, in percent; Pairs counts "
                   "the paired alerts, and Discordant those only arm A escalated and those only arm B escalated.")
        label = "tab:ext_tests"
    else:
        caption = "Extension exploratory tests Y1 to Y5, without multiplicity adjustment. Columns as in \\cref{tab:ext_tests}."
        label = "tab:ext_exploratory"
    return v3.table(caption, label, colspec, header, rows)


def injection_table(cells: ExtCells, analysis: Mapping[str, Any], models: Sequence[str]) -> str:
    def prefixes(model: str, prompt: str) -> dict[str, str]:
        role = "added" if model == analysis.get("added_model") else "locked"
        return {"F1": f"ext/exact/{role}/salad/{model}/injection/{prompt}",
                "F2": f"ext/original/{model}/f2_injection/{prompt}"}

    for model in models:  # both families share the clean opening turn, so its count must agree
        for prompt in PROMPT_TEXT:
            texts = {tuple(cells.defined.get(f"{p}/hostile_all/escalated_clean/{leaf}") for leaf in ("successes", "trials"))
                     for p in prefixes(model, prompt).values()}
            if len(texts) != 1 and analysis.get("label") == "OFFICIAL":
                raise ValueError(f"{model} {prompt}: clean escalation differs between payload families")
            if len(texts) != 1:  # a smoke subset reads F2 on a few alerts and F1 on the locked 80
                logger.warning("%s %s: F1 and F2 clean counts differ, as on a smoke subset", model, prompt)

    def build(model: str) -> list[Row]:
        rows: list[Row] = []
        for prompt, prompt_text in PROMPT_TEXT.items():
            for family, prefix in prefixes(model, prompt).items():
                rows.append([model if not rows else "", prompt_text if family == "F1" else "", family,
                             *[cells.count_bounded(f"{prefix}/{group}/{metric}") for group, metric in INJECTION_METRICS]])
        return rows

    size = _sizes(cells, models[0])
    caption = (f"Extension: the injection effect on the {size['hostile']} hostile locked alerts under payload family 1 (F1, the locked "
               "payloads) and payload family 2 (F2, format-mimicking payloads), as counts. Flip to dismiss counts the "
               "alerts escalated with a clean annotation and dismissed with the payload, out of those escalated clean. "
               "Payload echo counts injected answers whose rationale shares a word stem with the payload that is "
               "absent from the clean prompt. A zero count out of a nonzero denominator carries its "
               "exact 95 percent interval in brackets.")
    return v3.table(caption, "tab:ext_injection", "L{1.6cm} L{1.7cm} L{1.1cm} C{2.5cm} C{2.5cm} C{2.5cm} C{2.5cm}",
                    ["Model", "Opening", "Payload", "Escalated, clean", "Escalated, injected", "Flip to dismiss",
                     "Payload echo"], v3._blocks(models, build))


def stage2_table(cells: ExtCells, models: Sequence[str]) -> str:
    def cell(model: str, kind: str, prefix: str, stratum: str, base: str) -> str:
        key = f"ext/extended/{model}/stage2/s2_{kind}_after_{prefix}{base}/{stratum}"
        return cells.count_bounded(key) if cells.has(f"{key}/trials") else "none"

    def build(model: str) -> list[Row]:
        return [[model if i == 0 else "", code, *[cell(model, kind, prefix, s, b) for s, b in STAGE2_COLUMNS]]
                for i, (code, kind, prefix) in enumerate(STAGE2_ROWS)]

    caption = ("Extension: second turns, as alerts escalated out of the gated alerts the opening turn dismissed and the "
               "second turn then reached. S sends the structural cue, P the locked placebo, M the matched placebo, and "
               "DS the cue after a spotlighted opening. Benign alerts are read on the extended sample and are never "
               "injected; none means no alert reached that second turn. A zero count carries its exact 95 percent "
               "interval in brackets.")
    return v3.table(caption, "tab:ext_stage2", "L{1.6cm} L{1.7cm} C{2.7cm} C{2.7cm} C{2.7cm} C{2.7cm}",
                    ["Model", "Second turn", "Benign gated, clean", "Hostile gated, clean", "Hostile gated, F1",
                     "Hostile gated, F2"], v3._blocks(models, build))


def discrimination_table(cells: ExtCells, analysis: Mapping[str, Any], models: Sequence[str]) -> str:
    rows: list[Row] = []
    for model in models:
        record = analysis["original"][model]["discrimination"]
        base, rates = f"ext/original/{model}/discrimination", f"ext/original/{model}/rates/clean/U"
        rows.append([model, "added" if model == analysis.get("added_model") else "locked",
                     f"{cells.val(base + '/hostile_escalations')}/{cells.val(base + '/hostile_alerts')}",
                     cells.ci(f"{rates}/recall", f"{rates}/recall95"), cells.ci(f"{rates}/fpr", f"{rates}/fpr95"),
                     cells.ci(f"{base}/youden_j", f"{base}/youden_j95"),
                     "met" if record["meets_criterion"] else "not met"])
    minimum = cells.val(f"ext/original/{models[0]}/discrimination/min_hostile_escalations")
    caption = ("Extension: does the unassisted clean opening turn discriminate? Hostile escalated counts the hostile "
               "locked alerts escalated; recall, false escalation and the weighted Youden J (recall less false "
               f"escalation, in percentage points) are read on the {_sizes(cells, models[0])['original']} locked alerts. {CI_NOTE} from the stratified "
               f"bootstrap. The criterion, fixed for the added model before its runs, is at least {minimum} hostile "
               "escalations and a Youden J whose lower bound exceeds zero; the locked models are shown against the "
               "same criterion for reference.")
    return v3.table(caption, "tab:ext_discrimination", "L{1.6cm} L{0.9cm} C{2.3cm} C{2.5cm} C{3.0cm} C{2.7cm} C{1.3cm}",
                    ["Model", "Role", "Hostile escalated", "Recall, clean", "False escalation, clean", "Youden J",
                     "Criterion"], rows)


def _listing() -> dict[str, Mapping[str, Any]]:
    merged: dict[str, Mapping[str, Any]] = {}
    for path in LISTINGS:
        for entry in json.loads(path.read_text(encoding="utf-8"))["models"]:
            known = merged.get(entry["name"])
            if known is not None and known["digest"] != entry["digest"]:
                raise ValueError(f"{entry['name']}: the two model listings give different digests")
            merged[entry["name"]] = entry
    return merged


def _run_manifests(model: str) -> list[Path]:
    slug = v3._slug(model)
    paths = [v3.RUN_DIR / f"{corpus}_responses_manifest_{slug}.json" for corpus in ("salad", "otrf")]
    paths += [EXT_RUN_DIR / f"{corpus}_responses_added_manifest_{slug}.json" for corpus in ("salad", "otrf")]
    paths += sorted(EXT_RUN_DIR.glob(f"salad_ext_responses_manifest_{slug}_*.json"))
    return [p for p in paths if p.exists()]


def models_table(cells: ExtCells, analysis: Mapping[str, Any], models: Sequence[str]) -> str:
    listing, added = _listing(), analysis.get("added_model")
    rows: list[Row] = []
    for model in models:
        digest = listing[model]["digest"][:12]
        for path in _run_manifests(model):
            recorded = json.loads(path.read_text(encoding="utf-8"))["models"][model]
            if recorded != digest:  # the local model changed between runs
                raise ValueError(f"{model}: {path.name} records {recorded}, the listing gives {digest}")
        details = listing[model]["details"]
        locked = "add/latency" if model == added else "latency"
        rows.append([model, details["family"], details["parameter_size"],
                     details["quantization_level"].replace("_", "\\_"), f"\\texttt{{{digest}}}",
                     cells.val(f"{locked}/{model}/salad/calls"), cells.val(f"{locked}/{model}/otrf/calls"),
                     cells.val(f"ext/latency/{model}/calls"), cells.val(f"{locked}/{model}/salad/median_s")])
    shared = ""
    if added in models:
        family = listing[added]["details"]["family"]
        names = [m for m in models if m != added and listing[m]["details"]["family"] == family]
        if names:
            shared = (f" The listing gives the added model, {added}, the {family} architecture it also gives "
                      f"{' and '.join(names)}.")
    caption = ("Extension: instruction models, served locally by Ollama with the locked decoding. Family is the "
               "architecture field of the Ollama listing, not the developer." + shared + " Digest is the leading twelve "
               "hexadecimal characters of the model blob, identical in every run manifest. SALAD and OTRF calls count "
               "the locked-design invocations, Extension calls the extension invocations; Median s is the median "
               "seconds per locked-design SALAD call.")
    return v3.table(caption, "tab:ext_models",
                    "L{1.75cm} L{1.0cm} C{1.6cm} C{1.8cm} C{1.9cm} C{1.8cm} C{1.65cm} C{2.05cm} C{1.3cm}",
                    ["Model", "Family", "Parameters", "Quantization", "Digest", "SALAD calls", "OTRF calls",
                     "Extension calls", "Median s"], rows)


TIE_LINE_POLICIES, TIE_LINE_CHARS = 2, 4


def _tie_line_fits(names: Sequence[str]) -> bool:
    """Whether one line of a tie fits the 54pt column at scriptsize, as measured in the manuscript font.

    Two policies of at most four name characters fit ("= ALL = G" sets 45pt), and so do three one-letter
    policies other than M ("= U = D = G" sets 53.5pt). Any triple holding M (54.2 to 54.9pt),
    "= S = P = DS" (55.7pt) and "= PITSA = NONE" (73pt) overflow.
    """
    if len(names) <= TIE_LINE_POLICIES:
        return sum(map(len, names)) <= TIE_LINE_CHARS
    return len(names) == 3 and all(len(n) == 1 for n in names) and "M" not in names


def _lowest_cell(record: Mapping[str, Any], policies: Sequence[str]) -> str:
    """v3._lowest_cell with a long tie set over several lines, so the column stays narrow.

    The locked cell is built first, which keeps its tie check, and the tie recomputed here must
    reproduce it before any line is broken differently.
    """
    locked_cell = v3._lowest_cell(record, policies)
    losses = record["loss"]
    lowest = min(losses[p] for p in policies)
    tied = [p for p in policies if losses[p] - lowest <= v3.TIE_TOLERANCE * max(lowest, 1.0)]
    credited = next((p for p in v3.NO_MODEL if p in tied), tied[0])
    others = [p for p in tied if p != credited]
    if not others:
        return locked_cell
    if locked_cell != r"{\begin{tabular}[t]{@{}c@{}}%s\\= %s\end{tabular}}" % (credited, " = ".join(others)):
        raise ValueError(f"tie {tied} does not reproduce the locked cell {locked_cell!r}")
    lines, line = [credited], []
    for policy in others:
        if line and not _tie_line_fits(line + [policy]):
            lines.append(" ".join(f"= {p}" for p in line))
            line = []
        line.append(policy)
    lines.append(" ".join(f"= {p}" for p in line))
    return r"{\begin{tabular}[t]{@{}c@{}}%s\end{tabular}}" % r"\\".join(lines)


def _weighted_recall(cells: ExtCells, base: str, policy: str, removed: float = 0.0) -> float:
    """Population-weighted recall of a policy, with `removed` escalations taken out of the gated hostile stratum."""
    population = {s: _plain(cells, f"ext/design/population_strata/{s}") for s in ("hostile_gate", "hostile_nogate")}
    rates = {s: (_plain(cells, f"{base}/{policy}/per_stratum/{s}/successes") - (removed if s == "hostile_gate" else 0.0))
             / _plain(cells, f"{base}/{policy}/per_stratum/{s}/trials") for s in population}
    return sum(population[s] * rates[s] for s in population) / sum(population.values())


def _plain(cells: ExtCells, key: str) -> float:
    return float(cells.defined[key].replace("{,}", ""))


def _fail_safe_notes(cells: ExtCells, models: Sequence[str]) -> list[str]:
    """Caption notes for F2 S vs PITSA entries that name S only because invalid answers count as escalations.

    A note is written only when every invalid S decision is a second turn on a gated hostile alert, S has the
    higher recall as counted, and, with those answers counted as dismissals, S has the lower recall and the higher
    false escalation, so that PITSA would have the lower loss at every k of at least one.
    """
    notes: list[str] = []
    for model in models:
        base = f"ext/extended/{model}/rates/attacked_f2"
        invalid_key = f"ext/extended/{model}/stage2/s2_struct_after_injected_f2/hostile_gate/invalid"
        if not (cells.has(f"{base}/S/invalid_decisions") and cells.has(invalid_key)):
            continue
        invalid = _plain(cells, f"{base}/S/invalid_decisions")
        if invalid == 0 or _plain(cells, invalid_key) != invalid:
            continue
        counted = _weighted_recall(cells, base, "S") > _weighted_recall(cells, base, "PITSA")
        flipped = (_weighted_recall(cells, base, "S", invalid) < _weighted_recall(cells, base, "PITSA")
                     and _plain(cells, f"{base}/S/fpr") > _plain(cells, f"{base}/PITSA/fpr"))
        if counted and flipped:
            notes.append(f"For {model} under F2, the S entries of the S vs PITSA column rest on its "
                         f"{cells.val(invalid_key)} invalid answers to the structural cue, counted as escalations; "
                         "counted as dismissals, PITSA would have the lower loss at every k.")
    return notes


def loss_table(cells: ExtCells, analysis: Mapping[str, Any], models: Sequence[str]) -> str:
    first = analysis[LOSS_BLOCKS[0][0]]
    for name, _ in LOSS_BLOCKS[1:]:
        other = analysis[name]
        if any(first[k] != other[k] for k in ("prevalence_anchors", "cost_ratio_anchors", "policies")):
            raise ValueError(f"{name} uses other anchors or policies than {LOSS_BLOCKS[0][0]}")
    k_text = [f"{k:g}" for k in first["cost_ratio_anchors"]]

    def build(model: str) -> list[Row]:
        rows: list[Row] = [[f"\\multicolumn{{{len(header)}}}{{l}}{{{model}}}"]]  # nine columns do not fit the page
        for name, attack in LOSS_BLOCKS:
            block = analysis[name]
            records = block["models"][model]
            for index, prevalence in enumerate(block["prevalence_anchors"]):
                pi = f"pi{prevalence:g}"
                row = [attack if index == 0 else "", v3._prevalence_text(prevalence)]
                for a, b in v3.BREAK_EVEN_COLUMNS:
                    record = next(r for r in records["break_even"] if r["world"] == WORLD and r["a"] == a
                                  and r["b"] == b and f"pi{r['prevalence']:g}" == pi)
                    key = f"ext/{name}/models/{model}/break_even/{WORLD}/{a}-{b}/{pi}/k_star"
                    row.append(v3._break_even_cell(cells, record, key))
                for cost_ratio in block["cost_ratio_anchors"]:
                    record = next(r for r in records["anchors"] if r["world"] == WORLD
                                  and r["prevalence"] == prevalence and r["cost_ratio"] == cost_ratio)
                    row.append(_lowest_cell(record, block["policies"]))
                rows.append(row)
        return rows

    header = ["Attack", "Prevalence", "S vs U", "G vs U", "S vs PITSA", "G vs PITSA",
              f"Lowest, k {k_text[0]}", f"Lowest, k {k_text[1]}"]
    rows = v3._blocks(models, build)
    caption = ("Extension: break-even miss cost ratio between policies under attack by payload family 1 (F1) and payload "
               "family 2 (F2), read as in \\cref{tab:llm_loss}. Both attack "
               f"blocks read the extended sample, whose gated benign stratum holds {_sizes(cells, models[0])['benign_gate']} "
               "alerts, so they differ only in the payload family. The last columns name the policy with the lowest loss "
               f"at k of {k_text[0]} and {k_text[1]}, with any policy tied with it below. "
               + "".join(note + " " for note in _fail_safe_notes(cells, models)) + "Point estimates.")
    return v3.table(caption, "tab:ext_loss",
                    "L{0.95cm} L{1.7cm} C{2.05cm} C{2.15cm} C{2.15cm} C{2.25cm} C{1.9cm} C{1.9cm}", header, rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--locked-values", type=Path, default=DEFAULT_LOCKED_VALUES)
    parser.add_argument("--output-dir", type=Path, default=v3.DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    ext_defined, label, hashes = read_values(args.values)
    locked_defined, locked_label, _ = read_values(args.locked_values)
    if v3._sha256(args.analysis) not in hashes:
        logger.error("%s was not written from %s; regenerate the value file first", args.values, args.analysis)
        return 1
    if "OFFICIAL" != label or "OFFICIAL" != locked_label:
        if args.output_dir.resolve() == v3.DEFAULT_OUTPUT_DIR.resolve():
            logger.error("value files are %s and %s; only OFFICIAL files may write the manuscript tables",
                         label, locked_label)
            return 1
    if overlap := set(ext_defined) & set(locked_defined):
        logger.error("the value files define %d keys twice, e.g. %s", len(overlap), sorted(overlap)[0])
        return 1
    models = list(analysis["models"])
    cells = ExtCells({**locked_defined, **ext_defined})
    tables = {
        "table_ext_models.tex": models_table(cells, analysis, models),
        "table_ext_policies.tex": policy_table(cells, models),
        "table_ext_tests.tex": tests_table(cells, analysis, EXT_CONFIRMATORY, True),
        "table_ext_exploratory.tex": tests_table(cells, analysis, EXT_EXPLORATORY, False),
        "table_ext_injection.tex": injection_table(cells, analysis, models),
        "table_ext_stage2.tex": stage2_table(cells, models),
        "table_ext_discrimination.tex": discrimination_table(cells, analysis, models),
        "table_ext_loss.tex": loss_table(cells, analysis, models),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banner = f"% Generated by scripts/make_extension_tables.py from a {label} value file. Do not edit.\n"
    for name, text in tables.items():
        (args.output_dir / name).write_text(banner + text, encoding="utf-8")
    missing = sorted(set(cells.missing))
    manifest = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "value_source": label,
                "values_sha256": v3._sha256(args.values), "locked_values_sha256": v3._sha256(args.locked_values),
                "analysis_sha256": v3._sha256(args.analysis), "models": models, "tables": sorted(tables),
                "keys_printed_as_na": missing}
    (args.output_dir / "tables_ext_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %d tables to %s; %d keys printed as n/a", len(tables), args.output_dir, len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
