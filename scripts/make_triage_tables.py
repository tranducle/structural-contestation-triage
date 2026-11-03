"""LaTeX tables of the v3 manuscript, written from the analysis and the value file it produced.

Every number in a table is a \\val{key} reference into the value file that make_triage_macros.py
writes, so a table cannot disagree with the analysis it cites. A key the value file does not
define (a rate over zero trials) prints n/a and is listed in tables_manifest.json. Text that is
not a measured number, such as a model digest, is read from the run manifests and the Ollama model
listing, never typed by hand. The analysis decides only the layout: which test rows exist and on
which side of a break-even ratio a policy wins.

Added after the analysis lock of 2026-10-22; it reads the locked outputs and changes none of them.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger("make_v3_tables")

PAPER_ROOT = Path(__file__).resolve().parents[1]
V3 = PAPER_ROOT
DEFAULT_ANALYSIS = PAPER_ROOT / "results/triage/triage_study_analysis.json"
DEFAULT_VALUES = PAPER_ROOT / "results/generated/llm_values.tex"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "results/tables"
RUN_DIR = PAPER_ROOT / "data/responses"
MODEL_LISTING = PAPER_ROOT / "configs/ollama_models_main.json"

DEFINITION = re.compile(r"^\\expandafter\\def\\csname llmv@(?P<key>.+?)\\endcsname\{(?P<text>.*)\}$")
SOURCE_LABEL = re.compile(r"^\\def\\llmvaluesource\{(?P<label>[A-Z]+)\}$")
SOURCE_HASH = re.compile(r"^% source \S.* sha256 (?P<sha>[0-9a-f]{64})$")
NA = "n/a"
WORLD = "attacked"

POLICY_TEXT = {"U": "Unassisted", "D": "Spotlighting", "S": "Structural contestation", "P": "Placebo second turn",
               "DS": "Spotlighting and contestation", "G": "Model or gate", "PITSA": "Gate alone"}
SALAD_ROWS = ("U", "D", "S", "P", "DS", "G")
OTRF_COLUMNS = (("U_clean", "Unassisted, clean text"), ("U", "Unassisted"), ("D", "Spotlighting"),
                ("PROV", "Provenance"), ("PLACEBO", "Placebo"), ("DPROV", "Spotlighting and provenance"))
BREAK_EVEN_COLUMNS = (("S", "U"), ("G", "U"), ("S", "PITSA"), ("G", "PITSA"))
LOSS_COLUMNS = ("U", "S", "G", "PITSA")
LOSS_DIFFERENCES = (("S", "U"), ("G", "U"))
# A tie for the lowest loss is credited to a policy that uses no model, as in make_triage_figures.py.
NO_MODEL = ("PITSA", "ALL", "NONE")
TIE_TOLERANCE = 1e-12
V2_SALAD = (("PITSA_FullTrain", "PITSA full train"), ("PITSA_BudgetMatched", "PITSA budget matched"),
            ("StructuralLogistic", "Structural logistic"), ("StructuralRandomForest", "Structural random forest"),
            ("JointWordLogistic", "Joint word logistic"), ("CharacterNgramLogistic", "Character n-gram logistic"),
            ("JointNaiveBayes", "Joint naive Bayes"))
V2_LMD_SPLITS = (("signature_disjoint", "Signature disjoint"), ("temporal_future", "Future year"),
                 ("station_holdout", "Held-out station"))
V2_LMD_MODELS = (("ContextOnlyLogistic", "Context only logistic"), ("ContextAugmentedLogistic", "Context augmented logistic"),
                 ("CoreLogistic", "Core logistic"), ("CoreRandomForest", "Core random forest"), ("PITSA_Core", "PITSA core"))
CI_NOTE = "Brackets give the 95 percent interval"

Row = Sequence[str] | None  # None draws a \midrule


@dataclass
class Cells:
    """Turns value-file keys into table cells and remembers every key it could not find."""

    defined: Mapping[str, str]
    missing: list[str] = field(default_factory=list)

    def val(self, key: str) -> str:
        if key in self.defined:
            return f"\\val{{{key}}}"
        self.missing.append(key)
        return NA

    def ci(self, point: str, interval: str) -> str:
        text = self.val(point)
        if text == NA:
            return NA
        bounds = [f"{interval}/lo", f"{interval}/hi"]
        if all(b in self.defined for b in bounds):
            return f"{text} [{self.val(bounds[0])}, {self.val(bounds[1])}]"
        self.missing.extend(b for b in bounds if b not in self.defined)
        return text

    def count(self, prefix: str) -> str:
        successes, trials = self.val(f"{prefix}/successes"), self.val(f"{prefix}/trials")
        return NA if NA in (successes, trials) else f"{successes}/{trials}"


def read_values(path: Path) -> tuple[dict[str, str], str, set[str]]:
    """Key -> TeX text, the value-source label, and the sha256 of every source the file names."""
    defined: dict[str, str] = {}
    label, hashes = "", set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := DEFINITION.match(line):
            defined[match["key"]] = match["text"]
        elif match := SOURCE_LABEL.match(line):
            label = match["label"]
        elif match := SOURCE_HASH.match(line):
            hashes.add(match["sha"])
    if not defined or not label:
        raise ValueError(f"{path} is not a value file written by make_triage_macros.py")
    return defined, label, hashes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(caption: str, label: str, colspec: str, header: Sequence[str], rows: Sequence[Row]) -> str:
    lines = ["\\begin{table}[!htbp]", "\\centering", "\\singlespacing", "\\scriptsize",
             "\\setlength{\\tabcolsep}{2pt}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += ["\\midrule" if row is None else " & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _blocks(models: Sequence[str], build: Callable[[str], list[Row]]) -> list[Row]:
    rows: list[Row] = []
    for index, model in enumerate(models):
        if index:
            rows.append(None)
        rows += build(model)
    return rows


def policy_table(cells: Cells, models: Sequence[str]) -> str:
    def rate(model: str, world: str, policy: str, metric: str) -> str:
        base = f"salad/{model}/rates/{world}/{policy}/{metric}"
        return cells.ci(base, f"{base}95")

    for model in models:  # benign alerts are read clean in both worlds, so one FPR column must serve both
        for policy in SALAD_ROWS + ("PITSA",):
            pair = [cells.defined.get(f"salad/{model}/rates/{w}/{policy}/fpr{s}") for w in ("clean", WORLD)
                    for s in ("", "95/lo", "95/hi")]
            if pair[:3] != pair[3:]:
                raise ValueError(f"{model} {policy}: false-escalation rate differs between worlds; split the column")

    def build(model: str) -> list[Row]:
        return [[model if i == 0 else "", f"{POLICY_TEXT[p]} ({p})", rate(model, "clean", p, "recall"),
                 rate(model, WORLD, p, "recall"), rate(model, "clean", p, "fpr")] for i, p in enumerate(SALAD_ROWS)]

    gate = models[0]
    rows: list[Row] = [["No model", f"{POLICY_TEXT['PITSA']} (PITSA)", rate(gate, "clean", "PITSA", "recall"),
                        rate(gate, WORLD, "PITSA", "recall"), rate(gate, "clean", "PITSA", "fpr")], None]
    rows += _blocks(models, build)
    caption = ("Weighted operating points of each triage policy on the SALAD test population, in percent. "
               "Recall is the share of hostile alerts escalated, with endpoint annotations left clean or "
               "carrying an injection payload. False escalation is the share of benign alerts escalated; "
               f"benign annotations are never injected, so it is the same in both conditions. {CI_NOTE} from "
               "the stratified bootstrap. The gate alone has a zero-width interval because the strata are defined "
               "by its decision, which makes its rate on the test population exact. For model policies, "
               "zero-width intervals reflect constant within-stratum sampled decisions and do not establish "
               "zero population uncertainty; exact per-stratum bounds are described in Supplementary Material, Sect. S1.")
    # "Recall, attacked" holds at most 59.4pt (its widest cell), so C{2.7cm} left 17.4pt unused there
    # while "Recall, clean" held "100.0 [100.0, 100.0]" (70.8pt) with 6.0pt to spare. The row totals
    # are unchanged; the slack moves to the column that needs it.
    # Keep the added policy codes on the same line as their full names.
    return table(caption, "tab:llm_policies", "L{1.7cm} L{4.8cm} C{2.9cm} C{2.5cm} C{2.7cm}",
                 ["Model", "Policy", "Recall, clean", "Recall, attacked", "False escalation"], rows)


def _arm(text: str) -> tuple[str, str | None]:
    policy, _, world = text.partition("@")
    return policy, world or None


def _arm_text(text: str) -> str:
    policy, world = _arm(text)
    if policy.endswith("_clean"):  # OTRF policy codes carry the condition in the name, e.g. U_clean
        policy, world = policy.removesuffix("_clean"), "clean"
    return f"{policy} ({world})" if world else policy


def _test_rate(cells: Cells, corpus: str, model: str, arm: str, metric: str) -> str:
    policy, world = _arm(arm)
    if corpus == "otrf":
        return cells.val(f"otrf/{model}/escalation/{policy}/all/rate")
    return cells.val(f"salad/{model}/rates/{world}/{policy}/{metric}")


def tests_table(cells: Cells, analysis: Mapping[str, Any], models: Sequence[str], family: str) -> str:
    rows: list[Row] = []
    sources = [("salad", analysis["salad"])] + ([("otrf", analysis["otrf"])] if family == "exploratory" else [])
    for corpus, block in sources:
        names = [t["name"] for t in block[models[0]]["tests"] if t["family"] == family]
        for name in names:
            if rows:
                rows.append(None)
            for index, model in enumerate(models):
                test = next(t for t in block[model]["tests"] if t["name"] == name)
                hostile = any(s.startswith("hostile") for s in test.get("strata", ["hostile"]))
                metric = "recall" if hostile else "fpr"
                metric_text = "Escalation" if corpus == "otrf" else ("Recall" if hostile else "False esc.")
                arms = " vs ".join(_arm_text(test[side]) for side in ("first", "second"))
                key = f"{corpus}/{model}/tests/{name}"
                row = [name.split("_")[0] if index == 0 else "", model, arms, metric_text,
                       _test_rate(cells, corpus, model, test["first"], metric),
                       _test_rate(cells, corpus, model, test["second"], metric),
                       f"{cells.val(key + '/first_only')} : {cells.val(key + '/second_only')}",
                       cells.val(f"{key}/exact_p")]
                if family == "confirmatory":
                    row.append(cells.val(f"{key}/holm_p"))
                rows.append(row)
    header = ["Test", "Model", "Comparison", "Metric", "Arm A", "Arm B", "Discordant", "Exact p"]
    # The exploratory rows carry the longer Metric word "Escalation" (38.4pt), which the confirmatory
    # width L{1.4cm} (39.8pt) held with 1.4pt to spare; the comparison cell there is short, so the width
    # is taken from Comparison. Each family keeps its own total, and both total 13.1cm.
    comparison_width, metric_width = ("L{4.2cm}", "L{1.4cm}") if family == "confirmatory" else ("L{4.0cm}", "L{1.6cm}")
    colspec = f"L{{0.7cm}} L{{1.7cm}} {comparison_width} {metric_width} C{{1.1cm}} C{{1.1cm}} C{{1.6cm}} C{{1.3cm}}"
    if family == "confirmatory":
        header.append("Holm p")
        colspec += " C{1.3cm}"
        caption = ("Confirmatory tests H1 to H5. Each is a two-sided exact McNemar test on the paired sample "
                   "alerts; Holm adjustment runs over all models and hypotheses. A comparison reads arm A vs arm B, "
                   "and the Arm A and Arm B columns give their weighted rates, in percent. Discordant counts the "
                   "alerts that only arm A escalated and those that only arm B escalated. "
                   r"Policy codes are keyed in \cref{tab:llm_policies}.")
        label = "tab:llm_tests"
    else:
        caption = ("Exploratory tests, without multiplicity adjustment: E1 to E6 on SALAD and O1 to O5 on the "
                   "attack-only provenance probe, where every event is hostile and the rate is escalation. Columns as in "
                   "the confirmatory table.")
        label = "tab:llm_exploratory"
    return table(caption, label, colspec, header, rows)


def _prevalence_text(prevalence: float) -> str:
    inverse = 1.0 / prevalence
    return f"1 in {round(inverse)}" if abs(inverse - round(inverse)) < 1e-9 else "SALAD test"


def _break_even_cell(cells: Cells, record: Mapping[str, Any], key: str) -> str:
    a, b, k_star, side = record["a"], record["b"], record["k_star"], record["a_wins"]
    if k_star is None:
        return {"A_always": f"{a}, all k", "B_always": f"{b}, all k"}.get(side, "equal")
    if k_star <= 1.0:  # the miss cost ratio is at least one review, so one side wins everywhere
        return f"{a}, all k" if side == "above_k_star" else f"{b}, all k"
    return f"{a} above {cells.val(key)}" if side == "above_k_star" else f"{a} below {cells.val(key)}"


def _lowest_cell(record: Mapping[str, Any], policies: Sequence[str]) -> str:
    """Name the lowest-loss policy at an anchor, showing an exact tie as the loss-region figure does.

    ``best_policy`` comes from ``np.argmin``, so a tie goes to the policy listed first in
    ``loss.policies``. The recorded losses are only compared, never recomputed, and the cell
    stops the run unless the first tied policy is ``best_policy``.
    """
    losses = record["loss"]
    lowest = min(losses[p] for p in policies)
    tied = [p for p in policies if losses[p] - lowest <= TIE_TOLERANCE * max(lowest, 1.0)]
    if tied[0] != record["best_policy"]:
        raise ValueError(f"tie {tied} does not start with best_policy {record['best_policy']!r}")
    credited = next((p for p in NO_MODEL if p in tied), tied[0])
    others = [p for p in tied if p != credited]
    if not others:
        return credited
    # Braces keep the inner row break at depth one, where check_table_fit.py does not split rows.
    return r"{\begin{tabular}[t]{@{}c@{}}%s\\= %s\end{tabular}}" % (credited, " = ".join(others))


def loss_table(cells: Cells, analysis: Mapping[str, Any], models: Sequence[str]) -> str:
    loss = analysis["loss"]
    anchors = loss["prevalence_anchors"]
    k_values = [f"k{k:g}" for k in loss["cost_ratio_anchors"]]

    def build(model: str) -> list[Row]:
        rows: list[Row] = []
        records = loss["models"][model]["break_even"]
        for index, prevalence in enumerate(anchors):
            pi = f"pi{prevalence:g}"
            row = [model if index == 0 else "", _prevalence_text(prevalence)]
            for a, b in BREAK_EVEN_COLUMNS:
                record = next(r for r in records if r["world"] == WORLD and r["a"] == a and r["b"] == b
                              and f"pi{r['prevalence']:g}" == pi)
                row.append(_break_even_cell(cells, record, f"loss/models/{model}/break_even/{WORLD}/{a}-{b}/{pi}/k_star"))
            for cost_ratio in loss["cost_ratio_anchors"]:
                record = next(r for r in loss["models"][model]["anchors"] if r["world"] == WORLD
                              and r["prevalence"] == prevalence and r["cost_ratio"] == cost_ratio)
                row.append(_lowest_cell(record, loss["policies"]))
            rows.append(row)
        return rows

    k_text = [k[1:] for k in k_values]
    caption = ("Break-even miss cost ratio between policies under attack. A miss costs k analyst reviews and an "
               "escalation costs one. An entry such as S above x means S has the lower expected loss for every k "
               "above x, and S below x for every k below it; S, all k means it is lower for every k above one and "
               "no higher at one; "
               "equal means both have the same recall and false escalation, so the same loss at every k. "
               "The last columns name the policy "
               f"with the lowest loss at k of {k_text[0]} and {k_text[1]}; ALL escalates every alert and NONE "
               "dismisses every alert. Where policies tie exactly, the entry names a policy that uses no model, "
               "and the policies after the equals sign cost the same, as in \\cref{fig:loss_regions}. Point "
               "estimates; recall and false escalation are carried to each prevalence unchanged.")
    header = ["Model", "Prevalence", "S vs U", "G vs U", "S vs PITSA", "G vs PITSA",
              f"Lowest, k {k_text[0]}", f"Lowest, k {k_text[1]}"]
    # The two row-key columns held 42.5pt and 45.8pt in 44.1pt and 46.9pt, and the table already used
    # 16.33cm of the 16.51cm measure, so the widths are rebalanced rather than grown: the slack comes
    # from the four ratio columns, which held short cells. Total stays 15.20cm and every column is
    # left with at least 5pt of spare.
    return table(caption, "tab:llm_loss",
                 "L{1.75cm} L{1.80cm} C{2.0cm} C{1.95cm} C{1.85cm} C{1.85cm} C{2.0cm} C{2.0cm}",
                 header, _blocks(models, build))


def loss_anchor_table(cells: Cells, analysis: Mapping[str, Any], models: Sequence[str]) -> str:
    loss = analysis["loss"]

    def build(model: str) -> list[Row]:
        rows: list[Row] = []
        for i, prevalence in enumerate(loss["prevalence_anchors"]):
            for j, k in enumerate(loss["cost_ratio_anchors"]):
                base = f"loss/models/{model}/anchors/{WORLD}/pi{prevalence:g}/k{k:g}"
                rows.append([model if i == j == 0 else "", _prevalence_text(prevalence) if j == 0 else "", f"{k:g}",
                             *[cells.val(f"{base}/loss/{p}") for p in LOSS_COLUMNS],
                             *[cells.ci(f"{base}/differences/{a}-{b}/diff", f"{base}/differences/{a}-{b}/diff95")
                               for a, b in LOSS_DIFFERENCES]])
        return rows

    caption = ("Expected loss under attack, in analyst reviews per alert, at the anchor prevalences and cost ratios. "
               "A difference such as S minus U is the loss of S less that of U; a negative value favors S. "
               f"{CI_NOTE} from the stratified bootstrap.")
    header = ["Model", "Prevalence", "k", *LOSS_COLUMNS, "S minus U", "G minus U"]
    return table(caption, "tab:llm_loss_anchors",
                 "L{1.6cm} L{1.7cm} C{0.5cm} C{0.9cm} C{0.9cm} C{0.9cm} C{1.0cm} C{3.8cm} C{3.8cm}",
                 header, _blocks(models, build))


def otrf_table(cells: Cells, models: Sequence[str]) -> str:
    def build(model: str) -> list[Row]:
        row = [model]
        for policy, _ in OTRF_COLUMNS:
            base = f"otrf/{model}/escalation/{policy}"
            row.append(f"{cells.count(base + '/all')} ({cells.count(base + '/excluding_suspects')})")
        return [row + [cells.count(f"otrf/{model}/payload_echo_injected")]]

    caption = ("Attack-only provenance probe: hostile events escalated out of all events, with the count excluding the "
               "suspected label-noise event in parentheses. Except in the clean-text column, the model reads "
               "event text carrying an injection payload; Provenance, Placebo, and Spotlighting and provenance add "
               "a second turn after a dismissal. Payload echo counts the unassisted injected first-turn answers, "
               f"out of {cells.val(f'otrf/{models[0]}/n_events')}, whose rationale shares a payload word stem "
               "absent from the clean prompt. No false-escalation rate exists on this probe.")
    header = ["Model", *[text for _, text in OTRF_COLUMNS], "Payload echo"]
    # The five outcome columns used to share one C{1.85cm}. The Provenance and Placebo headers need
    # 51.9pt of that 52.6pt measure and the Payload echo label wrapped to two lines in C{1.3cm}, so the
    # widths are now per column. The float still measures 16.28cm inside the 16.51cm text measure.
    return table(caption, "tab:otrf_probe",
                 "L{1.7cm} C{1.85cm} C{1.75cm} C{1.85cm} C{2.0cm} C{2.0cm} C{2.2cm} C{1.8cm}",
                 header, _blocks(models, build))


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


def models_table(cells: Cells, models: Sequence[str]) -> str:
    listing = {m["name"]: m for m in json.loads(MODEL_LISTING.read_text())["models"]}
    rows: list[Row] = []
    for model in models:
        recorded = {}
        for corpus in ("salad", "otrf"):
            manifest = RUN_DIR / f"{corpus}_responses_manifest_{_slug(model)}.json"
            if manifest.exists():
                recorded[corpus] = json.loads(manifest.read_text())["models"][model]
        digest = listing[model]["digest"][:12]
        if any(d != digest for d in recorded.values()):  # the local model changed after it was run
            raise ValueError(f"{model}: run digest {recorded} differs from the model listing {digest}")
        details = listing[model]["details"]
        rows.append([model, details["family"], details["parameter_size"], details["quantization_level"].replace("_", "\\_"),
                     f"\\texttt{{{digest}}}", cells.val(f"latency/{model}/salad/calls"),
                     cells.val(f"latency/{model}/otrf/calls"), cells.val(f"latency/{model}/salad/median_s")])
    caption = ("Instruction models, served locally by Ollama with fixed decoding. Digest is the leading twelve "
               "hexadecimal characters of the model blob each run recorded. Calls count model invocations; latency "
               "is the median seconds per SALAD call on the study machine.")
    return table(caption, "tab:llm_models", "L{1.8cm} L{1.2cm} C{1.6cm} C{1.8cm} C{2.3cm} C{1.8cm} C{1.8cm} C{1.6cm}",
                 ["Model", "Family", "Parameters", "Quantization", "Digest", "SALAD calls", "OTRF calls",
                  "Median s"], rows)


def v2_salad_table(cells: Cells) -> str:
    rows = [[text, *[cells.val(f"v2/salad/{name}/clean/{m}") for m in ("auprc", "auroc", "recall", "fpr")],
             cells.val(f"v2/salad/{name}/full_itpi/recall")] for name, text in V2_SALAD]
    caption = ("Evaluation 1: classical detectors on the official SALAD test split. "
               "Thresholds and hyperparameters were selected on the official validation split. Attacked recall "
               "appends four benign-sounding clauses to the text of every hostile alert. The structural detectors do "
               "not read that text, so their attacked recall equals their clean recall by construction. Stochastic "
               "detectors show the mean over three seeds.")
    return table(caption, "tab:v2_salad", "L{3.7cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{2.1cm}",
                 ["Detector", "AUPRC", "AUROC", "Recall", "FPR", "Attacked recall"], rows)


def v2_lmd_table(cells: Cells) -> str:
    rows: list[Row] = []
    for index, (split, split_text) in enumerate(V2_LMD_SPLITS):
        if index:
            rows.append(None)
        rows += [[split_text if j == 0 else "", text,
                  *[cells.val(f"v2/lmd/{split}/{name}/{m}") for m in ("auprc", "auroc", "recall", "fpr", "f1")]]
                 for j, (name, text) in enumerate(V2_LMD_MODELS)]
    caption = ("Evaluation 1: LMD-2023 under three shifts. Signature disjoint keeps every "
               "core signature of the test attacks out of training; the structural gate does not generalize across it. "
               "Stochastic detectors show the mean over three seeds.")
    return table(caption, "tab:v2_lmd", "L{2.4cm} L{3.7cm} C{1.4cm} C{1.4cm} C{1.4cm} C{1.4cm} C{1.4cm}",
                 ["Split", "Detector", "AUPRC", "AUROC", "Recall", "FPR", "F1"], rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", help="restrict to these models (default: every analysed model)")
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    defined, label, hashes = read_values(args.values)
    if _sha256(args.analysis) not in hashes:
        logger.error("%s was not written from %s; regenerate the value file first", args.values, args.analysis)
        return 1
    if label != "OFFICIAL" and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        logger.error("value file is %s; only an OFFICIAL value file may write the manuscript tables", label)
        return 1
    models = args.models or list(analysis["models"])
    unknown = [m for m in models if m not in analysis["salad"]]
    if unknown:
        logger.error("models not in the analysis: %s", unknown)
        return 1

    cells = Cells(defined)
    tables = {
        "table_llm_models.tex": models_table(cells, models),
        "table_llm_policies.tex": policy_table(cells, models),
        "table_llm_tests.tex": tests_table(cells, analysis, models, "confirmatory"),
        "table_llm_exploratory.tex": tests_table(cells, analysis, models, "exploratory"),
        "table_llm_loss.tex": loss_table(cells, analysis, models),
        "table_llm_loss_anchors.tex": loss_anchor_table(cells, analysis, models),
        "table_otrf_probe.tex": otrf_table(cells, models),
        "table_v2_salad.tex": v2_salad_table(cells),
        "table_v2_lmd.tex": v2_lmd_table(cells),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banner = f"% Generated by scripts/make_triage_tables.py from a {label} value file. Do not edit.\n"
    for name, text in tables.items():
        (args.output_dir / name).write_text(banner + text, encoding="utf-8")
    missing = sorted(set(cells.missing))
    manifest = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "value_source": label,
                "values_sha256": _sha256(args.values), "analysis_sha256": _sha256(args.analysis), "models": models,
                "tables": sorted(tables), "keys_printed_as_na": missing}
    (args.output_dir / "tables_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %d tables to %s; %d keys printed as n/a", len(tables), args.output_dir, len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
