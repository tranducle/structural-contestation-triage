"""Write the value file the v3 manuscript reads every reported number from (gate G5).

Every quantity becomes \\val{<key>}, where <key> is the quantity's path in its result file, for
example \\val{salad/qwen2.5:3b/rates/attacked/S/recall}. \\val stops the compile on a key the file
does not define, so a number that no analysis produced cannot be typed into the manuscript and
still build.

Only an OFFICIAL analysis whose gate G4 passed may write llm_values.tex. Any other analysis writes
SMOKE_llm_values.tex, which the manuscript never inputs, so a code check cannot leak into the paper.

Instruction-model rates print as percentages with one decimal and no percent sign. The locked v2
classical results (SALAD, LMD, ablation, runtime) sit under v2/ keys and print as three-decimal
fractions, the form the v2 tables used.

  python3 make_triage_macros.py                                   # official analysis, official file
  python3 make_triage_macros.py --analysis <SMOKE json>           # writes SMOKE_llm_values.tex
  python3 make_triage_macros.py --check paper_v3.tex results.tex  # G5 report for the manuscript
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = PAPER_ROOT / "results/triage/triage_study_analysis.json"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "results/generated"
V2 = PAPER_ROOT / "results/detector"

KEY_PATTERN = re.compile(r"^[A-Za-z0-9.:/_+@-]+$")
PERCENT_BASES = {"rate", "recall", "fpr", "recall_diff", "fpr_diff", "share_a_lower", "wilson"}
P_BASES = {"exact_p", "holm_p", "exact_two_sided_p", "holm_adjusted_p"}
ONE_DECIMAL_BASES = {"median_s", "p90_s", "total_hours"}
STRING_LEAVES = {"best_policy", "a_wins", "family", "first", "second"}
SKIPPED_TOP_LEVEL = {"inputs", "label", "g4_passed", "models", "policies", "confirmatory_family"}
CLASSICAL_METRICS = ("precision", "recall", "f1", "fpr", "auroc", "auprc")
GUARD = (
    "\\providecommand{\\val}[1]{\\ifcsname llmv@#1\\endcsname\\csname llmv@#1\\endcsname"
    "\\else\\PackageError{llmvalues}{Undefined value #1}{The key is not defined in the value file.}\\fi}"
)

Key = tuple[str, ...]


def _validate_key(path: Key) -> str:
    key = "/".join(path)
    if not KEY_PATTERN.match(key) or "--" in key:
        raise ValueError(f"key {key!r} has characters a TeX control sequence name cannot carry safely")
    return key


def _kind(path: Key) -> str:
    leaf = path[-2] if path[-1] in {"lo", "hi"} else path[-1]
    base = leaf[:-2] if leaf.endswith("95") else leaf
    if base in P_BASES:
        return "p"
    if path[0] == "v2":
        return "three"
    if base in PERCENT_BASES:
        return "pct"
    if path[0] == "loss":
        return "kstar" if base == "k_star" else "loss"
    if base in ONE_DECIMAL_BASES:
        return "one"
    return "three"


def _signed(value: float, digits: int) -> str:
    text = f"{abs(value):.{digits}f}"
    negative = value < 0 and float(text) != 0.0
    return ("\\ensuremath{-}" if negative else "") + text


def _grouped(value: int) -> str:
    text = f"{abs(value):,}".replace(",", "{,}")
    return ("\\ensuremath{-}" if value < 0 else "") + text


def format_value(path: Key, value: Any) -> str | None:
    """TeX text for one leaf, or None when the leaf carries nothing reportable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if path[-1] not in STRING_LEAVES or re.search(r"[\\{}~^$#%&]", value):
            return None
        return value.replace("_", "\\_")
    if not isinstance(value, (int, float)):
        return None
    kind = _kind(path)
    if kind == "pct":
        return _signed(100.0 * value, 1)
    if kind == "p":
        return "\\ensuremath{<}0.001" if value < 0.001 else f"{value:.3f}"
    if kind in {"loss", "three"} and isinstance(value, int):
        return _grouped(value)
    if kind in {"kstar", "one"}:
        return _signed(float(value), 1)
    return _signed(float(value), 3)


def _list_item_key(name: str, item: Mapping[str, Any]) -> tuple[Key, set[str]]:
    if name == "tests":
        return (str(item["name"]),), {"name", "strata"}
    if name == "anchors":
        return (item["world"], f"pi{item['prevalence']:g}", f"k{item['cost_ratio']:g}"), \
            {"world", "prevalence", "cost_ratio"}
    if name == "break_even":
        return (item["world"], f"{item['a']}-{item['b']}", f"pi{item['prevalence']:g}"), \
            {"world", "a", "b", "prevalence"}
    raise ValueError(f"no key rule for a list of records named {name!r}; the analysis schema changed")


def _is_interval(path: Key, node: list) -> bool:
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node)
    return len(node) == 2 and numeric and (path[-1].endswith("95") or path[-1].endswith("95ci"))


def flatten(node: Any, path: Key, out: dict[Key, Any]) -> None:
    """Leaf path -> raw value. Intervals split into lo and hi; record lists are keyed by content."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            flatten(value, path + (str(key),), out)
    elif isinstance(node, list):
        if _is_interval(path, node):
            out[path + ("lo",)], out[path + ("hi",)] = node
        elif node and all(isinstance(item, Mapping) for item in node):
            for item in node:
                suffix, used = _list_item_key(path[-1], item)
                flatten({k: v for k, v in item.items() if k not in used}, path + suffix, out)
        # Lists of names, grid axes and grid cells carry no single reportable quantity.
    elif path in out:
        raise ValueError(f"duplicate key {'/'.join(path)}")
    else:
        out[path] = node


def analysis_values(analysis: Mapping[str, Any]) -> dict[Key, Any]:
    out: dict[Key, Any] = {}
    for key, value in analysis.items():
        if key not in SKIPPED_TOP_LEVEL:
            flatten(value, (key,), out)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(text: str) -> int | float | None:
    if text == "":
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return float(text)


def classical_values(v2: Path) -> dict[Key, Any]:
    """The locked v2 SALAD, LMD, ablation and runtime results under v2/ keys."""
    out: dict[Key, Any] = {}
    salad = json.loads((v2 / "salad_detector_results.json").read_text())
    lmd = json.loads((v2 / "lmd_detector_results.json").read_text())
    for row in salad.get("metrics_aggregate", []):
        for metric in CLASSICAL_METRICS:
            out[("v2", "salad", row["model"], row["condition"], metric)] = float(row[f"{metric}_mean"])
    for row in lmd.get("metrics_aggregate", []):
        for metric in CLASSICAL_METRICS:
            out[("v2", "lmd", row["split"], row["model"], metric)] = float(row[f"{metric}_mean"])
    flatten(salad["pitsa_full_cluster_bootstrap"], ("v2", "salad", "PITSA_FullTrain", "bootstrap"), out)
    for holdout in salad["source_holdouts"]:
        flatten(holdout["metrics"], ("v2", "salad", "holdout", f"{holdout['train_source']}-{holdout['test_source']}"),
                out)
    for test in salad["paired_exact_mcnemar_full_attack"]:
        flatten({k: v for k, v in test.items() if k not in {"reference", "comparator"}},
                ("v2", "salad", "mcnemar", f"{test['reference']}-{test['comparator']}"), out)
    protocol = salad["protocol"]
    for name in ("train_rows", "validation_rows", "test_rows", "learned_model_train_sample_rows",
                 "validation_max_fpr", "attack_clause_count"):
        out[("v2", "salad", "protocol", name)] = protocol[name]
    for model, curve in salad["graded_attack_curves"].items():
        for point in curve:
            out[("v2", "salad", "graded", model, f"c{point['clause_count']}", "hostile_recall")] = \
                point["hostile_recall"]

    flatten(lmd["split_report"], ("v2", "lmd", "split"), out)
    flatten(lmd["statistical_validation"]["pitsa_cluster_bootstrap"], ("v2", "lmd", "PITSA_Core", "bootstrap"), out)
    for test in lmd["statistical_validation"]["paired_exact_mcnemar"]:
        kept = {k: v for k, v in test.items() if k in {"a_correct_b_wrong", "a_wrong_b_correct", "discordant_pairs",
                                                        "exact_two_sided_p", "holm_adjusted_p"}}
        flatten(kept, ("v2", "lmd", "mcnemar", f"{test['reference']}-{test['comparator']}"), out)
    flatten(lmd["subclass_metrics"], ("v2", "lmd", "subclass"), out)

    ablation = json.loads((v2 / "ablation_results.json").read_text()).get("ablation_rows", [])
    full = {row["dataset"]: float(row["auprc"]) for row in ablation if row["configuration"] == "Full"}
    for row in ablation:
        base = ("v2", "ablation", row["dataset"], row["configuration"])
        for field in CLASSICAL_METRICS + ("tp", "fp", "tn", "fn"):
            out[base + (field,)] = _number(row[field])
        out[base + ("auprc_drop",)] = full[row["dataset"]] - float(row["auprc"])

    runtime = json.loads((v2 / "runtime_results.json").read_text())
    for name, block in runtime.items():
        if isinstance(block, Mapping) and "median_microseconds_per_row" in block:
            for field in ("median_microseconds_per_row", "rows", "measured_runs", "warmup_runs"):
                out[("v2", "runtime", name, field)] = block[field]
            out[("v2", "runtime", name, "median_rows_per_second")] = int(round(block["median_rows_per_second"]))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shown(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(PAPER_ROOT)) if resolved.is_relative_to(PAPER_ROOT) else str(resolved)


def render(values: Mapping[Key, Any], label: str, sources: Iterable[Path]) -> tuple[str, int]:
    lines = [
        "% Generated by scripts/make_triage_macros.py. Do not edit; regenerate.",
        f"% Analysis label: {label}",
        *(f"% source {_shown(path)} sha256 {_sha256(path)}" for path in sources),
        GUARD,
        f"\\def\\llmvaluesource{{{'OFFICIAL' if label == 'OFFICIAL' else 'SMOKE'}}}",
    ]
    count = 0
    for path in sorted(values):
        text = format_value(path, values[path])
        if text is None:
            continue
        lines.append(f"\\expandafter\\def\\csname llmv@{_validate_key(path)}\\endcsname{{{text}}}")
        count += 1
    return "\n".join(lines) + "\n", count


VAL_USE = re.compile(r"\\val\{([^}]*)\}")
BARE_NUMBER = re.compile(r"(?<![\w.\\])\d+\.\d+|(?<![\w.\\])\d+\\%")
NON_PROSE = re.compile(r"\\(?:ref|cref|Cref|label|cite[pt]?|input|includegraphics)(?:\[[^]]*\])?\{[^}]*\}")


def _strip_comment(line: str) -> str:
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def check_manuscript(files: list[Path], defined: set[str]) -> dict[str, Any]:
    """Keys a manuscript uses that the value file lacks, and numbers typed outside \\val for review."""
    used: set[str] = set()
    bare: list[dict[str, Any]] = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = _strip_comment(line)
            used.update(VAL_USE.findall(text))
            remainder = NON_PROSE.sub("", VAL_USE.sub("", text))
            for match in BARE_NUMBER.finditer(remainder):
                bare.append({"file": path.name, "line": number, "text": match.group(0)})
    missing = sorted(used - defined)
    return {"passed": not missing, "keys_used": len(used), "missing_keys": missing, "bare_numbers": bare}


def _defined_keys(tex: Path) -> set[str]:
    return set(re.findall(r"\\csname llmv@(.*?)\\endcsname", tex.read_text(encoding="utf-8")))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", type=Path, nargs="+", help="manuscript files to check against the value file")
    parser.add_argument("--values", type=Path, help="value file for --check (default: the official one)")
    parser.add_argument("--report", type=Path, help="write the --check result as JSON here")
    args = parser.parse_args()

    if args.check:
        values_file = (args.values or args.output_dir / "llm_values.tex").resolve()
        result = check_manuscript(args.check, _defined_keys(values_file))
        result["values_file"] = _shown(values_file)
        if args.report:
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.info("G5 %s: %d keys used, %d missing, %d bare numbers to review",
                    "PASS" if result["passed"] else "FAIL", result["keys_used"], len(result["missing_keys"]),
                    len(result["bare_numbers"]))
        for key in result["missing_keys"]:
            logger.error("undefined key %s", key)
        return 0 if result["passed"] else 1

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    official = analysis.get("label") == "OFFICIAL"
    if official and not analysis.get("g4_passed"):
        logger.error("refusing to write values: the OFFICIAL analysis did not pass gate G4")
        return 1
    values = {**analysis_values(analysis), **classical_values(V2)}
    sources = [args.analysis.resolve(), V2 / "salad_detector_results.json",
               V2 / "lmd_detector_results.json", V2 / "ablation_results.json",
               V2 / "runtime_results.json"]
    text, count = render(values, str(analysis.get("label", "")), sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (("" if official else "SMOKE_") + "llm_values.tex")
    output.write_text(text, encoding="utf-8")
    logger.info("wrote %d values to %s (%s)", count, output, "OFFICIAL" if official else "SMOKE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
