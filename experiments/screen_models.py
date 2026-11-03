"""Choose the added model by the screening rule fixed in ext_design.SCREENING_RULE.

Each candidate answers the clean opening turn, built by the locked prompt builder, on the 40
screening alerts (10 per stratum, disjoint from every other sample). The rule is then applied
mechanically. Screening answers select a model; they are never reported as evidence about it.
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
import statistics
from pathlib import Path
from typing import Any

import requests

import salad_triage_runner as salad
from ext_design import SCREENING_RULE, ScreeningRule
from llm_common import DecodingConfig, OllamaClient, ResponseStore, run_call, sha256_text
from llm_policies import escalates, is_invalid

logger = logging.getLogger(__name__)

SCREEN_CONDITION = "s1_clean"


def summarize(records: list[dict[str, Any]], stimuli_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Parse validity, unweighted escalation rates by label, Youden J and median latency."""
    hostile = [r for r in records if salad.is_hostile(stimuli_by_id[r["stimulus_id"]])]
    benign = [r for r in records if not salad.is_hostile(stimuli_by_id[r["stimulus_id"]])]
    if not hostile or not benign:
        raise ValueError("screening needs hostile and benign answers")
    tpr = sum(escalates(r) for r in hostile) / len(hostile)
    fpr = sum(escalates(r) for r in benign) / len(benign)
    return {
        "n": len(records),
        "n_hostile": len(hostile),
        "n_benign": len(benign),
        "parse_validity": sum(not is_invalid(r) for r in records) / len(records),
        "hostile_escalation_rate": tpr,
        "benign_escalation_rate": fpr,
        "youden_j": tpr - fpr,
        "median_latency_seconds": statistics.median(float(r["elapsed_seconds"]) for r in records),
    }


def select_model(summary: dict[str, dict[str, Any] | None], rule: ScreeningRule) -> str | None:
    """Largest Youden J among candidates meeting the parse-validity floor; ties to lower latency.

    A candidate whose summary is None (unavailable, or its run failed) does not qualify.
    """
    qualifying = [
        (name, row) for name, row in summary.items()
        if row is not None and row["parse_validity"] >= rule.min_parse_validity
    ]
    if not qualifying:
        return None
    best = min(qualifying, key=lambda item: (-item[1]["youden_j"], item[1]["median_latency_seconds"], item[0]))
    return best[0]


def tag_details(base_url: str, names: list[str]) -> list[dict[str, Any]]:
    response = requests.get(base_url.rstrip("/") + "/api/tags", timeout=30)
    response.raise_for_status()
    return [item for item in response.json().get("models", []) if item.get("name") in names]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tags-manifest", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stimuli_path = args.paper_root / "data/stimuli/salad_llm_screen_stimuli.jsonl"
    stimuli = load_jsonl(stimuli_path)
    for stimulus in stimuli:
        leaks = salad.leaked_fields(salad.stage1_messages(stimulus, SCREEN_CONDITION), stimulus)
        if leaks:
            raise RuntimeError(f"label leakage in {stimulus['stimulus_id']}: {leaks}")
    stimuli_by_id = {item["stimulus_id"]: item for item in stimuli}

    decoding = DecodingConfig()
    client = OllamaClient(args.base_url, decoding)
    digests = client.model_digests()
    store = ResponseStore(args.output_dir / "screen_responses.jsonl")
    summary: dict[str, dict[str, Any] | None] = {}
    notes: dict[str, str] = {}
    for model in SCREENING_RULE.candidates:
        if model not in digests:
            summary[model], notes[model] = None, "not available in Ollama"
            continue
        logger.info("screening %s (%s)", model, digests[model])
        client.make_exclusive(model)
        try:
            records = [
                run_call(client, store, model=model, model_digest=digests[model], stimulus=stimulus,
                         condition=SCREEN_CONDITION, stage=1, base_condition=None,
                         messages=salad.stage1_messages(stimulus, SCREEN_CONDITION))
                for stimulus in stimuli
            ]
        except RuntimeError as error:
            logger.error("screening %s failed: %s", model, error)
            summary[model], notes[model] = None, f"run failed: {error}"
            continue
        finally:
            client.unload(model)
        summary[model] = summarize(records, stimuli_by_id)

    selected = select_model(summary, SCREENING_RULE)
    args.tags_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.tags_manifest.write_text(json.dumps(
        {"models": tag_details(args.base_url, list(SCREENING_RULE.candidates))}, sort_keys=True) + "\n",
        encoding="utf-8")
    selection = {
        "rule": {"candidates": list(SCREENING_RULE.candidates), "per_stratum": SCREENING_RULE.per_stratum,
                 "min_parse_validity": SCREENING_RULE.min_parse_validity,
                 "criterion": "largest unweighted Youden J; ties to lower median latency"},
        "ollama_version": client.version(),
        "digests": {model: digests.get(model) for model in SCREENING_RULE.candidates},
        "decoding": decoding.options(),
        "screen_stimuli_sha256": sha256_text(stimuli_path.read_text(encoding="utf-8")),
        "summary": summary,
        "notes": notes,
        "selected": selected,
    }
    out = args.output_dir / "screen_selection.json"
    out.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("selected %s; wrote %s", selected, out)


if __name__ == "__main__":
    main()
