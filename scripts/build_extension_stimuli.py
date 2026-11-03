"""Build the extension stimulus sets: 80 further benign gated alerts and a screening sample.

Both are drawn from the same official-test pool as the locked v3 sample, after refitting PITSA
exactly as salad_llm_stimuli.py does, and both exclude every alert already sampled. Records are
built field by field with the locked helper functions. Before anything is written, the builder
rebuilds the 160 locked stimuli from their record ids and requires a byte-identical match, so an
extension stimulus cannot differ from a locked one in format or derivation.
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
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


from core import score_llr, stable_bucket, write_json  # noqa: E402
from ext_design import EXTENSION_SEED, SCREENING_RULE  # noqa: E402
from build_salad_stimuli import (  # noqa: E402
    INJECTION_PAYLOADS,
    STRATA,
    fit_full_train_table,
    pillar_contributions,
    sensor_view,
)
from salad_detector_study import STRUCT_COLUMNS, USE_COLUMNS, engineer_salad  # noqa: E402

logger = logging.getLogger(__name__)

BENIGN_EXTENSION_SIZE = 80
BENIGN_EXTENSION_STRATUM = "benign_gate"


def build_stimulus(
    stimulus_id: str, index: int, weight: float, *, features: pd.DataFrame, raw_test: pd.DataFrame,
    contributions: pd.DataFrame, scores: np.ndarray, gate: np.ndarray, labels: np.ndarray,
    stratum: np.ndarray, threshold: float,
) -> dict[str, Any]:
    """One stimulus record, field for field as in salad_llm_stimuli.main."""
    record_id = str(features.at[index, "record_id"])
    contrib = contributions.loc[index]
    ranked = sorted(STRUCT_COLUMNS, key=lambda column: float(contrib[column]), reverse=True)
    return {
        "stimulus_id": stimulus_id,
        "record_id": record_id,
        "label": int(labels[index]),
        "ground_truth": "escalate" if labels[index] == 1 else "dismiss",
        "stratum": str(stratum[index]),
        "stratum_weight": weight,
        "source_dataset": str(features.at[index, "source_dataset"]),
        "attack_category": str(features.at[index, "attack_category"]),
        "sensor_fields": sensor_view(raw_test.iloc[index]),
        "pitsa_score": float(scores[index]),
        "pitsa_threshold": threshold,
        "pitsa_gate": bool(gate[index]),
        "pillar_values": {column: str(features.at[index, column]) for column in STRUCT_COLUMNS},
        "pillar_contributions": {column: float(contrib[column]) for column in STRUCT_COLUMNS},
        "top_pillars": ranked[:2],
        "injection_payload": INJECTION_PAYLOADS[stable_bucket(record_id, len(INJECTION_PAYLOADS))],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = args.paper_root
    data_dir = root / "data/external/salad_soc"
    v2_out = root / "results/detector"
    locked_path = root / "data/stimuli/salad_llm_stimuli.jsonl"
    locked_manifest = json.loads(locked_path.with_name("salad_llm_stimuli_manifest.json").read_text())
    out_dir = root / "data/stimuli"
    out_dir.mkdir(parents=True, exist_ok=True)

    tuning = json.loads((v2_out / "tuning_report.json").read_text())["PITSA_FullTrain"]
    alpha, threshold = float(tuning["selected_alpha"]), float(tuning["threshold"])
    table, _ = fit_full_train_table(data_dir / "salad_train.csv", alpha)
    raw_test = pd.read_csv(data_dir / "salad_test.csv", low_memory=False)
    features = engineer_salad(raw_test[USE_COLUMNS])
    scores = score_llr(features, table, STRUCT_COLUMNS)

    locked = np.load(v2_out / "primary_predictions.npz", allow_pickle=True)
    if not np.array_equal(locked["record_id"].astype(str), features["record_id"].astype(str).to_numpy()):
        raise RuntimeError("test record order differs from the locked v2 prediction artifact")
    max_abs_diff = float(np.max(np.abs(locked["PITSA_FullTrain_clean_scores"] - scores)))
    if max_abs_diff > 1e-9:
        raise RuntimeError(f"PITSA refit does not reproduce v2 scores (max abs diff {max_abs_diff})")

    labels = features["label"].to_numpy(dtype=np.int8)
    gate = scores >= threshold
    stratum = np.where(labels == 1, np.where(gate, "hostile_gate", "hostile_nogate"),
                       np.where(gate, "benign_gate", "benign_nogate"))
    population = {name: int(np.sum(stratum == name)) for name in STRATA}
    if population != locked_manifest["population_strata"]:
        raise RuntimeError(f"stratum populations differ from the locked manifest: {population}")

    position_of = {rid: i for i, rid in enumerate(features["record_id"].astype(str).to_numpy())}
    locked_lines = [line for line in locked_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    locked_items = [json.loads(line) for line in locked_lines]
    locked_index = [position_of[item["record_id"]] for item in locked_items]

    rng = np.random.default_rng(EXTENSION_SEED)
    taken = set(locked_index)
    pool = np.array(sorted(set(np.flatnonzero(stratum == BENIGN_EXTENSION_STRATUM)) - taken))
    benign_ext = sorted(rng.choice(pool, size=BENIGN_EXTENSION_SIZE, replace=False).tolist())
    taken |= set(benign_ext)
    screen: list[int] = []
    for name in STRATA:
        pool = np.array(sorted(set(np.flatnonzero(stratum == name)) - taken))
        screen.extend(sorted(rng.choice(pool, size=SCREENING_RULE.per_stratum, replace=False).tolist()))

    contributions = pillar_contributions(features.iloc[locked_index + benign_ext + screen], table)
    common = dict(features=features, raw_test=raw_test, contributions=contributions, scores=scores,
                  gate=gate, labels=labels, stratum=stratum, threshold=threshold)

    per_stratum = int(locked_manifest["per_stratum"])
    rebuilt = [
        json.dumps(build_stimulus(item["stimulus_id"], index, population[str(stratum[index])] / per_stratum,
                                  **common), sort_keys=True)
        for item, index in zip(locked_items, locked_index)
    ]
    mismatches = [item["stimulus_id"] for item, line, again in zip(locked_items, locked_lines, rebuilt)
                  if line != again]
    if mismatches:
        raise RuntimeError(f"builder does not reproduce {len(mismatches)} locked stimuli: {mismatches[:5]}")

    benign_records = [
        build_stimulus(f"SALADX_{BENIGN_EXTENSION_STRATUM}_{pos:03d}", index,
                       population[BENIGN_EXTENSION_STRATUM] / BENIGN_EXTENSION_SIZE, **common)
        for pos, index in enumerate(benign_ext)
    ]
    screen_records = [
        build_stimulus(f"SCREEN_{stratum[index]}_{pos:03d}", index,
                       population[str(stratum[index])] / SCREENING_RULE.per_stratum, **common)
        for pos, index in enumerate(screen)
    ]
    write_jsonl(out_dir / "salad_llm_ext_benign_stimuli.jsonl", benign_records)
    write_jsonl(out_dir / "salad_llm_screen_stimuli.jsonl", screen_records)
    write_json(out_dir / "salad_llm_ext_stimuli_manifest.json", {
        "seed": EXTENSION_SEED,
        "benign_extension": {"stratum": BENIGN_EXTENSION_STRATUM, "n": len(benign_records)},
        "screening_per_stratum": SCREENING_RULE.per_stratum,
        "population_strata": population,
        "refit_reproduction_max_abs_diff": max_abs_diff,
        "locked_stimuli_rebuilt_identically": len(rebuilt),
        "disjoint_from_locked_and_each_other": len(set(locked_index) | set(benign_ext) | set(screen))
        == len(locked_index) + len(benign_ext) + len(screen),
    })
    logger.info("rebuilt %d locked stimuli identically; wrote %d benign and %d screening stimuli",
                len(rebuilt), len(benign_records), len(screen_records))


if __name__ == "__main__":
    main()
