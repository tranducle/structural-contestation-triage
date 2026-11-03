from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core import binary_metrics, choose_threshold, fit_llr_from_counters, fit_llr_table, score_llr, write_json
from lmd_shift_study import CORE_COLUMNS, load_and_engineer_lmd
from salad_detector_study import STRUCT_COLUMNS, USE_COLUMNS, _to_label, engineer_salad


def _update_salad_counters(
    counters: dict[str, dict[int, Counter]],
    totals: Counter,
    frame: pd.DataFrame,
) -> None:
    labels = frame["label"].to_numpy(dtype=np.int8)
    for label in [0, 1]:
        mask = labels == label
        totals[label] += int(mask.sum())
        for column in STRUCT_COLUMNS:
            counters[column][label].update(frame.loc[mask, column].astype(str))


def _salad_full_counters(train_path: Path) -> tuple[dict[str, dict[int, Counter]], Counter]:
    counters = {column: {0: Counter(), 1: Counter()} for column in STRUCT_COLUMNS}
    totals: Counter = Counter()
    for raw in pd.read_csv(train_path, usecols=USE_COLUMNS, chunksize=200000, low_memory=False):
        _update_salad_counters(counters, totals, engineer_salad(raw))
    return counters, totals


def _best_llr_configuration(
    train_frame: pd.DataFrame | None,
    counters: dict[str, dict[int, Counter]] | None,
    totals: Counter | None,
    validation: pd.DataFrame,
    columns: Sequence[str],
    label_column: str,
    alpha_grid: Sequence[float],
    max_fpr: float,
    objective: str,
) -> dict[str, Any]:
    y_validation = validation[label_column].to_numpy(dtype=np.int8)
    candidates: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        if counters is not None and totals is not None:
            table = fit_llr_from_counters(counters, totals, float(alpha))
        elif train_frame is not None:
            table = fit_llr_table(train_frame, columns, label_column, float(alpha))
        else:
            raise ValueError("training evidence is missing")
        validation_scores = score_llr(validation, table, columns)
        threshold = choose_threshold(
            y_validation,
            validation_scores,
            max_fpr=max_fpr,
            objective=objective,
        )
        candidates.append({"alpha": float(alpha), "table": table, "choice": threshold})
    return max(
        candidates,
        key=lambda item: (
            float(item["choice"].metrics["f1"]),
            float(item["choice"].metrics["recall"]),
            -float(item["choice"].metrics["fpr"]),
        ),
    )


def run_salad_ablation(paper_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    data_dir = paper_root / "data/external/salad_soc"
    counters, totals = _salad_full_counters(data_dir / "salad_train.csv")
    validation = engineer_salad(pd.read_csv(data_dir / "salad_val.csv", usecols=USE_COLUMNS, low_memory=False))
    test = engineer_salad(pd.read_csv(data_dir / "salad_test.csv", usecols=USE_COLUMNS, low_memory=False))
    y_test = test["label"].to_numpy(dtype=np.int8)
    configurations: dict[str, list[str]] = {"Full": list(STRUCT_COLUMNS)}
    for column in STRUCT_COLUMNS:
        configurations["Drop_" + column] = [item for item in STRUCT_COLUMNS if item != column]
        configurations["Only_" + column] = [column]
    rows: list[dict[str, Any]] = []
    for name, columns in configurations.items():
        best = _best_llr_configuration(
            None,
            counters,
            totals,
            validation,
            columns,
            "label",
            config["llr_alpha_grid"],
            float(config["validation_max_fpr"]),
            str(config["threshold_objective"]),
        )
        test_scores = score_llr(test, best["table"], columns)
        metrics = binary_metrics(y_test, test_scores, best["choice"].threshold)
        rows.append(
            {
                "dataset": "SALAD_SOC",
                "split": "official_test",
                "configuration": name,
                "active_columns": ",".join(columns),
                "selected_alpha": best["alpha"],
                **metrics,
            }
        )
    return rows


def run_lmd_ablation(paper_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = paper_root / "data/external/lmd_2023/full-csv(Evaluated Titles).csv"
    frame = load_and_engineer_lmd(path)
    train = frame[frame["signature_bucket"] <= 6].copy()
    validation = frame[frame["signature_bucket"] == 7].copy()
    test = frame[frame["signature_bucket"] >= 8].copy()
    y_test = test["label"].to_numpy(dtype=np.int8)
    configurations: dict[str, list[str]] = {"Full": list(CORE_COLUMNS)}
    for column in CORE_COLUMNS:
        configurations["Drop_" + column] = [item for item in CORE_COLUMNS if item != column]
        configurations["Only_" + column] = [column]
    rows: list[dict[str, Any]] = []
    for name, columns in configurations.items():
        best = _best_llr_configuration(
            train,
            None,
            None,
            validation,
            columns,
            "label",
            config["llr_alpha_grid"],
            float(config["validation_max_fpr"]),
            str(config["threshold_objective"]),
        )
        test_scores = score_llr(test, best["table"], columns)
        metrics = binary_metrics(y_test, test_scores, best["choice"].threshold)
        prediction = (test_scores >= best["choice"].threshold).astype(np.int8)
        label_class = test["label_class"].to_numpy(dtype=np.int8)
        eors = label_class == 1
        eoht = label_class == 2
        rows.append(
            {
                "dataset": "LMD_2023",
                "split": "signature_disjoint",
                "configuration": name,
                "active_columns": ",".join(columns),
                "selected_alpha": best["alpha"],
                "EoRS_recall": float(np.mean(prediction[eors] == 1)) if eors.any() else None,
                "EoHT_recall": float(np.mean(prediction[eoht] == 1)) if eoht.any() else None,
                **metrics,
            }
        )
    return rows


def run_ablation_pipeline(paper_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    result_dir = paper_root / "results/detector"
    salad_rows = run_salad_ablation(paper_root, config)
    lmd_rows = run_lmd_ablation(paper_root, config)
    frame = pd.DataFrame(salad_rows + lmd_rows)
    output_path = result_dir / "ablation_results.csv"
    frame.to_csv(output_path, index=False)
    payload = {
        "protocol": {
            "SALAD": "official train, validation, and test",
            "LMD": "core signature disjoint train, validation, and test",
            "selection": "alpha and threshold selected on validation for each configuration",
            "isolation_rule": "one structural pillar is removed or retained while the split and tuning budget remain fixed",
        },
        "result_file": str(output_path),
        "rows": int(len(frame)),
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(result_dir / "ablation_results.json", payload)
    return payload
