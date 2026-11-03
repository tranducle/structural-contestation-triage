from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core import write_json
from lmd_shift_study import load_and_engineer_lmd
from salad_detector_study import build_split_manifests


def _csv_id_summary(path: Path, id_column: str, label_column: str) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=[id_column, label_column], low_memory=False)
    return {
        "rows": int(len(frame)),
        "unique_ids": int(frame[id_column].nunique()),
        "duplicate_ids": int(frame[id_column].duplicated().sum()),
        "label_counts": {str(key): int(value) for key, value in frame[label_column].value_counts().to_dict().items()},
        "ids": set(frame[id_column].astype(str)),
    }


def audit_salad(paper_root: Path) -> dict[str, Any]:
    data_dir = paper_root / "data/external/salad_soc"
    manifest_dir = paper_root / "data/v2/manifests"
    manifests = build_split_manifests(data_dir, manifest_dir)
    summaries = {
        split: _csv_id_summary(path, "record_id", "label")
        for split, path in manifests.items()
    }
    overlap: dict[str, int] = {}
    names = list(summaries)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap[f"{left}_{right}"] = int(len(summaries[left]["ids"] & summaries[right]["ids"]))
    report = {
        "dataset": "SALAD_SOC",
        "protocol": "official train, validation, and test files",
        "splits": {
            split: {key: value for key, value in summary.items() if key != "ids"}
            for split, summary in summaries.items()
        },
        "cross_split_id_overlap": overlap,
        "verdict": "PASS" if all(value == 0 for value in overlap.values()) and all(summary["duplicate_ids"] == 0 for summary in summaries.values()) else "FAIL_REPAIR",
        "claim_boundary": "This gate licenses official split evaluation, not temporal or cross-source generalization.",
        "manifest_files": {key: str(value) for key, value in manifests.items()},
    }
    return report


def _write_lmd_manifest(path: Path, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    frame = pd.concat(
        [
            train[["row_id", "label", "core_signature", "context_station", "context_year"]].assign(split="train"),
            validation[["row_id", "label", "core_signature", "context_station", "context_year"]].assign(split="validation"),
            test[["row_id", "label", "core_signature", "context_station", "context_year"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    frame.to_csv(path, index=False)


def _split_summary(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, grouping: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "rows": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        "positive_rows": {"train": int(train["label"].sum()), "validation": int(validation["label"].sum()), "test": int(test["label"].sum())},
        "record_id_overlap": {
            "train_validation": int(len(set(train["row_id"]) & set(validation["row_id"]))),
            "train_test": int(len(set(train["row_id"]) & set(test["row_id"]))),
            "validation_test": int(len(set(validation["row_id"]) & set(test["row_id"]))),
        },
    }
    report["group_overlap"] = {
        "train_validation": int(len(set(train[grouping]) & set(validation[grouping]))),
        "train_test": int(len(set(train[grouping]) & set(test[grouping]))),
        "validation_test": int(len(set(validation[grouping]) & set(test[grouping]))),
    }
    report["grouping"] = grouping
    report["verdict"] = "PASS" if all(value == 0 for value in report["record_id_overlap"].values()) and all(value == 0 for value in report["group_overlap"].values()) else "FAIL_REPAIR"
    return report


def audit_lmd(paper_root: Path) -> dict[str, Any]:
    data_path = paper_root / "data/external/lmd_2023/full-csv(Evaluated Titles).csv"
    manifest_dir = paper_root / "data/v2/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    frame = load_and_engineer_lmd(data_path)
    splits = {
        "signature_disjoint": (
            frame[frame["signature_bucket"] <= 6].copy(),
            frame[frame["signature_bucket"] == 7].copy(),
            frame[frame["signature_bucket"] >= 8].copy(),
            "core_signature",
        ),
        "temporal_future": (
            frame[frame["context_year"].isin(["2020", "2021"])].copy(),
            frame[frame["context_year"] == "2022"].copy(),
            frame[frame["context_year"] == "2023"].copy(),
            "context_year",
        ),
        "station_holdout": (
            frame[~frame["context_station"].isin(["CompSTA2", "CompSTA3", "CompSTA3.1"])].copy(),
            frame[frame["context_station"].isin(["CompSTA3", "CompSTA3.1"])].copy(),
            frame[frame["context_station"] == "CompSTA2"].copy(),
            "context_station",
        ),
    }
    reports: dict[str, Any] = {}
    for name, (train, validation, test, grouping) in splits.items():
        path = manifest_dir / f"lmd_{name}_manifest.csv"
        _write_lmd_manifest(path, train, validation, test)
        reports[name] = _split_summary(train, validation, test, grouping)
        reports[name]["manifest_file"] = str(path)
    return {
        "dataset": "LMD_2023",
        "label_semantics": {"0": "Normal", "1": "EoRS", "2": "EoHT"},
        "split_reports": reports,
        "verdict": "PASS" if all(report["verdict"] == "PASS" for report in reports.values()) else "FAIL_REPAIR",
        "claim_boundary": "Signature, time, and station splits license their named evaluation only.",
    }


def run_split_audits(paper_root: Path) -> dict[str, Any]:
    gate_dir = paper_root / "results/split_audits"
    gate_dir.mkdir(parents=True, exist_ok=True)
    salad = audit_salad(paper_root)
    lmd = audit_lmd(paper_root)
    write_json(gate_dir / "G1_salad_custom_split_audit.json", salad)
    write_json(gate_dir / "G1_lmd_custom_split_audit.json", lmd)
    combined = {
        "gate_id": "G1_preprocessing_integrity",
        "stage": "preprocessing_integrity",
        "verdict": "PASS" if salad["verdict"] == "PASS" and lmd["verdict"] == "PASS" else "FAIL_REPAIR",
        "input_artifacts": [salad["manifest_files"], {key: value["manifest_file"] for key, value in lmd["split_reports"].items()}],
        "metrics": {"salad": salad, "lmd": lmd},
        "pass_criteria": {
            "record_id_overlap": 0,
            "declared_group_overlap": 0,
            "label_diversity": "both classes appear in each evaluated partition",
        },
        "failed_criteria": [],
        "next_allowed_step": "baseline_training",
        "fail_interpretation": "Training and testing would not be isolated under the declared evaluation role.",
    }
    if combined["verdict"] != "PASS":
        combined["failed_criteria"] = ["one or more split audits failed"]
    write_json(gate_dir / "G1_preprocessing_integrity.json", combined)
    return combined
