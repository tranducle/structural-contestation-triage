from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import gc
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from core import (
    binary_metrics,
    choose_threshold,
    cluster_bootstrap_ci,
    exact_mcnemar,
    fit_llr_table,
    holm_adjust,
    package_versions,
    score_llr,
    stable_bucket,
    write_json,
)


CORE_COLUMNS = ["p1_process_events", "p2_registry_file", "p3_network_pipe", "p4_destination_service"]
CONTEXT_COLUMNS = ["context_station", "context_year", "context_hour"]


def _bitmask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    accumulator = np.zeros(len(frame), dtype=np.int64)
    for index, column in enumerate(columns):
        accumulator |= (
            frame[column].to_numpy(dtype=np.float64) > 0.5
        ).astype(np.int64) << index
    return pd.Series(accumulator, index=frame.index).astype(str)


def _one_hot_name(frame: pd.DataFrame, columns: Sequence[str], prefix: str) -> pd.Series:
    matrix = frame[list(columns)].to_numpy(dtype=np.float32)
    selected = matrix.argmax(axis=1)
    present = matrix.max(axis=1) > 0.5
    names = np.array([column.replace(prefix, "") for column in columns], dtype=object)
    return pd.Series(np.where(present, names[selected], "none"), index=frame.index, dtype="string")


def load_and_engineer_lmd(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    event_columns = [column for column in header if column.startswith("EventID")]
    station_columns = [column for column in header if column.startswith("CompSTA")]
    destination_columns = [column for column in header if column.startswith("DstPortName_")]
    year_columns = [column for column in header if column.startswith("SysTimeYear_")]
    use_columns = list(
        dict.fromkeys(
            ["EventRecID", "SysTimehour", "Label"]
            + event_columns
            + station_columns
            + destination_columns
            + year_columns
            + ["Init_True", "SrcIpv6_False", "SrcIpv6_True"]
        )
    )
    raw = pd.read_csv(path, usecols=use_columns, low_memory=False)
    label_class = pd.to_numeric(raw["Label"], errors="coerce").fillna(0).astype(np.int8)
    frame = pd.DataFrame(index=raw.index)
    frame["row_id"] = np.arange(len(raw), dtype=np.int64).astype(str)
    frame["label_class"] = label_class
    frame["label"] = (label_class > 0).astype(np.int8)

    process_events = [column for column in ["EventID1", "EventID5", "EventID6", "EventID7", "EventID8", "EventID10"] if column in raw.columns]
    registry_file = [column for column in ["EventID2", "EventID11", "EventID12", "EventID13", "EventID15", "EventID16", "EventID23", "EventID255"] if column in raw.columns]
    network_pipe = [column for column in ["EventID3", "EventID17", "EventID18", "EventID22", "Init_True", "SrcIpv6_False", "SrcIpv6_True"] if column in raw.columns]
    frame["p1_process_events"] = _bitmask(raw, process_events)
    frame["p2_registry_file"] = _bitmask(raw, registry_file)
    frame["p3_network_pipe"] = _bitmask(raw, network_pipe)
    frame["p4_destination_service"] = _bitmask(raw, destination_columns)

    station_names = [column for column in raw.columns if column.startswith("CompSTA")]
    year_names = [column for column in raw.columns if column.startswith("SysTimeYear_")]
    frame["context_station"] = _one_hot_name(raw, station_names, "")
    frame["context_year"] = _one_hot_name(raw, year_names, "SysTimeYear_")
    hour = pd.cut(
        pd.to_numeric(raw["SysTimehour"], errors="coerce").fillna(0),
        bins=[-0.1, 0.25, 0.5, 0.75, 1.1],
        labels=["hour_q1", "hour_q2", "hour_q3", "hour_q4"],
    ).astype(str)
    frame["context_hour"] = hour
    frame["core_text"] = (
        "process=" + frame["p1_process_events"]
        + " registry_file=" + frame["p2_registry_file"]
        + " network_pipe=" + frame["p3_network_pipe"]
        + " destination=" + frame["p4_destination_service"]
    )
    frame["context_text"] = (
        "station=" + frame["context_station"].astype(str)
        + " year=" + frame["context_year"].astype(str)
        + " hour=" + frame["context_hour"].astype(str)
    )
    frame["augmented_text"] = frame["core_text"] + " " + frame["context_text"]
    frame["core_signature"] = frame[CORE_COLUMNS].astype(str).agg("|".join, axis=1)
    frame["cluster_id"] = (
        frame["context_station"].astype(str)
        + "_"
        + frame["context_year"].astype(str)
        + "_"
        + frame["context_hour"].astype(str)
    )
    frame["signature_bucket"] = frame["core_signature"].map(lambda value: stable_bucket(value, 10)).astype(np.int8)
    return frame.reset_index(drop=True)


def _sample_training(frame: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= target_rows:
        return frame.copy()
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    counts = frame["label"].value_counts().to_dict()
    for label, count in sorted(counts.items()):
        requested = int(round(target_rows * count / len(frame)))
        group = frame[frame["label"] == label]
        indices = rng.choice(group.index.to_numpy(), size=min(requested, len(group)), replace=False)
        pieces.append(group.loc[indices])
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _metric_row(split: str, model: str, seed: int | str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"dataset": "LMD_2023", "split": split, "model": model, "condition": "clean", "seed": seed, **metrics}


def _select_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    text_column: str,
    config: dict[str, Any],
) -> tuple[TfidfVectorizer, float, float, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\S+",
        max_features=6000,
        min_df=2,
        norm="l2",
    )
    x_train = vectorizer.fit_transform(train[text_column])
    x_validation = vectorizer.transform(validation[text_column])
    y_train = train["label"].to_numpy(dtype=np.int8)
    y_validation = validation["label"].to_numpy(dtype=np.int8)
    candidates: list[dict[str, Any]] = []
    for alpha in config["sgd_alpha_grid"]:
        model = SGDClassifier(
            loss="log_loss",
            alpha=float(alpha),
            class_weight="balanced",
            max_iter=60,
            tol=1e-4,
            random_state=int(config["seeds"][0]),
        )
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_validation)[:, 1]
        threshold_choice = choose_threshold(
            y_validation,
            scores,
            max_fpr=float(config["validation_max_fpr"]),
            objective=str(config["threshold_objective"]),
        )
        candidates.append({"alpha": float(alpha), "choice": threshold_choice})
    best = max(
        candidates,
        key=lambda item: (
            float(item["choice"].metrics["f1"]),
            float(item["choice"].metrics["recall"]),
            -float(item["choice"].metrics["fpr"]),
        ),
    )
    return vectorizer, best["alpha"], best["choice"].threshold, best["choice"].metrics


def _evaluate_split_family(
    split_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    tuning: dict[str, Any] = {}
    predictions: dict[str, dict[str, Any]] = {}
    sample = _sample_training(train, int(config["train_sample_rows"]), int(config["seeds"][0]))
    y_validation = validation["label"].to_numpy(dtype=np.int8)
    y_test = test["label"].to_numpy(dtype=np.int8)

    pit_candidates: list[dict[str, Any]] = []
    for alpha in config["llr_alpha_grid"]:
        table = fit_llr_table(train, CORE_COLUMNS, "label", float(alpha))
        validation_scores = score_llr(validation, table, CORE_COLUMNS)
        choice = choose_threshold(
            y_validation,
            validation_scores,
            max_fpr=float(config["validation_max_fpr"]),
            objective=str(config["threshold_objective"]),
        )
        pit_candidates.append({"alpha": float(alpha), "table": table, "choice": choice})
    pit_best = max(
        pit_candidates,
        key=lambda item: (
            float(item["choice"].metrics["f1"]),
            float(item["choice"].metrics["recall"]),
            -float(item["choice"].metrics["fpr"]),
        ),
    )
    pit_scores = score_llr(test, pit_best["table"], CORE_COLUMNS)
    pit_metrics = binary_metrics(y_test, pit_scores, pit_best["choice"].threshold)
    rows.append(_metric_row(split_name, "PITSA_Core", "deterministic", pit_metrics))
    predictions["PITSA_Core"] = {"scores": pit_scores, "threshold": pit_best["choice"].threshold}
    tuning["PITSA_Core"] = {
        "selected_alpha": pit_best["alpha"],
        "threshold": pit_best["choice"].threshold,
        "validation_metrics": pit_best["choice"].metrics,
    }

    for model_name, text_column in [
        ("CoreLogistic", "core_text"),
        ("ContextOnlyLogistic", "context_text"),
        ("ContextAugmentedLogistic", "augmented_text"),
    ]:
        vectorizer, alpha, threshold, validation_metrics = _select_model(sample, validation, text_column, config)
        x_train = vectorizer.transform(sample[text_column])
        x_test = vectorizer.transform(test[text_column])
        y_train = sample["label"].to_numpy(dtype=np.int8)
        tuning[model_name] = {
            "selected_alpha": alpha,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
        }
        for index, seed in enumerate(config["seeds"]):
            model = SGDClassifier(
                loss="log_loss",
                alpha=alpha,
                class_weight="balanced",
                max_iter=60,
                tol=1e-4,
                random_state=int(seed),
            )
            model.fit(x_train, y_train)
            scores = model.predict_proba(x_test)[:, 1]
            rows.append(_metric_row(split_name, model_name, int(seed), binary_metrics(y_test, scores, threshold)))
            if index == 0:
                predictions[model_name] = {"scores": scores, "threshold": threshold}
        del x_train, x_test
        gc.collect()

    rf_train = _sample_training(train, int(config["random_forest_train_rows"]), int(config["seeds"][0]))
    x_rf_train = np.column_stack(
        [
            pd.util.hash_pandas_object(rf_train[column].astype(str), index=False).to_numpy(dtype=np.uint64) % np.uint64(1048573)
            for column in CORE_COLUMNS
        ]
    ).astype(np.float32)
    x_rf_validation = np.column_stack(
        [
            pd.util.hash_pandas_object(validation[column].astype(str), index=False).to_numpy(dtype=np.uint64) % np.uint64(1048573)
            for column in CORE_COLUMNS
        ]
    ).astype(np.float32)
    x_rf_test = np.column_stack(
        [
            pd.util.hash_pandas_object(test[column].astype(str), index=False).to_numpy(dtype=np.uint64) % np.uint64(1048573)
            for column in CORE_COLUMNS
        ]
    ).astype(np.float32)
    x_rf_train /= 1048573.0
    x_rf_validation /= 1048573.0
    x_rf_test /= 1048573.0
    rf_candidates: list[dict[str, Any]] = []
    for depth in config["random_forest_depth_grid"]:
        model = RandomForestClassifier(
            n_estimators=int(config["random_forest_trees"]),
            max_depth=int(depth),
            class_weight="balanced_subsample",
            random_state=int(config["seeds"][0]),
            n_jobs=-1,
        )
        model.fit(x_rf_train, rf_train["label"].to_numpy(dtype=np.int8))
        scores = model.predict_proba(x_rf_validation)[:, 1]
        choice = choose_threshold(
            y_validation,
            scores,
            max_fpr=float(config["validation_max_fpr"]),
            objective=str(config["threshold_objective"]),
        )
        rf_candidates.append({"depth": int(depth), "choice": choice})
    rf_best = max(
        rf_candidates,
        key=lambda item: (
            float(item["choice"].metrics["f1"]),
            float(item["choice"].metrics["recall"]),
            -float(item["choice"].metrics["fpr"]),
        ),
    )
    tuning["CoreRandomForest"] = {
        "selected_max_depth": rf_best["depth"],
        "threshold": rf_best["choice"].threshold,
        "validation_metrics": rf_best["choice"].metrics,
    }
    for index, seed in enumerate(config["seeds"]):
        model = RandomForestClassifier(
            n_estimators=int(config["random_forest_trees"]),
            max_depth=rf_best["depth"],
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
        model.fit(x_rf_train, rf_train["label"].to_numpy(dtype=np.int8))
        scores = model.predict_proba(x_rf_test)[:, 1]
        rows.append(_metric_row(split_name, "CoreRandomForest", int(seed), binary_metrics(y_test, scores, rf_best["choice"].threshold)))
        if index == 0:
            predictions["CoreRandomForest"] = {"scores": scores, "threshold": rf_best["choice"].threshold}
    return rows, tuning, predictions


def run_lmd_pipeline(paper_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    data_path = paper_root / "data/external/lmd_2023/full-csv(Evaluated Titles).csv"
    output_dir = paper_root / "results/detector/lmd"
    result_dir = paper_root / "results/detector"
    manifest_dir = paper_root / "data/v2/manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    frame = load_and_engineer_lmd(data_path)
    signature_train = frame[frame["signature_bucket"] <= 6].copy()
    signature_validation = frame[frame["signature_bucket"] == 7].copy()
    signature_test = frame[frame["signature_bucket"] >= 8].copy()
    temporal_train = frame[frame["context_year"].isin(["2020", "2021"])].copy()
    temporal_validation = frame[frame["context_year"] == "2022"].copy()
    temporal_test = frame[frame["context_year"] == "2023"].copy()
    station_train = frame[~frame["context_station"].isin(["CompSTA2", "CompSTA3", "CompSTA3.1"])].copy()
    station_validation = frame[frame["context_station"].isin(["CompSTA3", "CompSTA3.1"])].copy()
    station_test = frame[frame["context_station"] == "CompSTA2"].copy()

    split_sets = {
        "signature_disjoint": (signature_train, signature_validation, signature_test),
        "temporal_future": (temporal_train, temporal_validation, temporal_test),
        "station_holdout": (station_train, station_validation, station_test),
    }
    split_report: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_tuning: dict[str, Any] = {}
    all_predictions: dict[str, dict[str, dict[str, Any]]] = {}

    for split_name, (train, validation, test) in split_sets.items():
        split_report[split_name] = {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "train_positive": int(train["label"].sum()),
            "validation_positive": int(validation["label"].sum()),
            "test_positive": int(test["label"].sum()),
            "core_signature_overlap_train_validation": int(len(set(train["core_signature"]) & set(validation["core_signature"]))),
            "core_signature_overlap_train_test": int(len(set(train["core_signature"]) & set(test["core_signature"]))),
        }
        if min(train["label"].nunique(), validation["label"].nunique(), test["label"].nunique()) < 2:
            split_report[split_name]["status"] = "not_evaluable_two_class_requirement"
            continue
        rows, tuning, predictions = _evaluate_split_family(split_name, train, validation, test, config)
        all_rows.extend(rows)
        all_tuning[split_name] = tuning
        all_predictions[split_name] = predictions
        y_test = test["label"].to_numpy(dtype=np.int8)
        for model_name, prediction in predictions.items():
            scores = np.asarray(prediction["scores"])
            threshold = float(prediction["threshold"])
            predicted = (scores >= threshold).astype(np.int8)
            class_metrics = {}
            for label_class, class_name in [(1, "EoRS"), (2, "EoHT")]:
                mask = test["label_class"].to_numpy(dtype=np.int8) == label_class
                class_metrics[f"{class_name}_rows"] = int(mask.sum())
                class_metrics[f"{class_name}_recall"] = float(np.mean(predicted[mask] == 1)) if mask.any() else None
            prediction["subclass_metrics"] = class_metrics

    metrics_frame = pd.DataFrame(all_rows)
    metrics_frame.to_csv(result_dir / "lmd_metrics_long.csv", index=False)
    aggregate = (
        metrics_frame.groupby(["dataset", "split", "model", "condition"], dropna=False)
        .agg(
            runs=("f1", "size"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_sd=("fpr", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
        )
        .reset_index()
    )
    aggregate.to_csv(result_dir / "lmd_metrics_aggregate.csv", index=False)

    primary_split = "signature_disjoint"
    primary_test = signature_test
    primary_y = primary_test["label"].to_numpy(dtype=np.int8)
    primary_predictions = all_predictions.get(primary_split, {})
    statistical: dict[str, Any] = {}
    if "PITSA_Core" in primary_predictions:
        pit = primary_predictions["PITSA_Core"]
        statistical["pitsa_cluster_bootstrap"] = cluster_bootstrap_ci(
            primary_y,
            np.asarray(pit["scores"]),
            float(pit["threshold"]),
            primary_test["cluster_id"].astype(str).tolist(),
            int(config["bootstrap_resamples"]),
            int(config["seeds"][0]),
        )
        reference_prediction = (np.asarray(pit["scores"]) >= float(pit["threshold"])).astype(np.int8)
        tests = []
        p_values = []
        for model_name, payload in primary_predictions.items():
            if model_name == "PITSA_Core":
                continue
            comparison_prediction = (np.asarray(payload["scores"]) >= float(payload["threshold"])).astype(np.int8)
            result = exact_mcnemar(primary_y, reference_prediction, comparison_prediction)
            p_values.append(float(result["exact_two_sided_p"]))
            tests.append({"reference": "PITSA_Core", "comparator": model_name, **result})
        for item, adjusted in zip(tests, holm_adjust(p_values)):
            item["holm_adjusted_p"] = float(adjusted)
        statistical["paired_exact_mcnemar"] = tests

    for split_name, (train, validation, test) in split_sets.items():
        manifest = pd.concat(
            [
                train[["row_id", "label", "core_signature"]].assign(split="train"),
                validation[["row_id", "label", "core_signature"]].assign(split="validation"),
                test[["row_id", "label", "core_signature"]].assign(split="test"),
            ],
            ignore_index=True,
        )
        manifest.to_csv(manifest_dir / f"lmd_{split_name}_manifest.csv", index=False)

    npz_payload: dict[str, Any] = {
        "row_id": primary_test["row_id"].astype(str).to_numpy(),
        "label": primary_y,
        "label_class": primary_test["label_class"].to_numpy(dtype=np.int8),
        "cluster_id": primary_test["cluster_id"].astype(str).to_numpy(),
    }
    for model_name, payload in primary_predictions.items():
        npz_payload[f"{model_name}_scores"] = np.asarray(payload["scores"])
    np.savez_compressed(output_dir / "signature_disjoint_primary_predictions.npz", **npz_payload)

    write_json(output_dir / "tuning_report.json", all_tuning)
    write_json(
        result_dir / "lmd_results.json",
        {
            "protocol": {
                "primary_split": "core signature disjoint",
                "robustness_splits": ["temporal future", "station holdout"],
                "label_semantics": {"0": "Normal", "1": "EoRS", "2": "EoHT"},
                "primary_features": CORE_COLUMNS,
                "context_features_only_for_diagnostic_baselines": CONTEXT_COLUMNS,
                "package_versions": package_versions(),
            },
            "split_report": split_report,
            "tuning": all_tuning,
            "subclass_metrics": {
                split_name: {
                    model_name: payload.get("subclass_metrics", {})
                    for model_name, payload in predictions.items()
                }
                for split_name, predictions in all_predictions.items()
            },
            "statistical_validation": statistical,
            "metrics_long_file": str(result_dir / "lmd_metrics_long.csv"),
            "metrics_aggregate_file": str(result_dir / "lmd_metrics_aggregate.csv"),
            "elapsed_seconds": float(time.time() - started),
        },
    )
    return {
        "results_file": str(result_dir / "lmd_results.json"),
        "metrics_file": str(result_dir / "lmd_metrics_long.csv"),
        "aggregate_file": str(result_dir / "lmd_metrics_aggregate.csv"),
        "elapsed_seconds": float(time.time() - started),
    }
