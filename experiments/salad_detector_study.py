from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from core import (
    binary_metrics,
    choose_threshold,
    cluster_bootstrap_ci,
    exact_mcnemar,
    fit_llr_from_counters,
    fit_llr_table,
    holm_adjust,
    package_versions,
    score_llr,
    stable_bucket,
    write_json,
)


USE_COLUMNS = [
    "alert_id",
    "timestamp",
    "source_dataset",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "flow_bytes_per_sec",
    "network_segment",
    "is_malicious",
    "attack_category",
    "difficulty_level",
    "alert_count_1h",
    "alert_count_24h",
    "is_repeated_target",
]

STRUCT_COLUMNS = [
    "s1_proto_ports",
    "s2_flow_volume",
    "s3_duration_segment",
    "s4_burst",
    "s5_network_scope",
]

ATTACK_CLAUSES = [
    "authorized maintenance activity confirmed by change control",
    "routine encrypted client session with no escalation required",
    "approved diagnostic task and expected administrative behavior",
    "verified internal service health check marked as benign",
    "scheduled operations ticket confirms normal enterprise traffic",
    "trusted support workflow and low risk event disposition",
    "known false positive pattern approved for automatic dismissal",
    "standard infrastructure monitoring action with no incident impact",
]


def _to_label(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(np.int8)
    )


def _private_scope(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    is_private = (
        text.str.startswith("10.")
        | text.str.startswith("192.168.")
        | text.str.match(r"^172\.(1[6-9]|2[0-9]|3[0-1])\.")
    )
    return np.where(is_private, "private", "other")


def engineer_salad(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw.index)
    frame["record_id"] = raw["alert_id"].astype(str)
    frame["label"] = _to_label(raw["is_malicious"])
    frame["source_dataset"] = raw["source_dataset"].fillna("unknown").astype(str)
    frame["timestamp"] = raw["timestamp"].fillna("").astype(str)
    parsed_time = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["cluster_id"] = (
        frame["source_dataset"]
        + "_"
        + parsed_time.dt.strftime("%Y%m%d").fillna("unknown")
    )
    frame["difficulty_level"] = raw["difficulty_level"].fillna("unknown").astype(str)
    frame["attack_category"] = raw["attack_category"].fillna("unknown").astype(str)

    source_port = pd.cut(
        pd.to_numeric(raw["src_port"], errors="coerce").fillna(-1),
        bins=[-2, 0, 1024, 10000, 49151, 65536],
        labels=["sp_zero", "sp_privileged", "sp_registered", "sp_mid", "sp_ephemeral"],
    ).astype(str)
    destination_port = pd.cut(
        pd.to_numeric(raw["dst_port"], errors="coerce").fillna(-1),
        bins=[-2, 0, 25, 80, 443, 1024, 3389, 5986, 10000, 65536],
        labels=["dp_zero", "dp_shell", "dp_http", "dp_https", "dp_privileged", "dp_rdp", "dp_winrm", "dp_app", "dp_ephemeral"],
    ).astype(str)
    forward = pd.cut(
        pd.to_numeric(raw["total_fwd_packets"], errors="coerce").fillna(0),
        bins=[-1, 0, 2, 10, 50, 500, 1e12],
        labels=["f_zero", "f_tiny", "f_small", "f_medium", "f_large", "f_extreme"],
    ).astype(str)
    backward = pd.cut(
        pd.to_numeric(raw["total_bwd_packets"], errors="coerce").fillna(0),
        bins=[-1, 0, 2, 10, 50, 500, 1e12],
        labels=["b_zero", "b_tiny", "b_small", "b_medium", "b_large", "b_extreme"],
    ).astype(str)
    rate = pd.cut(
        pd.to_numeric(raw["flow_bytes_per_sec"], errors="coerce").fillna(0),
        bins=[-1, 0, 100, 1500, 20000, 500000, 1e15],
        labels=["r_zero", "r_low", "r_light", "r_medium", "r_high", "r_extreme"],
    ).astype(str)
    duration = pd.cut(
        pd.to_numeric(raw["flow_duration"], errors="coerce").fillna(0),
        bins=[-1, 0, 0.01, 0.5, 5, 60, 600, 1e12],
        labels=["d_zero", "d_tiny", "d_short", "d_medium", "d_long", "d_extended", "d_extreme"],
    ).astype(str)
    count_1h = pd.cut(
        pd.to_numeric(raw["alert_count_1h"], errors="coerce").fillna(0),
        bins=[-1, 0, 2, 10, 50, 500, 1e9],
        labels=["h1_zero", "h1_low", "h1_small", "h1_medium", "h1_high", "h1_extreme"],
    ).astype(str)
    count_24h = pd.cut(
        pd.to_numeric(raw["alert_count_24h"], errors="coerce").fillna(0),
        bins=[-1, 0, 5, 25, 100, 1000, 1e9],
        labels=["h24_zero", "h24_low", "h24_small", "h24_medium", "h24_high", "h24_extreme"],
    ).astype(str)
    protocol = raw["protocol"].fillna("unknown").astype(str).str.lower()
    segment = raw["network_segment"].fillna("unknown").astype(str).str.lower()
    repeated = raw["is_repeated_target"].fillna(False).astype(str).str.lower()
    source_scope = pd.Series(_private_scope(raw["src_ip"]), index=raw.index)
    destination_scope = pd.Series(_private_scope(raw["dst_ip"]), index=raw.index)

    frame["s1_proto_ports"] = protocol + "_" + source_port + "_" + destination_port
    frame["s2_flow_volume"] = forward + "_" + backward + "_" + rate
    frame["s3_duration_segment"] = duration + "_" + segment
    frame["s4_burst"] = count_1h + "_" + count_24h + "_rep_" + repeated
    frame["s5_network_scope"] = source_scope + "_to_" + destination_scope + "_" + segment

    frame["struct_text"] = (
        "proto_ports=" + frame["s1_proto_ports"]
        + " flow_volume=" + frame["s2_flow_volume"]
        + " duration_segment=" + frame["s3_duration_segment"]
        + " burst=" + frame["s4_burst"]
        + " network_scope=" + frame["s5_network_scope"]
    )
    frame["neutral_text"] = (
        "network event protocol " + protocol
        + " source port class " + source_port
        + " destination service class " + destination_port
        + " duration " + duration
        + " forward packets " + forward
        + " backward packets " + backward
        + " byte rate " + rate
        + " alert history " + count_1h + " " + count_24h
        + " target repeated " + repeated
        + " segment " + segment
    )
    frame["joint_text"] = frame["struct_text"] + " text_channel " + frame["neutral_text"]
    return frame.reset_index(drop=True)


def attacked_joint_text(frame: pd.DataFrame, clause_count: int) -> pd.Series:
    if clause_count <= 0:
        return frame["joint_text"].copy()
    values = frame["joint_text"].astype(str).to_numpy(dtype=object).copy()
    labels = frame["label"].to_numpy(dtype=np.int8)
    ids = frame["record_id"].astype(str).to_numpy(dtype=object)
    hostile_indices = np.flatnonzero(labels == 1)
    for index in hostile_indices:
        start = stable_bucket(str(ids[index]), len(ATTACK_CLAUSES))
        payload = " ".join(
            ATTACK_CLAUSES[(start + offset) % len(ATTACK_CLAUSES)]
            for offset in range(clause_count)
        )
        values[index] = str(values[index]) + " untrusted_note " + payload
    return pd.Series(values, index=frame.index, dtype="string")


def build_split_manifests(data_dir: Path, manifest_dir: Path) -> dict[str, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in [
        ("train", "salad_train.csv"),
        ("validation", "salad_val.csv"),
        ("test", "salad_test.csv"),
    ]:
        output = manifest_dir / f"salad_{split}_ids.csv"
        first = True
        for chunk in pd.read_csv(
            data_dir / filename,
            usecols=["alert_id", "is_malicious"],
            chunksize=250000,
            low_memory=False,
        ):
            small = pd.DataFrame(
                {
                    "record_id": chunk["alert_id"].astype(str),
                    "label": _to_label(chunk["is_malicious"]),
                }
            )
            small.to_csv(output, mode="w" if first else "a", header=first, index=False)
            first = False
        paths[split] = output
    return paths


def _update_counters(
    counters: dict[str, dict[int, Counter]],
    label_totals: Counter,
    frame: pd.DataFrame,
) -> None:
    labels = frame["label"].to_numpy(dtype=np.int8)
    for label in [0, 1]:
        mask = labels == label
        count = int(mask.sum())
        label_totals[label] += count
        if count == 0:
            continue
        for column in STRUCT_COLUMNS:
            counters[column][label].update(frame.loc[mask, column].astype(str))


def load_train_sample_and_full_counters(
    train_path: Path,
    target_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[int, Counter]], Counter]:
    counters = {column: {0: Counter(), 1: Counter()} for column in STRUCT_COLUMNS}
    label_totals: Counter = Counter()
    pieces: list[pd.DataFrame] = []
    expected_rows = 1944887
    selection_rate = min(0.35, max(0.02, target_rows * 1.35 / expected_rows))
    threshold = int(selection_rate * 1_000_000)
    for raw in pd.read_csv(
        train_path,
        usecols=USE_COLUMNS,
        chunksize=160000,
        low_memory=False,
    ):
        features = engineer_salad(raw)
        _update_counters(counters, label_totals, features)
        hashed = pd.util.hash_pandas_object(features["record_id"], index=False).to_numpy(dtype=np.uint64)
        hashed = hashed ^ np.uint64(seed)
        mask = (hashed % np.uint64(1_000_000)) < np.uint64(threshold)
        if mask.any():
            pieces.append(features.loc[mask].copy())
    sample = pd.concat(pieces, ignore_index=True)
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for label, total in sorted(label_totals.items()):
        requested = int(round(target_rows * total / sum(label_totals.values())))
        group = sample[sample["label"] == label]
        if len(group) > requested:
            indices = rng.choice(group.index.to_numpy(), size=requested, replace=False)
            group = group.loc[indices]
        selected.append(group)
    sampled = pd.concat(selected, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled, counters, label_totals


def _metric_row(
    dataset: str,
    split: str,
    model: str,
    condition: str,
    seed: int | str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "model": model,
        "condition": condition,
        "seed": seed,
        **metrics,
    }


def _select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            float(item["threshold_choice"].metrics["f1"]),
            float(item["threshold_choice"].metrics["recall"]),
            -float(item["threshold_choice"].metrics["fpr"]),
        ),
    )


def _hashed_struct_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = []
    modulus = np.uint64(1048573)
    for column in STRUCT_COLUMNS:
        values = pd.util.hash_pandas_object(frame[column].astype(str), index=False).to_numpy(dtype=np.uint64)
        columns.append((values % modulus).astype(np.float32) / float(modulus))
    return np.column_stack(columns)


def run_salad_pipeline(
    paper_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    data_dir = paper_root / "data/external/salad_soc"
    output_dir = paper_root / "results/detector"
    result_dir = paper_root / "results/detector"
    manifest_dir = paper_root / "data/v2/manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    split_manifests = build_split_manifests(data_dir, manifest_dir)
    train_sample, full_counters, full_label_totals = load_train_sample_and_full_counters(
        data_dir / "salad_train.csv",
        int(config["train_sample_rows"]),
        int(config["seeds"][0]),
    )
    train_sample.to_pickle(output_dir / "train_sample.pkl")

    validation = engineer_salad(
        pd.read_csv(data_dir / "salad_val.csv", usecols=USE_COLUMNS, low_memory=False)
    )
    test = engineer_salad(
        pd.read_csv(data_dir / "salad_test.csv", usecols=USE_COLUMNS, low_memory=False)
    )
    y_train = train_sample["label"].to_numpy(dtype=np.int8)
    y_validation = validation["label"].to_numpy(dtype=np.int8)
    y_test = test["label"].to_numpy(dtype=np.int8)
    max_fpr = float(config["validation_max_fpr"])
    objective = str(config["threshold_objective"])
    rows: list[dict[str, Any]] = []
    tuning: dict[str, Any] = {}
    primary_predictions: dict[str, dict[str, Any]] = {}
    graded_curves: dict[str, list[dict[str, Any]]] = {}

    pit_full_candidates = []
    for alpha in config["llr_alpha_grid"]:
        table = fit_llr_from_counters(full_counters, full_label_totals, float(alpha))
        validation_scores = score_llr(validation, table, STRUCT_COLUMNS)
        choice = choose_threshold(y_validation, validation_scores, max_fpr=max_fpr, objective=objective)
        pit_full_candidates.append({"alpha": float(alpha), "table": table, "threshold_choice": choice})
    pit_full = _select_best_candidate(pit_full_candidates)
    pit_full_scores = score_llr(test, pit_full["table"], STRUCT_COLUMNS)
    pit_full_metrics = binary_metrics(y_test, pit_full_scores, pit_full["threshold_choice"].threshold)
    rows.append(_metric_row("SALAD_SOC", "official_test", "PITSA_FullTrain", "clean", "deterministic", pit_full_metrics))
    rows.append(_metric_row("SALAD_SOC", "official_test", "PITSA_FullTrain", "full_itpi", "deterministic", pit_full_metrics))
    primary_predictions["PITSA_FullTrain"] = {
        "clean_scores": pit_full_scores,
        "attack_scores": pit_full_scores,
        "threshold": pit_full["threshold_choice"].threshold,
    }
    tuning["PITSA_FullTrain"] = {
        "selected_alpha": pit_full["alpha"],
        "threshold": pit_full["threshold_choice"].threshold,
        "validation_metrics": pit_full["threshold_choice"].metrics,
    }

    pit_budget_candidates = []
    for alpha in config["llr_alpha_grid"]:
        table = fit_llr_table(train_sample, STRUCT_COLUMNS, "label", float(alpha))
        validation_scores = score_llr(validation, table, STRUCT_COLUMNS)
        choice = choose_threshold(y_validation, validation_scores, max_fpr=max_fpr, objective=objective)
        pit_budget_candidates.append({"alpha": float(alpha), "table": table, "threshold_choice": choice})
    pit_budget = _select_best_candidate(pit_budget_candidates)
    pit_budget_scores = score_llr(test, pit_budget["table"], STRUCT_COLUMNS)
    pit_budget_metrics = binary_metrics(y_test, pit_budget_scores, pit_budget["threshold_choice"].threshold)
    rows.append(_metric_row("SALAD_SOC", "official_test", "PITSA_BudgetMatched", "clean", "deterministic", pit_budget_metrics))
    rows.append(_metric_row("SALAD_SOC", "official_test", "PITSA_BudgetMatched", "full_itpi", "deterministic", pit_budget_metrics))
    primary_predictions["PITSA_BudgetMatched"] = {
        "clean_scores": pit_budget_scores,
        "attack_scores": pit_budget_scores,
        "threshold": pit_budget["threshold_choice"].threshold,
    }
    tuning["PITSA_BudgetMatched"] = {
        "selected_alpha": pit_budget["alpha"],
        "threshold": pit_budget["threshold_choice"].threshold,
        "validation_metrics": pit_budget["threshold_choice"].metrics,
    }

    full_attack_count = int(config["full_attack_clause_count"])
    attacked_test_text = attacked_joint_text(test, full_attack_count)

    structural_vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\S+",
        max_features=5000,
        min_df=2,
        norm="l2",
    )
    x_struct_train = structural_vectorizer.fit_transform(train_sample["struct_text"])
    x_struct_validation = structural_vectorizer.transform(validation["struct_text"])
    x_struct_test = structural_vectorizer.transform(test["struct_text"])
    structural_candidates = []
    for alpha in config["sgd_alpha_grid"]:
        model = SGDClassifier(
            loss="log_loss",
            alpha=float(alpha),
            class_weight="balanced",
            max_iter=60,
            tol=1e-4,
            random_state=int(config["seeds"][0]),
        )
        model.fit(x_struct_train, y_train)
        scores = model.predict_proba(x_struct_validation)[:, 1]
        choice = choose_threshold(y_validation, scores, max_fpr=max_fpr, objective=objective)
        structural_candidates.append({"alpha": float(alpha), "threshold_choice": choice})
    structural_best = _select_best_candidate(structural_candidates)
    tuning["StructuralLogistic"] = {
        "selected_alpha": structural_best["alpha"],
        "threshold": structural_best["threshold_choice"].threshold,
        "validation_metrics": structural_best["threshold_choice"].metrics,
    }
    for index, seed in enumerate(config["seeds"]):
        model = SGDClassifier(
            loss="log_loss",
            alpha=structural_best["alpha"],
            class_weight="balanced",
            max_iter=60,
            tol=1e-4,
            random_state=int(seed),
        )
        model.fit(x_struct_train, y_train)
        scores = model.predict_proba(x_struct_test)[:, 1]
        metrics = binary_metrics(y_test, scores, structural_best["threshold_choice"].threshold)
        rows.append(_metric_row("SALAD_SOC", "official_test", "StructuralLogistic", "clean", int(seed), metrics))
        rows.append(_metric_row("SALAD_SOC", "official_test", "StructuralLogistic", "full_itpi", int(seed), metrics))
        if index == 0:
            primary_predictions["StructuralLogistic"] = {
                "clean_scores": scores,
                "attack_scores": scores,
                "threshold": structural_best["threshold_choice"].threshold,
            }
    del x_struct_validation, x_struct_test
    gc.collect()

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=12000,
        min_df=3,
        sublinear_tf=True,
        norm="l2",
    )
    x_word_train = word_vectorizer.fit_transform(train_sample["joint_text"])
    x_word_validation = word_vectorizer.transform(validation["joint_text"])
    word_candidates = []
    for alpha in config["sgd_alpha_grid"]:
        model = SGDClassifier(
            loss="log_loss",
            alpha=float(alpha),
            class_weight="balanced",
            max_iter=60,
            tol=1e-4,
            random_state=int(config["seeds"][0]),
        )
        model.fit(x_word_train, y_train)
        scores = model.predict_proba(x_word_validation)[:, 1]
        choice = choose_threshold(y_validation, scores, max_fpr=max_fpr, objective=objective)
        word_candidates.append({"alpha": float(alpha), "threshold_choice": choice})
    word_best = _select_best_candidate(word_candidates)
    tuning["JointWordLogistic"] = {
        "selected_alpha": word_best["alpha"],
        "threshold": word_best["threshold_choice"].threshold,
        "validation_metrics": word_best["threshold_choice"].metrics,
    }
    x_word_test_clean = word_vectorizer.transform(test["joint_text"])
    x_word_test_attack = word_vectorizer.transform(attacked_test_text)
    word_models = []
    for index, seed in enumerate(config["seeds"]):
        model = SGDClassifier(
            loss="log_loss",
            alpha=word_best["alpha"],
            class_weight="balanced",
            max_iter=60,
            tol=1e-4,
            random_state=int(seed),
        )
        model.fit(x_word_train, y_train)
        clean_scores = model.predict_proba(x_word_test_clean)[:, 1]
        attack_scores = model.predict_proba(x_word_test_attack)[:, 1]
        threshold = word_best["threshold_choice"].threshold
        rows.append(_metric_row("SALAD_SOC", "official_test", "JointWordLogistic", "clean", int(seed), binary_metrics(y_test, clean_scores, threshold)))
        rows.append(_metric_row("SALAD_SOC", "official_test", "JointWordLogistic", "full_itpi", int(seed), binary_metrics(y_test, attack_scores, threshold)))
        word_models.append(model)
        if index == 0:
            primary_predictions["JointWordLogistic"] = {
                "clean_scores": clean_scores,
                "attack_scores": attack_scores,
                "threshold": threshold,
            }
    del x_word_validation, x_word_test_clean, x_word_test_attack
    gc.collect()

    naive_candidates = []
    for alpha in config["naive_bayes_alpha_grid"]:
        model = MultinomialNB(alpha=float(alpha))
        model.fit(x_word_train, y_train)
        scores = model.predict_proba(word_vectorizer.transform(validation["joint_text"]))[:, 1]
        choice = choose_threshold(y_validation, scores, max_fpr=max_fpr, objective=objective)
        naive_candidates.append({"alpha": float(alpha), "threshold_choice": choice})
    naive_best = _select_best_candidate(naive_candidates)
    naive_model = MultinomialNB(alpha=naive_best["alpha"])
    naive_model.fit(x_word_train, y_train)
    naive_clean = naive_model.predict_proba(word_vectorizer.transform(test["joint_text"]))[:, 1]
    naive_attack = naive_model.predict_proba(word_vectorizer.transform(attacked_test_text))[:, 1]
    naive_threshold = naive_best["threshold_choice"].threshold
    rows.append(_metric_row("SALAD_SOC", "official_test", "JointNaiveBayes", "clean", "deterministic", binary_metrics(y_test, naive_clean, naive_threshold)))
    rows.append(_metric_row("SALAD_SOC", "official_test", "JointNaiveBayes", "full_itpi", "deterministic", binary_metrics(y_test, naive_attack, naive_threshold)))
    primary_predictions["JointNaiveBayes"] = {
        "clean_scores": naive_clean,
        "attack_scores": naive_attack,
        "threshold": naive_threshold,
    }
    tuning["JointNaiveBayes"] = {
        "selected_alpha": naive_best["alpha"],
        "threshold": naive_threshold,
        "validation_metrics": naive_best["threshold_choice"].metrics,
    }
    del x_word_train
    gc.collect()

    char_vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=2 ** 16,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    x_char_train = char_vectorizer.transform(train_sample["joint_text"])
    x_char_validation = char_vectorizer.transform(validation["joint_text"])
    char_candidates = []
    for alpha in config["sgd_alpha_grid"]:
        model = SGDClassifier(
            loss="log_loss",
            alpha=float(alpha),
            class_weight="balanced",
            max_iter=60,
            tol=1e-4,
            random_state=int(config["seeds"][0]),
        )
        model.fit(x_char_train, y_train)
        scores = model.predict_proba(x_char_validation)[:, 1]
        choice = choose_threshold(y_validation, scores, max_fpr=max_fpr, objective=objective)
        char_candidates.append({"alpha": float(alpha), "threshold_choice": choice})
    char_best = _select_best_candidate(char_candidates)
    tuning["CharacterNgramLogistic"] = {
        "selected_alpha": char_best["alpha"],
        "threshold": char_best["threshold_choice"].threshold,
        "validation_metrics": char_best["threshold_choice"].metrics,
    }
    x_char_test_clean = char_vectorizer.transform(test["joint_text"])
    x_char_test_attack = char_vectorizer.transform(attacked_test_text)
    char_models = []
    for index, seed in enumerate(config["seeds"]):
        model = SGDClassifier(
            loss="log_loss",
            alpha=char_best["alpha"],
            class_weight="balanced",
            max_iter=60,
            tol=1e-4,
            random_state=int(seed),
        )
        model.fit(x_char_train, y_train)
        clean_scores = model.predict_proba(x_char_test_clean)[:, 1]
        attack_scores = model.predict_proba(x_char_test_attack)[:, 1]
        threshold = char_best["threshold_choice"].threshold
        rows.append(_metric_row("SALAD_SOC", "official_test", "CharacterNgramLogistic", "clean", int(seed), binary_metrics(y_test, clean_scores, threshold)))
        rows.append(_metric_row("SALAD_SOC", "official_test", "CharacterNgramLogistic", "full_itpi", int(seed), binary_metrics(y_test, attack_scores, threshold)))
        char_models.append(model)
        if index == 0:
            primary_predictions["CharacterNgramLogistic"] = {
                "clean_scores": clean_scores,
                "attack_scores": attack_scores,
                "threshold": threshold,
            }
    del x_char_validation, x_char_test_clean, x_char_test_attack
    gc.collect()

    x_rf_train = _hashed_struct_matrix(train_sample)
    x_rf_validation = _hashed_struct_matrix(validation)
    rf_candidates = []
    rf_rows = min(int(config["random_forest_train_rows"]), len(train_sample))
    rng = np.random.default_rng(int(config["seeds"][0]))
    rf_indices = rng.choice(len(train_sample), size=rf_rows, replace=False)
    for depth in config["random_forest_depth_grid"]:
        model = RandomForestClassifier(
            n_estimators=int(config["random_forest_trees"]),
            max_depth=int(depth),
            class_weight="balanced_subsample",
            random_state=int(config["seeds"][0]),
            n_jobs=-1,
        )
        model.fit(x_rf_train[rf_indices], y_train[rf_indices])
        scores = model.predict_proba(x_rf_validation)[:, 1]
        choice = choose_threshold(y_validation, scores, max_fpr=max_fpr, objective=objective)
        rf_candidates.append({"depth": int(depth), "threshold_choice": choice})
    rf_best = _select_best_candidate(rf_candidates)
    tuning["StructuralRandomForest"] = {
        "selected_max_depth": rf_best["depth"],
        "threshold": rf_best["threshold_choice"].threshold,
        "validation_metrics": rf_best["threshold_choice"].metrics,
    }
    x_rf_test = _hashed_struct_matrix(test)
    for index, seed in enumerate(config["seeds"]):
        model = RandomForestClassifier(
            n_estimators=int(config["random_forest_trees"]),
            max_depth=rf_best["depth"],
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
        model.fit(x_rf_train[rf_indices], y_train[rf_indices])
        scores = model.predict_proba(x_rf_test)[:, 1]
        threshold = rf_best["threshold_choice"].threshold
        metrics = binary_metrics(y_test, scores, threshold)
        rows.append(_metric_row("SALAD_SOC", "official_test", "StructuralRandomForest", "clean", int(seed), metrics))
        rows.append(_metric_row("SALAD_SOC", "official_test", "StructuralRandomForest", "full_itpi", int(seed), metrics))
        if index == 0:
            primary_predictions["StructuralRandomForest"] = {
                "clean_scores": scores,
                "attack_scores": scores,
                "threshold": threshold,
            }
    del x_rf_train, x_rf_validation, x_rf_test
    gc.collect()

    hostile_test = test[test["label"] == 1].reset_index(drop=True)
    for model_name in ["PITSA_FullTrain", "PITSA_BudgetMatched", "StructuralLogistic", "StructuralRandomForest"]:
        threshold = float(primary_predictions[model_name]["threshold"])
        if model_name.startswith("PITSA"):
            table = pit_full["table"] if model_name == "PITSA_FullTrain" else pit_budget["table"]
            scores = score_llr(hostile_test, table, STRUCT_COLUMNS)
        elif model_name == "StructuralLogistic":
            model = SGDClassifier(
                loss="log_loss",
                alpha=structural_best["alpha"],
                class_weight="balanced",
                max_iter=60,
                tol=1e-4,
                random_state=int(config["seeds"][0]),
            )
            model.fit(x_struct_train, y_train)
            scores = model.predict_proba(structural_vectorizer.transform(hostile_test["struct_text"]))[:, 1]
        else:
            model = RandomForestClassifier(
                n_estimators=int(config["random_forest_trees"]),
                max_depth=rf_best["depth"],
                class_weight="balanced_subsample",
                random_state=int(config["seeds"][0]),
                n_jobs=-1,
            )
            model.fit(_hashed_struct_matrix(train_sample)[rf_indices], y_train[rf_indices])
            scores = model.predict_proba(_hashed_struct_matrix(hostile_test))[:, 1]
        recall = float(np.mean(scores >= threshold))
        graded_curves[model_name] = [
            {"clause_count": int(count), "hostile_recall": recall}
            for count in config["attack_clause_counts"]
        ]

    word_primary = word_models[0]
    char_primary = char_models[0]
    for model_name in ["JointWordLogistic", "JointNaiveBayes", "CharacterNgramLogistic"]:
        threshold = float(primary_predictions[model_name]["threshold"])
        curve: list[dict[str, Any]] = []
        for count in config["attack_clause_counts"]:
            text = attacked_joint_text(hostile_test, int(count))
            if model_name == "JointWordLogistic":
                scores = word_primary.predict_proba(word_vectorizer.transform(text))[:, 1]
            elif model_name == "JointNaiveBayes":
                scores = naive_model.predict_proba(word_vectorizer.transform(text))[:, 1]
            else:
                scores = char_primary.predict_proba(char_vectorizer.transform(text))[:, 1]
            curve.append(
                {
                    "clause_count": int(count),
                    "hostile_recall": float(np.mean(scores >= threshold)),
                    "mean_hostile_score": float(np.mean(scores)),
                }
            )
        graded_curves[model_name] = curve

    source_holdouts: list[dict[str, Any]] = []
    for train_source, test_source in [("CICIDS2017", "UNSW_NB15"), ("UNSW_NB15", "CICIDS2017")]:
        train_source_frame = train_sample[train_sample["source_dataset"] == train_source]
        validation_source_frame = validation[validation["source_dataset"] == train_source]
        test_source_frame = test[test["source_dataset"] == test_source]
        if train_source_frame["label"].nunique() < 2 or validation_source_frame["label"].nunique() < 2 or test_source_frame["label"].nunique() < 2:
            continue
        table = fit_llr_table(train_source_frame, STRUCT_COLUMNS, "label", float(pit_budget["alpha"]))
        validation_scores = score_llr(validation_source_frame, table, STRUCT_COLUMNS)
        choice = choose_threshold(validation_source_frame["label"].to_numpy(dtype=np.int8), validation_scores, max_fpr=max_fpr, objective=objective)
        test_scores = score_llr(test_source_frame, table, STRUCT_COLUMNS)
        source_holdouts.append(
            {
                "train_source": train_source,
                "test_source": test_source,
                "model": "PITSA_BudgetMatched",
                "metrics": binary_metrics(test_source_frame["label"].to_numpy(dtype=np.int8), test_scores, choice.threshold),
                "validation_threshold": choice.threshold,
            }
        )

    bootstrap = cluster_bootstrap_ci(
        y_test,
        pit_full_scores,
        pit_full["threshold_choice"].threshold,
        test["cluster_id"].astype(str).tolist(),
        int(config["bootstrap_resamples"]),
        int(config["seeds"][0]),
    )

    comparison_tests: list[dict[str, Any]] = []
    p_values: list[float] = []
    reference_prediction = (
        primary_predictions["PITSA_FullTrain"]["attack_scores"]
        >= float(primary_predictions["PITSA_FullTrain"]["threshold"])
    ).astype(np.int8)
    for model_name, prediction_data in primary_predictions.items():
        if model_name == "PITSA_FullTrain":
            continue
        comparison_prediction = (
            np.asarray(prediction_data["attack_scores"])
            >= float(prediction_data["threshold"])
        ).astype(np.int8)
        test_result = exact_mcnemar(y_test, reference_prediction, comparison_prediction)
        p_values.append(float(test_result["exact_two_sided_p"]))
        comparison_tests.append({"reference": "PITSA_FullTrain", "comparator": model_name, **test_result})
    adjusted = holm_adjust(p_values)
    for item, value in zip(comparison_tests, adjusted):
        item["holm_adjusted_p"] = float(value)

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(result_dir / "salad_metrics_long.csv", index=False)
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
    aggregate.to_csv(result_dir / "salad_metrics_aggregate.csv", index=False)

    np.savez_compressed(
        output_dir / "primary_predictions.npz",
        record_id=test["record_id"].astype(str).to_numpy(),
        label=y_test,
        cluster_id=test["cluster_id"].astype(str).to_numpy(),
        **{
            f"{model}_{name}": np.asarray(value)
            for model, payload in primary_predictions.items()
            for name, value in payload.items()
            if name in {"clean_scores", "attack_scores"}
        },
    )
    write_json(output_dir / "tuning_report.json", tuning)
    write_json(
        result_dir / "salad_results.json",
        {
            "protocol": {
                "split": "official train, validation, and test",
                "train_rows": int(1944887),
                "validation_rows": int(len(validation)),
                "test_rows": int(len(test)),
                "learned_model_train_sample_rows": int(len(train_sample)),
                "validation_max_fpr": max_fpr,
                "threshold_objective": objective,
                "attack_clause_count": full_attack_count,
                "package_versions": package_versions(),
            },
            "split_manifests": {key: str(value) for key, value in split_manifests.items()},
            "tuning": tuning,
            "metrics_long_file": str(result_dir / "salad_metrics_long.csv"),
            "metrics_aggregate_file": str(result_dir / "salad_metrics_aggregate.csv"),
            "graded_attack_curves": graded_curves,
            "source_holdouts": source_holdouts,
            "pitsa_full_cluster_bootstrap": bootstrap,
            "paired_exact_mcnemar_full_attack": comparison_tests,
            "elapsed_seconds": float(time.time() - started),
        },
    )
    return {
        "results_file": str(result_dir / "salad_results.json"),
        "metrics_file": str(result_dir / "salad_metrics_long.csv"),
        "aggregate_file": str(result_dir / "salad_metrics_aggregate.csv"),
        "elapsed_seconds": float(time.time() - started),
    }
