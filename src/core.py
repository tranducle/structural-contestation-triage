from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    metrics: dict[str, float | int]
    objective: str
    max_fpr: float


def stable_sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def stable_bucket(value: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(stable_sha256_text(value)[:16], 16) % modulus


def fit_llr_table(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str,
    alpha: float = 1.0,
) -> dict[str, dict[str, Any]]:
    labels = frame[label_column].to_numpy(dtype=np.int8)
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("both labels are required to fit PITSA")

    table: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        positive_counts = Counter(frame.loc[positive, column].astype(str))
        negative_counts = Counter(frame.loc[negative, column].astype(str))
        values = sorted(set(positive_counts) | set(negative_counts))
        support_size = len(values) + 1
        positive_denominator = n_positive + alpha * support_size
        negative_denominator = n_negative + alpha * support_size
        scores: dict[str, float] = {}
        for value in values:
            p_positive = (positive_counts.get(value, 0) + alpha) / positive_denominator
            p_negative = (negative_counts.get(value, 0) + alpha) / negative_denominator
            scores[value] = float(math.log(p_positive / p_negative))
        unknown = float(
            math.log((alpha / positive_denominator) / (alpha / negative_denominator))
        )
        table[column] = {
            "scores": scores,
            "unknown": unknown,
            "support_size": support_size,
            "positive_rows": n_positive,
            "negative_rows": n_negative,
            "alpha": alpha,
        }
    return table


def fit_llr_from_counters(
    counters: Mapping[str, Mapping[int, Counter]],
    label_totals: Mapping[int, int],
    alpha: float = 1.0,
) -> dict[str, dict[str, Any]]:
    n_positive = int(label_totals[1])
    n_negative = int(label_totals[0])
    if n_positive == 0 or n_negative == 0:
        raise ValueError("both labels are required to fit PITSA")
    table: dict[str, dict[str, Any]] = {}
    for column, by_label in counters.items():
        positive_counts = by_label[1]
        negative_counts = by_label[0]
        values = sorted(set(positive_counts) | set(negative_counts))
        support_size = len(values) + 1
        positive_denominator = n_positive + alpha * support_size
        negative_denominator = n_negative + alpha * support_size
        scores = {
            value: float(
                math.log(
                    ((positive_counts.get(value, 0) + alpha) / positive_denominator)
                    / ((negative_counts.get(value, 0) + alpha) / negative_denominator)
                )
            )
            for value in values
        }
        unknown = float(
            math.log((alpha / positive_denominator) / (alpha / negative_denominator))
        )
        table[column] = {
            "scores": scores,
            "unknown": unknown,
            "support_size": support_size,
            "positive_rows": n_positive,
            "negative_rows": n_negative,
            "alpha": alpha,
        }
    return table


def score_llr(
    frame: pd.DataFrame,
    table: Mapping[str, Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> np.ndarray:
    total = np.zeros(len(frame), dtype=np.float64)
    for column in feature_columns:
        item = table[column]
        mapping = item["scores"]
        unknown = float(item["unknown"])
        total += (
            frame[column]
            .astype(str)
            .map(mapping)
            .fillna(unknown)
            .to_numpy(dtype=np.float64)
        )
    return total


def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    prediction = (s >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, fp + tn))
    tnr = float(tn / max(1, fp + tn))
    try:
        auroc = float(roc_auc_score(y, s))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, s))
    except ValueError:
        auprc = float("nan")
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "fpr": fpr,
        "specificity": tnr,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auroc": auroc,
        "auprc": auprc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
    }


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.10,
    objective: str = "f1",
    candidate_count: int = 301,
) -> ThresholdChoice:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        raise ValueError("threshold selection requires finite scores")
    quantiles = np.linspace(0.0, 1.0, candidate_count)
    candidates = np.unique(np.quantile(finite, quantiles))
    candidates = np.unique(np.concatenate(([finite.min() - 1e-12], candidates, [finite.max() + 1e-12])))
    feasible: list[tuple[float, dict[str, float | int]]] = []
    all_items: list[tuple[float, dict[str, float | int]]] = []
    for threshold in candidates:
        metrics = binary_metrics(y, s, float(threshold))
        all_items.append((float(threshold), metrics))
        if float(metrics["fpr"]) <= max_fpr:
            feasible.append((float(threshold), metrics))
    pool = feasible if feasible else all_items
    if objective == "balanced_accuracy":
        key = lambda item: (
            float(item[1]["balanced_accuracy"]),
            float(item[1]["recall"]),
            float(item[1]["f1"]),
            -float(item[1]["fpr"]),
        )
    else:
        key = lambda item: (
            float(item[1]["f1"]),
            float(item[1]["recall"]),
            float(item[1]["balanced_accuracy"]),
            -float(item[1]["fpr"]),
        )
    threshold, metrics = max(pool, key=key)
    return ThresholdChoice(
        threshold=threshold,
        metrics=metrics,
        objective=objective,
        max_fpr=max_fpr,
    )


def exact_mcnemar(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    a = np.asarray(prediction_a, dtype=np.int8)
    b = np.asarray(prediction_b, dtype=np.int8)
    correct_a = a == y
    correct_b = b == y
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(a_only, b_only), discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
        "p_value": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _weighted_auc_and_ap_from_sorted(
    y_sorted: np.ndarray,
    score_sorted: np.ndarray,
    weight_sorted: np.ndarray,
) -> tuple[float, float]:
    positive_total = float(weight_sorted[y_sorted == 1].sum())
    negative_total = float(weight_sorted[y_sorted == 0].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        return float("nan"), float("nan")

    group_start = np.r_[0, np.flatnonzero(np.diff(score_sorted) != 0.0) + 1]
    group_positive = np.add.reduceat(weight_sorted * (y_sorted == 1), group_start)
    group_negative = np.add.reduceat(weight_sorted * (y_sorted == 0), group_start)
    cumulative_negative_before = np.r_[0.0, np.cumsum(group_negative)[:-1]]
    auc_numerator = float(
        np.sum(group_positive * (cumulative_negative_before + 0.5 * group_negative))
    )
    auc = auc_numerator / (positive_total * negative_total)

    positive_descending = group_positive[::-1]
    negative_descending = group_negative[::-1]
    cumulative_positive = np.cumsum(positive_descending)
    cumulative_total = np.cumsum(positive_descending + negative_descending)
    precision_values = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=np.float64),
        where=cumulative_total > 0.0,
    )
    recall_increment = positive_descending / positive_total
    average_precision = float(np.sum(precision_values * recall_increment))
    return float(auc), average_precision


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    clusters: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, list[float] | int]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    cluster_series = pd.Series(clusters, dtype="string")
    unique_clusters, inverse = np.unique(cluster_series.astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {
        "recall": [],
        "precision": [],
        "f1": [],
        "fpr": [],
        "balanced_accuracy": [],
        "auroc": [],
        "auprc": [],
    }
    prediction = (s >= threshold).astype(np.int8)
    sort_order = np.argsort(s, kind="mergesort")
    y_sorted = y[sort_order]
    score_sorted = s[sort_order]
    inverse_sorted = inverse[sort_order]

    for _ in range(resamples):
        sampled = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        multiplicity = np.bincount(sampled, minlength=len(unique_clusters)).astype(np.float64)
        weights = multiplicity[inverse]
        positive_weight = float(weights[y == 1].sum())
        negative_weight = float(weights[y == 0].sum())
        if positive_weight == 0.0 or negative_weight == 0.0:
            continue
        tp = float(weights[(y == 1) & (prediction == 1)].sum())
        fn = float(weights[(y == 1) & (prediction == 0)].sum())
        fp = float(weights[(y == 0) & (prediction == 1)].sum())
        tn = float(weights[(y == 0) & (prediction == 0)].sum())
        precision_value = tp / max(1e-12, tp + fp)
        recall_value = tp / max(1e-12, tp + fn)
        f1_value = 2.0 * precision_value * recall_value / max(
            1e-12, precision_value + recall_value
        )
        fpr_value = fp / max(1e-12, fp + tn)
        balanced_value = 0.5 * (
            recall_value + tn / max(1e-12, tn + fp)
        )
        estimates["precision"].append(precision_value)
        estimates["recall"].append(recall_value)
        estimates["f1"].append(f1_value)
        estimates["fpr"].append(fpr_value)
        estimates["balanced_accuracy"].append(balanced_value)
        sorted_weights = multiplicity[inverse_sorted]
        auc_value, ap_value = _weighted_auc_and_ap_from_sorted(
            y_sorted, score_sorted, sorted_weights
        )
        if np.isfinite(auc_value):
            estimates["auroc"].append(auc_value)
        if np.isfinite(ap_value):
            estimates["auprc"].append(ap_value)

    output: dict[str, list[float] | int] = {
        "requested_resamples": int(resamples),
        "completed_resamples": int(len(estimates["recall"])),
        "cluster_count": int(len(unique_clusters)),
    }
    for name, values in estimates.items():
        if values:
            output[f"{name}_95ci"] = [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def append_attack_text(original: str, clause: str, repetitions: int = 1) -> str:
    if repetitions <= 0:
        return str(original)
    suffix = " ".join(str(clause) for _ in range(int(repetitions)))
    return str(original) + " untrusted_note " + suffix


def fit_categorical_llr(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str,
    alpha: float = 1.0,
) -> dict[str, Any]:
    columns = [str(column) for column in feature_columns]
    return {
        "feature_columns": columns,
        "table": fit_llr_table(frame, columns, label_column, alpha),
    }


def score_categorical_llr(frame: pd.DataFrame, model: Mapping[str, Any]) -> np.ndarray:
    return score_llr(frame, model["table"], model["feature_columns"])


def select_threshold_by_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    miss_cost: float,
    false_alarm_cost: float,
    review_cost: float = 0.0,
    candidate_count: int = 401,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        raise ValueError("threshold selection requires finite scores")
    candidates = np.unique(
        np.concatenate(
            [
                [finite.min() - 1e-12],
                np.quantile(finite, np.linspace(0.0, 1.0, candidate_count)),
                [finite.max() + 1e-12],
            ]
        )
    )
    best: dict[str, float | int] | None = None
    row_count = max(1, len(y))
    for threshold in candidates:
        prediction = (s >= float(threshold)).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        review_rows = int(prediction.sum())
        expected_cost = (
            float(miss_cost) * float(fn)
            + float(false_alarm_cost) * float(fp)
            + float(review_cost) * float(review_rows)
        ) / float(row_count)
        item: dict[str, float | int] = {
            "threshold": float(threshold),
            "expected_cost": float(expected_cost),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "review_rows": review_rows,
        }
        if best is None or (
            float(item["expected_cost"]),
            int(item["false_negatives"]),
            int(item["false_positives"]),
            -float(item["threshold"]),
        ) < (
            float(best["expected_cost"]),
            int(best["false_negatives"]),
            int(best["false_positives"]),
            -float(best["threshold"]),
        ):
            best = item
    if best is None:
        raise RuntimeError("no threshold candidate was evaluated")
    return best


def cluster_bootstrap_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    clusters: Sequence[str],
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    point = binary_metrics(y_true, scores, threshold)
    intervals = cluster_bootstrap_ci(
        y_true,
        scores,
        threshold,
        clusters,
        int(n_boot),
        int(seed),
    )
    return {
        "precision": float(point["precision"]),
        "recall": float(point["recall"]),
        "f1": float(point["f1"]),
        "fpr": float(point["fpr"]),
        "auroc": float(point["auroc"]),
        "auprc": float(point["auprc"]),
        "precision_ci": intervals.get("precision_95ci"),
        "recall_ci": intervals.get("recall_95ci"),
        "f1_ci": intervals.get("f1_95ci"),
        "fpr_ci": intervals.get("fpr_95ci"),
        "auroc_ci": intervals.get("auroc_95ci"),
        "auprc_ci": intervals.get("auprc_95ci"),
        "n_clusters": int(intervals["cluster_count"]),
        "n_boot": int(intervals["requested_resamples"]),
        "completed_boot": int(intervals["completed_resamples"]),
    }
