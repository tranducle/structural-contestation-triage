"""Flat CSV views of llm_analysis_v3.json, one row per reported quantity."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


def _interval(values: list[float] | None) -> tuple[Any, Any]:
    return (values[0], values[1]) if values else ("", "")


def _write(path: Path, header: list[str], rows: Iterable[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _policy_rows(analysis: Mapping[str, Any]) -> Iterable[list[Any]]:
    for model, block in analysis["salad"].items():
        for world, policies in block["rates"].items():
            for policy, rate in policies.items():
                yield [model, world, policy, rate["recall"], *_interval(rate["recall95"]), rate["fpr"],
                       *_interval(rate["fpr95"]), rate["invalid_decisions"], rate["contested_alerts"]]


def _test_rows(analysis: Mapping[str, Any]) -> Iterable[list[Any]]:
    for model, block in analysis["salad"].items():
        for test in block["tests"]:
            yield [model, "salad", test["family"], test["name"], test["first"], test["second"], test["n_pairs"],
                   test["first_only"], test["second_only"], test["exact_p"], test.get("holm_p", "")]
    for model, block in analysis["otrf"].items():
        for test in block["tests"]:
            yield [model, "otrf", test["family"], test["name"], test["first"], test["second"], test["n_pairs"],
                   test["first_only"], test["second_only"], test["exact_p"], ""]


def _stage2_rows(analysis: Mapping[str, Any]) -> Iterable[list[Any]]:
    for model, block in analysis["salad"].items():
        for condition, by_stratum in block["stage2"].items():
            for stratum, rate in by_stratum.items():
                yield [model, condition, stratum, rate["successes"], rate["trials"], rate["rate"],
                       *_interval(rate["wilson95"]), rate["invalid"]]


def _otrf_rows(analysis: Mapping[str, Any]) -> Iterable[list[Any]]:
    for model, block in analysis["otrf"].items():
        for policy, rate in block["escalation"].items():
            full, kept = rate["all"], rate["excluding_suspects"]
            yield [model, policy, full["successes"], full["trials"], full["rate"], *_interval(full["wilson95"]),
                   kept["successes"], kept["trials"], kept["rate"], rate["invalid_decisions"]]


def _loss_rows(analysis: Mapping[str, Any]) -> Iterable[list[Any]]:
    for model, block in analysis["loss"]["models"].items():
        for anchor in block["anchors"]:
            for policy, loss in anchor["loss"].items():
                yield [model, anchor["world"], anchor["prevalence"], anchor["cost_ratio"], policy, loss,
                       *_interval(anchor["loss95"][policy]), int(policy == anchor["best_policy"])]


def write_tables(analysis: Mapping[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    specs = (
        ("salad_policy_rates", ["model", "world", "policy", "recall", "recall_lo", "recall_hi", "fpr", "fpr_lo",
                                "fpr_hi", "invalid_decisions", "contested_alerts"], _policy_rows),
        ("llm_tests", ["model", "corpus", "family", "name", "first", "second", "n_pairs", "first_only",
                       "second_only", "exact_p", "holm_p"], _test_rows),
        ("salad_stage2", ["model", "condition", "stratum", "escalated", "issued", "rate", "wilson_lo",
                          "wilson_hi", "invalid"], _stage2_rows),
        ("otrf_policy_rates", ["model", "policy", "escalated", "events", "rate", "wilson_lo", "wilson_hi",
                               "escalated_excl_suspects", "events_excl_suspects", "rate_excl_suspects",
                               "invalid_decisions"], _otrf_rows),
        ("loss_anchors", ["model", "world", "prevalence", "cost_ratio", "policy", "loss", "loss_lo", "loss_hi",
                          "is_best"], _loss_rows),
    )
    written = []
    for name, header, rows in specs:
        path = output_dir / f"{prefix}{name}.csv"
        _write(path, header, rows(analysis))
        written.append(path)
    return written
