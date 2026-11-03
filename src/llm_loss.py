"""Expected triage loss of each policy, in units of one analyst review.

Recall and false-escalation rate are measured on the SALAD sample and are class-conditional, so
they are carried to other prevalences unchanged. That transfer assumes the hostile and benign
alert populations look the same at the new prevalence; the manuscript states this assumption.
ALL escalates every alert and NONE escalates nothing; they bound what any triage policy can buy.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from llm_policies import SALAD_POLICIES, WORLDS
from llm_stats import expected_loss, percentile_interval

PREVALENCE_ANCHORS = (0.01, 0.05)
COST_RATIO_ANCHORS = (10.0, 100.0)
GRID_POINTS = 49
GRID_PREVALENCE = np.logspace(-3.0, np.log10(0.5), GRID_POINTS)
GRID_COST_RATIO = np.logspace(0.0, 3.0, GRID_POINTS)
LOSS_POLICIES = SALAD_POLICIES + ("ALL", "NONE")
LOSS_DIFFERENCES = (("S", "U"), ("S", "P"), ("S", "PITSA"), ("G", "U"), ("G", "PITSA"), ("DS", "D"))
BREAK_EVEN_PAIRS = (("S", "U"), ("G", "U"), ("S", "PITSA"), ("G", "PITSA"))

# boot[(world, policy)] = (recall, fpr, recall per resample, fpr per resample)
Boot = Mapping[tuple[str, str], tuple[float, float, np.ndarray, np.ndarray]]


def _with_references(boot: Boot, world: str) -> dict[str, tuple[float, float, np.ndarray, np.ndarray]]:
    resamples = next(iter(boot.values()))[2].size
    rows = {policy: boot[(world, policy)] for policy in SALAD_POLICIES}
    rows["ALL"] = (1.0, 1.0, np.ones(resamples), np.ones(resamples))
    rows["NONE"] = (0.0, 0.0, np.zeros(resamples), np.zeros(resamples))
    return rows


def break_even(recall_a: float, fpr_a: float, recall_b: float, fpr_b: float, prevalence: float) -> dict[str, Any]:
    """Cost ratio at which policies A and B have equal loss, and which side of it favours A.

    Solving loss_A = loss_B gives k* = 1 + (1 - pi)(F_A - F_B) / (pi (R_A - R_B)). When A has the
    higher recall it wins for every k above k*; when recall is equal, the lower FPR wins for all k.
    """
    recall_gap, fpr_gap = recall_a - recall_b, fpr_a - fpr_b
    if abs(recall_gap) < 1e-12:
        verdict = "A_always" if fpr_gap < 0 else ("B_always" if fpr_gap > 0 else "tie")
        return {"k_star": None, "a_wins": verdict}
    k_star = 1.0 + (1.0 - prevalence) * fpr_gap / (prevalence * recall_gap)
    return {"k_star": float(k_star), "a_wins": "above_k_star" if recall_gap > 0 else "below_k_star"}


def anchor_table(boot: Boot, prevalences: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for world in WORLDS:
        policies = _with_references(boot, world)
        for prevalence in prevalences:
            for cost_ratio in COST_RATIO_ANCHORS:
                point = {p: float(expected_loss(v[0], v[1], prevalence, cost_ratio)) for p, v in policies.items()}
                samples = {p: expected_loss(v[2], v[3], prevalence, cost_ratio) for p, v in policies.items()}
                rows.append({
                    "world": world, "prevalence": prevalence, "cost_ratio": cost_ratio,
                    "loss": point,
                    "loss95": {p: percentile_interval(s) for p, s in samples.items()},
                    "best_policy": min(point, key=point.get),
                    "differences": {
                        f"{a}-{b}": {"diff": point[a] - point[b], "diff95": percentile_interval(samples[a] - samples[b]),
                                     "share_a_lower": float(np.mean(samples[a] < samples[b]))}
                        for a, b in LOSS_DIFFERENCES
                    },
                })
    return rows


def best_policy_grid(boot: Boot) -> dict[str, Any]:
    """Index into LOSS_POLICIES of the lowest-loss policy over the prevalence x cost-ratio grid."""
    output: dict[str, Any] = {}
    prevalence, cost_ratio = np.meshgrid(GRID_PREVALENCE, GRID_COST_RATIO, indexing="ij")
    for world in WORLDS:
        policies = _with_references(boot, world)
        stack = np.stack([expected_loss(policies[p][0], policies[p][1], prevalence, cost_ratio) for p in LOSS_POLICIES])
        best = np.argmin(stack, axis=0)
        output[world] = {
            "best_index": best.tolist(),
            "cells_won": {LOSS_POLICIES[i]: int(np.sum(best == i)) for i in np.unique(best)},
        }
    return output


def loss_analysis(boot_by_model: Mapping[str, Boot], salad_prevalence: float) -> dict[str, Any]:
    prevalences = PREVALENCE_ANCHORS + (float(salad_prevalence),)
    by_model: dict[str, Any] = {}
    for model, boot in boot_by_model.items():
        breaks = []
        for world in WORLDS:
            for a, b in BREAK_EVEN_PAIRS:
                for prevalence in prevalences:
                    ra, fa = boot[(world, a)][:2]
                    rb, fb = boot[(world, b)][:2]
                    breaks.append({"world": world, "a": a, "b": b, "prevalence": prevalence,
                                   **break_even(ra, fa, rb, fb, prevalence)})
        by_model[model] = {"anchors": anchor_table(boot, prevalences), "grid": best_policy_grid(boot),
                           "break_even": breaks}
    return {
        "units": "analyst reviews per alert; a missed hostile alert costs cost_ratio reviews",
        "formula": "pi*(1-R)*k + pi*R + (1-pi)*F",
        "assumption": "recall and FPR measured on SALAD transfer unchanged to the stated prevalence",
        "policies": list(LOSS_POLICIES),
        "prevalence_anchors": list(prevalences),
        "cost_ratio_anchors": list(COST_RATIO_ANCHORS),
        "grid_prevalence": GRID_PREVALENCE.tolist(),
        "grid_cost_ratio": GRID_COST_RATIO.tolist(),
        "models": by_model,
    }
