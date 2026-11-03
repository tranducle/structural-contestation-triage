"""Lowest-loss policy map of the v3 manuscript, drawn from the locked analysis.

llm_loss.py stores, for every model and world, the index of the policy with the lowest expected
loss at each point of a prevalence by cost-ratio grid (np.argmin, so a tie goes to the policy
listed first). Ties are common here: a model that dismisses every alert the gate does not flag
and escalates every alert it does has exactly the recall and false-escalation rate of the gate.
Drawn as stored, such a cell is credited to the model policy listed first, so a figure would show
a model policy winning where the gate alone costs exactly as much.

This script therefore recomputes the loss of every policy from the point rates the analysis
recorded, with the locked llm_stats.expected_loss on the locked grid, and stops unless its argmin
reproduces the stored index in every cell. It then draws each set of tied policies as one region,
credited to a policy that uses no model when one is in the tie, and names the tied policies in the
label. It shows the clean and attacked worlds side by side for every model, and recounts the cells
each stored index wins before drawing either world.

Added after the analysis lock of 2026-10-22; it reads the locked outputs and changes none of them.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

from llm_stats import expected_loss  # noqa: E402  locked; the loss the analysis itself used

logger = logging.getLogger("make_v3_figures")

PAPER_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = PAPER_ROOT / "results"
DEFAULT_ANALYSIS = PAPER_ROOT / "results/triage/triage_study_analysis.json"
DEFAULT_OUTPUT = MANUSCRIPT_DIR / "v3/figures/fig_loss_regions.pdf"
OFFICIAL_LABEL = "OFFICIAL"
WORLDS = ("clean", "attacked")
# A label sits inside its region when the region reaches this many cells from its edge in every
# direction; a thinner region of at least MIN_COMPONENT_CELLS cells gets a label on a leader line.
INSIDE_DEPTH = 2.0
MIN_COMPONENT_CELLS = 10
LEADER_RADIUS = 0.35  # decades, both axes are log10
EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=int)  # a diagonal staircase is one region, not many

SURFACE, INK, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
# Fixed family order. The hues are the first three slots of the dataviz reference palette and pass
# validate_palette.js --pairs all; the hatching keeps the families apart in grayscale print.
FAMILIES = (
    ("Model without structural evidence", ("U", "D", "P"), "#2a78d6", ""),
    ("Model with structural evidence", ("S", "DS", "G"), "#eb6834", "///"),
    ("No model", ("PITSA", "ALL", "NONE"), "#1baf7a", "\\\\\\"),
)
TINT = 0.30
NO_MODEL = FAMILIES[2][1]
# Two losses tie when they differ by less than this share of the lowest loss in the cell. Tied
# policies here have identical recall and false-escalation rates, so their losses differ by
# rounding alone, many orders of magnitude below this.
TIE_TOLERANCE = 1e-12
# The references llm_loss._with_references appends to the measured policies.
REFERENCE_RATES = {"ALL": (1.0, 1.0), "NONE": (0.0, 0.0)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tint(hex_color: str) -> tuple[float, float, float]:
    """Mix a hue into the chart surface, so a large region carries a wash, not a saturated block."""
    hue, surface = np.array(to_rgb(hex_color)), np.array(to_rgb(SURFACE))
    return tuple(float(c) for c in TINT * hue + (1.0 - TINT) * surface)


def _family_of(policy: str) -> tuple[str, str, str]:
    for name, members, color, hatch in FAMILIES:
        if policy in members:
            return name, color, hatch
    raise ValueError(f"policy {policy!r} belongs to no family; extend FAMILIES")


def tie_regions(analysis: Mapping[str, Any], model: str, world: str, policies: list[str],
                best: np.ndarray) -> tuple[np.ndarray, list[tuple[str, tuple[str, ...]]]]:
    """Per grid cell, the region of the set of policies tied for the lowest loss.

    Returns a field of region numbers and, per region, the policy it is credited to and the
    policies tied with it. A region is credited to a policy that uses no model whenever one is
    in the tie, because that cost is then reachable without a model.
    """
    loss = analysis["loss"]
    rates = analysis["salad"][model]["rates"][world]
    pairs = {p: REFERENCE_RATES.get(p) or (float(rates[p]["recall"]), float(rates[p]["fpr"]))
             for p in policies}
    prevalence, cost_ratio = np.meshgrid(np.asarray(loss["grid_prevalence"]),
                                         np.asarray(loss["grid_cost_ratio"]), indexing="ij")
    stack = np.stack([expected_loss(*pairs[p], prevalence, cost_ratio) for p in policies])
    if not np.array_equal(np.argmin(stack, axis=0), best):
        raise ValueError(f"{model}: losses recomputed from the recorded rates do not reproduce "
                         "the stored best_index; refusing to draw ties")
    lowest = stack.min(axis=0)
    tied = stack - lowest <= TIE_TOLERANCE * np.maximum(lowest, 1.0)

    field = np.empty(best.shape, dtype=int)
    regions: list[tuple[str, tuple[str, ...]]] = []
    for cell in np.ndindex(best.shape):
        members = tuple(p for i, p in enumerate(policies) if tied[(i, *cell)])
        credited = next((p for p in NO_MODEL if p in members), members[0])
        region = (credited, tuple(p for p in members if p != credited))
        if region not in regions:
            regions.append(region)
        field[cell] = regions.index(region)
    return field, regions


def region_label(credited: str, others: tuple[str, ...], separator: str = "\n") -> str:
    return credited if not others else f"{credited}{separator}= {' = '.join(others)}"


def _components(mask: np.ndarray) -> list[tuple[int, int, int, float]]:
    """(row, column, cells, depth) of the innermost cell of each connected region of one policy."""
    labelled, count = ndimage.label(mask, structure=EIGHT_NEIGHBOURS)
    found = []
    for index in range(1, count + 1):
        component = labelled == index
        # Pad so that the plot edge counts as a boundary and labels stay off the frame.
        depth = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
        row, column = np.unravel_index(np.argmax(depth), depth.shape)
        deepest = float(depth[row, column])
        if deepest < INSIDE_DEPTH:
            # Every cell of a thin band is equally shallow, so argmax would return the band's
            # first cell, at its end; anchor a leader at the cell nearest the band's centroid.
            cells = np.argwhere(component)
            row, column = cells[np.argmin(np.sum((cells - cells.mean(axis=0)) ** 2, axis=1))]
        found.append((int(row), int(column), int(component.sum()), deepest))
    return found


def _leader_target(anchor: tuple[float, float], taken: list[tuple[float, float]],
                   bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    """The point at LEADER_RADIUS from the anchor farthest from labels already placed."""
    x0, x1, y0, y1 = bounds
    margin = 0.2
    best_point, best_gap = anchor, -1.0
    for angle in np.deg2rad(np.arange(0, 360, 45) + 45):
        point = (anchor[0] + LEADER_RADIUS * math.cos(angle),
                 anchor[1] + LEADER_RADIUS * math.sin(angle))
        if not (x0 + margin <= point[0] <= x1 - margin and y0 + margin <= point[1] <= y1 - margin):
            continue
        gap = min((math.dist(point, other) for other in taken), default=math.inf)
        if gap > best_gap:
            best_point, best_gap = point, gap
    return best_point


def draw_panel(ax: plt.Axes, field: np.ndarray, regions: list[tuple[str, tuple[str, ...]]],
               log_k: np.ndarray, log_pi: np.ndarray,
               salad_prevalence: float) -> list[dict[str, Any]]:
    """Fill each tie region, separate regions by a surface line, label each region."""
    x, y = np.meshgrid(log_k, log_pi)  # grid rows are prevalence, columns cost ratio
    inside, thin, unlabeled = [], [], []
    for number, (credited, others) in enumerate(regions):
        mask = field == number
        _, color, hatch = _family_of(credited)
        surface = mask.astype(float)
        with plt.rc_context({"hatch.color": color}):
            ax.contourf(x, y, surface, levels=[0.5, 1.5], colors=[_tint(color)],
                        hatches=[hatch or None])
        ax.contour(x, y, surface, levels=[0.5], colors=[SURFACE], linewidths=1.0)
        label = region_label(credited, others)
        for row, column, cells, depth in _components(mask):
            point = (float(log_k[column]), float(log_pi[row]))
            if depth >= INSIDE_DEPTH:
                inside.append((label, point))
            elif cells >= MIN_COMPONENT_CELLS:
                thin.append((label, point))
            else:
                unlabeled.append({"region": region_label(credited, others, " "), "cells": cells})

    box = {"boxstyle": "round,pad=0.18", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}
    for label, point in inside:
        ax.text(*point, label, ha="center", va="center", fontsize=7.5, color=INK, bbox=box,
                linespacing=1.1)
    taken = [point for _, point in inside]
    bounds = (float(log_k[0]), float(log_k[-1]), float(log_pi[0]), float(log_pi[-1]))
    for label, point in thin:
        target = _leader_target(point, taken, bounds)
        ax.annotate(label, xy=point, xytext=target, ha="center", va="center", fontsize=7.5,
                    color=INK, bbox=box,
                    arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.6,
                                "shrinkA": 0, "shrinkB": 0})
        taken.append(target)

    level = math.log10(salad_prevalence)
    ax.axhline(level, color=MUTED, linewidth=0.6)
    ax.text(log_k[-1] - 0.05, level + 0.04, "SALAD test prevalence", ha="right", va="bottom",
            fontsize=6.5, color=MUTED, bbox=box)
    return unlabeled


def draw(analysis: Mapping[str, Any], models: list[str], output: Path) -> dict[str, Any]:
    """Write clean and attacked panels per model and return what each panel shows."""
    loss = analysis["loss"]
    policies = list(loss["policies"])
    for policy in policies:
        _family_of(policy)
    log_k = np.log10(np.asarray(loss["grid_cost_ratio"]))
    log_pi = np.log10(np.asarray(loss["grid_prevalence"]))
    salad_prevalence = float(loss["prevalence_anchors"][-1])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
        "hatch.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    })
    columns = len(WORLDS)
    rows = len(models)
    figure, axes = plt.subplots(rows, columns, figsize=(6.6, 1.65 * rows + 1.25),
                                sharex=True, sharey=True, squeeze=False, layout="constrained")
    record: dict[str, Any] = {}
    shown_families: set[str] = set()
    for row, model in enumerate(models):
        record[model] = {}
        for column, world in enumerate(WORLDS):
            ax = axes[row, column]
            cell = loss["models"][model]["grid"][world]
            best = np.asarray(cell["best_index"])
            if best.shape != (log_pi.size, log_k.size):
                raise ValueError(f"{model}/{world}: grid shape {best.shape} does not match the axes")
            recount = {policies[i]: int(np.sum(best == i)) for i in np.unique(best)}
            if recount != cell["cells_won"]:
                raise ValueError(
                    f"{model}/{world}: cells won {recount} differ from {cell['cells_won']}")
            field, regions = tie_regions(analysis, model, world, policies, best)
            unlabeled = draw_panel(ax, field, regions, log_k, log_pi, salad_prevalence)
            shown_families.update(_family_of(credited)[0] for credited, _ in regions)
            record[model][world] = {
                "cells_won": recount,
                "tie_regions": {region_label(c, o, " "): int(np.sum(field == n))
                                for n, (c, o) in enumerate(regions)},
                "cells_won_by_a_model_policy_alone": int(sum(
                    np.sum(field == n) for n, (c, _) in enumerate(regions) if c not in NO_MODEL)),
                "unlabeled_regions": unlabeled,
            }
            for item in unlabeled:
                logger.warning("%s/%s: a %s region of %d cells is too small to label",
                               model, world, item["region"], item["cells"])
        axes[row, 0].annotate(model, xy=(-0.28, 0.5), xycoords="axes fraction",
                              ha="right", va="center", fontsize=8.5, color=INK,
                              annotation_clip=False)

    for column, world in enumerate(WORLDS):
        axes[0, column].set_title(world.capitalize(), fontsize=8.5, color=INK)

    first = axes.flat[0]
    first.set_xticks([0, 1, 2, 3], ["1", "10", "100", "1000"])
    first.set_yticks([-3, -2, -1, math.log10(0.5)],
                     ["1 in 1000", "1 in 100", "1 in 10", "1 in 2"])
    first.set_xlim(log_k[0], log_k[-1])
    first.set_ylim(log_pi[0], log_pi[-1])
    figure.supxlabel("Cost of a missed hostile alert, k, in analyst reviews",
                     fontsize=8, color=INK)
    figure.supylabel("Hostile prevalence", fontsize=8, color=INK)
    # Only the families some region is credited to; a legend entry with no region is a question
    # the reader cannot answer from the figure.
    handles = [Patch(facecolor=_tint(color), edgecolor=color, hatch=hatch, linewidth=0.5,
                     label=name) for name, _, color, hatch in FAMILIES if name in shown_families]
    figure.legend(handles=handles, loc="outside upper center",
                  ncol=len(handles), frameon=False,
                  fontsize=7.5, handlelength=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata={"CreationDate": None} if output.suffix == ".pdf" else None)
    plt.close(figure)
    return record


def _inside(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)  # fontTools logs every subset step at INFO
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="figure file; the suffix picks the format (pdf for the manuscript)")
    parser.add_argument("--models", nargs="+", help="subset of models, in the order given")
    args = parser.parse_args()

    try:
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.error("cannot read the analysis %s: %s", args.analysis, error)
        return 1
    label = str(analysis.get("label", ""))
    if label != OFFICIAL_LABEL and _inside(args.output, MANUSCRIPT_DIR):
        logger.error("analysis label is %r; only an %s analysis may write into %s",
                     label, OFFICIAL_LABEL, MANUSCRIPT_DIR)
        return 1
    models = args.models or list(analysis["models"])
    unknown = [m for m in models if m not in analysis["loss"]["models"]]
    if unknown:
        logger.error("models not in the loss analysis: %s", unknown)
        return 1

    record = draw(analysis, models, args.output)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label, "analysis": str(args.analysis.resolve().relative_to(PAPER_ROOT))
        if _inside(args.analysis, PAPER_ROOT) else args.analysis.name,
        "analysis_sha256": _sha256(args.analysis), "script_sha256": _sha256(Path(__file__)),
        "worlds": list(WORLDS), "figure": args.output.name,
        "figure_sha256": _sha256(args.output),
        "tie_rule": ("best_index keeps np.argmin (lowest index in loss.policies); the figure "
                     "redraws exact ties (relative tolerance %g) as one region, credited to a "
                     "no-model policy when one is tied" % TIE_TOLERANCE),
        "models": record,
    }
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s and %s for %d models (%s)", args.output, manifest_path.name,
                len(models), label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
