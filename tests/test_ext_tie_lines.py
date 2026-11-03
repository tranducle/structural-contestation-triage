"""Layout check for the tie cells of the extension loss table, added after the extension lock.

A tie line may hold two policies of at most four name characters, or three one-letter policies
other than M; the widths behind that rule were measured in the manuscript font against the 54pt
"Lowest" columns of tab:ext_loss. Kept apart from the locked test_ext_tables.py so that file stays
byte-identical to its recorded sha256.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import make_extension_tables as xt  # noqa: E402


def test_tie_line_takes_three_one_letter_policies_but_never_m() -> None:
    policies = ("U", "D", "S", "P", "M", "DS", "G", "PITSA", "ALL", "NONE")
    losses = {p: 2.0 for p in policies} | {p: 1.0 for p in ("S", "P", "G", "PITSA")}
    record = {"loss": losses, "best_policy": "S"}
    assert xt._lowest_cell(record, policies) == r"{\begin{tabular}[t]{@{}c@{}}PITSA\\= S = P = G\end{tabular}}"
    assert xt._tie_line_fits(["U", "D", "G"]) and xt._tie_line_fits(["ALL", "G"])
    assert not xt._tie_line_fits(["S", "M", "G"])
    assert not xt._tie_line_fits(["S", "P", "DS"])
    assert not xt._tie_line_fits(["PITSA", "NONE"])
