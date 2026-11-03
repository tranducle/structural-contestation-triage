"""Gate G5: the value file the manuscript reads its numbers from.

Each test fixes one way a wrong number could reach the paper: a misformatted value, a key TeX
cannot carry, a smoke analysis written as the official file, or a manuscript key with no value.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import make_triage_macros as macros  # noqa: E402

SCRIPT = REPO_ROOT / "scripts/make_triage_macros.py"


def test_formats_follow_the_quantity() -> None:
    assert macros.format_value(("salad", "m", "rates", "clean", "S", "recall"), 0.8257) == "82.6"
    assert macros.format_value(("salad", "m", "rates", "clean", "S", "recall95", "lo"), 0.0) == "0.0"
    assert macros.format_value(("salad", "m", "differences", "clean:S-U", "fpr_diff"), -0.0123) == "\\ensuremath{-}1.2"
    assert macros.format_value(("salad", "m", "differences", "clean:S-U", "fpr_diff"), -0.0004) == "0.0"
    assert macros.format_value(("salad", "m", "tests", "H1", "holm_p"), 0.0004) == "\\ensuremath{<}0.001"
    assert macros.format_value(("salad", "m", "tests", "H1", "exact_p"), 0.0312) == "0.031"
    assert macros.format_value(("loss", "models", "m", "anchors", "clean", "pi0.01", "k10", "loss", "S"), 0.12345) \
        == "0.123"
    assert macros.format_value(("loss", "models", "m", "break_even", "clean", "S-U", "pi0.01", "k_star"), 12.83) \
        == "12.8"
    assert macros.format_value(("v2", "salad", "PITSA_FullTrain", "clean", "recall"), 0.8257) == "0.826"
    assert macros.format_value(("v2", "salad", "protocol", "train_rows"), 1944887) == "1{,}944{,}887"
    assert macros.format_value(("salad", "m", "rates", "clean", "S", "recall"), None) is None
    assert macros.format_value(("design", "invalid_answer_rule"), "escalate (fail-safe)") is None
    assert macros.format_value(("loss", "x", "best_policy"), "ALL") == "ALL"


def test_keys_reject_what_a_control_sequence_name_cannot_carry() -> None:
    assert macros._validate_key(("salad", "qwen2.5:3b", "differences", "attacked:S-U")) == \
        "salad/qwen2.5:3b/differences/attacked:S-U"
    for bad in (("a b",), ("a{b}",), ("a--b",), ("a%b",)):
        with pytest.raises(ValueError):
            macros._validate_key(bad)


def test_record_lists_are_keyed_by_content_and_unknown_lists_fail() -> None:
    out: dict = {}
    macros.flatten({"tests": [{"name": "H1", "exact_p": 0.5, "strata": ["a"]}],
                    "break_even": [{"world": "clean", "a": "S", "b": "U", "prevalence": 0.01, "k_star": 3.0}],
                    "recall95": [0.1, 0.2], "grid": [1, 2, 3]}, ("x",), out)
    assert out == {("x", "tests", "H1", "exact_p"): 0.5, ("x", "break_even", "clean", "S-U", "pi0.01", "k_star"): 3.0,
                   ("x", "recall95", "lo"): 0.1, ("x", "recall95", "hi"): 0.2}
    with pytest.raises(ValueError):
        macros.flatten({"mystery": [{"a": 1}]}, ("x",), {})


def _run(tmp_path: Path, analysis: dict) -> subprocess.CompletedProcess:
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(analysis))
    return subprocess.run([sys.executable, str(SCRIPT), "--analysis", str(source), "--output-dir", str(tmp_path)],
                          capture_output=True, text=True)


def test_only_a_passed_official_analysis_writes_the_official_file(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    base = {"salad": {"m": {"rates": {"clean": {"U": {"recall": 0.5}}}}}}
    smoke = _run(tmp_path, {**base, "label": "SMOKE (validates code only; never evidence)", "g4_passed": True})
    assert smoke.returncode == 0
    assert (tmp_path / "SMOKE_llm_values.tex").exists() and not (tmp_path / "llm_values.tex").exists()
    failed = _run(tmp_path, {**base, "label": "OFFICIAL", "g4_passed": False})
    assert failed.returncode == 1 and not (tmp_path / "llm_values.tex").exists()
    passed = _run(tmp_path, {**base, "label": "OFFICIAL", "g4_passed": True})
    assert passed.returncode == 0
    text = (tmp_path / "llm_values.tex").read_text()
    assert "\\def\\llmvaluesource{OFFICIAL}" in text and "llmv@salad/m/rates/clean/U/recall\\endcsname{50.0}" in text


def test_check_reports_missing_keys_and_bare_numbers(tmp_path: Path) -> None:
    manuscript = tmp_path / "results.tex"
    manuscript.write_text("Recall was \\val{a/recall}\\% and 0.83 by hand. % 0.99 in a comment\n"
                          "See \\cref{tab:x}; \\val{a/missing}.\n")
    result = macros.check_manuscript([manuscript], {"a/recall"})
    assert not result["passed"] and result["missing_keys"] == ["a/missing"]
    assert [item["text"] for item in result["bare_numbers"]] == ["0.83"]
