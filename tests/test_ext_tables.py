"""Extension gate E6: the table generator refuses what it must refuse.

Each test builds a minimal value mapping by hand, so the failure paths are exercised without an
official answer: a zero count must carry its exact bracket, a weighted rate printed 0.0 must have
no escalation behind it, and payload families 1 and 2 must agree on the clean opening turn they
share once the analysis is OFFICIAL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import make_extension_tables as xt  # noqa: E402

EXT_RES = REPO_ROOT / "results/triage_extension"
MODEL = "m1"


def _rate_keys(base: str, metric: str, printed: str, strata: tuple[str, ...], counts: tuple[int, ...],
               trials: tuple[int, ...], highs: tuple[str, ...]) -> dict[str, str]:
    keys = {f"{base}/{metric}": printed}
    for stratum, k, n, hi in zip(strata, counts, trials, highs):
        prefix = f"{base}/per_stratum/{stratum}"
        keys.update({f"{prefix}/successes": str(k), f"{prefix}/trials": str(n),
                     f"{prefix}/exact95/lo": "0.0", f"{prefix}/exact95/hi": hi})
    return keys


def test_zero_count_carries_exact_bracket_and_nonzero_does_not() -> None:
    cells = xt.ExtCells({"a/successes": "0", "a/trials": "40", "a/exact95/lo": "0.0", "a/exact95/hi": "8.8",
                         "b/successes": "3", "b/trials": "40", "b/exact95/lo": "1.6", "b/exact95/hi": "20.4"})
    assert cells.count_bounded("a") == r"\val{a/successes}/\val{a/trials} [\val{a/exact95/lo}, \val{a/exact95/hi}]"
    assert cells.count_bounded("b") == r"\val{b/successes}/\val{b/trials}"
    assert cells.count_bounded("c") == xt.NA and "c/successes" in cells.missing


def test_zero_rate_bound_picks_the_widest_stratum() -> None:
    base = f"ext/original/{MODEL}/rates/clean/U"
    defined = _rate_keys(base, "recall", "0.0", xt.HOSTILE, (0, 0), (40, 40), ("8.8", "8.8"))
    fpr_base = f"ext/extended/{MODEL}/rates/clean/U"
    defined.update(_rate_keys(fpr_base, "fpr", "0.0", xt.BENIGN, (0, 0), (120, 40), ("3.0", "8.8")))
    hi, trials = xt.zero_rate_bound(xt.ExtCells(defined), [MODEL])
    assert defined[hi] == "8.8" and defined[trials] == "40"


def test_zero_rate_bound_refuses_a_rounded_zero() -> None:
    base = f"ext/original/{MODEL}/rates/clean/U"
    defined = _rate_keys(base, "recall", "0.0", xt.HOSTILE, (0, 1), (40, 40), ("8.8", "13.2"))
    with pytest.raises(ValueError, match="counts an escalation"):
        xt.zero_rate_bound(xt.ExtCells(defined), [MODEL])


def test_zero_rate_bound_is_none_without_a_zero_rate() -> None:
    base = f"ext/original/{MODEL}/rates/clean/U"
    defined = _rate_keys(base, "recall", "12.5", xt.HOSTILE, (5, 0), (40, 40), ("26.8", "8.8"))
    assert xt.zero_rate_bound(xt.ExtCells(defined), [MODEL]) is None


def _injection_cells(f1_clean: int, f2_clean: int) -> xt.ExtCells:
    defined: dict[str, str] = {}
    for prompt in xt.PROMPT_TEXT:
        for prefix, clean in ((f"ext/exact/locked/salad/{MODEL}/injection/{prompt}", f1_clean),
                              (f"ext/original/{MODEL}/f2_injection/{prompt}", f2_clean)):
            for group, metric in xt.INJECTION_METRICS:
                k = clean if metric == "escalated_clean" else 0
                defined.update({f"{prefix}/{group}/{metric}/successes": str(k),
                                f"{prefix}/{group}/{metric}/trials": "80",
                                f"{prefix}/{group}/{metric}/exact95/lo": "0.0",
                                f"{prefix}/{group}/{metric}/exact95/hi": "4.5"})
    defined.update({"ext/design/original_alerts": "160", f"ext/original/{MODEL}/discrimination/hostile_alerts": "80",
                    f"ext/extended/{MODEL}/rates/clean/U/per_stratum/benign_gate/trials": "120"})
    return xt.ExtCells(defined)


def test_injection_families_must_share_the_clean_turn_when_official() -> None:
    official = {"label": "OFFICIAL", "added_model": None, "models": [MODEL]}
    assert "tab:ext_injection" in xt.injection_table(_injection_cells(7, 7), official, [MODEL])
    with pytest.raises(ValueError, match="differs between payload families"):
        xt.injection_table(_injection_cells(7, 6), official, [MODEL])
    smoke = {**official, "label": "SMOKE"}
    assert "tab:ext_injection" in xt.injection_table(_injection_cells(7, 6), smoke, [MODEL])


def test_read_values_accepts_both_value_files(tmp_path: Path) -> None:
    for label_macro in ("llmvaluesource", "llmextvaluesource"):
        path = tmp_path / f"{label_macro}.tex"
        path.write_text(f"% source x sha256 {'0' * 64}\n\\def\\{label_macro}{{SMOKE}}\n"
                        "\\expandafter\\def\\csname llmv@ext/a\\endcsname{1.0}\n", encoding="utf-8")
        defined, label, hashes = xt.read_values(path)
        assert defined == {"ext/a": "1.0"} and label == "SMOKE" and hashes == {"0" * 64}
    (tmp_path / "bad.tex").write_text("% nothing\n", encoding="utf-8")
    with pytest.raises(ValueError):
        xt.read_values(tmp_path / "bad.tex")


def test_official_tables_print_no_bare_measured_number(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import make_triage_macros as mm
    import make_extension_macros as mx
    mm_args = [sys.executable, str(REPO_ROOT / "scripts/make_triage_macros.py"), "--analysis", str(REPO_ROOT / "results/triage/triage_study_analysis.json"), "--output-dir", str(tmp_path)]
    mx_args = [sys.executable, str(REPO_ROOT / "scripts/make_extension_macros.py"), "--analysis", str(EXT_RES / "extension_study_analysis.json"), "--added-analysis", str(REPO_ROOT / "results/triage_added/triage_study_analysis.json"), "--output-dir", str(tmp_path)]
    import subprocess
    assert subprocess.run(mm_args, check=False).returncode == 0
    assert subprocess.run(mx_args, check=False).returncode == 0
    monkeypatch.setattr(sys, "argv", ["make_extension_tables.py", "--analysis", str(EXT_RES / "extension_study_analysis.json"),
                                      "--values", str(tmp_path / "llm_ext_values.tex"), "--locked-values", str(tmp_path / "llm_values.tex"), "--output-dir", str(tmp_path)])
    assert xt.main() == 0
    for path in tmp_path.glob("table_ext_*.tex"):
        body = "\n".join(line for line in path.read_text().splitlines() if not line.startswith("%"))
        for sample in ("160", "120", " 80 "):
            assert sample not in body, f"{path.name} types the sample size {sample.strip()} by hand"


def test_lowest_cell_wraps_a_long_tie_and_keeps_the_locked_check() -> None:
    policies = ("U", "D", "S", "P", "DS", "G", "PITSA", "ALL", "NONE")
    losses = {p: 2.0 for p in policies} | {p: 1.0 for p in ("S", "P", "DS", "G", "PITSA")}
    record = {"loss": losses, "best_policy": "S"}
    assert xt._lowest_cell(record, policies) == r"{\begin{tabular}[t]{@{}c@{}}PITSA\\= S = P\\= DS = G\end{tabular}}"
    single = {"loss": {p: 2.0 for p in policies} | {"U": 1.0}, "best_policy": "U"}
    assert xt._lowest_cell(single, policies) == "U"
    with pytest.raises(ValueError, match="best_policy"):
        xt._lowest_cell({**record, "best_policy": "P"}, policies)
