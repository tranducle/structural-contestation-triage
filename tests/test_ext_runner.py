"""Extension gate E3: prompt construction, call plans, stimuli and model selection.

The extension reuses the locked prompt builders; these tests pin that reuse, the fail-safe that a
benign alert is never injected, the read-only status of the locked response stores, the disjoint
draw of the extension stimuli and the mechanical screening rule.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT

import extension_triage_runner as er  # noqa: E402
import salad_triage_runner as salad  # noqa: E402
from ext_design import EXTENSION_SEED, SCREENING_RULE, ScreeningRule, payload_f2  # noqa: E402
from llm_common import DecodingConfig, ResponseStore  # noqa: E402
from screen_models import select_model, summarize  # noqa: E402

LOCKED = V3 / "data/stimuli/salad_llm_stimuli.jsonl"
LOCKED_MANIFEST = V3 / "data/stimuli/salad_llm_stimuli_manifest.json"
EXT_DIR = V3 / "data/stimuli"
RAW = '{"disposition": "dismiss", "confidence": 0.6, "evidence": ["protocol"], "rationale": "x"}'


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def locked() -> list[dict]:
    return _load(LOCKED)


@pytest.fixture(scope="module")
def benign_ext() -> list[dict]:
    return _load(EXT_DIR / "salad_llm_ext_benign_stimuli.jsonl")


@pytest.fixture(scope="module")
def screen() -> list[dict]:
    return _load(EXT_DIR / "salad_llm_screen_stimuli.jsonl")


class FakeClient:
    """Stands in for OllamaClient: every answer is a dismiss, so every eligible second turn runs."""

    decoding = DecodingConfig()

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, model: str, messages: list[dict[str, str]]) -> dict:
        self.calls += 1
        return {"disposition": "dismiss", "parse_valid": True, "raw_response": RAW, "elapsed_seconds": 0.0}


def test_locked_bases_rebuild_the_locked_second_turn_exactly(locked):
    for item in locked:
        for base, _, kind in salad.stage2_variants(item):
            assert er.ext_stage2_messages(item, base, RAW, kind) == salad.stage2_messages(item, base, RAW, kind)


def test_family_two_prompt_differs_from_family_one_only_in_the_payload(locked):
    for item in (item for item in locked if salad.is_hostile(item)):
        for f2, f1 in (("s1_injected_f2", "s1_injected"), ("s1_spot_injected_f2", "s1_spot_injected")):
            expected = json.loads(json.dumps(salad.stage1_messages(item, f1)).replace(
                json.dumps(item["injection_payload"])[1:-1], json.dumps(payload_f2(item["record_id"]))[1:-1]))
            assert er.ext_stage1_messages(item, f2) == expected


def test_family_two_never_injects_a_benign_alert(locked):
    benign = next(item for item in locked if not salad.is_hostile(item))
    with pytest.raises(ValueError):
        er.f2_stimulus(benign)
    with pytest.raises(ValueError):
        er.ext_stage1_messages(benign, "s1_injected_f2")


def test_call_plans_inject_hostile_alerts_only(locked, benign_ext):
    for part, corpus in (("mplacebo", locked), ("f2", locked), ("benign", benign_ext)):
        for item, condition, base, _ in [*er.stage1_steps(part, corpus), *er.stage2_steps(part, corpus)]:
            if not salad.is_hostile(item):
                assert "injected" not in condition and "injected" not in str(base)
    assert not list(er.stage1_steps("mplacebo", locked))


def test_second_turn_plans_have_the_designed_size(locked, benign_ext):
    n_hostile = sum(salad.is_hostile(item) for item in locked)
    assert len(list(er.stage2_steps("mplacebo", locked))) == len(locked) + n_hostile
    assert len(list(er.stage2_steps("f2", locked))) == 4 * n_hostile
    assert len(list(er.stage1_steps("f2", locked))) == 2 * n_hostile
    assert len(list(er.stage2_steps("benign", benign_ext))) == 4 * len(benign_ext)
    assert len(list(er.stage1_steps("benign", benign_ext))) == 2 * len(benign_ext)


def test_no_planned_prompt_leaks_a_label(locked, benign_ext):
    for part, corpus in (("mplacebo", locked), ("f2", locked), ("benign", benign_ext)):
        er.check_leaks(part, corpus)


def test_extension_stimuli_are_disjoint_and_built_like_the_locked_ones(locked, benign_ext, screen):
    ids = [item["record_id"] for item in locked + benign_ext + screen]
    assert len(ids) == len(set(ids))
    keys = set(locked[0])
    assert all(set(item) == keys for item in benign_ext + screen)
    population = json.loads(LOCKED_MANIFEST.read_text())["population_strata"]
    assert len(benign_ext) == 80
    assert all(item["stratum"] == "benign_gate" and item["pitsa_gate"] and item["label"] == 0 for item in benign_ext)
    assert sum(item["stratum_weight"] for item in benign_ext) == pytest.approx(population["benign_gate"])
    assert Counter(item["stratum"] for item in screen) == {name: SCREENING_RULE.per_stratum for name in population}
    manifest = json.loads((EXT_DIR / "salad_llm_ext_stimuli_manifest.json").read_text())
    assert manifest["seed"] == EXTENSION_SEED and manifest["disjoint_from_locked_and_each_other"] is True
    assert manifest["refit_reproduction_max_abs_diff"] == 0.0
    assert manifest["locked_stimuli_rebuilt_identically"] == len(locked)


def test_runner_never_writes_a_base_store(tmp_path, locked, benign_ext):
    base_path = tmp_path / "base.jsonl"
    base = ResponseStore(base_path)
    corpus = salad.select_stimuli(locked, 2)
    for item in corpus:
        for condition in ("s1_clean", "s1_injected"):
            base.append({"model": "m", "stimulus_id": item["stimulus_id"], "condition": condition,
                         "disposition": "dismiss", "raw_response": RAW})
    before = base_path.read_bytes()
    store = ResponseStore(tmp_path / "ext.jsonl")
    lookup = er.BaseLookup(store, [base_path])
    client = FakeClient()
    er.run_part(client, store, lookup, "m", "digest", "mplacebo", corpus)
    er.run_part(client, store, lookup, "m", "digest", "benign", salad.select_stimuli(benign_ext, 2))
    assert base_path.read_bytes() == before
    gated = [item for item in corpus if item["pitsa_gate"]]
    expected = sum(1 + salad.is_hostile(item) for item in gated)
    assert sum(record["condition"].startswith("s2_mplacebo") for record in store.records.values()
               if record["stimulus_id"].startswith("SALAD_")) == expected
    calls = client.calls
    er.run_part(client, store, lookup, "m", "digest", "mplacebo", corpus)
    assert client.calls == calls, "a resumed part issues no call twice"


def test_missing_opening_answer_stops_the_run(tmp_path, locked):
    base_path = tmp_path / "base.jsonl"
    base_path.write_text("")
    store = ResponseStore(tmp_path / "ext.jsonl")
    with pytest.raises(RuntimeError):
        er.run_part(FakeClient(), store, er.BaseLookup(store, [base_path]), "m", "d", "mplacebo", locked[:1])
    with pytest.raises(FileNotFoundError):
        er.BaseLookup(store, [tmp_path / "absent.jsonl"])


def _row(j: float, validity: float, latency: float) -> dict:
    return {"youden_j": j, "parse_validity": validity, "median_latency_seconds": latency}


def test_screening_rule_is_applied_mechanically():
    rule = ScreeningRule()
    assert select_model({"a": _row(0.5, 1.0, 9.0), "b": _row(0.3, 1.0, 1.0)}, rule) == "a"
    assert select_model({"a": _row(0.9, 0.90, 1.0), "b": _row(0.1, 0.95, 9.0)}, rule) == "b"
    assert select_model({"a": _row(0.4, 1.0, 5.0), "b": _row(0.4, 1.0, 2.0)}, rule) == "b"
    assert select_model({"a": None, "b": _row(0.9, 0.5, 1.0)}, rule) is None


def test_screening_summary_counts_by_label(screen):
    by_id = {item["stimulus_id"]: item for item in screen}
    records = [{"stimulus_id": item["stimulus_id"], "elapsed_seconds": 1.0,
                "disposition": "escalate" if salad.is_hostile(item) else "dismiss"} for item in screen]
    records[0]["disposition"] = None
    row = summarize(records, by_id)
    assert row["n"] == 40 and row["parse_validity"] == pytest.approx(39 / 40)
    assert row["hostile_escalation_rate"] == 1.0 and row["benign_escalation_rate"] == 0.0
    assert row["youden_j"] == 1.0
