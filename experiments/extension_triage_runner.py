"""Run the v3 extension calls for one or more models.

Parts, each resumable from the extension store:
  mplacebo   locked 160 alerts; matched-placebo second turn after s1_clean and s1_injected,
             issued under the locked rule (PITSA gate fired, opening answer dismiss). The opening
             answers are read from the read-only base stores, so the second turn continues the
             same conversation the locked cue and placebo continued.
  f2         locked 80 hostile alerts; opening turns s1_injected_f2 and s1_spot_injected_f2, then
             the cue, the placebo and the matched placebo after s1_injected_f2 and the cue after
             s1_spot_injected_f2.
  benign     80 extension benign gated alerts; the locked opening turns and second turns, plus the
             matched placebo after s1_clean.
No record is ever written to a base store.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ('src', 'experiments', 'scripts'):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator

import salad_triage_runner as salad
from ext_design import (
    F2_TO_LOCKED_BASE,
    MATCHED_PLACEBO_CUE,
    S1_F2,
    f2_stage2_variants,
    mplacebo_condition,
    payload_f2,
)
from llm_common import PLACEBO_CUE, DecodingConfig, OllamaClient, ResponseStore, run_call, sha256_text

logger = logging.getLogger(__name__)

PARTS = ("mplacebo", "f2", "benign")
Stimulus = dict[str, Any]
Step = tuple[Stimulus, str, str | None, str | None]  # stimulus, condition, base condition, cue kind


def f2_stimulus(stimulus: Stimulus) -> Stimulus:
    """Copy of a hostile stimulus carrying its family-2 payload in the locked payload field."""
    if not salad.is_hostile(stimulus):
        raise ValueError(f"benign alert {stimulus['stimulus_id']} must not be injected")
    return {**stimulus, "injection_payload": payload_f2(stimulus["record_id"])}


def ext_stage1_messages(stimulus: Stimulus, condition: str) -> list[dict[str, str]]:
    if condition in S1_F2:
        return salad.stage1_messages(f2_stimulus(stimulus), F2_TO_LOCKED_BASE[condition])
    return salad.stage1_messages(stimulus, condition)


def cue_text(stimulus: Stimulus, cue_kind: str) -> str:
    if cue_kind == "structural":
        return salad.structural_cue(stimulus)
    if cue_kind == "placebo":
        return PLACEBO_CUE
    if cue_kind == "matched_placebo":
        return MATCHED_PLACEBO_CUE
    raise ValueError(f"unknown cue kind {cue_kind!r}")


def ext_stage2_messages(
    stimulus: Stimulus, base_condition: str, base_raw_response: str, cue_kind: str
) -> list[dict[str, str]]:
    return ext_stage1_messages(stimulus, base_condition) + [
        {"role": "assistant", "content": base_raw_response},
        {"role": "user", "content": cue_text(stimulus, cue_kind)},
    ]


def stage1_steps(part: str, stimuli: list[Stimulus]) -> Iterator[Step]:
    for stimulus in stimuli:
        if part == "f2" and salad.is_hostile(stimulus):
            for condition in S1_F2:
                yield stimulus, condition, None, None
        elif part == "benign":
            for condition in salad.STAGE1_ALL:
                yield stimulus, condition, None, None


def stage2_steps(part: str, stimuli: list[Stimulus]) -> Iterator[Step]:
    """Every second turn a part may issue; each is run only if its opening answer qualifies."""
    for stimulus in stimuli:
        if part == "mplacebo":
            for base in ["s1_clean"] + (["s1_injected"] if salad.is_hostile(stimulus) else []):
                yield stimulus, mplacebo_condition(base), base, "matched_placebo"
        elif part == "f2" and salad.is_hostile(stimulus):
            for base, condition, kind in f2_stage2_variants():
                yield stimulus, condition, base, kind
        elif part == "benign":
            for base, condition, kind in salad.stage2_variants(stimulus):
                yield stimulus, condition, base, kind
            yield stimulus, mplacebo_condition("s1_clean"), "s1_clean", "matched_placebo"


class BaseLookup:
    """Opening answers: the extension store first, then the read-only base stores."""

    def __init__(self, ext_store: ResponseStore, base_paths: list[Path]) -> None:
        missing = [str(path) for path in base_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"base stores not found: {missing}")
        self.ext_store = ext_store
        self.base_stores = [ResponseStore(path) for path in base_paths]

    def get(self, model: str, stimulus_id: str, condition: str) -> dict[str, Any] | None:
        for store in [self.ext_store, *self.base_stores]:
            record = store.get(model, stimulus_id, condition)
            if record is not None:
                return record
        return None


def check_leaks(part: str, stimuli: list[Stimulus]) -> None:
    for stimulus, condition, _, _ in stage1_steps(part, stimuli):
        leaks = salad.leaked_fields(ext_stage1_messages(stimulus, condition), stimulus)
        if leaks:
            raise RuntimeError(f"label leakage in {stimulus['stimulus_id']} {condition}: {leaks}")


def run_part(
    client: OllamaClient, store: ResponseStore, lookup: BaseLookup, model: str, digest: str,
    part: str, stimuli: list[Stimulus],
) -> None:
    for stimulus, condition, _, _ in stage1_steps(part, stimuli):
        run_call(client, store, model=model, model_digest=digest, stimulus=stimulus, condition=condition,
                 stage=1, base_condition=None, messages=ext_stage1_messages(stimulus, condition))
    for stimulus, condition, base, kind in stage2_steps(part, stimuli):
        base_record = lookup.get(model, stimulus["stimulus_id"], str(base))
        if base_record is None:
            raise RuntimeError(f"{model} {stimulus['stimulus_id']}: opening answer {base} missing")
        if not salad.needs_stage2(stimulus, base_record):
            continue
        messages = ext_stage2_messages(stimulus, str(base), base_record["raw_response"], str(kind))
        run_call(client, store, model=model, model_digest=digest, stimulus=stimulus, condition=condition,
                 stage=2, base_condition=base, messages=messages)


def load_jsonl(path: Path) -> list[Stimulus]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-stores", nargs="+", type=Path, required=True)
    parser.add_argument("--limit-per-stratum", type=int, default=None, help="smoke checks only")
    parser.add_argument("--base-url", default="http://localhost:11434")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    v3 = args.paper_root
    locked_path = v3 / "data/stimuli/salad_llm_stimuli.jsonl"
    benign_path = v3 / "data/stimuli/salad_llm_ext_benign_stimuli.jsonl"
    corpora = {
        "mplacebo": salad.select_stimuli(load_jsonl(locked_path), args.limit_per_stratum),
        "f2": salad.select_stimuli(load_jsonl(locked_path), args.limit_per_stratum),
        "benign": salad.select_stimuli(load_jsonl(benign_path), args.limit_per_stratum),
    }
    for part in args.parts:
        check_leaks(part, corpora[part])

    decoding = DecodingConfig()
    client = OllamaClient(args.base_url, decoding)
    digests = client.model_digests()
    missing = [model for model in args.models if model not in digests]
    if missing:
        raise RuntimeError(f"models not available in Ollama: {missing}")

    store = ResponseStore(args.output)
    lookup = BaseLookup(store, args.base_stores)
    tag = "_".join(model.replace(":", "-").replace(".", "-") for model in args.models)
    manifest = {
        "ollama_version": client.version(),
        "models": {model: digests[model] for model in args.models},
        "decoding": decoding.options(),
        "parts": args.parts,
        "base_stores": [str(path) for path in args.base_stores],
        "locked_stimuli_sha256": sha256_text(locked_path.read_text(encoding="utf-8")),
        "benign_extension_stimuli_sha256": sha256_text(benign_path.read_text(encoding="utf-8")),
        "limit_per_stratum": args.limit_per_stratum,
    }
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest_{tag}_{'-'.join(args.parts)}.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for model in args.models:
        logger.info("starting model %s (%s), parts %s", model, digests[model], args.parts)
        client.make_exclusive(model)
        for part in args.parts:
            run_part(client, store, lookup, model, digests[model], part, corpora[part])
        client.unload(model)
    logger.info("finished: %d records in %s", len(store.records), args.output)


if __name__ == "__main__":
    main()
