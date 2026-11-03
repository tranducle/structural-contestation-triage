# Data Provenance and Schema Documentation

This document describes the **included stimulus benchmarks and language-model response stores** (`data/`) as well as the **upstream telemetry corpora** (`SALAD-SOC`, `LMD-2023`, and `OTRF Security-Datasets`) referenced in `configs/dataset_provenance.json`.

---

## 1. Included Stimulus Benchmarks (`data/stimuli/`)

| File | Records | Role |
| :--- | ---: | :--- |
| `salad_llm_stimuli.jsonl` | 160 | Locked SALAD triage sample (`40` alerts per stratum across `hostile_gate`, `hostile_nogate`, `benign_gate`, `benign_nogate`), sampled without replacement (seed `20261022`) from the official SALAD test split (`416,781` alerts). |
| `salad_llm_stimuli_manifest.json` | 1 | Stratum population counts (`hostile_gate: 73,138`, `hostile_nogate: 15,432`, `benign_gate: 32,378`, `benign_nogate: 295,833`), PITSA threshold (`0.719172`), and refit verification metadata. |
| `salad_llm_ext_benign_stimuli.jsonl` | 80 | Prespecified extension sample of `80` additional `benign_gate` alerts (seed `20261023`), disjoint from the locked sample, expanding `benign_gate` to `120` alerts (`240` extended alerts total). |
| `salad_llm_screen_stimuli.jsonl` | 40 | Disjoint screening sample (`10` alerts per stratum) used exclusively by `experiments/screen_models.py` to select the added fifth model (`mistral:7b`). |
| `salad_llm_ext_stimuli_manifest.json` | 1 | Provenance and disjointness verification manifest for the extension and screening stimuli. |
| `otrf_high_confidence_stimuli.jsonl` | 20 | Attack-only provenance probe comprising `20` host telemetry events across `20` OTRF Security-Datasets archives (`6` credential access, `6` lateral movement, `5` persistence, `3` execution). |
| `otrf_stimuli_summary.json` | 1 | Summary metadata for the OTRF provenance probe stimuli. |

### Schema of `salad_llm_stimuli.jsonl`

Each JSONL record contains:
- `stimulus_id` (`str`): Stratum-indexed identifier (replaced by an opaque SHA-256 handle `ALERT-<8hex>` before any prompt reaches a model).
- `record_id` (`str`): Upstream SALAD test-split record identifier.
- `label` (`int`): Ground-truth binary class (`1` = hostile, `0` = benign).
- `ground_truth` (`str`): `"escalate"` for hostile alerts, `"dismiss"` for benign alerts.
- `stratum` (`str`): One of `hostile_gate`, `hostile_nogate`, `benign_gate`, `benign_nogate`.
- `stratum_weight` (`float`): Inverse sampling probability $N_h / n_h$ mapping sample estimates back to the `416,781`-alert test population.
- `sensor_fields` (`dict`): The 13 pre-triage observable sensor fields presented to the language model (all 13 label-derived or post-triage fields are excluded).
- `pitsa_score` (`float`): Prompt-isolated additive log-likelihood ratio score $\Lambda(e)$.
- `pitsa_threshold` (`float`): Validation-selected operating threshold $\tau = 0.719172$.
- `pitsa_gate` (`bool`): Gate indicator $g(e) = \mathbf{1}[\Lambda(e) \ge \tau]$.
- `pillar_values` (`dict`): Categorical bin values for `s1_proto_ports`, `s2_flow_volume`, `s3_duration_segment`, `s4_burst`, `s5_network_scope`.
- `pillar_contributions` (`dict`): Signed per-pillar log-likelihood ratio contributions.
- `top_pillars` (`list[str]`): The two pillars with the largest signed contributions reported in the structural cue.
- `injection_payload` (`str`): Deterministically assigned payload from payload family 1 (`F1`).

---

## 2. Included Language-Model Response Stores (`data/screening/`, `data/responses/`, `data/responses_extension/`)

Every model invocation is logged with its full message list (`messages`), the SHA-256 digest of the message list (`messages_sha256`), the raw JSON response (`raw_response`), parsed disposition (`disposition`, `confidence`, `evidence`, `rationale`, `parse_valid`), the 12-character Ollama model blob digest (`model_digest`), and decoding options (`temperature: 0.0`, `seed: 20261014`, `num_predict: 256`, `num_ctx: 4096`).

1. **`data/screening/`**:
   - `screen_responses.jsonl` and `screen_selection.json`: Clean opening-turn screening calls on `salad_llm_screen_stimuli.jsonl` across candidate models, selecting `mistral:7b`.
2. **`data/responses/`**:
   - `salad_responses.jsonl` + per-model manifests: Complete Stage-1 (`s1_clean`, `s1_injected`, `s1_spot_clean`, `s1_spot_injected`), gate-triggered Stage-2 (`s2_struct_after_*`, `s2_placebo_after_*`, `s2_spotstruct_after_*`), and determinism replicate (`rep_s1_clean`) calls for the 4 locked models (`qwen2.5:3b`, `llama3.2:3b`, `phi3:latest`, `qwen2.5:7b`).
   - `otrf_responses.jsonl` + per-model manifests: Complete OTRF provenance probe calls for the 4 locked models.
3. **`data/responses_extension/`**:
   - `salad_responses_added.jsonl` and `otrf_responses_added.jsonl` + manifests: Locked-design SALAD and OTRF calls for the added fifth model (`mistral:7b`).
   - `salad_ext_responses.jsonl` + per-model manifests: Prespecified extension calls across all 5 models (`mplacebo` matched-placebo second turns, `f2` format-mimicking payload family 2 opening and second turns, and `benign` extended gated benign alerts).

---

## 3. Upstream Telemetry Corpora (Evaluation 1)

The full-corpus CSV/Parquet files for Evaluation 1 exceed GitHub repository size limits (`2,778,424` SALAD-SOC alerts and `1,777,930` LMD-2023 records). Their exact SHA-256 digests and record counts are locked in `configs/dataset_provenance.json`, and all precomputed Evaluation 1 metrics and bootstrap intervals are stored in `results/detector/`:
- **SALAD-SOC**: Chalaemwongwan et al. (2026), built from CICIDS2017 and UNSW-NB15 flow records (`1,944,887` train, `416,756` validation, `416,781` test).
- **LMD-2023**: Smiliotopoulos et al. (2023), lateral-movement Sysmon telemetry (`1,777,930` labeled records across Normal, EoRS, and EoHT).
- **OTRF Security-Datasets**: Open Threat Research Forge host attack archives (`20` archives sampled for the provenance probe).
