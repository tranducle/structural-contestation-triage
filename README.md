# Channel-Isolated Structural Contestation for Language-Model Alert Triage Under Indirect Prompt Injection

This repository contains the reproducibility package (source code, experimental runners, hashed analysis locks, deterministic stimulus benchmarks, complete language-model triage response logs, and statistical analysis outputs) for the study:

> **Channel-Isolated Structural Contestation for Language-Model Alert Triage Under Indirect Prompt Injection: A Design Science Evaluation**

## Overview

Security operations centers (SOCs) increasingly use large language models (LLMs) to summarize, prioritize, and recommend dispositions for security telemetry alerts. Because endpoint-controlled strings (such as annotations, command lines, and event descriptions) enter the triage prompt, LLM copilots are exposed to **indirect prompt injection**: adversarial instructions embedded inside untrusted telemetry that argue for dismissing hostile activity.

This repository implements and evaluates **Prompt-Isolated Telemetry Structural Assessor (PITSA)** and **gate-bounded structural contestation**:
1. **Evaluation 1 (Structural Detector Benchmark)**: Evaluates PITSA against structural and text-inclusive learned baselines (`StructuralLogistic`, `StructuralRandomForest`, `JointWordLogistic`, `CharacterNgramLogistic`, `JointNaiveBayes`) on the official SALAD-SOC split (`2,778,424` alerts) and on LMD-2023 (`1,777,930` records) across signature-disjoint, future-year, and held-out-station distribution shifts.
2. **Evaluation 2 (Language-Model Triage & Gate-Bounded Contestation)**: Evaluates four locked open-weight instruction models (`qwen2.5:3b`, `llama3.2:3b`, `phi3:latest`, `qwen2.5:7b`) on `160` stratified SALAD test alerts (`40` per label-by-gate stratum, weighted back to the `416,781`-alert test population) and `20` OTRF host attack telemetry probes, plus a prespecified extension adding a fifth screened model (`mistral:7b`), a second format-mimicking payload family (`F2`), `80` additional gated benign alerts (`240` extended alerts total), and a length-and-authority-matched placebo second turn (`M`).

## Repository Architecture

```text
.
|-- README.md                  # Repository overview, quickstart, and map
|-- DATA.md                    # Dataset provenance, stimulus schemas, and response store guide
|-- REPRODUCIBILITY.md         # Step-by-step commands to verify and regenerate all results
|-- LICENSE                    # MIT License
|-- RELEASE_SHA256.txt         # Cryptographic SHA-256 manifest of all released files
|-- requirements.txt           # Pinned Python dependencies
|-- configs/                   # Experiment configs, dataset provenance, hardware & analysis locks
|-- src/                       # Core modules (structural features, policies, loss, statistics)
|-- experiments/               # Experiment runners and gated statistical analysis pipelines
|-- scripts/                   # Stimulus builders, macro/table/figure generators, shell drivers
|-- data/                      # Deterministic JSONL stimuli and complete Ollama response stores
|   |-- stimuli/               # SALAD 160 locked + 80 benign extension + 40 screen + 20 OTRF
|   |-- screening/             # Mechanical model screening selection and response log
|   |-- responses/             # Locked 4-model SALAD and OTRF response logs and manifests
|   `-- responses_extension/   # Extension (mplacebo, F2, benign) and added-model (mistral:7b) logs
|-- results/                   # Precomputed JSON statistical analyses and completeness reports
|   |-- detector/              # Evaluation 1 SALAD and LMD-2023 detector, ablation, runtime JSONs
|   |-- triage/                # Primary 4-model triage analysis and G4 completeness report
|   |-- triage_added/          # Added 5th model (mistral:7b) locked-design analysis and report
|   `-- triage_extension/      # Extension analysis (5 models, F1/F2, matched placebo) and report
|-- docs/                      # Formal policy definitions, propositions, and design notes
`-- tests/                     # Automated pytest suite verifying gates, prompts, and statistics
```

## Quickstart Verification (CPU Only, < 1 Minute)

All raw language-model responses and deterministic stimulus files are included in `data/`. You can run the complete verification test suite and re-run the official statistical analyses without a GPU or external datasets:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the automated verification test suite (gates G2, G4, G5, E2, E3, E5, E6)
pytest -q

# 2. Re-run the primary 4-model statistical analysis from the locked response store
python3 experiments/analyze_triage_study.py \
  --paper-root . \
  --salad-responses data/responses/salad_responses.jsonl \
  --otrf-responses data/responses/otrf_responses.jsonl \
  --mode official \
  --output-dir /tmp/repro_triage

# 3. Re-run the 5-model extension statistical analysis (matched placebo, F2, extended benign)
python3 experiments/analyze_extension_study.py \
  --paper-root . \
  --mode official \
  --added-model mistral:7b \
  --output-dir /tmp/repro_extension
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full instructions on regenerating all LaTeX value files, tables, and figures, and [DATA.md](DATA.md) for dataset provenance and schema details.

## License

Code in this repository is released under the [MIT License](LICENSE). Derived stimulus benchmarks under `data/stimuli/` follow the open research terms of their upstream sources (SALAD-SOC and OTRF Security-Datasets; see [DATA.md](DATA.md)).
