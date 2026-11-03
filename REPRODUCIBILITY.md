# Reproducibility Guide

This guide gives the exact commands to run the verification test suite, reproduce the statistical analyses from the raw language-model response stores, and regenerate all manuscript macros, tables, and figures.

---

## 1. Environment Setup

- **Python**: Python `3.10+`
- **Dependencies**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`, `matplotlib`, `requests`, `pytest`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Automated Verification Suite (`pytest`)

Run the complete test suite verifying structural feature extraction, split hygiene, prompt integrity (Gate G2), completeness and hash verification (Gates G4 and G4-ext), macro formatting (Gate G5), extension design invariants (Gates E2 and E3), and table generation guards (Gate E6):

```bash
pytest -q
```

---

## 3. Reproducing the Statistical Analyses From Raw Response Stores

### 3.1 Primary 4-Model Triage Analysis (`results/triage/`)

```bash
python3 experiments/analyze_triage_study.py \
  --paper-root . \
  --salad-responses data/responses/salad_responses.jsonl \
  --otrf-responses data/responses/otrf_responses.jsonl \
  --mode official \
  --output-dir /tmp/repro_triage
```

This verifies Gate G4 for all 4 locked models (`qwen2.5:3b`, `llama3.2:3b`, `phi3:latest`, `qwen2.5:7b`) and writes `triage_completeness_report.json` and `triage_study_analysis.json`.

### 3.2 Added 5th Model (`mistral:7b`) Locked-Design Analysis (`results/triage_added/`)

```bash
python3 experiments/analyze_triage_study.py \
  --paper-root . \
  --salad-responses data/responses_extension/salad_responses_added.jsonl \
  --otrf-responses data/responses_extension/otrf_responses_added.jsonl \
  --mode official \
  --output-dir /tmp/repro_triage_added
```

### 3.3 Prespecified Extension Analysis (`results/triage_extension/`)

```bash
python3 experiments/analyze_extension_study.py \
  --paper-root . \
  --mode official \
  --added-model mistral:7b \
  --output-dir /tmp/repro_extension
```

This verifies Gate G4-ext across all 5 models, checks that every locked decision and rate matches the locked analysis exactly, and writes `extension_completeness_report.json` and `extension_study_analysis.json`.

---

## 4. Regenerating Manuscript Value Files, Tables, and Figures

```bash
mkdir -p /tmp/repro_manuscript/generated /tmp/repro_manuscript/tables /tmp/repro_manuscript/figures

# 1. Generate LaTeX value files (llm_values.tex and llm_ext_values.tex)
python3 scripts/make_triage_macros.py \
  --analysis results/triage/triage_study_analysis.json \
  --output-dir /tmp/repro_manuscript/generated

python3 scripts/make_extension_macros.py \
  --analysis results/triage_extension/extension_study_analysis.json \
  --added-analysis results/triage_added/triage_study_analysis.json \
  --output-dir /tmp/repro_manuscript/generated

# 2. Generate main and extension LaTeX tables
python3 scripts/make_triage_tables.py \
  --analysis results/triage/triage_study_analysis.json \
  --values /tmp/repro_manuscript/generated/llm_values.tex \
  --output-dir /tmp/repro_manuscript/tables

python3 scripts/make_extension_tables.py \
  --analysis results/triage_extension/extension_study_analysis.json \
  --values /tmp/repro_manuscript/generated/llm_ext_values.tex \
  --locked-values /tmp/repro_manuscript/generated/llm_values.tex \
  --output-dir /tmp/repro_manuscript/tables

# 3. Generate expected review cost region map and break-even curve data
python3 scripts/make_triage_figures.py \
  --analysis results/triage/triage_study_analysis.json \
  --output /tmp/repro_manuscript/figures/fig_loss_regions.pdf

python3 scripts/generate_break_even_data.py \
  --analysis results/triage/triage_study_analysis.json \
  --output-dir /tmp/repro_manuscript/figures
```

---

## 5. Re-Running Live Ollama Collection (Optional)

To re-run model inference locally with Ollama (`0.13.5` used in the study; exact model digests in `configs/ollama_models_main.json` and `configs/ollama_models_extension.json`):

```bash
# Pull pinned model tags
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull phi3:latest
ollama pull qwen2.5:7b
ollama pull mistral:7b

# Run or resume collection (ResponseStore skips already-completed (model, stimulus_id, condition) tuples)
python3 experiments/salad_triage_runner.py --paper-root . --models qwen2.5:3b llama3.2:3b phi3:latest qwen2.5:7b --output data/responses/salad_responses.jsonl
python3 experiments/otrf_probe_runner.py --paper-root . --models qwen2.5:3b llama3.2:3b phi3:latest qwen2.5:7b --output data/responses/otrf_responses.jsonl
```

---

## 6. Verifying Cryptographic File Integrity

```bash
shasum -a 256 -c RELEASE_SHA256.txt
```
