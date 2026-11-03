#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 experiments/analyze_extension_study.py \
  --paper-root "$REPO_ROOT" \
  --mode official \
  --added-model mistral:7b \
  --output-dir /tmp/repro_extension

python3 scripts/make_extension_macros.py \
  --analysis results/triage_extension/extension_study_analysis.json \
  --added-analysis results/triage_added/triage_study_analysis.json \
  --output-dir /tmp/repro_macros
