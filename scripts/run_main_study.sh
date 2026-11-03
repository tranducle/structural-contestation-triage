#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 experiments/analyze_triage_study.py \
  --paper-root "$REPO_ROOT" \
  --salad-responses data/responses/salad_responses.jsonl \
  --otrf-responses data/responses/otrf_responses.jsonl \
  --mode official \
  --output-dir /tmp/repro_triage

python3 scripts/make_triage_macros.py \
  --analysis results/triage/triage_study_analysis.json \
  --output-dir /tmp/repro_macros
