#!/usr/bin/env bash
# End-to-end runner for the C0/C1/C2 sycophancy source-manipulation experiment.
#
# Pre-req:  export TOGETHER_API_KEY=...
# Usage:    bash reasoning_extension/scripts/run_source_manipulation_pipeline.sh
#
# Steps:
#   1. Generate 52 dilemmas × 2 stances = 104 C2 arguments via Llama-3.3-70B (Together).
#   2. Run C1 (impersonal) — disagree-only, persona=respond, reuses Step-1 from C0 cache.
#   3. Run C2 (user_reasoned) — same scope, with C2 arguments injected.
#   4. Run the 3-arm analysis (chi-square, z-tests, bootstrap CIs, bar chart).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Auto-load .env if present
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${TOGETHER_API_KEY:-}" ]]; then
  echo "ERROR: TOGETHER_API_KEY not set. Add it to .env or export it first."
  exit 1
fi

PY=${PY:-python3}
if [[ -x "reasoning_extension/.venv/bin/python" ]]; then
  PY="reasoning_extension/.venv/bin/python"
fi
echo "Using Python: $PY"

MODEL=deepseek-r1-distill-qwen-7b
RESULTS=reasoning_extension/results/sycophancy
DATA=reasoning_extension/data
ARGS_FILE=$DATA/sycophancy_c2_arguments.json
STEP1_CACHE=$RESULTS/cot_bidirectional

mkdir -p "$DATA" "$RESULTS"

echo
echo "=================================================================="
echo " STEP 1/4 — Verify C2 arguments JSON (hand-authored, 104 total)"
echo "=================================================================="
if [[ ! -f "$ARGS_FILE" ]]; then
  echo "ERROR: $ARGS_FILE not found. Author it or run generate_c2_arguments.py first."
  exit 1
fi
COUNT=$($PY -c "import json; d=json.load(open('$ARGS_FILE')); print(sum(1 for v in d.values() for k in ('yes_argument','no_argument') if v.get(k)))")
echo "  -> $COUNT C2 arguments in $ARGS_FILE"
if [[ "$COUNT" != "104" ]]; then
  echo "ERROR: expected 104, got $COUNT. Inspect $ARGS_FILE."
  exit 1
fi

echo
echo "=================================================================="
echo " STEP 2/4 — Run C1 (impersonal) experiment"
echo "=================================================================="
$PY reasoning_extension/scripts/run_sycophancy_api.py \
    --model "$MODEL" \
    --output-dir "$RESULTS" \
    --source-condition impersonal \
    --persona-filter respond \
    --step1-cache-dir "$STEP1_CACHE"

echo
echo "=================================================================="
echo " STEP 3/4 — Run C2 (user_reasoned) experiment"
echo "=================================================================="
$PY reasoning_extension/scripts/run_sycophancy_api.py \
    --model "$MODEL" \
    --output-dir "$RESULTS" \
    --source-condition user_reasoned \
    --c2-arguments-file "$ARGS_FILE" \
    --persona-filter respond \
    --step1-cache-dir "$STEP1_CACHE"

echo
echo "=================================================================="
echo " STEP 4/4 — Analyze C0 / C1 / C2"
echo "=================================================================="
$PY reasoning_extension/scripts/analyze_source_manipulation.py \
    --base-dir "$RESULTS" \
    --model "$MODEL"

echo
echo "Pipeline complete."
echo "Figure (if matplotlib available): $RESULTS/source_manipulation_flip_rates.png"
