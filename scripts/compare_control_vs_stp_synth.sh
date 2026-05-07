#!/usr/bin/env bash
# Compare Control JEPA (control_v_geo ckpt) vs STP baseline on SYNTH with dual metrics.
# evaluate.py prints:
#   - Success Rate  -> strict EM (paper-comparable when eval matches upstream)
#   - SYNTH_relaxed_engineering -> prefix + non-alnum-boundary (deployment-style)
#   - SYNTH_relaxed_only_vs_strict -> how many failures are "near misses" under relaxed
#
# Usage (from repo root, GPU + conda env with deps):
#   SYNTH_TEST_FILE=datasets/synth_test_strict_smoke256.jsonl \
#   CONTROL_CKPT=gemma-2-2b-it/ft-c-v_geo-synth-g0.9-t0-2e-5-0.01-0-23 \
#   STP_CKPT=gemma-2-2b-it/ft-j-synth-2e-5-0.02-0-82 \
#   ORIGINAL=google/gemma-2-2b-it MAX_EXAMPLES=256 \
#   bash scripts/compare_control_vs_stp_synth.sh
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ENV="${CONDA_ENV:-controlJEPA}"
ORIGINAL="${ORIGINAL:-google/gemma-2-2b-it}"
TEST_FILE="${SYNTH_TEST_FILE:-datasets/synth_test.jsonl}"
CONTROL_CKPT="${CONTROL_CKPT:?set CONTROL_CKPT (e.g. .../ft-c-v_geo-synth-...)}"
STP_CKPT="${STP_CKPT:?set STP_CKPT (e.g. .../ft-j-synth-...)}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

run_one() {
  local tag="$1" ckpt="$2" log="$3"
  local -a cmd=(
    conda run -n "$ENV" --no-capture-output
    python evaluate.py
    --model_name="$ckpt"
    --original_model_name="$ORIGINAL"
    --input_file="$TEST_FILE"
    --output_file="eval_${tag}_synth.jsonl"
    --nosplit_data
    --split_tune_untune
    --spider_path="${SPIDER_PATH:-spider_data/database}"
    --max_new_tokens=96
  )
  if [[ -n "$MAX_EXAMPLES" ]]; then
    cmd+=(--max_examples="$MAX_EXAMPLES")
  fi
  echo "=== ${tag}: ${ckpt} ===" | tee "$log"
  "${cmd[@]}" 2>&1 | tee -a "$log"
}

LOG_DIR="${LOG_DIR:-.}"
mkdir -p "$LOG_DIR"

run_one "control_v_geo" "$CONTROL_CKPT" "${LOG_DIR}/synth_dual_control_jepa.log"
run_one "stp" "$STP_CKPT" "${LOG_DIR}/synth_dual_stp.log"

echo ""
echo "Grep summaries:"
grep -E "^(Success Rate|SYNTH_)" "${LOG_DIR}/synth_dual_control_jepa.log" || true
grep -E "^(Success Rate|SYNTH_)" "${LOG_DIR}/synth_dual_stp.log" || true
