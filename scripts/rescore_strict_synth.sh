#!/usr/bin/env bash
# Re-run synth eval with strict equality (evaluate.py `kind == "synth"` branch).
# Uses conda env `controlJEPA`.
#
# Usage:
#   bash scripts/rescore_strict_synth.sh <checkpoint_dir> [original_model_name] [optional_max_examples]
#
# Examples:
#   bash scripts/rescore_strict_synth.sh gemma-2-2b-it/ft-j-synth-2e-5-0.02-0-82 google/gemma-2-2b-it
#   bash scripts/rescore_strict_synth.sh gemma-2-2b-it/ft-j-synth-2e-5-0.02-0-82 google/gemma-2-2b-it 256
#
# Default test file: datasets/synth_test_strict_equal.jsonl (full copy of synth_test).
# Quick smoke (256 rows): SYNTH_TEST_FILE=datasets/synth_test_strict_smoke256.jsonl bash ...
# Override: SYNTH_TEST_FILE=datasets/synth_test.jsonl bash scripts/rescore_strict_synth.sh ...

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:?usage: $0 <checkpoint_dir> [original_hf_model_id] [max_examples]}"
ORIGINAL="${2:-google/gemma-2-2b-it}"
MAX_EX="${3:-}"
TEST_FILE="${SYNTH_TEST_FILE:-datasets/synth_test.jsonl}"
OUT_LOG="${OUT_LOG:-rescore_strict_synth.log}"

if [[ ! -d "$MODEL" ]]; then
  echo "Warning: checkpoint path is not a directory: $MODEL" >&2
fi

CMD=(
  conda run -n controlJEPA --no-capture-output
  python evaluate.py
  --model_name="$MODEL"
  --original_model_name="$ORIGINAL"
  --input_file="$TEST_FILE"
  --output_file=eval_strict_synth.jsonl
  --nosplit_data
  --split_tune_untune
  --spider_path="${SPIDER_PATH:-spider_data/database}"
  --max_new_tokens=96
)
if [[ -n "$MAX_EX" ]]; then
  CMD+=(--max_examples="$MAX_EX")
fi

echo "=== strict synth rescore ===" | tee -a "$OUT_LOG"
echo "test_file=$TEST_FILE model=$MODEL original=$ORIGINAL max_examples=${MAX_EX:-all}" | tee -a "$OUT_LOG"
"${CMD[@]}" 2>&1 | tee -a "$OUT_LOG"
