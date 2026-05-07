#!/usr/bin/env bash
# Run SYNTH generation + strict failure taxonomy for an ft-c-v_geo (or any) checkpoint.
# Uses conda env controlJEPA by default.
#
#   bash scripts/run_synth_vgeo_failure_report.sh
# Optional env vars:
#   MODEL_PATH   (default Llama 3.2 1B v_geo SYNTH ckpt seed 82)
#   ORIGINAL     (HF id for tokenizer/arch)
#   TEST_FILE    (default datasets/synth_test.jsonl)
#   MAX_EX       (empty = full file)
#   OUT          (JSONL path)
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV="${CONDA_ENV:-controlJEPA}"
MODEL_PATH="${MODEL_PATH:-Llama3.2-1B-Instruct/ft-c-v_geo-synth-g0.80-t1e-3-2e-5-0.01-0-82}"
ORIGINAL="${ORIGINAL:-meta-llama/Llama-3.2-1B-Instruct}"
TEST_FILE="${TEST_FILE:-datasets/synth_test.jsonl}"
OUT="${OUT:-results/synth_vgeo_failure_report.jsonl}"

CMD=(
  conda run -n "$ENV" --no-capture-output
  python scripts/synth_gen_failure_report.py
  --model_path="$MODEL_PATH"
  --original_model_name="$ORIGINAL"
  --input_file="$TEST_FILE"
  --out_jsonl="$OUT"
  --device_map="${DEVICE_MAP:-cuda:0}"
  --max_new_tokens="${MAX_NEW_TOKENS:-96}"
  --max_length="${MAX_LENGTH:-512}"
)
if [[ -n "${MAX_EX:-}" ]]; then
  CMD+=(--max_examples="$MAX_EX")
fi

"${CMD[@]}"
