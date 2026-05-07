#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Re-evaluate all 20 ft-* checkpoints with the patched evaluate.py.
#
# What changed vs. the first pass in output.txt:
#   * evaluate.py now routes `datasets/<name>_test.jsonl` to the right
#     dataset branch (was falling through to strict equality for all).
#   * gsm8k uses findall(r"####\s*([^\s#]+)")[-1] + numeric normalization.
#   * rotten_tomatoes uses first-word match (case-insensitive).
#   * spider truncates generation at the first `;` before sqlite3 compare.
#   * empty sim_list no longer crashes np.quantile.
#   * run_stp.sh (and this script) pass a dataset-appropriate
#     --max_new_tokens so the model actually gets to EMIT the answer.
#
# Output goes to output_reeval.txt (preserving the original output.txt).
#
# Usage:   bash reevaluate.sh
#   or:    GPU=1 bash reevaluate.sh        # pin to a specific GPU
#   or:    ONLY=rotten_tomatoes bash reevaluate.sh   # one dataset only
# ---------------------------------------------------------------------------
set -u

cd "$(dirname "$0")"

: "${GPU:=0}"
: "${ONLY:=}"           # comma- or space-separated include-list (rotten_tomatoes|synth|spider|gsm8k)
: "${EXCLUDE:=}"        # comma- or space-separated exclude-list; applied after ONLY
: "${MODEL_NAME:=meta-llama/Llama-3.2-1B-Instruct}"
: "${OUT_FILE:=output_reeval.txt}"

# Normalise ONLY/EXCLUDE to space-separated
ONLY=${ONLY//,/ }
EXCLUDE=${EXCLUDE//,/ }

PYTHON=/home/khanhnt/.conda/envs/LinOSS/bin/python

_mnt_for_dataset() {
  # GT-length quantiles (Llama-3.2 tokenizer, checked on the actual test sets):
  #   rotten_tomatoes : 1 token          -> 4   is generous
  #   synth           : <= 32            -> 96  is safe
  #   spider          : ~100 max         -> 128 is safe
  #   gsm8k           : q99=233, max=310 -> 256 covers 99% (vs 384 which doubles wall time)
  case "$1" in
    rotten_tomatoes)  echo 4   ;;
    synth)            echo 96  ;;
    spider)           echo 128 ;;
    gsm8k)            echo 256 ;;
    hellaswag)        echo 8   ;;
    nq_open)          echo 64  ;;
    *)                echo 128 ;;
  esac
}

_dataset_from_folder() {
  local f="$1"
  for ds in rotten_tomatoes synth spider gsm8k hellaswag nq_open; do
    if [[ "$f" == *"-${ds}-"* ]]; then
      echo "$ds"; return 0
    fi
  done
  echo ""
}

# Print a run banner identical in spirit to run_stp.sh so output_reeval.txt
# parses the same way as output.txt.
_print_header() {
  local folder="$1" dataset="$2"
  local method="unknown"
  case "$folder" in
    ft-r-*)          method="regular"                    ;;
    ft-j-*)          method="jepa"                       ;;
    ft-c-d_t-*)      method="control_JEPA norm=d_t"      ;;
    ft-c-v_geo-*)    method="control_JEPA norm=v_geo"    ;;
    ft-reach-*)      method="reach_JEPA"                 ;;
  esac
  echo "Success Rate: ${method} ${MODEL_NAME} dataset=datasets/${dataset} folder=${folder}" >> "$OUT_FILE"
}

mkdir -p logs

: > "$OUT_FILE"   # fresh file
echo "# Re-evaluation sweep at $(date)" >> "$OUT_FILE"
echo "# GPU=${GPU}  ONLY=${ONLY:-<all>}" >> "$OUT_FILE"

for folder in ft-*/; do
  folder="${folder%/}"
  [[ -f "${folder}/config.json" ]] || continue   # skip non-model dirs

  dataset=$(_dataset_from_folder "$folder")
  if [[ -z "$dataset" ]]; then
    echo "[SKIP] $folder (no known dataset tag in name)" | tee -a "$OUT_FILE"
    continue
  fi

  if [[ -n "$ONLY" ]]; then
    in_only=0
    for k in $ONLY; do [[ "$k" == "$dataset" ]] && in_only=1; done
    [[ $in_only -eq 1 ]] || continue
  fi
  if [[ -n "$EXCLUDE" ]]; then
    in_exc=0
    for k in $EXCLUDE; do [[ "$k" == "$dataset" ]] && in_exc=1; done
    [[ $in_exc -eq 0 ]] || continue
  fi

  mnt=$(_mnt_for_dataset "$dataset")
  echo ""
  echo "============================================================"
  echo "[$dataset][mnt=${mnt}] re-eval: ${folder}"
  echo "============================================================"

  _print_header "$folder" "$dataset"

  CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} evaluate.py \
    --model_name="${folder}" \
    --original_model_name="${MODEL_NAME}" \
    --input_file="datasets/${dataset}_test.jsonl" \
    --output_file="eval.jsonl" \
    --split_tune_untune \
    --nosplit_data \
    --spider_path=spider_data/database \
    --max_new_tokens=${mnt} 2>&1 | tee -a "$OUT_FILE"
done

echo ""
echo "============================================================"
echo "DONE. Per-run Success Rate lines are in ${OUT_FILE}."
echo "Extract summary with:"
echo "    grep -n '^Success Rate: ft-' ${OUT_FILE}"
echo "============================================================"
