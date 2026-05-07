#!/usr/bin/env bash
# Re-evaluate a subset of spider checkpoints on a single GPU.
#   GPU=0 FOLDERS="ft-c-d_t-spider-g0.9-t1e-4-2e-5-0.05-0-42 ft-c-v_geo-spider-g0.9-t1e-4-2e-5-0.05-0-42 ft-j-spider-2e-5-0.02-0-42" OUT=output_reeval_spider_A.txt bash reeval_spider_only.sh
#   GPU=1 FOLDERS="ft-reach-schedule_asym-spider-a1.0-b1.0-g0.9-t1e-4-2e-5-0.05-0-42 ft-r-spider-2e-5-42" OUT=output_reeval_spider_B.txt bash reeval_spider_only.sh
set -u

cd "$(dirname "$0")"

: "${GPU:=0}"
: "${FOLDERS:=}"
: "${OUT:=output_reeval_spider.txt}"
: "${MODEL_NAME:=meta-llama/Llama-3.2-1B-Instruct}"
: "${MNT:=128}"

PYTHON=/home/khanhnt/.conda/envs/LinOSS/bin/python

if [[ -z "$FOLDERS" ]]; then
  echo "ERROR: set FOLDERS= to a space-separated list of checkpoint directories." >&2
  exit 2
fi

: > "$OUT"
echo "# Spider re-evaluation on GPU=${GPU} at $(date)" >> "$OUT"

for folder in $FOLDERS; do
  [[ -f "${folder}/config.json" ]] || { echo "[SKIP] $folder (not a checkpoint)" | tee -a "$OUT"; continue; }

  method="unknown"
  case "$folder" in
    ft-r-*)          method="regular"                    ;;
    ft-j-*)          method="jepa"                       ;;
    ft-c-d_t-*)      method="control_JEPA norm=d_t"      ;;
    ft-c-v_geo-*)    method="control_JEPA norm=v_geo"    ;;
    ft-reach-*)      method="reach_JEPA"                 ;;
  esac

  echo "" | tee -a "$OUT"
  echo "=== [spider|mnt=${MNT}] ${method} ${folder}" | tee -a "$OUT"
  echo "Success Rate: ${method} ${MODEL_NAME} dataset=datasets/spider folder=${folder}" >> "$OUT"

  CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} evaluate.py \
    --model_name="${folder}" \
    --original_model_name="${MODEL_NAME}" \
    --input_file="datasets/spider_test.jsonl" \
    --output_file="eval_spider_${GPU}.jsonl" \
    --split_tune_untune \
    --nosplit_data \
    --spider_path=spider_data/database \
    --max_new_tokens=${MNT} 2>&1 | tee -a "$OUT"
done

echo "" | tee -a "$OUT"
echo "DONE [GPU=${GPU}]. Results in ${OUT}" | tee -a "$OUT"
