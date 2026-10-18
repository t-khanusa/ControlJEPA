#!/usr/bin/env bash
# Train + evaluate one or more methods over models x datasets x seeds.
#
#   bash scripts/run.sh                                   # ControlJEPA, Llama-3.2-1B, all benchmarks, 5 seeds
#   METHODS="regular stp controljepa" DATASETS="synth gsm8k" SEEDS=82 bash scripts/run.sh
#   MODELS="google/gemma-2-2b-it Qwen/Qwen3-1.7B" bash scripts/run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

MODELS="${MODELS:-meta-llama/Llama-3.2-1B-Instruct}"
DATASETS="${DATASETS:-synth turk gsm8k spider nq_open hellaswag}"
SEEDS="${SEEDS:-82 23 37 84 4}"
METHODS="${METHODS:-controljepa}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"

for model in ${MODELS}; do
  for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
      for method in ${METHODS}; do
        out="${CKPT_DIR}/${method}/$(echo "${model}" | tr '/' '_')/${dataset}/seed${seed}"
        if grep -qs "Success Rate" "${out}/eval_result.txt"; then echo "skip ${out}"; continue; fi
        train "${method}" "${model}" "datasets/${dataset}" "${seed}" "${out}"
        evaluate "${out}" "${model}" "datasets/${dataset}"
      done
    done
  done
done
