#!/usr/bin/env bash
# Stability-parameter sweep for ControlJEPA (Fig. 4b / Sec. 5.4) and the ablations of Sec. 6.
#
#   bash scripts/sweep.sh                                   # lambda x gamma x tau grid on NL-RX-SYNTH
#   SWEEP=norm  bash scripts/sweep.sh                       # normalisation ablation (Sec. 6.2)
#   SWEEP=data  bash scripts/sweep.sh                       # data efficiency (Sec. 5.3)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
DATASET="${DATASET:-synth}"
SEED="${SEED:-82}"
SWEEP="${SWEEP:-grid}"
CKPT_DIR="${CKPT_DIR:-checkpoints/sweep}"

run() {  # run <tag> <method> <train_prefix> [extra stp.py args...]
  local tag=$1 method=$2 prefix=$3; shift 3
  local out="${CKPT_DIR}/${DATASET}/${tag}-s${SEED}"
  if grep -qs "Success Rate" "${out}/eval_result.txt"; then echo "skip ${out}"; return; fi
  train "${method}" "${MODEL}" "${prefix}" "${SEED}" "${out}" "$@"
  evaluate "${out}" "${MODEL}" "datasets/${DATASET}"
}

case "${SWEEP}" in
  grid)
    for lbd in 0.01 0.02 0.05 0.08; do
      for gamma in 0.85 0.9 0.95 0.98; do
        for tau in 0 1e-5 1e-4 1e-3; do
          LAMBDA=${lbd} GAMMA=${gamma} TAU=${tau} \
            run "lbd${lbd}-g${gamma}-t${tau}" controljepa "datasets/${DATASET}"
        done
      done
    done ;;
  norm)
    for norm in v_geo d_model d_t; do
      CONTROL_NORM=${norm} run "norm-${norm}" controljepa "datasets/${DATASET}"
    done
    GAMMA=0.8 run "gamma0.8" controljepa "datasets/${DATASET}" ;;
  data)
    # Subsampled training sets: datasets/synth_train_frac1_{2,4,8,16,32}.jsonl
    for frac in 2 4 8 16 32; do
      for method in regular stp controljepa; do
        TRAIN_FILE="datasets/${DATASET}_train_frac1_${frac}.jsonl" \
          run "${method}-frac1_${frac}" "${method}" "datasets/${DATASET}"
      done
    done ;;
  *)
    echo "unknown SWEEP=${SWEEP}" >&2; exit 1 ;;
esac
