#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Data Efficiency experiment for control_v_geo with Llama-3.2-1B-Instruct
# on NL-RX-SYNTH, following the same data-fraction sweep protocol.
#
# Protocol (from paper):
#   - Randomly subsample 1/2, 1/4, 1/8, 1/16, 1/32 of synth_train.jsonl
#   - Compensate reduced steps: 1/n data -> n× epochs
#   - Also test "half compute" (n/2× epochs) with 2× learning rate
#   - Both full and half compute also tested with 2× lambda
#   - 5 seeds: 82, 23, 37, 84, 4
#
# Methods run at each data fraction:
#   1. control_v_geo (NTP + L_control): the method under study
#   2. regular NTP: baseline for comparison
#   3. STP (NTP + L_STP): STP baseline for comparison
#
# Hyperparameters (inherited from existing full-data control_v_geo runs):
#   control_v_geo: gamma=0.80, tau=1e-3, lbd_control=0.01, anchor_eps=1e-5
#   STP:           lbd_stp=0.02
#   Shared:        lr=2e-5, base_epoch=4, last_token=-2, predictors=0
#
# The full-data (fraction=1) runs already exist in Llama3.2-1B-Instruct/.
# This script only creates the subsampled-data runs.
#
# Usage:
#   bash run_data_efficiency.sh                    # run everything
#   FRACTIONS="1_2 1_4" bash run_data_efficiency.sh   # only 1/2 and 1/4
#   METHODS="control_v_geo" bash run_data_efficiency.sh  # only control_v_geo
#   SEEDS="82 23" bash run_data_efficiency.sh         # only 2 seeds
#   NPROC_PER_NODE=4 bash run_data_efficiency.sh      # use 4 GPUs
#   EVAL_PROFILE=llm_jepa_official bash run_data_efficiency.sh  # reference eval defaults (128/512)
# ---------------------------------------------------------------------------

set -e

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

# ---------------------------------------------------------------------------
# Helpers (mirrored from run_stp.sh)
# ---------------------------------------------------------------------------
_launch() {
  local run_folder="$1"; shift
  local port_suffix
  port_suffix=$(printf '%s' "$run_folder" | cksum | awk '{print $1 % 1000}')
  local port=$((MASTER_PORT + port_suffix))
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${port}" "$@"
}

banner() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
  echo ""
}

skip_if_model_exists() {
  local tag="$1" model_folder="$2"
  if [[ -e "${model_folder}" ]]; then
    banner "SKIP (exists) ${tag} -> ${model_folder}"
    echo "Skipped: ${tag} (model_folder exists: ${model_folder})" >> output_data_efficiency.txt
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------
base_model_name="meta-llama/Llama-3.2-1B-Instruct"
model_tag="Llama3.2-1B-Instruct"
lt=-2                   # last_token for Llama-3
predictors=0
base_epoch=4
base_lr=2e-5
double_lr=4e-5

# control_v_geo hyperparameters (from existing full-data runs)
lbd_control=0.01
lbd_control_2x=0.02
tube_gamma=0.80
tube_tau=1e-3
anchor_eps=1e-5
control_norm=v_geo

# STP hyperparameters
lbd_stp=0.02
lbd_stp_2x=0.04

# eval: fork uses a synth-safe cap; llm_jepa_official matches upstream evaluate.py (128/512).
max_new_tokens=96       # synth dataset (used when EVAL_PROFILE != llm_jepa_official)
EVAL_PROFILE="${EVAL_PROFILE:-fork}"

_run_evaluate_data_efficiency() {
  local model_folder="$1"
  if [[ "${EVAL_PROFILE}" == "llm_jepa_official" ]]; then
    python evaluate.py --model_name="${model_folder}" \
      --input_file="${test_file}" --output_file=eval.jsonl --split_tune_untune \
      --original_model_name="${base_model_name}" --nosplit_data \
      --spider_path=spider_data/database --eval_profile=llm_jepa_official \
      | tee -a output_data_efficiency.txt
  else
    python evaluate.py --model_name="${model_folder}" \
      --input_file="${test_file}" --output_file=eval.jsonl --split_tune_untune \
      --original_model_name="${base_model_name}" --nosplit_data \
      --spider_path=spider_data/database --max_new_tokens="${max_new_tokens}" \
      | tee -a output_data_efficiency.txt
  fi
}

# ---------------------------------------------------------------------------
# Dataset subsampling
# ---------------------------------------------------------------------------
dataset_base="datasets/synth"
train_file="${dataset_base}_train.jsonl"
test_file="${dataset_base}_test.jsonl"

# Create subsampled datasets if they don't exist yet.
# subsample_dataset.py uses seeded nested sampling so 1/4 is a strict
# subset of 1/2, etc. -- exactly what the paper prescribes.
if [[ ! -f "datasets/synth_train_frac1_2.jsonl" ]]; then
  banner "Creating subsampled training datasets"
  python subsample_dataset.py \
    --input_file "${train_file}" \
    --output_dir datasets \
    --fractions 0.5 0.25 0.125 0.0625 0.03125 \
    --seed 42
fi

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
# fraction_tag -> (float fraction, epoch_1x, epoch_half, train_file_suffix)
# epoch_1x   = base_epoch * (1/frac)     (full compute compensation)
# epoch_half = base_epoch * (1/frac) / 2 (half compute)
declare -A FRAC_EPOCH_1X FRAC_EPOCH_HALF FRAC_TRAIN
FRAC_EPOCH_1X[1_2]=8;     FRAC_EPOCH_HALF[1_2]=4;     FRAC_TRAIN[1_2]="datasets/synth_train_frac1_2.jsonl"
FRAC_EPOCH_1X[1_4]=16;    FRAC_EPOCH_HALF[1_4]=8;     FRAC_TRAIN[1_4]="datasets/synth_train_frac1_4.jsonl"
FRAC_EPOCH_1X[1_8]=32;    FRAC_EPOCH_HALF[1_8]=16;    FRAC_TRAIN[1_8]="datasets/synth_train_frac1_8.jsonl"
FRAC_EPOCH_1X[1_16]=64;   FRAC_EPOCH_HALF[1_16]=32;   FRAC_TRAIN[1_16]="datasets/synth_train_frac1_16.jsonl"
FRAC_EPOCH_1X[1_32]=128;  FRAC_EPOCH_HALF[1_32]=64;   FRAC_TRAIN[1_32]="datasets/synth_train_frac1_32.jsonl"

FRACTIONS="${FRACTIONS:-1_2 1_4 1_8 1_16 1_32}"
METHODS="${METHODS:-control_v_geo regular stp}"
SEEDS="${SEEDS:-82 23 37 84 4}"

mkdir -p logs

# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------
run_control_v_geo() {
  local lr=$1 epoch=$2 seed=$3 train_data=$4 model_folder=$5 lbd=$6
  local tag="control_v_geo|lr=${lr}|e=${epoch}|lbd=${lbd}|s=${seed}"

  echo "Success Rate: control_JEPA ${base_model_name} lr=${lr} e=${epoch} lt=${lt} p=${predictors} s=${seed} lbd_control=${lbd} norm=${control_norm} gamma=${tube_gamma} tau=${tube_tau} anchor_eps=${anchor_eps} rs_lyap=0 train=${train_data}" >> output_data_efficiency.txt
  _launch "${model_folder}" stp.py \
    --train_file "${train_data}" \
    --output_dir="${model_folder}" --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${lt} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${lr} \
    --linear=control_JEPA \
    --lbd_control=${lbd} \
    --control_norm=${control_norm} \
    --tube_gamma=${tube_gamma} \
    --tube_tau=${tube_tau} \
    --anchor_eps=${anchor_eps} \
    --tube_log_interval=50 \
    --batch_size=4 --grad_accum=4
  _run_evaluate_data_efficiency "${model_folder}"
}

run_regular() {
  local lr=$1 epoch=$2 seed=$3 train_data=$4 model_folder=$5

  echo "Success Rate: regular ${base_model_name} lr=${lr} e=${epoch} lt=${lt} p=${predictors} s=${seed} train=${train_data}" >> output_data_efficiency.txt
  _launch "${model_folder}" stp.py \
    --train_file "${train_data}" \
    --output_dir="${model_folder}" --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${lt} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${lr} \
    --regular \
    --batch_size=4 --grad_accum=4
  _run_evaluate_data_efficiency "${model_folder}"
}

run_stp() {
  local lr=$1 epoch=$2 seed=$3 train_data=$4 model_folder=$5 lbd=$6

  echo "Success Rate: stp ${base_model_name} lr=${lr} e=${epoch} lt=${lt} p=${predictors} s=${seed} lbd=${lbd} train=${train_data}" >> output_data_efficiency.txt
  _launch "${model_folder}" stp.py \
    --train_file "${train_data}" \
    --output_dir="${model_folder}" --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${lt} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${lr} \
    --linear=random_span \
    --batch_size=4 --grad_accum=4
  _run_evaluate_data_efficiency "${model_folder}"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
banner "DATA EFFICIENCY: control_v_geo vs NTP vs STP on SYNTH (Llama-3.2-1B)"

for seed in $SEEDS; do
for frac in $FRACTIONS; do
  epoch_1x=${FRAC_EPOCH_1X[$frac]}
  epoch_half=${FRAC_EPOCH_HALF[$frac]}
  train_data=${FRAC_TRAIN[$frac]}

  if [[ ! -f "${train_data}" ]]; then
    echo "ERROR: ${train_data} not found. Run subsample_dataset.py first." >&2
    exit 1
  fi

  for method in $METHODS; do
    case "$method" in

      # ===================================================================
      # 1. control_v_geo: full compute (n× epochs, base lr, base lambda)
      # ===================================================================
      control_v_geo)
        # --- 1a: 1Flop (full compute, base lambda) ---
        mf="${model_tag}/de-cv-${frac}-1F-g${tube_gamma}-t${tube_tau}-${base_lr}-${lbd_control}-${predictors}-${seed}"
        skip_if_model_exists "cv|${frac}|1F|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] control_v_geo 1Flop (e=${epoch_1x}, lr=${base_lr}, lbd=${lbd_control})"
          run_control_v_geo ${base_lr} ${epoch_1x} ${seed} "${train_data}" "${mf}" ${lbd_control}
        }

        # --- 1b: 1Flop, 2×lambda ---
        mf="${model_tag}/de-cv-${frac}-1F2L-g${tube_gamma}-t${tube_tau}-${base_lr}-${lbd_control_2x}-${predictors}-${seed}"
        skip_if_model_exists "cv|${frac}|1F2L|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] control_v_geo 1Flop,2λ (e=${epoch_1x}, lr=${base_lr}, lbd=${lbd_control_2x})"
          run_control_v_geo ${base_lr} ${epoch_1x} ${seed} "${train_data}" "${mf}" ${lbd_control_2x}
        }

        # --- 1c: 0.5Flop, 2×lr (half compute, double learning rate) ---
        mf="${model_tag}/de-cv-${frac}-hF2lr-g${tube_gamma}-t${tube_tau}-${double_lr}-${lbd_control}-${predictors}-${seed}"
        skip_if_model_exists "cv|${frac}|hF2lr|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] control_v_geo 0.5Flop,2lr (e=${epoch_half}, lr=${double_lr}, lbd=${lbd_control})"
          run_control_v_geo ${double_lr} ${epoch_half} ${seed} "${train_data}" "${mf}" ${lbd_control}
        }

        # --- 1d: 0.5Flop, 2×lr, 2×lambda ---
        mf="${model_tag}/de-cv-${frac}-hF2lr2L-g${tube_gamma}-t${tube_tau}-${double_lr}-${lbd_control_2x}-${predictors}-${seed}"
        skip_if_model_exists "cv|${frac}|hF2lr2L|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] control_v_geo 0.5Flop,2lr,2λ (e=${epoch_half}, lr=${double_lr}, lbd=${lbd_control_2x})"
          run_control_v_geo ${double_lr} ${epoch_half} ${seed} "${train_data}" "${mf}" ${lbd_control_2x}
        }
        ;;

      # ===================================================================
      # 2. Regular NTP baseline
      # ===================================================================
      regular)
        # --- 2a: 1Flop ---
        mf="${model_tag}/de-ntp-${frac}-1F-${base_lr}-${predictors}-${seed}"
        skip_if_model_exists "ntp|${frac}|1F|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] NTP 1Flop (e=${epoch_1x}, lr=${base_lr})"
          run_regular ${base_lr} ${epoch_1x} ${seed} "${train_data}" "${mf}"
        }

        # --- 2b: 0.5Flop, 2×lr ---
        mf="${model_tag}/de-ntp-${frac}-hF2lr-${double_lr}-${predictors}-${seed}"
        skip_if_model_exists "ntp|${frac}|hF2lr|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] NTP 0.5Flop,2lr (e=${epoch_half}, lr=${double_lr})"
          run_regular ${double_lr} ${epoch_half} ${seed} "${train_data}" "${mf}"
        }
        ;;

      # ===================================================================
      # 3. STP baseline (random_span cosine)
      # ===================================================================
      stp)
        # --- 3a: 1Flop ---
        mf="${model_tag}/de-stp-${frac}-1F-${base_lr}-${lbd_stp}-${predictors}-${seed}"
        skip_if_model_exists "stp|${frac}|1F|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] STP 1Flop (e=${epoch_1x}, lr=${base_lr}, lbd=${lbd_stp})"
          run_stp ${base_lr} ${epoch_1x} ${seed} "${train_data}" "${mf}" ${lbd_stp}
        }

        # --- 3b: 1Flop, 2×lambda ---
        mf="${model_tag}/de-stp-${frac}-1F2L-${base_lr}-${lbd_stp_2x}-${predictors}-${seed}"
        skip_if_model_exists "stp|${frac}|1F2L|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] STP 1Flop,2λ (e=${epoch_1x}, lr=${base_lr}, lbd=${lbd_stp_2x})"
          run_stp ${base_lr} ${epoch_1x} ${seed} "${train_data}" "${mf}" ${lbd_stp_2x}
        }

        # --- 3c: 0.5Flop, 2×lr ---
        mf="${model_tag}/de-stp-${frac}-hF2lr-${double_lr}-${lbd_stp}-${predictors}-${seed}"
        skip_if_model_exists "stp|${frac}|hF2lr|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] STP 0.5Flop,2lr (e=${epoch_half}, lr=${double_lr}, lbd=${lbd_stp})"
          run_stp ${double_lr} ${epoch_half} ${seed} "${train_data}" "${mf}" ${lbd_stp}
        }

        # --- 3d: 0.5Flop, 2×lr, 2×lambda ---
        mf="${model_tag}/de-stp-${frac}-hF2lr2L-${double_lr}-${lbd_stp_2x}-${predictors}-${seed}"
        skip_if_model_exists "stp|${frac}|hF2lr2L|s=${seed}" "${mf}" || {
          banner "[${frac}|s=${seed}] STP 0.5Flop,2lr,2λ (e=${epoch_half}, lr=${double_lr}, lbd=${lbd_stp_2x})"
          run_stp ${double_lr} ${epoch_half} ${seed} "${train_data}" "${mf}" ${lbd_stp_2x}
        }
        ;;

      *)
        echo "WARN: unknown method '${method}', skipping." >&2
        ;;
    esac
  done

done
done

banner "DATA EFFICIENCY RUNS COMPLETE -- see output_data_efficiency.txt"
