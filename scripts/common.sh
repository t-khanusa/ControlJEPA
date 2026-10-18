#!/usr/bin/env bash
# Shared helpers for the training / evaluation scripts. Source from the repository root.

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
RESULTS="${RESULTS:-results.txt}"

# Index of the token whose hidden state is used as the sequence embedding (STP / LLM-JEPA).
last_token_for() {
  case "$1" in
    apple/OpenELM*)        echo -4 ;;
    Qwen/Qwen*)            echo -3 ;;
    allenai/OLMo-2*)       echo -1 ;;
    deepseek-ai/DeepSeek*) echo -1 ;;
    *)                     echo -2 ;;  # Llama-3, Gemma-2
  esac
}

# Generation budget per benchmark.
max_new_tokens_for() {
  case "$(basename "$1")" in
    hellaswag) echo 8   ;;
    nq_open)   echo 64  ;;
    synth|turk) echo 96 ;;
    spider)    echo 128 ;;
    gsm8k)     echo 256 ;;
    *)         echo 128 ;;
  esac
}

launch() {
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" "$@"
}

# evaluate <checkpoint_dir> <base_model> <dataset_prefix>   e.g. datasets/synth
evaluate() {
  local ckpt=$1 model=$2 dataset=$3
  python evaluate.py --model_name="${ckpt}" --original_model_name="${model}" \
    --input_file="${dataset}_test.jsonl" --output_file="${ckpt}/eval.jsonl" \
    --split_tune_untune --nosplit_data --spider_path=spider_data/database \
    --max_new_tokens="$(max_new_tokens_for "${dataset}")" | tee "${ckpt}/eval_result.txt"
  grep "Success Rate" "${ckpt}/eval_result.txt" >> "${RESULTS}"
}

# train <method> <base_model> <dataset_prefix> <seed> <output_dir> [extra args...]
#   (set TRAIN_FILE to override <dataset_prefix>_train.jsonl)
#   method in {regular, llm_jepa, stp, controljepa}
#   experimental (not in the paper): {reach_jepa, reach_jepa_angle, reach_jepa_rs}
#
# controljepa also passes the stop-decision supervision (--supervise_eos --eos_ce_weight).
# stp.py keeps it only for the datasets in --eos_cure_datasets (synth, turk) and drops it
# elsewhere; set EOS_CURE=off to disable it everywhere or EOS_CURE=on to force it.
train() {
  local method=$1 model=$2 dataset=$3 seed=$4 out=$5; shift 5
  local common=(--train_file="${TRAIN_FILE:-${dataset}_train.jsonl}" --output_dir="${out}" --model_name="${model}"
                --finetune_seed="${seed}" --num_epochs="${EPOCHS:-4}" --learning_rate="${LR:-2e-5}"
                --batch_size="${BATCH_SIZE:-2}" --grad_accum="${GRAD_ACCUM:-4}")
  local lt; lt=$(last_token_for "${model}")
  case "${method}" in
    regular)
      launch stp.py "${common[@]}" --regular "$@" ;;
    llm_jepa)
      launch finetune.py "${common[@]}" --last_token="${lt}" \
        --lbd="${LLM_JEPA_LBD:-1.0}" --predictors="${LLM_JEPA_PREDICTORS:-1}" "$@" ;;
    stp)
      launch stp.py "${common[@]}" --linear=random_span --last_token="${lt}" \
        --lbd="${STP_LBD:-0.02}" --predictors=0 "$@" ;;
    controljepa)
      launch stp.py "${common[@]}" --linear=control_JEPA --last_token="${lt}" \
        --lbd="${LAMBDA:-0.05}" --tube_gamma="${GAMMA:-0.95}" --tube_tau="${TAU:-1e-4}" \
        --control_norm="${CONTROL_NORM:-v_geo}" --predictors=0 \
        --supervise_eos --eos_ce_weight="${EOS_CE_WEIGHT:-2}" --eos_cure="${EOS_CURE:-auto}" "$@" ;;
    reach_jepa|reach_jepa_angle|reach_jepa_rs)
      local mode=reach_JEPA rs=()
      [[ "${method}" == reach_jepa_angle ]] && mode=reach_JEPA_angle
      [[ "${method}" == reach_jepa_rs ]] && rs=(--random_span_lyap)
      launch stp.py "${common[@]}" --linear="${mode}" --last_token="${lt}" \
        --lbd_reach="${LAMBDA_REACH:-0.05}" --alpha="${REACH_ALPHA:-1.0}" --beta="${REACH_BETA:-1.0}" \
        --progress_mode="${PROGRESS_MODE:-schedule_asym}" --tube_gamma="${GAMMA:-0.95}" \
        --tube_tau="${TAU:-1e-4}" --predictors=0 "${rs[@]}" "$@" ;;
    *)
      echo "unknown method: ${method}" >&2; return 1 ;;
  esac
}
