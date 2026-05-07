# ---------------------------------------------------------------------------
# 6-dataset × 8-method Lyapunov-Reachability bench (Phase 1 / "beat-STP").
#
# We dropped the NTP-only `regular` baseline (the research goal is to beat
# STP, not NTP) and dropped `rotten_tomatoes` (saturated, not in the STP
# paper's matrix). We added `turk`, `nq_open`, and `hellaswag` to match the
# six columns of the STP paper's Figure 1 so our numbers are directly
# comparable.
#
# Methods:
#   1. stp               : STP with random_span cosine loss (baseline to beat).
#   2. control_d_t       : Lyapunov-tube, V_t = ||e_t||^2 / ||d_t||^2   (angle + anchor_eps=1e-3).
#   3. control_v_geo     : Lyapunov-tube, V_t = ||e_t||^2 / ||v_geo||^2 (chord).
#   4. reach_jepa        : Lyapunov-Reachability with schedule_asym progress
#                          V_prog_t = max(0, tau_t - p_t/L)^2 (structural
#                          token-rank schedule, teleportation-proof).
#   5. reach_jepa_angle  : same as (4) but with angle-form transverse term
#                          V_\perp = sin^2(theta_t) (bounded in [0,1],
#                          rotation-invariant). This is the "V_t =
#                          ||e_t||^2/||d_t||^2 + beta * max(0, tau_t -
#                          p_t/||v_geo||)^2" hybrid from the research plan,
#                          with anchor positions masked out of the transverse
#                          term and the contraction transition.
# Random-span Lyapunov variants (Phase B): same Monte Carlo sampler as
# --linear=random_span (STP), but applied to the *Lyapunov loss itself*,
# sampling random sub-segments of the token trajectory per step and
# enforcing V_{t+1} <= gamma V_t + tau over them. Interpretation: an
# Alg 2-style path-integral Monte Carlo estimator of the multi-scale
# contraction functional (see e.g. Alg 2 / continuous-token hypothesis),
# token hypothesis). Uses the exact same get_s_t distribution as STP so
# we can attribute any gap to the *loss*, not the sampler.
#   6. control_d_t_rs    : (2) evaluated on random sub-segments.
#   7. control_v_geo_rs  : (3) evaluated on random sub-segments.
#   8. reach_jepa_rs     : (4) evaluated on random sub-segments.
# reach_jepa_angle is deliberately not in the random-span matrix yet
# (its stand-alone regression needs to be diagnosed first; the soft-
# anchor anchor_eps fix applies only to control_JEPA d_t in this commit).
#
# Distributed:
#   torchrun --nproc_per_node=${NPROC_PER_NODE:-2}
#   batch_size=4, grad_accum=2 -> effective batch = 2 * 4 * 2 = 16 per step,
#   matching the single-GPU baseline (4 * 4 = 16) in output.txt.
#
# Hyperparameters are held fixed across datasets so that any cross-domain
# trend is attributable to the *method*, not to per-dataset tuning:
#   learning_rate = 2e-5, epochs = 4, seed = 42 (single seed for Phase 1)
#   tube_gamma = 0.9, tube_tau = 1e-4
#   lbd_stp = 0.02 (matches the STP paper); lbd_control = lbd_reach = 0.05
#   reach_alpha = reach_beta = 1.0, progress_mode = schedule_asym
#
# Bug fixes landed alongside this matrix (verified in the audit):
#   * stp.py: `data_file.startswith("hellaswag")` used basename-aware check
#     so `datasets/hellaswag_train.jsonl` correctly picks the text/code
#     JEPA pair instead of (context, letter).
#   * evaluate.py: same basename fix for the `relative_probability` router,
#     plus hellaswag/turk eval branches and standard NQ-Open normalization
#     (lowercase, strip articles/punct, `gt in gen` direction).
# ---------------------------------------------------------------------------

set -e

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
# Only export CUDA_VISIBLE_DEVICES when the caller explicitly asked for it.
# Exporting unconditionally makes modern torchrun (>= 2.0) remap visible
# GPUs per worker (rank i sees only the i-th GPU as local index 0), which
# collides with stp.py's `set_device(local_rank)` call. stp.py already
# detects the remap, but avoiding the export entirely is simpler and lets
# users pin GPUs from outside when they want to.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

# torchrun wrapper that keeps the master_port hashed to the run folder so
# concurrent runs in the same tmux don't collide.
_launch() {
  local run_folder="$1"; shift
  local port_suffix
  port_suffix=$(printf '%s' "$run_folder" | cksum | awk '{print $1 % 1000}')
  local port=$((MASTER_PORT + port_suffix))
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${port}" "$@"
}

# Dataset-appropriate generation cap for evaluate.py. Using the default (-1 ->
# None -> HF default 120) caused unbounded rambling that broke:
#   rotten_tomatoes  (1-token GT vs 120-token output vs strict equality)
#   gsm8k            (truncation BEFORE `#### X` emitted, plus regex $ anchor)
#   spider           (SQL followed by prose, broke sqlite3 execution compare)
# We now cap per-dataset; see evaluate.py for the matching comparator fixes.
#
# Optional reference evaluation profile (fixed generation defaults):
#   EVAL_PROFILE=llm_jepa_official bash run_stp.sh
# (forces max_new_tokens=128, max_length=512 in evaluate.py; ignores _mnt caps.)
EVAL_PROFILE="${EVAL_PROFILE:-fork}"

_run_evaluate_checkpoint() {
  local model_folder="$1" dataset="$2" base_model_name="$3"
  local log_file="${4:-output_gemma.txt}"
  if [[ "${EVAL_PROFILE}" == "llm_jepa_official" ]]; then
    python evaluate.py --model_name="${model_folder}" \
      --input_file="${dataset}_test.jsonl" --output_file=eval.jsonl --split_tune_untune \
      --original_model_name="${base_model_name}" --nosplit_data \
      --spider_path=spider_data/database --eval_profile=llm_jepa_official | tee -a "${log_file}"
  else
    local mnt; mnt=$(_mnt_for_dataset "${dataset}")
    python evaluate.py --model_name="${model_folder}" \
      --input_file="${dataset}_test.jsonl" --output_file=eval.jsonl --split_tune_untune \
      --original_model_name="${base_model_name}" --nosplit_data \
      --spider_path=spider_data/database --max_new_tokens="${mnt}" | tee -a "${log_file}"
  fi
}

_mnt_for_dataset() {
  # GT-length quantiles on the actual test sets (Llama-3.2 tokenizer):
  #   rotten_tomatoes : 1 token          -> 4   is generous
  #   synth / turk    : <= 32            -> 96  is safe
  #   spider          : ~100 max         -> 128 is safe
  #   gsm8k           : q99=233, max=310 -> 256 covers 99% (384 doubles wall time)
  #   nq_open         : short answers    -> 64  plenty
  #   hellaswag       : 1 letter         -> 8   (not actually generated; scored
  #                                              by relative_probability)
  local ds_base
  ds_base=$(basename "$1")
  case "${ds_base}" in
    rotten_tomatoes)  echo 4   ;;
    synth|turk)       echo 96  ;;
    spider)           echo 128 ;;
    gsm8k)            echo 256 ;;
    hellaswag)        echo 8   ;;
    nq_open)          echo 64  ;;
    *)                echo 128 ;;
  esac
}

run_jepa() {
  base_model_name=${1}; learning_rate=${2}; epoch=${3}; last_token=${4}
  predictors=${5}; seed=${6}; lbd=${7}; dataset=${8}; model_folder=${9}

  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  _launch "${model_folder}" stp.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=random_span \
    --batch_size=4 --grad_accum=4
  _run_evaluate_checkpoint "${model_folder}" "${dataset}" "${base_model_name}" output_gemma.txt
}

run_control_jepa() {
  # Lyapunov-tube regularizer on token trajectories.
  # V_t = ||e_t||^2 / N_t where N_t is selected by --control_norm.
  # Trailing args (Phase A + B):
  #   ${13}=anchor_eps (float, default 1e-3; only used when control_norm=d_t)
  #   ${14}=random_span_lyap (0/1, default 0; if 1 passes --random_span_lyap
  #         so the Lyapunov contraction is evaluated on a random STP-style
  #         sub-segment per step instead of the full user+assistant span)
  base_model_name=${1}; learning_rate=${2}; epoch=${3}; last_token=${4}
  predictors=${5}; seed=${6}; lbd_control=${7}; dataset=${8}; model_folder=${9}
  control_norm=${10:-d_t}
  tube_gamma=${11:-0.95}
  tube_tau=${12:-1e-4}
  anchor_eps=${13:-1e-3}
  random_span_lyap=${14:-0}

  local rs_flag=""
  if [[ "${random_span_lyap}" == "1" ]]; then
    rs_flag="--random_span_lyap"
  fi

  echo "Success Rate: control_JEPA ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd_control=${lbd_control} norm=${control_norm} gamma=${tube_gamma} tau=${tube_tau} anchor_eps=${anchor_eps} rs_lyap=${random_span_lyap} dataset=${dataset}" >> output.txt
  _launch "${model_folder}" stp.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=control_JEPA \
    --lbd_control=${lbd_control} \
    --control_norm=${control_norm} \
    --tube_gamma=${tube_gamma} \
    --tube_tau=${tube_tau} \
    --anchor_eps=${anchor_eps} \
    ${rs_flag} \
    --tube_log_interval=50 \
    --batch_size=4 --grad_accum=4
  _run_evaluate_checkpoint "${model_folder}" "${dataset}" "${base_model_name}" output_gemma.txt
}

run_reach_jepa() {
  # Lyapunov-Reachability regularizer.
  # V_t = alpha * V_\perp  +  beta * V_prog, with progress_mode selecting
  # the longitudinal term and --linear selecting the V_\perp form:
  #   linear=reach_JEPA        -> V_\perp = ||e_t||^2 / L^2   (classical)
  #   linear=reach_JEPA_angle  -> V_\perp = sin^2 theta_t     (angle, bounded)
  # schedule_asym is the proposed longitudinal form; endpoint is the
  # legacy (teleportation-prone) form kept for ablations.
  # Trailing arg (Phase B):
  #   ${16}=random_span_lyap (0/1, default 0; reach_JEPA_angle is ignored
  #         since the angle form is kept step-based while the d_t
  #         regression is diagnosed).
  base_model_name=${1}; learning_rate=${2}; epoch=${3}; last_token=${4}
  predictors=${5}; seed=${6}; lbd_reach=${7}; dataset=${8}; model_folder=${9}
  alpha=${10:-1.0}
  beta=${11:-1.0}
  tube_gamma=${12:-0.95}
  tube_tau=${13:-1e-4}
  progress_mode=${14:-schedule_asym}
  linear_mode=${15:-reach_JEPA}
  random_span_lyap=${16:-0}

  local rs_flag=""
  if [[ "${random_span_lyap}" == "1" ]]; then
    rs_flag="--random_span_lyap"
  fi

  echo "Success Rate: ${linear_mode} ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd_reach=${lbd_reach} alpha=${alpha} beta=${beta} gamma=${tube_gamma} tau=${tube_tau} pm=${progress_mode} rs_lyap=${random_span_lyap} dataset=${dataset}" >> output.txt
  _launch "${model_folder}" stp.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=${linear_mode} \
    --lbd_reach=${lbd_reach} \
    --alpha=${alpha} --beta=${beta} \
    --progress_mode=${progress_mode} \
    --tube_gamma=${tube_gamma} \
    --tube_tau=${tube_tau} \
    ${rs_flag} \
    --tube_log_interval=50 \
    --batch_size=4 --grad_accum=4
  _run_evaluate_checkpoint "${model_folder}" "${dataset}" "${base_model_name}" output_gemma.txt
}

# ---------------------------------------------------------------------------
# Model dispatch: `last_token` + folder-tag per model family.
# ---------------------------------------------------------------------------
# `--last_token` picks the representation position used for the
# JEPA/Lyapunov losses. It depends on what `apply_chat_template(...,
# add_generation_prompt=False)` appends *after* the assistant content, so
# it is per-family. Values below replay the commented-out table in
# run.sh:44-61 which was validated against each tokenizer's trailer:
#   llama-3.*      : ... content <|eot_id|>                 -> -2
#   gemma-2        : ... content <end_of_turn>\n            -> -2
#   openelm (L2)   : ... content </s> \n                    -> -4
#   olmo-2         : ... content <|endoftext|>             -> -1
#   qwen3          : ... content <|im_end|>\n               -> -3
#   deepseek r1-d  : ... content <|end_of_sentence|>       -> -1
#
# `_model_tag` is used as the *directory prefix* for the fine-tuned
# checkpoint, so that runs from different model families do not collide
# on disk (all the run_* helpers just concatenate `$model_folder`). For
# meta-llama/Llama-3.2-1B-Instruct we keep the historical tag
# "Llama3.2-1B-Instruct" to preserve existing `logs/` layout and let
# `skip_if_model_exists` short-circuit completed runs.
_last_token_for_model() {
  case "$1" in
    google/gemma*)              echo -2 ;;
    apple/OpenELM*)             echo -4 ;;
    allenai/OLMo*)              echo -1 ;;
    Qwen/*)                     echo -3 ;;
    deepseek-ai/DeepSeek*)      echo -1 ;;
    meta-llama/Llama*)          echo -2 ;;
    *)                          echo -2 ;;
  esac
}

_model_tag() {
  case "$1" in
    meta-llama/Llama-3.2-1B-Instruct) echo "Llama3.2-1B-Instruct" ;;
    *)                                basename "$1" ;;
  esac
}

# ---------------------------------------------------------------------------
# Driver. Shared hyperparameters.
# ---------------------------------------------------------------------------
# Set of models to sweep. Override with e.g.
#   MODELS="meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3-1.7B" bash run_stp.sh
# The six families validated by _lora_target_modules in stp.py are:
#   meta-llama/Llama-3.2-1B-Instruct
#   google/gemma-2-2b-it
#   apple/OpenELM-1_1B-Instruct
#   Qwen/Qwen3-1.7B
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   allenai/OLMo-2-0425-1B-Instruct
MODELS="${MODELS:-meta-llama/Llama-3.2-1B-Instruct}" #Qwen/Qwen3-1.7B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B allenai/OLMo-2-0425-1B-Instruct}"
predictors=0
epoch=4
learning_rate=2e-5
seed=42

lbd_stp=0.02
# Override any of these from the shell for hyperparameter sweeps, e.g.:
#   for lbd in $(seq 0.01 0.01 0.2); do
#     for g in $(seq 0.8 0.01 0.98); do
#       for t in 1e-3 1e-4 1e-5 0; do
#         LBD_CONTROL=$lbd TUBE_GAMMA=$g TUBE_TAU=$t METHODS=control_v_geo SEEDS=42 bash run_stp.sh
#       done
#     done0.9-t1e-4-2e-5-0.01
#   done
lbd_control="${LBD_CONTROL:-0.05}"
lbd_reach="${LBD_REACH:-0.05}"
tube_gamma="${TUBE_GAMMA:-0.9}"
tube_tau="${TUBE_TAU:-1e-4}"
reach_alpha=1.0
reach_beta=1.0
progress_mode=schedule_asym
# Phase A: scale-invariant soft anchor for control_norm=d_t. See
# lyapunov_loss.py::LyapunovControlLoss docstring for the derivation.
# 1e-3 is the default across the matrix; overridable via env.
anchor_eps="${ANCHOR_EPS:-1e-5}"

# Datasets (short -> long token budgets). Paths are passed *without* suffix;
# run_* functions append _train.jsonl / _test.jsonl.
#   synth     : n=8000, u+a median 28 tokens (synthetic NL -> regex)
#   turk      : n=8000, u+a similar (Turker paraphrases of the same regexes)
#   spider    : n=6587, u+a median 46 tokens (text-to-SQL)
#   gsm8k     : n=7473, u+a median 154 tokens (multi-step math)
#   nq_open   : n=8000, short QA; answers 1-5 tokens
#   hellaswag : n=8042, 4-way MC commonsense; scored by relative_probability
datasets=(datasets/synth) # datasets/nq_open datasets/spider datasets/gsm8k datasets/hellaswag datasets/turk )

# Allow overriding which methods to run via the METHODS env var. The
# default below matches the subset of case-branches that are *currently
# active* (uncommented) in the dispatcher just below. If you comment a
# case branch out for a focused session, also drop its name from this
# default, OR rely on the warn-and-skip behaviour in the `*)` fallback
# so no single method crashes the whole matrix.
#
# Full method vocabulary (all 8 Phase 1 + Phase B variants):
#   stp  control_d_t  control_v_geo  reach_jepa  reach_jepa_angle
#   control_d_t_rs  control_v_geo_rs  reach_jepa_rs
#
# Example overrides:
#   METHODS="stp control_d_t_rs"          bash run_stp.sh
#   METHODS="reach_jepa reach_jepa_rs"    bash run_stp.shs
METHODS="${METHODS:-stp control_v_geo}"

# Seed loop (single seed for Phase 1; extendable by `SEEDS="82 23 37 84 4"` env).
SEEDS="${SEEDS:-82}" #23 37 84 4

mkdir -p logs

banner() {
  local msg="$1"
  echo ""
  echo "============================================================"
  echo "$msg"
  echo "============================================================"
  echo ""
}

skip_if_model_exists() {
  local method="$1"
  local ds_tag="$2"
  local seed="$3"
  local model_folder="$4"
  if [[ -e "${model_folder}" ]]; then
    banner "[${ds_tag}|s=${seed}] method=${method} SKIP (model_folder exists) -> ${model_folder}"
    echo "Skipped: ${method} on ${ds_tag}|s=${seed} (model_folder exists: ${model_folder})" >> output.txt
    return 0
  fi
  return 1
}

for model_name in $MODELS; do
model_tag=$(_model_tag "$model_name")
lt=$(_last_token_for_model "$model_name")
banner "MODEL: ${model_name}  (tag=${model_tag}, last_token=${lt})"

for seed in $SEEDS; do
for dataset in "${datasets[@]}"; do
  ds_tag=$(basename "$dataset")

  for method in $METHODS; do
    case "$method" in
      # stp)
      #   model_folder=${model_tag}/ft-j-${ds_tag}-${learning_rate}-${lbd_stp}-${predictors}-${seed}
      #   # skip_if_model_exists "stp" "${ds_tag}" "${seed}" "${model_folder}" && continue
      #   banner "[${model_tag}|${ds_tag}|s=${seed}] method=stp (random_span, lbd=${lbd_stp}) -> ${model_folder}"
      #   run_jepa ${model_name} ${learning_rate} ${epoch} ${lt} ${predictors} ${seed} ${lbd_stp} ${dataset} ${model_folder}
      #   ;;
      
      # control_v_geo_rs)
      #   model_folder=${model_tag}/ft-crs-v_geo-${ds_tag}-g${tube_gamma}-t${tube_tau}-${learning_rate}-${lbd_control}-${predictors}-${seed}
      #   skip_if_model_exists "control_v_geo_rs" "${ds_tag}" "${seed}" "${model_folder}" && continue
      #   banner "[${model_tag}|${ds_tag}|s=${seed}] method=control_JEPA/v_geo +random_span_lyap (chord) -> ${model_folder}"
      #   run_control_jepa ${model_name} ${learning_rate} ${epoch} ${lt} ${predictors} ${seed} ${lbd_control} ${dataset} ${model_folder} v_geo ${tube_gamma} ${tube_tau} ${anchor_eps} 1
      #   ;;

      # reach_jepa_rs)
      #   model_folder=${model_tag}/ft-reachrs-${progress_mode}-${ds_tag}-a${reach_alpha}-b${reach_beta}-g${tube_gamma}-t${tube_tau}-${learning_rate}-${lbd_reach}-${predictors}-${seed}
      #   skip_if_model_exists "reach_jepa_rs" "${ds_tag}" "${seed}" "${model_folder}" && continue
      #   banner "[${model_tag}|${ds_tag}|s=${seed}] method=reach_JEPA +random_span_lyap (${progress_mode}, a=${reach_alpha}, b=${reach_beta}) -> ${model_folder}"
      #   run_reach_jepa ${model_name} ${learning_rate} ${epoch} ${lt} ${predictors} ${seed} ${lbd_reach} ${dataset} ${model_folder} ${reach_alpha} ${reach_beta} ${tube_gamma} ${tube_tau} ${progress_mode} reach_JEPA 1
      #   ;;
      # # control_d_t)
      # #   # Step-wise Lyapunov-tube, angle form with Phase A soft anchor
      # #   # (anchor_eps * L^2 + ||d_t||^2) replacing the legacy 1e-8 clamp.
      # #   model_folder=${model_tag}/ft-c-d_t-${ds_tag}-g${tube_gamma}-t${tube_tau}-ae${anchor_eps}-${learning_rate}-${lbd_control}-${predictors}-${seed}
      # #   banner "[${model_tag}|${ds_tag}|s=${seed}] method=control_JEPA/d_t (angle, anchor_eps=${anchor_eps}) -> ${model_folder}"
      # #   run_control_jepa ${model_name} ${learning_rate} ${epoch} ${lt} ${predictors} ${seed} ${lbd_control} ${dataset} ${model_folder} d_t ${tube_gamma} ${tube_tau} ${anchor_eps} 0
      # #   ;;

      control_v_geo)
        model_folder=${model_tag}/ft-c-v_geo-${ds_tag}-g${tube_gamma}-t${tube_tau}-${learning_rate}-${lbd_control}-${predictors}-${seed}
        # skip_if_model_exists "control_v_geo" "${ds_tag}" "${seed}" "${model_folder}" && continue
        banner "[${model_tag}|${ds_tag}|s=${seed}] method=control_JEPA/v_geo (chord) -> ${model_folder}"
        run_control_jepa ${model_name} ${learning_rate} ${epoch} ${lt} ${predictors} ${seed} ${lbd_control} ${dataset} ${model_folder} v_geo ${tube_gamma} ${tube_tau} ${anchor_eps} 0
        ;;
      # reach_jepa)
      #   model_folder=${model_tag}/ft-reach-${progress_mode}-${ds_tag}-a${reach_alpha}-b${reach_beta}-g${tube_gamma}-t${tube_tau}-${learning_rate}-${lbd_reach}-${predictors}-${seed}
      #   skip_if_model_exists "reach_jepa" "${ds_tag}" "${seed}" "${model_folder}" && continue
      #   banner "[${model_tag}|${ds_tag}|s=${seed}] method=reach_JEPA (${progress_mode}, a=${reach_alpha}, b=${reach_beta}) -> ${model_folder}"
      #   run_reach_jepa ${model_name} ${learning_rate} ${epoch} ${lt} ${predictors} ${seed} ${lbd_reach} ${dataset} ${model_folder} ${reach_alpha} ${reach_beta} ${tube_gamma} ${tube_tau} ${progress_mode} reach_JEPA 0
      #   ;;

      # reach_jepa_angle)
      #   # Angle-form transverse term: V_\perp = sin^2 theta_t (bounded in
      #   # [0,1], rotation-invariant), anchor positions masked. Combined
      #   # with the schedule_asym progress term this is the hybrid
      #   # "V_t = ||e_t||^2 / ||d_t||^2 + beta * max(0, tau_t - p_t/L)^2".
      #   model_folder=${model_tag}/ft-reachA-${progress_mode}-${ds_tag}-a${reach_alpha}-b${reach_beta}-g${tube_gamma}-t${tube_tau}-${learning_rate}-${lbd_reach}-${predictors}-${seed}
      #   skip_if_model_exists "reach_jepa_angle" "${ds_tag}" "${seed}" "${model_folder}" && continue
      #   banner "[${model_tag}|${ds_tag}|s=${seed}] method=reach_JEPA_angle (${progress_mode}, a=${reach_alpha}, b=${reach_beta}) -> ${model_folder}"
      #   run_reach_jepa ${model_name} ${learning_rate} ${epoch} ${lt} ${predictors} ${seed} ${lbd_reach} ${dataset} ${model_folder} ${reach_alpha} ${reach_beta} ${tube_gamma} ${tube_tau} ${progress_mode} reach_JEPA_angle 0
      #   ;;


      *)
        # Warn and skip instead of `exit 2`: if a method is commented out
        # above or misspelled in METHODS, we do NOT want the entire bench
        # (which can be 40+ runs) to die on the second cell. The warning
        # is loud enough to notice in `output.txt` and on stderr.
        echo "WARN: skipping unknown/disabled method '${method}'. " \
             "Known names: stp control_d_t control_v_geo reach_jepa " \
             "reach_jepa_angle control_d_t_rs control_v_geo_rs reach_jepa_rs. " \
             "Check that the matching case-branch is uncommented below." >&2
        echo "Skipped: ${method} on ${ds_tag}|s=${seed} (case branch missing)" >> output.txt
        continue
        ;;
    esac
  done
done
done
done

banner "ALL RUNS COMPLETE -- see output.txt for accuracies"
