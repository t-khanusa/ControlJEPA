#!/usr/bin/env bash
set -e
# Launch a single reach_JEPA experiment (alpha=1, beta=1)
# to validate the pipeline end-to-end before running the full sweep.
# Assumes your environment already has the dependencies installed.

# Shared configs (match compare_three_runs_dt / run1 so we can compare heads-up)
model_name=meta-llama/Llama-3.2-1B-Instruct
dataset=datasets/synth
learning_rate=2e-5
seed=42
epoch=4
last_token=-2
predictors=0

# reach_JEPA specifics
lbd_reach=0.05
alpha=1.0
beta=1.0
reach_gamma=0.9
reach_tau=1e-4

model_folder=run2/ft-reach-a${alpha}-b${beta}-g${reach_gamma}-t${reach_tau}-${learning_rate}-${lbd_reach}-${predictors}-${seed}

echo "==== reach_JEPA: alpha=${alpha} beta=${beta} lbd=${lbd_reach} -> ${model_folder}"

python stp.py \
    --train_file ${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --predictors=${predictors} \
    --model_name=${model_name} --learning_rate=${learning_rate} \
    --linear=reach_JEPA \
    --lbd_reach=${lbd_reach} \
    --alpha=${alpha} \
    --beta=${beta} \
    --tube_gamma=${reach_gamma} \
    --tube_tau=${reach_tau} \
    --tube_log_interval=50

python evaluate.py --model_name=${model_folder} \
    --input_file=${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 | tee -a output.txt
