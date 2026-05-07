#!/usr/bin/env bash
# Minimal example: control-JEPA Semantic Tube Lyapunov tube + polymorphism auxiliary.
#
# Run only after activating the project environment:
#
#     conda activate controlJEPA
#
# Or delegate to conda:
#
#     bash scripts/run_under_control_jepa.sh bash scripts/example_preserving_diversity_training.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
TRAIN="${TRAIN:-$ROOT/datasets/synth_train.jsonl}"
OUT="${OUT:-$ROOT/outputs/control_jepa_preserving_diversity_demo}"

PYTHON="${PYTHON:-python}"

exec "$PYTHON" stp.py \
  --train_file "$TRAIN" \
  --output_dir "$OUT" \
  --model_name "$MODEL" \
  --dynamics_tube \
  --preserve_diversity \
  --lambda_diversity "${LAMBD_DIV:-0.01}" \
  --diversity_proj_dim "${DIV_PROJ:-64}" \
  --tube_gamma "${TUBE_GAMMA:-0.9}" \
  --tube_tau "${TUBE_TAU:-1e-4}" \
  --tube_log_interval "${TUBE_LOG:-50}" \
  --learning_rate "${LR:-2e-5}" \
  --batch_size "${BS:-1}" \
  --grad_accum "${GA:-8}" \
  --max_length "${MAXL:-512}" \
  --num_epochs "${EP:-1}" \
  "$@"
