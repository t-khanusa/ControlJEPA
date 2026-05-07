#!/usr/bin/env bash
# Reproduce suffix-bin accuracy for ft-c-v_geo (conda env controlJEPA).
#
#   conda activate controlJEPA
#   bash scripts/run_nl_rx_table1_ft_c_v_geo.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
ENV="${CONDA_ENV:-controlJEPA}"

CKPT="${CKPT:-$REPO/Llama3.2-1B-Instruct/ft-c-v_geo-synth-g0.80-t1e-3-2e-5-0.01-0-82}"
BASE="${BASE_MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"

exec conda run -n "$ENV" --no-capture-output python "$REPO/scripts/nl_rx_table1_suffix_accuracy.py" \
  --checkpoint "$CKPT" \
  --base_model_name "$BASE" \
  --column_label "ft-c-v_geo" \
  --json_out "${JSON_OUT:-$REPO/results/table1_ft-c-v_geo_latest.json}" \
  "$@"
